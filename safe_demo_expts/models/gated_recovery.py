"""A1c gated recovery residual: frozen base BC + δa trained on oracle stuck labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from safe_demo_expts.models.bc import BCPolicy, act_policy, build_features

PathLike = Union[str, Path]


class ResidualMLP(nn.Module):
    """Small residual head predicting δa given (obs [+ history])."""

    def __init__(self, in_dim: int, act_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.in_dim = int(in_dim)
        self.act_dim = int(act_dim)
        self.hidden = int(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _residual_features(
    obs: np.ndarray,
    reject: Optional[np.ndarray],
    use_history: bool,
    traj_id: Optional[np.ndarray],
    actions_for_hist: Optional[np.ndarray],
) -> np.ndarray:
    """Build the same feature vector the residual sees at train / act time."""
    if reject is None:
        reject = np.zeros(len(obs), dtype=np.float64)
    if not use_history:
        return obs.astype(np.float32)
    assert actions_for_hist is not None
    return build_features(
        obs,
        actions_for_hist,
        reject,
        use_history=True,
        traj_id=traj_id,
    )


@torch.no_grad()
def _base_actions_batch(
    base_policy: BCPolicy,
    obs: np.ndarray,
    reject: Optional[np.ndarray],
    use_history: bool,
    traj_id: Optional[np.ndarray],
    oracle_actions: np.ndarray,
    device: str,
) -> np.ndarray:
    """Frozen-base predictions on the oracle-labelled states (teacher-forced history)."""
    if reject is None:
        reject = np.zeros(len(obs), dtype=np.float64)
    # History for base uses oracle actions as the executed stream during collection.
    feats = build_features(
        obs,
        oracle_actions,
        reject,
        use_history=use_history,
        traj_id=traj_id,
    )
    x = torch.from_numpy(feats).to(device)
    base_policy.eval()
    pred = base_policy(x).cpu().numpy()
    return np.clip(pred, -1.0, 1.0).astype(np.float64)


def train_residual(
    base_policy: BCPolicy,
    obs: np.ndarray,
    oracle_actions: np.ndarray,
    reject: Optional[np.ndarray] = None,
    use_history: bool = False,
    traj_id: Optional[np.ndarray] = None,
    a_base: Optional[np.ndarray] = None,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[ResidualMLP, Dict[str, float]]:
    """
    Train residual δa so that clip(a_base + δ) ≈ oracle on stuck labels.

    ``base_policy`` is frozen. If ``a_base`` is provided (from the collect buffer),
    it is used; otherwise base predictions are recomputed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(obs) == 0:
        in_dim = int(obs.shape[1]) if obs.ndim == 2 and obs.shape[1] else 60
        if use_history:
            in_dim = in_dim + 3
        residual = ResidualMLP(in_dim=in_dim, act_dim=2, hidden=hidden).to(device)
        residual.eval()
        return residual, {
            "residual_train_mse": float("nan"),
            "recon_oracle_mse": float("nan"),
            "n_oracle": 0.0,
            "epochs": float(epochs),
        }

    if reject is None:
        reject = np.zeros(len(obs), dtype=np.float64)
    oracle_actions = np.asarray(oracle_actions, dtype=np.float64).reshape(-1, 2)

    if a_base is None:
        a_base = _base_actions_batch(
            base_policy, obs, reject, use_history, traj_id, oracle_actions, device
        )
    else:
        a_base = np.asarray(a_base, dtype=np.float64).reshape(-1, 2)

    targets = (oracle_actions - a_base).astype(np.float32)
    feats = _residual_features(obs, reject, use_history, traj_id, oracle_actions)

    residual = ResidualMLP(in_dim=feats.shape[1], act_dim=2, hidden=hidden).to(device)
    # Freeze base (defensive — we never step its optimizer).
    for p in base_policy.parameters():
        p.requires_grad_(False)
    base_policy.eval()

    opt = torch.optim.Adam(residual.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x = torch.from_numpy(feats.astype(np.float32))
    y = torch.from_numpy(targets)
    loader = DataLoader(TensorDataset(x, y), batch_size=min(batch_size, max(1, len(x))), shuffle=True)

    last = 0.0
    residual.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(residual(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        last = epoch_loss / max(1, n_batches)

    residual.eval()
    with torch.no_grad():
        delta = residual(x.to(device)).cpu().numpy()
        recon = np.clip(a_base + delta, -1.0, 1.0)
        recon_mse = float(np.mean((recon - oracle_actions) ** 2))

    return residual, {
        "residual_train_mse": float(last),
        "recon_oracle_mse": recon_mse,
        "n_oracle": float(len(obs)),
        "epochs": float(epochs),
    }


@torch.no_grad()
def act_with_residual(
    base: BCPolicy,
    residual: ResidualMLP,
    obs: np.ndarray,
    prev_a: Optional[np.ndarray] = None,
    prev_rej: float = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    """Return clip(a_base + δ). History features mirror ``act_policy``."""
    a_base = act_policy(base, obs, prev_action=prev_a, prev_reject=prev_rej, device=device)
    o = obs.astype(np.float32)
    if getattr(base, "use_history", False):
        if prev_a is None:
            prev_a = np.zeros(2, dtype=np.float32)
        feat = np.concatenate(
            [o, np.asarray(prev_a, dtype=np.float32).reshape(2), np.array([prev_rej], dtype=np.float32)]
        )
    else:
        feat = o
    # Residual may have been trained with/without history independently; match its in_dim.
    in_dim = int(getattr(residual, "in_dim", residual.net[0].in_features))
    if feat.shape[0] < in_dim:
        pad = np.zeros(in_dim - feat.shape[0], dtype=np.float32)
        feat = np.concatenate([feat, pad])
    elif feat.shape[0] > in_dim:
        feat = feat[:in_dim]
    x = torch.from_numpy(feat[None, :].astype(np.float32)).to(device)
    delta = residual(x).cpu().numpy()[0]
    return np.clip(a_base + delta, -1.0, 1.0).astype(np.float64)


def save_residual(
    residual: ResidualMLP,
    path: PathLike,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("in_dim", int(residual.in_dim))
    meta.setdefault("act_dim", int(residual.act_dim))
    meta.setdefault("hidden", int(residual.hidden))
    payload = {
        "state_dict": residual.state_dict(),
        "meta": meta,
        "in_dim": int(residual.in_dim),
        "act_dim": int(residual.act_dim),
        "hidden": int(residual.hidden),
    }
    torch.save(payload, path)
    return path


def load_residual(
    path: PathLike,
    device: str = "cpu",
) -> Tuple[ResidualMLP, Dict[str, Any]]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    meta = dict(payload.get("meta") or {})
    in_dim = int(payload.get("in_dim", meta.get("in_dim", 60)))
    act_dim = int(payload.get("act_dim", meta.get("act_dim", 2)))
    hidden = int(payload.get("hidden", meta.get("hidden", 64)))
    residual = ResidualMLP(in_dim=in_dim, act_dim=act_dim, hidden=hidden).to(device)
    residual.load_state_dict(payload["state_dict"])
    residual.eval()
    return residual, meta
