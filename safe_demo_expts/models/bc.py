"""Behavioral cloning MLP with optional short history features."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 128, history_extra: int = 0):
        super().__init__()
        in_dim = obs_dim + history_extra
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_features(
    obs: np.ndarray,
    actions: Optional[np.ndarray] = None,
    reject: Optional[np.ndarray] = None,
    use_history: bool = False,
    traj_id: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not use_history:
        return obs.astype(np.float32)
    assert actions is not None and reject is not None
    prev_a = np.zeros_like(actions)
    prev_r = np.zeros_like(reject)
    prev_a[1:] = actions[:-1]
    prev_r[1:] = reject[:-1]
    if traj_id is not None:
        boundary = np.zeros(len(traj_id), dtype=bool)
        boundary[1:] = traj_id[1:] != traj_id[:-1]
        prev_a[boundary] = 0.0
        prev_r[boundary] = 0.0
    return np.concatenate([obs, prev_a, prev_r.reshape(-1, 1)], axis=1).astype(np.float32)


def _mse(policy: BCPolicy, xb: torch.Tensor, yb: torch.Tensor) -> float:
    with torch.no_grad():
        return float(nn.functional.mse_loss(policy(xb), yb).item())


def train_bc(
    obs: np.ndarray,
    actions: np.ndarray,
    reject: Optional[np.ndarray] = None,
    traj_id: Optional[np.ndarray] = None,
    use_history: bool = False,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    val_frac: float = 0.2,
) -> Tuple[BCPolicy, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if reject is None:
        reject = np.zeros(len(obs), dtype=np.float64)
    feats = build_features(obs, actions, reject, use_history=use_history, traj_id=traj_id)
    history_extra = 0 if not use_history else (actions.shape[1] + 1)

    if traj_id is not None and len(np.unique(traj_id)) >= 5 and val_frac > 0:
        uniq = np.unique(traj_id)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_val = max(1, int(round(len(uniq) * val_frac)))
        val_ids = set(uniq[:n_val].tolist())
        train_mask = np.array([t not in val_ids for t in traj_id], dtype=bool)
        val_mask = ~train_mask
    else:
        n = len(obs)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_val = max(1, int(round(n * val_frac))) if val_frac > 0 and n >= 10 else 0
        val_mask = np.zeros(n, dtype=bool)
        if n_val:
            val_mask[idx[:n_val]] = True
        train_mask = ~val_mask if n_val else np.ones(n, dtype=bool)

    x_tr = torch.from_numpy(feats[train_mask])
    y_tr = torch.from_numpy(actions[train_mask].astype(np.float32))
    x_va = torch.from_numpy(feats[val_mask]) if val_mask.any() else None
    y_va = torch.from_numpy(actions[val_mask].astype(np.float32)) if val_mask.any() else None

    policy = BCPolicy(obs_dim=obs.shape[1], act_dim=actions.shape[1], history_extra=history_extra).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)

    last_train = 0.0
    last_val = float("nan")
    policy.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(policy(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        last_train = epoch_loss / max(1, n_batches)
        if x_va is not None and len(x_va):
            policy.eval()
            last_val = _mse(policy, x_va.to(device), y_va.to(device))
            policy.train()

    policy.eval()
    policy.history_extra = history_extra  # type: ignore[attr-defined]
    policy.use_history = use_history  # type: ignore[attr-defined]
    return policy, {
        "train_mse": last_train,
        "val_mse": last_val,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "epochs": float(epochs),
    }


def finetune_bc(
    policy: BCPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    reject: Optional[np.ndarray] = None,
    traj_id: Optional[np.ndarray] = None,
    use_history: bool = False,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 5e-4,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[BCPolicy, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if reject is None:
        reject = np.zeros(len(obs), dtype=np.float64)
    feats = build_features(obs, actions, reject, use_history=use_history, traj_id=traj_id)
    x = torch.from_numpy(feats)
    y = torch.from_numpy(actions.astype(np.float32))
    # load_policy / deepcopy can leave params frozen; re-enable grads for FT
    policy.to(device)
    for p in policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    policy.train()
    last = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(policy(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        last = epoch_loss / max(1, n_batches)
    policy.eval()
    policy.use_history = use_history  # type: ignore[attr-defined]
    return policy, {"ft_train_mse": last, "ft_epochs": float(epochs), "ft_n": float(len(obs))}


@torch.no_grad()
def act_policy(
    policy: BCPolicy,
    obs: np.ndarray,
    prev_action: Optional[np.ndarray] = None,
    prev_reject: float = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    o = obs.astype(np.float32)
    if getattr(policy, "use_history", False):
        if prev_action is None:
            prev_action = np.zeros(2, dtype=np.float32)
        feat = np.concatenate([o, prev_action.astype(np.float32), np.array([prev_reject], dtype=np.float32)])
    else:
        feat = o
    x = torch.from_numpy(feat[None, :]).to(device)
    a = policy(x).cpu().numpy()[0]
    return np.clip(a, -1.0, 1.0).astype(np.float64)
