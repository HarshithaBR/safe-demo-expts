"""H3 query-vs-escape gate: predict whether autonomous escape will fail."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from safe_demo_expts.envs.safety_point_goal import SafetyPointGoalEnv

PathLike = Union[str, Path]

FEATURE_DIM = 8
NEARBY_RADIUS = 1.25

GATE_MODES = (
    "escape_always",
    "oracle_always",
    "try_escape_then_query",
    "learned",
)


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _heading_alignment(env: "SafetyPointGoalEnv", world_dir: np.ndarray) -> float:
    world_dir = np.asarray(world_dir, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(world_dir))
    if n < 1e-8:
        return 0.0
    desired = float(np.arctan2(world_dir[1], world_dir[0]))
    err = _wrap_angle(desired - env.yaw())
    return float(np.cos(err))


def stuck_features(
    env: "SafetyPointGoalEnv",
    clearance: Optional[float] = None,
    margin: float = 0.14,
    nearby_radius: float = NEARBY_RADIUS,
) -> np.ndarray:
    """
    Privileged 8-d stuck feature vector:

      [clearance, penetrate_depth, nearest_hazard_dist, n_nearby,
       goal_dist, away_heading_align, goal_heading_align, hazard_count]
    """
    if clearance is None:
        from safe_demo_expts.shield.safety_point_shield import SafetyPointShield

        clearance = float(SafetyPointShield(margin=margin, escape_mode=True).clearance(env))
    clr = float(clearance)
    penetrate = float(max(0.0, -clr))

    agent = env.agent_xy()
    haz = env.hazard_xy()
    hazard_count = float(len(haz))
    if len(haz):
        dists = np.linalg.norm(haz - agent[None, :], axis=1)
        nearest_dist = float(dists.min())
        n_nearby = float(np.sum(dists <= nearby_radius))
        nearest = haz[int(np.argmin(dists))]
        away = agent - nearest
    else:
        nearest_dist = 10.0
        n_nearby = 0.0
        away = np.zeros(2, dtype=np.float64)

    goal = env.goal_xy()
    goal_delta = goal - agent
    goal_dist = float(np.linalg.norm(goal_delta))
    away_align = _heading_alignment(env, away)
    goal_align = _heading_alignment(env, goal_delta)

    return np.array(
        [
            clr,
            penetrate,
            nearest_dist,
            n_nearby,
            goal_dist,
            away_align,
            goal_align,
            hazard_count,
        ],
        dtype=np.float32,
    )


class GateMLP(nn.Module):
    """P(escape fails | stuck features)."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def prob_fail(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def train_gate(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-2,
    hidden: int = 32,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[GateMLP, Dict[str, float]]:
    """
    Train binary classifier: label=1 means escape failed (still stuck after try).

    Returns gate + stats ``{bce, acc, n, fail_rate, epochs}``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    features = np.asarray(features, dtype=np.float32).reshape(-1, FEATURE_DIM)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)

    gate = GateMLP(in_dim=FEATURE_DIM, hidden=hidden).to(device)
    if len(features) == 0:
        gate.eval()
        return gate, {
            "bce": float("nan"),
            "acc": float("nan"),
            "n": 0.0,
            "fail_rate": 0.0,
            "epochs": float(epochs),
        }

    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=min(batch_size, max(1, len(x))),
        shuffle=True,
    )

    last_bce = 0.0
    gate.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(gate(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        last_bce = epoch_loss / max(1, n_batches)

    gate.eval()
    with torch.no_grad():
        logits = gate(x.to(device)).cpu().numpy()
        preds = (1.0 / (1.0 + np.exp(-logits))) >= 0.5
        acc = float(np.mean(preds.astype(np.float32) == labels))
    return gate, {
        "bce": float(last_bce),
        "acc": acc,
        "n": float(len(labels)),
        "fail_rate": float(np.mean(labels)),
        "epochs": float(epochs),
    }


@torch.no_grad()
def gate_fail_prob(
    gate: GateMLP,
    feats: np.ndarray,
    device: str = "cpu",
) -> float:
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32).reshape(1, -1)).to(device)
    return float(gate.prob_fail(x).cpu().numpy()[0])


def decide_query(
    mode: str,
    gate: Optional[GateMLP] = None,
    feats: Optional[np.ndarray] = None,
    fail_threshold: float = 0.55,
    escape_try_exhausted: bool = False,
    device: str = "cpu",
) -> str:
    """
    Return ``'escape'`` or ``'oracle'`` for the given mode.

    ``try_escape_then_query`` returns escape until ``escape_try_exhausted``.
    """
    mode = (mode or "escape_always").lower()
    if mode == "escape_always":
        return "escape"
    if mode == "oracle_always":
        return "oracle"
    if mode == "try_escape_then_query":
        return "oracle" if escape_try_exhausted else "escape"
    if mode == "learned":
        if gate is None or feats is None:
            return "escape"
        p = gate_fail_prob(gate, feats, device=device)
        return "oracle" if p >= float(fail_threshold) else "escape"
    raise ValueError(f"unknown gate mode={mode}")


def save_gate(gate: GateMLP, path: PathLike, meta: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("in_dim", int(gate.in_dim))
    meta.setdefault("hidden", int(gate.hidden))
    torch.save(
        {
            "state_dict": gate.state_dict(),
            "meta": meta,
            "in_dim": int(gate.in_dim),
            "hidden": int(gate.hidden),
        },
        path,
    )
    return path


def load_gate(path: PathLike, device: str = "cpu") -> Tuple[GateMLP, Dict[str, Any]]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    meta = dict(payload.get("meta") or {})
    in_dim = int(payload.get("in_dim", meta.get("in_dim", FEATURE_DIM)))
    hidden = int(payload.get("hidden", meta.get("hidden", 32)))
    gate = GateMLP(in_dim=in_dim, hidden=hidden).to(device)
    gate.load_state_dict(payload["state_dict"])
    gate.eval()
    return gate, meta


def collect_escape_fail_labels(
    policy,
    n_episodes: int,
    seed: int,
    deploy_margin: float = 0.14,
    stuck_horizon: int = 40,
    stuck_eps: float = 1e-3,
    escape_try_steps: int = 30,
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
    device: str = "cpu",
    deep_penetration: float = -0.05,
    escape_shield: str = "escape",
    mpc_horizon: int = 3,
    mpc_samples: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Label collection: on stuck, try autonomous escape for ``escape_try_steps``.
    Label = 1 if still stuck afterwards (escape failed).
    """
    from safe_demo_expts.models.bc import act_policy
    from safe_demo_expts.pipelines.evaluate_flexible import make_shield
    from safe_demo_expts.pipelines.evaluate_safety_point import goal_distance, is_stuck
    from safe_demo_expts.envs.safety_point_goal import SafetyPointConfig, SafetyPointGoalEnv

    cfg = SafetyPointConfig(env_id=env_id, max_steps=max_steps, seed=seed, stop_on_first_goal=True)
    env = SafetyPointGoalEnv(cfg)
    shield = make_shield(
        mode=escape_shield,
        deploy_margin=deploy_margin,
        mpc_horizon=mpc_horizon,
        mpc_samples=mpc_samples,
    )

    feats_l: List[np.ndarray] = []
    labels_l: List[float] = []

    for ep in range(n_episodes):
        o, _ = env.reset(seed=(seed + 10_000 + ep))
        prev_a = np.zeros(2, dtype=np.float64)
        prev_rej = 0.0
        dist_hist: List[float] = []
        info: Dict = {}
        escaping = False
        escape_left = 0
        feat_at_trigger: Optional[np.ndarray] = None

        for t in range(max_steps):
            a_base = act_policy(policy, o, prev_action=prev_a, prev_reject=prev_rej, device=device)
            clr = float(shield.clearance(env))
            dist_hist.append(goal_distance(env))
            stuck_now = is_stuck(
                dist_hist,
                stuck_horizon=stuck_horizon,
                stuck_eps=stuck_eps,
                clearance=clr,
                deep_penetration=deep_penetration,
            )
            gate = stuck_now or (clr < deep_penetration and len(dist_hist) >= stuck_horizon)

            if escaping:
                a_prop = a_base
                escape_left -= 1
            elif gate:
                feat_at_trigger = stuck_features(env, clearance=clr, margin=deploy_margin)
                escaping = True
                escape_left = int(escape_try_steps)
                a_prop = a_base
            else:
                a_prop = a_base

            sr = shield(env, a_prop)
            o, _r, terminated, truncated, info = env.step(sr.a_exec)
            prev_a = sr.a_exec.copy()
            prev_rej = 1.0 if sr.reject else 0.0

            if escaping and escape_left <= 0:
                # Evaluate whether still stuck after the try window.
                clr2 = float(shield.clearance(env))
                still = is_stuck(
                    dist_hist,
                    stuck_horizon=stuck_horizon,
                    stuck_eps=stuck_eps,
                    clearance=clr2,
                    deep_penetration=deep_penetration,
                ) or (clr2 < deep_penetration)
                success = bool(info.get("success"))
                label = 0.0 if (success or not still) else 1.0
                if feat_at_trigger is not None:
                    feats_l.append(feat_at_trigger)
                    labels_l.append(label)
                escaping = False
                feat_at_trigger = None
                # If still stuck after failed escape, end episode as stuck for labeling purity.
                if label >= 0.5 and not success:
                    break

            if terminated or truncated:
                if escaping and feat_at_trigger is not None:
                    success = bool(info.get("success"))
                    label = 0.0 if success else 1.0
                    feats_l.append(feat_at_trigger)
                    labels_l.append(label)
                break

    env.close()
    feats_arr = np.stack(feats_l) if feats_l else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    labels_arr = np.asarray(labels_l, dtype=np.float32)
    stats = {
        "n_labels": float(len(labels_arr)),
        "fail_rate": float(labels_arr.mean()) if len(labels_arr) else 0.0,
    }
    return feats_arr, labels_arr, stats
