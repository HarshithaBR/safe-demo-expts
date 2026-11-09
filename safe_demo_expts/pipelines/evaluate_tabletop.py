"""Closed-loop Meta-World Sawyer reach evaluation under TabletopShield."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from safe_demo_expts.envs.metaworld_reach import MetaWorldReachConfig, MetaWorldReachEnv
from safe_demo_expts.models.bc import BCPolicy, act_policy
from safe_demo_expts.shield.tabletop_shield import (
    DEFAULT_PILLAR_RADIUS,
    DEFAULT_PILLAR_XY,
    TabletopShield,
    make_tabletop_shield,
)


def goal_distance(env: MetaWorldReachEnv) -> float:
    return float(np.linalg.norm(env.goal_pos() - env.ee_pos()))


def is_stuck(
    dist_hist: List[float],
    stuck_horizon: int,
    stuck_eps: float = 1e-3,
    clearance: Optional[float] = None,
    deep_penetration: float = -0.03,
) -> bool:
    """Stuck if goal-distance range over the last window is tiny."""
    if len(dist_hist) < stuck_horizon:
        return False
    window = dist_hist[-stuck_horizon:]
    span = max(window) - min(window)
    if span < stuck_eps:
        return True
    if clearance is not None and clearance < deep_penetration and span < max(stuck_eps * 10.0, 1e-2):
        return True
    return False


def evaluate_policy(
    policy: BCPolicy,
    deploy_margin: float = 0.05,
    z_min: float = 0.05,
    pillar_xy: Sequence[float] = DEFAULT_PILLAR_XY,
    pillar_radius: float = DEFAULT_PILLAR_RADIUS,
    shield: Union[str, TabletopShield] = "escape",
    n_episodes: int = 20,
    seed: int = 0,
    stuck_horizon: int = 40,
    stuck_eps: float = 1e-3,
    device: str = "cpu",
    max_steps: int = 200,
    env_name: str = "reach-v3",
    deep_penetration: float = -0.03,
    mpc_horizon: int = 3,
    mpc_samples: int = 24,
) -> Dict[str, float]:
    """
    Autonomous closed-loop eval.

    Metrics: success_rate, ever_stuck_rate, terminal_stuck_rate, mean_reject_rate,
    mean_length, n_episodes.
    """
    cfg = MetaWorldReachConfig(env_name=env_name, max_steps=max_steps, seed=seed, stop_on_success=True)
    env = MetaWorldReachEnv(cfg)
    if isinstance(shield, TabletopShield):
        shield_fn = shield
        shield_name = shield.mode
    else:
        shield_name = str(shield).lower()
        shield_fn = make_tabletop_shield(
            mode=shield_name,
            margin=deploy_margin,
            z_min=z_min,
            pillar_xy=pillar_xy,
            pillar_radius=pillar_radius,
            mpc_horizon=mpc_horizon,
            mpc_samples=mpc_samples,
        )

    successes = 0
    ever_stuck = 0
    terminal_stuck = 0
    reject_rates: List[float] = []
    lengths: List[float] = []

    act_dim = int(getattr(policy.net[-1], "out_features", 4))

    for ep in range(n_episodes):
        o, _ = env.reset(seed=(seed + 10_000 + ep))
        prev_a = np.zeros(act_dim, dtype=np.float64)
        prev_rej = 0.0
        dist_hist: List[float] = []
        stuck = False
        ep_rejects = 0
        info: Dict = {}
        t = 0

        for t in range(max_steps):
            a_prop = act_policy(policy, o, prev_action=prev_a, prev_reject=prev_rej, device=device)
            if a_prop.shape[0] != env.action_size:
                # Pad / trim to env action dim (BC act_dim=4 for tabletop).
                a = np.zeros(env.action_size, dtype=np.float64)
                n = min(env.action_size, a_prop.shape[0])
                a[:n] = a_prop[:n]
                a_prop = a
            sr = shield_fn(env, a_prop)
            o, _r, terminated, truncated, info = env.step(sr.a_exec)
            prev_a = sr.a_exec.copy()
            prev_rej = 1.0 if sr.reject else 0.0
            if sr.reject:
                ep_rejects += 1

            dist_hist.append(goal_distance(env))
            if is_stuck(
                dist_hist,
                stuck_horizon=stuck_horizon,
                stuck_eps=stuck_eps,
                clearance=sr.clearance,
                deep_penetration=deep_penetration,
            ):
                stuck = True
                break

            if terminated or truncated:
                break

        lengths.append(float(t + 1))
        reject_rates.append(ep_rejects / max(1, t + 1))
        if info.get("success"):
            successes += 1
        if stuck:
            ever_stuck += 1
            terminal_stuck += 1

    env.close()
    n = max(1, n_episodes)
    return {
        "success_rate": successes / n,
        "ever_stuck_rate": ever_stuck / n,
        "terminal_stuck_rate": terminal_stuck / n,
        "mean_reject_rate": float(np.mean(reject_rates)) if reject_rates else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": float(n_episodes),
        "shield": shield_name,
    }
