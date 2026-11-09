"""Closed-loop SafetyPointGoal evaluation under SafetyPointShield."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from safe_demo_expts.envs.safety_point_goal import SafetyPointConfig, SafetyPointGoalEnv
from safe_demo_expts.models.bc import BCPolicy, act_policy
from safe_demo_expts.shield.safety_point_shield import SafetyPointShield


def goal_distance(env: SafetyPointGoalEnv) -> float:
    return float(np.linalg.norm(env.goal_xy() - env.agent_xy()))


def is_stuck(
    dist_hist: List[float],
    stuck_horizon: int,
    stuck_eps: float = 1e-3,
    clearance: Optional[float] = None,
    deep_penetration: float = -0.05,
) -> bool:
    """
    Stuck if goal-distance range over the last ``stuck_horizon`` steps is < ``stuck_eps``.

    Optionally also treat deep hazard penetration with no progress as stuck.
    """
    if len(dist_hist) < stuck_horizon:
        return False
    window = dist_hist[-stuck_horizon:]
    span = max(window) - min(window)
    if span < stuck_eps:
        return True
    # Optional: deep hazard penetration with essentially no goal progress.
    if clearance is not None and clearance < deep_penetration and span < max(stuck_eps * 10.0, 1e-2):
        return True
    return False


def evaluate_policy(
    policy: BCPolicy,
    deploy_margin: float = 0.14,
    n_episodes: int = 20,
    seed: int = 0,
    stuck_horizon: int = 40,
    stuck_eps: float = 1e-3,
    device: str = "cpu",
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
    escape_mode: bool = True,
    deep_penetration: float = -0.05,
) -> Dict[str, float]:
    """
    Autonomous closed-loop eval.

    Metrics: success_rate, terminal_stuck_rate (alias stuck_rate), mean_reject_rate,
    mean_episode_cost, mean_length.
    """
    cfg = SafetyPointConfig(env_id=env_id, max_steps=max_steps, seed=seed, stop_on_first_goal=True)
    env = SafetyPointGoalEnv(cfg)
    shield = SafetyPointShield(margin=deploy_margin, escape_mode=escape_mode)

    successes = 0
    stuck_count = 0
    reject_rates: List[float] = []
    lengths: List[float] = []
    costs: List[float] = []

    for ep in range(n_episodes):
        o, _ = env.reset(seed=(seed + 10_000 + ep))
        prev_a = np.zeros(2, dtype=np.float64)
        prev_rej = 0.0
        dist_hist: List[float] = []
        stuck = False
        ep_rejects = 0
        ep_cost = 0.0
        info: Dict = {}
        t = 0

        for t in range(max_steps):
            a_prop = act_policy(policy, o, prev_action=prev_a, prev_reject=prev_rej, device=device)
            sr = shield(env, a_prop)
            o, _r, terminated, truncated, info = env.step(sr.a_exec)
            prev_a = sr.a_exec.copy()
            prev_rej = 1.0 if sr.reject else 0.0
            if sr.reject:
                ep_rejects += 1
            ep_cost += float(info.get("cost", 0.0))

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
        costs.append(ep_cost)
        if info.get("success"):
            successes += 1
        if stuck:
            stuck_count += 1

    env.close()
    stuck_rate = stuck_count / max(1, n_episodes)
    return {
        "success_rate": successes / max(1, n_episodes),
        "terminal_stuck_rate": stuck_rate,
        "stuck_rate": stuck_rate,
        "mean_reject_rate": float(np.mean(reject_rates)) if reject_rates else 0.0,
        "mean_episode_cost": float(np.mean(costs)) if costs else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": float(n_episodes),
        "escape_mode": float(1.0 if escape_mode else 0.0),
    }
