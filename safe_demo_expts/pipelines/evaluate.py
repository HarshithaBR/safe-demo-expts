"""Closed-loop evaluation under deploy shield + stuck detection."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from safe_demo_expts.envs.point_nav import PointNavConfig, PointNavEnv
from safe_demo_expts.models.bc import BCPolicy, act_policy
from safe_demo_expts.shield.obstacle_shield import ObstacleShield


def evaluate_policy(
    policy: BCPolicy,
    deploy_margin: float = 0.10,
    n_episodes: int = 30,
    seed: int = 0,
    stuck_horizon: int = 25,
    stuck_eps: float = 1e-3,
    device: str = "cpu",
    max_steps: int = 200,
) -> Dict[str, float]:
    cfg = PointNavConfig(seed=seed, max_steps=max_steps)
    env = PointNavEnv(cfg)
    shield = ObstacleShield(margin=deploy_margin)

    successes = 0
    stuck_eps_count = 0
    rejects = []
    lengths = []
    collisions = 0

    for ep in range(n_episodes):
        env.rng = np.random.default_rng(seed + 10_000 + ep)
        o = env.reset()
        prev_a = np.zeros(2, dtype=np.float64)
        prev_rej = 0.0
        progress_hist = []
        stuck = False
        ep_rejects = 0

        for t in range(max_steps):
            a_prop = act_policy(policy, o, prev_action=prev_a, prev_reject=prev_rej, device=device)
            sr = shield(env, a_prop)
            o, _, terminated, truncated, info = env.step(sr.a_exec)
            prev_a = sr.a_exec
            prev_rej = 1.0 if sr.reject else 0.0
            if sr.reject:
                ep_rejects += 1
            if info["collision"]:
                collisions += 1

            progress_hist.append(env.geodesic_progress_proxy())
            if len(progress_hist) >= stuck_horizon:
                window = progress_hist[-stuck_horizon:]
                if max(window) - min(window) < stuck_eps:
                    stuck = True
                    break

            if terminated or truncated:
                break

        lengths.append(t + 1)
        rejects.append(ep_rejects / max(1, t + 1))
        if info.get("success"):
            successes += 1
        if stuck:
            stuck_eps_count += 1

    return {
        "success_rate": successes / n_episodes,
        "stuck_rate": stuck_eps_count / n_episodes,
        "mean_reject_rate": float(np.mean(rejects)),
        "mean_length": float(np.mean(lengths)),
        "collision_events": float(collisions),
        "n_episodes": float(n_episodes),
    }
