"""Generate demos with a privileged waypoint expert under a relaxed (demo) shield."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from safe_demo_expts.envs.point_nav import PointNavConfig, PointNavEnv
from safe_demo_expts.shield.obstacle_shield import ObstacleShield


def expert_action(env: PointNavEnv, shield: ObstacleShield) -> np.ndarray:
    """Follow waypoints toward the goal; only use the demo shield if seek is unsafe."""
    target = env.current_waypoint()
    delta = target - env.pos
    n = np.linalg.norm(delta)
    if n < 1e-8:
        delta = np.array(env.cfg.goal, dtype=np.float64) - env.pos
        n = np.linalg.norm(delta)
    if n < 1e-8:
        return np.zeros(2, dtype=np.float64)
    seek = delta * (env.cfg.max_speed / n)
    sr = shield(env, seek)
    if not sr.reject:
        return seek  # keep nominal seek when already demo-safe (better labels)

    tang = np.array([-seek[1], seek[0]], dtype=np.float64)
    tn = np.linalg.norm(tang)
    if tn > 1e-8:
        tang = tang / tn * env.cfg.max_speed
    for cand in (tang, -tang, 0.6 * tang + 0.4 * seek, -0.6 * tang + 0.4 * seek, -0.4 * seek):
        sr = shield(env, cand)
        if not sr.reject:
            return cand
    return sr.a_exec


def generate_demos(
    n_traj: int = 40,
    deploy_margin: float = 0.16,
    demo_margin: float = 0.08,
    seed: int = 0,
    max_steps: int = 200,
    path_bias: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    path_bias: shift waypoints toward one corridor wall (increases deploy illegality).
    """
    cfg = PointNavConfig(seed=seed, max_steps=max_steps)
    env = PointNavEnv(cfg)
    if abs(path_bias) > 1e-8:
        v = np.array([-1.0, 1.0]) / np.sqrt(2.0)
        env.waypoints = env.waypoints + path_bias * v
    demo_shield = ObstacleShield(margin=demo_margin)
    deploy_shield = ObstacleShield(margin=deploy_margin)

    obs_l: List[np.ndarray] = []
    a_prop_l: List[np.ndarray] = []
    a_exec_demo_l: List[np.ndarray] = []
    a_exec_deploy_l: List[np.ndarray] = []
    reject_deploy_l: List[float] = []
    done_l: List[float] = []
    traj_id_l: List[int] = []
    success_count = 0

    for tid in range(n_traj):
        env.rng = np.random.default_rng(seed + 1000 + tid)
        o = env.reset()
        for _ in range(max_steps):
            a_demo = expert_action(env, demo_shield)
            sr_dep = deploy_shield(env, a_demo)

            obs_l.append(o.copy())
            a_prop_l.append(a_demo.copy())
            a_exec_demo_l.append(a_demo.copy())
            a_exec_deploy_l.append(sr_dep.a_exec.copy())
            reject_deploy_l.append(1.0 if sr_dep.reject else 0.0)
            traj_id_l.append(tid)

            o, _, terminated, truncated, info = env.step(a_demo)
            done_l.append(1.0 if (terminated or truncated) else 0.0)
            if terminated:
                success_count += 1
                break
            if truncated:
                break

    return {
        "obs": np.stack(obs_l),
        "a_prop": np.stack(a_prop_l),
        "a_exec_demo": np.stack(a_exec_demo_l),
        "a_exec_deploy": np.stack(a_exec_deploy_l),
        "reject_deploy": np.asarray(reject_deploy_l, dtype=np.float64),
        "done": np.asarray(done_l, dtype=np.float64),
        "traj_id": np.asarray(traj_id_l, dtype=np.int64),
        "meta": {
            "n_traj": n_traj,
            "success_traj": success_count,
            "deploy_margin": deploy_margin,
            "demo_margin": demo_margin,
            "illegal_rate": float(np.mean(reject_deploy_l)) if reject_deploy_l else 0.0,
            "seed": seed,
        },
    }


def filter_dataset(data: Dict[str, np.ndarray], mode: str = "A") -> Dict[str, np.ndarray]:
    obs = data["obs"]
    reject = data["reject_deploy"].astype(bool)
    if mode == "A":
        keep = ~reject
        labels = data["a_prop"][keep]
        out_obs = obs[keep]
        hist_reject = data["reject_deploy"][keep]
    elif mode == "B":
        keep = np.ones(len(obs), dtype=bool)
        labels = data["a_prop"]
        out_obs = obs
        hist_reject = data["reject_deploy"]
    elif mode == "C":
        keep = np.ones(len(obs), dtype=bool)
        labels = data["a_exec_deploy"]
        out_obs = obs
        hist_reject = data["reject_deploy"]
    else:
        raise ValueError(mode)

    return {
        "obs": out_obs,
        "actions": labels,
        "reject_deploy": hist_reject,
        "traj_id": data["traj_id"][keep],
        "kept_frac": float(keep.mean()),
        "mode": mode,
        "meta": data["meta"],
    }
