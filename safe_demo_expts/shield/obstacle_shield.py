"""Hard safety filter: reject / project actions that would enter inflated obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from safe_demo_expts.envs.point_nav import PointNavEnv


@dataclass
class ShieldResult:
    a_exec: np.ndarray
    reject: bool
    a_prop: np.ndarray


class ObstacleShield:
    """Project unsafe actions onto a progress-aware safe fallback when possible."""

    def __init__(self, margin: float = 0.08, n_proj_samples: int = 20):
        self.margin = float(margin)
        self.n_proj_samples = int(n_proj_samples)

    def __call__(self, env: PointNavEnv, a_prop: np.ndarray) -> ShieldResult:
        a_prop = np.asarray(a_prop, dtype=np.float64).reshape(2)
        speed = np.linalg.norm(a_prop)
        if speed > env.cfg.max_speed:
            a_prop = a_prop * (env.cfg.max_speed / (speed + 1e-8))

        if self._safe(env, a_prop):
            return ShieldResult(a_exec=a_prop.copy(), reject=False, a_prop=a_prop.copy())

        goal = np.array(env.cfg.goal, dtype=np.float64)
        best = None
        best_score = -1e9

        # scaled proposals
        for alpha in np.linspace(0.9, 0.0, self.n_proj_samples):
            cand = a_prop * alpha
            if self._safe(env, cand):
                nxt = np.clip(env.pos + cand, 0.0, env.cfg.world_size)
                score = -np.linalg.norm(goal - nxt) + 0.05 * np.linalg.norm(cand)
                if score > best_score:
                    best_score = score
                    best = cand

        # goal-directed and tangential safe probes
        to_goal = goal - env.pos
        n = np.linalg.norm(to_goal)
        basis = []
        if n > 1e-8:
            g = to_goal / n * env.cfg.max_speed
            basis.append(g)
            basis.append(np.array([-g[1], g[0]]))
            basis.append(np.array([g[1], -g[0]]))
        for b in basis:
            for alpha in (1.0, 0.7, 0.4, 0.2):
                cand = b * alpha
                if self._safe(env, cand):
                    nxt = np.clip(env.pos + cand, 0.0, env.cfg.world_size)
                    score = -np.linalg.norm(goal - nxt) + 0.05 * np.linalg.norm(cand)
                    if score > best_score:
                        best_score = score
                        best = cand

        if best is None:
            best = np.zeros(2, dtype=np.float64)
        return ShieldResult(a_exec=best, reject=True, a_prop=a_prop.copy())

    def _safe(self, env: PointNavEnv, action: np.ndarray) -> bool:
        nxt = env.pos + action * env.cfg.dt
        nxt = np.clip(nxt, 0.0, env.cfg.world_size)
        return not env._in_collision(nxt, margin=self.margin)


def margin_pair(deploy_margin: float, illegal_level: str) -> Tuple[float, float]:
    """Return (demo_margin, deploy_margin) for controllable illegality."""
    # centerline clearance ≈ 0.12; deploy should be < that so centerline remains feasible
    deltas = {"low": 0.02, "mid": 0.05, "high": 0.08}
    if illegal_level not in deltas:
        raise ValueError(f"unknown illegal_level={illegal_level}")
    demo_margin = max(0.0, deploy_margin - deltas[illegal_level])
    return demo_margin, deploy_margin
