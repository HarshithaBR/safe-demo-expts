"""S0 diagnostic: 2D point robot among circular obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PointNavConfig:
    world_size: float = 4.0
    goal: Tuple[float, float] = (3.5, 3.5)
    start: Tuple[float, float] = (0.35, 0.35)
    max_speed: float = 0.20
    dt: float = 1.0
    goal_radius: float = 0.25
    robot_radius: float = 0.10
    max_steps: int = 200
    seed: int = 0


def _build_corridor_obstacles() -> np.ndarray:
    """
    Place obstacle pairs perpendicular to the diagonal corridor.
    Nominal clearance from the diagonal centerline ≈ 0.12 (after robot radius),
    so demo_margin≲0.08 succeeds while deploy_margin≳0.14 rejects near pinch points.
    """
    v = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    offset = 0.57
    radius = 0.35
    centers_on_diag = [
        np.array([1.1, 1.1]),
        np.array([2.0, 2.0]),
        np.array([2.9, 2.9]),
    ]
    obs = []
    for p in centers_on_diag:
        obs.append([*(p - offset * v), radius])
        obs.append([*(p + offset * v), radius])
    return np.asarray(obs, dtype=np.float64)


DEFAULT_OBSTACLES = _build_corridor_obstacles()

DEFAULT_WAYPOINTS = np.array(
    [
        [0.55, 0.55],
        [1.1, 1.1],
        [1.55, 1.55],
        [2.0, 2.0],
        [2.45, 2.45],
        [2.9, 2.9],
        [3.5, 3.5],
    ],
    dtype=np.float64,
)


class PointNavEnv:
    """Continuous 2D point navigation. Actions are (dx, dy) clipped to max_speed."""

    def __init__(
        self,
        config: Optional[PointNavConfig] = None,
        obstacles: Optional[np.ndarray] = None,
    ):
        self.cfg = config or PointNavConfig()
        self.obstacles = DEFAULT_OBSTACLES.copy() if obstacles is None else np.asarray(obstacles, dtype=np.float64)
        self.waypoints = DEFAULT_WAYPOINTS.copy()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.pos = np.zeros(2, dtype=np.float64)
        self.t = 0
        self._goal = np.array(self.cfg.goal, dtype=np.float64)
        self._wp_idx = 0

    @property
    def observation_size(self) -> int:
        return 2 + 2 + 1 + 6

    @property
    def action_size(self) -> int:
        return 2

    def reset(self, start: Optional[Tuple[float, float]] = None) -> np.ndarray:
        self._wp_idx = 0
        if start is None:
            for _ in range(50):
                jitter = self.rng.uniform(-0.06, 0.06, size=2)
                cand = np.array(self.cfg.start, dtype=np.float64) + jitter
                if not self._in_collision(cand, margin=0.0):
                    self.pos = cand
                    break
            else:
                self.pos = np.array(self.cfg.start, dtype=np.float64)
        else:
            self.pos = np.array(start, dtype=np.float64)
        self.t = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        goal_rel = self._goal - self.pos
        dists = self._obstacle_dists(self.pos)
        nearest = float(dists.min()) if len(dists) else self.cfg.world_size
        order = np.argsort(dists)[:3]
        rels = []
        for idx in order:
            c = self.obstacles[idx, :2]
            rels.extend((c - self.pos).tolist())
        while len(rels) < 6:
            rels.extend([0.0, 0.0])
        return np.asarray(
            [self.pos[0], self.pos[1], goal_rel[0], goal_rel[1], nearest, *rels[:6]],
            dtype=np.float64,
        )

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float64).reshape(2)
        speed = np.linalg.norm(a)
        if speed > self.cfg.max_speed:
            a = a * (self.cfg.max_speed / (speed + 1e-8))
        nxt = self.pos + a * self.cfg.dt
        nxt = np.clip(nxt, 0.0, self.cfg.world_size)
        collided = self._in_collision(nxt, margin=0.0)
        if collided:
            nxt = self.pos.copy()
        self.pos = nxt
        self.t += 1
        while self._wp_idx < len(self.waypoints) - 1:
            if np.linalg.norm(self.pos - self.waypoints[self._wp_idx]) < 0.26:
                self._wp_idx += 1
            else:
                break
        obs = self.observe()
        dist_goal = float(np.linalg.norm(self._goal - self.pos))
        success = dist_goal <= self.cfg.goal_radius
        truncated = self.t >= self.cfg.max_steps
        reward = -dist_goal + (10.0 if success else 0.0) - (5.0 if collided else 0.0)
        info = {
            "success": success,
            "collision": collided,
            "dist_goal": dist_goal,
            "pos": self.pos.copy(),
        }
        return obs, reward, success, truncated, info

    def current_waypoint(self) -> np.ndarray:
        return self.waypoints[min(self._wp_idx, len(self.waypoints) - 1)]

    def geodesic_progress_proxy(self) -> float:
        return -float(np.linalg.norm(self._goal - self.pos))

    def _obstacle_dists(self, pos: np.ndarray) -> np.ndarray:
        if len(self.obstacles) == 0:
            return np.array([self.cfg.world_size], dtype=np.float64)
        centers = self.obstacles[:, :2]
        radii = self.obstacles[:, 2]
        return np.linalg.norm(centers - pos[None, :], axis=1) - radii - self.cfg.robot_radius

    def _in_collision(self, pos: np.ndarray, margin: float) -> bool:
        return bool(np.any(self._obstacle_dists(pos) < margin))
