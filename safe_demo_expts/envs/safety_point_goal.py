"""Safety-Gymnasium SafetyPointGoal wrapper with privileged layout access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import safety_gymnasium
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "SafetyPointGoal requires safety-gymnasium (.venv-safety). "
        "See docs/MAC_SETUP.md / BASELINE_ENV_SETUP.md"
    ) from e


@dataclass
class SafetyPointConfig:
    env_id: str = "SafetyPointGoal1-v0"
    max_steps: int = 500
    seed: int = 0
    stop_on_first_goal: bool = True


class SafetyPointGoalEnv:
    """Thin wrapper: reset/step + privileged agent/goal/hazard XY."""

    def __init__(self, config: Optional[SafetyPointConfig] = None):
        self.cfg = config or SafetyPointConfig()
        self._env = safety_gymnasium.make(self.cfg.env_id)
        self.observation_size = int(np.prod(self._env.observation_space.shape))
        self.action_size = int(np.prod(self._env.action_space.shape))
        self._steps = 0

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def task(self):
        return self._env.unwrapped.task

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        seed = self.cfg.seed if seed is None else seed
        obs, info = self._env.reset(seed=seed)
        self._steps = 0
        return np.asarray(obs, dtype=np.float64), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(self.action_size)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        self._steps += 1
        if self._steps >= self.cfg.max_steps:
            truncated = True
        info = dict(info)
        info["cost"] = float(cost)
        goal_met = bool(info.get("goal_met", False) or self.task.goal_achieved)
        info["success"] = goal_met
        info["agent_xy"] = self.agent_xy()
        info["goal_xy"] = self.goal_xy()
        if goal_met and self.cfg.stop_on_first_goal:
            terminated = True
            self._env.unwrapped.terminated = True
        return (
            np.asarray(obs, dtype=np.float64),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def agent_xy(self) -> np.ndarray:
        return np.asarray(self.task.agent.pos[:2], dtype=np.float64)

    def goal_xy(self) -> np.ndarray:
        return np.asarray(self.task.goal.pos[:2], dtype=np.float64)

    def hazard_xy(self) -> np.ndarray:
        haz = getattr(self.task, "hazards", None)
        if haz is None:
            return np.zeros((0, 2), dtype=np.float64)
        pos = np.asarray(haz.pos, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        return pos[:, :2]

    def hazard_radius(self) -> float:
        haz = getattr(self.task, "hazards", None)
        if haz is None:
            return 0.2
        return float(getattr(haz, "size", 0.2))

    def yaw(self) -> float:
        mat = np.asarray(self.task.agent.mat, dtype=np.float64)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    def close(self) -> None:
        self._env.close()
