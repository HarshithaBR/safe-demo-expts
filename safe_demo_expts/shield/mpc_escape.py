"""Short-horizon kinematic MPC escape / projection for the hazard shield."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from safe_demo_expts.envs.safety_point_goal import SafetyPointGoalEnv
from safe_demo_expts.shield.safety_point_shield import SafetyPointShield, ShieldResult


class MPCEscapeShield(SafetyPointShield):
    def __init__(
        self,
        margin: float = 0.14,
        step_scale: float = 0.08,
        horizon: int = 3,
        n_samples: int = 24,
        n_alternatives: int = 12,
    ):
        super().__init__(margin=margin, step_scale=step_scale, n_alternatives=n_alternatives, escape_mode=True)
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)

    def _roll_clearance(self, env: SafetyPointGoalEnv, actions: np.ndarray) -> Tuple[float, np.ndarray]:
        xy = env.agent_xy().copy()
        mat = np.asarray(env.task.agent.mat, dtype=np.float64)
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        min_clr = 1e9
        for a in actions:
            fwd = float(np.clip(a[0], -1, 1))
            turn = float(np.clip(a[1], -1, 1))
            yaw = yaw + 0.35 * turn
            body_x = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
            xy = xy + self.step_scale * fwd * body_x
            min_clr = min(min_clr, self.clearance(env, xy))
        return float(min_clr), xy

    def __call__(self, env: SafetyPointGoalEnv, a_prop: np.ndarray) -> ShieldResult:
        a_prop = np.asarray(a_prop, dtype=np.float64).reshape(2)
        a_prop = np.clip(a_prop, env.action_space.low, env.action_space.high)
        base = super().__call__(env, a_prop)
        cur = self.clearance(env)
        pred_clr = self.clearance(env, self.predict_xy(env, a_prop))
        if pred_clr >= 0.0 and cur >= -0.02:
            return base

        goal = env.goal_xy()
        best_a0 = base.a_exec
        best_score = -1e18
        rng = np.random.default_rng(0)
        for _ in range(self.n_samples):
            seq = rng.uniform(-1, 1, size=(self.horizon, 2))
            # bias first action near base proposal
            seq[0] = 0.5 * seq[0] + 0.5 * a_prop
            min_clr, xy_h = self._roll_clearance(env, seq)
            if cur < 0 and min_clr <= cur:
                continue
            if cur >= 0 and min_clr < 0:
                continue
            prog = -np.linalg.norm(goal - xy_h)
            score = 8.0 * min_clr + prog
            if score > best_score:
                best_score = score
                best_a0 = seq[0].copy()

        return ShieldResult(
            a_exec=np.clip(best_a0, env.action_space.low, env.action_space.high),
            reject=True,
            a_prop=a_prop.copy(),
            clearance=self.clearance(env),
        )
