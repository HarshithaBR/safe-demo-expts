"""Geometric hazard shield for Safety PointGoal (no speculative env.step)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from safe_demo_expts.envs.safety_point_goal import SafetyPointGoalEnv


@dataclass
class ShieldResult:
    a_exec: np.ndarray
    reject: bool
    a_prop: np.ndarray
    clearance: float


class SafetyPointShield:
    """Inflated-hazard keep-out filter using privileged positions."""

    def __init__(
        self,
        margin: float = 0.12,
        step_scale: float = 0.08,
        n_alternatives: int = 12,
        escape_mode: bool = True,
    ):
        self.margin = float(margin)
        self.step_scale = float(step_scale)
        self.n_alternatives = int(n_alternatives)
        self.escape_mode = bool(escape_mode)

    def clearance(self, env: SafetyPointGoalEnv, xy: np.ndarray | None = None) -> float:
        xy = env.agent_xy() if xy is None else np.asarray(xy, dtype=np.float64)
        haz = env.hazard_xy()
        r = env.hazard_radius() + self.margin
        d = np.linalg.norm(haz - xy[None, :], axis=1) - r
        return float(d.min()) if len(d) else 1e9

    def predict_xy(self, env: SafetyPointGoalEnv, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(2)
        mat = np.asarray(env.task.agent.mat, dtype=np.float64)
        body_x = mat[:2, 0]
        return env.agent_xy() + self.step_scale * float(action[0]) * body_x

    def __call__(self, env: SafetyPointGoalEnv, a_prop: np.ndarray) -> ShieldResult:
        a_prop = np.asarray(a_prop, dtype=np.float64).reshape(2)
        a_prop = np.clip(a_prop, env.action_space.low, env.action_space.high)

        pred = self.predict_xy(env, a_prop)
        if self.clearance(env, pred) >= 0.0 and self.clearance(env) >= -0.02:
            return ShieldResult(
                a_exec=a_prop.copy(),
                reject=False,
                a_prop=a_prop.copy(),
                clearance=self.clearance(env),
            )

        if not self.escape_mode:
            # freeze: hold still
            z = np.zeros(2, dtype=np.float64)
            return ShieldResult(
                a_exec=z,
                reject=True,
                a_prop=a_prop.copy(),
                clearance=self.clearance(env),
            )

        goal = env.goal_xy()
        best = None
        best_score = -1e18
        candidates: List[np.ndarray] = []
        for alpha in np.linspace(0.9, 0.0, 8):
            candidates.append(np.array([a_prop[0] * alpha, a_prop[1]], dtype=np.float64))
        for turn in np.linspace(-1.0, 1.0, self.n_alternatives):
            candidates.append(np.array([0.35, turn], dtype=np.float64))
            candidates.append(np.array([0.0, turn], dtype=np.float64))
        candidates.append(np.zeros(2, dtype=np.float64))

        for cand in candidates:
            cand = np.clip(cand, env.action_space.low, env.action_space.high)
            pxy = self.predict_xy(env, cand)
            clr = self.clearance(env, pxy)
            cur = self.clearance(env)
            if cur < 0:
                # recovery mode: require clearance increase
                if clr <= cur:
                    continue
            elif clr < 0:
                continue
            prog = -np.linalg.norm(goal - pxy)
            score = 8.0 * clr + prog
            if score > best_score:
                best_score = score
                best = cand

        if best is None:
            best = np.zeros(2, dtype=np.float64)
            return ShieldResult(
                a_exec=best,
                reject=True,
                a_prop=a_prop.copy(),
                clearance=self.clearance(env),
            )
        return ShieldResult(
            a_exec=best.copy(),
            reject=True,
            a_prop=a_prop.copy(),
            clearance=self.clearance(env, self.predict_xy(env, best)),
        )
