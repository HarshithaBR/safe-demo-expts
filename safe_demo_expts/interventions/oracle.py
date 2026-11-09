"""Privileged recovery oracle for SafetyPointGoal."""

from __future__ import annotations

from typing import Optional

import numpy as np

from safe_demo_expts.envs.safety_point_goal import SafetyPointGoalEnv
from safe_demo_expts.shield.safety_point_shield import SafetyPointShield


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _dir_to_action(env: SafetyPointGoalEnv, world_dir: np.ndarray, speed: float = 0.9) -> np.ndarray:
    world_dir = np.asarray(world_dir, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(world_dir))
    if n < 1e-8:
        return np.zeros(2, dtype=np.float64)
    desired = float(np.arctan2(world_dir[1], world_dir[0]))
    err = _wrap_angle(desired - env.yaw())
    turn = float(np.clip(err / 0.5, -1.0, 1.0))
    forward = float(np.clip(speed * (1.0 - 0.8 * min(1.0, abs(err) / np.pi)), -1.0, 1.0))
    if abs(err) > 2.0:
        forward = -0.4
    return np.array([forward, turn], dtype=np.float64)


def nearest_hazard_xy(env: SafetyPointGoalEnv) -> Optional[np.ndarray]:
    haz = env.hazard_xy()
    if len(haz) == 0:
        return None
    agent = env.agent_xy()
    d = np.linalg.norm(haz - agent[None, :], axis=1)
    return haz[int(np.argmin(d))].copy()


def oracle_recovery_action(
    env: SafetyPointGoalEnv,
    oracle_margin: float = 0.04,
    shield: Optional[SafetyPointShield] = None,
) -> np.ndarray:
    """
    Privileged recovery:
      - if clearance < 0: back away from nearest hazard
      - else: seek goal
    Action is projected through an ``oracle_margin`` shield (escape enabled).
    """
    if shield is None:
        shield = SafetyPointShield(margin=oracle_margin, escape_mode=True)
    clr = shield.clearance(env)
    if clr < 0.0:
        nearest = nearest_hazard_xy(env)
        if nearest is None:
            a_prop = np.array([-0.6, 0.0], dtype=np.float64)
        else:
            away = env.agent_xy() - nearest
            a_prop = _dir_to_action(env, away, speed=1.0)
    else:
        delta = env.goal_xy() - env.agent_xy()
        a_prop = _dir_to_action(env, delta, speed=0.95)

    sr = shield(env, a_prop)
    return sr.a_exec.copy()


class RecoveryOracle:
    """Callable wrapper used by intervention controllers / evaluate_flexible."""

    def __init__(self, oracle_margin: float = 0.04):
        self.oracle_margin = float(oracle_margin)
        self.shield = SafetyPointShield(margin=self.oracle_margin, escape_mode=True)

    def __call__(self, env: SafetyPointGoalEnv, obs: Optional[np.ndarray] = None) -> np.ndarray:
        del obs  # privileged geometry only
        return oracle_recovery_action(env, oracle_margin=self.oracle_margin, shield=self.shield)

    def clearance(self, env: SafetyPointGoalEnv) -> float:
        return self.shield.clearance(env)
