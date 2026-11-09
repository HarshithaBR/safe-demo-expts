"""Oracle intervention variants V1–V4 for shield-induced stuck recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from safe_demo_expts.envs.safety_point_goal import SafetyPointGoalEnv
from safe_demo_expts.interventions.oracle import RecoveryOracle, oracle_recovery_action


OracleFn = Callable[[SafetyPointGoalEnv], np.ndarray]


@dataclass
class InterventionState:
    active: bool = False
    name: str = "none"
    steps_taken: int = 0
    hold_action: Optional[np.ndarray] = None
    bad_action: Optional[np.ndarray] = None
    max_steps: int = 0
    done: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def make_oracle_fn(oracle_margin: float = 0.04) -> OracleFn:
    oracle = RecoveryOracle(oracle_margin=oracle_margin)
    return oracle


def action_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def default_variant_kwargs(name: str) -> Dict[str, Any]:
    name = name.upper() if name else "NONE"
    if name == "V2":
        return {"v2_n": 10}
    if name == "V3A":
        return {"v3_agree_tol": 0.35, "v3_max_steps": 40}
    if name == "V3B":
        return {"v3_disagree_tol": 0.45, "v3_max_steps": 40}
    if name == "V4":
        return {"v4_horizon": 20}
    return {}


class InterventionController:
    """
    Stateful V1–V4 controller.

    Call ``maybe_start`` when stuck (and budgets allow), then ``step`` each timestep
    while ``state.active``. Returns (action, used_oracle).
    """

    def __init__(
        self,
        variant: str = "none",
        oracle_fn: Optional[OracleFn] = None,
        oracle_margin: float = 0.04,
        v2_n: int = 10,
        v3_agree_tol: float = 0.35,
        v3_disagree_tol: float = 0.45,
        v3_max_steps: int = 40,
        v4_horizon: int = 20,
        clearance_fn: Optional[Callable[[SafetyPointGoalEnv], float]] = None,
    ):
        self.variant = (variant or "none").strip()
        self.oracle_fn = oracle_fn or make_oracle_fn(oracle_margin)
        self.v2_n = int(v2_n)
        self.v3_agree_tol = float(v3_agree_tol)
        self.v3_disagree_tol = float(v3_disagree_tol)
        self.v3_max_steps = int(v3_max_steps)
        self.v4_horizon = int(v4_horizon)
        self.clearance_fn = clearance_fn
        self.state = InterventionState(name=self.variant)

    def reset_episode(self) -> None:
        self.state = InterventionState(name=self.variant)

    @property
    def active(self) -> bool:
        return bool(self.state.active)

    def _clearance(self, env: SafetyPointGoalEnv) -> float:
        if self.clearance_fn is not None:
            return float(self.clearance_fn(env))
        # Prefer oracle shield clearance when available.
        if hasattr(self.oracle_fn, "clearance"):
            return float(self.oracle_fn.clearance(env))  # type: ignore[attr-defined]
        from safe_demo_expts.shield.safety_point_shield import SafetyPointShield

        return SafetyPointShield(margin=0.04, escape_mode=True).clearance(env)

    def maybe_start(
        self,
        env: SafetyPointGoalEnv,
        a_base: np.ndarray,
        stuck_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Begin an intervention burst if this variant is not ``none``."""
        del stuck_info
        name = self.variant.upper()
        if name in ("", "NONE", "NULL"):
            return False
        a_base = np.asarray(a_base, dtype=np.float64).reshape(2)
        a_orc = np.asarray(self.oracle_fn(env), dtype=np.float64).reshape(2)

        st = InterventionState(active=True, name=self.variant, steps_taken=0, done=False)
        if name == "V1":
            st.hold_action = a_orc
            st.max_steps = 1
        elif name == "V2":
            st.hold_action = a_orc
            st.max_steps = self.v2_n
        elif name == "V3A":
            st.hold_action = a_orc
            st.bad_action = a_base.copy()
            st.max_steps = self.v3_max_steps
        elif name == "V3B":
            st.hold_action = a_orc
            st.bad_action = a_base.copy()
            st.max_steps = self.v3_max_steps
        elif name == "V4":
            st.hold_action = a_orc
            st.max_steps = self.v4_horizon
        else:
            raise ValueError(f"unknown intervention variant={self.variant}")
        self.state = st
        return True

    def step(
        self,
        env: SafetyPointGoalEnv,
        a_base: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Advance one intervention step.

        Returns ``(action, used_oracle)``. When the intervention ends, sets
        ``state.active=False`` after returning the final oracle action for that step.
        """
        if not self.state.active:
            return np.asarray(a_base, dtype=np.float64).reshape(2), False

        name = self.variant.upper()
        a_base = np.asarray(a_base, dtype=np.float64).reshape(2)
        used = True

        if name == "V1":
            a = self.state.hold_action if self.state.hold_action is not None else self.oracle_fn(env)
            self.state.steps_taken += 1
            self.state.active = False
            self.state.done = True
            return np.asarray(a, dtype=np.float64).reshape(2), used

        if name == "V2":
            a = self.state.hold_action if self.state.hold_action is not None else self.oracle_fn(env)
            self.state.steps_taken += 1
            if self.state.steps_taken >= self.state.max_steps:
                self.state.active = False
                self.state.done = True
            return np.asarray(a, dtype=np.float64).reshape(2), used

        if name == "V3A":
            # Recompute oracle; stop when π ≈ oracle.
            a_orc = np.asarray(self.oracle_fn(env), dtype=np.float64).reshape(2)
            self.state.hold_action = a_orc
            self.state.steps_taken += 1
            if action_distance(a_base, a_orc) <= self.v3_agree_tol:
                self.state.active = False
                self.state.done = True
            elif self.state.steps_taken >= self.state.max_steps:
                self.state.active = False
                self.state.done = True
            return a_orc, used

        if name == "V3B":
            # Until π leaves the bad mode AND agent is safe (clearance >= 0).
            a_orc = np.asarray(self.oracle_fn(env), dtype=np.float64).reshape(2)
            self.state.hold_action = a_orc
            self.state.steps_taken += 1
            bad = self.state.bad_action if self.state.bad_action is not None else a_base
            left_bad = action_distance(a_base, bad) >= self.v3_disagree_tol
            safe = self._clearance(env) >= 0.0
            if left_bad and safe:
                self.state.active = False
                self.state.done = True
            elif self.state.steps_taken >= self.state.max_steps:
                self.state.active = False
                self.state.done = True
            return a_orc, used

        if name == "V4":
            # Trajectorylet: fresh oracle each step for H steps.
            a_orc = np.asarray(self.oracle_fn(env), dtype=np.float64).reshape(2)
            self.state.hold_action = a_orc
            self.state.steps_taken += 1
            if self.state.steps_taken >= self.state.max_steps:
                self.state.active = False
                self.state.done = True
            return a_orc, used

        raise ValueError(f"unknown intervention variant={self.variant}")


def build_controller(
    variant: Optional[str],
    oracle_margin: float = 0.04,
    **kwargs: Any,
) -> InterventionController:
    name = variant or "none"
    defaults = default_variant_kwargs(name)
    defaults.update({k: v for k, v in kwargs.items() if v is not None})
    return InterventionController(variant=name, oracle_margin=oracle_margin, **defaults)


# Convenience one-shot helpers (used by thin wrappers / tests) -----------------

def v1_one_step(env: SafetyPointGoalEnv, oracle_margin: float = 0.04) -> np.ndarray:
    return oracle_recovery_action(env, oracle_margin=oracle_margin)


def v2_hold_action(a_oracle: np.ndarray, n: int = 10) -> np.ndarray:
    """Return the action to hold; caller repeats ``n`` times."""
    del n
    return np.asarray(a_oracle, dtype=np.float64).reshape(2)


def v3a_should_stop(a_base: np.ndarray, a_oracle: np.ndarray, tol: float = 0.35) -> bool:
    return action_distance(a_base, a_oracle) <= tol


def v3b_should_stop(
    a_base: np.ndarray,
    bad_action: np.ndarray,
    clearance: float,
    tol: float = 0.45,
) -> bool:
    return action_distance(a_base, bad_action) >= tol and clearance >= 0.0
