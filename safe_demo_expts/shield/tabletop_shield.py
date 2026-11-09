"""Workspace + vertical-pillar shield for Meta-World Sawyer reach.

Keep-out: cylindrical pillar at ``pillar_xy`` with ``radius + margin``, plus
workspace box (XY from env hand limits, Z ≥ ``z_min``).

Modes:
  - freeze: reject → zero action
  - escape: reject → maximize predicted clearance (goal as tie-break)
  - mpc: short-horizon kinematic sample / replanning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from safe_demo_expts.envs.metaworld_reach import MetaWorldReachEnv


DEFAULT_PILLAR_XY = (0.0, 0.72)
DEFAULT_PILLAR_RADIUS = 0.08


@dataclass
class ShieldResult:
    a_exec: np.ndarray
    reject: bool
    a_prop: np.ndarray
    clearance: float


class TabletopShield:
    """Geometric filter; no speculative ``env.step``."""

    def __init__(
        self,
        margin: float = 0.05,
        z_min: float = 0.05,
        pillar_xy: Sequence[float] = DEFAULT_PILLAR_XY,
        pillar_radius: float = DEFAULT_PILLAR_RADIUS,
        mode: str = "escape",
        n_alternatives: int = 16,
        mpc_horizon: int = 3,
        mpc_samples: int = 24,
        workspace_xy_margin: float = 0.0,
    ):
        self.margin = float(margin)
        self.z_min = float(z_min)
        self.pillar_xy = np.asarray(pillar_xy, dtype=np.float64).reshape(2)
        self.pillar_radius = float(pillar_radius)
        self.mode = str(mode).lower()
        self.n_alternatives = int(n_alternatives)
        self.mpc_horizon = int(mpc_horizon)
        self.mpc_samples = int(mpc_samples)
        self.workspace_xy_margin = float(workspace_xy_margin)

    # --- geometry ---------------------------------------------------------

    def pillar_clearance(self, xy: np.ndarray) -> float:
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        r = self.pillar_radius + self.margin
        return float(np.linalg.norm(xy - self.pillar_xy) - r)

    def workspace_clearance(self, env: MetaWorldReachEnv, pos: np.ndarray) -> float:
        """Positive inside the inflated workspace box (z ≥ z_min, XY in hand box)."""
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        low = env.hand_low.copy()
        high = env.hand_high.copy()
        low[2] = max(float(low[2]), self.z_min)
        # Optional inward XY margin so demos can scrape the rim under soft margin=0.
        low[:2] = low[:2] + self.workspace_xy_margin
        high[:2] = high[:2] - self.workspace_xy_margin
        # Distance to each face; min over faces.
        d_low = pos - low
        d_high = high - pos
        return float(np.min(np.concatenate([d_low, d_high])))

    def clearance(self, env: MetaWorldReachEnv, pos: Optional[np.ndarray] = None) -> float:
        pos = env.ee_pos() if pos is None else np.asarray(pos, dtype=np.float64).reshape(3)
        return float(min(self.pillar_clearance(pos[:2]), self.workspace_clearance(env, pos)))

    def predict_pos(self, env: MetaWorldReachEnv, action: np.ndarray) -> np.ndarray:
        return env.predict_ee(action)

    def _clip_action(self, env: MetaWorldReachEnv, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64).reshape(env.action_size)
        return np.clip(a, env.action_space.low, env.action_space.high).astype(np.float64)

    def _is_safe_transition(self, env: MetaWorldReachEnv, action: np.ndarray) -> bool:
        cur = self.clearance(env)
        pred = self.clearance(env, self.predict_pos(env, action))
        # Allow tiny penetration hysteresis so numerical noise doesn't thrash.
        return pred >= 0.0 and cur >= -0.02

    # --- modes ------------------------------------------------------------

    def _freeze(self, env: MetaWorldReachEnv, a_prop: np.ndarray) -> ShieldResult:
        z = np.zeros(env.action_size, dtype=np.float64)
        return ShieldResult(
            a_exec=z,
            reject=True,
            a_prop=a_prop.copy(),
            clearance=self.clearance(env),
        )

    def _candidate_actions(self, env: MetaWorldReachEnv, a_prop: np.ndarray) -> List[np.ndarray]:
        a_prop = np.asarray(a_prop, dtype=np.float64).reshape(env.action_size)
        cands: List[np.ndarray] = []
        for alpha in np.linspace(0.9, 0.0, 8):
            c = a_prop.copy()
            c[:3] *= float(alpha)
            cands.append(c)
        # Radial escape from pillar in XY; preserve z/grip guesses.
        ee = env.ee_pos()
        away = ee[:2] - self.pillar_xy
        n = float(np.linalg.norm(away))
        if n < 1e-8:
            away = np.array([1.0, 0.0], dtype=np.float64)
            n = 1.0
        away = away / n
        tangents = [
            away,
            -away,
            np.array([-away[1], away[0]], dtype=np.float64),
            np.array([away[1], -away[0]], dtype=np.float64),
        ]
        to_goal = env.goal_pos() - ee
        if float(np.linalg.norm(to_goal)) > 1e-8:
            tangents.append(to_goal[:2] / (np.linalg.norm(to_goal[:2]) + 1e-8))
            tangents.append(to_goal / (np.linalg.norm(to_goal) + 1e-8))  # type: ignore[arg-type]
        for direction in tangents:
            d = np.asarray(direction, dtype=np.float64).reshape(-1)
            for scale in (1.0, 0.6, 0.3):
                c = a_prop.copy()
                if d.size == 2:
                    c[0] = float(d[0]) * scale
                    c[1] = float(d[1]) * scale
                else:
                    c[:3] = d[:3] * scale
                cands.append(c)
        # Pure Z lifts (escape floor / scrape).
        for dz in (1.0, 0.5, -0.5):
            c = np.zeros(env.action_size, dtype=np.float64)
            c[2] = dz
            c[3] = a_prop[3] if env.action_size > 3 else 0.0
            cands.append(c)
        cands.append(np.zeros(env.action_size, dtype=np.float64))
        # Random probes
        rng = np.random.default_rng(0)
        for _ in range(self.n_alternatives):
            c = rng.uniform(-1.0, 1.0, size=env.action_size)
            c[3] = a_prop[3] if env.action_size > 3 else 0.0
            cands.append(c)
        return cands

    def _score(self, env: MetaWorldReachEnv, cand: np.ndarray, cur_clr: float) -> Optional[float]:
        pred = self.predict_pos(env, cand)
        clr = self.clearance(env, pred)
        if cur_clr < 0.0:
            if clr <= cur_clr:
                return None
        elif clr < 0.0:
            return None
        prog = -float(np.linalg.norm(env.goal_pos() - pred))
        return 8.0 * clr + prog

    def _escape(self, env: MetaWorldReachEnv, a_prop: np.ndarray) -> ShieldResult:
        cur = self.clearance(env)
        best = None
        best_score = -1e18
        for cand in self._candidate_actions(env, a_prop):
            cand = self._clip_action(env, cand)
            score = self._score(env, cand, cur)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best = cand
        if best is None:
            return self._freeze(env, a_prop)
        return ShieldResult(
            a_exec=best.copy(),
            reject=True,
            a_prop=a_prop.copy(),
            clearance=self.clearance(env, self.predict_pos(env, best)),
        )

    def _roll_clearance(
        self, env: MetaWorldReachEnv, actions: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        pos = env.ee_pos().copy()
        scale = float(env.action_scale)
        min_clr = 1e9
        for a in actions:
            a = np.asarray(a, dtype=np.float64).reshape(env.action_size)
            pos = pos + a[:3] * scale
            # Soft clip to workspace for rollout stability
            pos = np.minimum(np.maximum(pos, env.hand_low), env.hand_high)
            min_clr = min(min_clr, self.clearance(env, pos))
        return float(min_clr), pos

    def _mpc(self, env: MetaWorldReachEnv, a_prop: np.ndarray) -> ShieldResult:
        # Prefer escape seed; then try short open-loop samples.
        base = self._escape(env, a_prop)
        cur = self.clearance(env)
        goal = env.goal_pos()
        best_a0 = base.a_exec.copy()
        best_score = -1e18
        # Score the escape action as a 1-step plan
        pred0 = self.predict_pos(env, best_a0)
        best_score = 8.0 * self.clearance(env, pred0) - float(np.linalg.norm(goal - pred0))

        rng = np.random.default_rng(0)
        H = self.mpc_horizon
        for _ in range(self.mpc_samples):
            seq = rng.uniform(-1.0, 1.0, size=(H, env.action_size))
            seq[0] = 0.5 * seq[0] + 0.5 * a_prop
            seq[:, 3] = a_prop[3] if env.action_size > 3 else 0.0
            min_clr, xy_h = self._roll_clearance(env, seq)
            if cur < 0.0 and min_clr <= cur:
                continue
            if cur >= 0.0 and min_clr < 0.0:
                continue
            prog = -float(np.linalg.norm(goal - xy_h))
            score = 8.0 * min_clr + prog
            if score > best_score:
                best_score = score
                best_a0 = self._clip_action(env, seq[0])

        return ShieldResult(
            a_exec=best_a0,
            reject=True,
            a_prop=a_prop.copy(),
            clearance=self.clearance(env),
        )

    def __call__(self, env: MetaWorldReachEnv, a_prop: np.ndarray) -> ShieldResult:
        a_prop = self._clip_action(env, a_prop)
        if self._is_safe_transition(env, a_prop):
            return ShieldResult(
                a_exec=a_prop.copy(),
                reject=False,
                a_prop=a_prop.copy(),
                clearance=self.clearance(env),
            )
        mode = self.mode
        if mode in ("freeze", "reject", "hold"):
            return self._freeze(env, a_prop)
        if mode in ("mpc", "a3", "mpc_escape"):
            return self._mpc(env, a_prop)
        # default escape
        return self._escape(env, a_prop)


def make_tabletop_shield(
    mode: str = "escape",
    margin: float = 0.05,
    z_min: float = 0.05,
    pillar_xy: Sequence[float] = DEFAULT_PILLAR_XY,
    pillar_radius: float = DEFAULT_PILLAR_RADIUS,
    **kwargs,
) -> TabletopShield:
    return TabletopShield(
        margin=margin,
        z_min=z_min,
        pillar_xy=pillar_xy,
        pillar_radius=pillar_radius,
        mode=mode,
        **kwargs,
    )
