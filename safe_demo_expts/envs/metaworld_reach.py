"""Meta-World Sawyer ``reach-v3`` wrapper with privileged EE / goal access.

Import is guarded: ``python -m py_compile`` and importlib work without metaworld.
Runtime construction raises a clear ImportError if metaworld / mujoco are missing.

Assumes Meta-World ≥ 2.x / 3.x Sawyer reach-v3 (4D delta+grip, action_scale≈0.01).
Prefers Gymnasium ``Meta-World/MT1``; falls back to legacy ``metaworld.ML1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import metaworld  # noqa: F401
except ImportError as e:  # pragma: no cover
    metaworld = None  # type: ignore
    _METAWORLD_IMPORT_ERROR: Optional[BaseException] = e
else:
    _METAWORLD_IMPORT_ERROR = None


# Reach-v3 hand box (matches Farama SawyerReachEnvV3).
DEFAULT_HAND_LOW = np.array([-0.5, 0.40, 0.05], dtype=np.float64)
DEFAULT_HAND_HIGH = np.array([0.5, 1.0, 0.5], dtype=np.float64)
DEFAULT_ACTION_SCALE = 0.01  # SawyerXYZEnv default 1/100
DEFAULT_SUCCESS_DIST = 0.05


@dataclass
class MetaWorldReachConfig:
    env_name: str = "reach-v3"
    max_steps: int = 200
    seed: int = 0
    stop_on_success: bool = True
    action_scale: float = DEFAULT_ACTION_SCALE
    success_dist: float = DEFAULT_SUCCESS_DIST
    hand_low: Tuple[float, float, float] = (-0.5, 0.40, 0.05)
    hand_high: Tuple[float, float, float] = (0.5, 1.0, 0.5)


def _require_metaworld() -> None:
    if _METAWORLD_IMPORT_ERROR is not None:
        raise ImportError(
            "MetaWorldReachEnv requires metaworld (+ mujoco). "
            "Activate .venv-tabletop (MUJOCO_GL=egl, PYTHONPATH=.). "
            "See docs/tabletop.md"
        ) from _METAWORLD_IMPORT_ERROR


def _unwrap(env: Any) -> Any:
    cur = env
    seen = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
            nxt = cur.unwrapped
            if nxt is cur:
                break
            cur = nxt
            continue
        if hasattr(cur, "env") and cur.env is not cur:
            cur = cur.env
            continue
        break
    return cur


def _make_reach_env(env_name: str, seed: int) -> Any:
    """Construct reach-v3 via Gymnasium MT1 or legacy ML1."""
    _require_metaworld()
    # New Farama Gymnasium API
    try:
        import gymnasium as gym

        try:
            env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed)
            return env
        except Exception:
            # Some builds register as metaworld/... or need ML1 task set.
            pass
    except ImportError:
        pass

    # Legacy ML1 API (metaworld < ~3 Gymnasium rewrite)
    import metaworld as mw

    if hasattr(mw, "ML1"):
        ml1 = mw.ML1(env_name)
        env_cls = ml1.train_classes[env_name]
        env = env_cls()
        task = ml1.train_tasks[0]
        env.set_task(task)
        if hasattr(env, "seed"):
            try:
                env.seed(seed)
            except Exception:
                pass
        return env

    raise ImportError(
        f"Could not construct Meta-World env {env_name!r}; "
        "need gymnasium Meta-World/MT1 or metaworld.ML1"
    )


class MetaWorldReachEnv:
    """Thin wrapper: reset/step + EE XYZ, goal, success, 4D actions."""

    def __init__(self, config: Optional[MetaWorldReachConfig] = None):
        self.cfg = config or MetaWorldReachConfig()
        self._env = _make_reach_env(self.cfg.env_name, self.cfg.seed)
        self._inner = _unwrap(self._env)
        obs_space = getattr(self._env, "observation_space", None)
        if obs_space is None:
            obs_space = getattr(self._inner, "observation_space", None)
        act_space = getattr(self._env, "action_space", None)
        if act_space is None:
            act_space = getattr(self._inner, "action_space", None)
        self.observation_size = int(np.prod(obs_space.shape)) if obs_space is not None else 39
        self.action_size = int(np.prod(act_space.shape)) if act_space is not None else 4
        self._steps = 0
        self._last_info: Dict[str, Any] = {}
        # Prefer env's native scale when present.
        native_scale = getattr(self._inner, "action_scale", None)
        if native_scale is not None:
            try:
                self.action_scale = float(np.asarray(native_scale).reshape(-1)[0])
            except Exception:
                self.action_scale = float(self.cfg.action_scale)
        else:
            self.action_scale = float(self.cfg.action_scale)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def hand_low(self) -> np.ndarray:
        hl = getattr(self._inner, "hand_low", None)
        if hl is not None:
            return np.asarray(hl, dtype=np.float64).reshape(3)
        return np.asarray(self.cfg.hand_low, dtype=np.float64)

    @property
    def hand_high(self) -> np.ndarray:
        hh = getattr(self._inner, "hand_high", None)
        if hh is not None:
            return np.asarray(hh, dtype=np.float64).reshape(3)
        return np.asarray(self.cfg.hand_high, dtype=np.float64)

    def ee_pos(self) -> np.ndarray:
        """End-effector / TCP position (x, y, z)."""
        inner = self._inner
        if hasattr(inner, "tcp_center"):
            try:
                return np.asarray(inner.tcp_center, dtype=np.float64).reshape(3)
            except Exception:
                pass
        if hasattr(inner, "get_endeff_pos"):
            return np.asarray(inner.get_endeff_pos(), dtype=np.float64).reshape(3)
        # Obs prefix is often hand XYZ.
        raise RuntimeError("cannot read end-effector position from Meta-World env")

    def ee_xy(self) -> np.ndarray:
        return self.ee_pos()[:2].copy()

    def goal_pos(self) -> np.ndarray:
        inner = self._inner
        for attr in ("_target_pos", "goal"):
            g = getattr(inner, attr, None)
            if g is not None:
                return np.asarray(g, dtype=np.float64).reshape(3)
        raise RuntimeError("cannot read goal position from Meta-World env")

    def goal_xy(self) -> np.ndarray:
        return self.goal_pos()[:2].copy()

    def _normalize_step_result(self, result) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Gymnasium: obs, reward, terminated, truncated, info
        # Old gym: obs, reward, done, info
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return (
                np.asarray(obs, dtype=np.float64),
                float(reward),
                bool(terminated),
                bool(truncated),
                dict(info or {}),
            )
        if len(result) == 4:
            obs, reward, done, info = result
            return (
                np.asarray(obs, dtype=np.float64),
                float(reward),
                bool(done),
                False,
                dict(info or {}),
            )
        raise RuntimeError(f"unexpected env.step return of length {len(result)}")

    def _augment_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info)
        ee = self.ee_pos()
        goal = self.goal_pos()
        dist = float(np.linalg.norm(ee - goal))
        success = bool(info.get("success", False))
        # Meta-World often stores success as float 0/1
        if not success:
            try:
                success = float(info.get("success", 0.0)) >= 1.0 - 1e-6
            except Exception:
                success = dist <= float(self.cfg.success_dist)
        else:
            success = bool(float(success) >= 1.0 - 1e-6) if not isinstance(success, bool) else success
        if not success and dist <= float(self.cfg.success_dist):
            success = True
        info["success"] = bool(success)
        info["ee_pos"] = ee
        info["ee_xy"] = ee[:2].copy()
        info["goal"] = goal
        info["goal_xy"] = goal[:2].copy()
        info["reach_dist"] = dist
        return info

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        seed = self.cfg.seed if seed is None else int(seed)
        # Gymnasium reset(seed=...)
        try:
            out = self._env.reset(seed=seed)
        except TypeError:
            if hasattr(self._env, "seed"):
                try:
                    self._env.seed(seed)
                except Exception:
                    pass
            # Legacy ML1: re-sample task for diversity
            try:
                import metaworld as mw

                if hasattr(mw, "ML1"):
                    ml1 = mw.ML1(self.cfg.env_name)
                    rng = np.random.default_rng(seed)
                    task = ml1.train_tasks[int(rng.integers(0, len(ml1.train_tasks)))]
                    self._env.set_task(task)
            except Exception:
                pass
            out = self._env.reset()

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            info = dict(info or {})
        else:
            obs, info = out, {}

        self._inner = _unwrap(self._env)
        self._steps = 0
        info = self._augment_info(info)
        self._last_info = info
        return np.asarray(obs, dtype=np.float64), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(self.action_size)
        low = np.asarray(self.action_space.low, dtype=np.float64)
        high = np.asarray(self.action_space.high, dtype=np.float64)
        action = np.clip(action, low, high)
        obs, reward, terminated, truncated, info = self._normalize_step_result(
            self._env.step(action.astype(np.float32))
        )
        self._steps += 1
        if self._steps >= int(self.cfg.max_steps):
            truncated = True
        info = self._augment_info(info)
        if info["success"] and self.cfg.stop_on_success:
            terminated = True
        self._last_info = info
        return obs, reward, bool(terminated), bool(truncated), info

    def predict_ee(self, action: np.ndarray) -> np.ndarray:
        """Kinematic one-step EE prediction (no env.step)."""
        action = np.asarray(action, dtype=np.float64).reshape(self.action_size)
        delta = action[:3] * float(self.action_scale)
        return self.ee_pos() + delta

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
