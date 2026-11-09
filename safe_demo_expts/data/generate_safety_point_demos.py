"""Scripted expert demos for SafetyPointGoal with controllable illegality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lazy / soft import so ``python -m py_compile`` and importlib work without MuJoCo.
try:
    from safe_demo_expts.envs.safety_point_goal import SafetyPointConfig, SafetyPointGoalEnv
    from safe_demo_expts.shield.safety_point_shield import SafetyPointShield
except ImportError as _IMPORT_ERR:  # pragma: no cover
    SafetyPointConfig = None  # type: ignore
    SafetyPointGoalEnv = None  # type: ignore
    SafetyPointShield = None  # type: ignore
    _SAFETY_IMPORT_ERROR = _IMPORT_ERR
else:
    _SAFETY_IMPORT_ERROR = None


# Deploy margin fixed at 0.14; illegal level only shrinks demo keep-out.
# Matches ``results/s1_matrix_results.json`` (low=0.10, mid=0.04, high=0.0).
S1_DEPLOY_MARGIN = 0.14
S1_DEMO_MARGINS = {"low": 0.10, "mid": 0.04, "high": 0.0}


def s1_margin_pair(illegal_level: str, deploy_margin: float = S1_DEPLOY_MARGIN) -> Tuple[float, float]:
    if illegal_level not in S1_DEMO_MARGINS:
        raise ValueError(f"unknown illegal_level={illegal_level}")
    # Scale demo margin if deploy_margin differs from the paper default.
    base_demo = S1_DEMO_MARGINS[illegal_level]
    if abs(deploy_margin - S1_DEPLOY_MARGIN) < 1e-12:
        return float(base_demo), float(deploy_margin)
    gap = S1_DEPLOY_MARGIN - base_demo
    demo = max(0.0, deploy_margin - gap)
    return float(demo), float(deploy_margin)


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _body_xy_to_action(env: "SafetyPointGoalEnv", world_dir: np.ndarray, speed: float = 0.85) -> np.ndarray:
    """Map a world-frame desired direction into [forward, turn]."""
    world_dir = np.asarray(world_dir, dtype=np.float64).reshape(2)
    n = np.linalg.norm(world_dir)
    if n < 1e-8:
        return np.zeros(2, dtype=np.float64)
    desired_yaw = float(np.arctan2(world_dir[1], world_dir[0]))
    err = _wrap_angle(desired_yaw - env.yaw())
    turn = float(np.clip(err / 0.55, -1.0, 1.0))
    forward = float(np.clip(speed * (1.0 - 0.75 * min(1.0, abs(err) / np.pi)), -1.0, 1.0))
    # Prefer reversing slightly when the goal/hazard is nearly behind.
    if abs(err) > 2.2:
        forward = float(np.clip(-0.35, -1.0, 1.0))
    return np.array([forward, turn], dtype=np.float64)


def seek_goal_action(env: "SafetyPointGoalEnv") -> np.ndarray:
    delta = env.goal_xy() - env.agent_xy()
    dist = float(np.linalg.norm(delta))
    speed = 0.95 if dist > 0.4 else 0.55
    return _body_xy_to_action(env, delta, speed=speed)


def expert_action(env: "SafetyPointGoalEnv", demo_shield: "SafetyPointShield") -> np.ndarray:
    """
    Privileged seek-goal expert; clips through the *demo* shield (relaxed keep-out).

    Returns the action that will be executed under the demo shield (label source for A/B).
    """
    seek = seek_goal_action(env)
    sr = demo_shield(env, seek)
    if not sr.reject:
        return seek

    # Try tangential / slowed alternatives before accepting projection.
    mat = np.asarray(env.task.agent.mat, dtype=np.float64)
    body_x = mat[:2, 0]
    body_y = mat[:2, 1]
    candidates = [
        np.array([0.5 * seek[0], seek[1]], dtype=np.float64),
        np.array([0.25 * seek[0], seek[1]], dtype=np.float64),
        _body_xy_to_action(env, body_y, speed=0.7),
        _body_xy_to_action(env, -body_y, speed=0.7),
        _body_xy_to_action(env, -body_x, speed=0.55),
        np.array([0.0, float(np.sign(seek[1]) or 1.0)], dtype=np.float64),
    ]
    for cand in candidates:
        sr_c = demo_shield(env, cand)
        if not sr_c.reject:
            return cand
    return sr.a_exec.copy()


def _require_safety() -> None:
    if _SAFETY_IMPORT_ERROR is not None:
        raise ImportError(
            "generate_safety_point_demos requires safety-gymnasium "
            "(activate .venv-safety; see docs/MAC_SETUP.md)"
        ) from _SAFETY_IMPORT_ERROR


def generate_demos(
    n_traj: int = 40,
    deploy_margin: float = S1_DEPLOY_MARGIN,
    demo_margin: float = 0.10,
    seed: int = 0,
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
    escape_mode_demo: bool = True,
    escape_mode_deploy: bool = True,
    max_attempts: Optional[int] = None,
    save_path: Optional[PathLike] = None,
) -> Dict[str, np.ndarray]:
    """
    Collect ``n_traj`` *successful* scripted-expert episodes.

    Logs per transition: obs, a_prop, a_exec_demo, a_exec_deploy, reject_deploy, traj_id.
    ``meta`` includes ``illegal_rate`` = mean deploy-shield reject over logged steps.
    """
    _require_safety()

    cfg = SafetyPointConfig(env_id=env_id, max_steps=max_steps, seed=seed, stop_on_first_goal=True)
    env = SafetyPointGoalEnv(cfg)
    demo_shield = SafetyPointShield(margin=demo_margin, escape_mode=escape_mode_demo)
    deploy_shield = SafetyPointShield(margin=deploy_margin, escape_mode=escape_mode_deploy)

    obs_l: List[np.ndarray] = []
    a_prop_l: List[np.ndarray] = []
    a_exec_demo_l: List[np.ndarray] = []
    a_exec_deploy_l: List[np.ndarray] = []
    reject_deploy_l: List[float] = []
    done_l: List[float] = []
    traj_id_l: List[int] = []
    success_flags: List[float] = []

    success_count = 0
    attempts = 0
    limit = int(max_attempts) if max_attempts is not None else max(n_traj * 25, n_traj + 10)

    while success_count < n_traj and attempts < limit:
        tid = success_count
        ep_seed = seed + 1000 + attempts
        attempts += 1
        o, _ = env.reset(seed=ep_seed)

        ep_obs: List[np.ndarray] = []
        ep_prop: List[np.ndarray] = []
        ep_demo: List[np.ndarray] = []
        ep_dep: List[np.ndarray] = []
        ep_rej: List[float] = []
        ep_done: List[float] = []
        success = False

        for _t in range(max_steps):
            a_demo = expert_action(env, demo_shield)
            sr_dep = deploy_shield(env, a_demo)

            ep_obs.append(o.copy())
            ep_prop.append(a_demo.copy())
            ep_demo.append(a_demo.copy())
            ep_dep.append(sr_dep.a_exec.copy())
            ep_rej.append(1.0 if sr_dep.reject else 0.0)

            o, _r, terminated, truncated, info = env.step(a_demo)
            ep_done.append(1.0 if (terminated or truncated) else 0.0)
            if info.get("success"):
                success = True
                break
            if terminated or truncated:
                break

        if not success:
            continue

        for i in range(len(ep_obs)):
            obs_l.append(ep_obs[i])
            a_prop_l.append(ep_prop[i])
            a_exec_demo_l.append(ep_demo[i])
            a_exec_deploy_l.append(ep_dep[i])
            reject_deploy_l.append(ep_rej[i])
            done_l.append(ep_done[i])
            traj_id_l.append(tid)
            success_flags.append(1.0)
        success_count += 1

    env.close()

    if success_count < n_traj:
        raise RuntimeError(
            f"only collected {success_count}/{n_traj} successful demos after {attempts} attempts"
        )

    meta = {
        "n_traj": int(n_traj),
        "success_traj": int(success_count),
        "attempts": int(attempts),
        "deploy_margin": float(deploy_margin),
        "demo_margin": float(demo_margin),
        "illegal_rate": float(np.mean(reject_deploy_l)) if reject_deploy_l else 0.0,
        "seed": int(seed),
        "env_id": env_id,
        "max_steps": int(max_steps),
    }
    data: Dict[str, np.ndarray] = {
        "obs": np.stack(obs_l),
        "a_prop": np.stack(a_prop_l),
        "a_exec_demo": np.stack(a_exec_demo_l),
        "a_exec_deploy": np.stack(a_exec_deploy_l),
        "reject_deploy": np.asarray(reject_deploy_l, dtype=np.float64),
        "done": np.asarray(done_l, dtype=np.float64),
        "traj_id": np.asarray(traj_id_l, dtype=np.int64),
        "success": np.asarray(success_flags, dtype=np.float64),
        "meta": meta,  # type: ignore[dict-item]
    }

    if save_path is not None:
        save_demos_npz(data, save_path)
    return data


def save_demos_npz(data: Dict, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = data.get("meta", {})
    arrays = {k: v for k, v in data.items() if k != "meta" and isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays, meta_json=np.asarray(json.dumps(meta)))
    return path


def load_demos_npz(path: PathLike) -> Dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if k != "meta_json"}
        if "meta_json" in z.files:
            data["meta"] = json.loads(str(z["meta_json"]))
        else:
            data["meta"] = {}
    return data


def filter_dataset(data: Dict, mode: str = "A") -> Dict:
    """A/B/C label filtering — same contract as S0 ``generate_demos.filter_dataset``."""
    obs = data["obs"]
    reject = np.asarray(data["reject_deploy"]).astype(bool)
    if mode == "A":
        keep = ~reject
        labels = data["a_prop"][keep]
        out_obs = obs[keep]
        hist_reject = np.asarray(data["reject_deploy"])[keep]
    elif mode == "B":
        keep = np.ones(len(obs), dtype=bool)
        labels = data["a_prop"]
        out_obs = obs
        hist_reject = np.asarray(data["reject_deploy"])
    elif mode == "C":
        keep = np.ones(len(obs), dtype=bool)
        labels = data["a_exec_deploy"]
        out_obs = obs
        hist_reject = np.asarray(data["reject_deploy"])
    else:
        raise ValueError(mode)

    return {
        "obs": out_obs,
        "actions": labels,
        "reject_deploy": hist_reject,
        "traj_id": np.asarray(data["traj_id"])[keep],
        "kept_frac": float(keep.mean()) if len(keep) else 0.0,
        "mode": mode,
        "meta": data.get("meta", {}),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate S1 SafetyPointGoal demo banks")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "demos"))
    parser.add_argument("--n-traj", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deploy-margin", type=float, default=S1_DEPLOY_MARGIN)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--env-id", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument(
        "--levels",
        type=str,
        default="low,mid,high",
        help="Comma-separated illegal levels to generate",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in [s.strip() for s in args.levels.split(",") if s.strip()]:
        demo_m, deploy_m = s1_margin_pair(level, deploy_margin=args.deploy_margin)
        out = out_dir / f"s1_pointgoal1_{level}_n{args.n_traj}_seed{args.seed}.npz"
        print(f"Generating {level}: demo_margin={demo_m} deploy_margin={deploy_m} -> {out}")
        data = generate_demos(
            n_traj=args.n_traj,
            deploy_margin=deploy_m,
            demo_margin=demo_m,
            seed=args.seed,
            max_steps=args.max_steps,
            env_id=args.env_id,
            save_path=out,
        )
        print(
            f"  success_traj={data['meta']['success_traj']} "
            f"illegal_rate={data['meta']['illegal_rate']:.4f} "
            f"steps={len(data['obs'])}"
        )


if __name__ == "__main__":
    main()
