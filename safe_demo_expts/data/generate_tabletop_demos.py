"""Scripted expert demos for Meta-World Sawyer reach with controllable illegality.

Expert seeks the goal under a *demo* shield (relaxed pillar margin / z_min).
Each step is also scored by the *deploy* shield → ``reject_deploy`` for A/B/C.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from safe_demo_expts.envs.metaworld_reach import MetaWorldReachConfig, MetaWorldReachEnv
    from safe_demo_expts.shield.tabletop_shield import (
        DEFAULT_PILLAR_RADIUS,
        DEFAULT_PILLAR_XY,
        TabletopShield,
    )
except ImportError as _IMPORT_ERR:  # pragma: no cover
    MetaWorldReachConfig = None  # type: ignore
    MetaWorldReachEnv = None  # type: ignore
    TabletopShield = None  # type: ignore
    DEFAULT_PILLAR_XY = (0.0, 0.72)  # type: ignore
    DEFAULT_PILLAR_RADIUS = 0.08  # type: ignore
    _TABLETOP_IMPORT_ERROR: Optional[BaseException] = _IMPORT_ERR
else:
    _TABLETOP_IMPORT_ERROR = None


def _require_tabletop() -> None:
    if _TABLETOP_IMPORT_ERROR is not None:
        raise ImportError(
            "generate_tabletop_demos requires metaworld (.venv-tabletop). "
            "See docs/tabletop.md"
        ) from _TABLETOP_IMPORT_ERROR
    # Wrapper import succeeds without metaworld; construction must not.
    from safe_demo_expts.envs import metaworld_reach as _mw

    if getattr(_mw, "_METAWORLD_IMPORT_ERROR", None) is not None:
        raise ImportError(
            "generate_tabletop_demos requires metaworld (.venv-tabletop). "
            "See docs/tabletop.md"
        ) from _mw._METAWORLD_IMPORT_ERROR


def seek_goal_action(env: "MetaWorldReachEnv", gain: float = 0.85) -> np.ndarray:
    """Map EE→goal delta into Meta-World 4D delta+grip action."""
    delta = env.goal_pos() - env.ee_pos()
    dist = float(np.linalg.norm(delta))
    # Soften near the goal so we don't overshoot / oscilate.
    g = gain if dist > 0.08 else 0.45
    a = np.zeros(env.action_size, dtype=np.float64)
    scale = float(env.action_scale) + 1e-12
    a[:3] = np.clip((delta / scale) * g, -1.0, 1.0)
    # Reach does not need the gripper; keep open.
    if env.action_size > 3:
        a[3] = 1.0
    return a


def expert_action(env: "MetaWorldReachEnv", demo_shield: "TabletopShield") -> np.ndarray:
    """Privileged seek-goal expert clipped through the demo shield."""
    seek = seek_goal_action(env)
    sr = demo_shield(env, seek)
    if not sr.reject:
        return seek

    ee = env.ee_pos()
    goal = env.goal_pos()
    pillar = np.asarray(demo_shield.pillar_xy, dtype=np.float64)
    away = ee[:2] - pillar
    n = float(np.linalg.norm(away))
    if n < 1e-8:
        away = np.array([1.0, 0.0], dtype=np.float64)
    else:
        away = away / n
    tang = np.array([-away[1], away[0]], dtype=np.float64)
    to_goal_xy = goal[:2] - ee[:2]

    candidates: List[np.ndarray] = []
    for alpha in (0.7, 0.4, 0.2):
        c = seek.copy()
        c[:3] *= alpha
        candidates.append(c)
    for direction, scale in (
        (away, 0.9),
        (away, 0.5),
        (tang, 0.7),
        (-tang, 0.7),
        (to_goal_xy / (np.linalg.norm(to_goal_xy) + 1e-8), 0.6),
    ):
        c = seek.copy()
        c[0] = float(direction[0]) * scale
        c[1] = float(direction[1]) * scale
        candidates.append(c)
    # Lift then translate — useful when z_min is tight.
    lift = np.zeros(env.action_size, dtype=np.float64)
    lift[2] = 0.8
    if env.action_size > 3:
        lift[3] = 1.0
    candidates.append(lift)

    for cand in candidates:
        sr_c = demo_shield(env, cand)
        if not sr_c.reject:
            return cand
    return sr.a_exec.copy()


def generate_demos(
    n_traj: int = 40,
    deploy_margin: float = 0.05,
    demo_margin: float = 0.0,
    deploy_z_min: float = 0.05,
    demo_z_min: float = 0.05,
    pillar_xy: Sequence[float] = DEFAULT_PILLAR_XY,
    pillar_radius: float = DEFAULT_PILLAR_RADIUS,
    seed: int = 0,
    max_steps: int = 200,
    env_name: str = "reach-v3",
    max_attempts: Optional[int] = None,
    require_success: bool = False,
    save_path: Optional[PathLike] = None,
) -> Dict[str, np.ndarray]:
    """
    Collect ``n_traj`` scripted-expert episodes.

    Logs: obs, a_prop, a_exec_demo, a_exec_deploy, reject_deploy, traj_id.
    ``meta.illegal_rate`` = mean deploy-shield reject over logged steps.
    """
    _require_tabletop()

    cfg = MetaWorldReachConfig(env_name=env_name, max_steps=max_steps, seed=seed, stop_on_success=True)
    env = MetaWorldReachEnv(cfg)
    demo_shield = TabletopShield(
        margin=demo_margin,
        z_min=demo_z_min,
        pillar_xy=pillar_xy,
        pillar_radius=pillar_radius,
        mode="escape",
    )
    deploy_shield = TabletopShield(
        margin=deploy_margin,
        z_min=deploy_z_min,
        pillar_xy=pillar_xy,
        pillar_radius=pillar_radius,
        mode="escape",
    )

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
    kept = 0
    limit = int(max_attempts) if max_attempts is not None else max(n_traj * 20, n_traj + 5)

    while kept < n_traj and attempts < limit:
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

        if require_success and not success:
            continue

        tid = kept
        for i in range(len(ep_obs)):
            obs_l.append(ep_obs[i])
            a_prop_l.append(ep_prop[i])
            a_exec_demo_l.append(ep_demo[i])
            a_exec_deploy_l.append(ep_dep[i])
            reject_deploy_l.append(ep_rej[i])
            done_l.append(ep_done[i])
            traj_id_l.append(tid)
            success_flags.append(1.0 if success else 0.0)
        if success:
            success_count += 1
        kept += 1

    env.close()

    if kept < n_traj:
        raise RuntimeError(f"only collected {kept}/{n_traj} demos after {attempts} attempts")

    meta = {
        "n_traj": int(n_traj),
        "success_traj": int(success_count),
        "attempts": int(attempts),
        "deploy_z_min": float(deploy_z_min),
        "demo_z_min": float(demo_z_min),
        "pillar_xy": [float(pillar_xy[0]), float(pillar_xy[1])],
        "deploy_pillar_radius": float(pillar_radius),
        "deploy_pillar_margin": float(deploy_margin),
        "demo_pillar_margin": float(demo_margin),
        "illegal_rate": float(np.mean(reject_deploy_l)) if reject_deploy_l else 0.0,
        "seed": int(seed),
        "env": env_name,
        "max_steps": int(max_steps),
        "deploy_margin": float(deploy_margin),
        "demo_margin": float(demo_margin),
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
    """A/B/C label filtering — same contract as nav / SafetyPoint."""
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
    parser = argparse.ArgumentParser(description="Generate Meta-World Sawyer tabletop demos")
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "demos" / "s1_tabletop_harden.npz"))
    parser.add_argument("--n-traj", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-margin", type=float, default=0.0)
    parser.add_argument("--deploy-margin", type=float, default=0.05)
    parser.add_argument("--demo-z-min", type=float, default=0.05)
    parser.add_argument("--deploy-z-min", type=float, default=0.05)
    parser.add_argument("--pillar-radius", type=float, default=DEFAULT_PILLAR_RADIUS)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    print(
        f"Generating tabletop demos demo_margin={args.demo_margin} "
        f"deploy_margin={args.deploy_margin} -> {out}"
    )
    data = generate_demos(
        n_traj=args.n_traj,
        deploy_margin=args.deploy_margin,
        demo_margin=args.demo_margin,
        deploy_z_min=args.deploy_z_min,
        demo_z_min=args.demo_z_min,
        pillar_radius=args.pillar_radius,
        seed=args.seed,
        max_steps=args.max_steps,
        require_success=bool(args.require_success),
        save_path=out,
    )
    print(
        f"  success_traj={data['meta']['success_traj']}/{data['meta']['n_traj']} "
        f"illegal_rate={data['meta']['illegal_rate']:.4f} steps={len(data['obs'])}"
    )


if __name__ == "__main__":
    main()
