#!/usr/bin/env python3
"""Run S1 tabletop harden: A/B/C × freeze/escape/MPC over seeds {7,77,177}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_demo_expts.data.generate_tabletop_demos import (
    filter_dataset,
    generate_demos,
    load_demos_npz,
)
from safe_demo_expts.models.bc import train_bc
from safe_demo_expts.pipelines.evaluate_tabletop import evaluate_policy
from safe_demo_expts.util_ckpt import save_policy


ROBOT_TAG = "Meta-World Sawyer (reach-v3) — NOT Franka"


def demo_path(demo_dir: Path, cfg: dict) -> Path:
    dm = cfg["demo_margin"]
    pm = cfg["deploy_margin"]
    n = cfg["n_demos"]
    seed = cfg.get("demo_seed", 0)
    return demo_dir / f"s1_tabletop_dm{dm}_pm{pm}_n{n}_seed{seed}.npz"


def load_or_generate_demos(cfg: dict, demo_dir: Path) -> tuple[dict, str]:
    path = demo_path(demo_dir, cfg)
    if path.exists():
        raw = load_demos_npz(path)
        meta = dict(raw.get("meta") or {})
        meta.setdefault("demo_margin", cfg["demo_margin"])
        meta.setdefault("deploy_margin", cfg["deploy_margin"])
        meta.setdefault("demo_pillar_margin", cfg["demo_margin"])
        meta.setdefault("deploy_pillar_margin", cfg["deploy_margin"])
        meta.setdefault(
            "illegal_rate",
            float(np.mean(raw["reject_deploy"])) if len(raw["reject_deploy"]) else 0.0,
        )
        raw["meta"] = meta
        return raw, str(path)

    print(f"  demo bank missing; generating {path}")
    raw = generate_demos(
        n_traj=int(cfg["n_demos"]),
        deploy_margin=float(cfg["deploy_margin"]),
        demo_margin=float(cfg["demo_margin"]),
        deploy_z_min=float(cfg.get("deploy_z_min", 0.05)),
        demo_z_min=float(cfg.get("demo_z_min", 0.05)),
        pillar_xy=cfg.get("pillar_xy", [0.0, 0.72]),
        pillar_radius=float(cfg.get("pillar_radius", 0.08)),
        seed=int(cfg.get("demo_seed", 0)),
        max_steps=int(cfg.get("max_steps", 200)),
        env_name=cfg.get("env_name", "reach-v3"),
        require_success=bool(cfg.get("require_success", False)),
        save_path=path,
    )
    return raw, str(path)


def _n_val_traj(traj_id: np.ndarray, val_frac: float, seed: int) -> int:
    uniq = np.unique(traj_id)
    if len(uniq) < 5 or val_frac <= 0:
        return 0
    n_val = max(1, int(round(len(uniq) * val_frac)))
    return int(n_val)


def train_mode(
    cfg: dict,
    raw: dict,
    mode: str,
    ckpt_dir: Path,
) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
    ds = filter_dataset(raw, mode=mode)
    if len(ds["obs"]) < 16:
        raise RuntimeError(f"too_few_samples_after_filter mode={mode} n={len(ds['obs'])}")

    policy, train_metrics = train_bc(
        obs=ds["obs"],
        actions=ds["actions"],
        reject=ds["reject_deploy"],
        traj_id=ds["traj_id"],
        use_history=False,
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg.get("batch_size", 256)),
        lr=float(cfg.get("lr", 1e-3)),
        device=str(cfg.get("device", "cpu")),
        seed=int(cfg.get("demo_seed", 0)),
        val_frac=float(cfg.get("val_frac", 0.2)),
    )

    n_train_traj = int(len(np.unique(ds["traj_id"]))) if len(ds["traj_id"]) else 0
    n_val_traj = _n_val_traj(ds["traj_id"], float(cfg.get("val_frac", 0.2)), int(cfg.get("demo_seed", 0)))
    # train_bc holds out ~val_frac of trajs; n_train_traj reported as remaining.
    if n_val_traj and n_train_traj >= n_val_traj:
        n_train_traj_out = n_train_traj - n_val_traj
    else:
        n_train_traj_out = n_train_traj

    stem = f"s1_table_illegal_mode{mode}_seed{int(cfg.get('demo_seed', 0))}"
    ckpt_path = ckpt_dir / f"{stem}.pt"
    meta = {
        "mode": mode,
        "seed": int(cfg.get("demo_seed", 0)),
        "obs_dim": int(ds["obs"].shape[1]),
        "act_dim": int(ds["actions"].shape[1]),
        "history_extra": 0,
        "use_history": False,
        "deploy_margin": float(cfg["deploy_margin"]),
        "demo_margin": float(cfg["demo_margin"]),
        "env_name": cfg.get("env_name", "reach-v3"),
        "tag": stem,
        "robot": ROBOT_TAG,
    }
    save_policy(policy, ckpt_path, meta=meta, write_sidecar=True)

    extras = {
        "kept_frac": float(ds["kept_frac"]),
        "illegal_rate": float(raw["meta"].get("illegal_rate", 0.0)),
        "demo_success": int(raw["meta"].get("success_traj", cfg["n_demos"])),
        "n_train_traj": int(n_train_traj_out),
        "n_val_traj": int(n_val_traj),
        "ckpt": str(ckpt_path),
        **train_metrics,
    }
    return policy, extras, ds


def run_matrix(cfg: dict, demo_dir: Path, ckpt_dir: Path) -> Dict[str, Any]:
    raw, demo_source = load_or_generate_demos(cfg, demo_dir)
    print(f"demos: {demo_source} illegal_rate={raw['meta'].get('illegal_rate')} "
          f"success={raw['meta'].get('success_traj')}/{raw['meta'].get('n_traj')}")

    modes = list(cfg.get("modes", ["A", "B", "C"]))
    shields = list(cfg.get("shields", ["freeze", "escape", "mpc"]))
    eval_seeds = [int(s) for s in cfg.get("eval_seeds", [7, 77, 177])]

    results: List[Dict[str, Any]] = []
    trained: Dict[str, Any] = {}

    for mode in modes:
        print(f"=== train mode={mode} ===")
        policy, extras, _ds = train_mode(cfg, raw, mode, ckpt_dir)
        trained[mode] = (policy, extras)

        for shield in shields:
            per_seed: List[Dict[str, Any]] = []
            for seed in eval_seeds:
                print(f"  eval mode={mode} shield={shield} seed={seed}")
                metrics = evaluate_policy(
                    policy,
                    deploy_margin=float(cfg["deploy_margin"]),
                    z_min=float(cfg.get("deploy_z_min", 0.05)),
                    pillar_xy=cfg.get("pillar_xy", [0.0, 0.72]),
                    pillar_radius=float(cfg.get("pillar_radius", 0.08)),
                    shield=shield,
                    n_episodes=int(cfg.get("n_eval", 20)),
                    seed=int(seed),
                    stuck_horizon=int(cfg.get("stuck_horizon", 40)),
                    device=str(cfg.get("device", "cpu")),
                    max_steps=int(cfg.get("max_steps", 200)),
                    env_name=cfg.get("env_name", "reach-v3"),
                    mpc_horizon=int(cfg.get("mpc_horizon", 3)),
                    mpc_samples=int(cfg.get("mpc_samples", 24)),
                )
                per_seed.append(
                    {
                        "seed": int(seed),
                        "success_rate": float(metrics["success_rate"]),
                        "ever_stuck_rate": float(metrics["ever_stuck_rate"]),
                        "terminal_stuck_rate": float(metrics["terminal_stuck_rate"]),
                        "mean_reject_rate": float(metrics["mean_reject_rate"]),
                        "mean_length": float(metrics["mean_length"]),
                        "n_episodes": float(metrics["n_episodes"]),
                    }
                )

            succ = np.asarray([r["success_rate"] for r in per_seed], dtype=np.float64)
            stuck = np.asarray([r["ever_stuck_rate"] for r in per_seed], dtype=np.float64)
            row = {
                "mode": mode,
                "shield": shield,
                "kept_frac": extras["kept_frac"],
                "illegal_rate": extras["illegal_rate"],
                "demo_success": extras["demo_success"],
                "success_mean": float(succ.mean()),
                "success_std": float(succ.std(ddof=0)),
                "stuck_mean": float(stuck.mean()),
                "stuck_std": float(stuck.std(ddof=0)),
                "per_seed": per_seed,
                "train_mse": extras["train_mse"],
                "val_mse": extras["val_mse"],
                "n_train": extras["n_train"],
                "n_val": extras["n_val"],
                "n_train_traj": extras["n_train_traj"],
                "n_val_traj": extras["n_val_traj"],
                "epochs": extras["epochs"],
            }
            results.append(row)
            print(
                f"  -> mode={mode} shield={shield} "
                f"succ={row['success_mean']:.2f}±{row['success_std']:.2f} "
                f"stuck={row['stuck_mean']:.2f}±{row['stuck_std']:.2f}"
            )

    out_obj = {
        "robot": cfg.get("robot", ROBOT_TAG),
        "config": {
            "n_demos": int(cfg["n_demos"]),
            "deploy_margin": float(cfg["deploy_margin"]),
            "demo_margin": float(cfg["demo_margin"]),
            "pillar_radius": float(cfg.get("pillar_radius", 0.08)),
            "eval_seeds": eval_seeds,
            "modes": modes,
        },
        "demo_meta": dict(raw.get("meta") or {}),
        "results": results,
        "demo_source": demo_source,
    }
    return out_obj


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="S1 tabletop illegal-demo harden")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_tabletop_harden.yaml"))
    parser.add_argument("--demo-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "s1_tabletop_harden.json"))
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--demo-margin", type=float, default=None)
    parser.add_argument("--deploy-margin", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.demo_margin is not None:
        cfg["demo_margin"] = float(args.demo_margin)
    if args.deploy_margin is not None:
        cfg["deploy_margin"] = float(args.deploy_margin)

    if args.smoke:
        cfg.update(
            {
                "n_demos": 4,
                "n_eval": 2,
                "epochs": 3,
                "eval_seeds": [7],
                "modes": ["B"],
                "shields": ["freeze", "mpc"],
            }
        )

    demo_dir = Path(args.demo_dir or cfg.get("demo_dir", "data/demos"))
    if not demo_dir.is_absolute():
        demo_dir = ROOT / demo_dir
    ckpt_dir = Path(args.checkpoint_dir or cfg.get("checkpoint_dir", "results/checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    demo_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = run_matrix(cfg, demo_dir, ckpt_dir)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")

    print("\nmode shield  succ_mean  stuck_mean")
    for r in payload["results"]:
        print(
            f"{r['mode']:4} {r['shield']:6}  {r['success_mean']:.2f}±{r['success_std']:.2f}  "
            f"{r['stuck_mean']:.2f}±{r['stuck_std']:.2f}"
        )


if __name__ == "__main__":
    main()
