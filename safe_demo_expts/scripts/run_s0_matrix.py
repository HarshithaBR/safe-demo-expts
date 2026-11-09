#!/usr/bin/env python3
"""Run S0 PointNav matrix: pipelines A/B/C × illegal levels × optional history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# allow running without install
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_demo_expts.data.generate_demos import filter_dataset, generate_demos
from safe_demo_expts.models.bc import train_bc
from safe_demo_expts.pipelines.evaluate import evaluate_policy
from safe_demo_expts.shield.obstacle_shield import margin_pair


def run_one(cfg: dict, mode: str, illegal_level: str, use_history: bool, seed: int) -> dict:
    demo_m, deploy_m = margin_pair(cfg["deploy_margin"], illegal_level)
    # hug the wall more as illegality increases (still demo-feasible with smaller margin)
    bias = {"low": 0.0, "mid": 0.035, "high": 0.06}[illegal_level]
    raw = generate_demos(
        n_traj=cfg["n_demos"],
        deploy_margin=deploy_m,
        demo_margin=demo_m,
        seed=seed,
        max_steps=cfg["max_steps"],
        path_bias=bias,
    )
    ds = filter_dataset(raw, mode=mode)
    if len(ds["obs"]) < 16:
        return {
            "mode": mode,
            "illegal_level": illegal_level,
            "use_history": use_history,
            "seed": seed,
            "error": "too_few_samples_after_filter",
            "kept_frac": ds["kept_frac"],
            "illegal_rate": raw["meta"]["illegal_rate"],
        }

    policy, _train_stats = train_bc(
        obs=ds["obs"],
        actions=ds["actions"],
        reject=ds["reject_deploy"],
        use_history=use_history,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        device=cfg["device"],
        seed=seed,
    )
    metrics = evaluate_policy(
        policy,
        deploy_margin=deploy_m,
        n_episodes=cfg["n_eval"],
        seed=seed + 7,
        stuck_horizon=cfg["stuck_horizon"],
        device=cfg["device"],
        max_steps=cfg["max_steps"],
    )
    return {
        "mode": mode,
        "illegal_level": illegal_level,
        "use_history": use_history,
        "seed": seed,
        "demo_margin": demo_m,
        "deploy_margin": deploy_m,
        "illegal_rate": raw["meta"]["illegal_rate"],
        "kept_frac": ds["kept_frac"],
        "demo_success_traj": raw["meta"]["success_traj"],
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s0_pointnav.yaml"))
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "s0_matrix.json"))
    parser.add_argument("--smoke", action="store_true", help="Tiny run for CI/smoke")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        cfg.update(
            {
                "n_demos": 12,
                "n_eval": 8,
                "epochs": 15,
                "seeds": [0],
                "illegal_levels": ["mid"],
                "modes": ["A", "B", "C"],
                "history_options": [False],
            }
        )

    results = []
    for seed in cfg["seeds"]:
        for illegal in cfg["illegal_levels"]:
            for mode in cfg["modes"]:
                for hist in cfg["history_options"]:
                    print(f"=== seed={seed} illegal={illegal} mode={mode} hist={hist} ===")
                    row = run_one(cfg, mode=mode, illegal_level=illegal, use_history=hist, seed=seed)
                    print(json.dumps({k: row[k] for k in row if k != "error"}, indent=2))
                    results.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)
    print(f"Wrote {out}")

    # compact table
    print("\nmode  illegal  hist  success  stuck  reject  illegal_rate  kept")
    for r in results:
        if "error" in r:
            print(f"{r['mode']:4} {r['illegal_level']:6} {str(r['use_history']):5} ERROR {r['error']}")
            continue
        print(
            f"{r['mode']:4} {r['illegal_level']:6} {str(r['use_history']):5} "
            f"{r['success_rate']:.2f}     {r['stuck_rate']:.2f}  {r['mean_reject_rate']:.2f}   "
            f"{r['illegal_rate']:.2f}          {r['kept_frac']:.2f}"
        )


if __name__ == "__main__":
    main()
