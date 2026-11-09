#!/usr/bin/env python3
"""Oracle budget–success / stuck curves (V2 / V3b / V4)."""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List

from safe_demo_expts.pipelines.evaluate_interventions import evaluate_interventions
from safe_demo_expts.scripts._s1_common import (
    ROOT,
    apply_smoke,
    dump_result,
    ensure_policy,
    load_yaml,
    resolve_path,
    specs_from_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_budget_curves.yaml"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    if args.smoke and "smoke_budgets" in cfg:
        cfg["budgets"] = list(cfg["smoke_budgets"])
    specs = specs_from_config(cfg)
    variants = list(cfg.get("variants") or ["V2", "V3b", "V4"])
    budgets = list(cfg.get("budgets") or [0, 5, 10, 20, 40, 80])
    seed = int(cfg.get("seed", 7))

    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))

    results: List[Dict[str, Any]] = []
    for spec in specs:
        tag = spec["tag"]
        print(f"=== budget curves policy={tag} ===")
        policy, _meta, _ckpt = ensure_policy(
            spec,
            checkpoint_dir=ckpt_dir,
            demo_dir=demo_dir,
            device=cfg.get("device", "cpu"),
            policy_seed=int(cfg.get("policy_seed", 0)),
            n_demos=int(cfg.get("n_demos", 40)),
            epochs=int(cfg.get("train_epochs", 60)),
            deploy_margin=float(cfg["deploy_margin"]),
            max_steps=int(cfg["max_steps"]),
            env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
        )
        for variant in variants:
            for budget in budgets:
                vname = "none" if int(budget) == 0 else variant
                print(f"  variant={vname} budget={budget}")
                m = evaluate_interventions(
                    policy,
                    variant=vname,
                    n_episodes=int(cfg["n_eval"]),
                    seed=seed,
                    deploy_margin=float(cfg["deploy_margin"]),
                    oracle_margin=float(cfg["oracle_margin"]),
                    stuck_horizon=int(cfg["stuck_horizon"]),
                    device=cfg.get("device", "cpu"),
                    max_steps=int(cfg["max_steps"]),
                    env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
                    escape_mode=bool(cfg.get("escape_mode", False)),
                    assist_budget=int(budget) if int(budget) > 0 else 0,
                    max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
                    v2_n=int(cfg.get("v2_n", 10)),
                    v3_disagree_tol=float(cfg.get("v3_disagree_tol", 0.45)),
                    v3_max_steps=int(cfg.get("v3_max_steps", 40)),
                    v4_horizon=int(cfg.get("v4_horizon", 20)),
                )
                results.append(
                    {
                        "tag": tag,
                        "mode": spec["mode"],
                        "use_history": spec["use_history"],
                        "illegal_level": spec["illegal_level"],
                        "success_rate": m["success_rate"],
                        "ever_stuck_rate": m["ever_stuck_rate"],
                        "terminal_stuck_rate": m["terminal_stuck_rate"],
                        "assisted_success_rate": m["assisted_success_rate"],
                        "unassisted_success_rate": m["unassisted_success_rate"],
                        "mean_oracle_actions": m["mean_oracle_actions"],
                        "mean_interventions": m["mean_interventions"],
                        "mean_reject_rate": m["mean_reject_rate"],
                        "mean_episode_cost": m["mean_episode_cost"],
                        "mean_length": m["mean_length"],
                        "n_episodes": m["n_episodes"],
                        "variant": "none" if int(budget) == 0 else variant,
                        "budget": float(budget),
                    }
                )

    payload = {"config": copy.deepcopy(cfg), "results": results, "smoke": bool(args.smoke)}
    out = args.out or str(ROOT / "results" / "s1_budget_curves.json")
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
