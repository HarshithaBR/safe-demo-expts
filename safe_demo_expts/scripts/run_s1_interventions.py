#!/usr/bin/env python3
"""V1–V4 oracle intervention probe under escape shield."""

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
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_interventions.yaml"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="Comma-separated tags, e.g. A_nohist,C_hist (default: config policies)",
    )
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    from safe_demo_expts.scripts._s1_common import parse_policy_tags

    specs = (
        specs_from_config(cfg, tags=parse_policy_tags(args.policies))
        if args.policies
        else specs_from_config(cfg)
    )
    variants = cfg.get("variants") or [{"name": "none"}, {"name": "V3b"}]

    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))
    seed = int(cfg.get("seed", 7))

    results: List[Dict[str, Any]] = []
    for spec in specs:
        tag = spec["tag"]
        print(f"=== interventions policy={tag} ===")
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
        for v in variants:
            if isinstance(v, str):
                name, extra = v, {}
            else:
                name = v.get("name", "none")
                extra = {k: val for k, val in v.items() if k != "name"}
            print(f"  variant={name}")
            m = evaluate_interventions(
                policy,
                variant=name,
                n_episodes=int(cfg["n_eval"]),
                seed=seed,
                deploy_margin=float(cfg["deploy_margin"]),
                oracle_margin=float(cfg["oracle_margin"]),
                stuck_horizon=int(cfg["stuck_horizon"]),
                device=cfg.get("device", "cpu"),
                max_steps=int(cfg["max_steps"]),
                env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
                escape_mode=bool(cfg.get("escape_mode", True)),
                assist_budget=int(cfg.get("assist_budget", 80)),
                max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 3)),
                **extra,
            )
            results.append(
                {
                    "tag": tag,
                    "mode": spec["mode"],
                    "use_history": spec["use_history"],
                    "illegal_level": spec["illegal_level"],
                    "variant": name,
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
                    "budget": m.get("budget", float(cfg.get("assist_budget", 80))),
                }
            )

    payload = {"config": copy.deepcopy(cfg), "results": results, "smoke": bool(args.smoke)}
    out = args.out or str(ROOT / "results" / "s1_interventions_escape.json")
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
