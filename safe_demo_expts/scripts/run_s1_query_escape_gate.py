#!/usr/bin/env python3
"""H3 query-vs-escape gate runner (four modes)."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

from safe_demo_expts.interventions.query_escape_gate import (
    collect_escape_fail_labels,
    save_gate,
    train_gate,
)
from safe_demo_expts.pipelines.evaluate_query_escape import evaluate_query_escape
from safe_demo_expts.scripts._s1_common import (
    ROOT,
    apply_smoke,
    dump_result,
    ensure_policy,
    load_yaml,
    parse_policy_tags,
    resolve_path,
    specs_from_config,
)
from safe_demo_expts.util_ckpt import aggregate_seed_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=str(ROOT / "configs" / "s1_query_escape_gate.yaml")
    )
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    tags = parse_policy_tags(args.policies, default=[p["tag"] for p in cfg.get("policies", [])])
    specs = specs_from_config(cfg, tags)
    modes = list(cfg.get("modes") or [
        "escape_always",
        "oracle_always",
        "try_escape_then_query",
        "learned",
    ])

    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))
    artifact_dir = resolve_path(cfg.get("artifact_dir", "results/query_escape_gate"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    methods: Dict[str, Any] = {}
    for spec in specs:
        tag = spec["tag"]
        print(f"=== query-escape policy={tag} ===")
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
        per_mode: Dict[str, list] = {m: [] for m in modes}
        for seed in cfg["seeds"]:
            print(f"  seed={seed}")
            # Train gate labels once per seed (for learned mode).
            feats, labels, _ = collect_escape_fail_labels(
                policy,
                n_episodes=int(cfg.get("label_episodes", 40)),
                seed=int(cfg.get("label_seed_base", 11)) + int(seed),
                deploy_margin=float(cfg["deploy_margin"]),
                stuck_horizon=int(cfg["stuck_horizon"]),
                escape_try_steps=int(cfg.get("escape_try_steps", 30)),
                max_steps=int(cfg["max_steps"]),
                env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
                device=cfg.get("device", "cpu"),
                escape_shield=cfg.get("escape_shield", "escape"),
                mpc_horizon=int(cfg.get("mpc_horizon", 3)),
                mpc_samples=int(cfg.get("mpc_samples", 20)),
            )
            gate, gate_stats = train_gate(
                feats,
                labels,
                epochs=int(cfg.get("gate_epochs", 50)),
                lr=float(cfg.get("gate_lr", 1e-2)),
                hidden=int(cfg.get("gate_hidden", 32)),
                device=cfg.get("device", "cpu"),
                seed=int(seed),
            )
            gpath = artifact_dir / f"{tag}_seed{seed}_gate.pt"
            save_gate(gate, gpath, meta={"tag": tag, "seed": int(seed), **gate_stats})

            common = dict(
                n_episodes=int(cfg["n_eval"]),
                seed=int(seed),
                deploy_margin=float(cfg["deploy_margin"]),
                oracle_margin=float(cfg["oracle_margin"]),
                stuck_horizon=int(cfg["stuck_horizon"]),
                device=cfg.get("device", "cpu"),
                max_steps=int(cfg["max_steps"]),
                env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
                fail_threshold=float(cfg.get("fail_threshold", 0.55)),
                escape_try_steps=int(cfg.get("escape_try_steps", 30)),
                escape_shield=cfg.get("escape_shield", "escape"),
                mpc_horizon=int(cfg.get("mpc_horizon", 3)),
                mpc_samples=int(cfg.get("mpc_samples", 20)),
                assist_variant=cfg.get("assist_variant", "V3b"),
                assist_budget=int(cfg.get("assist_budget", 40)),
                max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
                v3_disagree_tol=float(cfg.get("v3_disagree_tol", 0.45)),
                v3_max_steps=int(cfg.get("v3_max_steps", 40)),
            )
            for mode in modes:
                m = evaluate_query_escape(
                    policy,
                    mode=mode,
                    gate=gate if mode == "learned" else None,
                    **common,
                )
                row = {
                    "seed": int(seed),
                    "mode_tag": mode,
                    "success_rate": m["success_rate"],
                    "terminal_stuck_rate": m["terminal_stuck_rate"],
                    "mean_oracle_actions": m["mean_oracle_actions"],
                    "mean_gate_oracle_decisions": m["mean_gate_oracle_decisions"],
                    "mean_gate_escape_decisions": m["mean_gate_escape_decisions"],
                    "mean_reject_rate": m["mean_reject_rate"],
                }
                if mode == "learned":
                    row["gate_stats"] = gate_stats
                per_mode[mode].append(row)

        # Aggregate; keep gate_stats out of mean/std keys via aggregate on numeric only.
        methods[tag] = {}
        for mode, rows in per_mode.items():
            agg = aggregate_seed_metrics(rows)
            methods[tag][mode] = agg

    cfg_out = copy.deepcopy(cfg)
    cfg_out["policies"] = specs
    payload = {"methods": methods, "config": cfg_out, "smoke": bool(args.smoke)}
    out = args.out or str(ROOT / "results" / "s1_query_escape_gate.json")
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
