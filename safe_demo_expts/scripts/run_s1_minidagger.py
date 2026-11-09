#!/usr/bin/env python3
"""Mini-DAgger: V3b collect → fine-tune → autonomous re-eval."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from safe_demo_expts.data.generate_safety_point_demos import filter_dataset
from safe_demo_expts.models.bc import finetune_bc
from safe_demo_expts.pipelines.evaluate_flexible import evaluate_flexible
from safe_demo_expts.scripts._s1_common import (
    ROOT,
    apply_smoke,
    dump_result,
    ensure_policy,
    load_or_generate_demos,
    load_yaml,
    resolve_path,
    specs_from_config,
)
from safe_demo_expts.util_ckpt import save_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_minidagger.yaml"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Oracle-only labels, lower lr / fewer epochs (minidagger_safe)",
    )
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    if args.safe:
        cfg["mix_original_demos"] = False
        cfg["ft_lr"] = float(cfg.get("safe_ft_lr", 1e-4))
        cfg["ft_epochs"] = int(cfg.get("safe_ft_epochs", 10))

    specs = specs_from_config(cfg)
    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))
    ft_dir = resolve_path(cfg.get("ft_checkpoint_dir", "results/checkpoints_ft"))
    ft_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for spec in specs:
        tag = spec["tag"]
        use_history = bool(spec["use_history"])
        print(f"=== minidagger policy={tag} ===")
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

        common = dict(
            n_episodes=int(cfg["n_eval"]),
            seed=int(cfg.get("eval_seed", 77)),
            deploy_margin=float(cfg["deploy_margin"]),
            oracle_margin=float(cfg["oracle_margin"]),
            stuck_horizon=int(cfg["stuck_horizon"]),
            device=cfg.get("device", "cpu"),
            max_steps=int(cfg["max_steps"]),
            env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
            max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
            v3_disagree_tol=float(cfg.get("v3_disagree_tol", 0.45)),
            v3_max_steps=int(cfg.get("v3_max_steps", 40)),
        )

        # Freeze deploy (oracle steps still escape-exec inside evaluate_flexible).
        # Escape-alone already solves most mid layouts, so V3b never fires and
        # mini-DAgger collects 0 labels — freeze is the regime where assist/FT matter.
        deploy_escape = bool(cfg.get("escape_mode", False))
        shield_name = "escape" if deploy_escape else "freeze"

        base_auto = evaluate_flexible(
            policy,
            shield=shield_name,
            escape_mode=deploy_escape,
            intervention_variant=None,
            **common,
        )
        base_assist = evaluate_flexible(
            policy,
            shield=shield_name,
            escape_mode=deploy_escape,
            intervention_variant=cfg.get("collect_variant", "V3b"),
            assist_budget=int(cfg.get("collect_budget", 40)),
            **common,
        )

        collect = evaluate_flexible(
            policy,
            n_episodes=int(cfg["collect_episodes"]),
            seed=int(cfg.get("collect_seed", 21)),
            deploy_margin=float(cfg["deploy_margin"]),
            oracle_margin=float(cfg["oracle_margin"]),
            stuck_horizon=int(cfg["stuck_horizon"]),
            device=cfg.get("device", "cpu"),
            max_steps=int(cfg["max_steps"]),
            env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
            shield=shield_name,
            escape_mode=deploy_escape,
            intervention_variant=cfg.get("collect_variant", "V3b"),
            assist_budget=int(cfg.get("collect_budget", 40)),
            max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
            v3_disagree_tol=float(cfg.get("v3_disagree_tol", 0.45)),
            v3_max_steps=int(cfg.get("v3_max_steps", 40)),
            collect_oracle_buffer=True,
        )
        buf = collect.get("oracle_transitions") or {}
        n_oracle = int(collect.get("n_oracle_transitions", 0))

        obs = buf.get("obs", np.zeros((0, 60)))
        acts = buf.get("actions", np.zeros((0, 2)))
        rej = buf.get("reject_deploy", np.zeros((0,)))
        traj = buf.get("traj_id", np.zeros((0,), dtype=np.int64))

        if cfg.get("mix_original_demos", True):
            raw = load_or_generate_demos(
                demo_dir,
                illegal=spec["illegal_level"],
                n_demos=int(cfg.get("n_demos", 40)),
                seed=int(cfg.get("policy_seed", 0)),
                deploy_margin=float(cfg["deploy_margin"]),
                max_steps=int(cfg["max_steps"]),
                env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
            )
            ds = filter_dataset(raw, mode=spec["mode"])
            if len(ds["obs"]):
                offset = int(ds["traj_id"].max()) + 1 if len(ds["traj_id"]) else 0
                if n_oracle > 0:
                    obs = np.concatenate([ds["obs"], obs], axis=0)
                    acts = np.concatenate([ds["actions"], acts], axis=0)
                    rej = np.concatenate([ds["reject_deploy"], rej], axis=0)
                    traj = np.concatenate([ds["traj_id"], traj + offset], axis=0)
                else:
                    obs, acts, rej, traj = ds["obs"], ds["actions"], ds["reject_deploy"], ds["traj_id"]

        ft_policy = copy.deepcopy(policy)
        ft_stats: Dict[str, float] = {"ft_train_mse": float("nan"), "ft_epochs": 0.0, "ft_n": 0.0}
        if len(obs):
            ft_policy, ft_stats = finetune_bc(
                ft_policy,
                obs=obs,
                actions=acts,
                reject=rej,
                traj_id=traj,
                use_history=use_history,
                epochs=int(cfg.get("ft_epochs", 25)),
                lr=float(cfg.get("ft_lr", 5e-4)),
                device=cfg.get("device", "cpu"),
                seed=int(cfg.get("eval_seed", 77)),
            )

        hist = 1 if use_history else 0
        suffix = "safe" if args.safe else "minidagger"
        ft_path = ft_dir / f"s1_{spec['illegal_level']}_mode{spec['mode']}_hist{hist}_ft_{suffix}.pt"
        save_policy(ft_policy, ft_path, meta={"tag": tag, **ft_stats})

        ft_auto = evaluate_flexible(
            ft_policy,
            shield=shield_name,
            escape_mode=deploy_escape,
            intervention_variant=None,
            **common,
        )

        def _slim(m: Dict[str, Any]) -> Dict[str, Any]:
            keys = [
                "success_rate",
                "ever_stuck_rate",
                "terminal_stuck_rate",
                "assisted_success_rate",
                "unassisted_success_rate",
                "mean_oracle_actions",
                "mean_interventions",
                "mean_reject_rate",
                "mean_episode_cost",
                "mean_length",
                "n_episodes",
                "variant",
                "budget",
            ]
            return {k: m[k] for k in keys if k in m}

        results.append(
            {
                "tag": tag,
                "mode": spec["mode"],
                "use_history": use_history,
                "n_oracle_transitions": n_oracle,
                "ft_stats": ft_stats,
                "ft_ckpt": str(ft_path),
                "base_auto": _slim(base_auto),
                "base_assist": _slim(base_assist),
                "ft_auto": _slim(ft_auto),
            }
        )

    payload = {"config": copy.deepcopy(cfg), "results": results, "smoke": bool(args.smoke)}
    default = "s1_minidagger_safe.json" if args.safe else "s1_minidagger.json"
    out = args.out or str(ROOT / "results" / default)
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
