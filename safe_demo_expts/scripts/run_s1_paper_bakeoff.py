#!/usr/bin/env python3
"""Matched nav bake-off: freeze / A1a / A1b_postft / A2_escape / A3_mpc."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

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
    parse_policy_tags,
    resolve_path,
    specs_from_config,
)
from safe_demo_expts.util_ckpt import aggregate_seed_metrics, save_policy


def _ek(cfg: dict) -> dict:
    return dict(
        n_episodes=int(cfg["n_eval"]),
        deploy_margin=float(cfg["deploy_margin"]),
        oracle_margin=float(cfg["oracle_margin"]),
        stuck_horizon=int(cfg["stuck_horizon"]),
        device=cfg.get("device", "cpu"),
        max_steps=int(cfg["max_steps"]),
        env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
        assist_budget=int(cfg.get("assist_budget", 40)),
        max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
        v3_disagree_tol=float(cfg.get("v3_disagree_tol", 0.45)),
        v3_max_steps=int(cfg.get("v3_max_steps", 40)),
        mpc_horizon=int(cfg.get("mpc_horizon", 3)),
        mpc_samples=int(cfg.get("mpc_samples", 20)),
    )


def run_seed(policy, spec: dict, cfg: dict, seed: int, ft_dir: Path, demo_dir: Path) -> Dict[str, Any]:
    device = cfg.get("device", "cpu")
    use_history = bool(spec["use_history"])
    ek = _ek(cfg)
    out: Dict[str, Any] = {}

    m = evaluate_flexible(
        policy, seed=seed, shield="freeze", escape_mode=False, intervention_variant=None, **ek
    )
    out["freeze"] = {
        "seed": seed,
        "success_rate": m["success_rate"],
        "terminal_stuck_rate": m["terminal_stuck_rate"],
        "mean_reject_rate": m["mean_reject_rate"],
        "mean_episode_cost": m["mean_episode_cost"],
        "mean_length": m["mean_length"],
    }

    # A1a — interactive assist, no FT
    m = evaluate_flexible(
        policy,
        seed=seed,
        shield="escape",
        escape_mode=True,
        intervention_variant=cfg.get("assist_variant", "V3b"),
        **ek,
    )
    out["A1a"] = {
        "seed": seed,
        "success_rate": m["success_rate"],
        "terminal_stuck_rate": m["terminal_stuck_rate"],
        "mean_reject_rate": m["mean_reject_rate"],
        "mean_oracle_actions": m["mean_oracle_actions"],
        "mean_episode_cost": m["mean_episode_cost"],
        "mean_length": m["mean_length"],
    }

    # Collect oracle buffer for A1b
    collect = evaluate_flexible(
        policy,
        seed=int(cfg.get("collect_seed_base", 21)) + seed,
        n_episodes=int(cfg["collect_episodes"]),
        deploy_margin=float(cfg["deploy_margin"]),
        oracle_margin=float(cfg["oracle_margin"]),
        stuck_horizon=int(cfg["stuck_horizon"]),
        device=device,
        max_steps=int(cfg["max_steps"]),
        env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
        shield="escape",
        escape_mode=True,
        intervention_variant=cfg.get("assist_variant", "V3b"),
        assist_budget=int(cfg.get("assist_budget", 40)),
        max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
        v3_disagree_tol=float(cfg.get("v3_disagree_tol", 0.45)),
        v3_max_steps=int(cfg.get("v3_max_steps", 40)),
        collect_oracle_buffer=True,
    )
    buf = collect.get("oracle_transitions") or {}
    n_oracle = int(collect.get("n_oracle_transitions", 0))

    ft_policy = copy.deepcopy(policy)
    ft_stats: Dict[str, float] = {"ft_train_mse": float("nan"), "ft_n": 0.0}
    obs_ft = buf.get("obs", np.zeros((0, 60)))
    act_ft = buf.get("actions", np.zeros((0, 2)))
    rej_ft = buf.get("reject_deploy", np.zeros((0,)))
    traj_ft = buf.get("traj_id", np.zeros((0,), dtype=np.int64))

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
        if n_oracle > 0 and len(ds["obs"]):
            obs_ft = np.concatenate([ds["obs"], obs_ft], axis=0)
            act_ft = np.concatenate([ds["actions"], act_ft], axis=0)
            rej_ft = np.concatenate([ds["reject_deploy"], rej_ft], axis=0)
            # offset oracle traj ids
            offset = int(ds["traj_id"].max()) + 1 if len(ds["traj_id"]) else 0
            traj_ft = np.concatenate([ds["traj_id"], traj_ft + offset], axis=0)
        elif len(ds["obs"]):
            obs_ft, act_ft, rej_ft, traj_ft = (
                ds["obs"],
                ds["actions"],
                ds["reject_deploy"],
                ds["traj_id"],
            )

    if len(obs_ft):
        ft_policy, ft_stats = finetune_bc(
            ft_policy,
            obs=obs_ft,
            actions=act_ft,
            reject=rej_ft,
            traj_id=traj_ft,
            use_history=use_history,
            epochs=int(cfg.get("ft_epochs", 25)),
            lr=float(cfg.get("ft_lr", 5e-4)),
            device=device,
            seed=seed,
        )

    hist = 1 if use_history else 0
    ft_name = f"s1_{spec['illegal_level']}_mode{spec['mode']}_hist{hist}_ft_bakeoff.pt"
    ft_path = ft_dir / ft_name
    save_policy(
        ft_policy,
        ft_path,
        meta={"tag": spec["tag"], "seed": seed, "method": "A1b_postft", **ft_stats},
    )

    m = evaluate_flexible(
        ft_policy, seed=seed, shield="freeze", escape_mode=False, intervention_variant=None, **ek
    )
    out["A1b_postft"] = {
        "seed": seed,
        "success_rate": m["success_rate"],
        "terminal_stuck_rate": m["terminal_stuck_rate"],
        "mean_reject_rate": m["mean_reject_rate"],
        "mean_oracle_actions": 0.0,
        "mean_episode_cost": m["mean_episode_cost"],
        "mean_length": m["mean_length"],
        "n_oracle_transitions": n_oracle,
        "ft_ckpt": str(ft_path),
        "ft_train_mse": float(ft_stats.get("ft_train_mse", float("nan"))),
        "n_ft_transitions": int(len(obs_ft)),
    }

    m = evaluate_flexible(
        policy, seed=seed, shield="escape", escape_mode=True, intervention_variant=None, **ek
    )
    out["A2_escape"] = {
        "seed": seed,
        "success_rate": m["success_rate"],
        "terminal_stuck_rate": m["terminal_stuck_rate"],
        "mean_reject_rate": m["mean_reject_rate"],
        "mean_episode_cost": m["mean_episode_cost"],
        "mean_length": m["mean_length"],
    }

    m = evaluate_flexible(
        policy, seed=seed, shield="mpc", intervention_variant=None, **ek
    )
    out["A3_mpc"] = {
        "seed": seed,
        "success_rate": m["success_rate"],
        "terminal_stuck_rate": m["terminal_stuck_rate"],
        "mean_reject_rate": m["mean_reject_rate"],
        "mean_episode_cost": m["mean_episode_cost"],
        "mean_length": m["mean_length"],
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_paper_bakeoff.yaml"))
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    tags = parse_policy_tags(args.policies, default=[p["tag"] for p in cfg.get("policies", [])])
    specs = specs_from_config(cfg, tags)

    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))
    ft_dir = resolve_path(cfg.get("ft_checkpoint_dir", "results/checkpoints_ft"))
    ft_dir.mkdir(parents=True, exist_ok=True)

    methods: Dict[str, Any] = {}
    for spec in specs:
        tag = spec["tag"]
        print(f"=== paper bakeoff policy={tag} ===")
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
        per: Dict[str, list] = {
            "freeze": [],
            "A1a": [],
            "A1b_postft": [],
            "A2_escape": [],
            "A3_mpc": [],
        }
        for seed in cfg["seeds"]:
            print(f"  seed={seed}")
            rows = run_seed(policy, spec, cfg, int(seed), ft_dir, demo_dir)
            for k, v in rows.items():
                per[k].append(v)
        methods[tag] = {k: aggregate_seed_metrics(v) for k, v in per.items()}

    cfg_out = copy.deepcopy(cfg)
    cfg_out["policies"] = specs
    payload = {"methods": methods, "config": cfg_out, "smoke": bool(args.smoke)}
    out = args.out or str(ROOT / "results" / "s1_paper_bakeoff.json")
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
