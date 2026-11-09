#!/usr/bin/env python3
"""A1c gated residual bake-off: freeze / A1b_naive_ft / A1c_gated_residual / A2_escape."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np

from safe_demo_expts.models.bc import finetune_bc
from safe_demo_expts.models.gated_recovery import (
    act_with_residual,
    save_residual,
    train_residual,
)
from safe_demo_expts.pipelines.evaluate_flexible import evaluate_flexible
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


def _eval_kwargs(cfg: dict) -> dict:
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
    )


def _collect_oracle(policy, cfg: dict, seed: int) -> Dict[str, Any]:
    ek = _eval_kwargs(cfg)
    ek["n_episodes"] = int(cfg["collect_episodes"])
    return evaluate_flexible(
        policy,
        seed=int(cfg.get("collect_seed_base", 21)) + seed,
        shield="freeze",
        escape_mode=False,
        intervention_variant=cfg.get("assist_variant", "V3b"),
        collect_oracle_buffer=True,
        **ek,
    )


def run_seed(policy, spec: dict, cfg: dict, seed: int, artifact_dir: Path) -> Dict[str, Dict[str, Any]]:
    device = cfg.get("device", "cpu")
    tag = spec["tag"]
    use_history = bool(spec["use_history"])
    ek = _eval_kwargs(cfg)
    out: Dict[str, Dict[str, Any]] = {}

    # --- freeze ---
    m_freeze = evaluate_flexible(
        policy, seed=seed, shield="freeze", escape_mode=False, intervention_variant=None, **ek
    )
    out["freeze"] = {
        "seed": seed,
        "success_rate": m_freeze["success_rate"],
        "terminal_stuck_rate": m_freeze["terminal_stuck_rate"],
        "mean_reject_rate": m_freeze["mean_reject_rate"],
        "mean_episode_cost": m_freeze["mean_episode_cost"],
        "mean_length": m_freeze["mean_length"],
    }

    # --- collect oracle labels under freeze + V3b ---
    collect = _collect_oracle(policy, cfg, seed)
    buf = collect.get("oracle_transitions") or {}
    n_oracle = int(collect.get("n_oracle_transitions", 0))

    # --- A1b naïve FT ---
    ft_policy = copy.deepcopy(policy)
    ft_stats: Dict[str, float] = {"ft_train_mse": float("nan")}
    if n_oracle > 0:
        ft_policy, ft_stats = finetune_bc(
            ft_policy,
            obs=buf["obs"],
            actions=buf["actions"],
            reject=buf.get("reject_deploy"),
            traj_id=buf.get("traj_id"),
            use_history=use_history,
            epochs=int(cfg.get("ft_epochs", 25)),
            lr=float(cfg.get("ft_lr", 5e-4)),
            device=device,
            seed=seed,
        )
    m_ft = evaluate_flexible(
        ft_policy, seed=seed, shield="freeze", escape_mode=False, intervention_variant=None, **ek
    )
    out["A1b_naive_ft"] = {
        "seed": seed,
        "success_rate": m_ft["success_rate"],
        "terminal_stuck_rate": m_ft["terminal_stuck_rate"],
        "mean_reject_rate": m_ft["mean_reject_rate"],
        "mean_episode_cost": m_ft["mean_episode_cost"],
        "mean_length": m_ft["mean_length"],
        "n_oracle": n_oracle,
        "ft_train_mse": float(ft_stats.get("ft_train_mse", float("nan"))),
    }

    # --- A1c gated residual ---
    residual, rstats = train_residual(
        policy,
        obs=buf.get("obs", np.zeros((0, 60))),
        oracle_actions=buf.get("actions", np.zeros((0, 2))),
        reject=buf.get("reject_deploy"),
        use_history=use_history,
        traj_id=buf.get("traj_id"),
        a_base=buf.get("a_base"),
        epochs=int(cfg.get("residual_epochs", 40)),
        lr=float(cfg.get("residual_lr", 1e-3)),
        hidden=int(cfg.get("residual_hidden", 64)),
        device=device,
        seed=seed,
    )
    ckpt_rel = f"{tag}_seed{seed}_gated.pt"
    ckpt_path = artifact_dir / ckpt_rel
    save_residual(
        residual,
        ckpt_path,
        meta={"tag": tag, "seed": seed, "use_history": use_history, **rstats},
    )

    def recovery_fn(obs, a_base, env, stuck_info):
        del a_base, env
        return act_with_residual(
            policy,
            residual,
            obs,
            prev_a=stuck_info.get("prev_action"),
            prev_rej=float(stuck_info.get("prev_reject", 0.0)),
            device=device,
        )

    m_a1c = evaluate_flexible(
        policy,
        seed=seed,
        shield="freeze",
        escape_mode=False,
        intervention_variant=None,
        use_recovery_fn=recovery_fn,
        **ek,
    )
    out["A1c_gated_residual"] = {
        "seed": seed,
        "success_rate": m_a1c["success_rate"],
        "terminal_stuck_rate": m_a1c["terminal_stuck_rate"],
        "mean_reject_rate": m_a1c["mean_reject_rate"],
        "mean_episode_cost": m_a1c["mean_episode_cost"],
        "mean_length": m_a1c["mean_length"],
        "n_oracle": n_oracle,
        "residual_train_mse": float(rstats.get("residual_train_mse", float("nan"))),
        "recon_oracle_mse": float(rstats.get("recon_oracle_mse", float("nan"))),
        "ckpt": str(Path(cfg.get("artifact_dir", "results/gated_recovery")) / ckpt_rel),
    }

    # --- A2 escape ---
    m_esc = evaluate_flexible(
        policy, seed=seed, shield="escape", escape_mode=True, intervention_variant=None, **ek
    )
    out["A2_escape"] = {
        "seed": seed,
        "success_rate": m_esc["success_rate"],
        "terminal_stuck_rate": m_esc["terminal_stuck_rate"],
        "mean_reject_rate": m_esc["mean_reject_rate"],
        "mean_episode_cost": m_esc["mean_episode_cost"],
        "mean_length": m_esc["mean_length"],
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_gated_recovery.yaml"))
    parser.add_argument("--policies", type=str, default=None, help="Comma tags e.g. A_nohist,C_hist")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    tags = parse_policy_tags(args.policies, default=[p["tag"] for p in cfg.get("policies", [])])
    specs = specs_from_config(cfg, tags)

    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))
    artifact_dir = resolve_path(cfg.get("artifact_dir", "results/gated_recovery"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    methods: Dict[str, Any] = {}
    for spec in specs:
        tag = spec["tag"]
        print(f"=== gated recovery policy={tag} ===")
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
        per_method: Dict[str, list] = {
            "freeze": [],
            "A1b_naive_ft": [],
            "A1c_gated_residual": [],
            "A2_escape": [],
        }
        for seed in cfg["seeds"]:
            print(f"  seed={seed}")
            rows = run_seed(policy, spec, cfg, int(seed), artifact_dir)
            for k, v in rows.items():
                per_method[k].append(v)
        methods[tag] = {k: aggregate_seed_metrics(v) for k, v in per_method.items()}

    # Persist only the policies that were run in the config snapshot.
    cfg_out = copy.deepcopy(cfg)
    cfg_out["policies"] = specs
    payload = {"methods": methods, "config": cfg_out, "smoke": bool(args.smoke)}
    out = args.out or str(ROOT / "results" / "s1_gated_recovery.json")
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
