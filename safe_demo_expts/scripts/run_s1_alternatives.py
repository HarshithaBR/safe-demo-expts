#!/usr/bin/env python3
"""
Thin alternatives stub: freeze / escape / MPC / V3b assist / gated residual.

Full TD3+BC / shielded PPO are intentionally omitted (heavy); this runner covers
the paths already supported by evaluate_flexible + gated_recovery.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List

import numpy as np

from safe_demo_expts.models.bc import finetune_bc
from safe_demo_expts.models.gated_recovery import act_with_residual, train_residual
from safe_demo_expts.pipelines.evaluate_flexible import evaluate_flexible
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
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "s1_paper_bakeoff.yaml"),
        help="Reuse bake-off defaults (policies / margins / seeds)",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    specs = specs_from_config(cfg)
    # Alternatives probe historically used a single seed.
    seed = int(args.seed)
    n_eval = int(cfg.get("n_eval", 20))

    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))
    device = cfg.get("device", "cpu")

    results: List[Dict[str, Any]] = []
    for spec in specs:
        tag = spec["tag"]
        use_history = bool(spec["use_history"])
        print(f"=== alternatives policy={tag} ===")
        policy, _meta, _ckpt = ensure_policy(
            spec,
            checkpoint_dir=ckpt_dir,
            demo_dir=demo_dir,
            device=device,
            policy_seed=int(cfg.get("policy_seed", 0)),
            n_demos=int(cfg.get("n_demos", 40)),
            epochs=int(cfg.get("train_epochs", 60)),
            deploy_margin=float(cfg["deploy_margin"]),
            max_steps=int(cfg["max_steps"]),
            env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
        )

        common = dict(
            n_episodes=n_eval,
            seed=seed,
            deploy_margin=float(cfg["deploy_margin"]),
            oracle_margin=float(cfg.get("oracle_margin", 0.04)),
            stuck_horizon=int(cfg.get("stuck_horizon", 40)),
            device=device,
            max_steps=int(cfg["max_steps"]),
            env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
            mpc_horizon=int(cfg.get("mpc_horizon", 3)),
            mpc_samples=int(cfg.get("mpc_samples", 20)),
            assist_budget=int(cfg.get("assist_budget", 40)),
            max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
            v3_disagree_tol=float(cfg.get("v3_disagree_tol", 0.45)),
            v3_max_steps=int(cfg.get("v3_max_steps", 40)),
        )

        def row(alt: str, m: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
            return {
                "tag": tag,
                "alt": alt,
                "success_rate": m["success_rate"],
                "ever_stuck_rate": m.get("ever_stuck_rate", m.get("terminal_stuck_rate")),
                "terminal_stuck_rate": m["terminal_stuck_rate"],
                "mean_reject_rate": m["mean_reject_rate"],
                "mean_episode_cost": m.get("mean_episode_cost", 0.0),
                "mean_length": m.get("mean_length", 0.0),
                "n_episodes": m.get("n_episodes", float(n_eval)),
                **extra,
            }

        # 0 BC auto (escape)
        m0 = evaluate_flexible(
            policy, shield="escape", escape_mode=True, intervention_variant=None, **common
        )
        results.append(row("0_bc_auto", m0))

        # 3 eval assist V3b
        m3 = evaluate_flexible(
            policy,
            shield="escape",
            escape_mode=True,
            intervention_variant=cfg.get("assist_variant", "V3b"),
            **common,
        )
        results.append(
            row(
                "3_eval_assist_V3b",
                m3,
                assisted_success_rate=m3.get("assisted_success_rate"),
                unassisted_success_rate=m3.get("unassisted_success_rate"),
                mean_oracle_actions=m3.get("mean_oracle_actions"),
                mean_interventions=m3.get("mean_interventions"),
                variant=m3.get("variant"),
                budget=m3.get("budget"),
            )
        )

        # Collect oracle for recovery / residual
        collect = evaluate_flexible(
            policy,
            n_episodes=int(cfg.get("collect_episodes", 16 if args.smoke else 40)),
            seed=int(cfg.get("collect_seed_base", 21)),
            deploy_margin=float(cfg["deploy_margin"]),
            oracle_margin=float(cfg.get("oracle_margin", 0.04)),
            stuck_horizon=int(cfg.get("stuck_horizon", 40)),
            device=device,
            max_steps=int(cfg["max_steps"]),
            env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
            shield="escape",
            escape_mode=True,
            intervention_variant=cfg.get("assist_variant", "V3b"),
            assist_budget=int(cfg.get("assist_budget", 40)),
            max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
            collect_oracle_buffer=True,
        )
        buf = collect.get("oracle_transitions") or {}

        # 1 recovery policy — BC on oracle labels, used via use_recovery_fn
        rec_policy = copy.deepcopy(policy)
        if len(buf.get("obs", [])):
            rec_policy, _ = finetune_bc(
                rec_policy,
                obs=buf["obs"],
                actions=buf["actions"],
                reject=buf.get("reject_deploy"),
                traj_id=buf.get("traj_id"),
                use_history=use_history,
                epochs=int(cfg.get("ft_epochs", 8 if args.smoke else 25)),
                lr=float(cfg.get("ft_lr", 5e-4)),
                device=device,
                seed=seed,
            )

        def recovery_switch(obs, a_base, env, stuck_info):
            del a_base, env
            from safe_demo_expts.models.bc import act_policy

            return act_policy(
                rec_policy,
                obs,
                prev_action=stuck_info.get("prev_action"),
                prev_reject=float(stuck_info.get("prev_reject", 0.0)),
                device=device,
            )

        m1 = evaluate_flexible(
            policy,
            shield="escape",
            escape_mode=True,
            intervention_variant=None,
            use_recovery_fn=recovery_switch,
            **common,
        )
        results.append(row("1_recovery_policy", m1))

        # 2 residual (always-on via recovery gate — approximates always-on when stuck)
        residual, _ = train_residual(
            policy,
            obs=buf.get("obs", np.zeros((0, 60))),
            oracle_actions=buf.get("actions", np.zeros((0, 2))),
            reject=buf.get("reject_deploy"),
            use_history=use_history,
            traj_id=buf.get("traj_id"),
            a_base=buf.get("a_base"),
            epochs=int(cfg.get("residual_epochs", 25 if args.smoke else 40)),
            device=device,
            seed=seed,
        )

        def residual_fn(obs, a_base, env, stuck_info):
            del a_base, env
            return act_with_residual(
                policy,
                residual,
                obs,
                prev_a=stuck_info.get("prev_action"),
                prev_rej=float(stuck_info.get("prev_reject", 0.0)),
                device=device,
            )

        m2 = evaluate_flexible(
            policy,
            shield="escape",
            escape_mode=True,
            intervention_variant=None,
            use_recovery_fn=residual_fn,
            **common,
        )
        results.append(row("2_residual", m2))

        # 4 escape-projected C is just the C-mode BC under escape — alias of 0 for C tags;
        # for A tags we still report escape deploy of the loaded policy.
        results.append(row("4_escape_projected_C", m0))

        # 7 MPC
        m7 = evaluate_flexible(policy, shield="mpc", intervention_variant=None, **common)
        results.append(row("7_mpc_escape", m7))

        # Stubs for heavy alts so schema stays recognizable
        results.append(
            {
                "tag": tag,
                "alt": "5_td3bc",
                "skipped": True,
                "reason": "TD3+BC omitted in thin stub; use dedicated offline RL runner",
            }
        )
        results.append(
            {
                "tag": tag,
                "alt": "6_ppo",
                "skipped": True,
                "reason": "shielded PPO omitted in thin stub",
            }
        )

    payload = {
        "results": results,
        "config": {
            "note": "thin stub: freeze/escape/MPC/V3b/recovery/residual; TD3/PPO skipped",
            "seed": seed,
            "n_eval": n_eval,
            "smoke": bool(args.smoke),
        },
        "smoke": bool(args.smoke),
    }
    out = args.out or str(ROOT / "results" / "s1_alternatives.json")
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
