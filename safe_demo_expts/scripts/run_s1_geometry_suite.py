#!/usr/bin/env python3
"""E3 geometry suite: simple / U-trap / narrow / infeasible stratified eval."""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List

import numpy as np

from safe_demo_expts.envs.safety_point_goal import SafetyPointConfig, SafetyPointGoalEnv
from safe_demo_expts.interventions.oracle import RecoveryOracle
from safe_demo_expts.interventions.variants import build_controller
from safe_demo_expts.models.bc import act_policy
from safe_demo_expts.pipelines.evaluate_flexible import make_shield
from safe_demo_expts.pipelines.evaluate_safety_point import goal_distance, is_stuck
from safe_demo_expts.scenarios import LAYOUT_KINDS, reset_with_layout
from safe_demo_expts.scripts._s1_common import (
    ROOT,
    apply_smoke,
    dump_result,
    ensure_policy,
    load_yaml,
    parse_policy_tags,
    policy_spec,
    resolve_path,
    specs_from_config,
)
from safe_demo_expts.util_ckpt import aggregate_seed_metrics


def evaluate_on_layout(
    policy,
    layout_kind: str,
    n_episodes: int,
    seed: int,
    method: str = "freeze",
    deploy_margin: float = 0.14,
    oracle_margin: float = 0.04,
    stuck_horizon: int = 40,
    stuck_eps: float = 1e-3,
    device: str = "cpu",
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
    prefer_override: bool = True,
    layout_scale: float = 1.0,
    assist_variant: str = "V3b",
    assist_budget: int = 40,
    max_interventions_per_episode: int = 5,
    v3_disagree_tol: float = 0.45,
    v3_max_steps: int = 40,
    mpc_horizon: int = 3,
    mpc_samples: int = 20,
    deep_penetration: float = -0.05,
) -> Dict[str, Any]:
    method = method.lower()
    if method in ("freeze",):
        shield = make_shield("freeze", deploy_margin=deploy_margin)
        variant = None
    elif method in ("a2_escape", "escape", "a2"):
        shield = make_shield("escape", deploy_margin=deploy_margin)
        variant = None
    elif method in ("a3_mpc", "mpc", "a3"):
        shield = make_shield(
            "mpc", deploy_margin=deploy_margin, mpc_horizon=mpc_horizon, mpc_samples=mpc_samples
        )
        variant = None
    elif method in ("a1a", "assist", "v3b"):
        shield = make_shield("escape", deploy_margin=deploy_margin)
        variant = assist_variant
    else:
        raise ValueError(f"unknown geometry method={method}")

    oracle = RecoveryOracle(oracle_margin=oracle_margin)
    controller = build_controller(
        variant=variant or "none",
        oracle_margin=oracle_margin,
        v3_disagree_tol=v3_disagree_tol,
        v3_max_steps=v3_max_steps,
        clearance_fn=lambda e: shield.clearance(e),
    )
    controller.oracle_fn = oracle

    cfg = SafetyPointConfig(env_id=env_id, max_steps=max_steps, seed=seed, stop_on_first_goal=True)
    env = SafetyPointGoalEnv(cfg)

    successes = 0
    terminal_stuck = 0
    realized_kinds: List[str] = []
    reject_rates: List[float] = []
    lengths: List[float] = []
    oracle_actions_eps: List[float] = []

    for ep in range(n_episodes):
        o, info0 = reset_with_layout(
            env,
            kind=layout_kind,
            seed=seed + 10_000 + ep,
            prefer_override=prefer_override,
            scale=layout_scale,
            deploy_margin=deploy_margin,
        )
        realized_kinds.append(str(info0.get("layout_kind", layout_kind)))
        prev_a = np.zeros(2, dtype=np.float64)
        prev_rej = 0.0
        dist_hist: List[float] = []
        ep_rejects = 0
        ep_oracle = 0
        ep_interventions = 0
        stuck_terminal = False
        info: Dict = {}
        t = 0
        controller.reset_episode()

        for t in range(max_steps):
            a_base = act_policy(policy, o, prev_action=prev_a, prev_reject=prev_rej, device=device)
            clr = float(shield.clearance(env))
            dist_hist.append(goal_distance(env))
            stuck_now = is_stuck(
                dist_hist,
                stuck_horizon=stuck_horizon,
                stuck_eps=stuck_eps,
                clearance=clr,
                deep_penetration=deep_penetration,
            )
            gate = stuck_now or (clr < deep_penetration and len(dist_hist) >= stuck_horizon)
            used_oracle = False
            a_prop = a_base

            if controller.active:
                if ep_oracle >= assist_budget:
                    controller.state.active = False
                    if gate:
                        stuck_terminal = True
                        break
                else:
                    a_prop, used_oracle = controller.step(env, a_base)
            elif gate:
                if variant is not None and ep_interventions < max_interventions_per_episode and ep_oracle < assist_budget:
                    if controller.maybe_start(env, a_base, stuck_info={"clearance": clr}):
                        ep_interventions += 1
                        a_prop, used_oracle = controller.step(env, a_base)
                    else:
                        stuck_terminal = True
                        break
                else:
                    stuck_terminal = True
                    break

            sr = shield(env, a_prop)
            if used_oracle:
                ep_oracle += 1
            o, _r, terminated, truncated, info = env.step(sr.a_exec)
            prev_a = sr.a_exec.copy()
            prev_rej = 1.0 if sr.reject else 0.0
            if sr.reject:
                ep_rejects += 1
            if terminated or truncated:
                break

        success = bool(info.get("success"))
        if stuck_terminal and not success:
            terminal_stuck += 1
        elif (not success) and is_stuck(
            dist_hist,
            stuck_horizon=stuck_horizon,
            stuck_eps=stuck_eps,
            clearance=float(shield.clearance(env)),
            deep_penetration=deep_penetration,
        ):
            terminal_stuck += 1
        if success:
            successes += 1
        lengths.append(float(t + 1))
        reject_rates.append(ep_rejects / max(1, t + 1))
        oracle_actions_eps.append(float(ep_oracle))

    env.close()
    n = max(1, n_episodes)
    return {
        "layout_requested": layout_kind,
        "layout_realized_mode": max(set(realized_kinds), key=realized_kinds.count) if realized_kinds else layout_kind,
        "success_rate": successes / n,
        "terminal_stuck_rate": terminal_stuck / n,
        "mean_reject_rate": float(np.mean(reject_rates)) if reject_rates else 0.0,
        "mean_oracle_actions": float(np.mean(oracle_actions_eps)) if oracle_actions_eps else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": float(n_episodes),
        "method": method,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_geometry_suite.yaml"))
    parser.add_argument("--policy-tag", type=str, default=None)
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    tag = args.policy_tag or (parse_policy_tags(args.policies, default=[cfg.get("policy_tag", "C_hist")])[0])
    spec = policy_spec(tag)
    layouts = list(cfg.get("layouts") or LAYOUT_KINDS)
    methods = list(cfg.get("methods") or ["freeze", "A2_escape", "A3_mpc", "A1a"])

    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))
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

    results: Dict[str, Any] = {}
    for layout in layouts:
        results[layout] = {}
        for method in methods:
            per_seed = []
            for seed in cfg["seeds"]:
                print(f"=== geometry layout={layout} method={method} seed={seed} ===")
                m = evaluate_on_layout(
                    policy,
                    layout_kind=layout,
                    n_episodes=int(cfg["n_eval"]),
                    seed=int(seed),
                    method=method,
                    deploy_margin=float(cfg["deploy_margin"]),
                    oracle_margin=float(cfg.get("oracle_margin", 0.04)),
                    stuck_horizon=int(cfg["stuck_horizon"]),
                    device=cfg.get("device", "cpu"),
                    max_steps=int(cfg["max_steps"]),
                    env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
                    prefer_override=bool(cfg.get("prefer_override", True)),
                    layout_scale=float(cfg.get("layout_scale", 1.0)),
                    assist_variant=cfg.get("assist_variant", "V3b"),
                    assist_budget=int(cfg.get("assist_budget", 40)),
                    max_interventions_per_episode=int(cfg.get("max_interventions_per_episode", 5)),
                    v3_disagree_tol=float(cfg.get("v3_disagree_tol", 0.45)),
                    v3_max_steps=int(cfg.get("v3_max_steps", 40)),
                    mpc_horizon=int(cfg.get("mpc_horizon", 3)),
                    mpc_samples=int(cfg.get("mpc_samples", 20)),
                )
                per_seed.append({"seed": int(seed), **m})
            results[layout][method] = aggregate_seed_metrics(per_seed)

    cfg_out = copy.deepcopy(cfg)
    cfg_out["policies"] = [spec]
    cfg_out["policy_tag"] = tag
    out_name = "s1_geometry_suite_smoke.json" if args.smoke else "s1_geometry_suite.json"
    out = args.out or str(ROOT / "results" / out_name)
    payload = {"methods": {tag: results}, "config": cfg_out, "smoke": bool(args.smoke)}
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
