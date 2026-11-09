#!/usr/bin/env python3
"""MPC escape harden: shield family + H/N sweeps."""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List

import numpy as np

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


def _agg(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=0)) if len(arr) else float("nan"),
    }


def eval_shield(policy, cfg: dict, seed: int, shield: str, **kwargs) -> Dict[str, float]:
    return evaluate_flexible(
        policy,
        n_episodes=int(cfg["n_eval"]),
        seed=seed,
        deploy_margin=float(cfg["deploy_margin"]),
        stuck_horizon=int(cfg["stuck_horizon"]),
        device=cfg.get("device", "cpu"),
        max_steps=int(cfg["max_steps"]),
        env_id=cfg.get("env_id", "SafetyPointGoal1-v0"),
        shield=shield,
        intervention_variant=None,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "s1_mpc_harden.yaml"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = apply_smoke(load_yaml(args.config), args.smoke)
    if args.smoke and "smoke_seeds" in cfg:
        cfg["seeds"] = list(cfg["smoke_seeds"])
    specs = specs_from_config(cfg)

    demo_dir = resolve_path(cfg.get("demo_dir", "data/demos"))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "results/checkpoints"))

    shield_family: List[Dict[str, Any]] = []
    horizon_sweep: List[Dict[str, Any]] = []
    samples_sweep: List[Dict[str, Any]] = []

    for spec in specs:
        tag = spec["tag"]
        print(f"=== mpc harden policy={tag} ===")
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

        for sm in cfg.get("shield_modes") or []:
            name = sm["name"]
            cls = sm.get("cls", name)
            kw = dict(sm.get("kwargs") or {})
            succ, stuck = [], []
            for seed in cfg["seeds"]:
                if cls == "mpc":
                    m = eval_shield(
                        policy,
                        cfg,
                        int(seed),
                        "mpc",
                        mpc_horizon=int(kw.get("horizon", 3)),
                        mpc_samples=int(kw.get("n_samples", 20)),
                    )
                else:
                    escape = bool(kw.get("escape_mode", True))
                    m = eval_shield(
                        policy,
                        cfg,
                        int(seed),
                        "escape" if escape else "freeze",
                        escape_mode=escape,
                    )
                succ.append(m["success_rate"])
                stuck.append(m["terminal_stuck_rate"])
            sa, st = _agg(succ), _agg(stuck)
            shield_family.append(
                {
                    "tag": tag,
                    "mode": spec["mode"],
                    "use_history": spec["use_history"],
                    "shield": name,
                    "success_mean": sa["mean"],
                    "success_std": sa["std"],
                    "stuck_mean": st["mean"],
                    "stuck_std": st["std"],
                }
            )

        fixed_n = int(cfg.get("mpc_horizon_fixed_n", 20))
        for h in cfg.get("mpc_horizons") or [1, 2, 3, 5]:
            stuck = []
            succ = []
            for seed in cfg["seeds"]:
                m = eval_shield(
                    policy, cfg, int(seed), "mpc", mpc_horizon=int(h), mpc_samples=fixed_n
                )
                stuck.append(m["terminal_stuck_rate"])
                succ.append(m["success_rate"])
            st, sa = _agg(stuck), _agg(succ)
            horizon_sweep.append(
                {
                    "tag": tag,
                    "horizon": int(h),
                    "n_samples": fixed_n,
                    "stuck_mean": st["mean"],
                    "stuck_std": st["std"],
                    "success_mean": sa["mean"],
                    "success_std": sa["std"],
                }
            )

        fixed_h = int(cfg.get("mpc_samples_fixed_h", 3))
        for n in cfg.get("mpc_samples") or [8, 20, 40]:
            stuck, succ = [], []
            for seed in cfg["seeds"]:
                m = eval_shield(
                    policy, cfg, int(seed), "mpc", mpc_horizon=fixed_h, mpc_samples=int(n)
                )
                stuck.append(m["terminal_stuck_rate"])
                succ.append(m["success_rate"])
            st, sa = _agg(stuck), _agg(succ)
            samples_sweep.append(
                {
                    "tag": tag,
                    "horizon": fixed_h,
                    "n_samples": int(n),
                    "stuck_mean": st["mean"],
                    "stuck_std": st["std"],
                    "success_mean": sa["mean"],
                    "success_std": sa["std"],
                }
            )

    cfg_out = copy.deepcopy(cfg)
    payload = {
        "config": cfg_out,
        "results": {
            "shield_family": shield_family,
            "horizon_sweep": horizon_sweep,
            "samples_sweep": samples_sweep,
        },
        "smoke": bool(args.smoke),
    }
    out = args.out or str(ROOT / "results" / "s1_mpc_harden.json")
    path = dump_result(resolve_path(out), payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
