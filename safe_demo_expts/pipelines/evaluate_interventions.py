"""Thin wrapper: V1–V4 oracle intervention probe on SafetyPointGoal."""

from __future__ import annotations

from typing import Any, Dict, Optional

from safe_demo_expts.models.bc import BCPolicy
from safe_demo_expts.pipelines.evaluate_flexible import evaluate_flexible


def evaluate_interventions(
    policy: BCPolicy,
    variant: str = "V3b",
    n_episodes: int = 20,
    seed: int = 0,
    deploy_margin: float = 0.14,
    oracle_margin: float = 0.04,
    stuck_horizon: int = 40,
    device: str = "cpu",
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
    escape_mode: bool = True,
    assist_budget: int = 80,
    max_interventions_per_episode: int = 3,
    collect_oracle_buffer: bool = False,
    v2_n: int = 10,
    v3_agree_tol: float = 0.35,
    v3_disagree_tol: float = 0.45,
    v3_max_steps: int = 40,
    v4_horizon: int = 20,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Probe one intervention variant under an escape-capable shield by default.

    Pass ``variant='none'`` for the unassisted baseline.
    """
    shield = "escape" if escape_mode else "freeze"
    return evaluate_flexible(
        policy,
        n_episodes=n_episodes,
        seed=seed,
        deploy_margin=deploy_margin,
        oracle_margin=oracle_margin,
        stuck_horizon=stuck_horizon,
        device=device,
        max_steps=max_steps,
        env_id=env_id,
        shield=shield,
        escape_mode=escape_mode,
        intervention_variant=variant,
        assist_budget=assist_budget,
        max_interventions_per_episode=max_interventions_per_episode,
        collect_oracle_buffer=collect_oracle_buffer,
        v2_n=v2_n,
        v3_agree_tol=v3_agree_tol,
        v3_disagree_tol=v3_disagree_tol,
        v3_max_steps=v3_max_steps,
        v4_horizon=v4_horizon,
        **kwargs,
    )


def evaluate_variant_grid(
    policy: BCPolicy,
    variants: Optional[list] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Run a small grid of variants; returns ``{variant_name: metrics}``."""
    variants = variants or ["none", "V1", "V2", "V3a", "V3b", "V4"]
    out: Dict[str, Dict[str, Any]] = {}
    for v in variants:
        name = v if isinstance(v, str) else v.get("name", "none")
        extra = {} if isinstance(v, str) else {k: val for k, val in v.items() if k != "name"}
        merged = dict(kwargs)
        merged.update(extra)
        out[name] = evaluate_interventions(policy, variant=name, **merged)
    return out
