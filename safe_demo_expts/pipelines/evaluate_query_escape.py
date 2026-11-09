"""Eval loop for query-vs-escape gate modes (H3)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from safe_demo_expts.envs.safety_point_goal import SafetyPointConfig, SafetyPointGoalEnv
from safe_demo_expts.interventions.oracle import RecoveryOracle
from safe_demo_expts.interventions.query_escape_gate import (
    GATE_MODES,
    GateMLP,
    decide_query,
    stuck_features,
)
from safe_demo_expts.interventions.variants import InterventionController, build_controller
from safe_demo_expts.models.bc import BCPolicy, act_policy
from safe_demo_expts.pipelines.evaluate_flexible import make_shield
from safe_demo_expts.pipelines.evaluate_safety_point import goal_distance, is_stuck
from safe_demo_expts.shield.safety_point_shield import ShieldResult


def evaluate_query_escape(
    policy: BCPolicy,
    mode: str = "escape_always",
    n_episodes: int = 20,
    seed: int = 0,
    deploy_margin: float = 0.14,
    oracle_margin: float = 0.04,
    stuck_horizon: int = 40,
    stuck_eps: float = 1e-3,
    device: str = "cpu",
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
    gate: Optional[GateMLP] = None,
    fail_threshold: float = 0.55,
    escape_try_steps: int = 30,
    escape_shield: str = "escape",
    mpc_horizon: int = 3,
    mpc_samples: int = 20,
    assist_variant: str = "V3b",
    assist_budget: int = 40,
    max_interventions_per_episode: int = 5,
    v3_disagree_tol: float = 0.45,
    v3_max_steps: int = 40,
    v2_n: int = 10,
    v3_agree_tol: float = 0.35,
    v4_horizon: int = 20,
    deep_penetration: float = -0.05,
) -> Dict[str, Any]:
    """
    Closed-loop eval with four gate modes.

    Metrics include ``mean_oracle_actions``, ``mean_gate_oracle_decisions``,
    ``mean_gate_escape_decisions``.
    """
    mode = (mode or "escape_always").lower()
    if mode not in GATE_MODES:
        raise ValueError(f"unknown mode={mode}; expected one of {GATE_MODES}")

    # Autonomous recovery shield (escape or MPC).
    auto_shield = make_shield(
        mode=escape_shield,
        deploy_margin=deploy_margin,
        mpc_horizon=mpc_horizon,
        mpc_samples=mpc_samples,
    )
    # When querying oracle we still need a shield that can exit; use escape.
    oracle_shield = make_shield(mode="escape", deploy_margin=deploy_margin)

    oracle = RecoveryOracle(oracle_margin=oracle_margin)
    controller = build_controller(
        variant=assist_variant,
        oracle_margin=oracle_margin,
        v2_n=v2_n,
        v3_agree_tol=v3_agree_tol,
        v3_disagree_tol=v3_disagree_tol,
        v3_max_steps=v3_max_steps,
        v4_horizon=v4_horizon,
        clearance_fn=lambda e: oracle_shield.clearance(e),
    )
    controller.oracle_fn = oracle

    cfg = SafetyPointConfig(env_id=env_id, max_steps=max_steps, seed=seed, stop_on_first_goal=True)
    env = SafetyPointGoalEnv(cfg)

    successes = 0
    terminal_stuck = 0
    ever_stuck = 0
    oracle_actions_eps: List[float] = []
    gate_oracle_eps: List[float] = []
    gate_escape_eps: List[float] = []
    reject_rates: List[float] = []
    lengths: List[float] = []
    costs: List[float] = []

    for ep in range(n_episodes):
        o, _ = env.reset(seed=(seed + 10_000 + ep))
        prev_a = np.zeros(2, dtype=np.float64)
        prev_rej = 0.0
        dist_hist: List[float] = []
        ep_rejects = 0
        ep_cost = 0.0
        ep_oracle_actions = 0
        ep_gate_oracle = 0
        ep_gate_escape = 0
        ep_interventions = 0
        ep_ever_stuck = False
        stuck_terminal = False
        info: Dict = {}
        t = 0
        controller.reset_episode()

        # Bout state: one gate decision per stuck bout.
        escaping = False
        escape_left = 0
        decision: Optional[str] = None
        in_bout = False

        for t in range(max_steps):
            a_base = act_policy(policy, o, prev_action=prev_a, prev_reject=prev_rej, device=device)
            clr = float(auto_shield.clearance(env))
            dist_hist.append(goal_distance(env))
            stuck_now = is_stuck(
                dist_hist,
                stuck_horizon=stuck_horizon,
                stuck_eps=stuck_eps,
                clearance=clr,
                deep_penetration=deep_penetration,
            )
            gated = stuck_now or (clr < deep_penetration and len(dist_hist) >= stuck_horizon)

            used_oracle = False
            a_prop = a_base
            active_shield = auto_shield

            if not gated and not controller.active and not escaping:
                in_bout = False
                decision = None

            if controller.active:
                if ep_oracle_actions >= assist_budget:
                    controller.state.active = False
                    if gated:
                        stuck_terminal = True
                        break
                else:
                    a_prop, used_oracle = controller.step(env, a_base)
                    active_shield = oracle_shield
            elif escaping:
                a_prop = a_base
                active_shield = auto_shield
                escape_left -= 1
                if escape_left <= 0:
                    escaping = False
                    clr2 = float(auto_shield.clearance(env))
                    still = is_stuck(
                        dist_hist,
                        stuck_horizon=stuck_horizon,
                        stuck_eps=stuck_eps,
                        clearance=clr2,
                        deep_penetration=deep_penetration,
                    ) or (clr2 < deep_penetration)
                    if still and mode == "try_escape_then_query":
                        decision = "oracle"
                        ep_gate_oracle += 1
                        if (
                            ep_interventions < max_interventions_per_episode
                            and ep_oracle_actions < assist_budget
                        ):
                            started = controller.maybe_start(
                                env, a_base, stuck_info={"clearance": clr2}
                            )
                            if started:
                                ep_interventions += 1
                                a_prop, used_oracle = controller.step(env, a_base)
                                active_shield = oracle_shield
                            else:
                                stuck_terminal = True
                                break
                        else:
                            stuck_terminal = True
                            break
                    else:
                        decision = None
                        in_bout = False
            elif gated and decision == "escape":
                # Continue autonomous escape for this bout (escape_always / learned).
                a_prop = a_base
                active_shield = auto_shield
            elif gated and not in_bout:
                ep_ever_stuck = True
                in_bout = True
                feats = stuck_features(env, clearance=clr, margin=deploy_margin)
                if mode == "try_escape_then_query":
                    choice = decide_query(
                        mode,
                        escape_try_exhausted=False,
                        fail_threshold=fail_threshold,
                        device=device,
                    )
                else:
                    choice = decide_query(
                        mode,
                        gate=gate,
                        feats=feats,
                        fail_threshold=fail_threshold,
                        device=device,
                    )
                decision = choice
                if choice == "escape":
                    ep_gate_escape += 1
                    if mode == "try_escape_then_query":
                        escaping = True
                        escape_left = int(escape_try_steps)
                    a_prop = a_base
                    active_shield = auto_shield
                else:
                    ep_gate_oracle += 1
                    if (
                        ep_interventions < max_interventions_per_episode
                        and ep_oracle_actions < assist_budget
                    ):
                        started = controller.maybe_start(
                            env, a_base, stuck_info={"clearance": clr, "features": feats}
                        )
                        if started:
                            ep_interventions += 1
                            a_prop, used_oracle = controller.step(env, a_base)
                            active_shield = oracle_shield
                        else:
                            stuck_terminal = True
                            break
                    else:
                        stuck_terminal = True
                        break

            sr: ShieldResult = active_shield(env, a_prop)
            if used_oracle:
                ep_oracle_actions += 1

            o, _r, terminated, truncated, info = env.step(sr.a_exec)
            prev_a = sr.a_exec.copy()
            prev_rej = 1.0 if sr.reject else 0.0
            if sr.reject:
                ep_rejects += 1
            ep_cost += float(info.get("cost", 0.0))

            if terminated or truncated:
                break

        success = bool(info.get("success"))
        if stuck_terminal and not success:
            terminal_stuck += 1
        elif (not success) and is_stuck(
            dist_hist,
            stuck_horizon=stuck_horizon,
            stuck_eps=stuck_eps,
            clearance=float(auto_shield.clearance(env)),
            deep_penetration=deep_penetration,
        ):
            terminal_stuck += 1
            ep_ever_stuck = True

        if ep_ever_stuck:
            ever_stuck += 1
        if success:
            successes += 1

        lengths.append(float(t + 1))
        reject_rates.append(ep_rejects / max(1, t + 1))
        costs.append(ep_cost)
        oracle_actions_eps.append(float(ep_oracle_actions))
        gate_oracle_eps.append(float(ep_gate_oracle))
        gate_escape_eps.append(float(ep_gate_escape))

    env.close()
    n = max(1, n_episodes)
    return {
        "mode_tag": mode,
        "success_rate": successes / n,
        "ever_stuck_rate": ever_stuck / n,
        "terminal_stuck_rate": terminal_stuck / n,
        "stuck_rate": terminal_stuck / n,
        "mean_oracle_actions": float(np.mean(oracle_actions_eps)) if oracle_actions_eps else 0.0,
        "mean_gate_oracle_decisions": float(np.mean(gate_oracle_eps)) if gate_oracle_eps else 0.0,
        "mean_gate_escape_decisions": float(np.mean(gate_escape_eps)) if gate_escape_eps else 0.0,
        "mean_reject_rate": float(np.mean(reject_rates)) if reject_rates else 0.0,
        "mean_episode_cost": float(np.mean(costs)) if costs else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": float(n_episodes),
        "fail_threshold": float(fail_threshold),
        "escape_try_steps": float(escape_try_steps),
        "escape_shield": escape_shield,
    }
