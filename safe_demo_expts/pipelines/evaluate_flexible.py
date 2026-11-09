"""General SafetyPointGoal eval loop: shield modes, interventions, recovery hooks."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from safe_demo_expts.envs.safety_point_goal import SafetyPointConfig, SafetyPointGoalEnv
from safe_demo_expts.interventions.oracle import RecoveryOracle
from safe_demo_expts.interventions.variants import InterventionController, build_controller
from safe_demo_expts.models.bc import BCPolicy, act_policy
from safe_demo_expts.pipelines.evaluate_safety_point import goal_distance, is_stuck
from safe_demo_expts.shield.mpc_escape import MPCEscapeShield
from safe_demo_expts.shield.safety_point_shield import SafetyPointShield, ShieldResult


RecoveryFn = Callable[
    [np.ndarray, np.ndarray, SafetyPointGoalEnv, Dict[str, Any]],
    np.ndarray,
]


def make_shield(
    mode: str = "escape",
    deploy_margin: float = 0.14,
    mpc_horizon: int = 3,
    mpc_samples: int = 24,
) -> Union[SafetyPointShield, MPCEscapeShield]:
    mode = (mode or "escape").lower()
    if mode in ("freeze", "reject", "hold"):
        return SafetyPointShield(margin=deploy_margin, escape_mode=False)
    if mode in ("escape", "a2", "clearance"):
        return SafetyPointShield(margin=deploy_margin, escape_mode=True)
    if mode in ("mpc", "a3", "mpc_escape"):
        return MPCEscapeShield(margin=deploy_margin, horizon=mpc_horizon, n_samples=mpc_samples)
    raise ValueError(f"unknown shield mode={mode}")


def evaluate_flexible(
    policy: BCPolicy,
    n_episodes: int = 20,
    seed: int = 0,
    deploy_margin: float = 0.14,
    oracle_margin: float = 0.04,
    stuck_horizon: int = 40,
    stuck_eps: float = 1e-3,
    device: str = "cpu",
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
    shield: str = "escape",
    escape_mode: Optional[bool] = None,
    intervention_variant: Optional[str] = None,
    assist_budget: Optional[int] = None,
    max_oracle_actions_per_episode: Optional[int] = None,
    max_interventions_per_episode: int = 5,
    use_recovery_fn: Optional[RecoveryFn] = None,
    collect_oracle_buffer: bool = False,
    v2_n: int = 10,
    v3_agree_tol: float = 0.35,
    v3_disagree_tol: float = 0.45,
    v3_max_steps: int = 40,
    v4_horizon: int = 20,
    mpc_horizon: int = 3,
    mpc_samples: int = 24,
    deep_penetration: float = -0.05,
    return_transitions: bool = False,
) -> Dict[str, Any]:
    """
    Closed-loop eval with optional oracle interventions and recovery hooks.

    Shield modes: ``freeze`` / ``escape`` / ``mpc``.
    If ``escape_mode`` is set, it overrides freeze/escape for ``SafetyPointShield``
    (ignored for MPC).

    Budgets:
      - ``assist_budget`` / ``max_oracle_actions_per_episode``: max oracle steps per ep
      - ``max_interventions_per_episode``: max intervention bursts per ep

    ``use_recovery_fn(obs, a_base, env, stuck_info) -> action`` fires when the
    stuck/clearance gate triggers and no intervention variant is active (A1c).

    When ``collect_oracle_buffer`` is True, returns ``oracle_transitions`` suitable
    for mini-DAgger / residual labels.
    """
    # Resolve shield
    if escape_mode is not None and (shield or "escape").lower() in ("freeze", "escape", "reject", "hold", "clearance", "a2"):
        shield_obj: Union[SafetyPointShield, MPCEscapeShield] = SafetyPointShield(
            margin=deploy_margin, escape_mode=bool(escape_mode)
        )
        shield_name = "escape" if escape_mode else "freeze"
    else:
        shield_obj = make_shield(
            mode=shield or "escape",
            deploy_margin=deploy_margin,
            mpc_horizon=mpc_horizon,
            mpc_samples=mpc_samples,
        )
        shield_name = (shield or "escape").lower()

    budget_cap = max_oracle_actions_per_episode
    if assist_budget is not None:
        budget_cap = assist_budget
    if budget_cap is None:
        budget_cap = 10**9

    variant = intervention_variant
    if variant is not None and str(variant).lower() in ("none", "null", ""):
        variant = None

    # Oracle / recovery actions must be able to *exit* a penetrated keep-out.
    # If the deploy shield is freeze, execute assisted steps with an escape shield.
    if isinstance(shield_obj, SafetyPointShield) and not getattr(shield_obj, "escape_mode", True):
        oracle_exec_shield: Union[SafetyPointShield, MPCEscapeShield] = SafetyPointShield(
            margin=deploy_margin, escape_mode=True
        )
    else:
        oracle_exec_shield = shield_obj

    oracle = RecoveryOracle(oracle_margin=oracle_margin)
    controller = build_controller(
        variant=variant or "none",
        oracle_margin=oracle_margin,
        v2_n=v2_n,
        v3_agree_tol=v3_agree_tol,
        v3_disagree_tol=v3_disagree_tol,
        v3_max_steps=v3_max_steps,
        v4_horizon=v4_horizon,
        clearance_fn=lambda e: shield_obj.clearance(e),
    )
    # Prefer shared RecoveryOracle instance for consistency.
    controller.oracle_fn = oracle

    cfg = SafetyPointConfig(env_id=env_id, max_steps=max_steps, seed=seed, stop_on_first_goal=True)
    env = SafetyPointGoalEnv(cfg)

    successes = 0
    ever_stuck = 0
    terminal_stuck = 0
    assisted_success = 0
    unassisted_success = 0
    oracle_actions_eps: List[float] = []
    interventions_eps: List[float] = []
    reject_rates: List[float] = []
    lengths: List[float] = []
    costs: List[float] = []

    oracle_obs: List[np.ndarray] = []
    oracle_actions: List[np.ndarray] = []
    oracle_a_base: List[np.ndarray] = []
    oracle_reject: List[float] = []
    oracle_traj: List[int] = []
    all_transitions: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        o, _ = env.reset(seed=(seed + 10_000 + ep))
        prev_a = np.zeros(2, dtype=np.float64)
        prev_rej = 0.0
        dist_hist: List[float] = []
        ep_rejects = 0
        ep_cost = 0.0
        ep_oracle_actions = 0
        ep_interventions = 0
        ep_ever_stuck = False
        ep_assisted = False
        stuck_terminal = False
        info: Dict = {}
        t = 0
        controller.reset_episode()

        can_assist = (variant is not None) or (use_recovery_fn is not None)
        # Start assist only after a post-step stuck detection (same criterion as
        # autonomous termination). Avoids pre-step false triggers that make
        # V2/V3b look worse than unassisted escape.
        pending_assist = False

        for t in range(max_steps):
            a_base = act_policy(policy, o, prev_action=prev_a, prev_reject=prev_rej, device=device)
            clr = float(shield_obj.clearance(env))
            stuck_info = {
                "clearance": clr,
                "goal_distance": goal_distance(env),
                "stuck": pending_assist,
                "t": t,
                "ep": ep,
                "prev_action": prev_a.copy(),
                "prev_reject": prev_rej,
            }

            used_oracle = False
            used_recovery = False
            a_prop = a_base

            if controller.active:
                if ep_oracle_actions >= budget_cap:
                    controller.state.active = False
                else:
                    a_prop, used_oracle = controller.step(env, a_base)
            elif pending_assist and can_assist:
                ep_ever_stuck = True
                pending_assist = False
                if use_recovery_fn is not None:
                    a_prop = np.asarray(
                        use_recovery_fn(o, a_base, env, stuck_info), dtype=np.float64
                    ).reshape(2)
                    used_recovery = True
                    ep_assisted = True
                elif (
                    variant is not None
                    and ep_interventions < max_interventions_per_episode
                    and ep_oracle_actions < budget_cap
                ):
                    started = controller.maybe_start(env, a_base, stuck_info=stuck_info)
                    if started:
                        ep_interventions += 1
                        ep_assisted = True
                        a_prop, used_oracle = controller.step(env, a_base)
                    else:
                        stuck_terminal = True
                        break
                elif variant is not None:
                    stuck_terminal = True
                    break

            exec_shield = oracle_exec_shield if (used_oracle or used_recovery or controller.active) else shield_obj
            sr: ShieldResult = exec_shield(env, a_prop)
            if used_oracle:
                ep_oracle_actions += 1
                ep_assisted = True
                if collect_oracle_buffer:
                    oracle_obs.append(o.copy())
                    oracle_actions.append(np.asarray(a_prop, dtype=np.float64).reshape(2))
                    oracle_a_base.append(np.asarray(a_base, dtype=np.float64).reshape(2))
                    oracle_reject.append(1.0 if sr.reject else 0.0)
                    oracle_traj.append(ep)

            if return_transitions:
                all_transitions.append(
                    {
                        "obs": o.copy(),
                        "a_base": a_base.copy(),
                        "a_prop": np.asarray(a_prop, dtype=np.float64).reshape(2),
                        "a_exec": sr.a_exec.copy(),
                        "reject": 1.0 if sr.reject else 0.0,
                        "oracle": 1.0 if used_oracle else 0.0,
                        "recovery": 1.0 if used_recovery else 0.0,
                        "traj_id": ep,
                        "clearance": sr.clearance,
                    }
                )

            o, _r, terminated, truncated, info = env.step(sr.a_exec)
            prev_a = sr.a_exec.copy()
            prev_rej = 1.0 if sr.reject else 0.0
            if sr.reject:
                ep_rejects += 1
            ep_cost += float(info.get("cost", 0.0))

            dist_hist.append(goal_distance(env))
            post_stuck = is_stuck(
                dist_hist,
                stuck_horizon=stuck_horizon,
                stuck_eps=stuck_eps,
                clearance=float(sr.clearance),
                deep_penetration=deep_penetration,
            )
            if post_stuck and (not controller.active):
                ep_ever_stuck = True
                if can_assist:
                    # Defer assist to the next step so detection matches autonomous.
                    if ep_interventions >= max_interventions_per_episode or ep_oracle_actions >= budget_cap:
                        stuck_terminal = True
                        break
                    pending_assist = True
                else:
                    stuck_terminal = True
                    break

            if terminated or truncated:
                break

        success = bool(info.get("success"))
        if stuck_terminal and not success:
            terminal_stuck += 1
        elif (not success) and is_stuck(
            dist_hist,
            stuck_horizon=stuck_horizon,
            stuck_eps=stuck_eps,
            clearance=float(shield_obj.clearance(env)),
            deep_penetration=deep_penetration,
        ):
            terminal_stuck += 1
            ep_ever_stuck = True

        if ep_ever_stuck:
            ever_stuck += 1
        if success:
            successes += 1
            if ep_assisted:
                assisted_success += 1
            else:
                unassisted_success += 1

        lengths.append(float(t + 1))
        reject_rates.append(ep_rejects / max(1, t + 1))
        costs.append(ep_cost)
        oracle_actions_eps.append(float(ep_oracle_actions))
        interventions_eps.append(float(ep_interventions))

    env.close()
    n = max(1, n_episodes)
    metrics: Dict[str, Any] = {
        "success_rate": successes / n,
        "ever_stuck_rate": ever_stuck / n,
        "terminal_stuck_rate": terminal_stuck / n,
        "stuck_rate": terminal_stuck / n,
        "assisted_success_rate": assisted_success / n,
        "unassisted_success_rate": unassisted_success / n,
        "mean_oracle_actions": float(np.mean(oracle_actions_eps)) if oracle_actions_eps else 0.0,
        "mean_interventions": float(np.mean(interventions_eps)) if interventions_eps else 0.0,
        "mean_reject_rate": float(np.mean(reject_rates)) if reject_rates else 0.0,
        "mean_episode_cost": float(np.mean(costs)) if costs else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": float(n_episodes),
        "variant": variant or "none",
        "budget": float(0.0 if variant is None else (budget_cap if budget_cap < 10**8 else -1.0)),
        "shield": shield_name,
    }

    if collect_oracle_buffer:
        if oracle_obs:
            metrics["oracle_transitions"] = {
                "obs": np.stack(oracle_obs),
                "actions": np.stack(oracle_actions),
                "a_base": np.stack(oracle_a_base),
                "reject_deploy": np.asarray(oracle_reject, dtype=np.float64),
                "traj_id": np.asarray(oracle_traj, dtype=np.int64),
            }
            metrics["n_oracle_transitions"] = float(len(oracle_obs))
        else:
            metrics["oracle_transitions"] = {
                "obs": np.zeros((0, getattr(policy.net[0], "in_features", 60) - int(getattr(policy, "history_extra", 0))), dtype=np.float64),
                "actions": np.zeros((0, 2), dtype=np.float64),
                "a_base": np.zeros((0, 2), dtype=np.float64),
                "reject_deploy": np.zeros((0,), dtype=np.float64),
                "traj_id": np.zeros((0,), dtype=np.int64),
            }
            metrics["n_oracle_transitions"] = 0.0

    if return_transitions:
        metrics["transitions"] = all_transitions

    return metrics
