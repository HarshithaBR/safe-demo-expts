"""E3 geometry suite: layout kinds + placement / classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from safe_demo_expts.envs.safety_point_goal import SafetyPointGoalEnv

LAYOUT_KINDS = ("simple", "u_trap", "narrow", "infeasible")


@dataclass
class LayoutSpec:
    kind: str
    agent_xy: np.ndarray
    goal_xy: np.ndarray
    hazard_xy: np.ndarray
    notes: str = ""


def _set_body_xy(env: "SafetyPointGoalEnv", body_name: str, xy: np.ndarray) -> bool:
    """Best-effort MuJoCo body pose override. Returns True on success."""
    try:
        model = env._env.unwrapped.model  # type: ignore[attr-defined]
        data = env._env.unwrapped.data  # type: ignore[attr-defined]
        bid = int(model.body(body_name).id) if hasattr(model, "body") else None
        if bid is None:
            # Fallback: name lookup
            names = [model.body(i).name for i in range(model.nbody)]
            if body_name not in names:
                return False
            bid = names.index(body_name)
        data.xpos[bid, 0] = float(xy[0])
        data.xpos[bid, 1] = float(xy[1])
        return True
    except Exception:
        return False


def try_set_hazards_goal(
    env: "SafetyPointGoalEnv",
    hazard_xy: np.ndarray,
    goal_xy: np.ndarray,
    agent_xy: Optional[np.ndarray] = None,
) -> bool:
    """
    Attempt to override privileged layout via the task / MuJoCo API.

    Returns True if enough objects were moved to match the requested layout.
    """
    hazard_xy = np.asarray(hazard_xy, dtype=np.float64).reshape(-1, 2)
    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    ok = True
    try:
        # Prefer high-level task object setters when available.
        haz = getattr(env.task, "hazards", None)
        if haz is not None and hasattr(haz, "pos"):
            cur = np.asarray(haz.pos, dtype=np.float64)
            n = min(len(cur), len(hazard_xy))
            for i in range(n):
                try:
                    haz.pos[i][:2] = hazard_xy[i]
                except Exception:
                    ok = False
        goal = getattr(env.task, "goal", None)
        if goal is not None and hasattr(goal, "pos"):
            try:
                goal.pos[:2] = goal_xy
            except Exception:
                ok = False
        if agent_xy is not None:
            agent = getattr(env.task, "agent", None)
            if agent is not None and hasattr(agent, "pos"):
                try:
                    agent.pos[:2] = np.asarray(agent_xy, dtype=np.float64).reshape(2)
                except Exception:
                    ok = False
    except Exception:
        return False
    return ok


def layout_simple(scale: float = 1.0) -> LayoutSpec:
    """Open field: goal ahead, sparse hazards off the direct path."""
    return LayoutSpec(
        kind="simple",
        agent_xy=np.array([0.0, 0.0]),
        goal_xy=np.array([1.2 * scale, 0.0]),
        hazard_xy=np.array([[0.6 * scale, 0.9 * scale], [0.6 * scale, -0.9 * scale]]),
        notes="open path to goal",
    )


def layout_u_trap(scale: float = 1.0) -> LayoutSpec:
    """U-shaped hazard pocket: agent inside, opening opposite the goal."""
    # Three hazards forming a U opening toward -x; goal outside to +x.
    return LayoutSpec(
        kind="u_trap",
        agent_xy=np.array([0.0, 0.0]),
        goal_xy=np.array([1.5 * scale, 0.0]),
        hazard_xy=np.array(
            [
                [0.35 * scale, 0.0],  # front (blocks goal)
                [0.0, 0.45 * scale],  # left wall
                [0.0, -0.45 * scale],  # right wall
            ]
        ),
        notes="U-trap pocket; escape must reverse before seeking goal",
    )


def layout_narrow(scale: float = 1.0) -> LayoutSpec:
    """Narrow corridor between two hazards toward the goal."""
    gap = 0.55 * scale
    return LayoutSpec(
        kind="narrow",
        agent_xy=np.array([-0.8 * scale, 0.0]),
        goal_xy=np.array([1.0 * scale, 0.0]),
        hazard_xy=np.array([[0.0, gap / 2], [0.0, -gap / 2]]),
        notes="pinched corridor",
    )


def layout_infeasible(scale: float = 1.0) -> LayoutSpec:
    """Goal sealed by a hazard ring — autonomous escape cannot reach goal safely."""
    ring = []
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ring.append([0.55 * scale * np.cos(ang), 0.55 * scale * np.sin(ang)])
    return LayoutSpec(
        kind="infeasible",
        agent_xy=np.array([0.0, 0.0]),
        goal_xy=np.array([1.6 * scale, 0.0]),
        hazard_xy=np.asarray(ring, dtype=np.float64),
        notes="goal sealed; success only via oracle / luck",
    )


LAYOUT_BUILDERS = {
    "simple": layout_simple,
    "u_trap": layout_u_trap,
    "narrow": layout_narrow,
    "infeasible": layout_infeasible,
}


def make_layout(kind: str, scale: float = 1.0) -> LayoutSpec:
    kind = kind.lower().replace("-", "_")
    if kind == "utrap":
        kind = "u_trap"
    if kind not in LAYOUT_BUILDERS:
        raise ValueError(f"unknown layout kind={kind}; expected {LAYOUT_KINDS}")
    return LAYOUT_BUILDERS[kind](scale=scale)


def apply_layout(env: "SafetyPointGoalEnv", spec: LayoutSpec) -> bool:
    """Apply a LayoutSpec after reset; returns whether override succeeded."""
    return try_set_hazards_goal(env, spec.hazard_xy, spec.goal_xy, agent_xy=spec.agent_xy)


def classify_layout(
    env: "SafetyPointGoalEnv",
    deploy_margin: float = 0.14,
) -> str:
    """
    Heuristic layout classifier from privileged geometry (seed-filtering fallback).

    - infeasible: goal inside / behind a dense hazard cluster from agent
    - u_trap: hazards surround agent on 3+ sides with opening ≠ goal direction
    - narrow: two hazards form a tight gap on the agent→goal line
    - simple: otherwise
    """
    agent = env.agent_xy()
    goal = env.goal_xy()
    haz = env.hazard_xy()
    if len(haz) == 0:
        return "simple"

    r = float(env.hazard_radius()) + float(deploy_margin)
    goal_delta = goal - agent
    goal_dist = float(np.linalg.norm(goal_delta))
    goal_dir = goal_delta / max(goal_dist, 1e-8)

    # Count hazards near the goal and near the agent.
    d_agent = np.linalg.norm(haz - agent[None, :], axis=1)
    d_goal = np.linalg.norm(haz - goal[None, :], axis=1)
    near_agent = int(np.sum(d_agent < 1.0))
    near_goal = int(np.sum(d_goal < 0.8))

    # Infeasible: many hazards tightly around goal or agent boxed with goal blocked.
    if near_goal >= 3 or (near_agent >= 4 and near_goal >= 2):
        return "infeasible"

    # Narrow: hazards on opposite sides of the path with small gap.
    # Project hazards onto perpendicular of goal_dir; look for pair straddling the path.
    perp = np.array([-goal_dir[1], goal_dir[0]])
    along = (haz - agent) @ goal_dir
    across = (haz - agent) @ perp
    mid = (along > 0.15 * goal_dist) & (along < 0.85 * goal_dist)
    if mid.any():
        left = across[mid]
        if np.any(left > 0) and np.any(left < 0):
            gap = float(left[left > 0].min() - left[left < 0].max()) if (
                np.any(left > 0) and np.any(left < 0)
            ) else 10.0
            # gap between nearest left/right hazards (center-to-center minus radii)
            if gap - 2 * r < 0.7:
                return "narrow"

    # U-trap: hazards in ≥3 angular sectors around the agent, goal sector occupied.
    angs = np.arctan2(haz[:, 1] - agent[1], haz[:, 0] - agent[0])
    sectors = ((angs + np.pi) / (np.pi / 2)).astype(int) % 4
    occupied = set(int(s) for s in sectors if True)
    goal_ang = float(np.arctan2(goal_dir[1], goal_dir[0]))
    goal_sector = int(((goal_ang + np.pi) / (np.pi / 2)) % 4)
    if len(occupied) >= 3 and goal_sector in occupied and near_agent >= 3:
        return "u_trap"

    return "simple"


def find_seeds_for_layout(
    env_factory,
    kind: str,
    n: int,
    start_seed: int = 0,
    max_tries: int = 500,
    deploy_margin: float = 0.14,
) -> List[int]:
    """
    Scan environment seeds and keep those whose random layout classifies as ``kind``.

    ``env_factory`` should return a fresh ``SafetyPointGoalEnv``.
    """
    kind = kind.lower().replace("-", "_")
    if kind == "utrap":
        kind = "u_trap"
    found: List[int] = []
    env = env_factory()
    try:
        for offset in range(max_tries):
            if len(found) >= n:
                break
            seed = start_seed + offset
            env.reset(seed=seed)
            if classify_layout(env, deploy_margin=deploy_margin) == kind:
                found.append(seed)
    finally:
        env.close()
    return found


def reset_with_layout(
    env: "SafetyPointGoalEnv",
    kind: str,
    seed: int,
    prefer_override: bool = True,
    scale: float = 1.0,
    deploy_margin: float = 0.14,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reset env and try to realize ``kind``.

    Strategy: reset(seed) → try apply_layout override → if override fails, accept
    the random layout and report its classified kind.
    """
    obs, info = env.reset(seed=seed)
    info = dict(info)
    realized = "simple"
    overridden = False
    if prefer_override:
        spec = make_layout(kind, scale=scale)
        overridden = apply_layout(env, spec)
        if overridden:
            realized = spec.kind
        else:
            realized = classify_layout(env, deploy_margin=deploy_margin)
    else:
        realized = classify_layout(env, deploy_margin=deploy_margin)
    info["layout_kind"] = realized
    info["layout_requested"] = kind
    info["layout_overridden"] = bool(overridden)
    return obs, info


__all__ = [
    "LAYOUT_KINDS",
    "LayoutSpec",
    "make_layout",
    "apply_layout",
    "classify_layout",
    "find_seeds_for_layout",
    "reset_with_layout",
    "try_set_hazards_goal",
    "layout_simple",
    "layout_u_trap",
    "layout_narrow",
    "layout_infeasible",
]
