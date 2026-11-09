"""Checkpoint I/O and metric aggregation for S1 SafetyPoint BC policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from safe_demo_expts.models.bc import BCPolicy


PathLike = Union[str, Path]


def checkpoint_stem(
    illegal_level: str,
    mode: str,
    use_history: bool,
    seed: int = 0,
) -> str:
    """Canonical stem: ``s1_{illegal}_mode{A|B|C}_hist{0|1}_seed{seed}``."""
    hist = 1 if use_history else 0
    return f"s1_{illegal_level}_mode{mode}_hist{hist}_seed{seed}"


def resolve_policy_path(
    checkpoint_dir: PathLike,
    illegal: str,
    mode: str,
    history: bool,
    seed: int = 0,
) -> Path:
    """
    Resolve a BC checkpoint path under ``checkpoint_dir``.

    Primary convention (used by bake-off / interventions / mini-DAgger results):
      ``s1_{illegal}_mode{mode}_hist{0|1}_seed{seed}.pt``

    Also accepts legacy / shorthand names if present on disk:
      ``s1_{illegal}_{mode}_{hist|nohist}_seed{seed}.pt``
      ``s1_{illegal}_{mode}_hist_seed{seed}.pt`` (hist only)
    """
    root = Path(checkpoint_dir)
    primary = root / f"{checkpoint_stem(illegal, mode, history, seed)}.pt"
    if primary.exists():
        return primary

    hist_tag = "hist" if history else "nohist"
    alts = [
        root / f"s1_{illegal}_{mode}_{hist_tag}_seed{seed}.pt",
        root / f"s1_{illegal}_{mode}_hist_seed{seed}.pt" if history else None,
        root / f"s1_{illegal}_{mode}_nohist_seed{seed}.pt" if not history else None,
        root / f"s1_{illegal}_mode{mode}_hist{1 if history else 0}.pt",
    ]
    for alt in alts:
        if alt is not None and alt.exists():
            return alt
    return primary


def save_policy(
    policy: BCPolicy,
    path: PathLike,
    meta: Optional[Dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    """Save ``BCPolicy`` state_dict + meta; optional JSON sidecar next to the ``.pt``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    obs_dim = int(meta.get("obs_dim", getattr(policy.net[0], "in_features", 0) - int(getattr(policy, "history_extra", 0))))
    history_extra = int(getattr(policy, "history_extra", meta.get("history_extra", 0)))
    use_history = bool(getattr(policy, "use_history", meta.get("use_history", history_extra > 0)))
    act_dim = int(policy.net[-1].out_features)
    hidden = int(policy.net[0].out_features)
    meta.setdefault("obs_dim", obs_dim)
    meta.setdefault("act_dim", act_dim)
    meta.setdefault("hidden", hidden)
    meta.setdefault("history_extra", history_extra)
    meta.setdefault("use_history", use_history)
    meta.setdefault("history", use_history)

    payload = {
        "state_dict": policy.state_dict(),
        "meta": meta,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden": hidden,
        "history_extra": history_extra,
        "use_history": use_history,
    }
    torch.save(payload, path)
    if write_sidecar:
        side = path.with_suffix(".json")
        with open(side, "w") as f:
            json.dump(meta, f, indent=2)
    return path


def load_policy(
    path: PathLike,
    device: str = "cpu",
    map_location: Optional[str] = None,
) -> tuple[BCPolicy, Dict[str, Any]]:
    """Load a ``BCPolicy`` and its meta dict from ``save_policy`` output."""
    path = Path(path)
    loc = map_location or device
    try:
        payload = torch.load(path, map_location=loc, weights_only=False)
    except TypeError:  # older torch
        payload = torch.load(path, map_location=loc)

    if isinstance(payload, dict) and "state_dict" in payload:
        meta = dict(payload.get("meta") or {})
        obs_dim = int(payload.get("obs_dim", meta.get("obs_dim", 60)))
        act_dim = int(payload.get("act_dim", meta.get("act_dim", 2)))
        hidden = int(payload.get("hidden", meta.get("hidden", 128)))
        history_extra = int(payload.get("history_extra", meta.get("history_extra", 0)))
        use_history = bool(payload.get("use_history", meta.get("use_history", history_extra > 0)))
        state = payload["state_dict"]
    else:
        # raw state_dict + sidecar
        state = payload
        side = path.with_suffix(".json")
        meta = {}
        if side.exists():
            with open(side) as f:
                meta = json.load(f)
        obs_dim = int(meta.get("obs_dim", 60))
        act_dim = int(meta.get("act_dim", 2))
        hidden = int(meta.get("hidden", 128))
        history_extra = int(meta.get("history_extra", 0))
        use_history = bool(meta.get("use_history", history_extra > 0))

    policy = BCPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, history_extra=history_extra)
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    policy.history_extra = history_extra  # type: ignore[attr-defined]
    policy.use_history = use_history  # type: ignore[attr-defined]
    meta.setdefault("obs_dim", obs_dim)
    meta.setdefault("act_dim", act_dim)
    meta.setdefault("hidden", hidden)
    meta.setdefault("history_extra", history_extra)
    meta.setdefault("use_history", use_history)
    meta.setdefault("history", use_history)
    return policy, meta


def aggregate_seed_metrics(
    per_seed: Sequence[Dict[str, Any]],
    keys: Optional[Iterable[str]] = None,
    skip_non_numeric: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate a list of per-seed metric dicts into ``{key}_mean`` / ``{key}_std``.

    Matches the shape used in ``results/s1_paper_bakeoff_*.json``.
    """
    rows = list(per_seed)
    out: Dict[str, Any] = {"n_seeds": len(rows), "per_seed": rows}
    if not rows:
        return out

    if keys is None:
        numeric_keys: List[str] = []
        for k, v in rows[0].items():
            if k == "seed":
                continue
            if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                numeric_keys.append(k)
            elif skip_non_numeric:
                continue
        keys = numeric_keys

    for k in keys:
        vals = []
        for r in rows:
            if k not in r:
                continue
            v = r[k]
            if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                vals.append(float(v))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std(ddof=0))
    return out
