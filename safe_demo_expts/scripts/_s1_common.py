"""Shared helpers for S1 paper-queue runners (policy tags, ckpts, demos, smoke)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_demo_expts.data.generate_safety_point_demos import (
    filter_dataset,
    generate_demos,
    load_demos_npz,
    s1_margin_pair,
)
from safe_demo_expts.models.bc import BCPolicy, train_bc
from safe_demo_expts.util_ckpt import load_policy, resolve_policy_path, save_policy

PathLike = Union[str, Path]

POLICY_TAG_MAP: Dict[str, Dict[str, Any]] = {
    "A_nohist": {"illegal_level": "mid", "mode": "A", "use_history": False, "tag": "A_nohist"},
    "C_nohist": {"illegal_level": "mid", "mode": "C", "use_history": False, "tag": "C_nohist"},
    "C_hist": {"illegal_level": "mid", "mode": "C", "use_history": True, "tag": "C_hist"},
    "B_hist": {"illegal_level": "mid", "mode": "B", "use_history": True, "tag": "B_hist"},
}


def project_root() -> Path:
    return ROOT


def load_yaml(path: PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p: PathLike, root: Optional[Path] = None) -> Path:
    root = root or ROOT
    path = Path(p)
    return path if path.is_absolute() else root / path


def parse_policy_tags(s: Optional[str], default: Optional[Sequence[str]] = None) -> List[str]:
    if s is None or str(s).strip() == "":
        return list(default or ["A_nohist", "C_hist"])
    return [t.strip() for t in str(s).split(",") if t.strip()]


def policy_spec(tag: str) -> Dict[str, Any]:
    if tag not in POLICY_TAG_MAP:
        raise ValueError(f"unknown policy tag={tag}; known={list(POLICY_TAG_MAP)}")
    return dict(POLICY_TAG_MAP[tag])


def specs_from_config(cfg: dict, tags: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    if tags:
        return [policy_spec(t) for t in tags]
    policies = cfg.get("policies") or []
    out = []
    for p in policies:
        if isinstance(p, str):
            out.append(policy_spec(p))
        else:
            tag = p.get("tag")
            if tag and tag in POLICY_TAG_MAP:
                out.append(policy_spec(tag))
            else:
                out.append(
                    {
                        "illegal_level": p.get("illegal_level", "mid"),
                        "mode": p["mode"],
                        "use_history": bool(p.get("use_history", False)),
                        "tag": tag
                        or f"{p['mode']}_{'hist' if p.get('use_history') else 'nohist'}",
                    }
                )
    return out


def demo_path(demo_dir: Path, illegal: str, n_demos: int = 40, seed: int = 0) -> Path:
    return demo_dir / f"s1_pointgoal1_{illegal}_n{n_demos}_seed{seed}.npz"


def load_or_generate_demos(
    demo_dir: Path,
    illegal: str = "mid",
    n_demos: int = 40,
    seed: int = 0,
    deploy_margin: float = 0.14,
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
) -> Dict[str, Any]:
    path = demo_path(demo_dir, illegal, n_demos=n_demos, seed=seed)
    demo_m, deploy_m = s1_margin_pair(illegal, deploy_margin=deploy_margin)
    if path.exists():
        raw = load_demos_npz(path)
        meta = dict(raw.get("meta") or {})
        meta.setdefault("demo_margin", demo_m)
        meta.setdefault("deploy_margin", deploy_m)
        raw["meta"] = meta
        return raw
    print(f"  demo bank missing; generating {path}")
    return generate_demos(
        n_traj=n_demos,
        deploy_margin=deploy_m,
        demo_margin=demo_m,
        seed=seed,
        max_steps=max_steps,
        env_id=env_id,
        save_path=path,
    )


def ensure_policy(
    spec: Dict[str, Any],
    checkpoint_dir: Path,
    demo_dir: Path,
    device: str = "cpu",
    policy_seed: int = 0,
    n_demos: int = 40,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    deploy_margin: float = 0.14,
    max_steps: int = 500,
    env_id: str = "SafetyPointGoal1-v0",
) -> Tuple[BCPolicy, Dict[str, Any], Path]:
    """
    Load ``results/checkpoints/s1_{illegal}_mode{A|B|C}_hist{0|1}_seed{policy_seed}.pt``
    if present; otherwise train from demos and save.
    """
    illegal = spec["illegal_level"]
    mode = spec["mode"]
    use_history = bool(spec["use_history"])
    ckpt = resolve_policy_path(checkpoint_dir, illegal, mode, use_history, seed=policy_seed)
    if ckpt.exists():
        policy, meta = load_policy(ckpt, device=device)
        return policy, meta, ckpt

    print(f"  checkpoint missing ({ckpt}); training from demos")
    raw = load_or_generate_demos(
        demo_dir,
        illegal=illegal,
        n_demos=n_demos,
        seed=policy_seed,
        deploy_margin=deploy_margin,
        max_steps=max_steps,
        env_id=env_id,
    )
    ds = filter_dataset(raw, mode=mode)
    policy, train_metrics = train_bc(
        obs=ds["obs"],
        actions=ds["actions"],
        reject=ds["reject_deploy"],
        traj_id=ds["traj_id"],
        use_history=use_history,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=policy_seed,
    )
    demo_m, deploy_m = s1_margin_pair(illegal, deploy_margin=deploy_margin)
    meta = {
        "illegal_level": illegal,
        "mode": mode,
        "use_history": use_history,
        "history": use_history,
        "seed": policy_seed,
        "obs_dim": int(ds["obs"].shape[1]),
        "act_dim": int(ds["actions"].shape[1]),
        "history_extra": int(getattr(policy, "history_extra", 0)),
        "deploy_margin": deploy_m,
        "demo_margin": demo_m,
        "env_id": env_id,
        "tag": spec.get("tag"),
        **train_metrics,
    }
    save_policy(policy, ckpt, meta=meta)
    return policy, meta, ckpt


def apply_smoke(cfg: dict, smoke: bool) -> dict:
    """Shrink seeds / n_eval / epochs using smoke_* keys when ``smoke`` is True."""
    cfg = copy.deepcopy(cfg)
    if not smoke:
        return cfg
    if "smoke_n_eval" in cfg:
        cfg["n_eval"] = cfg["smoke_n_eval"]
    else:
        cfg["n_eval"] = min(int(cfg.get("n_eval", 20)), 8)
    if "smoke_collect_episodes" in cfg:
        cfg["collect_episodes"] = cfg["smoke_collect_episodes"]
    if "smoke_label_episodes" in cfg:
        cfg["label_episodes"] = cfg["smoke_label_episodes"]
    if "smoke_ft_epochs" in cfg:
        cfg["ft_epochs"] = cfg["smoke_ft_epochs"]
    if "smoke_residual_epochs" in cfg:
        cfg["residual_epochs"] = cfg["smoke_residual_epochs"]
    if "smoke_gate_epochs" in cfg:
        cfg["gate_epochs"] = cfg["smoke_gate_epochs"]
    if "smoke_epochs" in cfg:
        cfg["epochs"] = cfg["smoke_epochs"]
    # Prefer a single seed for smoke unless smoke_seeds is set.
    if "smoke_seeds" in cfg:
        cfg["seeds"] = list(cfg["smoke_seeds"])
    else:
        seeds = list(cfg.get("seeds") or [7])
        cfg["seeds"] = seeds[:1]
    return cfg


def dump_result(path: PathLike, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def metric_subset(m: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {k: m[k] for k in keys if k in m}
