# Tabletop track (Meta-World Sawyer reach-v3)

**Robot: Meta-World Sawyer (`reach-v3`) — NOT Franka.**  
robosuite / Franka–Panda is deferred; this track uses Sawyer for a quick pillar keep-out analog of the nav spine.

Spine: **escape / MPC shields** with frozen BC (not DAgger). Venv: `.venv-tabletop` (`MUJOCO_GL=egl`, `PYTHONPATH=.`).

## Setup

- Env: Meta-World `reach-v3` (Sawyer; 4D delta+grip)
- Deploy shield: workspace + **vertical pillar** keep-out at `(0, 0.72)`  
  `radius=0.08`
- Train: BC modes A/B/C · Eval: freeze / escape / MPC · 20 eps × seeds `{7,77,177}`

## Illegal-demo harden (multi-seed) — `docs/s1-tabletop-harden.json`

Controlled mismatch: `demo_margin=0.0`, `deploy_margin=0.05` → **illegal_rate≈0.188**, demos 39/40 success.

| Mode | Freeze succ / stuck | Escape | **MPC** |
|---|---|---|---|
| A (filter illegal) | 0.00±0.00 / 0.82±0.05 | 0.00±0.00 / 0.78±0.05 | 0.03±0.02 / **0.03±0.02** |
| **B (keep all)** | 0.43±0.10 / 0.22±0.05 | 0.55±0.04 / 0.10±0.08 | **0.77±0.10 / 0.03±0.02** |
| C (relabel) | 0.22±0.10 / 0.15±0.11 | 0.30±0.08 / 0.08±0.05 | 0.28±0.12 / 0.08±0.02 |

**Paper panel:** Mode **B + MPC**.  
Mode **A**: **MPC unsticks but can’t invent competence** (stuck↓, success stays ~0).

## Visual samples (promote)

| File | Note |
|---|---|
| [`media/irl-expts/tabletop_freeze_vs_mpc_contrast.gif`](../media/irl-expts/tabletop_freeze_vs_mpc_contrast.gif) | **canonical** seed **7** |
| [`media/irl-expts/tabletop_freeze_vs_mpc_seed7.gif`](../media/irl-expts/tabletop_freeze_vs_mpc_seed7.gif) | same |
| [`media/irl-expts/tabletop_freeze_vs_mpc_seed177.gif`](../media/irl-expts/tabletop_freeze_vs_mpc_seed177.gif) | second contrast seed |
| [`media/irl-expts/tabletop_freeze_vs_mpc_seed77.gif`](../media/irl-expts/tabletop_freeze_vs_mpc_seed77.gif) | freeze FAIL / MPC SUCCESS |

## Status

Illegal-demo harden **done**. Franka deferred. Next compute: thin VoI selective-assist on nav.
