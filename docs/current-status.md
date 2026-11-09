# Current status — SAFE demonstrations

## Spine

Nav: **escape / MPC shields** with frozen BC (multi-seed hardened).  
Tabletop: **illegal-demo harden done** on Meta-World **Sawyer** reach-v3 + pillar (NOT Franka).

## Tabletop (Sawyer, not Franka)

Harden (`results/s1_tabletop_harden.json`): `demo_margin=0`, `deploy_margin=0.05`, illegal_rate≈0.188, seeds `{7,77,177}`.

| Mode | Freeze succ/stuck | Escape | **MPC** |
|---|---|---|---|
| A | 0.00 / 0.82 | 0.00 / 0.78 | 0.03 / 0.03 |
| **B** | 0.43 / 0.22 | 0.55 / 0.10 | **0.77 / 0.03** |
| C | 0.22 / 0.15 | 0.30 / 0.08 | 0.28 / 0.08 |

**Panel = B + MPC.** Mode A: MPC unsticks but can’t invent competence.  
Promoted GIFs: seeds **7** + **177** (+ contrast composite). Docs: `docs/TABLETOP.md`.

## Nav (locked)

Freeze stuck ~0.30–0.35 → escape ~0.05–0.13 → MPC ≤ escape.  
Docs: `docs/MPC_HARDEN.md`, `docs/PAPER_OUTLINE.md`.

## VoI selective-assist probe — **DROP**

Mid-illegal A_nohist + freeze; gates stuck / uncertainty / VoI; budgets 10/20/40.  
Stuck stays ~0.20 for all gates (no absolute lift). Docs: `docs/VOI_PROBE.md`.  
Franka still deferred. No extra tabletop seed / H–N sweep needed (paper E2 already done).
