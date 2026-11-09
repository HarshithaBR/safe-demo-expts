# Experiment queue (sequential)

Primary policy: **C_hist**. A_nohist = ablation only.

| # | Task | Status |
|---|---|---|
| 1 | A1c gated residual on **C_hist** (3 seeds) | **done** |
| 2 | Query-vs-escape gate on **C_hist** (3 seeds) | **done** |
| 3 | Matched bake-off on **C_hist** (freeze / A1a / A1b / A2 / A3) | **done** |
| 4 | A1c hist ablation: **C_nohist** gated residual | queued |
| 5 | Geometry suite E3 (simple / U-trap / narrow / infeasible) | queued |
| 6 | Thin ablations (budget, MPC H/N, margin) after tables lock | queued |
| 7 | Tabletop transfer (Sawyer B) | queued last |

## VoI selective querying?

Already probed (`docs/VOI_PROBE.md` / `results/s1_voi_probe.json`): stuck stays ~0.20 across stuck / uncertainty / VoI gates; no absolute lift → **dropped from spine**.

Do **not** put VoI in the main queue. Optional late revisit only if the query-vs-escape gate fails to show an H3 story on C_hist + hard geometries.


## C_hist results (just finished)

**A1c:** freeze 0.63/0.35; A1b 0.42/0.55; A1c 0.63/0.35; A2 0.85/0.13

**Query-escape:** escape_always 0.83/0.15@0; oracle_always 0.57/0.42@20; try_escape_then_query 0.85/0.13@5; learned 0.82/0.17@9

Nothing running now — next is matched bake-off C_hist.

## Bake-off C_hist
A3 0.93/0.05 > A2 0.85/0.13 > A1a 0.75/0.23 > freeze 0.63/0.35 > A1b 0.42/0.55
Next: A1c C_nohist hist ablation.
