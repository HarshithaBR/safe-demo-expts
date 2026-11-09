# A1c — Gated recovery residual

## Idea

Naïve A1b fine-tunes BC on oracle stuck corrections and **destroys task competence** (A_nohist bake-off: post-FT stuck 0.67 vs freeze 0.30).

**A1c** keeps the base BC frozen and trains a small residual \(\delta a\) only on oracle (stuck-time) labels:

\[
a_{\mathrm{rec}} = \mathrm{clip}(a_{\mathrm{base}} + \delta_\theta(x))
\]

At deploy under a **freeze** shield, the residual is used **only when the stuck/clearance gate fires** (`evaluate_flexible.use_recovery_fn`); otherwise the frozen base acts alone.

This is still interactive learning (labels update \(\delta_\theta\)), without overwriting the task policy.

## Code

| Piece | Path |
|---|---|
| Model / train | `safe_demo_expts/models/gated_recovery.py` |
| Runner | `python -m safe_demo_expts.scripts.run_s1_gated_recovery` |
| Config | `configs/s1_gated_recovery.yaml` |

Compared methods per seed: `freeze`, `A1b_naive_ft`, `A1c_gated_residual`, `A2_escape`.

## Smoke (A_nohist, seed 7, n_eval=8)

| Method | succ | stuck |
|---|---|---|
| freeze | 0.75 | 0.25 |
| A1b naïve FT | 0.50 | 0.50 |
| **A1c gated residual** | **0.75** | **0.25** |
| A2 escape | 1.00 | 0.00 |

A1c does **not** collapse like A1b. Beating A2 on easy A_nohist may be hard; the interesting regime is when escape fails (U-traps / C policies).

## Commands

```bash
python -m safe_demo_expts.scripts.run_s1_gated_recovery --smoke --policies A_nohist
python -m safe_demo_expts.scripts.run_s1_gated_recovery --policies A_nohist
python -m safe_demo_expts.scripts.run_s1_gated_recovery --policies C_nohist,C_hist
```

## Full A_nohist (seeds 7,77,177; n_eval=20)

| Method | succ | stuck |
|---|---|---|
| freeze | 0.68±0.06 | 0.30±0.08 |
| A1b naïve FT | 0.27±0.02 | 0.65±0.07 |
| **A1c gated residual** | **0.68±0.02** | **0.30±0.04** |
| A2 escape | 0.93±0.05 | 0.05±0.04 |

A1c **preserves** base competence (matches freeze; no A1b collapse) but does not yet beat A2 on this easy policy.
