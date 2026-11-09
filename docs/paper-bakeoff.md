# Paper bake-off — matched nav comparison

**Script:** `python -m safe_demo_expts.scripts.run_s1_paper_bakeoff`  
**Config:** `configs/s1_paper_bakeoff.yaml`  
**Smoke result:** `results/s1_paper_bakeoff_smoke.json`

## Methods (matched info)

| Tag | Role |
|---|---|
| freeze | Reject-only baseline |
| **A1a** | Interactive assistance (V3b, no FT) |
| **A1b_postft** | Interactive learning (collect → FT → autonomous re-eval) — **core IRL arm** |
| A2_escape | Autonomous baseline (clearance escape) |
| A3_mpc | Autonomous baseline (short-horizon MPC) |

Same frozen mid-illegal BC, same `deploy_margin=0.14`, same stuck detector, same privileged geometry.

## Smoke (A_nohist, seed 7, n_eval=8)

| Method | succ | stuck |
|---|---|---|
| freeze | 0.75 | 0.25 |
| A1a | 0.75 | 0.25 |
| A1b_postft | 0.50 | 0.50 |
| A2_escape | 1.00 | 0.00 |
| A3_mpc | 1.00 | 0.00 |

Smoke is too small to conclude; A1b looking worse than freeze matches earlier mini-DAgger probes (assist ≠ lasting learning). Full multi-seed run is next.

## Commands

```bash
# smoke
python -m safe_demo_expts.scripts.run_s1_paper_bakeoff --smoke

# full (all policies × seeds)
python -m safe_demo_expts.scripts.run_s1_paper_bakeoff

# one policy
python -m safe_demo_expts.scripts.run_s1_paper_bakeoff --policies A_nohist
```

## Next

1. Full bake-off JSON for A_nohist / C_nohist / C_hist  
2. Geometry suite (simple / U-trap / narrow / infeasible)  
3. Budget / H–N thin ablations after the main table locks  

## Full A_nohist (seeds 7,77,177; n_eval=20)
| Method | succ (mean±std) | stuck (mean±std) |
|---|---|---|
| freeze | 0.68±0.06 | 0.30±0.08 |
| A1a | 0.72±0.05 | 0.27±0.06 |
| A1b_postft | 0.27±0.02 | 0.67±0.05 |
| A2_escape | 0.93±0.05 | 0.05±0.04 |
| A3_mpc | 0.93±0.02 | 0.03±0.02 |

**Takeaway:** A2/A3 ≫ freeze; A1a only slightly helps online; **A1b post-FT hurts** on this policy (learning transfer fails).


## Full C_hist (seeds 7,77,177; n_eval=20)

| Method | succ | stuck |
|---|---|---|
| freeze | 0.63±0.08 | 0.35±0.07 |
| A1a assist | 0.75±0.04 | 0.23±0.05 |
| A1b post-FT | 0.42±0.10 | 0.55±0.12 |
| A2 escape | 0.85±0.04 | 0.13±0.02 |
| **A3 MPC** | **0.93±0.05** | **0.05±0.04** |

**Takeaway:** On C_hist, A1a helps online vs freeze; A1b FT still hurts; A3 ≳ A2 ≫ A1a ≫ freeze ≫ A1b.
