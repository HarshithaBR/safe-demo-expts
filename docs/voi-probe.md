# Thin VoI selective-assist probe (nav)

Eval-time only — **no fine-tuning**. Mid-illegal **A_nohist**, **freeze** shield (`escape_mode=false`).  
Gates choose *when* to start a V3b oracle bout under fixed budgets `{10,20,40}`.

| Gate | Trigger |
|---|---|
| `stuck` | Flat progress / deep penetration (baseline V3b) |
| `uncertainty` | Stuck **or** shield would reject learner **or** clearance ≤ 0 |
| `voi` | Stuck **or** utility score ≥ 1.0 (`stuck + penetration + reject + flat`) |

Results: `results/s1_voi_probe.json` · script: `safe_demo_expts/scripts/run_s1_voi_probe.py`

## Results (succ / stuck / mean oracle)

| Gate | budget 0 | 10 | 20 | 40 |
|---|---|---|---|---|
| none / stuck | **0.80 / 0.20** / 0 | 0.80 / 0.20 / 3.5 | 0.80 / 0.20 / 6.5 | 0.80 / 0.20 / 12.5 |
| uncertainty | — | 0.75 / 0.25 / 3.9 | 0.75 / 0.25 / 7.3 | 0.80 / 0.20 / 12.6 |
| voi | — | 0.80 / 0.20 / 3.8 | 0.80 / 0.20 / 7.2 | 0.80 / 0.20 / 12.3 |

Primary DV `stuck_per_oracle`: VoI slightly better at budget 10–20 (0.053 / 0.028 vs stuck 0.057 / 0.031) but **absolute stuck never moves** (~0.20).

## Decision

**Flat → drop VoI from the spine.** No clear efficiency lift; uncertainty is weakly worse at low budget. Keep escape/MPC + budgeted V3b-on-stuck as the assist story when the base is weak; do not pursue utility-gated querying as a method claim.
