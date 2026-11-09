# Query-vs-escape gate (H3)

## Claim

Call the oracle **only when autonomous escape/MPC is predicted to fail**. Otherwise recover with A2/A3 and save labels.

## Modes

| Mode | Behavior |
|---|---|
| `escape_always` | On stuck → A2/A3 only (0 oracle) |
| `oracle_always` | On stuck → V3b assist |
| `try_escape_then_query` | Try escape for `escape_try_steps`; escalate to oracle if still stuck |
| `learned` | MLP on privileged stuck features → \(P(\text{escape fails})\); query if \(P \ge \tau\) |

Features (dim 8): clearance, penetrate depth, nearest hazard dist, #nearby hazards, goal dist, away/goal heading alignment, hazard count.

## Code

| Piece | Path |
|---|---|
| Gate + trainer | `safe_demo_expts/interventions/query_escape_gate.py` |
| Eval loop | `safe_demo_expts/pipelines/evaluate_query_escape.py` |
| Runner | `python -m safe_demo_expts.scripts.run_s1_query_escape_gate` |
| Config | `configs/s1_query_escape_gate.yaml` |

## Smoke (A_nohist, seed 7, n_eval=8)

| Mode | succ | stuck | mean oracle acts |
|---|---|---|---|
| escape_always | 1.00 | 0.00 | 0.0 |
| oracle_always | 0.50 | 0.50 | 20.0 |
| try_escape_then_query | 1.00 | 0.00 | 0.0 |
| learned | 1.00 | 0.00 | 15.0 |

On easy A_nohist, escape already solves stuck — the gate’s job is to **avoid unnecessary queries**. Harder C policies / U-trap geometries are where `learned` should beat `escape_always` on stuck while using fewer oracle acts than `oracle_always`.

## Commands

```bash
python -m safe_demo_expts.scripts.run_s1_query_escape_gate --smoke --policies A_nohist
python -m safe_demo_expts.scripts.run_s1_query_escape_gate --policies A_nohist,C_nohist
```
