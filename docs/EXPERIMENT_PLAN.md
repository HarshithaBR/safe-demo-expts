# Experiment plan

Aligned with the research note: filter-first (A) vs train-then-constrain (B) vs projected labels (C), plus history and (later) interventions.

## Phase 0 — smoke (done on cloud)

```bash
PYTHONPATH=. python safe_demo_expts/scripts/run_s0_matrix.py --smoke
```

Checks: demos generate, A/B/C train, eval metrics JSON writes.

## Phase 1 — S0 matrix (this pack)

Config: `configs/s0_pointnav.yaml`

Axes:

- `illegal_levels`: low / mid / high (demo margin gap)
- `modes`: A / B / C
- `history_options`: false / true
- `seeds`: 0,1,2

Primary plots (post-process `results/s0_matrix.json`):

1. success vs illegal_rate by mode  
2. stuck_rate vs illegal_rate  
3. history on/off delta for stuck_rate  

**Kill criteria:** if C ≥ B ≥ A everywhere with no interesting regime split, do not expand the dichotomy — pivot to shield-in-loop method or interventions (see research note §12).

## Phase 2 — Safety-Gymnasium

- Env: `SafetyPointGoal1-v0` / `SafetyCarGoal1-v0`
- Same A/B/C API
- Scripted or MPC expert with relaxed hazard inflation

## Phase 3 — Tabletop

- robomimic Lift/Can or Meta-World reach/push
- Shields: `z >= z_min`, workspace box
- Same logging schema

## Phase 4 — Interventions (V1–V4)

On stuck: single step / hold-n / disagree-until / oracle sequence, budget-matched.

## Logging schema (all phases)

```text
obs, a_prop, a_exec, shield_reject, traj_id, success, stuck
```
