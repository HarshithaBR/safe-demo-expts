# S1 Oracle interventions (V1–V4) — probe, budgets, mini-DAgger

## Shift in question

| Before | Now |
|---|---|
| Headline: A vs B vs C | **Baselines only** |
| Primary metric: success | **Primary: `terminal_stuck_rate`** |
| Stall = episode fail | Stall **triggers oracle recovery** |

## Critical systems finding (before rankings matter)

First probe with a **freeze shield** (project to `a=0` when already inside the keep-out): **all V1–V4 had zero lift**. Diagnosis: stuck states had `clearance < 0` and reject rate 1.0 — the shield zeroed every oracle action.

**Fix:** shield **escape mode** — if penetrating, choose the action that *maximizes clearance* instead of freezing. Recovery oracle also backs away from the nearest hazard before goal-seeking.

→ Interventions only make sense on a shield that can **exit** unsafe sets, not only prevent entry.

## Variants

| ID | Behavior | Probe outcome |
|---|---|---|
| `none` | No help | baseline |
| **V1** | One oracle step | **≈ none** (too weak) |
| **V2** | Hold action `n=10` | clear lift |
| **V3a** | Until π≈oracle | weak / inconsistent |
| **V3b** | Until π leaves bad mode + safe | **best / tied best** |
| **V4** | Oracle trajectorylet `H=20` | strong, near V3b |

## Probe results (mid illegal, escape shield) — `results/s1_interventions_escape.json`

| Policy | none succ / stuck | best variant | best succ / stuck |
|---|---|---|---|
| A_nohist | 0.65 / **0.35** | **V3b** | **0.95 / 0.05** |
| B_hist | 0.75 / 0.25 | V2 / V3b / V4 | **0.95 / 0.05** |
| C_nohist | 0.35 / **0.45** | V3b / V4 | **0.60 / 0.20** |
| C_hist | 0.75 / 0.25 | V3b / V4 | **0.95 / 0.05** |

Plot: `results/figures/s1_interventions_escape.png`

## Budget–success / stuck curves — `results/s1_budget_curves.json`

Policies: A_nohist, C_nohist, C_hist. Variants: V2 / V3b / V4. Budgets: 0, 5, 10, 20, 40, 80. `n_eval=20`.

| Policy | none stuck | best @80 (variant) | stuck @40 V3b | notes |
|---|---|---|---|---|
| A_nohist | 0.35 | **0.05 (V3b)** / 0.10 (V4) | 0.20 | V2 plateaus ~0.20 |
| C_nohist | 0.45 | **0.20 (V3b/V4)** | 0.30 | never gets clean |
| C_hist | 0.25 | **0.05 (V3b/V4)** | 0.15 | smoothest curve |

Plot: `results/figures/s1_budget_curves.png`

Takeaway: primary DV moves monotonically with oracle budget on A and C_hist; need ~80 oracle actions/ep for stuck≈0.05. Eval-time assist is real.

## Mini-DAgger (V3b collect → fine-tune → autonomous) — `results/s1_minidagger.json`

Collect: V3b, budget 40, 40 eps; mix with original demos; 25 FT epochs; eval autonomous (no help).

| Policy | n_oracle | base auto succ/stuck | assist@40 succ/stuck | **ft auto succ/stuck** |
|---|---|---|---|---|
| A_nohist | 900 | 0.65 / 0.35 | 0.75 / 0.25 | **0.15 / 0.60** (worse) |
| C_nohist | 593 | 0.35 / 0.60 | 0.55 / 0.40 | 0.30 / 0.45 (flat/mixed) |
| C_hist | 712 | 0.50 / 0.50 | 0.80 / 0.20 | 0.50 / 0.45 (no transfer) |

Takeaway: **eval-time assist helps; aggregating interventions into BC fine-tune does not** — A collapses, C_hist stuck barely moves. Mini-DAgger is not clean on nav.

### Safer FT retry — `results/s1_minidagger_safe.json`

Oracle-only labels, `lr=1e-4`, 10 epochs (no demo mix). Still collapses:

| Policy | base auto | safe FT auto |
|---|---|---|
| A_nohist | 0.65 / 0.35 | **0.10 / 0.60** |
| C_nohist | 0.35 / 0.60 | 0.25 / 0.55 |
| C_hist | 0.50 / 0.50 | 0.25 / 0.55 |

Likely cause: fine-tuning only on near-stuck recovery states overwrites the nominal goal-seeking policy (catastrophic forgetting), not a logging bug.

## Go / no-go

**Probe + budgets: GO for eval-time oracle interventions** (V3b/V4 + escape shield).

**Mini-DAgger transfer: NO-GO** (hurts or flat; safer FT still fails).

**Tabletop: NO-GO for now.** Gate was “nav budget + DAgger stays clean”; budgets are clean, DAgger is not. Stay on nav if continuing — e.g. weighted mix / replay buffer / freeze early layers — before leaving the domain.

## Code

- `safe_demo_expts/interventions/variants.py`
- `safe_demo_expts/pipelines/evaluate_interventions.py`
- `safe_demo_expts/scripts/run_s1_interventions.py`
- `safe_demo_expts/scripts/run_s1_budget_curves.py`
- `safe_demo_expts/scripts/run_s1_minidagger.py`
- `safe_demo_expts/scripts/plot_s1_budget_curves.py`
- `safe_demo_expts/shield/safety_point_shield.py` (escape mode)
- `configs/s1_interventions.yaml`, `configs/s1_budget_curves.yaml`, `configs/s1_minidagger.yaml`
