# Experiment pack location

Runnable SAFE-demo experiment code lives on this machine at:

**`/home/ubuntu/Documents/IRL/project`**

(Mac Pro mirror path: `~/Documents/IRL/project`)

## What’s there

- S0 PointNav env + obstacle shield
- Demo generation with controllable illegality
- Pipelines **A / B / C** (+ optional action history)
- Smoke + matrix runner: `safe_demo_expts/scripts/run_s0_matrix.py`
- Mac setup: `docs/MAC_SETUP.md`
- Plan: `docs/EXPERIMENT_PLAN.md`
- Research framing: [`demo-filtering-vs-constrained-deploy.md`](./demo-filtering-vs-constrained-deploy.md)

## Quick commands

```bash
cd /home/ubuntu/Documents/IRL/project
PYTHONPATH=. python3 safe_demo_expts/scripts/run_s0_matrix.py --smoke
PYTHONPATH=. python3 safe_demo_expts/scripts/run_s0_matrix.py --out results/s0_matrix.json
```

## Status (cloud VM)

- S0 PointNav = unit test only
- **S1 SafetyPointGoal1 main track built and smoke-tested**
  - `safe_demo_expts/scripts/run_s1_safety_point.py`
  - wrapper + geometric hazard shield + scripted demos + A/B/C BC
  - smoke: demos 8/8 success, ~11% illegal; A success 0.60 vs B/C 0.40 (`results/s1_smoke.json`)
- Meta-World tabletop venv ready; A/B/C harness not wired yet
- Setup: `/home/ubuntu/Documents/IRL/project/docs/BASELINE_ENV_SETUP.md`
