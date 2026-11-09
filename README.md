# SAFE Demo Experiments

Filter-first vs train-then-constrain vs projected-label BC under a deploy-time obstacle shield (S0 PointNav + S1 SafetyPoint).

## Quick setup

```bash
cd /path/to/this/repo
bash scripts/setup.sh          # micromamba/conda/venv + deps + SG patch
source scripts/env_safe311.sh  # or: micromamba activate safe311
export PYTHONPATH=. MUJOCO_GL=glfw
```

`scripts/setup.sh` is idempotent-ish: creates Python 3.11 env `safe311` (micromamba preferred), installs `requirements.txt` plus a pinned Safety-Gymnasium stack, and applies `scripts/patch_safety_gymnasium_dataclasses.py` when needed.

Cloud VM reference env:

```bash
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
eval "$(/home/ubuntu/bin/micromamba shell hook -s bash 2>/dev/null || /tmp/bin/micromamba shell hook -s bash)"
micromamba activate safe311
cd /home/ubuntu/Documents/IRL/project
export PYTHONPATH=. MUJOCO_GL=glfw
```

Mac notes: [`docs/MAC_SETUP.md`](docs/MAC_SETUP.md). Restore / regen notes: [`docs/RESTORE.md`](docs/RESTORE.md). Training layout: [`docs/training-protocol.md`](docs/training-protocol.md).

**MUJOCO_GL:** `glfw` (default) or `egl` on headless hosts with libEGL.  
**Tabletop / Meta-World:** not installed by setup — see [`docs/tabletop.md`](docs/tabletop.md) / Mac setup (separate env).

Large GIFs/figures are gitignored; keep JSON under `results/` and regenerate plots locally if needed.

## Pipelines

| Mode | Meaning |
|---|---|
| **A** filter-first | Drop transitions the deploy shield would reject, then BC |
| **B** train-then-constrain | BC on proposed actions; shield only at deploy |
| **C** projected labels | BC on shield-executed actions (shield-in-the-loop labels) |

History option: previous executed action + reject bit (`*_hist` vs `*_nohist`).

## S0 PointNav (no MuJoCo)

```bash
PYTHONPATH=. python -m safe_demo_expts.scripts.run_s0_matrix --smoke
PYTHONPATH=. python -m safe_demo_expts.scripts.run_s0_matrix \
  --config configs/s0_pointnav.yaml \
  --out results/s0_matrix.json
```

Plan: [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md).

## S1 SafetyPoint

```bash
# demos / A·B·C matrix (see configs/s1_safety_point.yaml)
python -m safe_demo_expts.scripts.run_s1_safety_point --smoke
```

Mid-illegal demo bank lives at `data/demos/s1_pointgoal1_mid_n40_seed0.npz`. Small BC checkpoints under `results/checkpoints/` (regen via runners if missing — see RESTORE).

### Interventions (V1–V4)

Oracle assist at stuck time; needs an **escape** shield (freeze-only gives zero lift). Docs: [`docs/interventions.md`](docs/interventions.md).

```bash
python -m safe_demo_expts.scripts.run_s1_interventions \
  --out results/s1_interventions_escape.json

# budget curves (V2 / V3b / V4 × budgets)
python -m safe_demo_expts.scripts.run_s1_budget_curves \
  --out results/s1_budget_curves.json

# mini-DAgger (collect → FT → autonomous)
python -m safe_demo_expts.scripts.run_s1_minidagger \
  --out results/s1_minidagger.json
```

### Paper bake-off (matched methods)

Freeze / A1a assist / A1b post-FT / A2 escape / A3 MPC. Docs: [`docs/paper-bakeoff.md`](docs/paper-bakeoff.md).

```bash
python -m safe_demo_expts.scripts.run_s1_paper_bakeoff --smoke
python -m safe_demo_expts.scripts.run_s1_paper_bakeoff --policies C_hist \
  --out results/s1_paper_bakeoff_C_hist.json
```

### A1c gated recovery

Frozen base BC + residual used only when the stuck gate fires (avoids naïve FT collapse). Docs: [`docs/gated-recovery.md`](docs/gated-recovery.md).

```bash
python -m safe_demo_expts.scripts.run_s1_gated_recovery --smoke
python -m safe_demo_expts.scripts.run_s1_gated_recovery --policies C_hist \
  --out results/s1_gated_recovery_C_hist.json
```

### Query-vs-escape gate / geometry / MPC harden

```bash
python -m safe_demo_expts.scripts.run_s1_query_escape_gate --policies C_hist \
  --out results/s1_query_escape_gate_C_hist.json
python -m safe_demo_expts.scripts.run_s1_geometry_suite --policy-tag C_hist \
  --out results/s1_geometry_suite.json
python -m safe_demo_expts.scripts.run_s1_mpc_harden --out results/s1_mpc_harden.json
```

## Full experiment queue

Seven sequential steps (A1c → query-escape → bakeoff → … → tabletop). Skips existing outputs unless `FORCE=1`.

```bash
mkdir -p results/queue
nohup bash scripts/run_all_paper_queue.sh >> results/queue/queue.nohup.out 2>&1 &
# SMOKE=1 / START_FROM=N / FORCE=1 supported
```

Details: [`docs/README_paper_queue.md`](docs/README_paper_queue.md). Early subset: `bash scripts/run_early_experiments.sh`.

## Layout

```
configs/                 # S0/S1 YAML
safe_demo_expts/         # envs, shield, BC, interventions, runners
scripts/                 # setup, env activate, SG patch, queues
data/demos/              # demo banks (.npz)
results/                 # JSON metrics + small checkpoints
docs/                    # scientific experiment docs (not proposals)
```

## Publishing this repo

If `origin` is not set (no GitHub credentials in the environment):

```bash
gh auth login
gh repo create safe-demo-expts --public --source=. --remote=origin --push
# or: git remote add origin git@github.com:<you>/safe-demo-expts.git && git push -u origin main
```

Proposal / Heilmeier / Overleaf / course-paper sources are intentionally excluded (see `.gitignore`).
