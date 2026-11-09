# Project restore (post wipe)

Rebuilt **2026-11-09** from S0 bundle + session docs/results + rewritten S1 stack.

## Runtime

```bash
source scripts/env_safe311.sh   # micromamba env `safe311` (Py3.11 + safety-gymnasium)
# If SG dataclass errors after reinstall:
python scripts/patch_safety_gymnasium_dataclasses.py
```

- Nav: `MUJOCO_GL=glfw` (egl needs system libEGL)
- Tabletop: separate `.venv-tabletop` + metaworld (not installed on this VM yet)

## Regenerated artifacts

| Artifact | Status |
|---|---|
| `data/demos/s1_pointgoal1_mid_n40_seed0.npz` | regenerated (illegal_rate≈0.26) |
| `results/checkpoints/s1_mid_modeA_hist0_seed0.pt` | regenerated |
| `results/checkpoints/s1_mid_modeC_hist1_seed0.pt` | regenerated |
| Prior result JSONs under `results/` / store `docs/` | preserved (historical) |
| Smoke rebuilds `*_smoke_rebuild.json` | pass qualitatively (A2≫freeze; A1b hurts; A1c≈freeze) |

## Not bit-identical

Shields, oracle, and BC training were re-authored from docs/configs. Numbers will drift vs Nov 5–8 JSONs until the full paper queue is re-run.

```bash
PYTHON=/home/ubuntu/micromamba/envs/safe311/bin/python \
  nohup bash scripts/run_all_paper_queue.sh >> results/queue/queue.nohup.out 2>&1 &
```
