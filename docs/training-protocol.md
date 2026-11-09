# S1 Training Protocol — where things live

## Layout

```
/home/ubuntu/Documents/IRL/project/
├── configs/s1_safety_point.yaml          # hyperparameters
├── data/demos/s1_pointgoal1_*.npz      # 40-traj demo banks (low/mid/high)
├── results/
│   ├── s1_matrix.json                    # train+eval metrics
│   └── checkpoints/s1_*.pt               # BC weights
└── safe_demo_expts/
    ├── models/bc.py                      # MLP + MSE training
    ├── data/generate_safety_point_demos.py
    ├── shield/safety_point_shield.py
    ├── pipelines/evaluate_safety_point.py
    └── scripts/run_s1_safety_point.py    # A/B/C matrix runner
```

## Architecture

**Policy:** 2-layer MLP  
`obs (60) [+ prev_action (2) + prev_reject (1)] → 128 → ReLU → 128 → ReLU → action (2)`

- Action = `[forward, turn]` in `[-1, 1]`
- Optional history: previous *executed* action + whether the deploy shield rejected last step
- Code: `safe_demo_expts/models/bc.py` → `BCPolicy`

**Not used yet:** RL, AIRL, recurrence beyond 1-step history.

## Loss

**MSE** between predicted action and training labels (`torch.nn.MSELoss`).

Labels depend on pipeline mode (`filter_dataset`):

| Mode | Training labels | Transitions kept |
|---|---|---|
| **A** filter-first | expert proposed `a_prop` | only transitions the deploy shield would *accept* |
| **B** train-then-constrain | expert proposed `a_prop` | all |
| **C** projected labels | shield-projected `a_exec_deploy` | all |

Optimizer: Adam, `lr=1e-3`, `batch_size=256`, `epochs=60`.

## Trajectories used for learning

From `data/demos/s1_pointgoal1_{low,mid,high}_n40_seed0.npz`:

- **40 successful** scripted-expert episodes per illegal level
- Same deploy margin `0.14`; illegal level only changes the *demo* keep-out (`demo_margin`)
- Demo seeds: `seed + 1000 + traj_id` during collection

Illegal levels control how close the expert clips hazards vs the deploy shield.

## Train / val / test — what we do and don’t do

| Split | What |
|---|---|
| **Train** | ~80% of *trajectories* (by `traj_id`) after mode filtering |
| **Val** | held-out ~20% of trajectories — **supervised MSE only** (does not drive early stopping yet) |
| **Test / real metric** | **not** a held-out demo set |

There is **no offline test MSE that defines success**. The real evaluation is **online closed-loop rollouts** under the deploy shield (see below).

Why: BC MSE on demos can look fine while the policy still stalls or fails to reach the goal once the shield edits actions.

## Online inference / “new setup”

Yes — eval uses **fresh environment layouts**, not demo start states.

For each of `n_eval=20` episodes:

1. `env.reset(seed = (train_seed + 7) + 10_000 + ep)` → new agent / goal / hazard placement
2. Policy proposes `a_prop` from current obs (+ history if enabled)
3. **Deploy shield** may reject and project → `a_exec`
4. Environment steps with `a_exec`
5. Repeat until success, timeout (`max_steps=500`), or **stuck**

Stuck = goal-distance progress range over the last `stuck_horizon=40` steps `< 1e-3`.

Code: `pipelines/evaluate_safety_point.py`.

## What defines success?

Per episode:

- **`success`** = first goal reached (`info["goal_met"]` / `task.goal_achieved`), then episode stops (`stop_on_first_goal=True`)

Aggregate metrics (reported in `results/s1_matrix.json`):

| Metric | Meaning |
|---|---|
| `success_rate` | fraction of eval episodes that hit the goal |
| `stuck_rate` | fraction terminated as shield-induced stalls |
| `mean_reject_rate` | fraction of steps where deploy shield intervened |
| `mean_episode_cost` | Safety-Gymnasium hazard cost accumulated |
| `train_mse` / `val_mse` | supervised fit quality (secondary) |

**Primary outcome for the research question:** compare A vs B vs C on `success_rate` and `stuck_rate` across illegal levels (and history on/off).

## Run command

```bash
cd /home/ubuntu/Documents/IRL/project
. .venv-safety/bin/activate
MUJOCO_GL=egl PYTHONPATH=. python safe_demo_expts/scripts/run_s1_safety_point.py \
  --config configs/s1_safety_point.yaml \
  --demo-dir data/demos \
  --out results/s1_matrix.json
```
