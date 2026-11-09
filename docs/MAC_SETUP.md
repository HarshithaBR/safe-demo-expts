# Mac Pro setup (`~/Documents/IRL/project`)

This repo was scaffolded on the Cursor cloud Linux VM at the same relative path. On your Mac Pro, keep the project at:

```text
~/Documents/IRL/project
```

Copy/sync this folder from the cloud agent machine or from the Project store export, then:

## 1. Python env

```bash
# Apple Silicon or Intel — use conda/mamba if you already do for robotics
conda create -n safe-demo python=3.11 -y
conda activate safe-demo
cd ~/Documents/IRL/project
pip install -r requirements.txt
```

For Apple Silicon PyTorch, prefer the official Mac wheels:

```bash
pip install torch torchvision
```

## 2. Run the S0 diagnostic (no MuJoCo required)

```bash
cd ~/Documents/IRL/project
PYTHONPATH=. python safe_demo_expts/scripts/run_s0_matrix.py --smoke
PYTHONPATH=. python safe_demo_expts/scripts/run_s0_matrix.py --out results/s0_matrix_mac.json
```

## 3. Optional baselines used in related work

### Safety-Gymnasium (nav)

```bash
pip install safety-gymnasium
python -c "import gymnasium as gym; import safety_gymnasium; env=gym.make('SafetyPointGoal1-v0'); print(env.reset())"
```

Next step (not yet wrapped in this pack): write `safe_demo_expts/envs/safety_gym_wrapper.py` that exposes the same `(obs, a_prop, a_exec, reject)` logging as S0.

### F1TENTH Gym (racing / constraint-aware IL neighbor)

```bash
# follow https://github.com/f1tenth/f1tenth_gym README for your OS
git clone https://github.com/f1tenth/f1tenth_gym.git
# build per upstream docs, then add a wrapper later
```

### robosuite + robomimic (tabletop IL baseline)

Follow current ARISE docs (versions matter):

- https://github.com/ARISE-Initiative/robosuite  
- https://github.com/ARISE-Initiative/robomimic  

Suggested first task: **Lift** low-dim PH demos, then add a table/workspace shield wrapper analogous to `ObstacleShield`.

```bash
# illustrative — check current robomimic docs for exact pins
pip install robosuite
pip install -e /path/to/robomimic
python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph --hdf5_types low_dim
```

## 4. Sync with cloud results

Cloud smoke/full JSON lives under this same `results/` folder on the VM. Compare Mac vs cloud with identical `configs/s0_pointnav.yaml` seeds.

## 5. GPU note

Mac Pro GPU (Metal) helps PyTorch; Safety-Gymnasium/MuJoCo are often CPU-bound for Point/Car. Prefer CPU for S0 apples-to-apples with the cloud VM.
