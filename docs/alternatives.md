# Alternatives to IL + DAgger (tried)

Focused mid-illegal probe on A_nohist / C_nohist / C_hist. Primary DV: `terminal_stuck_rate`.  
Results: `results/s1_alternatives.json` · plot: `results/figures/s1_alternatives.png`

## What we ran

| # | Method | Implementation |
|---|---|---|
| **0** | Frozen BC autonomous | Escape-mode deploy shield (no oracle) |
| **3** | Eval-time V3b assist | Existing intervention stack, budget 40 |
| **1** | Frozen BC + recovery policy | BC on oracle stuck labels; switch 20 steps when stuck |
| **2** | Residual head | Freeze BC; learn `Δa`; always on |
| **4** | Escape-projected C | Fresh demos; BC on escape-shield `a_exec` |
| **5** | TD3+BC offline | 4k updates on demo transitions (proxy rewards) |
| **6** | Shielded PPO | 15k env steps, actions always shielded |
| **7** | MPC escape shield | H=3 kinematic clearance search + frozen BC |

## Results (succ / stuck)

| Method | A_nohist | C_nohist | C_hist |
|---|---|---|---|
| 0 BC auto | **0.90 / 0.10** | **0.95 / 0.00** | **0.90 / 0.10** |
| 3 Assist V3b | 0.75 / 0.25 | 0.55 / 0.40 | 0.80 / 0.20 |
| 1 Recovery | 0.80 / 0.20 | 0.60 / 0.35 | 0.85 / 0.15 |
| 2 Residual | 0.20 / 0.55 | 0.30 / 0.45 | 0.20 / 0.50 |
| 4 Escape-C | 0.85 / 0.10 | 0.85 / 0.10 | 0.85 / 0.15 |
| 5 TD3+BC | 0.15 / 0.55 | 0.15 / 0.55 | 0.15 / 0.55 |
| 6 PPO | 0.05 / 0.35 | 0.05 / 0.35 | 0.05 / 0.35 |
| **7 MPC** | **0.95 / 0.05** | **0.90 / 0.05** | **0.90 / 0.10** |

## Interpretation

1. **Escape-mode shield alone** already lifts frozen BC a lot vs the old freeze-shield era. That changes the baseline.
2. **Best learning-free add-on: MPC escape (7)** — lowest stuck on A/C_nohist.
3. **Assist (3) / recovery (1)** can *hurt* when the base is already strong (early stuck triggers / takeover noise). They mattered more under the freeze shield.
4. **Residual (2), TD3+BC (5), short PPO (6)** failed here — residual overwrites like bad DAgger; offline reward proxy is weak; 15k PPO steps are far too few.
5. **Escape-projected C (4)** is competitive with BC auto, not a breakthrough over MPC.

## What to keep pursuing

| Keep | Drop / defer |
|---|---|
| Escape shield + **MPC escape (7)** | Naive residual FT |
| Assist/recovery **only when base is weak** (budgeted) | Short PPO as currently configured |
| Stronger offline RL later (real rewards, more data) | Claiming DAgger/residual transfer |

## Code

- `safe_demo_expts/scripts/run_s1_alternatives.py`
- `safe_demo_expts/shield/mpc_escape.py`
- `safe_demo_expts/models/{residual,td3_bc,shielded_ppo}.py`
- `safe_demo_expts/pipelines/evaluate_flexible.py`
