# MPC escape harden — multi-seed results

Config: `configs/s1_mpc_harden.yaml` · JSON: `results/s1_mpc_harden.json` · plot: `results/figures/s1_mpc_harden.png`  
Seeds: `{7, 77, 177}`, 20 eval eps each. Frozen mid-illegal BC policies.

## Shield family (stuck / success, mean±std)

| Policy | Freeze | Escape | MPC H=3 N=20 |
|---|---|---|---|
| A_nohist | 0.30±0.08 / 0.68±0.06 | 0.05±0.04 / 0.93±0.05 | **0.03±0.02 / 0.93±0.02** |
| C_nohist | 0.32±0.12 / 0.53±0.10 | **0.07±0.05 / 0.78±0.12** | 0.08±0.02 / 0.75±0.11 |
| C_hist | 0.35±0.07 / 0.63±0.08 | 0.13±0.02 / 0.85±0.04 | **0.05±0.04 / 0.93±0.05** |

**Takeaway:** freeze ≫ stuck; escape fixes most of it; MPC helps further on A and especially **C_hist** (escape 0.13 → MPC 0.05). On C_nohist, escape ≈ MPC at H=3 N=20; **N=40** drops stuck to **0.03±0.02**.

## MPC horizon @ N=20 (stuck)

| H | A_nohist | C_nohist | C_hist |
|---|---|---|---|
| 1 | 0.05±0.04 | 0.08±0.02 | 0.07±0.02 |
| 2 | **0.00±0.00** | 0.08±0.02 | 0.05±0.00 |
| 3 | 0.03±0.02 | 0.08±0.02 | 0.05±0.04 |
| 5 | **0.00±0.00** | 0.08±0.02 | **0.02±0.02** |

Default recommendation: **H=2 or H=5** (A/C_hist); C_nohist needs more samples more than longer H.

## MPC samples @ H=3 (stuck)

| N | A_nohist | C_nohist | C_hist |
|---|---|---|---|
| 8 | **0.00±0.00** | 0.08±0.02 | 0.03±0.02 |
| 20 | 0.03±0.02 | 0.08±0.02 | 0.05±0.04 |
| 40 | 0.02±0.02 | **0.03±0.02** | 0.07±0.02 |

## Locked defaults for the paper spine

- **Negative control:** freeze (`escape_mode=false`)  
- **Strong baseline:** escape (`escape_mode=true`)  
- **Proposed:** MPC with **H=2–5**, **N=20** (use **N=40** if emphasizing C_nohist)
