# Paper experiment queue

```bash
# full 7-step queue in background (skips steps whose outputs already exist)
mkdir -p results/queue
nohup bash scripts/run_all_paper_queue.sh >> results/queue/queue.nohup.out 2>&1 &

# smoke (faster)
SMOKE=1 nohup bash scripts/run_all_paper_queue.sh >> results/queue/queue.nohup.out 2>&1 &

# resume from step 5
START_FROM=5 nohup bash scripts/run_all_paper_queue.sh >> results/queue/queue.nohup.out 2>&1 &

# force redo everything
FORCE=1 nohup bash scripts/run_all_paper_queue.sh >> results/queue/queue.nohup.out 2>&1 &
```

Monitor: `tail -f results/queue/queue.log` · status: `cat results/queue/status.json` · done: `ls results/queue/COMPLETE`

Steps: 1 A1c C_hist · 2 query-escape C_hist · 3 bake-off C_hist · 4 A1c C_nohist · 5 geometry E3 · 6 MPC+budget ablations · 7 tabletop harden
