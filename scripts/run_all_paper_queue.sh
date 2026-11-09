#!/usr/bin/env bash
# Sequential paper experiment queue (7 steps). Skip steps whose output already exists
# unless FORCE=1. Designed to be nohup'd:
#   nohup bash scripts/run_all_paper_queue.sh > results/queue/queue.log 2>&1 &
#
# Env overrides:
#   FORCE=1          redo even if outputs exist
#   SMOKE=1          fast/smoke modes where supported
#   START_FROM=N     start at step N (1..7)
#   STOP_AFTER=N     stop after step N
#   PYTHON=python    interpreter

set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/ubuntu/micromamba/envs/safe311/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
FORCE="${FORCE:-0}"
SMOKE="${SMOKE:-0}"
START_FROM="${START_FROM:-1}"
STOP_AFTER="${STOP_AFTER:-7}"

QUEUE_DIR="$ROOT/results/queue"
mkdir -p "$QUEUE_DIR" results/checkpoints results/gated_recovery results/query_escape_gate

MASTER_LOG="$QUEUE_DIR/queue.log"
STATUS="$QUEUE_DIR/status.json"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

write_status() {
  local step="$1" state="$2" detail="${3:-}"
  "$PYTHON" - "$STATUS" "$step" "$state" "$detail" <<'PY' || true
import json, sys, datetime
path, step, state, detail = sys.argv[1:5]
try:
    data = json.load(open(path))
except Exception:
    data = {"steps": {}}
data["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
data["current_step"] = int(step)
data["steps"][str(step)] = {"state": state, "detail": detail, "at": data["updated_at"]}
json.dump(data, open(path, "w"), indent=2)
PY
}

need() {
  # need OUTFILE → 0 if should run, 1 if skip
  local out="$1"
  if [[ "$FORCE" == "1" ]]; then return 0; fi
  if [[ -f "$out" && -s "$out" ]]; then
    # treat empty methods {} as incomplete for some runners
    if grep -q '"methods": {}' "$out" 2>/dev/null; then return 0; fi
    return 1
  fi
  return 0
}

run_step() {
  local n="$1" name="$2" outfile="$3"
  shift 3
  if (( n < START_FROM || n > STOP_AFTER )); then
    log "SKIP step $n ($name) — outside START_FROM/STOP_AFTER"
    return 0
  fi
  if ! need "$outfile"; then
    log "SKIP step $n ($name) — exists: $outfile"
    write_status "$n" "skipped_exists" "$outfile"
    return 0
  fi
  local lock="$QUEUE_DIR/step${n}.lock"
  exec {lock_fd}>"$lock"
  if ! flock -n "$lock_fd"; then
    log "WAIT  step $n ($name) — another process holds $lock; waiting"
    flock "$lock_fd"
  fi
  # re-check after lock (another runner may have finished)
  if ! need "$outfile"; then
    log "SKIP step $n ($name) — exists after lock: $outfile"
    write_status "$n" "skipped_exists" "$outfile"
    exec {lock_fd}>&-
    return 0
  fi
  log "START step $n: $name"
  write_status "$n" "running" "$*"
  local step_log="$QUEUE_DIR/step${n}_${name}.log"
  local rc=0
  if "$@" >"$step_log" 2>&1; then
    log "DONE  step $n: $name → $outfile"
    write_status "$n" "done" "$outfile"
  else
    rc=$?
    log "FAIL  step $n: $name (exit $rc). See $step_log"
    write_status "$n" "failed" "exit=$rc log=$step_log"
  fi
  exec {lock_fd}>&-
  return $rc
}

SMOKE_FLAG=()
if [[ "$SMOKE" == "1" ]]; then SMOKE_FLAG=(--smoke); fi

log "==== paper queue begin ROOT=$ROOT FORCE=$FORCE SMOKE=$SMOKE START_FROM=$START_FROM STOP_AFTER=$STOP_AFTER ===="

# 1) A1c gated residual — C_hist
run_step 1 a1c_C_hist results/s1_gated_recovery_C_hist.json \
  "$PYTHON" -m safe_demo_expts.scripts.run_s1_gated_recovery \
    --policies C_hist \
    --out results/s1_gated_recovery_C_hist.json \
    "${SMOKE_FLAG[@]}"

# 2) Query-vs-escape — C_hist
run_step 2 query_escape_C_hist results/s1_query_escape_gate_C_hist.json \
  "$PYTHON" -m safe_demo_expts.scripts.run_s1_query_escape_gate \
    --policies C_hist \
    --out results/s1_query_escape_gate_C_hist.json \
    "${SMOKE_FLAG[@]}"

# 3) Matched bake-off — C_hist
run_step 3 bakeoff_C_hist results/s1_paper_bakeoff_C_hist.json \
  "$PYTHON" -m safe_demo_expts.scripts.run_s1_paper_bakeoff \
    --policies C_hist \
    --out results/s1_paper_bakeoff_C_hist.json \
    "${SMOKE_FLAG[@]}"

# 4) A1c hist ablation — C_nohist
run_step 4 a1c_C_nohist results/s1_gated_recovery_C_nohist.json \
  "$PYTHON" -m safe_demo_expts.scripts.run_s1_gated_recovery \
    --policies C_nohist \
    --out results/s1_gated_recovery_C_nohist.json \
    "${SMOKE_FLAG[@]}"

# 5) Geometry suite E3 — C_hist stratified layouts
GEOM_OUT=results/s1_geometry_suite.json
if [[ "$SMOKE" == "1" ]]; then GEOM_OUT=results/s1_geometry_suite_smoke.json; fi
run_step 5 geometry_suite "$GEOM_OUT" \
  "$PYTHON" -m safe_demo_expts.scripts.run_s1_geometry_suite \
    --policy-tag C_hist \
    --out results/s1_geometry_suite.json \
    "${SMOKE_FLAG[@]}"

# 6) Thin ablations — MPC H/N harden + A1 budget curves (one step, two artifacts)
run_step 6 ablations results/s1_ablations_done.flag \
  bash -c '
    set -e
    '"$PYTHON"' -m safe_demo_expts.scripts.run_s1_mpc_harden --out results/s1_mpc_harden.json
    '"$PYTHON"' -m safe_demo_expts.scripts.run_s1_budget_curves --out results/s1_budget_curves.json
    date -u +"%Y-%m-%dT%H:%M:%SZ" > results/s1_ablations_done.flag
  '

# 7) Tabletop transfer — Sawyer illegal-demo harden
run_step 7 tabletop_harden results/s1_tabletop_harden.json \
  "$PYTHON" -m safe_demo_expts.scripts.run_s1_tabletop_harden \
    --out results/s1_tabletop_harden.json

log "==== paper queue finished ===="
write_status 0 "queue_complete" "all requested steps finished"
echo "OK" > "$QUEUE_DIR/COMPLETE"
