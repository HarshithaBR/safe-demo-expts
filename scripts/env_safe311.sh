#!/usr/bin/env bash
# Activate the Safety-Gymnasium Python 3.11 env used for S1 nav.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/ubuntu/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/bin/micromamba}"
if [[ ! -x "$MICROMAMBA_BIN" ]]; then MICROMAMBA_BIN=/tmp/bin/micromamba; fi
eval "$("$MICROMAMBA_BIN" shell hook -s bash)"
micromamba activate safe311
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
echo "safe311 ready: $(python -V) MUJOCO_GL=$MUJOCO_GL"
