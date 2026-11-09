#!/usr/bin/env bash
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/ubuntu/micromamba}"
# shellcheck disable=SC1091
eval "$(${MICROMAMBA_BIN:-/tmp/bin/micromamba} shell hook -s bash)"
micromamba activate safe311
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
