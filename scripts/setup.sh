#!/usr/bin/env bash
# Idempotent-ish bootstrap for SAFE demo experiments (S0/S1).
# Creates a Python 3.11 env (micromamba preferred, else conda, else venv),
# installs requirements + Safety-Gymnasium stack, applies SG dataclass patch,
# and prints MUJOCO_GL / tabletop notes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${SAFE_ENV_NAME:-safe311}"
PYTHON_VERSION="${SAFE_PYTHON_VERSION:-3.11}"
MUJOCO_GL_DEFAULT="${MUJOCO_GL:-glfw}"

log() { printf '[setup] %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

find_micromamba() {
  if [[ -n "${MICROMAMBA_BIN:-}" && -x "${MICROMAMBA_BIN}" ]]; then
    echo "$MICROMAMBA_BIN"; return
  fi
  for c in "$HOME/bin/micromamba" /tmp/bin/micromamba /usr/local/bin/micromamba micromamba; do
    if [[ -x "$c" ]] || have "$c"; then
      if [[ -x "$c" ]]; then echo "$c"; else command -v "$c"; fi
      return
    fi
  done
  return 1
}

ensure_mamba_env() {
  local mm
  mm="$(find_micromamba)" || return 1
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
  mkdir -p "$MAMBA_ROOT_PREFIX"
  log "using micromamba: $mm (root=$MAMBA_ROOT_PREFIX)"
  # shellcheck disable=SC1091
  eval "$("$mm" shell hook -s bash)"
  if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "creating env $ENV_NAME (python=$PYTHON_VERSION)"
    micromamba create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
  else
    log "env $ENV_NAME already exists"
  fi
  micromamba activate "$ENV_NAME"
  PYTHON_BIN="$(command -v python)"
  PIP_BIN="$(command -v pip)"
}

ensure_conda_env() {
  have conda || return 1
  log "using conda"
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "creating conda env $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
  else
    log "conda env $ENV_NAME already exists"
  fi
  conda activate "$ENV_NAME"
  PYTHON_BIN="$(command -v python)"
  PIP_BIN="$(command -v pip)"
}

ensure_venv() {
  local venv_dir="$ROOT/.venv-safe311"
  log "falling back to venv at $venv_dir"
  if ! have "python${PYTHON_VERSION}" && ! have python3; then
    log "ERROR: need python${PYTHON_VERSION} or python3 on PATH"
    exit 1
  fi
  local py
  if have "python${PYTHON_VERSION}"; then
    py="python${PYTHON_VERSION}"
  else
    py=python3
  fi
  if [[ ! -d "$venv_dir" ]]; then
    "$py" -m venv "$venv_dir"
  fi
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
  PYTHON_BIN="$(command -v python)"
  PIP_BIN="$(command -v pip)"
  local ver
  ver="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [[ "$ver" != "$PYTHON_VERSION" ]]; then
    log "WARNING: venv Python is $ver (wanted $PYTHON_VERSION). Safety-Gymnasium may need 3.11."
  fi
}

bootstrap_env() {
  if ensure_mamba_env; then return; fi
  if ensure_conda_env; then return; fi
  ensure_venv
}

install_python_deps() {
  log "python: $("$PYTHON_BIN" -V) ($PYTHON_BIN)"
  "$PIP_BIN" install --upgrade pip wheel setuptools
  log "installing requirements.txt"
  "$PIP_BIN" install -r "$ROOT/requirements.txt"

  # Pinned Safety-Gymnasium / MuJoCo stack used on the cloud VM (safe311).
  # Override with SKIP_SAFETY_GYM=1 for S0-only installs.
  if [[ "${SKIP_SAFETY_GYM:-0}" == "1" ]]; then
    log "SKIP_SAFETY_GYM=1 — skipping safety-gymnasium stack"
    return
  fi
  log "installing Safety-Gymnasium stack (pinned)"
  "$PIP_BIN" install \
    "numpy==1.23.5" \
    "gymnasium==0.28.1" \
    "mujoco==2.3.3" \
    "safety-gymnasium==1.0.0" \
    "gymnasium-robotics==1.2.2" \
    "imageio>=2.31" \
    "pillow>=9"
}

apply_sg_patch() {
  local patch="$ROOT/scripts/patch_safety_gymnasium_dataclasses.py"
  if [[ ! -f "$patch" ]]; then
    log "no SG dataclass patch script present — skip"
    return
  fi
  if ! "$PYTHON_BIN" -c "import safety_gymnasium" >/dev/null 2>&1; then
    log "safety_gymnasium not importable — skip patch"
    return
  fi
  log "applying scripts/patch_safety_gymnasium_dataclasses.py"
  "$PYTHON_BIN" "$patch" || log "WARNING: SG patch returned non-zero (may already be patched)"
}

smoke_check() {
  export PYTHONPATH="${PYTHONPATH:-}:$ROOT"
  export MUJOCO_GL="${MUJOCO_GL:-$MUJOCO_GL_DEFAULT}"
  log "smoke import (PYTHONPATH + MUJOCO_GL=$MUJOCO_GL)"
  "$PYTHON_BIN" - <<'PY' || log "WARNING: smoke import failed (ok if SG/Mujoco GL missing on this host)"
import os
print("MUJOCO_GL=", os.environ.get("MUJOCO_GL"))
import numpy, torch, gymnasium, yaml
print("core ok:", numpy.__version__, torch.__version__, gymnasium.__version__)
try:
    import safety_gymnasium  # noqa: F401
    print("safety_gymnasium ok")
except Exception as e:
    print("safety_gymnasium import skipped/failed:", e)
PY
}

print_notes() {
  cat <<EOF

[setup] done.

Activate / run (micromamba example):
  export MAMBA_ROOT_PREFIX=\${MAMBA_ROOT_PREFIX:-\$HOME/micromamba}
  eval "\$(micromamba shell hook -s bash)"
  micromamba activate ${ENV_NAME}
  cd ${ROOT}
  export PYTHONPATH=. MUJOCO_GL=${MUJOCO_GL_DEFAULT}

Or:  source scripts/env_safe311.sh

MUJOCO_GL:
  - glfw  — desktop / cloud VMs with a display (default here)
  - egl   — headless servers with libEGL (export MUJOCO_GL=egl)

Optional tabletop / Meta-World:
  Not installed by this script. Create a separate env (e.g. .venv-tabletop),
  then follow docs/MAC_SETUP.md and docs/tabletop.md (robosuite/robomimic or metaworld).

S0 smoke (no MuJoCo required):
  PYTHONPATH=. python -m safe_demo_expts.scripts.run_s0_matrix --smoke

See README.md for S1 / interventions / bakeoff / A1c commands.
EOF
}

bootstrap_env
install_python_deps
apply_sg_patch
smoke_check
print_notes
