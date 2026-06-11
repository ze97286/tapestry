#!/bin/bash
# Sourced by every rltf SLURM job. Self-contained: resolves the project dir from
# its own location, sets PYTHONPATH, loads the compiler module, and activates the
# rltf venv if present. Submit jobs from the tapestry repo root.
RLTF_DIR="${RLTF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RLTF_DIR
export PYTHONPATH="${RLTF_DIR}:${PYTHONPATH:-}"
export PYTHONHASHSEED=42

module load GCC/12.3.0 2>/dev/null || true

RLTF_VENV="${RLTF_VENV:-${RLTF_DIR}/.venv}"
export RLTF_VENV
if [ -f "${RLTF_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${RLTF_VENV}/bin/activate"
else
    echo "WARNING: ${RLTF_VENV} not found; run readlevel_tf/slurm/setup_env.sh first." >&2
fi
