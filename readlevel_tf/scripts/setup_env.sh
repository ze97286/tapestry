#!/bin/bash
# Build the standalone rltf venv. Deps: numpy/pandas/scipy/scikit-learn/plotly
# (always) + torch/tabicl for the detector head. No h5py, no tapestry — standalone.
# pip + the TabICL checkpoint download need internet; run on a login node if
# compute nodes are offline.
#
# Env: RLTF_VENV (default ${RLTF_DIR}/.venv), WITH_TORCH=1, TORCH_INDEX_URL,
#      PREFETCH_TABICL=1, RLTF_VENV_PYTHON=python3.
#
# HPC note: pip and HuggingFace caches live under ~/.cache, which on a quota'd
# home will fill up (the TabICL checkpoint download in particular). Symlink
# ~/.cache (or ~/.cache/pip and ~/.cache/huggingface) onto scratch once, before
# running this — see RUNBOOK.md step 0.

set -euo pipefail

RLTF_DIR="${RLTF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RLTF_VENV="${RLTF_VENV:-${RLTF_DIR}/.venv}"
PYTHON_BIN="${RLTF_VENV_PYTHON:-python3}"
WITH_TORCH="${WITH_TORCH:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PREFETCH_TABICL="${PREFETCH_TABICL:-1}"

if [ ! -d "${RLTF_VENV}" ]; then
    echo "Creating venv: ${RLTF_VENV}"
    "${PYTHON_BIN}" -m venv "${RLTF_VENV}"
fi
# shellcheck disable=SC1091
source "${RLTF_VENV}/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install "numpy>=1.26" "pandas>=2.1" "scipy>=1.12" "scikit-learn>=1.4" "plotly>=5.18" "pysam>=0.22"

if [ "${WITH_TORCH}" = "1" ]; then
    python -m pip install torch --index-url "${TORCH_INDEX_URL}"
    python -m pip install tabicl
    if [ "${PREFETCH_TABICL}" = "1" ]; then
        # Non-fatal: if the prefetch fails (e.g. offline), the detector will
        # fetch the checkpoint at runtime (into ~/.cache/huggingface).
        python - <<'PY' || echo "WARN: TabICL prefetch failed; detector will fetch at runtime."
import numpy as np
from tabicl import TabICLClassifier
TabICLClassifier().fit(np.random.RandomState(0).normal(size=(20, 4)), [0, 1] * 10)
print("TabICL checkpoint cached")
PY
    fi
fi

PYTHONPATH="${RLTF_DIR}:${PYTHONPATH:-}" python -c "import rltf; print('rltf env OK')"
echo "Done: ${RLTF_VENV}"
