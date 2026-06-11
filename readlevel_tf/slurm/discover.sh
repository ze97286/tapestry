#!/bin/bash
#SBATCH --job-name=rltf_discover
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rltf_discover_%j.out
#SBATCH --error=logs/rltf_discover_%j.err
#
# Step 1 — discover the panel from reference PATs. Submit from repo root.
# Required: MANIFEST, OUT_DIR. Optional: WINDOW TOP_N MIN_TOTAL MIN_EFFECT DIRECTION.
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
: "${MANIFEST:?set MANIFEST}"
: "${OUT_DIR:?set OUT_DIR}"

python "${RLTF_DIR}/scripts/discover_panel.py" \
    --manifest "${MANIFEST}" --out-dir "${OUT_DIR}" \
    --window "${WINDOW:-5}" --top-n "${TOP_N:-2000}" \
    --min-total "${MIN_TOTAL:-10}" --min-effect "${MIN_EFFECT:-0.3}" \
    --direction "${DIRECTION:-any}"
