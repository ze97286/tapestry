#!/bin/bash
#SBATCH --job-name=rltf_detect
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rltf_detect_%j.out
#SBATCH --error=logs/rltf_detect_%j.err
#
# Step 3 — read-level tabular detector. Submit from repo root, after the oracle
# gate passes. Required: MANIFEST, PANEL_DIR, OUT_DIR.
# Optional: BACKEND (tabicl|tabpfn|sklearn) DEVICE GROUP_COL REGRESSION_MIN_TF SEED.
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
: "${MANIFEST:?set MANIFEST}"
: "${PANEL_DIR:?set PANEL_DIR}"
: "${OUT_DIR:?set OUT_DIR}"

python "${RLTF_DIR}/scripts/run_detector.py" \
    --manifest "${MANIFEST}" --panel-dir "${PANEL_DIR}" --out-dir "${OUT_DIR}" \
    --backend "${BACKEND:-tabicl}" --device "${DEVICE:-cpu}" \
    --group-col "${GROUP_COL:-cohort}" --regression-min-tf "${REGRESSION_MIN_TF:-0.0}" \
    --seed "${SEED:-0}"
