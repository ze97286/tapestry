#!/bin/bash
#SBATCH --job-name=rltf_clinical
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rltf_clinical_%j.out
#SBATCH --error=logs/rltf_clinical_%j.err
#
# Step 4 — clinical evaluation. Submit from repo root, after the detector.
# Required: DETECTOR_DIR, CLINICAL_TSV, OUT_DIR. Optional: SPECIFICITY (default 0.95).
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
: "${DETECTOR_DIR:?set DETECTOR_DIR}"
: "${CLINICAL_TSV:?set CLINICAL_TSV}"
: "${OUT_DIR:?set OUT_DIR}"

python "${RLTF_DIR}/scripts/run_clinical_eval.py" \
    --detector-dir "${DETECTOR_DIR}" --clinical-tsv "${CLINICAL_TSV}" \
    --out-dir "${OUT_DIR}" --specificity "${SPECIFICITY:-0.95}"
