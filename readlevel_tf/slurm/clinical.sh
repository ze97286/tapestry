#!/bin/bash
#SBATCH --job-name=rltf_clinical
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rltf_clinical_%j.out
#SBATCH --error=logs/rltf_clinical_%j.err
#
# Step 4 — clinical evaluation. Config-driven. Submit from repo root, after the detector.
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
CONFIG="${CONFIG:-readlevel_tf/configs/config.toml}"
python "${RLTF_DIR}/scripts/run_clinical_eval.py" --config "${CONFIG}"
