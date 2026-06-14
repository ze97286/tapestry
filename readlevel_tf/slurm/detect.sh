#!/bin/bash
#SBATCH --job-name=rltf_detect
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=28:00:00
#SBATCH --output=logs/rltf_detect_%j.out
#SBATCH --error=logs/rltf_detect_%j.err
#
# Step 3 — read-level tabular detector. Config-driven. Submit from repo root,
# after the oracle gate passes.
#   Full run (scores all query samples):        sbatch readlevel_tf/slurm/detect.sh
#   Re-run CV only on saved features.tsv:       sbatch --export=ALL,FROM_FEATURES=1 readlevel_tf/slurm/detect.sh
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
CONFIG="${CONFIG:-readlevel_tf/configs/config.toml}"
EXTRA=()
[ -n "${FROM_FEATURES:-}" ] && EXTRA+=(--from-features)
python "${RLTF_DIR}/scripts/run_detector.py" --config "${CONFIG}" "${EXTRA[@]}"
