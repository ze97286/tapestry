#!/bin/bash
#SBATCH --job-name=rltf_monitor
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=28:00:00
#SBATCH --output=logs/rltf_monitor_%j.out
#SBATCH --error=logs/rltf_monitor_%j.err
#
# Step 5 — longitudinal monitoring. Config-driven. Submit from repo root, after the
# detector (step 3) has produced runs/detector/{features.tsv, classification_oof.tsv}.
# Scores the on-treatment (followup) timepoints out-of-sample and relates the per-patient
# score trajectory to survival.
#   sbatch readlevel_tf/slurm/monitor.sh
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
CONFIG="${CONFIG:-readlevel_tf/configs/config.toml}"
python "${RLTF_DIR}/scripts/run_monitor.py" --config "${CONFIG}"
