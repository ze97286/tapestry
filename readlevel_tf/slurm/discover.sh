#!/bin/bash
#SBATCH --job-name=rltf_discover
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rltf_discover_%j.out
#SBATCH --error=logs/rltf_discover_%j.err
#
# Step 1 (single-job) — discover the panel genome-wide. Config-driven.
# Memory-heavy single-threaded streaming of the reference per-read calls; for a
# faster, lower-risk run use the sharded path instead:
#   sbatch readlevel_tf/slurm/discover_shard.sh                       # array 1-22, tabix per chrom
#   sbatch --dependency=afterok:<arrayJobID> readlevel_tf/slurm/discover_merge.sh
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
CONFIG="${CONFIG:-readlevel_tf/configs/config.toml}"
python "${RLTF_DIR}/scripts/discover_panel.py" --config "${CONFIG}"
