#!/bin/bash
#SBATCH --job-name=rltf_disc_merge
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rltf_disc_merge_%j.out
#SBATCH --error=logs/rltf_disc_merge_%j.err
#
# Step 1 (merge) — combine the per-chromosome partials into the final top-N panel
# (runs/panel/). Run after discover_shard.sh completes:
#   sbatch --dependency=afterok:<arrayJobID> readlevel_tf/slurm/discover_merge.sh
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
CONFIG="${CONFIG:-readlevel_tf/configs/config.toml}"
python "${RLTF_DIR}/scripts/discover_panel.py" --config "${CONFIG}" --merge
