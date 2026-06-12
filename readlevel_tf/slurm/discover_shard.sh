#!/bin/bash
#SBATCH --job-name=rltf_disc_shard
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/rltf_disc_shard_%A_%a.out
#SBATCH --error=logs/rltf_disc_shard_%A_%a.err
#
# Step 1 (sharded) — discover one chromosome per array task via tabix region
# reads (~1/22 of each file → fast, low memory). Writes runs/panel/partials/chrN/.
# Then merge with discover_merge.sh. Submit from the repo root:
#   sbatch readlevel_tf/slurm/discover_shard.sh
#   sbatch --dependency=afterok:<arrayJobID> readlevel_tf/slurm/discover_merge.sh
# Override the chromosome set with e.g. --array=1-22,X.
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
CONFIG="${CONFIG:-readlevel_tf/configs/config.toml}"
CHR="chr${SLURM_ARRAY_TASK_ID:?run as a SLURM array, e.g. --array=1-22}"
python "${RLTF_DIR}/scripts/discover_panel.py" --config "${CONFIG}" --chrom "${CHR}"
