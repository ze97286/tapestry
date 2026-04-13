#!/bin/bash
#SBATCH --job-name=vis_atlas
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/visualise_atlas.out
#SBATCH --error=logs/visualise_atlas.err

source slurm/common.sh

PLOTS_DIR="${OUTPUT_DIR}/markers/plots"
mkdir -p "${PLOTS_DIR}"

python scripts/visualise_atlas.py \
    --markers "${OUTPUT_DIR}/markers/markers.tsv" \
    --output-dir "${PLOTS_DIR}"
