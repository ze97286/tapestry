#!/bin/bash
#SBATCH --job-name=eval_lts
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_lts.out
#SBATCH --error=logs/eval_lts.err

# Least Trimmed Squares (Rousseeuw) deconvolution on the eval set.
# Sweeps trim_frac — the assumed fraction of atlas entries to treat as outliers.
# NNLS included as reference row.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"

python scripts/eval_lts.py \
    --data-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}" \
    --trim-fracs 0.05 0.10 0.15 0.20 0.30 \
    --n-starts 3
