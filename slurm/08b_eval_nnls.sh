#!/bin/bash
#SBATCH --job-name=eval_nnls
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_nnls.out
#SBATCH --error=logs/eval_nnls.err

# Coverage-weighted NNLS baseline on the tapestry eval set.
# Use the same atlas + eval data as tapestry training to get a fair comparison.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"

python scripts/eval_nnls.py \
    --data-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}"
