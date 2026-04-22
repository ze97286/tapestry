#!/bin/bash
#SBATCH --job-name=eval_binomial
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_binomial.out
#SBATCH --error=logs/eval_binomial.err

# Binomial-likelihood MLE deconvolution on the eval set.
# Correct observation model for count data — no Gaussian approximation.
# NNLS reported alongside for direct comparison.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"

python scripts/eval_binomial.py \
    --data-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}" \
    --per-cell-type
