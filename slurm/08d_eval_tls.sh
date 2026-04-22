#!/bin/bash
#SBATCH --job-name=eval_tls
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_tls.out
#SBATCH --error=logs/eval_tls.err

# Scaled Total Least Squares baseline on the tapestry eval set.
# Accounts for both observation noise (binomial via coverage) and atlas noise.
# Runs NNLS alongside for a direct side-by-side in the same log.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"

python scripts/eval_tls.py \
    --data-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}" \
    --atlas-sigma 0.05 \
    --also-nnls
