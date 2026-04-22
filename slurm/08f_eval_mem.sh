#!/bin/bash
#SBATCH --job-name=eval_mem
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_mem.out
#SBATCH --error=logs/eval_mem.err

# Maximum Entropy Method (Gull-Skilling / Sibisi) deconvolution on the eval set.
# Sweeps lambda_reg (data/entropy tradeoff). NNLS included as reference row.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"

python scripts/eval_mem.py \
    --data-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}" \
    --lambdas 0.01 0.1 1.0 10.0 100.0
