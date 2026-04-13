#!/bin/bash
#SBATCH --job-name=gen_props
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/generate_proportions.out
#SBATCH --error=logs/generate_proportions.err

# Generate proportion tables for synthetic mixture training data.
# Quick job — just creates CSV files specifying what to generate.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
mkdir -p "${TRAINING_DIR}"

python scripts/generate_proportions.py \
    --manifest "${MANIFEST}" \
    --output-dir "${TRAINING_DIR}" \
    --n-train 100000 \
    --n-eval 20000 \
    --n-dilution-per-level 50 \
    --seed 42
