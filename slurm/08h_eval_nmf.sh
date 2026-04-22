#!/bin/bash
#SBATCH --job-name=eval_nmf
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_nmf.out
#SBATCH --error=logs/eval_nmf.err

# Cohort-level NMF deconvolution with atlas anchor — sweeps anchor_weight.
# NNLS as reference. S_drift column = how much learned signatures moved from
# the atlas (0 = signatures pinned to atlas, larger = more data-driven).

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"

python scripts/eval_nmf.py \
    --data-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}" \
    --anchor-weights 0 1 10 100 1000
