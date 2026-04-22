#!/bin/bash
#SBATCH --job-name=sweep_tls
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep_tls.out
#SBATCH --error=logs/sweep_tls.err

# Sweep atlas_sigma for the TLS solver against NNLS reference on the eval set.
# Single log with side-by-side table.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"

python scripts/sweep_tls_sigma.py \
    --data-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}" \
    --sigmas 0.005 0.01 0.02 0.05 0.10 0.20
