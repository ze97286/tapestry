#!/bin/bash
#SBATCH --job-name=predict_calibrated
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/predict_calibrated.out
#SBATCH --error=logs/predict_calibrated.err

# Post-hoc calibrated + NNLS-gated predictions for the best tapestry checkpoint.
# Compares raw / recalibrated / NNLS-gated / both variants side by side.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"
MODEL_DIR="${OUTPUT_DIR}/models/tapestry"
CHECKPOINT="${MODEL_DIR}/best_model.pt"

python scripts/predict_calibrated.py \
    --checkpoint "${CHECKPOINT}" \
    --train-dir "${TRAINING_DIR}/train" \
    --eval-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}" \
    --output-dir "${MODEL_DIR}/calibrated"
