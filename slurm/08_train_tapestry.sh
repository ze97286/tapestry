#!/bin/bash
#SBATCH --job-name=train_tapestry
#SBATCH --partition=gpu_rtx8000_48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_tapestry.out
#SBATCH --error=logs/train_tapestry.err

# Train the TapestryModel for cfDNA deconvolution.
# Requires training data from step 7.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
MARKERS="${OUTPUT_DIR}/markers/markers.tsv"
MODEL_DIR="${OUTPUT_DIR}/models/tapestry"

mkdir -p "${MODEL_DIR}"

python scripts/train_tapestry.py \
    --train-dir "${TRAINING_DIR}/train" \
    --eval-dir "${TRAINING_DIR}/eval" \
    --atlas "${MARKERS}" \
    --output-dir "${MODEL_DIR}" \
    --feature-dim 64 \
    --l1-heads 4 \
    --l1-layers 2 \
    --l2-heads 4 \
    --l2-layers 1 \
    --dropout 0.1 \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 0.01 \
    --epochs 10000 \
    --patience 20 \
    --grad-accum-steps 4 \
    --save-interval 10 \
    --phi 50.0 \
    --detection-epochs 20 \
    --detection-threshold 0.5
