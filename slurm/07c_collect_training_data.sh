#!/bin/bash
#SBATCH --job-name=collect_data
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/collect_training_data.out
#SBATCH --error=logs/collect_training_data.err

# Collect batch npz files into final parquet training data.
# Run after all 07b jobs have completed.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"

for DATASET in train eval oac_dilution tcell_dilution; do
    DATA_DIR="${TRAINING_DIR}/${DATASET}"
    if [ -d "${DATA_DIR}" ] && ls "${DATA_DIR}"/batch_*.npz 1>/dev/null 2>&1; then
        echo "Collecting ${DATASET}..."
        python scripts/collect_training_data.py \
            --input-dir "${DATA_DIR}" \
            --output-dir "${DATA_DIR}"
    else
        echo "Skipping ${DATASET} (no batch files found)"
    fi
done
