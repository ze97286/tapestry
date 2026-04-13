#!/bin/bash
#SBATCH --job-name=vis_data
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/visualise_training_data.out
#SBATCH --error=logs/visualise_training_data.err

# Visualise training and evaluation data distributions.
# Run after 07c has collected all batch data into parquets.
# Generates interactive HTML plots. Add --png flag for PNG export (slow on headless nodes).
# For paper-quality PNGs, download HTML files and render locally.

source slurm/common.sh

TRAINING_DIR="${OUTPUT_DIR}/training"
PLOTS_DIR="${TRAINING_DIR}/plots"
mkdir -p "${PLOTS_DIR}"

for DATASET in train eval oac_dilution tcell_dilution; do
    DATA_DIR="${TRAINING_DIR}/${DATASET}"
    PROPS_CSV="${TRAINING_DIR}/${DATASET}_proportions.csv"

    if [ -f "${DATA_DIR}/ground_truth_y.parquet" ]; then
        echo "Visualising ${DATASET}..."
        python scripts/visualise_training_data.py \
            --data-dir "${DATA_DIR}" \
            --proportions "${PROPS_CSV}" \
            --output-dir "${PLOTS_DIR}/${DATASET}" \
            --dataset-name "${DATASET}"
    else
        echo "Skipping ${DATASET} (no parquet files found)"
    fi
done
