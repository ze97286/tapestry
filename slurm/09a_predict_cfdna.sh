#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/predict_cfdna_%j.out
#SBATCH --error=logs/predict_cfdna_%j.err

# Run TapestryModel predictions on cfDNA cohorts.
#
# Usage:
#   sbatch --export=ALL,COHORT=AB slurm/09a_predict_cfdna.sh
#   sbatch --export=ALL,COHORT=CD slurm/09a_predict_cfdna.sh

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set. Use --export=ALL,COHORT=AB or COHORT=CD"
    exit 1
fi

CFDNA_DIR="${OUTPUT_DIR}/filtered_pats/cfdna/${COHORT}"
MARKERS_BED="${OUTPUT_DIR}/markers/markers.bed"
ATLAS="${OUTPUT_DIR}/markers/markers.tsv"
MODEL="${OUTPUT_DIR}/models/tapestry/best_model.pt"
PRED_DIR="${OUTPUT_DIR}/predictions"

mkdir -p "${PRED_DIR}"

echo "Predicting cohort ${COHORT}"
echo "cfDNA dir: ${CFDNA_DIR}"
echo "Model: ${MODEL}"

# Tapestry model predictions
python scripts/predict_cfdna.py \
    --cfdna-dir "${CFDNA_DIR}" \
    --markers-bed "${MARKERS_BED}" \
    --atlas "${ATLAS}" \
    --model "${MODEL}" \
    --output "${PRED_DIR}/${COHORT}_predictions.csv" \
    --wgbstools "${WGBSTOOLS}" \
    --cohort "${COHORT}"

# NNLS baseline predictions
python scripts/predict_cfdna_nnls.py \
    --cfdna-dir "${CFDNA_DIR}" \
    --markers-bed "${MARKERS_BED}" \
    --atlas "${ATLAS}" \
    --output "${PRED_DIR}/${COHORT}_nnls_predictions.csv" \
    --wgbstools "${WGBSTOOLS}" \
    --cohort "${COHORT}"
