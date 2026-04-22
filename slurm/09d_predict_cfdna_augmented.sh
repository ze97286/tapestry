#!/bin/bash
#SBATCH --job-name=predict_aug
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/predict_cfdna_augmented_%j.out
#SBATCH --error=logs/predict_cfdna_augmented_%j.err

# Augmented-basis NNLS cfDNA deconvolution.
# Uses healthy controls in the cohort to build a K-dim unknown-tissue basis
# via SVD of their NNLS residuals, then augments the atlas with that basis
# for per-sample prediction. Non-atlas signal is absorbed by free-sign
# coefficients on the basis, preventing misattribution to atlas cell types.
#
# Usage:
#   sbatch --export=ALL,COHORT=AB slurm/09d_predict_cfdna_augmented.sh
#   sbatch --export=ALL,COHORT=CD slurm/09d_predict_cfdna_augmented.sh

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set. Use --export=ALL,COHORT=AB or COHORT=CD"
    exit 1
fi

CFDNA_DIR="${OUTPUT_DIR}/filtered_pats/cfdna/${COHORT}"
MARKERS_BED="${OUTPUT_DIR}/markers/markers.bed"
ATLAS="${OUTPUT_DIR}/markers/markers.tsv"
PRED_DIR="${OUTPUT_DIR}/predictions"

mkdir -p "${PRED_DIR}"

echo "Cohort: ${COHORT}"
echo "cfDNA dir: ${CFDNA_DIR}"

python scripts/predict_cfdna_augmented.py \
    --cfdna-dir "${CFDNA_DIR}" \
    --markers-bed "${MARKERS_BED}" \
    --atlas "${ATLAS}" \
    --output "${PRED_DIR}/${COHORT}_augmented_predictions.csv" \
    --wgbstools "${WGBSTOOLS}" \
    --cohort "${COHORT}" \
    --control-pattern "_Ctrl_|^Ctrl_|_healthy_" \
    --n-components 2
