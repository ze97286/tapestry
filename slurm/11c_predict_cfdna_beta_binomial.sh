#!/bin/bash
#SBATCH --job-name=predict_bb
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/predict_cfdna_beta_binomial_%j.out
#SBATCH --error=logs/predict_cfdna_beta_binomial_%j.err

# Beta-Binomial cfDNA deconvolution with empirical per-marker Beta priors.
#
# Prereqs (run in order):
#   sbatch slurm/11a_homog_atlas_references.sh
#   sbatch slurm/11b_build_atlas_priors.sh
#
# Usage:
#   sbatch --export=ALL,COHORT=AB slurm/11c_predict_cfdna_beta_binomial.sh
#   sbatch --export=ALL,COHORT=CD slurm/11c_predict_cfdna_beta_binomial.sh

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set. Use --export=ALL,COHORT=AB or COHORT=CD"
    exit 1
fi

CFDNA_DIR="${OUTPUT_DIR}/filtered_pats/cfdna/${COHORT}"
MARKERS_BED="${OUTPUT_DIR}/markers/markers.bed"
ATLAS="${OUTPUT_DIR}/markers/markers.tsv"
PRIORS="${PROJECT_DIR}/data/atlas_priors_ben.tsv"
PRED_DIR="${OUTPUT_DIR}/predictions"

mkdir -p "${PRED_DIR}"

if [ ! -f "${PRIORS}" ]; then
    echo "ERROR: priors file ${PRIORS} not found. Run 11b_build_atlas_priors.sh first."
    exit 1
fi

echo "Cohort: ${COHORT}"
echo "cfDNA dir: ${CFDNA_DIR}"
echo "Priors: ${PRIORS}"

python scripts/predict_cfdna_beta_binomial.py \
    --cfdna-dir "${CFDNA_DIR}" \
    --markers-bed "${MARKERS_BED}" \
    --atlas "${ATLAS}" \
    --priors "${PRIORS}" \
    --output "${PRED_DIR}/${COHORT}_beta_binomial_predictions.csv" \
    --wgbstools "${WGBSTOOLS}" \
    --cohort "${COHORT}" \
    --control-pattern "_Ctrl_|^Ctrl_|_healthy_"
