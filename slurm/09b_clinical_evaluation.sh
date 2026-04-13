#!/bin/bash
#SBATCH --job-name=eval_clin
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/clinical_evaluation.out
#SBATCH --error=logs/clinical_evaluation.err

# Clinical evaluation of tapestry predictions.
# Run after 09a_predict_cfdna.sh has completed for AB and CD cohorts.

source slurm/common.sh

PRED_DIR="${OUTPUT_DIR}/predictions"
EVAL_DIR="${OUTPUT_DIR}/evaluation"
CLINICAL_FILE="${PROJECT_DIR}/data/AB_patient_summary_HannahFuchs2023.csv"
ICHORCNA_FILE="${PROJECT_DIR}/data/cfDNA_tumour_fraction_ichorCNA.json"

mkdir -p "${EVAL_DIR}"

# AB cohort
if [ -f "${PRED_DIR}/AB_predictions.csv" ]; then
    echo "Evaluating AB cohort..."
    NNLS_FLAG=""
    if [ -f "${PRED_DIR}/AB_nnls_predictions.csv" ]; then
        NNLS_FLAG="--nnls-predictions ${PRED_DIR}/AB_nnls_predictions.csv"
    fi
    python scripts/clinical_evaluation.py \
        --predictions "${PRED_DIR}/AB_predictions.csv" \
        ${NNLS_FLAG} \
        --clinical-file "${CLINICAL_FILE}" \
        --ichorcna-file "${ICHORCNA_FILE}" \
        --output-dir "${EVAL_DIR}/AB" \
        --cohort AB
fi

# CD cohort
if [ -f "${PRED_DIR}/CD_predictions.csv" ]; then
    echo "Evaluating CD cohort..."
    NNLS_FLAG=""
    if [ -f "${PRED_DIR}/CD_nnls_predictions.csv" ]; then
        NNLS_FLAG="--nnls-predictions ${PRED_DIR}/CD_nnls_predictions.csv"
    fi
    python scripts/clinical_evaluation.py \
        --predictions "${PRED_DIR}/CD_predictions.csv" \
        ${NNLS_FLAG} \
        --output-dir "${EVAL_DIR}/CD" \
        --cohort CD
fi
