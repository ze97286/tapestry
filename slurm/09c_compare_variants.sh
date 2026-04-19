#!/bin/bash
#SBATCH --job-name=compare_variants
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/compare_variants.out
#SBATCH --error=logs/compare_variants.err

# Diagnose NNLS-gate impact on clinical correlation.
# Reports OAC-vs-ichorCNA Pearson r for gated / raw / nnls-only variants.

source slurm/common.sh

PRED_DIR="${OUTPUT_DIR}/predictions"
ICHORCNA_FILE="${PROJECT_DIR}/data/cfDNA_tumour_fraction_ichorCNA.json"

for COHORT in AB CD; do
    PRED_FILE="${PRED_DIR}/${COHORT}_predictions.csv"
    if [ -f "${PRED_FILE}" ]; then
        python scripts/clinical_compare_variants.py \
            --predictions "${PRED_FILE}" \
            --ichorcna-file "${ICHORCNA_FILE}" \
            --cohort "${COHORT}"
    fi
done
