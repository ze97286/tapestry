#!/bin/bash
#SBATCH --job-name=predict_rlb
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/predict_cfdna_robust_lower_bound_%j.out
#SBATCH --error=logs/predict_cfdna_robust_lower_bound_%j.err

# Robust OAC lower-bound prediction.
#
# Usage:
#   sbatch --export=ALL,COHORT=AB slurm/09e_predict_cfdna_robust_lower_bound.sh
# Optional:
#   ALPHA=0.05 FACTOR_RANK=auto FACTOR_RANK_GRID=0,3,5,10

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set. Use --export=ALL,COHORT=AB or COHORT=CD"
    exit 1
fi

CFDNA_DIR="${OUTPUT_DIR}/filtered_pats/cfdna/${COHORT}"
MARKERS_BED="${OUTPUT_DIR}/markers/markers.bed"
ATLAS="${OUTPUT_DIR}/markers/markers.tsv"
PRED_DIR="${OUTPUT_DIR}/predictions"
TARGET_CELL_TYPE="${TARGET_CELL_TYPE:-OAC}"
ALPHA="${ALPHA:-0.05}"
FACTOR_RANK="${FACTOR_RANK:-auto}"
FACTOR_RANK_GRID="${FACTOR_RANK_GRID:-0,3,5,10}"
MIN_CONTROL_COVERAGE_FRACTION="${MIN_CONTROL_COVERAGE_FRACTION:-0.8}"
MIN_SAMPLE_COVERAGE="${MIN_SAMPLE_COVERAGE:-1}"

mkdir -p "${PRED_DIR}"

echo "Cohort: ${COHORT}"
echo "cfDNA dir: ${CFDNA_DIR}"
echo "target cell type: ${TARGET_CELL_TYPE}"
echo "alpha: ${ALPHA}"
echo "factor rank: ${FACTOR_RANK}"
echo "factor rank grid: ${FACTOR_RANK_GRID}"

python scripts/predict_cfdna_robust_lower_bound.py \
    --cfdna-dir "${CFDNA_DIR}" \
    --markers-bed "${MARKERS_BED}" \
    --atlas "${ATLAS}" \
    --output "${PRED_DIR}/${COHORT}_robust_lower_bound_predictions.csv" \
    --wgbstools "${WGBSTOOLS}" \
    --cohort "${COHORT}" \
    --target-cell-type "${TARGET_CELL_TYPE}" \
    --control-pattern "_Ctrl_|^Ctrl_|_healthy_" \
    --alpha "${ALPHA}" \
    --factor-rank "${FACTOR_RANK}" \
    --factor-rank-grid "${FACTOR_RANK_GRID}" \
    --min-control-coverage-fraction "${MIN_CONTROL_COVERAGE_FRACTION}" \
    --min-sample-coverage "${MIN_SAMPLE_COVERAGE}"
