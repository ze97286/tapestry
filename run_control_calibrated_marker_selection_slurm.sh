#!/bin/bash
#SBATCH --job-name=ctrl_markers
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/control_markers_%j.out
#SBATCH --error=logs/control_markers_%j.err

# Select markers for weighted UXM NNLS with an explicit healthy-control OAC
# penalty.  This expects marker-value and coverage matrices exported by
# run_deconvolution_slurm.sh / scripts/predict_cfdna_augmented.py so selection
# uses the same cview/homog/alignment path as deconvolution.

set -euo pipefail

source slurm/common.sh

RUN_LABEL="${RUN_LABEL:-AB_ctrl56_l4_oac_excluded_unknown}"
ATLAS_TSV="${ATLAS_TSV:-${HOME}/sharedscratch/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed}"
MATRIX_DIR="${MATRIX_DIR:-${OUTPUT_DIR}/marker_matrices/${RUN_LABEL}}"
MARKER_VALUES_TSV="${MARKER_VALUES_TSV:-${MATRIX_DIR}/marker_values.tsv}"
COVERAGE_TSV="${COVERAGE_TSV:-${MATRIX_DIR}/coverage.tsv}"
MARKERS_DIR="${MARKERS_DIR:-${OUTPUT_DIR}/markers_control_calibrated/${RUN_LABEL}}"
MARKERS_TSV="${MARKERS_TSV:-${MARKERS_DIR}/markers.tsv}"
MARKERS_BED="${MARKERS_BED:-${MARKERS_DIR}/markers.bed}"
ICHORCNA_FILE="${ICHORCNA_FILE:-${PROJECT_DIR}/data/cfDNA_tumour_fraction_ichorCNA.json}"

TARGET_CELL_TYPE="${TARGET_CELL_TYPE:-OAC}"
CONTROL_PATTERN="${CONTROL_PATTERN:-Ctrl|healthy|^(GI|SCAN)}"
CELL_TYPES="${CELL_TYPES:-}"
TOP_BACKBONE_PER_CELL="${TOP_BACKBONE_PER_CELL:-50}"
TOP_TARGET="${TOP_TARGET:-250}"
N_FINAL="${N_FINAL:-1200}"
MIN_TARGET_DELTA="${MIN_TARGET_DELTA:-0.02}"
MIN_CONTROL_OBSERVED_FRAC="${MIN_CONTROL_OBSERVED_FRAC:-0.6}"
MIN_CONTROL_MEDIAN_COV="${MIN_CONTROL_MEDIAN_COV:-1.0}"
SEPARATION_WEIGHT="${SEPARATION_WEIGHT:-1.0}"
ICHOR_WEIGHT="${ICHOR_WEIGHT:-0.5}"
COVERAGE_WEIGHT="${COVERAGE_WEIGHT:-0.25}"
CONTROL_PENALTY_WEIGHT="${CONTROL_PENALTY_WEIGHT:-1.0}"
MISSING_PENALTY_WEIGHT="${MISSING_PENALTY_WEIGHT:-0.5}"

mkdir -p "${MARKERS_DIR}" logs

echo "=== control-calibrated marker selection ==="
echo "RUN_LABEL=${RUN_LABEL}"
echo "ATLAS_TSV=${ATLAS_TSV}"
echo "MARKER_VALUES_TSV=${MARKER_VALUES_TSV}"
echo "COVERAGE_TSV=${COVERAGE_TSV}"
echo "MARKERS_TSV=${MARKERS_TSV}"
echo "MARKERS_BED=${MARKERS_BED}"
echo "TARGET_CELL_TYPE=${TARGET_CELL_TYPE}"
echo "CONTROL_PATTERN=${CONTROL_PATTERN}"

for required in "${ATLAS_TSV}" "${MARKER_VALUES_TSV}" "${COVERAGE_TSV}"; do
    if [ ! -f "${required}" ]; then
        echo "ERROR: required input not found: ${required}"
        echo "If marker matrices are missing, rerun run_deconvolution_slurm.sh with:"
        echo "  MARKER_VALUES_OUTPUT=${MARKER_VALUES_TSV}"
        echo "  COVERAGE_OUTPUT=${COVERAGE_TSV}"
        exit 1
    fi
done

ARGS=(
    --atlas "${ATLAS_TSV}"
    --marker-values "${MARKER_VALUES_TSV}"
    --coverage "${COVERAGE_TSV}"
    --output "${MARKERS_TSV}"
    --bed-output "${MARKERS_BED}"
    --candidate-output "${MARKERS_DIR}/candidate_scores.tsv"
    --diagnostics-output "${MARKERS_DIR}/markers.diagnostics.json"
    --ichorcna-file "${ICHORCNA_FILE}"
    --target-cell-type "${TARGET_CELL_TYPE}"
    --control-pattern "${CONTROL_PATTERN}"
    --top-backbone-per-cell "${TOP_BACKBONE_PER_CELL}"
    --top-target "${TOP_TARGET}"
    --n-final "${N_FINAL}"
    --min-target-delta "${MIN_TARGET_DELTA}"
    --min-control-observed-frac "${MIN_CONTROL_OBSERVED_FRAC}"
    --min-control-median-cov "${MIN_CONTROL_MEDIAN_COV}"
    --separation-weight "${SEPARATION_WEIGHT}"
    --ichor-weight "${ICHOR_WEIGHT}"
    --coverage-weight "${COVERAGE_WEIGHT}"
    --control-penalty-weight "${CONTROL_PENALTY_WEIGHT}"
    --missing-penalty-weight "${MISSING_PENALTY_WEIGHT}"
)

if [ -n "${CELL_TYPES}" ]; then
    ARGS+=(--cell-types "${CELL_TYPES}")
fi

python scripts/select_control_calibrated_markers.py "${ARGS[@]}"

echo
echo "Done."
echo "Markers TSV: ${MARKERS_TSV}"
echo "Markers BED: ${MARKERS_BED}"
