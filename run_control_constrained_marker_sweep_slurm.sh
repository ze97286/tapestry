#!/bin/bash
#SBATCH --job-name=ctrl_sweep
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/control_sweep_%j.out
#SBATCH --error=logs/control_sweep_%j.err

# Build several marker panels under hard healthy-control OAC-direction
# constraints.  This consumes the broad candidate atlas and the cfDNA/control
# marker matrices already produced by run_extract_marker_matrix_slurm.sh.

set -euo pipefail

source slurm/common.sh

RUN_LABEL="${RUN_LABEL:-AB_ctrl56_v04_candidates}"
ATLAS_TSV="${ATLAS_TSV:-${OUTPUT_DIR}/markers_unknown_robust/candidate_markers.tsv}"
MATRIX_DIR="${MATRIX_DIR:-${OUTPUT_DIR}/marker_matrices/${RUN_LABEL}}"
MARKER_VALUES_TSV="${MARKER_VALUES_TSV:-${MATRIX_DIR}/marker_values.tsv}"
COVERAGE_TSV="${COVERAGE_TSV:-${MATRIX_DIR}/coverage.tsv}"
SWEEP_DIR="${SWEEP_DIR:-${OUTPUT_DIR}/markers_control_constrained_sweep/${RUN_LABEL}}"
ICHORCNA_FILE="${ICHORCNA_FILE:-${PROJECT_DIR}/data/cfDNA_tumour_fraction_ichorCNA.json}"

TARGET_CELL_TYPE="${TARGET_CELL_TYPE:-OAC}"
CONTROL_PATTERN="${CONTROL_PATTERN:-Ctrl|healthy|^(GI|SCAN)}"
CONTROL_P95_GRID="${CONTROL_P95_GRID:-0,0.1,0.25,0.5,1,2,5,inf}"
CELL_TYPES="${CELL_TYPES:-}"
TOP_BACKBONE_PER_CELL="${TOP_BACKBONE_PER_CELL:-50}"
TOP_TARGET="${TOP_TARGET:-250}"
N_FINAL="${N_FINAL:-1200}"
MIN_TARGET_DELTA="${MIN_TARGET_DELTA:-0.02}"
MIN_BACKBONE_SCORE="${MIN_BACKBONE_SCORE:-0}"
MIN_CONTROL_OBSERVED_FRAC="${MIN_CONTROL_OBSERVED_FRAC:-0.6}"
MIN_CONTROL_MEDIAN_COV="${MIN_CONTROL_MEDIAN_COV:-1.0}"
SEPARATION_WEIGHT="${SEPARATION_WEIGHT:-1.0}"
ICHOR_WEIGHT="${ICHOR_WEIGHT:-0.5}"
COVERAGE_WEIGHT="${COVERAGE_WEIGHT:-0.25}"
CONTROL_PENALTY_WEIGHT="${CONTROL_PENALTY_WEIGHT:-1.0}"
MISSING_PENALTY_WEIGHT="${MISSING_PENALTY_WEIGHT:-0.5}"

mkdir -p "${SWEEP_DIR}" logs

echo "=== hard control-constrained marker sweep ==="
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_LABEL=${RUN_LABEL}"
echo "ATLAS_TSV=${ATLAS_TSV}"
echo "MARKER_VALUES_TSV=${MARKER_VALUES_TSV}"
echo "COVERAGE_TSV=${COVERAGE_TSV}"
echo "SWEEP_DIR=${SWEEP_DIR}"
echo "CONTROL_PATTERN=${CONTROL_PATTERN}"
echo "CONTROL_P95_GRID=${CONTROL_P95_GRID}"

for required in "${ATLAS_TSV}" "${MARKER_VALUES_TSV}" "${COVERAGE_TSV}"; do
    if [ ! -f "${required}" ]; then
        echo "ERROR: required input not found: ${required}"
        exit 1
    fi
done

ARGS=(
    --atlas "${ATLAS_TSV}"
    --marker-values "${MARKER_VALUES_TSV}"
    --coverage "${COVERAGE_TSV}"
    --output-dir "${SWEEP_DIR}"
    --ichorcna-file "${ICHORCNA_FILE}"
    --target-cell-type "${TARGET_CELL_TYPE}"
    --control-pattern "${CONTROL_PATTERN}"
    --control-p95-grid "${CONTROL_P95_GRID}"
    --top-backbone-per-cell "${TOP_BACKBONE_PER_CELL}"
    --top-target "${TOP_TARGET}"
    --n-final "${N_FINAL}"
    --min-target-delta "${MIN_TARGET_DELTA}"
    --min-backbone-score "${MIN_BACKBONE_SCORE}"
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

python scripts/sweep_control_constrained_markers.py "${ARGS[@]}"

echo
echo "Done."
echo "Sweep summary: ${SWEEP_DIR}/sweep_summary.tsv"
