#!/bin/bash
#SBATCH --job-name=extract_uxm
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/extract_uxm_%j.out
#SBATCH --error=logs/extract_uxm_%j.err

# Extract observed cfDNA UXM marker values and coverage for a candidate atlas.
# This is intentionally not a deconvolution job; it only materialises the
# marker-by-sample matrices needed by control-calibrated marker selection.

set -euo pipefail

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set. Use: sbatch --export=ALL,COHORT=AB run_extract_marker_matrix_slurm.sh"
    exit 1
fi

RUN_LABEL="${RUN_LABEL:-${COHORT}_candidate_l4}"
ATLAS_TSV="${ATLAS_TSV:-${MARKERS_TSV:-${HOME}/sharedscratch/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed}}"
MATRIX_DIR="${MATRIX_DIR:-${OUTPUT_DIR}/marker_matrices/${RUN_LABEL}}"
MARKERS_BED="${MARKERS_BED:-${MATRIX_DIR}/candidate_markers.bed}"
FILTERED_DIR="${FILTERED_DIR:-${OUTPUT_DIR}/filtered_pats/marker_matrix/${RUN_LABEL}}"
MARKER_VALUES_OUTPUT="${MARKER_VALUES_OUTPUT:-${MATRIX_DIR}/marker_values.tsv}"
COVERAGE_OUTPUT="${COVERAGE_OUTPUT:-${MATRIX_DIR}/coverage.tsv}"
SAMPLE_MANIFEST_OUTPUT="${SAMPLE_MANIFEST_OUTPUT:-${MATRIX_DIR}/sample_manifest.tsv}"

CFDNA_INPUT_DIR="${CFDNA_INPUT_DIR:-${PROJECT_DIR}/data/${COHORT}}"
CFDNA_EXTRA_CONTROL_DIRS="${CFDNA_EXTRA_CONTROL_DIRS:-}"
EXTRA_CONTROL_PATTERN="${EXTRA_CONTROL_PATTERN:-^(GI|SCAN)}"
CFDNA_CVIEW_ARGS="${CFDNA_CVIEW_ARGS:-}"
FORCE_REFILTER="${FORCE_REFILTER:-0}"
HOMOG_LEN="${HOMOG_LEN:-4}"
CELL_TYPES="${CELL_TYPES:-}"

mkdir -p "${MATRIX_DIR}" "${FILTERED_DIR}" logs

echo "=== extract cfDNA UXM marker matrix ==="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "COHORT=${COHORT}"
echo "RUN_LABEL=${RUN_LABEL}"
echo "ATLAS_TSV=${ATLAS_TSV}"
echo "MARKERS_BED=${MARKERS_BED}"
echo "CFDNA_INPUT_DIR=${CFDNA_INPUT_DIR}"
echo "CFDNA_EXTRA_CONTROL_DIRS=${CFDNA_EXTRA_CONTROL_DIRS}"
echo "EXTRA_CONTROL_PATTERN=${EXTRA_CONTROL_PATTERN}"
echo "CFDNA_CVIEW_ARGS=${CFDNA_CVIEW_ARGS}"
echo "FILTERED_DIR=${FILTERED_DIR}"
echo "MARKER_VALUES_OUTPUT=${MARKER_VALUES_OUTPUT}"
echo "COVERAGE_OUTPUT=${COVERAGE_OUTPUT}"
echo "HOMOG_LEN=${HOMOG_LEN}"

if [ ! -f "${ATLAS_TSV}" ]; then
    echo "ERROR: atlas TSV not found: ${ATLAS_TSV}"
    exit 1
fi
if [ ! -d "${CFDNA_INPUT_DIR}" ]; then
    echo "ERROR: cfDNA input directory not found: ${CFDNA_INPUT_DIR}"
    exit 1
fi

EXTRA_ARGS=()
if [ -n "${CFDNA_EXTRA_CONTROL_DIRS}" ]; then
    IFS=':' read -r -a EXTRA_DIR_ARRAY <<< "${CFDNA_EXTRA_CONTROL_DIRS}"
    for EXTRA_DIR in "${EXTRA_DIR_ARRAY[@]}"; do
        if [ -z "${EXTRA_DIR}" ]; then
            continue
        fi
        if [ ! -d "${EXTRA_DIR}" ]; then
            echo "ERROR: extra control directory not found: ${EXTRA_DIR}"
            exit 1
        fi
        EXTRA_ARGS+=(--extra-control-dir "${EXTRA_DIR}")
    done
    EXTRA_ARGS+=(--extra-control-pattern "${EXTRA_CONTROL_PATTERN}")
fi
if [ "${FORCE_REFILTER}" = "1" ]; then
    EXTRA_ARGS+=(--force-refilter)
fi
if [ -n "${CELL_TYPES}" ]; then
    EXTRA_ARGS+=(--cell-types "${CELL_TYPES}")
fi

python scripts/extract_cfdna_marker_matrix.py \
    --cfdna-dir "${CFDNA_INPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    --atlas "${ATLAS_TSV}" \
    --markers-bed "${MARKERS_BED}" \
    --filtered-dir "${FILTERED_DIR}" \
    --marker-values-output "${MARKER_VALUES_OUTPUT}" \
    --coverage-output "${COVERAGE_OUTPUT}" \
    --sample-manifest-output "${SAMPLE_MANIFEST_OUTPUT}" \
    --wgbstools "${WGBSTOOLS}" \
    --cview-args "${CFDNA_CVIEW_ARGS}" \
    --homog-len "${HOMOG_LEN}"

echo
echo "Done."
echo "Marker values: ${MARKER_VALUES_OUTPUT}"
echo "Coverage: ${COVERAGE_OUTPUT}"
echo "Sample manifest: ${SAMPLE_MANIFEST_OUTPUT}"
