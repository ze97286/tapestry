#!/bin/bash
#SBATCH --job-name=clean_markers
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/control_clean_markers_%j.out
#SBATCH --error=logs/control_clean_markers_%j.err

# Upstream marker selection with healthy cfDNA controls included in the OAC
# candidate definition.

set -euo pipefail

source slurm/common.sh

RUN_LABEL="${RUN_LABEL:-AB_ctrl56_v04_control_clean}"
REF_HOMOG_DIR="${REF_HOMOG_DIR:-${OUTPUT_DIR}/homog}"
CONTROL_HOMOG_DIR="${CONTROL_HOMOG_DIR:-${OUTPUT_DIR}/control_homog_blocks/AB_CD_controls_v04_blocks}"
MARKERS_DIR="${MARKERS_DIR:-${OUTPUT_DIR}/markers_control_clean/${RUN_LABEL}}"
MARKERS_TSV="${MARKERS_TSV:-${MARKERS_DIR}/markers.tsv}"
MARKERS_BED="${MARKERS_BED:-${MARKERS_DIR}/markers.bed}"
CONTROL_PATTERN="${CONTROL_PATTERN:-Ctrl|healthy|^(GI|SCAN)}"
TARGET_CELL_TYPE="${TARGET_CELL_TYPE:-OAC}"
TOP_N="${TOP_N:-100}"
MAX_TARGET_CONTROL_P95="${MAX_TARGET_CONTROL_P95:-0.02}"
MIN_TARGET_CONTROL_DELTA="${MIN_TARGET_CONTROL_DELTA:-0.2}"
MAX_BACKBONE_CONTROL_P95="${MAX_BACKBONE_CONTROL_P95:-inf}"
MIN_CONTROL_OBSERVED_FRAC="${MIN_CONTROL_OBSERVED_FRAC:-0.6}"
MIN_CONTROL_COV="${MIN_CONTROL_COV:-1}"
MIN_SNR="${MIN_SNR:-2.0}"
MIN_SIGNAL="${MIN_SIGNAL:-0.2}"
MAX_BG="${MAX_BG:-0.2}"
MAX_SINGLE_BG="${MAX_SINGLE_BG:-0.35}"
MIN_CONSISTENCY="${MIN_CONSISTENCY:-0.1}"
MIN_COV_PER_SAMPLE="${MIN_COV_PER_SAMPLE:-5}"

mkdir -p "${MARKERS_DIR}" logs

echo "=== control-clean marker selection ==="
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_LABEL=${RUN_LABEL}"
echo "REF_HOMOG_DIR=${REF_HOMOG_DIR}"
echo "CONTROL_HOMOG_DIR=${CONTROL_HOMOG_DIR}"
echo "MARKERS_TSV=${MARKERS_TSV}"
echo "MAX_TARGET_CONTROL_P95=${MAX_TARGET_CONTROL_P95}"
echo "MIN_TARGET_CONTROL_DELTA=${MIN_TARGET_CONTROL_DELTA}"

for required_dir in "${REF_HOMOG_DIR}" "${CONTROL_HOMOG_DIR}"; do
    if [ ! -d "${required_dir}" ]; then
        echo "ERROR: required directory not found: ${required_dir}"
        exit 1
    fi
done

python scripts/select_control_clean_markers.py \
    --ref-homog-dir "${REF_HOMOG_DIR}" \
    --manifest "${MANIFEST}" \
    --control-homog-dir "${CONTROL_HOMOG_DIR}" \
    --output "${MARKERS_TSV}" \
    --bed-output "${MARKERS_BED}" \
    --diagnostics-output "${MARKERS_DIR}/markers.diagnostics.tsv" \
    --candidate-output "${MARKERS_DIR}/candidate_markers.tsv" \
    --summary-output "${MARKERS_DIR}/markers.summary.json" \
    --target-cell-type "${TARGET_CELL_TYPE}" \
    --control-pattern "${CONTROL_PATTERN}" \
    --top-n "${TOP_N}" \
    --max-target-control-p95 "${MAX_TARGET_CONTROL_P95}" \
    --min-target-control-delta "${MIN_TARGET_CONTROL_DELTA}" \
    --max-backbone-control-p95 "${MAX_BACKBONE_CONTROL_P95}" \
    --min-control-observed-frac "${MIN_CONTROL_OBSERVED_FRAC}" \
    --min-control-cov "${MIN_CONTROL_COV}" \
    --min-snr "${MIN_SNR}" \
    --min-signal "${MIN_SIGNAL}" \
    --max-bg "${MAX_BG}" \
    --max-single-bg "${MAX_SINGLE_BG}" \
    --min-consistency "${MIN_CONSISTENCY}" \
    --min-cov-per-sample "${MIN_COV_PER_SAMPLE}" \
    --direction U

echo
echo "Done."
echo "Markers TSV: ${MARKERS_TSV}"
echo "Markers BED: ${MARKERS_BED}"
echo "Summary: ${MARKERS_DIR}/markers.summary.json"
