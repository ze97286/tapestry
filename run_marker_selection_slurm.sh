#!/bin/bash
#SBATCH --job-name=robust_markers
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/robust_markers_%j.out
#SBATCH --error=logs/robust_markers_%j.err

# End-to-end marker selection for coverage-weighted UXM NNLS with an unknown
# component.
#
# Produces:
#   ${MARKERS_DIR}/candidate_markers.tsv
#   ${MARKERS_DIR}/markers.tsv
#   ${MARKERS_DIR}/markers.bed
#   ${MARKERS_DIR}/markers.diagnostics.json
#   ${MARKERS_DIR}/markers.selection_annotations.tsv
#
# Submit:
#   sbatch run_marker_selection_slurm.sh
#
# Common overrides:
#   sbatch --export=ALL,MARKERS_DIR=/path/to/out run_marker_selection_slurm.sh
#   sbatch --export=ALL,TARGET_CELL_TYPE=OAC,TOP_TARGET=150 run_marker_selection_slurm.sh

set -euo pipefail

source slurm/common.sh

MARKERS_DIR="${MARKERS_DIR:-${OUTPUT_DIR}/markers_unknown_robust}"
HOMOG_DIR="${HOMOG_DIR:-${OUTPUT_DIR}/homog}"
CANDIDATE_MARKERS_TSV="${CANDIDATE_MARKERS_TSV:-${MARKERS_DIR}/candidate_markers.tsv}"
FINAL_MARKERS_TSV="${FINAL_MARKERS_TSV:-${MARKERS_DIR}/markers.tsv}"
FINAL_MARKERS_BED="${FINAL_MARKERS_BED:-${MARKERS_DIR}/markers.bed}"

TARGET_CELL_TYPE="${TARGET_CELL_TYPE:-OAC}"
EXCLUDE_FROM_BACKBONE="${EXCLUDE_FROM_BACKBONE:-}"
UNKNOWN_BASIS_NPY="${UNKNOWN_BASIS_NPY:-}"

CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-750}"
MIN_SNR="${MIN_SNR:-2.0}"
MIN_SIGNAL="${MIN_SIGNAL:-0.2}"
MAX_BG="${MAX_BG:-0.2}"
MAX_SINGLE_BG="${MAX_SINGLE_BG:-0.35}"
MIN_CONSISTENCY="${MIN_CONSISTENCY:-0.1}"
MIN_COV_PER_SAMPLE="${MIN_COV_PER_SAMPLE:-5}"

TOP_BACKBONE_PER_CELL="${TOP_BACKBONE_PER_CELL:-100}"
TOP_TARGET="${TOP_TARGET:-100}"
TARGET_CANDIDATE_POOL="${TARGET_CANDIDATE_POOL:-300}"
CONDITION_PENALTY="${CONDITION_PENALTY:-0}"
MIN_SINGLE_MARKER_SCORE="${MIN_SINGLE_MARKER_SCORE:-0}"
CONE_RECONSTRUCTION="${CONE_RECONSTRUCTION:-0}"

mkdir -p "${MARKERS_DIR}" logs

echo "=== robust marker selection ==="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HOMOG_DIR=${HOMOG_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "MARKERS_DIR=${MARKERS_DIR}"
echo "TARGET_CELL_TYPE=${TARGET_CELL_TYPE}"

if [ ! -d "${HOMOG_DIR}" ]; then
    echo "ERROR: HOMOG_DIR not found: ${HOMOG_DIR}"
    echo "Run the reference homog step first, or set HOMOG_DIR."
    exit 1
fi

echo
echo "Step 1/3: build broad candidate UXM atlas"
python scripts/select_markers.py \
    --homog-dir "${HOMOG_DIR}" \
    --manifest "${MANIFEST}" \
    --output "${CANDIDATE_MARKERS_TSV}" \
    --top-n "${CANDIDATE_TOP_N}" \
    --min-snr "${MIN_SNR}" \
    --min-signal "${MIN_SIGNAL}" \
    --max-bg "${MAX_BG}" \
    --max-single-bg "${MAX_SINGLE_BG}" \
    --min-consistency "${MIN_CONSISTENCY}" \
    --min-cov-per-sample "${MIN_COV_PER_SAMPLE}" \
    --direction U

echo
echo "Step 2/3: select unknown-robust marker subset"
ROBUST_ARGS=(
    --atlas "${CANDIDATE_MARKERS_TSV}"
    --output "${FINAL_MARKERS_TSV}"
    --diagnostics-output "${MARKERS_DIR}/markers.diagnostics.json"
    --annotation-output "${MARKERS_DIR}/markers.selection_annotations.tsv"
    --target-cell-type "${TARGET_CELL_TYPE}"
    --exclude-from-backbone "${EXCLUDE_FROM_BACKBONE}"
    --top-backbone-per-cell "${TOP_BACKBONE_PER_CELL}"
    --top-target "${TOP_TARGET}"
    --target-candidate-pool "${TARGET_CANDIDATE_POOL}"
    --condition-penalty "${CONDITION_PENALTY}"
    --min-single-marker-score "${MIN_SINGLE_MARKER_SCORE}"
)

if [ -n "${UNKNOWN_BASIS_NPY}" ]; then
    ROBUST_ARGS+=(--unknown-basis-npy "${UNKNOWN_BASIS_NPY}")
fi

if [ "${CONE_RECONSTRUCTION}" = "1" ]; then
    ROBUST_ARGS+=(--cone-reconstruction)
fi

python scripts/select_unknown_robust_markers.py "${ROBUST_ARGS[@]}"

echo
echo "Step 3/3: write markers BED for PAT filtering/homog"
python -c '
import sys
import pandas as pd
markers_tsv, markers_bed = sys.argv[1], sys.argv[2]
df = pd.read_csv(markers_tsv, sep="\t")
required = ["chr", "start", "end", "startCpG", "endCpG"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"missing columns for BED: {missing}")
df[required].to_csv(markers_bed, sep="\t", header=False, index=False)
print(f"wrote {len(df)} regions to {markers_bed}")
' "${FINAL_MARKERS_TSV}" "${FINAL_MARKERS_BED}"

echo
echo "Done."
echo "Final atlas: ${FINAL_MARKERS_TSV}"
echo "Final BED:   ${FINAL_MARKERS_BED}"
