#!/bin/bash
#SBATCH --job-name=robust_deconv
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/robust_deconv_%j.out
#SBATCH --error=logs/robust_deconv_%j.err

# End-to-end cfDNA deconvolution for the unknown-robust UXM NNLS project.
#
# Steps:
#   1. Filter cfDNA PAT files to the selected marker BED.
#   2. Run augmented/unknown-channel weighted NNLS over a lambda path.
#
# Submit:
#   sbatch --export=ALL,COHORT=AB run_deconvolution_slurm.sh
#
# Common overrides:
#   MARKERS_DIR=/path/to/markers_unknown_robust
#   CFDNA_INPUT_DIR=/path/to/raw/pats
#   CFDNA_CVIEW_ARGS="..."        # e.g. read2-specific cview flags if needed
#   FORCE_REFILTER=1
#   CONTROL_PATTERN="Ctrl|healthy"
#   N_COMPONENTS=3
#   PRIMARY_LAMBDA_UNKNOWN=10

set -euo pipefail

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set. Use: sbatch --export=ALL,COHORT=AB run_deconvolution_slurm.sh"
    exit 1
fi

MARKERS_DIR="${MARKERS_DIR:-${OUTPUT_DIR}/markers_unknown_robust}"
MARKERS_TSV="${MARKERS_TSV:-${MARKERS_DIR}/markers.tsv}"
MARKERS_BED="${MARKERS_BED:-${MARKERS_DIR}/markers.bed}"

CFDNA_INPUT_DIR="${CFDNA_INPUT_DIR:-${PROJECT_DIR}/data/${COHORT}}"
FILTERED_DIR="${FILTERED_DIR:-${OUTPUT_DIR}/filtered_pats/unknown_robust/${COHORT}}"
PRED_DIR="${PRED_DIR:-${OUTPUT_DIR}/predictions_unknown_robust}"
PRED_OUTPUT="${PRED_OUTPUT:-${PRED_DIR}/${COHORT}_unknown_robust_nnls.csv}"
PATH_OUTPUT="${PATH_OUTPUT:-${PRED_DIR}/${COHORT}_unknown_robust_nnls_lambda_path.csv}"

CFDNA_CVIEW_ARGS="${CFDNA_CVIEW_ARGS:-}"
FORCE_REFILTER="${FORCE_REFILTER:-0}"

CONTROL_PATTERN="${CONTROL_PATTERN:-Ctrl|healthy}"
N_COMPONENTS="${N_COMPONENTS:-3}"
LAMBDA_UNKNOWN_GRID="${LAMBDA_UNKNOWN_GRID:-0,0.01,0.1,1,10,100,1000,10000}"
PRIMARY_LAMBDA_UNKNOWN="${PRIMARY_LAMBDA_UNKNOWN:-10}"
ORTHOGONALIZE_TARGET="${ORTHOGONALIZE_TARGET:-OAC}"

mkdir -p "${FILTERED_DIR}" "${PRED_DIR}" logs

echo "=== unknown-robust deconvolution ==="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "COHORT=${COHORT}"
echo "CFDNA_INPUT_DIR=${CFDNA_INPUT_DIR}"
echo "FILTERED_DIR=${FILTERED_DIR}"
echo "MARKERS_TSV=${MARKERS_TSV}"
echo "MARKERS_BED=${MARKERS_BED}"
echo "CFDNA_CVIEW_ARGS=${CFDNA_CVIEW_ARGS}"

if [ ! -f "${MARKERS_TSV}" ]; then
    echo "ERROR: markers TSV not found: ${MARKERS_TSV}"
    echo "Run marker selection first, or set MARKERS_TSV/MARKERS_DIR."
    exit 1
fi

if [ ! -f "${MARKERS_BED}" ]; then
    echo "markers BED missing; creating ${MARKERS_BED}"
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
' "${MARKERS_TSV}" "${MARKERS_BED}"
fi

if [ ! -d "${CFDNA_INPUT_DIR}" ]; then
    echo "ERROR: cfDNA input directory not found: ${CFDNA_INPUT_DIR}"
    exit 1
fi

FILE_LIST="${FILTERED_DIR}/pat_files.list"
find "${CFDNA_INPUT_DIR}" -maxdepth 1 -name "*.pat.gz" | sort > "${FILE_LIST}.tmp"
mv "${FILE_LIST}.tmp" "${FILE_LIST}"
N_FILES=$(wc -l < "${FILE_LIST}")
echo "Found ${N_FILES} cfDNA PAT files"
if [ "${N_FILES}" -eq 0 ]; then
    echo "ERROR: no *.pat.gz files found in ${CFDNA_INPUT_DIR}"
    exit 1
fi

echo
echo "Step 1/2: filter cfDNA PAT files to robust marker BED"
while IFS= read -r FPATH; do
    SID=$(basename "${FPATH}" .pat.gz)
    OUT_FILE="${FILTERED_DIR}/${SID}.markers.pat.gz"
    if [ -f "${OUT_FILE}" ] && [ "${FORCE_REFILTER}" != "1" ]; then
        echo "Skipping existing ${OUT_FILE}"
        continue
    fi

    echo "Filtering ${SID}"
    # CFDNA_CVIEW_ARGS is intentionally unquoted so callers can pass additional
    # wgbstools cview flags such as read-pair/read2 filters when needed.
    ${WGBSTOOLS} cview "${FPATH}" -L "${MARKERS_BED}" ${CFDNA_CVIEW_ARGS} \
        | bgzip > "${OUT_FILE}.tmp"
    mv "${OUT_FILE}.tmp" "${OUT_FILE}"
    tabix -f -s 1 -b 2 -e 2 "${OUT_FILE}"

    N_LINES=$(zcat "${OUT_FILE}" | wc -l)
    N_READS=$(zcat "${OUT_FILE}" | awk -F'\t' '{s+=$4} END{print s+0}')
    echo "  ${SID}: patterns=${N_LINES}, reads=${N_READS}"
    if [ "${N_LINES}" -eq 0 ]; then
        echo "  WARNING: filtered PAT is empty"
    fi
done < "${FILE_LIST}"

echo
echo "Step 2/2: run unknown-channel weighted NNLS"
python scripts/predict_cfdna_augmented.py \
    --cfdna-dir "${FILTERED_DIR}" \
    --markers-bed "${MARKERS_BED}" \
    --atlas "${MARKERS_TSV}" \
    --output "${PRED_OUTPUT}" \
    --path-output "${PATH_OUTPUT}" \
    --wgbstools "${WGBSTOOLS}" \
    --cohort "${COHORT}" \
    --control-pattern "${CONTROL_PATTERN}" \
    --n-components "${N_COMPONENTS}" \
    --lambda-unknown-grid "${LAMBDA_UNKNOWN_GRID}" \
    --primary-lambda-unknown "${PRIMARY_LAMBDA_UNKNOWN}" \
    --orthogonalize-target "${ORTHOGONALIZE_TARGET}"

if [ ! -s "${PRED_OUTPUT}" ]; then
    echo "ERROR: prediction output was not created or is empty: ${PRED_OUTPUT}"
    echo "Check the Python log above for the real failure, usually no matched controls"
    echo "or no successfully processed PAT files."
    exit 1
fi

if [ ! -s "${PATH_OUTPUT}" ]; then
    echo "ERROR: lambda-path output was not created or is empty: ${PATH_OUTPUT}"
    exit 1
fi

echo
echo "Done."
echo "Predictions: ${PRED_OUTPUT}"
echo "Lambda path: ${PATH_OUTPUT}"
