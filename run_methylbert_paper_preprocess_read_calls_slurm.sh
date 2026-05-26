#!/bin/bash
#SBATCH --job-name=mbert_rc
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --output=logs/methylbert_preprocess_read_calls_%j.out
#SBATCH --error=logs/methylbert_preprocess_read_calls_%j.err

set -euo pipefail

export METHYLBERT_STEP=PREPROCESS
source scripts/methylbert_common.sh
bootstrap_methylbert_job
source scripts/methylbert_reference.sh

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
READ_CALL_PREPROCESS_DIR="${READ_CALL_PREPROCESS_DIR:-${METHYLBERT_WORK_DIR}/preprocess_taps_read_calls}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv}"

MIN_INFORMATIVE="${MIN_INFORMATIVE:-2}"
MAX_READS_PER_SAMPLE="${MAX_READS_PER_SAMPLE:-200000}"
MAX_READS_PER_LABEL="${MAX_READS_PER_LABEL:-500000}"
READ_CALL_MODE="${READ_CALL_MODE:-overlap}"
DMR_START_BASE="${DMR_START_BASE:-1}"

mkdir -p logs "${READ_CALL_PREPROCESS_DIR}"

if [ ! -s "${METHYLBERT_DMRS}" ]; then
    echo "Missing DMRs: ${METHYLBERT_DMRS}" >&2
    exit 1
fi

stage_methylbert_reference

if [ ! -s "${METHYLBERT_REF_FASTA}" ]; then
    echo "Missing staged/reference FASTA: ${METHYLBERT_REF_FASTA}" >&2
    exit 1
fi
if [ ! -s "${METHYLBERT_REF_FASTA}.fai" ]; then
    echo "Missing FASTA index: ${METHYLBERT_REF_FASTA}.fai" >&2
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required." >&2
    exit 1
fi

args=(
    scripts/preprocess_methylbert_taps_read_calls.py
    --dmrs "${METHYLBERT_DMRS}"
    --reference "${METHYLBERT_REF_FASTA}"
    --output-dir "${READ_CALL_PREPROCESS_DIR}"
    --mode "${READ_CALL_MODE}"
    --dmr-start-base "${DMR_START_BASE}"
    --min-informative "${MIN_INFORMATIVE}"
    --max-reads-per-sample "${MAX_READS_PER_SAMPLE}"
    --max-reads-per-label "${MAX_READS_PER_LABEL}"
)

if [ -n "${METHYLBERT_READ_CALL_SAMPLE_SHEET:-}" ]; then
    args+=(--sample-sheet "${METHYLBERT_READ_CALL_SAMPLE_SHEET}")
else
    if [ -n "${METHYLBERT_DMR_TUMOUR_READ_CALL_LIST:-}" ]; then
        args+=(--tumour-list "${METHYLBERT_DMR_TUMOUR_READ_CALL_LIST}")
    fi
    if [ -n "${METHYLBERT_DMR_NORMAL_READ_CALL_LIST:-}" ]; then
        args+=(--normal-list "${METHYLBERT_DMR_NORMAL_READ_CALL_LIST}")
    fi
fi

echo "=== MethylBERT TAPS read-call preprocessing ==="
echo "READ_CALL_PREPROCESS_DIR=${READ_CALL_PREPROCESS_DIR}"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
echo "METHYLBERT_REF_FASTA=${METHYLBERT_REF_FASTA}"
echo "READ_CALL_MODE=${READ_CALL_MODE}"
echo "MIN_INFORMATIVE=${MIN_INFORMATIVE}"
"${PYTHON_BIN}" "${args[@]}"

test -s "${READ_CALL_PREPROCESS_DIR}/train_seq.csv"
test -s "${READ_CALL_PREPROCESS_DIR}/test_seq.csv"
echo "Done."
