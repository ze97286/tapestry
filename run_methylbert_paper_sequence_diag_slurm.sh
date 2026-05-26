#!/bin/bash
#SBATCH --job-name=mbert_sq
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/methylbert_sequence_diag_%j.out
#SBATCH --error=logs/methylbert_sequence_diag_%j.err

set -euo pipefail

export METHYLBERT_STEP=FINETUNE
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
PREPROCESS_DIR="${PREPROCESS_DIR:-${METHYLBERT_WORK_DIR}/preprocess}"
SEQUENCE_DIAG_DIR="${SEQUENCE_DIAG_DIR:-${PREPROCESS_DIR}/sequence_diagnostics}"

mkdir -p logs "${SEQUENCE_DIAG_DIR}"

if [ ! -s "${PREPROCESS_DIR}/train_seq.csv" ]; then
    echo "Missing train dataset: ${PREPROCESS_DIR}/train_seq.csv" >&2
    exit 1
fi
if [ ! -s "${PREPROCESS_DIR}/test_seq.csv" ]; then
    echo "Missing test dataset: ${PREPROCESS_DIR}/test_seq.csv" >&2
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required for sequence diagnostics." >&2
    exit 1
fi

echo "=== MethylBERT sequence diagnostics ==="
echo "PREPROCESS_DIR=${PREPROCESS_DIR}"
echo "SEQUENCE_DIAG_DIR=${SEQUENCE_DIAG_DIR}"
"${PYTHON_BIN}" scripts/diagnose_methylbert_sequences.py \
    --train "${PREPROCESS_DIR}/train_seq.csv" \
    --test "${PREPROCESS_DIR}/test_seq.csv" \
    --output-dir "${SEQUENCE_DIAG_DIR}"

test -s "${SEQUENCE_DIAG_DIR}/sequence_diagnostics_summary.json"
test -s "${SEQUENCE_DIAG_DIR}/sequence_diagnostics_by_label.tsv"
echo "Done."
