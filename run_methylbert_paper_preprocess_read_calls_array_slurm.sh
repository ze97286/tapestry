#!/bin/bash
#SBATCH --job-name=mbert_rc
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=03:59:00
#SBATCH --output=logs/methylbert_preprocess_read_calls_array_%A_%a.out
#SBATCH --error=logs/methylbert_preprocess_read_calls_array_%A_%a.err

set -euo pipefail

export METHYLBERT_STEP=PREPROCESS_PAT
source scripts/methylbert_common.sh
bootstrap_methylbert_job
source scripts/methylbert_reference.sh

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
READ_CALL_SAMPLE_SHEET="${READ_CALL_SAMPLE_SHEET:-${METHYLBERT_WORK_DIR}/read_call_lists/oac_dmr_read_calls.sample_sheet.tsv}"
READ_CALL_SHARD_DIR="${READ_CALL_SHARD_DIR:-${METHYLBERT_WORK_DIR}/preprocess_taps_read_call_shards}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv}"

MIN_INFORMATIVE="${MIN_INFORMATIVE:-2}"
MAX_READS_PER_SAMPLE="${MAX_READS_PER_SAMPLE:-200000}"
STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE="${STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE:-0}"
READ_CALL_MODE="${READ_CALL_MODE:-contained}"
DMR_START_BASE="${DMR_START_BASE:-1}"

# Optional ablation controls (see docs/methylbert_shortcut_diagnostics.md).
PREPROCESS_ABLATION_ARGS=()
[ "${BLANK_METHYL:-0}" = "1" ] && PREPROCESS_ABLATION_ARGS+=(--blank-methyl)
[ "${BLANK_DNA:-0}" = "1" ] && PREPROCESS_ABLATION_ARGS+=(--blank-dna)
[ "${COLLAPSE_DMR_LABEL:-0}" = "1" ] && PREPROCESS_ABLATION_ARGS+=(--collapse-dmr-label)

mkdir -p logs "${READ_CALL_SHARD_DIR}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Submit this script with --array=1-N, where N is the sample-sheet row count." >&2
    exit 1
fi
if [ ! -s "${READ_CALL_SAMPLE_SHEET}" ]; then
    echo "Missing read-call sample sheet: ${READ_CALL_SAMPLE_SHEET}" >&2
    exit 1
fi
if [ ! -s "${METHYLBERT_DMRS}" ]; then
    echo "Missing DMRs: ${METHYLBERT_DMRS}" >&2
    exit 1
fi

METHYLBERT_REF_STAGE_DIR="${METHYLBERT_REF_STAGE_DIR:-${METHYLBERT_WORK_DIR}/reference_stage}"
export METHYLBERT_REF_STAGE_DIR
stage_methylbert_reference

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required." >&2
    exit 1
fi

SHARD_NAME="$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")"
SHARD_DIR="${READ_CALL_SHARD_DIR}/${SHARD_NAME}"
mkdir -p "${SHARD_DIR}"
SHARD_SAMPLE_SHEET="${SHARD_DIR}/sample_sheet.tsv"

awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR == idx { print; found=1 } END { exit(found ? 0 : 1) }' \
    "${READ_CALL_SAMPLE_SHEET}" > "${SHARD_SAMPLE_SHEET}"

echo "=== MethylBERT TAPS read-call preprocessing shard ==="
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "SHARD_DIR=${SHARD_DIR}"
echo "SAMPLE=$(cat "${SHARD_SAMPLE_SHEET}")"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
echo "METHYLBERT_REF_FASTA=${METHYLBERT_REF_FASTA}"
echo "READ_CALL_MODE=${READ_CALL_MODE}"
echo "MIN_INFORMATIVE=${MIN_INFORMATIVE}"
echo "MAX_READS_PER_SAMPLE=${MAX_READS_PER_SAMPLE}"
echo "STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE=${STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE}"

"${PYTHON_BIN}" scripts/preprocess_methylbert_taps_read_calls.py \
    --sample-sheet "${SHARD_SAMPLE_SHEET}" \
    --dmrs "${METHYLBERT_DMRS}" \
    --reference "${METHYLBERT_REF_FASTA}" \
    --output-dir "${SHARD_DIR}" \
    --rows-output "${SHARD_DIR}/rows.tsv" \
    --summary-output "${SHARD_DIR}/read_call_preprocess_summary.tsv" \
    --mode "${READ_CALL_MODE}" \
    --dmr-start-base "${DMR_START_BASE}" \
    --min-informative "${MIN_INFORMATIVE}" \
    --max-reads-per-sample "${MAX_READS_PER_SAMPLE}" \
    --stop-after-output-rows-per-sample "${STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE}" \
    --max-reads-per-label 0 \
    ${PREPROCESS_ABLATION_ARGS[@]+"${PREPROCESS_ABLATION_ARGS[@]}"}

test -s "${SHARD_DIR}/rows.tsv"
echo "Done."
