#!/bin/bash
#SBATCH --job-name=mbert_rcm
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/methylbert_merge_read_call_shards_%j.out
#SBATCH --error=logs/methylbert_merge_read_call_shards_%j.err

set -euo pipefail

export METHYLBERT_STEP=PREPROCESS_PAT
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
READ_CALL_SHARD_DIR="${READ_CALL_SHARD_DIR:-${METHYLBERT_WORK_DIR}/preprocess_taps_read_call_shards}"
READ_CALL_PREPROCESS_DIR="${READ_CALL_PREPROCESS_DIR:-${METHYLBERT_WORK_DIR}/preprocess_taps_read_calls_collapsed_100kb}"
MAX_READS_PER_LABEL="${MAX_READS_PER_LABEL:-500000}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
SPLIT_BY="${SPLIT_BY:-sample}"

# Optional shortcut-diagnostic controls (see docs/methylbert_shortcut_diagnostics.md).
MERGE_EXTRA_ARGS=()
[ "${SHUFFLE_LABELS:-0}" = "1" ] && MERGE_EXTRA_ARGS+=(--shuffle-labels)
[ "${LENGTH_MATCH:-0}" = "1" ] && MERGE_EXTRA_ARGS+=(--length-match --length-match-bin "${LENGTH_MATCH_BIN:-10}")
[ -n "${MIN_READ_LENGTH:-}" ] && MERGE_EXTRA_ARGS+=(--min-read-length "${MIN_READ_LENGTH}")
[ -n "${MAX_READ_LENGTH_FILTER:-}" ] && MERGE_EXTRA_ARGS+=(--max-read-length "${MAX_READ_LENGTH_FILTER}")
[ -n "${HOLDOUT_SAMPLES:-}" ] && MERGE_EXTRA_ARGS+=(--holdout-samples "${HOLDOUT_SAMPLES}")
[ -n "${HOLDOUT_COHORT:-}" ] && MERGE_EXTRA_ARGS+=(--holdout-cohort "${HOLDOUT_COHORT}")
[ "${BALANCE_COHORTS:-0}" = "1" ] && MERGE_EXTRA_ARGS+=(--balance-cohorts)
[ "${BALANCE_LABELS:-0}" = "1" ] && MERGE_EXTRA_ARGS+=(--balance-labels)

mkdir -p logs "${READ_CALL_PREPROCESS_DIR}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required." >&2
    exit 1
fi

echo "=== MethylBERT TAPS read-call shard merge ==="
echo "READ_CALL_SHARD_DIR=${READ_CALL_SHARD_DIR}"
echo "READ_CALL_PREPROCESS_DIR=${READ_CALL_PREPROCESS_DIR}"
echo "MAX_READS_PER_LABEL=${MAX_READS_PER_LABEL}"
echo "SPLIT_BY=${SPLIT_BY}"
echo "MERGE_EXTRA_ARGS=${MERGE_EXTRA_ARGS[*]:-}"

"${PYTHON_BIN}" scripts/merge_methylbert_read_call_shards.py \
    --shard-dir "${READ_CALL_SHARD_DIR}" \
    --output-dir "${READ_CALL_PREPROCESS_DIR}" \
    --max-reads-per-label "${MAX_READS_PER_LABEL}" \
    --split-ratio "${SPLIT_RATIO}" \
    --split-by "${SPLIT_BY}" \
    ${MERGE_EXTRA_ARGS[@]+"${MERGE_EXTRA_ARGS[@]}"}

test -s "${READ_CALL_PREPROCESS_DIR}/train_seq.csv"
test -s "${READ_CALL_PREPROCESS_DIR}/test_seq.csv"
echo "Done."
