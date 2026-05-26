#!/bin/bash
#SBATCH --job-name=mbert_ev
#SBATCH --account=gpu_ludwig.prj
#SBATCH --qos=gpu_bmrc_4hr
#SBATCH --partition=gpu_p100_16gb,gpu_v100_32gb,gpu_rtx8000_48gb,gpu_a100_40gb,gpu_a100_80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=16G
#SBATCH --time=03:59:00
#SBATCH --output=logs/methylbert_eval_%j.out
#SBATCH --error=logs/methylbert_eval_%j.err

set -euo pipefail

export METHYLBERT_STEP=FINETUNE
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_DIR="${METHYLBERT_DIR:-${PROJECT_DIR}/external/methylbert}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
PREPROCESS_DIR="${PREPROCESS_DIR:-${METHYLBERT_WORK_DIR}/preprocess}"
MODEL_DIR="${MODEL_DIR:-${METHYLBERT_WORK_DIR}/model}"
EVAL_DIR="${EVAL_DIR:-${MODEL_DIR}/heldout_eval}"

N_ENCODER="${N_ENCODER:-12}"
SEQ_LEN="${SEQ_LEN:-150}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-2}}"
WITH_CUDA="${WITH_CUDA:-1}"

mkdir -p logs "${EVAL_DIR}"

if [ ! -s "${PREPROCESS_DIR}/test_seq.csv" ]; then
    echo "Missing test dataset: ${PREPROCESS_DIR}/test_seq.csv" >&2
    exit 1
fi
if [ ! -d "${MODEL_DIR}/bert.model" ]; then
    echo "Missing model directory: ${MODEL_DIR}/bert.model" >&2
    exit 1
fi

export PYTHONPATH="${METHYLBERT_DIR}/src:${PYTHONPATH:-}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required for MethylBERT evaluation." >&2
    exit 1
fi

args=(
    scripts/evaluate_methylbert_finetune.py
    --test-dataset "${PREPROCESS_DIR}/test_seq.csv"
    --model-dir "${MODEL_DIR}"
    --output-dir "${EVAL_DIR}"
    --n-encoder "${N_ENCODER}"
    --seq-len "${SEQ_LEN}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
)

if [ "${WITH_CUDA}" = "1" ]; then
    args+=(--with-cuda)
fi

echo "=== MethylBERT held-out evaluation ==="
echo "PREPROCESS_DIR=${PREPROCESS_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "EVAL_DIR=${EVAL_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" "${args[@]}"

test -s "${EVAL_DIR}/summary.json"
test -s "${EVAL_DIR}/summary_by_label.tsv"
test -s "${EVAL_DIR}/test_predictions.tsv"
echo "Done."
