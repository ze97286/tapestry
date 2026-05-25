#!/bin/bash
#SBATCH --job-name=mbert_ft
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/methylbert_finetune_%j.out
#SBATCH --error=logs/methylbert_finetune_%j.err

set -euo pipefail

export METHYLBERT_STEP=FINETUNE
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_DIR="${METHYLBERT_DIR:-${PROJECT_DIR}/external/methylbert}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
PREPROCESS_DIR="${PREPROCESS_DIR:-${METHYLBERT_WORK_DIR}/preprocess}"
MODEL_DIR="${MODEL_DIR:-${METHYLBERT_WORK_DIR}/model}"

N_ENCODER="${N_ENCODER:-12}"
SEQ_LEN="${SEQ_LEN:-150}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
STEPS="${STEPS:-600}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_FREQ="${LOG_FREQ:-10}"
EVAL_FREQ="${EVAL_FREQ:-10}"
WARM_UP="${WARM_UP:-100}"
DECREASE_STEPS="${DECREASE_STEPS:-200}"
LR="${LR:-4e-4}"
LOSS="${LOSS:-bce}"
WITH_CUDA="${WITH_CUDA:-1}"

mkdir -p logs "${MODEL_DIR}"

if [ ! -s "${PREPROCESS_DIR}/train_seq.csv" ]; then
    echo "Missing train dataset: ${PREPROCESS_DIR}/train_seq.csv" >&2
    exit 1
fi
if [ ! -s "${PREPROCESS_DIR}/test_seq.csv" ]; then
    echo "Missing test dataset: ${PREPROCESS_DIR}/test_seq.csv" >&2
    exit 1
fi

export PYTHONPATH="${METHYLBERT_DIR}/src:${PYTHONPATH:-}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required for MethylBERT fine-tuning. On BMRC, load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 and Transformers/4.39.3-gfbf-2023a or set METHYLBERT_FINETUNE_MODULES." >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = ["numpy", "pandas", "sklearn", "torch", "tqdm", "transformers"]
missing = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name} ({exc})")

if missing:
    sys.stderr.write(
        "Missing Python modules for MethylBERT fine-tuning after module load:\n"
        + "\n".join(f"  - {entry}" for entry in missing)
        + "\nSet METHYLBERT_FINETUNE_MODULES or METHYLBERT_BMRC_FINETUNE_MODULES to a BMRC module stack that provides them.\n"
    )
    raise SystemExit(1)
PY

args=(
    scripts/run_methylbert_finetune_direct.py
    --train_dataset "${PREPROCESS_DIR}/train_seq.csv"
    --test_dataset "${PREPROCESS_DIR}/test_seq.csv"
    --output_path "${MODEL_DIR}"
    --n_encoder "${N_ENCODER}"
    --n_mers 3
    --seq_len "${SEQ_LEN}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --steps "${STEPS}"
    --num_workers "${NUM_WORKERS}"
    --log_freq "${LOG_FREQ}"
    --eval_freq "${EVAL_FREQ}"
    --warm_up "${WARM_UP}"
    --decrease_steps "${DECREASE_STEPS}"
    --lr "${LR}"
    --loss "${LOSS}"
)
if [ "${WITH_CUDA}" = "1" ]; then
    args+=(--with_cuda)
fi

echo "=== MethylBERT paper-style fine-tuning ==="
echo "PREPROCESS_DIR=${PREPROCESS_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "N_ENCODER=${N_ENCODER}"
echo "STEPS=${STEPS}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "LR=${LR}"
"${PYTHON_BIN}" "${args[@]}"

test -s "${MODEL_DIR}/train_param.txt"
test -d "${MODEL_DIR}/bert.model"
echo "Done."
echo "Model: ${MODEL_DIR}/bert.model"
