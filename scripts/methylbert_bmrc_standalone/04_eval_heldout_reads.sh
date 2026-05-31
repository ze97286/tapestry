#!/bin/bash
set -euo pipefail

# Evaluate the fine-tuned read classifier on the held-out sample-split test set.

export PROJECT_DIR="${PROJECT_DIR:-/gpfs3/well/ludwig/users/uii408/tapestry}"
cd "${PROJECT_DIR}"

export RUN_LABEL="OAC_methylbert_paper_bmrc"
export OUTPUT_DIR="/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.5_methylbert_bmrc"
export METHYLBERT_DIR="${PROJECT_DIR}/external/methylbert"
export METHYLBERT_WORK_DIR="${OUTPUT_DIR}/methylbert/${RUN_LABEL}"
# VARIANT suffixes all output dirs so diagnostic runs do not clobber the main run.
export VARIANT="${VARIANT:-contained_sample_split}"
export METHYLBERT_R_LIBS="${HOME}/R/library"
export METHYLBERT_ENV_COMMAND='export R_LIBS_USER="${METHYLBERT_R_LIBS}"'
export METHYLBERT_DEACTIVATE_CONDA="1"
unset METHYLBERT_CONFIG
unset METHYLBERT_STEP_MODULES

export METHYLBERT_MODULES=""
export METHYLBERT_FINETUNE_MODULES="PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 Transformers/4.39.3-gfbf-2023a scikit-learn/1.3.1-gfbf-2023a"

export READ_CALL_PREPROCESS_DIR="${METHYLBERT_WORK_DIR}/preprocess_taps_read_calls_${VARIANT}"
export PREPROCESS_DIR="${READ_CALL_PREPROCESS_DIR}"
export MODEL_DIR="${METHYLBERT_WORK_DIR}/model_taps_read_calls_${VARIANT}"
export EVAL_DIR="${MODEL_DIR}/heldout_eval"

export N_ENCODER="12"
export SEQ_LEN="150"
export EVAL_BATCH_SIZE="256"
export NUM_WORKERS="8"
export WITH_CUDA="1"

test -s "${PREPROCESS_DIR}/test_seq.csv"
test -d "${MODEL_DIR}/bert.model"

echo "Submitting held-out read evaluation"
echo "EVAL_DIR=${EVAL_DIR}"

sbatch --parsable \
  --export=ALL \
  run_methylbert_paper_eval_slurm.sh
