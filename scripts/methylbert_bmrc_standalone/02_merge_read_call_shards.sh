#!/bin/bash
set -euo pipefail

# Merge previously completed read-call preprocessing shards.
# This does not depend on Slurm job IDs; run it manually after shard jobs finish.

export PROJECT_DIR="${PROJECT_DIR:-/gpfs3/well/ludwig/users/uii408/tapestry}"
cd "${PROJECT_DIR}"

export RUN_LABEL="OAC_methylbert_paper_bmrc"
export OUTPUT_DIR="/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.5_methylbert_bmrc"
export METHYLBERT_WORK_DIR="${OUTPUT_DIR}/methylbert/${RUN_LABEL}"
# VARIANT suffixes all output dirs so diagnostic runs do not clobber the main run.
export VARIANT="${VARIANT:-contained_sample_split}"
export METHYLBERT_R_LIBS="${HOME}/R/library"
export METHYLBERT_ENV_COMMAND='export R_LIBS_USER="${METHYLBERT_R_LIBS}"'
export METHYLBERT_DEACTIVATE_CONDA="1"
unset METHYLBERT_CONFIG
unset METHYLBERT_STEP_MODULES

export METHYLBERT_MODULES=""
export METHYLBERT_PREPROCESS_PAT_MODULES="Python/3.11.3-GCCcore-12.3.0"

export READ_CALL_SHARD_DIR="${METHYLBERT_WORK_DIR}/preprocess_taps_read_call_shards_${VARIANT}"
export READ_CALL_PREPROCESS_DIR="${METHYLBERT_WORK_DIR}/preprocess_taps_read_calls_${VARIANT}"

export MAX_READS_PER_LABEL="500000"
export SPLIT_RATIO="0.8"
export SPLIT_BY="sample"

test -d "${READ_CALL_SHARD_DIR}"

echo "Merging read-call shards"
echo "READ_CALL_SHARD_DIR=${READ_CALL_SHARD_DIR}"
echo "READ_CALL_PREPROCESS_DIR=${READ_CALL_PREPROCESS_DIR}"

sbatch --parsable \
  --export=ALL \
  run_methylbert_paper_merge_read_call_shards_slurm.sh
