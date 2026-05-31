#!/bin/bash
set -euo pipefail

# Submit read-call preprocessing shards for the contained-read/sample-split rerun.
# This script is deliberately self-contained: run it from the repo checkout on BMRC.

export PROJECT_DIR="${PROJECT_DIR:-/gpfs3/well/ludwig/users/uii408/tapestry}"
cd "${PROJECT_DIR}"

export RUN_LABEL="OAC_methylbert_paper_bmrc"
export OUTPUT_DIR="/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.5_methylbert_bmrc"
export METHYLBERT_DIR="${PROJECT_DIR}/external/methylbert"
export METHYLBERT_WORK_DIR="${OUTPUT_DIR}/methylbert/${RUN_LABEL}"
# VARIANT suffixes all output dirs so diagnostic runs do not clobber the main run.
export VARIANT="${VARIANT:-contained_sample_split}"

export METHYLBERT_REF_FASTA=""
export METHYLBERT_REF_FASTA_GZ="/well/ludwig/shared/genomes/hg38_full_gatk_HPV_HBV_HCV_spike-ins_v2.fa.gz"
export METHYLBERT_REF_STAGE_DIR="${METHYLBERT_WORK_DIR}/reference_stage"
export METHYLBERT_R_LIBS="${HOME}/R/library"
export METHYLBERT_ENV_COMMAND='export R_LIBS_USER="${METHYLBERT_R_LIBS}"'
export METHYLBERT_DEACTIVATE_CONDA="1"
unset METHYLBERT_CONFIG
unset METHYLBERT_STEP_MODULES

export METHYLBERT_MODULES=""
export METHYLBERT_PREPROCESS_PAT_MODULES="Python/3.11.3-GCCcore-12.3.0 SAMtools/1.18-GCC-12.3.0"

export METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv}"
export READ_CALL_SAMPLE_SHEET="${READ_CALL_SAMPLE_SHEET:-${METHYLBERT_WORK_DIR}/read_call_lists/oac_dmr_read_calls.sample_sheet.tsv}"
export READ_CALL_SHARD_DIR="${METHYLBERT_WORK_DIR}/preprocess_taps_read_call_shards_${VARIANT}"

export READ_CALL_MODE="contained"
export DMR_START_BASE="1"
export MIN_INFORMATIVE="2"
export MAX_READS_PER_SAMPLE="200000"
export STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE="0"

test -s "${READ_CALL_SAMPLE_SHEET}"
test -s "${METHYLBERT_DMRS}"

N=$(wc -l < "${READ_CALL_SAMPLE_SHEET}")
echo "Submitting ${N} read-call preprocessing shards"
echo "READ_CALL_SHARD_DIR=${READ_CALL_SHARD_DIR}"

sbatch --parsable --array=1-"${N}"%130 \
  --export=ALL \
  run_methylbert_paper_preprocess_read_calls_array_slurm.sh
