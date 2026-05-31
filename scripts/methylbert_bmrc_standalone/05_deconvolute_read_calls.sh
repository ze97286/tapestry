#!/bin/bash
set -euo pipefail

# Run read-call MethylBERT/MLE deconvolution for cfDNA samples.

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
export METHYLBERT_DECONVOLUTE_MODULES="PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 Transformers/4.39.3-gfbf-2023a scikit-learn/1.3.1-gfbf-2023a SAMtools/1.18-GCC-12.3.0"

export MODEL_DIR="${METHYLBERT_WORK_DIR}/model_taps_read_calls_${VARIANT}"
export DECONV_DIR="${METHYLBERT_WORK_DIR}/deconvolution_taps_read_calls_${VARIANT}_adjusted"
export METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv}"
export METHYLBERT_BULK_READ_CALL_LIST="${METHYLBERT_BULK_READ_CALL_LIST:-${METHYLBERT_WORK_DIR}/read_call_lists/oac_bulk_read_calls.sample_sheet.tsv}"
# Optional: set DECONV_PRIOR_T (e.g. 0.5) to override the training prior at deploy.

export READ_CALL_MODE="contained"
export DMR_START_BASE="1"
export MIN_INFORMATIVE="2"
export READ_CALL_MAX_READS_PER_SAMPLE="0"
export STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE="0"
export DECONV_BATCH_SIZE="128"
export ADJUSTMENT="1"

test -d "${MODEL_DIR}/bert.model"
test -s "${METHYLBERT_DMRS}"
test -s "${METHYLBERT_BULK_READ_CALL_LIST}"

N=$(wc -l < "${METHYLBERT_BULK_READ_CALL_LIST}")
echo "Submitting ${N} deconvolution array tasks"
echo "DECONV_DIR=${DECONV_DIR}"

sbatch --parsable --array=1-"${N}"%130 \
  --export=ALL \
  run_methylbert_paper_deconvolute_read_calls_slurm.sh
