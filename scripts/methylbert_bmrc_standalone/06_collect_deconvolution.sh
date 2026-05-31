#!/bin/bash
set -euo pipefail

# Collect per-sample deconvolution.csv files into one summary table.

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
export METHYLBERT_COLLECT_MODULES="Python/3.11.3-GCCcore-12.3.0"

export DECONV_DIR="${METHYLBERT_WORK_DIR}/deconvolution_taps_read_calls_${VARIANT}_adjusted"
export SUMMARY_OUTPUT="${DECONV_DIR}/deconvolution_summary.csv"

test -d "${DECONV_DIR}"

echo "Submitting deconvolution collection"
echo "SUMMARY_OUTPUT=${SUMMARY_OUTPUT}"

sbatch --parsable \
  --export=ALL \
  run_methylbert_paper_collect_slurm.sh
