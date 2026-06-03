#!/bin/bash
set -euo pipefail

# Validate the deconvolution tumour fraction against ichorCNA (orthogonal, CNA-based) and
# against fragmentomics features. Run after 06. See RUNBOOK.md.

export PROJECT_DIR="${PROJECT_DIR:-/gpfs3/well/ludwig/users/uii408/tapestry}"
cd "${PROJECT_DIR}"

export RUN_LABEL="OAC_methylbert_paper_bmrc"
export OUTPUT_DIR="/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.5_methylbert_bmrc"
export METHYLBERT_WORK_DIR="${OUTPUT_DIR}/methylbert/${RUN_LABEL}"
# VARIANT must match the run being validated.
export VARIANT="${VARIANT:-collapsed_100kb_balanced}"
export METHYLBERT_R_LIBS="${HOME}/R/library"
export METHYLBERT_ENV_COMMAND='export R_LIBS_USER="${METHYLBERT_R_LIBS}"'
export METHYLBERT_DEACTIVATE_CONDA="1"
unset METHYLBERT_CONFIG
unset METHYLBERT_STEP_MODULES

export METHYLBERT_MODULES=""
export METHYLBERT_COLLECT_MODULES="Python/3.11.3-GCCcore-12.3.0"

export DECONV_DIR="${METHYLBERT_WORK_DIR}/deconvolution_taps_read_calls_${VARIANT}_adjusted"
export SUMMARY_OUTPUT="${DECONV_DIR}/deconvolution_summary.csv"
export ICHORCNA_FILE="${ICHORCNA_FILE:-${PROJECT_DIR}/data/cfDNA_tumour_fraction_ichorCNA.json}"
export VALIDATE_DIR="${DECONV_DIR}/validation"
export ESTIMATE_COL="T"

test -s "${SUMMARY_OUTPUT}"

echo "Submitting validation vs ichorCNA"
echo "SUMMARY_OUTPUT=${SUMMARY_OUTPUT}"
echo "VALIDATE_DIR=${VALIDATE_DIR}"

sbatch --parsable \
  --export=ALL \
  run_methylbert_paper_validate_slurm.sh
