#!/bin/bash
set -euo pipefail

# Prepare inputs for the read-call workflow: build the reproducible collapsed_100kb DMR
# panel from the DSS calls and (optionally) the read-call sample sheet + bulk deconvolution
# sheet. Run this once before 01. See RUNBOOK.md.

export PROJECT_DIR="${PROJECT_DIR:-/gpfs3/well/ludwig/users/uii408/tapestry}"
cd "${PROJECT_DIR}"

export RUN_LABEL="OAC_methylbert_paper_bmrc"
export OUTPUT_DIR="/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.5_methylbert_bmrc"
export METHYLBERT_WORK_DIR="${OUTPUT_DIR}/methylbert/${RUN_LABEL}"
export METHYLBERT_R_LIBS="${HOME}/R/library"
export METHYLBERT_ENV_COMMAND='export R_LIBS_USER="${METHYLBERT_R_LIBS}"'
export METHYLBERT_DEACTIVATE_CONDA="1"
unset METHYLBERT_CONFIG
unset METHYLBERT_STEP_MODULES

export METHYLBERT_MODULES=""
export METHYLBERT_COLLECT_MODULES="Python/3.11.3-GCCcore-12.3.0"

export DSS_DMRS="${DSS_DMRS:-${METHYLBERT_WORK_DIR}/dmr_pat/dss_dmrs.tsv}"
export METHYLBERT_DMRS="${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv"

# To also (re)build the read-call sample sheet and bulk sheet, set BUILD_READ_CALL_LISTS=1
# and the per-read-call directory variables before running (see RUNBOOK.md):
#   export BUILD_READ_CALL_LISTS=1
#   export TUMOUR_SAMPLE_LIST=data/methylbert/oac_dmr_tumour_pats.list
#   export NORMAL_SAMPLE_LIST=data/methylbert/oac_dmr_normal_pats.list
#   export TUMOUR_READ_CALL_DIR=... AB_READ_CALL_DIR=... CD_READ_CALL_DIR=...
#   export BULK_SAMPLE_LIST=...   BULK_READ_CALL_DIR=...

echo "Submitting input preparation"
echo "DSS_DMRS=${DSS_DMRS}"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"

sbatch --parsable \
  --export=ALL \
  run_methylbert_paper_prepare_inputs_slurm.sh
