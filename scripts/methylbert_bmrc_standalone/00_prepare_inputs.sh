#!/bin/bash
set -euo pipefail

# Prepare inputs for the read-call workflow: build the reproducible collapsed_100kb DMR
# panel from the DSS calls and rebuild the balanced read-call training sample sheet.
# Run this once before 01. See RUNBOOK.md.

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

export BUILD_READ_CALL_LISTS="${BUILD_READ_CALL_LISTS:-1}"
export BALANCE_COHORTS="${BALANCE_COHORTS:-1}"
export DMR_MAX_BACKGROUND_PER_COHORT="${DMR_MAX_BACKGROUND_PER_COHORT:-4}"
export READ_CALL_MAX_NORMAL_PER_COHORT="${READ_CALL_MAX_NORMAL_PER_COHORT:-${DMR_MAX_BACKGROUND_PER_COHORT}}"

export TUMOUR_SAMPLE_LIST="${TUMOUR_SAMPLE_LIST:-${PROJECT_DIR}/data/methylbert/oac_dmr_tumour_pats.list}"
export NORMAL_SAMPLE_LIST="${NORMAL_SAMPLE_LIST:-${PROJECT_DIR}/data/methylbert/oac_dmr_normal_pats.list}"

export TUMOUR_READ_CALL_DIR="${TUMOUR_READ_CALL_DIR:-/well/ludwig/users/benjamin/OAC_Immuno_Trial/TAPS_Tissue/PerReadCalls}"
export AB_READ_CALL_DIR="${AB_READ_CALL_DIR:-/well/ludwig2/projects/processed/Lu_lab/OAC_immuno_Trial/TAPS_cfDNA/Results/1.7/PerReadCalls}"
export CD_READ_CALL_DIR="${CD_READ_CALL_DIR:-/well/ludwig2/projects/processed/Lu_lab/OAC_immuno_Trial/TAPS_cfDNA/CD/Results/1.1/PerReadCalls}"
export READ_CALL_LISTS_DIR="${READ_CALL_LISTS_DIR:-${METHYLBERT_WORK_DIR}/read_call_lists}"

echo "Submitting input preparation"
echo "DSS_DMRS=${DSS_DMRS}"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
echo "BUILD_READ_CALL_LISTS=${BUILD_READ_CALL_LISTS}"
echo "BALANCE_COHORTS=${BALANCE_COHORTS}"
echo "READ_CALL_MAX_NORMAL_PER_COHORT=${READ_CALL_MAX_NORMAL_PER_COHORT}"
echo "TUMOUR_SAMPLE_LIST=${TUMOUR_SAMPLE_LIST}"
echo "NORMAL_SAMPLE_LIST=${NORMAL_SAMPLE_LIST}"
echo "READ_CALL_LISTS_DIR=${READ_CALL_LISTS_DIR}"

sbatch --parsable \
  --export=ALL \
  run_methylbert_paper_prepare_inputs_slurm.sh
