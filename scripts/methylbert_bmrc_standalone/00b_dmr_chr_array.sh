#!/bin/bash
set -euo pipefail

# Step 00b: run DSS DMR calling per chromosome after 00a has completed.

export PROJECT_DIR="${PROJECT_DIR:-/gpfs3/well/ludwig/users/uii408/tapestry}"
cd "${PROJECT_DIR}"

if [ -s "${PROJECT_DIR}/configs/methylbert_oac_paper_bmrc.env" ]; then
    source "${PROJECT_DIR}/configs/methylbert_oac_paper_bmrc.env"
fi

export RUN_LABEL="OAC_methylbert_paper_bmrc"
export OUTPUT_DIR="/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.5_methylbert_bmrc"
export METHYLBERT_WORK_DIR="${OUTPUT_DIR}/methylbert/${RUN_LABEL}"

export DMR_REGION_MODE="${DMR_REGION_MODE:-balanced_ab_cd}"
case "${DMR_REGION_MODE}" in
    balanced_ab_cd)
        export DMR_BACKGROUND_COHORTS="${DMR_BACKGROUND_COHORTS:-AB_plasma,CD_plasma}"
        export DMR_MAX_BACKGROUND_PER_COHORT="${DMR_MAX_BACKGROUND_PER_COHORT:-4}"
        ;;
    ab_only)
        export DMR_BACKGROUND_COHORTS="${DMR_BACKGROUND_COHORTS:-AB_plasma}"
        export DMR_MAX_BACKGROUND_PER_COHORT="${DMR_MAX_BACKGROUND_PER_COHORT:-4}"
        ;;
    *)
        echo "Unsupported DMR_REGION_MODE=${DMR_REGION_MODE}; use balanced_ab_cd or ab_only" >&2
        exit 1
        ;;
esac

export DMR_DIR="${METHYLBERT_WORK_DIR}/dmr_pat_${DMR_REGION_MODE}"
export DMR_BY_CHROM_DIR="${DMR_DIR}/by_chrom"
export DSS_SPLIT_DIR="${DMR_DIR}/counts_by_chrom"
export DMR_COUNT_DIR="${METHYLBERT_WORK_DIR}/dmr_pat_counts"
export METHYLBERT_DMRS="${METHYLBERT_WORK_DIR}/dmrs_top100.${DMR_REGION_MODE}.tsv"
export METHYLBERT_DMRS_BED="${METHYLBERT_WORK_DIR}/dmrs_top100.${DMR_REGION_MODE}.bed"
export CHROMOSOMES_FILE="${DMR_DIR}/chromosomes.txt"
export SAMPLE_SHEET_BY_CHROM="${DMR_DIR}/dss_samples_by_chrom.tsv"

test -s "${CHROMOSOMES_FILE}"
test -s "${SAMPLE_SHEET_BY_CHROM}"

N_CHROM="$(wc -l < "${CHROMOSOMES_FILE}")"
echo "Submitting DMR chromosome array"
echo "DMR_REGION_MODE=${DMR_REGION_MODE}"
echo "CHROMOSOMES_FILE=${CHROMOSOMES_FILE}"
echo "N_CHROM=${N_CHROM}"

sbatch --parsable \
    --array=1-"${N_CHROM}" \
    --mem="${METHYLBERT_DMR_CHR_MEM:-256G}" \
    --time="${METHYLBERT_DMR_CHR_TIME:-24:00:00}" \
    --export=ALL \
    run_methylbert_paper_dmr_pat_chr_slurm.sh
