#!/bin/bash
# Submit PAT/DSS DMR discovery as prepare -> chromosome array -> merge.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
export PROJECT_DIR
cd "${PROJECT_DIR}"

if [ -s "${PROJECT_DIR}/configs/methylbert_oac_paper_bmrc.env" ]; then
    source "${PROJECT_DIR}/configs/methylbert_oac_paper_bmrc.env"
else
    export RUN_LABEL="${RUN_LABEL:-OAC_methylbert_paper_bmrc}"
    export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/run_v0.5_methylbert_bmrc}"
    export METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
    export METHYLBERT_DMR_TUMOUR_PAT_LIST="${METHYLBERT_DMR_TUMOUR_PAT_LIST:-${PROJECT_DIR}/data/methylbert/oac_dmr_tumour_pats.list}"
    export METHYLBERT_DMR_NORMAL_PAT_LIST="${METHYLBERT_DMR_NORMAL_PAT_LIST:-${PROJECT_DIR}/data/methylbert/oac_dmr_normal_pats.list}"
fi

if [ -z "${METHYLBERT_DMR_TUMOUR_PAT_LIST:-}" ] || [ ! -s "${METHYLBERT_DMR_TUMOUR_PAT_LIST}" ]; then
    echo "Missing METHYLBERT_DMR_TUMOUR_PAT_LIST: ${METHYLBERT_DMR_TUMOUR_PAT_LIST:-unset}" >&2
    exit 1
fi
if [ -z "${METHYLBERT_DMR_NORMAL_PAT_LIST:-}" ] || [ ! -s "${METHYLBERT_DMR_NORMAL_PAT_LIST}" ]; then
    echo "Missing METHYLBERT_DMR_NORMAL_PAT_LIST: ${METHYLBERT_DMR_NORMAL_PAT_LIST:-unset}" >&2
    exit 1
fi

mkdir -p logs

array_range="${METHYLBERT_DMR_CHROM_ARRAY:-1-22}"
chr_mem="${METHYLBERT_DMR_CHR_MEM:-256G}"
chr_time="${METHYLBERT_DMR_CHR_TIME:-24:00:00}"

prepare_jid="$(sbatch --parsable --export=ALL run_methylbert_paper_dmr_pat_prepare_slurm.sh)"
prepare_dep="${prepare_jid%%;*}"

chrom_jid="$(
    sbatch \
        --parsable \
        --dependency="afterok:${prepare_dep}" \
        --array="${array_range}" \
        --mem="${chr_mem}" \
        --time="${chr_time}" \
        --export=ALL \
        run_methylbert_paper_dmr_pat_chr_slurm.sh
)"
chrom_dep="${chrom_jid%%;*}"

merge_jid="$(
    sbatch \
        --parsable \
        --dependency="afterok:${chrom_dep}" \
        --export=ALL \
        run_methylbert_paper_dmr_pat_merge_slurm.sh
)"

echo "prepare_jid=${prepare_jid}"
echo "chrom_array_jid=${chrom_jid}"
echo "merge_jid=${merge_jid}"
echo "chrom_array=${array_range}"
echo "chrom_mem=${chr_mem}"
echo "chrom_time=${chr_time}"
echo "METHYLBERT_DMR_TUMOUR_PAT_LIST=${METHYLBERT_DMR_TUMOUR_PAT_LIST}"
echo "METHYLBERT_DMR_NORMAL_PAT_LIST=${METHYLBERT_DMR_NORMAL_PAT_LIST}"
echo "DMR_MAX_BACKGROUND_PER_COHORT=${DMR_MAX_BACKGROUND_PER_COHORT:-0}"
