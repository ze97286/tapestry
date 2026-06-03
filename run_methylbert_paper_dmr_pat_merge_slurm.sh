#!/bin/bash
#SBATCH --job-name=mbert_dmr_merge
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/methylbert_dmr_pat_merge_%j.out
#SBATCH --error=logs/methylbert_dmr_pat_merge_%j.err

set -euo pipefail

export METHYLBERT_STEP=DMR
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DMR_DIR="${DMR_DIR:-${METHYLBERT_WORK_DIR}/dmr_pat}"
DMR_BY_CHROM_DIR="${DMR_BY_CHROM_DIR:-${DMR_DIR}/by_chrom}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.tsv}"
METHYLBERT_DMRS_BED="${METHYLBERT_DMRS_BED:-${METHYLBERT_DMRS%.tsv}.bed}"
DMR_TOP_N="${DMR_TOP_N:-100}"

if ! command -v python >/dev/null 2>&1; then
    echo "python is required for DMR merge" >&2
    exit 1
fi

python scripts/merge_dss_dmrs.py \
    --input-dir "${DMR_BY_CHROM_DIR}" \
    --output-all "${DMR_DIR}/dss_dmrs.tsv" \
    --output-top "${METHYLBERT_DMRS}" \
    --top-n "${DMR_TOP_N}" \
    --target-group T

python scripts/dss_dmrs_to_bed.py \
    --input "${METHYLBERT_DMRS}" \
    --output "${METHYLBERT_DMRS_BED}" \
    --start-base "${DMR_POSITION_START_BASE:-1}"

echo "Done."
echo "All DMRs: ${DMR_DIR}/dss_dmrs.tsv"
echo "Selected DMRs: ${METHYLBERT_DMRS}"
echo "Selected DMR BED: ${METHYLBERT_DMRS_BED}"
