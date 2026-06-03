#!/bin/bash
#SBATCH --job-name=mbert_dmr_chr
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/methylbert_dmr_pat_chr_%A_%a.out
#SBATCH --error=logs/methylbert_dmr_pat_chr_%A_%a.err

set -euo pipefail

# The DMR pipeline gets its toolchain (Rscript, DSS) from conda; keep conda active
# rather than deactivating it (a module stack does not provide DSS here).
export METHYLBERT_STEP=DMR
export METHYLBERT_DEACTIVATE_CONDA=0
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DMR_DIR="${DMR_DIR:-${METHYLBERT_WORK_DIR}/dmr_pat}"
SAMPLE_SHEET_BY_CHROM="${SAMPLE_SHEET_BY_CHROM:-${DMR_DIR}/dss_samples_by_chrom.tsv}"
CHROMOSOMES_FILE="${CHROMOSOMES_FILE:-${DMR_DIR}/chromosomes.txt}"
DMR_BY_CHROM_DIR="${DMR_BY_CHROM_DIR:-${DMR_DIR}/by_chrom}"

DMR_TOP_N="${DMR_TOP_N:-100}"
DMR_DELTA="${DMR_DELTA:-0.2}"
DMR_P_THRESHOLD="${DMR_P_THRESHOLD:-0.05}"
DMR_MIN_CPG="${DMR_MIN_CPG:-4}"
DMR_MIN_LEN="${DMR_MIN_LEN:-50}"
DMR_MERGE_DISTANCE="${DMR_MERGE_DISTANCE:-50}"

if [ -n "${DSS_CHROM:-}" ]; then
    chrom="${DSS_CHROM}"
else
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "Set DSS_CHROM or run as a Slurm array using ${CHROMOSOMES_FILE}" >&2
        exit 1
    fi
    if [ ! -s "${CHROMOSOMES_FILE}" ]; then
        echo "Missing chromosome list: ${CHROMOSOMES_FILE}" >&2
        exit 1
    fi
    chrom="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CHROMOSOMES_FILE}")"
fi

if [ -z "${chrom}" ]; then
    echo "No chromosome found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 1
fi
if [ ! -s "${SAMPLE_SHEET_BY_CHROM}" ]; then
    echo "Missing chromosome sample sheet: ${SAMPLE_SHEET_BY_CHROM}" >&2
    exit 1
fi
if ! command -v Rscript >/dev/null 2>&1; then
    echo "Rscript is required for DSS DMR calling" >&2
    exit 1
fi

out_dir="${DMR_BY_CHROM_DIR}/${chrom}"
mkdir -p "${out_dir}"

echo "Calling DSS DMRs for ${chrom}"
echo "SAMPLE_SHEET_BY_CHROM=${SAMPLE_SHEET_BY_CHROM}"
echo "out_dir=${out_dir}"

Rscript scripts/call_methylbert_dmrs_dss.R \
    --sample-sheet "${SAMPLE_SHEET_BY_CHROM}" \
    --output-dir "${out_dir}" \
    --chrom "${chrom}" \
    --target-group T \
    --background-group N \
    --top-n "${DMR_TOP_N}" \
    --delta "${DMR_DELTA}" \
    --p-threshold "${DMR_P_THRESHOLD}" \
    --min-cpg "${DMR_MIN_CPG}" \
    --min-len "${DMR_MIN_LEN}" \
    --merge-distance "${DMR_MERGE_DISTANCE}" \
    --max-background-per-cohort "${DMR_MAX_BACKGROUND_PER_COHORT:-0}" \
    --seed "${DMR_SEED:-950410}"

echo "Done: ${chrom}"
