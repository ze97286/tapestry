#!/bin/bash
#SBATCH --job-name=mbert_dmr_pat
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/methylbert_dmr_pat_%j.out
#SBATCH --error=logs/methylbert_dmr_pat_%j.err

set -euo pipefail

export METHYLBERT_STEP=DMR
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DMR_COUNT_DIR="${DMR_COUNT_DIR:-${METHYLBERT_WORK_DIR}/dmr_pat_counts}"
DMR_DIR="${DMR_DIR:-${METHYLBERT_WORK_DIR}/dmr_pat}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.tsv}"
METHYLBERT_DMRS_BED="${METHYLBERT_DMRS_BED:-${METHYLBERT_DMRS%.tsv}.bed}"

METHYLBERT_DMR_TUMOUR_PAT_LIST="${METHYLBERT_DMR_TUMOUR_PAT_LIST:-}"
METHYLBERT_DMR_NORMAL_PAT_LIST="${METHYLBERT_DMR_NORMAL_PAT_LIST:-}"
METHYLBERT_CPG_FILE="${METHYLBERT_CPG_FILE:-/users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz}"

PAT_METHYLATED_CHAR="${PAT_METHYLATED_CHAR:-C}"
PAT_UNMETHYLATED_CHAR="${PAT_UNMETHYLATED_CHAR:-T}"
DMR_TOP_N="${DMR_TOP_N:-100}"
DMR_DELTA="${DMR_DELTA:-0.2}"
DMR_P_THRESHOLD="${DMR_P_THRESHOLD:-0.05}"
DMR_MIN_CPG="${DMR_MIN_CPG:-4}"
DMR_MIN_LEN="${DMR_MIN_LEN:-50}"
DMR_MERGE_DISTANCE="${DMR_MERGE_DISTANCE:-50}"
MIN_CPG_COVERAGE="${MIN_CPG_COVERAGE:-1}"

mkdir -p logs "${DMR_COUNT_DIR}" "${DMR_DIR}"

if ! command -v python >/dev/null 2>&1; then
    echo "python is required for PAT count extraction" >&2
    exit 1
fi
if ! command -v Rscript >/dev/null 2>&1; then
    echo "Rscript is required for DSS DMR calling" >&2
    exit 1
fi
Rscript -e 'missing <- setdiff(c("optparse", "DSS"), rownames(installed.packages())); if (length(missing) > 0) stop("Missing R packages required for DMR calling: ", paste(missing, collapse=", ")); suppressPackageStartupMessages({ library(optparse); library(DSS) })'

if [ -z "${METHYLBERT_DMR_TUMOUR_PAT_LIST}" ] || [ ! -s "${METHYLBERT_DMR_TUMOUR_PAT_LIST}" ]; then
    echo "Set METHYLBERT_DMR_TUMOUR_PAT_LIST to a file listing tumour PATs" >&2
    exit 1
fi
if [ -z "${METHYLBERT_DMR_NORMAL_PAT_LIST}" ] || [ ! -s "${METHYLBERT_DMR_NORMAL_PAT_LIST}" ]; then
    echo "Set METHYLBERT_DMR_NORMAL_PAT_LIST to a file listing control/background PATs" >&2
    exit 1
fi
if [ ! -s "${METHYLBERT_CPG_FILE}" ]; then
    echo "Missing METHYLBERT_CPG_FILE=${METHYLBERT_CPG_FILE}" >&2
    exit 1
fi

echo "=== MethylBERT paper-style DMRs from PAT counts ==="
echo "METHYLBERT_DMR_TUMOUR_PAT_LIST=${METHYLBERT_DMR_TUMOUR_PAT_LIST}"
echo "METHYLBERT_DMR_NORMAL_PAT_LIST=${METHYLBERT_DMR_NORMAL_PAT_LIST}"
echo "METHYLBERT_CPG_FILE=${METHYLBERT_CPG_FILE}"
echo "PAT_METHYLATED_CHAR=${PAT_METHYLATED_CHAR}"
echo "PAT_UNMETHYLATED_CHAR=${PAT_UNMETHYLATED_CHAR}"
echo "DMR_COUNT_DIR=${DMR_COUNT_DIR}"
echo "DMR_DIR=${DMR_DIR}"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
echo "METHYLBERT_DMRS_BED=${METHYLBERT_DMRS_BED}"

SAMPLE_SHEET="${DMR_DIR}/dss_samples.tsv"
echo -e "sample\tgroup\tcounts_path" > "${SAMPLE_SHEET}"

extract_counts() {
    local pat="$1"
    local group="$2"
    local sample
    local out
    sample="$(basename "${pat}")"
    sample="${sample%.gz}"
    sample="${sample%.pat}"
    out="${DMR_COUNT_DIR}/${sample}.dss_counts.tsv"

    if [ ! -s "${pat}" ]; then
        echo "Missing ${group} PAT: ${pat}" >&2
        exit 1
    fi

    if [ ! -s "${out}" ] || [ "${FORCE_REBUILD_DMR_COUNTS:-0}" = "1" ]; then
        echo "Extracting PAT CpG counts for ${sample} (${group})"
        python scripts/extract_pat_cpg_counts.py \
            --pat "${pat}" \
            --cpg-file "${METHYLBERT_CPG_FILE}" \
            --output "${out}" \
            --methylated-char "${PAT_METHYLATED_CHAR}" \
            --unmethylated-char "${PAT_UNMETHYLATED_CHAR}" \
            --min-coverage "${MIN_CPG_COVERAGE}"
    else
        echo "Skipping existing counts for ${sample}"
    fi

    echo -e "${sample}\t${group}\t${out}" >> "${SAMPLE_SHEET}"
}

while read -r pat; do
    [ -n "${pat}" ] || continue
    extract_counts "${pat}" T
done < <(awk 'NF && $1 !~ /^#/ {print $1}' "${METHYLBERT_DMR_TUMOUR_PAT_LIST}")

while read -r pat; do
    [ -n "${pat}" ] || continue
    extract_counts "${pat}" N
done < <(awk 'NF && $1 !~ /^#/ {print $1}' "${METHYLBERT_DMR_NORMAL_PAT_LIST}")

echo "Calling DSS DMRs"
Rscript scripts/call_methylbert_dmrs_dss.R \
    --sample-sheet "${SAMPLE_SHEET}" \
    --output-dir "${DMR_DIR}" \
    --target-group T \
    --background-group N \
    --top-n "${DMR_TOP_N}" \
    --delta "${DMR_DELTA}" \
    --p-threshold "${DMR_P_THRESHOLD}" \
    --min-cpg "${DMR_MIN_CPG}" \
    --min-len "${DMR_MIN_LEN}" \
    --merge-distance "${DMR_MERGE_DISTANCE}"

cp "${DMR_DIR}/dmrs_top${DMR_TOP_N}.tsv" "${METHYLBERT_DMRS}"
python scripts/dss_dmrs_to_bed.py \
    --input "${METHYLBERT_DMRS}" \
    --output "${METHYLBERT_DMRS_BED}" \
    --start-base "${DMR_POSITION_START_BASE:-1}"

echo "Done."
echo "DSS sample sheet: ${SAMPLE_SHEET}"
echo "All DMRs: ${DMR_DIR}/dss_dmrs.tsv"
echo "Selected DMRs: ${METHYLBERT_DMRS}"
echo "Selected DMR BED: ${METHYLBERT_DMRS_BED}"
