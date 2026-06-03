#!/bin/bash
# Prepare DSS count files and chromosome split manifests for PAT-based DMRs.

set -euo pipefail

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DMR_COUNT_DIR="${DMR_COUNT_DIR:-${METHYLBERT_WORK_DIR}/dmr_pat_counts}"
DMR_DIR="${DMR_DIR:-${METHYLBERT_WORK_DIR}/dmr_pat}"
DSS_SPLIT_DIR="${DSS_SPLIT_DIR:-${DMR_DIR}/counts_by_chrom}"

METHYLBERT_DMR_TUMOUR_PAT_LIST="${METHYLBERT_DMR_TUMOUR_PAT_LIST:-}"
METHYLBERT_DMR_NORMAL_PAT_LIST="${METHYLBERT_DMR_NORMAL_PAT_LIST:-}"
METHYLBERT_CPG_FILE="${METHYLBERT_CPG_FILE:-/users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz}"

PAT_METHYLATED_CHAR="${PAT_METHYLATED_CHAR:-C}"
PAT_UNMETHYLATED_CHAR="${PAT_UNMETHYLATED_CHAR:-T}"
MIN_CPG_COVERAGE="${MIN_CPG_COVERAGE:-1}"

mkdir -p logs "${DMR_COUNT_DIR}" "${DMR_DIR}" "${DSS_SPLIT_DIR}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required for PAT count extraction" >&2
    exit 1
fi

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

echo "=== Preparing MethylBERT PAT DSS inputs ==="
echo "METHYLBERT_DMR_TUMOUR_PAT_LIST=${METHYLBERT_DMR_TUMOUR_PAT_LIST}"
echo "METHYLBERT_DMR_NORMAL_PAT_LIST=${METHYLBERT_DMR_NORMAL_PAT_LIST}"
echo "METHYLBERT_CPG_FILE=${METHYLBERT_CPG_FILE}"
echo "PAT_METHYLATED_CHAR=${PAT_METHYLATED_CHAR}"
echo "PAT_UNMETHYLATED_CHAR=${PAT_UNMETHYLATED_CHAR}"
echo "DMR_COUNT_DIR=${DMR_COUNT_DIR}"
echo "DMR_DIR=${DMR_DIR}"
echo "DSS_SPLIT_DIR=${DSS_SPLIT_DIR}"

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
        "${PYTHON_BIN}" scripts/extract_pat_cpg_counts.py \
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

SAMPLE_SHEET_BY_CHROM="${DMR_DIR}/dss_samples_by_chrom.tsv"
echo -e "sample\tgroup\tchrom\tcounts_path" > "${SAMPLE_SHEET_BY_CHROM}"
while IFS=$'\t' read -r sample group counts_path; do
    [ "${sample}" != "sample" ] || continue
    manifest="${DSS_SPLIT_DIR}/${sample}.manifest.tsv"
    split_force_args=()
    if [ "${FORCE_REBUILD_DSS_CHROM_SPLITS:-0}" = "1" ]; then
        split_force_args=(--force)
    fi
    "${PYTHON_BIN}" scripts/split_dss_counts_by_chrom.py \
        --input "${counts_path}" \
        --sample "${sample}" \
        --group "${group}" \
        --output-dir "${DSS_SPLIT_DIR}/${sample}" \
        --manifest "${manifest}" \
        "${split_force_args[@]}"
    awk 'NR > 1' "${manifest}" >> "${SAMPLE_SHEET_BY_CHROM}"
done < "${SAMPLE_SHEET}"

CHROMOSOMES_FILE="${DMR_DIR}/chromosomes.txt"
awk 'NR > 1 {print $3}' "${SAMPLE_SHEET_BY_CHROM}" | sort -V | uniq > "${CHROMOSOMES_FILE}"

echo "Done."
echo "DSS sample sheet: ${SAMPLE_SHEET}"
echo "DSS chromosome sample sheet: ${SAMPLE_SHEET_BY_CHROM}"
echo "Chromosomes: ${CHROMOSOMES_FILE}"
