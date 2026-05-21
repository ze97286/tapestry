#!/bin/bash
#SBATCH --job-name=mbert_dmr
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/methylbert_dmr_%j.out
#SBATCH --error=logs/methylbert_dmr_%j.err

set -euo pipefail

export METHYLBERT_STEP=DMR
source scripts/methylbert_common.sh
bootstrap_methylbert_job
source scripts/methylbert_reference.sh

check_dmr_dependencies() {
    if ! command -v python >/dev/null 2>&1; then
        echo "python is required for DMR CpG count extraction" >&2
        exit 1
    fi
    python - <<'PY'
try:
    import pysam  # noqa: F401
except ModuleNotFoundError:
    raise SystemExit(
        "Python package pysam is required for DMR CpG count extraction. "
        "Add the exact pysam/Python module to METHYLBERT_DMR_MODULES, "
        "or activate an environment with pysam via METHYLBERT_ENV_COMMAND."
    )
PY

    if ! command -v Rscript >/dev/null 2>&1; then
        echo "Rscript is required for DSS DMR calling" >&2
        exit 1
    fi
    Rscript -e 'missing <- setdiff(c("optparse", "DSS"), rownames(installed.packages())); if (length(missing) > 0) stop("Missing R packages required for DMR calling: ", paste(missing, collapse=", ")); suppressPackageStartupMessages({ library(optparse); library(DSS) })'
}

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DMR_COUNT_DIR="${DMR_COUNT_DIR:-${METHYLBERT_WORK_DIR}/dmr_counts}"
DMR_DIR="${DMR_DIR:-${METHYLBERT_WORK_DIR}/dmr}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.tsv}"

METHYLBERT_DMR_TUMOUR_BAM_LIST="${METHYLBERT_DMR_TUMOUR_BAM_LIST:-${METHYLBERT_TUMOUR_BAM_LIST:-}}"
METHYLBERT_DMR_NORMAL_BAM_LIST="${METHYLBERT_DMR_NORMAL_BAM_LIST:-${METHYLBERT_CONTROL_BAM_LIST:-}}"
METHYLBERT_METHYLCALLER="${METHYLBERT_METHYLCALLER:-bismark}"
DMR_TOP_N="${DMR_TOP_N:-100}"
DMR_DELTA="${DMR_DELTA:-0.2}"
DMR_P_THRESHOLD="${DMR_P_THRESHOLD:-0.05}"
DMR_MIN_CPG="${DMR_MIN_CPG:-4}"
DMR_MIN_LEN="${DMR_MIN_LEN:-50}"
DMR_MERGE_DISTANCE="${DMR_MERGE_DISTANCE:-50}"
MIN_MAPQ="${MIN_MAPQ:-10}"
MIN_CPG_COVERAGE="${MIN_CPG_COVERAGE:-1}"

mkdir -p logs "${DMR_COUNT_DIR}" "${DMR_DIR}"

check_dmr_dependencies
stage_methylbert_reference
if [ -z "${METHYLBERT_DMR_TUMOUR_BAM_LIST}" ] || [ ! -s "${METHYLBERT_DMR_TUMOUR_BAM_LIST}" ]; then
    echo "Set METHYLBERT_DMR_TUMOUR_BAM_LIST, or METHYLBERT_TUMOUR_BAM_LIST" >&2
    exit 1
fi
if [ -z "${METHYLBERT_DMR_NORMAL_BAM_LIST}" ] || [ ! -s "${METHYLBERT_DMR_NORMAL_BAM_LIST}" ]; then
    echo "Set METHYLBERT_DMR_NORMAL_BAM_LIST, or METHYLBERT_CONTROL_BAM_LIST" >&2
    exit 1
fi

SAMPLE_SHEET="${DMR_DIR}/dss_samples.tsv"
echo -e "sample\tgroup\tcounts_path" > "${SAMPLE_SHEET}"

extract_counts() {
    local bam="$1"
    local group="$2"
    local sample
    local out
    sample="$(basename "${bam}")"
    sample="${sample%.bam}"
    sample="${sample%.cram}"
    out="${DMR_COUNT_DIR}/${sample}.dss_counts.tsv"

    if [ ! -s "${bam}" ]; then
        echo "Missing ${group} BAM: ${bam}" >&2
        exit 1
    fi

    if [ ! -s "${out}" ] || [ "${FORCE_REBUILD_DMR_COUNTS:-0}" = "1" ]; then
        echo "Extracting CpG counts for ${sample} (${group})"
        python scripts/extract_methylbert_cpg_counts.py \
            --bam "${bam}" \
            --reference "${METHYLBERT_REF_FASTA}" \
            --output "${out}" \
            --methylcaller "${METHYLBERT_METHYLCALLER}" \
            --min-mapq "${MIN_MAPQ}" \
            --min-coverage "${MIN_CPG_COVERAGE}"
    else
        echo "Skipping existing counts for ${sample}"
    fi

    echo -e "${sample}\t${group}\t${out}" >> "${SAMPLE_SHEET}"
}

while read -r bam; do
    [ -n "${bam}" ] || continue
    extract_counts "${bam}" T
done < <(awk 'NF && $1 !~ /^#/ {print $1}' "${METHYLBERT_DMR_TUMOUR_BAM_LIST}")

while read -r bam; do
    [ -n "${bam}" ] || continue
    extract_counts "${bam}" N
done < <(awk 'NF && $1 !~ /^#/ {print $1}' "${METHYLBERT_DMR_NORMAL_BAM_LIST}")

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

echo "Done."
echo "DSS sample sheet: ${SAMPLE_SHEET}"
echo "All DMRs: ${DMR_DIR}/dss_dmrs.tsv"
echo "Selected DMRs: ${METHYLBERT_DMRS}"
