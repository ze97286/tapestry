#!/bin/bash
#SBATCH --job-name=mbert_prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/methylbert_prepare_inputs_%j.out
#SBATCH --error=logs/methylbert_prepare_inputs_%j.err

set -euo pipefail

export METHYLBERT_STEP=COLLECT
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DSS_DMRS="${DSS_DMRS:-${METHYLBERT_WORK_DIR}/dmr_pat/dss_dmrs.tsv}"
DMR_VARIANTS_DIR="${DMR_VARIANTS_DIR:-${METHYLBERT_WORK_DIR}/dmr_variants}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv}"
TOP_N="${TOP_N:-100}"

mkdir -p logs "${DMR_VARIANTS_DIR}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required. On BMRC load Python/3.11.3-GCCcore-12.3.0 or set METHYLBERT_COLLECT_MODULES." >&2
    exit 1
fi

# 1. Build the locus-collapsed DMR variants (including the collapsed_100kb panel the
#    read-call workflow consumes) so the regions actually used are reproducible.
if [ -s "${DSS_DMRS}" ]; then
    "${PYTHON_BIN}" scripts/select_methylbert_dmr_variants.py \
        --input "${DSS_DMRS}" \
        --output-dir "${DMR_VARIANTS_DIR}" \
        --top-n "${TOP_N}" \
        --collapse-distances 100000 500000 1000000
    cp "${DMR_VARIANTS_DIR}/collapsed_100kb_top${TOP_N}.tsv" "${METHYLBERT_DMRS}"
    echo "DMR panel: ${METHYLBERT_DMRS}"
else
    echo "Skipping DMR panel build: DSS DMRs not found at ${DSS_DMRS} (set DSS_DMRS)." >&2
fi

# 2. Optionally (re)build the read-call sample sheet and the bulk deconvolution sheet.
#    Set BUILD_READ_CALL_LISTS=1 and the per-read-call directory variables below.
if [ "${BUILD_READ_CALL_LISTS:-0}" = "1" ]; then
    : "${TUMOUR_SAMPLE_LIST:?set TUMOUR_SAMPLE_LIST}"
    : "${NORMAL_SAMPLE_LIST:?set NORMAL_SAMPLE_LIST}"
    : "${TUMOUR_READ_CALL_DIR:?set TUMOUR_READ_CALL_DIR}"
    : "${AB_READ_CALL_DIR:?set AB_READ_CALL_DIR}"
    : "${CD_READ_CALL_DIR:?set CD_READ_CALL_DIR}"
    READ_CALL_LISTS_DIR="${READ_CALL_LISTS_DIR:-${METHYLBERT_WORK_DIR}/read_call_lists}"

    prep_args=(
        scripts/prepare_methylbert_read_call_lists.py
        --tumour-sample-list "${TUMOUR_SAMPLE_LIST}"
        --normal-sample-list "${NORMAL_SAMPLE_LIST}"
        --tumour-read-call-dir "${TUMOUR_READ_CALL_DIR}"
        --ab-read-call-dir "${AB_READ_CALL_DIR}"
        --cd-read-call-dir "${CD_READ_CALL_DIR}"
        --output-dir "${READ_CALL_LISTS_DIR}"
    )
    if [ "${BALANCE_COHORTS:-0}" = "1" ]; then
        READ_CALL_MAX_NORMAL_PER_COHORT="${READ_CALL_MAX_NORMAL_PER_COHORT:-${DMR_MAX_BACKGROUND_PER_COHORT:-4}}"
        prep_args+=(--max-normal-per-cohort "${READ_CALL_MAX_NORMAL_PER_COHORT}")
        echo "Balancing read-call normal cohorts with max ${READ_CALL_MAX_NORMAL_PER_COHORT} per cohort"
    fi
    if [ -n "${BULK_SAMPLE_LIST:-}" ]; then
        prep_args+=(--bulk-sample-list "${BULK_SAMPLE_LIST}")
        [ -n "${BULK_READ_CALL_DIR:-}" ] && prep_args+=(--bulk-read-call-dir "${BULK_READ_CALL_DIR}")
    fi
    "${PYTHON_BIN}" "${prep_args[@]}"
else
    echo "Skipping read-call list build (set BUILD_READ_CALL_LISTS=1 with the *_READ_CALL_DIR vars; see RUNBOOK.md)."
fi

echo "Done."
