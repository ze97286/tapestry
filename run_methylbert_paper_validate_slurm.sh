#!/bin/bash
#SBATCH --job-name=mbert_val
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/methylbert_validate_%j.out
#SBATCH --error=logs/methylbert_validate_%j.err

set -euo pipefail

export METHYLBERT_STEP=COLLECT
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DECONV_DIR="${DECONV_DIR:-${METHYLBERT_WORK_DIR}/deconvolution}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${DECONV_DIR}/deconvolution_summary.csv}"
ICHORCNA_FILE="${ICHORCNA_FILE:-${PROJECT_DIR}/data/cfDNA_tumour_fraction_ichorCNA.json}"
VALIDATE_DIR="${VALIDATE_DIR:-${DECONV_DIR}/validation}"
ESTIMATE_COL="${ESTIMATE_COL:-T}"
PREFIX="${PREFIX:-methylbert}"

mkdir -p logs "${VALIDATE_DIR}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required. plotly and pandas must be importable; set METHYLBERT_COLLECT_MODULES or a venv if not." >&2
    exit 1
fi

if [ ! -s "${SUMMARY_OUTPUT}" ]; then
    echo "Missing deconvolution summary: ${SUMMARY_OUTPUT} (run 06 first)." >&2
    exit 1
fi
if [ ! -s "${ICHORCNA_FILE}" ]; then
    echo "Missing ichorCNA file: ${ICHORCNA_FILE}" >&2
    exit 1
fi

# Writes the theta-vs-ichorCNA scatter and the theta-vs-fragmentomics correlation CSV.
"${PYTHON_BIN}" scripts/plot_methylbert_deconvolution_vs_ichorcna.py \
    --deconvolution-summary "${SUMMARY_OUTPUT}" \
    --ichorcna-file "${ICHORCNA_FILE}" \
    --output-dir "${VALIDATE_DIR}" \
    --estimate-col "${ESTIMATE_COL}" \
    --prefix "${PREFIX}"

echo "Done."
echo "Validation outputs: ${VALIDATE_DIR}"
