#!/bin/bash
#SBATCH --job-name=mbert_col
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/methylbert_collect_%j.out
#SBATCH --error=logs/methylbert_collect_%j.err

set -euo pipefail

export METHYLBERT_STEP=COLLECT
source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DECONV_DIR="${DECONV_DIR:-${METHYLBERT_WORK_DIR}/deconvolution}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${DECONV_DIR}/deconvolution_summary.csv}"

mkdir -p logs

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required to collect MethylBERT deconvolution outputs. On BMRC, load Python/3.11.3-GCCcore-12.3.0 or set METHYLBERT_COLLECT_MODULES." >&2
    exit 1
fi

"${PYTHON_BIN}" scripts/collect_methylbert_deconvolution.py \
    --deconvolution-dir "${DECONV_DIR}" \
    --output "${SUMMARY_OUTPUT}"

echo "Done."
echo "Summary: ${SUMMARY_OUTPUT}"
