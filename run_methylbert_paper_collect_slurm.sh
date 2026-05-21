#!/bin/bash
#SBATCH --job-name=mbert_col
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/methylbert_collect_%j.out
#SBATCH --error=logs/methylbert_collect_%j.err

set -euo pipefail

source slurm/common.sh
if [ -n "${METHYLBERT_CONFIG:-}" ]; then
    source "${METHYLBERT_CONFIG}"
fi

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
DECONV_DIR="${DECONV_DIR:-${METHYLBERT_WORK_DIR}/deconvolution}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${DECONV_DIR}/deconvolution_summary.csv}"

mkdir -p logs

python scripts/collect_methylbert_deconvolution.py \
    --deconvolution-dir "${DECONV_DIR}" \
    --output "${SUMMARY_OUTPUT}"

echo "Done."
echo "Summary: ${SUMMARY_OUTPUT}"
