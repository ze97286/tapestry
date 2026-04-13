#!/bin/bash
#SBATCH --job-name=select_markers
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/select_markers.out
#SBATCH --error=logs/select_markers.err

source slurm/common.sh

MARKERS_DIR="${OUTPUT_DIR}/markers"
mkdir -p "${MARKERS_DIR}"

python scripts/select_markers.py \
    --homog-dir "${OUTPUT_DIR}/homog" \
    --manifest "${MANIFEST}" \
    --output "${MARKERS_DIR}/markers.tsv" \
    --top-n 250 \
    --min-snr 3.0 \
    --min-signal 0.3 \
    --max-bg 0.1 \
    --max-single-bg 0.2 \
    --direction U
