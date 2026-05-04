#!/bin/bash
#SBATCH --job-name=atlas_priors
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_atlas_priors_%j.out
#SBATCH --error=logs/build_atlas_priors_%j.err

# Aggregate per-reference homog outputs into per-(cell type, marker) Beta
# priors via method-of-moments + cell-type-median variance flooring.
# Output: ${PROJECT_DIR}/data/atlas_priors_ben.tsv

source slurm/common.sh

ATLAS_HOMOG_DIR="${OUTPUT_DIR}/atlas_homog"
ATLAS="${OUTPUT_DIR}/markers/markers.tsv"
OUT="${PROJECT_DIR}/data/atlas_priors_ben.tsv"

if [ ! -d "${ATLAS_HOMOG_DIR}" ]; then
    echo "ERROR: ${ATLAS_HOMOG_DIR} missing. Run 11a_homog_atlas_references.sh first."
    exit 1
fi

python scripts/build_atlas_priors.py \
    --manifest "${MANIFEST}" \
    --homog-dir "${ATLAS_HOMOG_DIR}" \
    --atlas "${ATLAS}" \
    --output "${OUT}" \
    --var-floor-frac 0.5
