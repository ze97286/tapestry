#!/bin/bash
#SBATCH --job-name=homog
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=1-70
#SBATCH --output=logs/homog_%a.out
#SBATCH --error=logs/homog_%a.err

source slurm/common.sh

HOMOG_DIR="${OUTPUT_DIR}/homog"
BLOCKS="${OUTPUT_DIR}/segmentation/blocks.bed"
mkdir -p "${HOMOG_DIR}"

if [ ! -f "${BLOCKS}" ] && [ -f "${BLOCKS}.gz" ]; then
    echo "Creating ${BLOCKS} from ${BLOCKS}.gz"
    zcat "${BLOCKS}.gz" > "${BLOCKS}"
fi

if [ ! -f "${BLOCKS}" ]; then
    echo "ERROR: blocks BED not found: ${BLOCKS}"
    echo "Run segmentation + merge first."
    exit 1
fi

# Get the Nth sample from the manifest
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
SID=$(echo "$LINE" | cut -f1)
FPATH=$(echo "$LINE" | cut -f3)

echo "Task ${SLURM_ARRAY_TASK_ID}: ${SID}"

# Check if output already exists
EXPECTED_OUT="${HOMOG_DIR}/${SID}.uxm.bed.gz"
if [ -f "${EXPECTED_OUT}" ]; then
    echo "  Skipping (exists)"
    exit 0
fi

${WGBSTOOLS} homog \
    -b "${BLOCKS}" \
    -l 4 \
    -o "${HOMOG_DIR}" \
    "${FPATH}" \
    2>&1

# Rename output to sample_id
SRC_NAME=$(basename "${FPATH}" .pat.gz).uxm.bed.gz
if [ -f "${HOMOG_DIR}/${SRC_NAME}" ] && [ "${SRC_NAME}" != "${SID}.uxm.bed.gz" ]; then
    mv "${HOMOG_DIR}/${SRC_NAME}" "${EXPECTED_OUT}"
fi

echo "  Done: ${EXPECTED_OUT}"
