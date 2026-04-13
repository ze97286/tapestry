#!/bin/bash
#SBATCH --job-name=pat2beta
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-70
#SBATCH --output=logs/pat2beta_%a.out
#SBATCH --error=logs/pat2beta_%a.err

source slurm/common.sh

BETA_DIR="${OUTPUT_DIR}/betas"

# Get the Nth line from the manifest (skip header)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
SID=$(echo "$LINE" | cut -f1)
FPATH=$(echo "$LINE" | cut -f3)

echo "Task ${SLURM_ARRAY_TASK_ID}: ${SID}"

BETA_OUT="${BETA_DIR}/${SID}.beta"
if [ -f "${BETA_OUT}" ]; then
    echo "  Skipping (beta exists)"
    exit 0
fi

${WGBSTOOLS} pat2beta "${FPATH}" --out_dir "${BETA_DIR}" -@ 4 2>&1

# pat2beta names output by input filename; rename to sample_id
SRC_NAME=$(basename "${FPATH}" .pat.gz).beta
if [ -f "${BETA_DIR}/${SRC_NAME}" ] && [ "${SRC_NAME}" != "${SID}.beta" ]; then
    mv "${BETA_DIR}/${SRC_NAME}" "${BETA_OUT}"
fi

echo "  Done: ${BETA_OUT}"
