#!/bin/bash
#SBATCH --job-name=homog_atlas_refs
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=1-66
#SBATCH --output=logs/homog_atlas_refs_%a.out
#SBATCH --error=logs/homog_atlas_refs_%a.err

# Run wgbstools homog on every reference PAT in manifest_ben_atlas.tsv,
# restricted to the 1105 selected atlas markers. Per-sample U/M counts feed
# the Beta-prior aggregator (scripts/build_atlas_priors.py) used by the
# Beta-Binomial deconvolution variant.
#
# Output: ${OUTPUT_DIR}/atlas_homog/<sample_id>.uxm.bed.gz
# One file per reference sample, aligned to markers.bed.

source slurm/common.sh

ATLAS_HOMOG_DIR="${OUTPUT_DIR}/atlas_homog"
MARKERS_BED="${OUTPUT_DIR}/markers/markers.bed"
mkdir -p "${ATLAS_HOMOG_DIR}"

if [ ! -f "${MARKERS_BED}" ]; then
    echo "ERROR: ${MARKERS_BED} not found. Run 04_select_markers.sh first."
    exit 1
fi

# manifest line N+1 (skip header)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
if [ -z "${LINE}" ]; then
    echo "No manifest line for task ${SLURM_ARRAY_TASK_ID} — array size mismatch."
    exit 0
fi

SID=$(echo "${LINE}" | cut -f1)
CELL_TYPE=$(echo "${LINE}" | cut -f2)
FPATH=$(echo "${LINE}" | cut -f3)

echo "Task ${SLURM_ARRAY_TASK_ID}: ${SID} (${CELL_TYPE})"
echo "  PAT: ${FPATH}"

EXPECTED_OUT="${ATLAS_HOMOG_DIR}/${SID}.uxm.bed.gz"
if [ -f "${EXPECTED_OUT}" ]; then
    echo "  Skipping (exists)"
    exit 0
fi

if [ ! -f "${FPATH}" ]; then
    echo "  ERROR: PAT file missing: ${FPATH}"
    exit 1
fi

${WGBSTOOLS} homog \
    -b "${MARKERS_BED}" \
    -l 4 \
    -o "${ATLAS_HOMOG_DIR}" \
    "${FPATH}" \
    2>&1

# wgbstools writes <pat-basename>.uxm.bed.gz; rename to <sample_id>.uxm.bed.gz
SRC_NAME=$(basename "${FPATH}" .pat.gz).uxm.bed.gz
SRC_PATH="${ATLAS_HOMOG_DIR}/${SRC_NAME}"
if [ -f "${SRC_PATH}" ] && [ "${SRC_NAME}" != "${SID}.uxm.bed.gz" ]; then
    mv "${SRC_PATH}" "${EXPECTED_OUT}"
fi

if [ -f "${EXPECTED_OUT}" ]; then
    echo "  Done: ${EXPECTED_OUT}"
else
    echo "  ERROR: expected output not produced"
    exit 1
fi
