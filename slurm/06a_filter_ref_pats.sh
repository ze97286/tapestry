#!/bin/bash
#SBATCH --job-name=filter_ref
#SBATCH --partition=short
#SBATCH --array=1-72%35
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/filter_ref_%a.out
#SBATCH --error=logs/filter_ref_%a.err

# Filter reference PAT files to marker regions only.
# Produces much smaller PAT files for efficient mixture generation.

source slurm/common.sh

FILTERED_DIR="${OUTPUT_DIR}/filtered_pats/ref"
MARKERS_TSV="${OUTPUT_DIR}/markers/markers.tsv"
MARKERS_BED="${OUTPUT_DIR}/markers/markers.bed"

mkdir -p "${FILTERED_DIR}"

# Create markers BED from markers.tsv (idempotent, atomic rename)
if [ ! -f "${MARKERS_BED}" ]; then
    echo "Creating markers.bed from markers.tsv"
    tail -n +2 "${MARKERS_TSV}" \
        | awk -F'\t' 'BEGIN{OFS="\t"} {print $1, $2, $3, $4, $5}' \
        > "${MARKERS_BED}.tmp"
    mv "${MARKERS_BED}.tmp" "${MARKERS_BED}"
    echo "Created ${MARKERS_BED} with $(wc -l < "${MARKERS_BED}") regions"
fi

# Get this task's sample from manifest (skip header)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
SID=$(echo "${LINE}" | cut -f1)
FPATH=$(echo "${LINE}" | cut -f3)

echo "Task ${SLURM_ARRAY_TASK_ID}: filtering ${SID}"
echo "Input: ${FPATH}"

if [ ! -f "${FPATH}" ]; then
    echo "ERROR: PAT file not found: ${FPATH}"
    exit 1
fi

OUT_FILE="${FILTERED_DIR}/${SID}.markers.pat.gz"

# Skip if already done
if [ -f "${OUT_FILE}" ]; then
    echo "Skipping (exists): ${OUT_FILE}"
    exit 0
fi

# Extract reads overlapping marker regions (bgzip + index for wgbstools compatibility)
${WGBSTOOLS} cview "${FPATH}" -L "${MARKERS_BED}" | bgzip > "${OUT_FILE}.tmp"
mv "${OUT_FILE}.tmp" "${OUT_FILE}"
tabix -s 1 -b 2 -e 2 "${OUT_FILE}"

# Report stats
N_LINES=$(zcat "${OUT_FILE}" | wc -l)
N_READS=$(zcat "${OUT_FILE}" | awk -F'\t' '{s+=$4} END{print s+0}')
echo "Output: ${OUT_FILE}"
echo "Unique patterns: ${N_LINES}, Total reads: ${N_READS}"

# Sanity check: filtered file should be non-empty
if [ "${N_LINES}" -eq 0 ]; then
    echo "WARNING: filtered file is empty — check markers.bed vs PAT coordinates"
fi
