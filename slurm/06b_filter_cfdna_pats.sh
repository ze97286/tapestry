#!/bin/bash
#SBATCH --job-name=filter_cfdna
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/filter_cfdna_%x_%a.out
#SBATCH --error=logs/filter_cfdna_%x_%a.err

# Filter cfDNA PAT files to marker regions AND flip TAPS→bisulfite convention.
# Outputs to a cohort-specific subdirectory (AB or CD).
#
# Usage:
#   N_AB=$(ls /gpfs3/users/ludwig/uii408/sharedscratch/tapestry/data/AB/*.pat.gz | wc -l)
#   sbatch --array=1-${N_AB} --export=ALL,COHORT=AB slurm/06b_filter_cfdna_pats.sh
#
#   N_CD=$(ls /gpfs3/users/ludwig/uii408/sharedscratch/tapestry/data/CD/*.pat.gz | wc -l)
#   sbatch --array=1-${N_CD} --export=ALL,COHORT=CD slurm/06b_filter_cfdna_pats.sh

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set. Use --export=ALL,COHORT=AB or COHORT=CD"
    exit 1
fi

COHORT_DIR="${PROJECT_DIR}/data/${COHORT}"
FILTERED_DIR="${OUTPUT_DIR}/filtered_pats/cfdna/${COHORT}"
MARKERS_BED="${OUTPUT_DIR}/markers/markers.bed"

mkdir -p "${FILTERED_DIR}"

# Markers BED must exist (created by 06a_filter_ref_pats.sh)
if [ ! -f "${MARKERS_BED}" ]; then
    echo "ERROR: ${MARKERS_BED} not found — run 06a_filter_ref_pats.sh first"
    exit 1
fi

# Build sorted file list for this cohort (idempotent, atomic rename)
FILE_LIST="${FILTERED_DIR}/pat_files.list"
if [ ! -f "${FILE_LIST}" ]; then
    ls "${COHORT_DIR}"/*.pat.gz | sort > "${FILE_LIST}.tmp"
    mv "${FILE_LIST}.tmp" "${FILE_LIST}"
    echo "Created file list with $(wc -l < "${FILE_LIST}") samples"
fi

# Get this task's PAT file
FPATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${FILE_LIST}")
SID=$(basename "${FPATH}" .pat.gz)

echo "Task ${SLURM_ARRAY_TASK_ID}: filtering + flipping ${SID} (cohort ${COHORT})"
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

# Extract reads overlapping marker regions.
# Input PATs are already in bisulfite convention (flipped during bam2pat).
${WGBSTOOLS} cview "${FPATH}" -L "${MARKERS_BED}" \
    | bgzip > "${OUT_FILE}.tmp"
mv "${OUT_FILE}.tmp" "${OUT_FILE}"
tabix -s 1 -b 2 -e 2 "${OUT_FILE}"

# Report stats
N_LINES=$(zcat "${OUT_FILE}" | wc -l)
N_READS=$(zcat "${OUT_FILE}" | awk -F'\t' '{s+=$4} END{print s+0}')
echo "Output: ${OUT_FILE}"
echo "Unique patterns: ${N_LINES}, Total reads: ${N_READS}"

if [ "${N_LINES}" -eq 0 ]; then
    echo "WARNING: filtered file is empty — check markers.bed vs PAT coordinates"
fi
