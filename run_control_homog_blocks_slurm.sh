#!/bin/bash
#SBATCH --job-name=ctrl_homog
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/control_homog_%A_%a.out
#SBATCH --error=logs/control_homog_%A_%a.err

# Run wgbstools homog for healthy cfDNA controls over the full segmentation
# block universe. Submit as an array over the generated file list.

set -euo pipefail

source slurm/common.sh

COHORT="${COHORT:-AB}"
RUN_LABEL="${RUN_LABEL:-AB_CD_controls_v04_blocks}"
CFDNA_INPUT_DIR="${CFDNA_INPUT_DIR:-${PROJECT_DIR}/data/${COHORT}}"
CFDNA_EXTRA_CONTROL_DIRS="${CFDNA_EXTRA_CONTROL_DIRS:-${PROJECT_DIR}/data/CD}"
CONTROL_PATTERN="${CONTROL_PATTERN:-Ctrl|healthy|^(GI|SCAN)}"
EXTRA_CONTROL_PATTERN="${EXTRA_CONTROL_PATTERN:-^(GI|SCAN)}"
BLOCKS_BED="${BLOCKS_BED:-${OUTPUT_DIR}/segmentation/blocks.bed}"
CONTROL_HOMOG_DIR="${CONTROL_HOMOG_DIR:-${OUTPUT_DIR}/control_homog_blocks/${RUN_LABEL}}"
FILE_LIST="${FILE_LIST:-${CONTROL_HOMOG_DIR}/control_pat_files.list}"
HOMOG_LEN="${HOMOG_LEN:-4}"

mkdir -p "${CONTROL_HOMOG_DIR}" logs

if [ ! -f "${BLOCKS_BED}" ]; then
    echo "ERROR: blocks BED not found: ${BLOCKS_BED}"
    exit 1
fi

build_file_list() {
    local tmp
    tmp="${FILE_LIST}.tmp"
    : > "${tmp}"
    find "${CFDNA_INPUT_DIR}" -maxdepth 1 -name "*.pat.gz" | sort | while IFS= read -r f; do
        sid=$(basename "${f}" .pat.gz)
        if [[ "${sid}" =~ ${CONTROL_PATTERN} ]]; then
            echo "${f}"
        fi
    done >> "${tmp}"
    if [ -n "${CFDNA_EXTRA_CONTROL_DIRS}" ]; then
        IFS=':' read -r -a extra_dirs <<< "${CFDNA_EXTRA_CONTROL_DIRS}"
        for d in "${extra_dirs[@]}"; do
            [ -z "${d}" ] && continue
            find "${d}" -maxdepth 1 -name "*.pat.gz" | sort | while IFS= read -r f; do
                sid=$(basename "${f}" .pat.gz)
                if [[ "${sid}" =~ ${EXTRA_CONTROL_PATTERN} ]]; then
                    echo "${f}"
                fi
            done >> "${tmp}"
        done
    fi
    sort -u "${tmp}" -o "${tmp}"
    mv "${tmp}" "${FILE_LIST}"
}

if [ ! -f "${FILE_LIST}" ]; then
    build_file_list
    echo "Created ${FILE_LIST} with $(wc -l < "${FILE_LIST}") controls"
fi

N_FILES=$(wc -l < "${FILE_LIST}")
if [ "${N_FILES}" -eq 0 ]; then
    echo "ERROR: no control PAT files found"
    exit 1
fi

run_one() {
    local task_id f sid expected src_name
    task_id="$1"
    f=$(sed -n "${task_id}p" "${FILE_LIST}")
    sid=$(basename "${f}" .pat.gz)
    expected="${CONTROL_HOMOG_DIR}/${sid}.uxm.bed.gz"

    echo "Task ${task_id}/${N_FILES}: ${sid}"
    echo "Input: ${f}"
    echo "Output: ${expected}"

    if [ -s "${expected}" ]; then
        echo "Skipping existing ${expected}"
        return
    fi

    "${WGBSTOOLS}" homog \
        -b "${BLOCKS_BED}" \
        -l "${HOMOG_LEN}" \
        -o "${CONTROL_HOMOG_DIR}" \
        "${f}" \
        2>&1

    src_name="$(basename "${f}" .pat.gz).uxm.bed.gz"
    if [ -f "${CONTROL_HOMOG_DIR}/${src_name}" ] && [ "${src_name}" != "${sid}.uxm.bed.gz" ]; then
        mv "${CONTROL_HOMOG_DIR}/${src_name}" "${expected}"
    fi

    if [ ! -s "${expected}" ]; then
        echo "ERROR: expected homog output was not created or is empty: ${expected}"
        exit 1
    fi
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    run_one "${SLURM_ARRAY_TASK_ID}"
else
    for i in $(seq 1 "${N_FILES}"); do
        run_one "${i}"
    done
fi
