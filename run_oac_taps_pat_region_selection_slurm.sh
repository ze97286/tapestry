#!/bin/bash
#SBATCH --job-name=oac_pat_regions
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/oac_taps_pat_regions_%j.out
#SBATCH --error=logs/oac_taps_pat_regions_%j.err

set -euo pipefail

if [ -z "${PROJECT_DIR:-}" ]; then
    PROJECT_DIR="$(pwd)"
    export PROJECT_DIR
fi

if [ -n "${TAPS_PAT_REGION_CONFIG:-}" ]; then
    source "${TAPS_PAT_REGION_CONFIG}"
else
    source "${PROJECT_DIR}/configs/oac_taps_pat_regions.env"
fi

mkdir -p "${PROJECT_DIR}/logs" "${TAPS_PAT_MARKER_DIR}" "$(dirname "${TAPS_PAT_METHYLBERT_DMRS}")"
cd "${PROJECT_DIR}"

if [ -n "${TAPS_PAT_MODULE_INIT:-}" ]; then
    eval "${TAPS_PAT_MODULE_INIT}"
fi

if [ -n "${TAPS_PAT_R_MODULES:-}" ]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "TAPS_PAT_R_MODULES is set but the module command is unavailable. Set TAPS_PAT_MODULE_INIT or load R before submission." >&2
        exit 1
    fi
    for module_name in ${TAPS_PAT_R_MODULES}; do
        module load "${module_name}"
    done
fi

export R_LIBS_USER="${TAPS_PAT_R_LIBS:-${HOME}/R/library}"
mkdir -p "${R_LIBS_USER}"

if ! command -v Rscript >/dev/null 2>&1; then
    echo "Rscript is required. Load an R module or set TAPS_PAT_R_MODULES." >&2
    exit 1
fi

require_file() {
    local path="$1"
    local label="$2"
    if [ ! -s "${path}" ]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

require_file "${TAPS_PAT_GENERATE_ATLAS_R}" "generate_atlas.R"
require_file "${TAPS_PAT_CPG_FILE}" "hg38 CpG file"
require_file "${TAPS_PAT_MAP_FILE}" "read-to-class map"
require_file "${TAPS_PAT_INDEX_FILE}" "PAT coverage index"
require_file "${TAPS_PAT_BASE_DIR}/dmr_by_read/blood+tum+gi_scores-by-position_chr1.txt.gz" "chromosome score input"

if [ "${TAPS_PAT_ASSEMBLY}" != "hg38" ]; then
    echo "Refusing to run: TAPS_PAT_ASSEMBLY=${TAPS_PAT_ASSEMBLY}, but these OAC BAM/PAT inputs are hg38." >&2
    exit 1
fi

echo "=== OAC TAPS/PAT hg38 region selection ==="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TAPS_PAT_GENERATE_ATLAS_R=${TAPS_PAT_GENERATE_ATLAS_R}"
echo "TAPS_PAT_CPG_FILE=${TAPS_PAT_CPG_FILE}"
echo "TAPS_PAT_BASE_DIR=${TAPS_PAT_BASE_DIR}"
echo "TAPS_PAT_MAP_FILE=${TAPS_PAT_MAP_FILE}"
echo "TAPS_PAT_INDEX_FILE=${TAPS_PAT_INDEX_FILE}"
echo "TAPS_PAT_OUT_FILE=${TAPS_PAT_OUT_FILE}"
echo "TAPS_PAT_ASSEMBLY=${TAPS_PAT_ASSEMBLY}"
echo "METHYLBERT_PRETRAIN_ASSEMBLY=${METHYLBERT_PRETRAIN_ASSEMBLY:-unset}"
echo "Rscript=$(command -v Rscript)"
echo "R_LIBS_USER=${R_LIBS_USER}"

cmd=(
    Rscript "${TAPS_PAT_GENERATE_ATLAS_R}"
    --cpg_file "${TAPS_PAT_CPG_FILE}"
    --map_file "${TAPS_PAT_MAP_FILE}"
    --base_dir "${TAPS_PAT_BASE_DIR}"
    --out_file "${TAPS_PAT_OUT_FILE}"
    --index_file "${TAPS_PAT_INDEX_FILE}"
    --top_n "${TAPS_PAT_TOP_N}"
    --min_reads "${TAPS_PAT_MIN_READS}"
    --min_cpgs "${TAPS_PAT_MIN_CPGS}"
    --threads "${TAPS_PAT_THREADS}"
)

if [ "${TAPS_PAT_VERBOSE:-0}" = "1" ]; then
    cmd+=(--verbose)
fi
if [ -n "${TAPS_PAT_GROUP_MAPPING:-}" ]; then
    cmd+=(--group_mapping "${TAPS_PAT_GROUP_MAPPING}")
fi

"${cmd[@]}"

test -s "${TAPS_PAT_OUT_FILE}"
echo "Marker regions: ${TAPS_PAT_OUT_FILE}"

if [ "${TAPS_PAT_WRITE_METHYLBERT_DMRS:-0}" = "1" ]; then
    python scripts/prepare_methylbert_dmrs.py \
        --input "${TAPS_PAT_OUT_FILE}" \
        --output "${TAPS_PAT_METHYLBERT_DMRS}" \
        --target-filter "${TAPS_PAT_METHYLBERT_TARGET}" \
        --target-ctype T \
        --top-n "${TAPS_PAT_METHYLBERT_TOP_N}" \
        --sort-column none
    echo "Prepared hg38 OAC DMRs: ${TAPS_PAT_METHYLBERT_DMRS}"
fi

echo "Done."
