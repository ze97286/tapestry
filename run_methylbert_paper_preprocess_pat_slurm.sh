#!/bin/bash
#SBATCH --job-name=mbert_pat_pre
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/methylbert_preprocess_pat_%j.out
#SBATCH --error=logs/methylbert_preprocess_pat_%j.err

set -euo pipefail

export METHYLBERT_STEP=PREPROCESS_PAT
source scripts/methylbert_common.sh
bootstrap_methylbert_job
source scripts/methylbert_reference.sh

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
PREPROCESS_DIR="${PREPROCESS_DIR:-${METHYLBERT_WORK_DIR}/preprocess_pat}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.tsv}"
METHYLBERT_CPG_FILE="${METHYLBERT_CPG_FILE:-${PROJECT_DIR}/data/CpG.bed.gz}"

N_DMRS="${N_DMRS:-100}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
PAT_MAX_READS_PER_SAMPLE="${PAT_MAX_READS_PER_SAMPLE:-200000}"
PAT_MAX_READS_PER_LABEL="${PAT_MAX_READS_PER_LABEL:-500000}"
PAT_MIN_CPGS="${PAT_MIN_CPGS:-1}"
PAT_MAX_SEQ_BASES="${PAT_MAX_SEQ_BASES:-500}"

mkdir -p logs "${METHYLBERT_WORK_DIR}" "${PREPROCESS_DIR}"

export METHYLBERT_REF_REQUIRE_INDEX="${METHYLBERT_REF_REQUIRE_INDEX:-0}"

if [ -z "${METHYLBERT_DMR_TUMOUR_PAT_LIST:-}" ] || [ ! -s "${METHYLBERT_DMR_TUMOUR_PAT_LIST:-}" ]; then
    echo "Set METHYLBERT_DMR_TUMOUR_PAT_LIST to a file listing tumour PATs" >&2
    exit 1
fi
if [ -z "${METHYLBERT_DMR_NORMAL_PAT_LIST:-}" ] || [ ! -s "${METHYLBERT_DMR_NORMAL_PAT_LIST:-}" ]; then
    echo "Set METHYLBERT_DMR_NORMAL_PAT_LIST to a file listing normal/control PATs" >&2
    exit 1
fi
if [ ! -s "${METHYLBERT_DMRS}" ]; then
    echo "Missing selected DMR TSV: ${METHYLBERT_DMRS}" >&2
    exit 1
fi
if [ ! -s "${METHYLBERT_CPG_FILE}" ]; then
    echo "Missing CpG index: ${METHYLBERT_CPG_FILE}" >&2
    exit 1
fi

stage_methylbert_reference

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required for PAT preprocessing. On BMRC, load Python/3.11.3-GCCcore-12.3.0 or set METHYLBERT_PREPROCESS_PAT_MODULES." >&2
    exit 1
fi

echo "=== MethylBERT PAT preprocessing ==="
echo "METHYLBERT_WORK_DIR=${METHYLBERT_WORK_DIR}"
echo "PREPROCESS_DIR=${PREPROCESS_DIR}"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
echo "METHYLBERT_CPG_FILE=${METHYLBERT_CPG_FILE}"
echo "METHYLBERT_REF_FASTA=${METHYLBERT_REF_FASTA}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PAT_MAX_READS_PER_SAMPLE=${PAT_MAX_READS_PER_SAMPLE}"
echo "PAT_MAX_READS_PER_LABEL=${PAT_MAX_READS_PER_LABEL}"

"${PYTHON_BIN}" scripts/preprocess_methylbert_pats.py \
    --tumour-pat-list "${METHYLBERT_DMR_TUMOUR_PAT_LIST}" \
    --normal-pat-list "${METHYLBERT_DMR_NORMAL_PAT_LIST}" \
    --dmrs "${METHYLBERT_DMRS}" \
    --cpg-file "${METHYLBERT_CPG_FILE}" \
    --reference "${METHYLBERT_REF_FASTA}" \
    --output-dir "${PREPROCESS_DIR}" \
    --top-n "${N_DMRS}" \
    --split-ratio "${SPLIT_RATIO}" \
    --max-reads-per-sample "${PAT_MAX_READS_PER_SAMPLE}" \
    --max-reads-per-label "${PAT_MAX_READS_PER_LABEL}" \
    --min-cpgs "${PAT_MIN_CPGS}" \
    --max-seq-bases "${PAT_MAX_SEQ_BASES}" \
    --methylated-char "${PAT_METHYLATED_CHAR:-C}" \
    --unmethylated-char "${PAT_UNMETHYLATED_CHAR:-T}" \
    --cpg-position-base "${CPG_POSITION_BASE:-1}"

test -s "${PREPROCESS_DIR}/train_seq.csv"
test -s "${PREPROCESS_DIR}/test_seq.csv"
echo "Done."
echo "Train reads: ${PREPROCESS_DIR}/train_seq.csv"
echo "Test reads: ${PREPROCESS_DIR}/test_seq.csv"
echo "Selected DMRs: ${PREPROCESS_DIR}/dmrs.csv"
