#!/bin/bash
#SBATCH --job-name=mbert_pre
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/methylbert_preprocess_%j.out
#SBATCH --error=logs/methylbert_preprocess_%j.err

set -euo pipefail

source slurm/common.sh
if [ -n "${METHYLBERT_CONFIG:-}" ]; then
    source "${METHYLBERT_CONFIG}"
fi
source scripts/methylbert_reference.sh

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_DIR="${METHYLBERT_DIR:-${PROJECT_DIR}/external/methylbert}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
PREPROCESS_DIR="${PREPROCESS_DIR:-${METHYLBERT_WORK_DIR}/preprocess}"
TRAIN_BAMS="${TRAIN_BAMS:-${METHYLBERT_WORK_DIR}/train_bams.tsv}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.tsv}"

N_DMRS="${N_DMRS:-100}"
N_CORES="${N_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
METHYLBERT_METHYLCALLER="${METHYLBERT_METHYLCALLER:-bismark}"
IGNORE_SEX_CHROMO="${IGNORE_SEX_CHROMO:-1}"

mkdir -p logs "${METHYLBERT_WORK_DIR}" "${PREPROCESS_DIR}"

if [ -n "${METHYLBERT_ENV_COMMAND:-}" ]; then
    eval "${METHYLBERT_ENV_COMMAND}"
fi

if [ ! -d "${METHYLBERT_DIR}/src/methylbert" ]; then
    echo "Missing upstream MethylBERT source: ${METHYLBERT_DIR}/src/methylbert" >&2
    exit 1
fi
stage_methylbert_reference
if [ -z "${METHYLBERT_TUMOUR_BAM_LIST:-}" ] || [ ! -s "${METHYLBERT_TUMOUR_BAM_LIST:-}" ]; then
    echo "Set METHYLBERT_TUMOUR_BAM_LIST to a file listing tumour tissue BAMs" >&2
    exit 1
fi
if [ -z "${METHYLBERT_CONTROL_BAM_LIST:-}" ] || [ ! -s "${METHYLBERT_CONTROL_BAM_LIST:-}" ]; then
    echo "Set METHYLBERT_CONTROL_BAM_LIST to a file listing healthy-control cfDNA BAMs" >&2
    exit 1
fi

if [ -n "${METHYLBERT_DMR_INPUT:-}" ]; then
    python scripts/prepare_methylbert_dmrs.py \
        --input "${METHYLBERT_DMR_INPUT}" \
        --output "${METHYLBERT_DMRS}" \
        --top-n "${N_DMRS}" \
        --target-ctype T
elif [ ! -s "${METHYLBERT_DMRS}" ]; then
    echo "Set METHYLBERT_DMR_INPUT, or provide prepared METHYLBERT_DMRS=${METHYLBERT_DMRS}" >&2
    exit 1
fi

{
    awk 'NF && $1 !~ /^#/ {print $1 "\tT"}' "${METHYLBERT_TUMOUR_BAM_LIST}"
    awk 'NF && $1 !~ /^#/ {print $1 "\tN"}' "${METHYLBERT_CONTROL_BAM_LIST}"
} > "${TRAIN_BAMS}"

missing=0
while IFS=$'\t' read -r bam label; do
    if [ ! -s "${bam}" ]; then
        echo "Missing ${label} BAM: ${bam}" >&2
        missing=$((missing + 1))
    fi
done < "${TRAIN_BAMS}"
if [ "${missing}" -ne 0 ]; then
    echo "Refusing to run with ${missing} missing BAM files" >&2
    exit 1
fi

export PYTHONPATH="${METHYLBERT_DIR}/src:${PYTHONPATH:-}"

args=(
    scripts/run_upstream_methylbert.py preprocess_finetune
    --sc_dataset "${TRAIN_BAMS}"
    --f_dmr "${METHYLBERT_DMRS}"
    --output_path "${PREPROCESS_DIR}"
    --f_ref "${METHYLBERT_REF_FASTA}"
    --n_mers 3
    --methylcaller "${METHYLBERT_METHYLCALLER}"
    --split_ratio "${SPLIT_RATIO}"
    --n_dmrs "${N_DMRS}"
    --n_cores "${N_CORES}"
)
if [ "${IGNORE_SEX_CHROMO}" = "1" ]; then
    args+=(--ignore_sex_chromo)
fi

echo "=== MethylBERT paper-style preprocessing ==="
echo "METHYLBERT_DIR=${METHYLBERT_DIR}"
echo "METHYLBERT_WORK_DIR=${METHYLBERT_WORK_DIR}"
echo "METHYLBERT_REF_FASTA=${METHYLBERT_REF_FASTA}"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
echo "TRAIN_BAMS=${TRAIN_BAMS}"
echo "PREPROCESS_DIR=${PREPROCESS_DIR}"
echo "METHYLCALLER=${METHYLBERT_METHYLCALLER}"
python "${args[@]}"

test -s "${PREPROCESS_DIR}/train_seq.csv"
test -s "${PREPROCESS_DIR}/test_seq.csv"
echo "Done."
echo "Train reads: ${PREPROCESS_DIR}/train_seq.csv"
echo "Test reads: ${PREPROCESS_DIR}/test_seq.csv"
echo "Selected DMRs: ${PREPROCESS_DIR}/dmrs.csv"
