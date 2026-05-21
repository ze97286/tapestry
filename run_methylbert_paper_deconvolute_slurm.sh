#!/bin/bash
#SBATCH --job-name=mbert_dec
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/methylbert_deconvolute_%A_%a.out
#SBATCH --error=logs/methylbert_deconvolute_%A_%a.err

set -euo pipefail

source scripts/methylbert_common.sh
bootstrap_methylbert_job
source scripts/methylbert_reference.sh

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_DIR="${METHYLBERT_DIR:-${PROJECT_DIR}/external/methylbert}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
PREPROCESS_DIR="${PREPROCESS_DIR:-${METHYLBERT_WORK_DIR}/preprocess}"
MODEL_DIR="${MODEL_DIR:-${METHYLBERT_WORK_DIR}/model}"
DECONV_DIR="${DECONV_DIR:-${METHYLBERT_WORK_DIR}/deconvolution}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${PREPROCESS_DIR}/dmrs.csv}"

N_CORES="${N_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
METHYLBERT_METHYLCALLER="${METHYLBERT_METHYLCALLER:-bismark}"
DECONV_BATCH_SIZE="${DECONV_BATCH_SIZE:-128}"
ADJUSTMENT="${ADJUSTMENT:-0}"
IGNORE_SEX_CHROMO="${IGNORE_SEX_CHROMO:-1}"

mkdir -p logs "${DECONV_DIR}"

if [ -z "${METHYLBERT_BULK_BAM_LIST:-}" ] || [ ! -s "${METHYLBERT_BULK_BAM_LIST:-}" ]; then
    echo "Set METHYLBERT_BULK_BAM_LIST to a file listing cfDNA BAMs to deconvolute" >&2
    exit 1
fi
stage_methylbert_reference
if [ ! -s "${METHYLBERT_DMRS}" ]; then
    echo "Missing MethylBERT DMRs: ${METHYLBERT_DMRS}" >&2
    exit 1
fi
if [ ! -s "${MODEL_DIR}/train_param.txt" ] || [ ! -d "${MODEL_DIR}/bert.model" ]; then
    echo "Missing fine-tuned model under ${MODEL_DIR}" >&2
    exit 1
fi

mapfile -t bams < <(awk 'NF && $1 !~ /^#/ {print $1}' "${METHYLBERT_BULK_BAM_LIST}")
if [ "${#bams[@]}" -eq 0 ]; then
    echo "No BAMs found in ${METHYLBERT_BULK_BAM_LIST}" >&2
    exit 1
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    idx=$((SLURM_ARRAY_TASK_ID - 1))
    if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#bams[@]}" ]; then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside BAM list length ${#bams[@]}" >&2
        exit 1
    fi
    selected_bams=("${bams[$idx]}")
else
    selected_bams=("${bams[@]}")
fi

export PYTHONPATH="${METHYLBERT_DIR}/src:${PYTHONPATH:-}"

echo "=== MethylBERT paper-style deconvolution ==="
echo "METHYLBERT_WORK_DIR=${METHYLBERT_WORK_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
echo "METHYLBERT_BULK_BAM_LIST=${METHYLBERT_BULK_BAM_LIST}"
echo "DECONV_DIR=${DECONV_DIR}"

for bam in "${selected_bams[@]}"; do
    if [ ! -s "${bam}" ]; then
        echo "Missing bulk BAM: ${bam}" >&2
        exit 1
    fi

    sample="$(basename "${bam}")"
    sample="${sample%.bam}"
    sample="${sample%.cram}"
    sample_dir="${DECONV_DIR}/${sample}"
    bulk_preprocess="${sample_dir}/preprocess"
    mkdir -p "${bulk_preprocess}" "${sample_dir}"

    preprocess_args=(
        scripts/run_upstream_methylbert.py preprocess_finetune
        --input_file "${bam}"
        --f_dmr "${METHYLBERT_DMRS}"
        --output_path "${bulk_preprocess}"
        --f_ref "${METHYLBERT_REF_FASTA}"
        --n_mers 3
        --methylcaller "${METHYLBERT_METHYLCALLER}"
        --split_ratio 1.0
        --n_dmrs -1
        --n_cores "${N_CORES}"
    )
    if [ "${IGNORE_SEX_CHROMO}" = "1" ]; then
        preprocess_args+=(--ignore_sex_chromo)
    fi

    echo "Preprocessing ${sample}"
    python "${preprocess_args[@]}"
    test -s "${bulk_preprocess}/data.csv"

    deconv_args=(
        scripts/run_upstream_methylbert.py deconvolute
        --input_data "${bulk_preprocess}/data.csv"
        --model_dir "${MODEL_DIR}"
        --output_path "${sample_dir}"
        --batch_size "${DECONV_BATCH_SIZE}"
    )
    if [ "${ADJUSTMENT}" = "1" ]; then
        deconv_args+=(--adjustment)
    fi

    echo "Deconvoluting ${sample}"
    python "${deconv_args[@]}"
    test -s "${sample_dir}/deconvolution.csv"
done

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    python scripts/collect_methylbert_deconvolution.py \
        --deconvolution-dir "${DECONV_DIR}" \
        --output "${DECONV_DIR}/deconvolution_summary.csv"
    echo "Summary: ${DECONV_DIR}/deconvolution_summary.csv"
else
    echo "Array task complete. Run run_methylbert_paper_collect_slurm.sh after all array tasks finish."
fi

echo "Done."
