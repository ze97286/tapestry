#!/bin/bash
# Shared environment and submit helpers for the v0.6 MethylBERT BMRC experiments.

set -euo pipefail

mbert_v06_init() {
    local caller="$1"
    local caller_dir
    caller_dir="$(cd "$(dirname "${caller}")" && pwd)"

    export METHYLBERT_RUN_NAME
    METHYLBERT_RUN_NAME="$(basename "${caller_dir}")"

    export PROJECT_DIR="${PROJECT_DIR:-/gpfs3/well/ludwig/users/uii408/tapestry}"
    cd "${PROJECT_DIR}"

    if [ -s "${PROJECT_DIR}/configs/methylbert_oac_paper_bmrc.env" ]; then
        source "${PROJECT_DIR}/configs/methylbert_oac_paper_bmrc.env"
    fi

    export RUN_LABEL="${METHYLBERT_RUN_NAME}"
    export OUTPUT_DIR="/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc"
    export METHYLBERT_DIR="${PROJECT_DIR}/external/methylbert"
    export METHYLBERT_WORK_DIR="${OUTPUT_DIR}/methylbert/${RUN_LABEL}"
    export VARIANT="${METHYLBERT_RUN_NAME}"

    export METHYLBERT_REF_FASTA=""
    export METHYLBERT_REF_FASTA_GZ="/well/ludwig/shared/genomes/hg38_full_gatk_HPV_HBV_HCV_spike-ins_v2.fa.gz"
    export METHYLBERT_REF_STAGE_DIR="${METHYLBERT_WORK_DIR}/reference_stage"

    export METHYLBERT_R_LIBS="${HOME}/R/library"
    export METHYLBERT_ENV_COMMAND='export R_LIBS_USER="${METHYLBERT_R_LIBS}"'
    export METHYLBERT_DEACTIVATE_CONDA="1"
    export METHYLBERT_VENV_SCOPE="none"
    export METHYLBERT_SETUP_R_DEPS="1"
    unset METHYLBERT_CONFIG
    unset METHYLBERT_STEP_MODULES

    export METHYLBERT_MODULES=""
    export METHYLBERT_BMRC_DMR_DEFAULT_MODULES="${METHYLBERT_BMRC_DMR_DEFAULT_MODULES:-Python/3.11.3-GCCcore-12.3.0 R/4.3.2-gfbf-2023a R-bundle-Bioconductor/3.18-foss-2023a-R-4.3.2}"
    export METHYLBERT_DMR_MODULES="${METHYLBERT_DMR_MODULES:-${METHYLBERT_BMRC_DMR_DEFAULT_MODULES}}"
    export METHYLBERT_SETUP_MODULES="${METHYLBERT_SETUP_MODULES:-${METHYLBERT_DMR_MODULES}}"
    export METHYLBERT_PREPROCESS_PAT_MODULES="${METHYLBERT_PREPROCESS_PAT_MODULES:-Python/3.11.3-GCCcore-12.3.0 SAMtools/1.18-GCC-12.3.0}"
    export METHYLBERT_FINETUNE_MODULES="${METHYLBERT_FINETUNE_MODULES:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 Transformers/4.39.3-gfbf-2023a scikit-learn/1.3.1-gfbf-2023a}"
    export METHYLBERT_DECONVOLUTE_MODULES="${METHYLBERT_FINETUNE_MODULES} SAMtools/1.18-GCC-12.3.0"
    export METHYLBERT_COLLECT_MODULES="${METHYLBERT_COLLECT_MODULES:-Python/3.11.3-GCCcore-12.3.0}"

    export DMR_COUNT_DIR="${METHYLBERT_WORK_DIR}/dmr_pat_counts"
    export DMR_DIR="${METHYLBERT_WORK_DIR}/dmr_pat"
    export DMR_BY_CHROM_DIR="${DMR_DIR}/by_chrom"
    export DSS_SPLIT_DIR="${DMR_DIR}/counts_by_chrom"
    export CHROMOSOMES_FILE="${DMR_DIR}/chromosomes.txt"
    export SAMPLE_SHEET_BY_CHROM="${DMR_DIR}/dss_samples_by_chrom.tsv"
    export METHYLBERT_RAW_DMRS="${METHYLBERT_WORK_DIR}/dmrs_top100.dss.tsv"
    export METHYLBERT_RAW_DMRS_BED="${METHYLBERT_WORK_DIR}/dmrs_top100.dss.bed"
    export METHYLBERT_COLLAPSED_DMRS="${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv"
    export METHYLBERT_COLLAPSED_DMRS_BED="${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.bed"

    case "${METHYLBERT_RUN_NAME}" in
        06_AB|06_AB_lenmatch_exact|06_AB_full_length_only)
            export DMR_BACKGROUND_COHORTS="AB_plasma"
            export READ_CALL_NORMAL_COHORTS="AB_plasma"
            export DMR_MAX_BACKGROUND_PER_COHORT="4"
            export READ_CALL_MAX_NORMAL_PER_COHORT="4"
            ;;
        06_ABCD)
            export DMR_BACKGROUND_COHORTS="AB_plasma,CD_plasma"
            export READ_CALL_NORMAL_COHORTS="AB_plasma,CD_plasma"
            export DMR_MAX_BACKGROUND_PER_COHORT="4"
            export READ_CALL_MAX_NORMAL_PER_COHORT="4"
            ;;
        06_CD_full_length_only)
            export DMR_BACKGROUND_COHORTS="CD_plasma"
            export READ_CALL_NORMAL_COHORTS="CD_plasma"
            export DMR_MAX_BACKGROUND_PER_COHORT="0"
            export READ_CALL_MAX_NORMAL_PER_COHORT="0"
            ;;
        *)
            echo "Unsupported v0.6 MethylBERT run directory: ${METHYLBERT_RUN_NAME}" >&2
            exit 1
            ;;
    esac

    export METHYLBERT_DMR_TUMOUR_PAT_LIST="${PROJECT_DIR}/data/methylbert/oac_dmr_tumour_pats.list"
    export METHYLBERT_DMR_NORMAL_PAT_LIST="${PROJECT_DIR}/data/methylbert/oac_dmr_normal_pats.list"
    export TUMOUR_SAMPLE_LIST="${METHYLBERT_DMR_TUMOUR_PAT_LIST}"
    export NORMAL_SAMPLE_LIST="${METHYLBERT_DMR_NORMAL_PAT_LIST}"
    export METHYLBERT_CPG_FILE="${METHYLBERT_CPG_FILE:-${PROJECT_DIR}/data/CpG.bed.gz}"
    export PAT_METHYLATED_CHAR="C"
    export PAT_UNMETHYLATED_CHAR="T"
    export DMR_POSITION_START_BASE="1"

    export TUMOUR_READ_CALL_DIR="/well/ludwig/users/benjamin/OAC_Immuno_Trial/TAPS_Tissue/PerReadCalls"
    export AB_READ_CALL_DIR="/well/ludwig2/projects/processed/Lu_lab/OAC_immuno_Trial/TAPS_cfDNA/Results/1.7/PerReadCalls"
    export CD_READ_CALL_DIR="/well/ludwig2/projects/processed/Lu_lab/OAC_immuno_Trial/TAPS_cfDNA/CD/Results/1.1/PerReadCalls"
    export READ_CALL_LISTS_DIR="${METHYLBERT_WORK_DIR}/read_call_lists"
    export READ_CALL_SAMPLE_SHEET="${READ_CALL_LISTS_DIR}/oac_dmr_read_calls.sample_sheet.tsv"

    export READ_CALL_MODE="contained"
    export DMR_START_BASE="1"
    export MIN_INFORMATIVE="2"
    export MAX_READS_PER_SAMPLE="200000"
    export STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE="0"
    export BUILD_READ_CALL_LISTS="1"
    export BALANCE_COHORTS="1"

    case "${METHYLBERT_RUN_NAME}" in
        06_AB_lenmatch_exact)
            export LENGTH_MATCH="1"
            export LENGTH_MATCH_BIN="1"
            export BALANCE_LABELS="1"
            ;;
        06_AB_full_length_only|06_CD_full_length_only)
            export MIN_READ_LENGTH="150"
            export BALANCE_LABELS="1"
            ;;
    esac

    export READ_CALL_SHARD_DIR="${METHYLBERT_WORK_DIR}/preprocess_taps_read_call_shards_${VARIANT}"
    export READ_CALL_PREPROCESS_DIR="${METHYLBERT_WORK_DIR}/preprocess_taps_read_calls_${VARIANT}"
    export PREPROCESS_DIR="${READ_CALL_PREPROCESS_DIR}"
    export MODEL_DIR="${METHYLBERT_WORK_DIR}/model_taps_read_calls_${VARIANT}"
    export EVAL_DIR="${MODEL_DIR}/heldout_eval"
    export DECONV_DIR="${METHYLBERT_WORK_DIR}/deconvolution_taps_read_calls_${VARIANT}_adjusted"
    export SUMMARY_OUTPUT="${DECONV_DIR}/deconvolution_summary.csv"
    export VALIDATE_DIR="${DECONV_DIR}/validation"

    export N_ENCODER="12"
    export SEQ_LEN="150"
    export BATCH_SIZE="128"
    export GRAD_ACCUM="4"
    export STEPS="600"
    export NUM_WORKERS="8"
    export LOG_FREQ="10"
    export EVAL_FREQ="10"
    export WARM_UP="100"
    export DECREASE_STEPS="200"
    export LR="4e-4"
    export LOSS="bce"
    export WITH_CUDA="1"
    export EVAL_BATCH_SIZE="256"
    export DECONV_BATCH_SIZE="128"
    export ADJUSTMENT="1"
    export READ_CALL_MAX_READS_PER_SAMPLE="0"

    export ICHORCNA_FILE="${PROJECT_DIR}/data/cfDNA_tumour_fraction_ichorCNA.json"
    export ESTIMATE_COL="T"
}

mbert_v06_context() {
    echo "METHYLBERT_RUN_NAME=${METHYLBERT_RUN_NAME}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "METHYLBERT_WORK_DIR=${METHYLBERT_WORK_DIR}"
    echo "DMR_BACKGROUND_COHORTS=${DMR_BACKGROUND_COHORTS}"
    echo "READ_CALL_NORMAL_COHORTS=${READ_CALL_NORMAL_COHORTS}"
    echo "LENGTH_MATCH=${LENGTH_MATCH:-0}"
    echo "LENGTH_MATCH_BIN=${LENGTH_MATCH_BIN:-}"
    echo "MIN_READ_LENGTH=${MIN_READ_LENGTH:-}"
    echo "BALANCE_LABELS=${BALANCE_LABELS:-0}"
}

mbert_v06_submit_dmr_prepare() {
    mbert_v06_context
    echo "DMR_DIR=${DMR_DIR}"
    sbatch --parsable --export=ALL run_methylbert_paper_dmr_pat_prepare_slurm.sh
}

mbert_v06_submit_dmr_chr_array() {
    test -s "${CHROMOSOMES_FILE}"
    test -s "${SAMPLE_SHEET_BY_CHROM}"
    local n_chrom
    n_chrom="$(wc -l < "${CHROMOSOMES_FILE}")"
    mbert_v06_context
    echo "CHROMOSOMES_FILE=${CHROMOSOMES_FILE}"
    echo "N_CHROM=${n_chrom}"
    sbatch --parsable \
        --array=1-"${n_chrom}"%130 \
        --mem="${METHYLBERT_DMR_CHR_MEM:-256G}" \
        --time="${METHYLBERT_DMR_CHR_TIME:-24:00:00}" \
        --export=ALL \
        run_methylbert_paper_dmr_pat_chr_slurm.sh
}

mbert_v06_submit_dmr_merge() {
    test -d "${DMR_BY_CHROM_DIR}"
    export METHYLBERT_DMRS="${METHYLBERT_RAW_DMRS}"
    export METHYLBERT_DMRS_BED="${METHYLBERT_RAW_DMRS_BED}"
    mbert_v06_context
    echo "DMR_BY_CHROM_DIR=${DMR_BY_CHROM_DIR}"
    echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
    sbatch --parsable --export=ALL run_methylbert_paper_dmr_pat_merge_slurm.sh
}

mbert_v06_submit_prepare_inputs() {
    export DSS_DMRS="${DMR_DIR}/dss_dmrs.tsv"
    export METHYLBERT_DMRS="${METHYLBERT_COLLAPSED_DMRS}"
    test -s "${DSS_DMRS}"
    mbert_v06_context
    echo "DSS_DMRS=${DSS_DMRS}"
    echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
    sbatch --parsable --export=ALL run_methylbert_paper_prepare_inputs_slurm.sh
}

mbert_v06_submit_preprocess_shards() {
    export METHYLBERT_DMRS="${METHYLBERT_COLLAPSED_DMRS}"
    test -s "${READ_CALL_SAMPLE_SHEET}"
    test -s "${METHYLBERT_DMRS}"
    local n
    n="$(wc -l < "${READ_CALL_SAMPLE_SHEET}")"
    mbert_v06_context
    echo "READ_CALL_SAMPLE_SHEET=${READ_CALL_SAMPLE_SHEET}"
    echo "Sample-sheet label counts:"
    cut -f2 "${READ_CALL_SAMPLE_SHEET}" | sort | uniq -c
    echo "READ_CALL_SHARD_DIR=${READ_CALL_SHARD_DIR}"
    sbatch --parsable --array=1-"${n}"%130 --export=ALL run_methylbert_paper_preprocess_read_calls_array_slurm.sh
}

mbert_v06_submit_merge_shards() {
    test -d "${READ_CALL_SHARD_DIR}"
    export MAX_READS_PER_LABEL="500000"
    export SPLIT_RATIO="0.8"
    export SPLIT_BY="sample"
    mbert_v06_context
    echo "READ_CALL_SHARD_DIR=${READ_CALL_SHARD_DIR}"
    echo "READ_CALL_PREPROCESS_DIR=${READ_CALL_PREPROCESS_DIR}"
    sbatch --parsable --export=ALL run_methylbert_paper_merge_read_call_shards_slurm.sh
}

mbert_v06_submit_finetune() {
    test -s "${PREPROCESS_DIR}/train_seq.csv"
    test -s "${PREPROCESS_DIR}/test_seq.csv"
    mbert_v06_context
    echo "PREPROCESS_DIR=${PREPROCESS_DIR}"
    echo "MODEL_DIR=${MODEL_DIR}"
    sbatch --parsable --export=ALL run_methylbert_paper_finetune_slurm.sh
}

mbert_v06_submit_eval() {
    test -s "${PREPROCESS_DIR}/test_seq.csv"
    test -d "${MODEL_DIR}/bert.model"
    mbert_v06_context
    echo "EVAL_DIR=${EVAL_DIR}"
    sbatch --parsable --export=ALL run_methylbert_paper_eval_slurm.sh
}

mbert_v06_submit_deconvolute() {
    export METHYLBERT_DMRS="${METHYLBERT_COLLAPSED_DMRS}"
    export METHYLBERT_BULK_READ_CALL_LIST="${METHYLBERT_WORK_DIR}/read_call_lists/oac_bulk_read_calls.sample_sheet.tsv"
    test -d "${MODEL_DIR}/bert.model"
    test -s "${METHYLBERT_DMRS}"
    test -s "${METHYLBERT_BULK_READ_CALL_LIST}"
    local n
    n="$(wc -l < "${METHYLBERT_BULK_READ_CALL_LIST}")"
    mbert_v06_context
    echo "METHYLBERT_BULK_READ_CALL_LIST=${METHYLBERT_BULK_READ_CALL_LIST}"
    echo "DECONV_DIR=${DECONV_DIR}"
    sbatch --parsable --array=1-"${n}"%130 --export=ALL run_methylbert_paper_deconvolute_read_calls_slurm.sh
}

mbert_v06_submit_collect() {
    test -d "${DECONV_DIR}"
    mbert_v06_context
    echo "SUMMARY_OUTPUT=${SUMMARY_OUTPUT}"
    sbatch --parsable --export=ALL run_methylbert_paper_collect_slurm.sh
}

mbert_v06_submit_validate() {
    test -s "${SUMMARY_OUTPUT}"
    mbert_v06_context
    echo "SUMMARY_OUTPUT=${SUMMARY_OUTPUT}"
    echo "VALIDATE_DIR=${VALIDATE_DIR}"
    sbatch --parsable --export=ALL run_methylbert_paper_validate_slurm.sh
}
