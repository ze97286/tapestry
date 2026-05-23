#!/bin/bash
#SBATCH --job-name=mbert_submit
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/methylbert_submit_%j.out
#SBATCH --error=logs/methylbert_submit_%j.err

# Submit the full paper-style MethylBERT pipeline:
# BAMs -> DSS DMRs -> upstream MethylBERT fine-tuning -> MLE TF estimation.

set -euo pipefail

source scripts/methylbert_common.sh
bootstrap_methylbert_job

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.tsv}"
export RUN_LABEL METHYLBERT_WORK_DIR METHYLBERT_DMRS

if [ -z "${METHYLBERT_BULK_BAM_LIST:-}" ] || [ ! -s "${METHYLBERT_BULK_BAM_LIST:-}" ]; then
    echo "Set METHYLBERT_BULK_BAM_LIST before submitting the e2e pipeline" >&2
    exit 1
fi

n_bulk="$(awk 'NF && $1 !~ /^#/ {n++} END {print n+0}' "${METHYLBERT_BULK_BAM_LIST}")"
if [ "${n_bulk}" -lt 1 ]; then
    echo "No BAMs found in METHYLBERT_BULK_BAM_LIST=${METHYLBERT_BULK_BAM_LIST}" >&2
    exit 1
fi

dmr_job="$(sbatch --parsable --export=ALL run_methylbert_paper_dmr_slurm.sh)"
pre_job="$(sbatch --parsable --dependency=afterok:${dmr_job} --export=ALL run_methylbert_paper_preprocess_slurm.sh)"
ft_job="$(sbatch --parsable --dependency=afterok:${pre_job} --export=ALL run_methylbert_paper_finetune_slurm.sh)"
dec_job="$(sbatch --parsable --array=1-${n_bulk} --dependency=afterok:${ft_job} --export=ALL run_methylbert_paper_deconvolute_slurm.sh)"
col_job="$(sbatch --parsable --dependency=afterok:${dec_job} --export=ALL run_methylbert_paper_collect_slurm.sh)"

echo "Submitted MethylBERT e2e pipeline"
echo "DMR job: ${dmr_job}"
echo "Preprocess job: ${pre_job}"
echo "Fine-tune job: ${ft_job}"
echo "Deconvolution array: ${dec_job}"
echo "Collect job: ${col_job}"
echo "Work dir: ${METHYLBERT_WORK_DIR}"
