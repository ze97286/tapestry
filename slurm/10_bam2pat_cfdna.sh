#!/bin/bash
#SBATCH --job-name=bam2pat
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bam2pat_%a.out
#SBATCH --error=logs/bam2pat_%a.err

# Convert cfDNA BAMs to PAT files with vclip clipping.
# Array job — one task per BAM file.
#
# Usage:
#   # First create the BAM file list:
#   ls /well/ludwig/processed/Lu_lab/OAC_immuno_Trial/TAPS_cfDNA/Results/1.6.1/Alignments/*_md.bam > data/cfdna_bams_AB.list
#   N=$(wc -l < data/cfdna_bams_AB.list)
#   sbatch --array=1-${N}%20 --export=ALL,COHORT=AB slurm/10_bam2pat_cfdna.sh

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set."
    exit 1
fi

BAM_LIST="${PROJECT_DIR}/data/cfdna_bams_${COHORT}.list"
OUTDIR="${PROJECT_DIR}/data/cfdna_pats/${COHORT}"
VCLIP="/users/ludwig/uii408/sharedscratch/vclip/target/release/vclip"
RULES="${PROJECT_DIR}/data/vclip_rules.tsv"

mkdir -p "${OUTDIR}"

BAM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${BAM_LIST}")

if [ -z "${BAM}" ]; then
    echo "No BAM for task ${SLURM_ARRAY_TASK_ID}"
    exit 0
fi

bash scripts/bam2pat_cfdna.sh "${BAM}" "${OUTDIR}" "${WGBSTOOLS}" "${VCLIP}" "${RULES}" 4
