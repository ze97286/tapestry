#!/bin/bash
#SBATCH --job-name=bam2pat_cf
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/bam2pat_cfdna_%j.out
#SBATCH --error=logs/bam2pat_cfdna_%j.err

# Convert cfDNA BAMs to PAT files with vclip fragment-length-dependent clipping.
# Removes artefactual hypomethylation from TAPS cfDNA data.
#
# Usage:
#   sbatch --export=ALL,COHORT=AB slurm/10_bam2pat_cfdna.sh
#   sbatch --export=ALL,COHORT=CD slurm/10_bam2pat_cfdna.sh

source slurm/common.sh

if [ -z "${COHORT:-}" ]; then
    echo "ERROR: COHORT not set. Use --export=ALL,COHORT=AB or COHORT=CD"
    exit 1
fi

bash scripts/bam2pat_cfdna.sh "${COHORT}"
