#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=long
#SBATCH --array=1-22
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/preprocess_%a.out
#SBATCH --error=logs/preprocess_%a.err

source slurm/common.sh

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 \
        chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 \
        chr20 chr21 chr22)
CHROM=${CHROMS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "Processing ${CHROM} on $(hostname)"

python scripts/preprocess.py \
    --config ${CONFIG} \
    --chrom ${CHROM}
