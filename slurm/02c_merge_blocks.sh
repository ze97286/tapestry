#!/bin/bash
#SBATCH --job-name=merge_blocks
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_blocks.out
#SBATCH --error=logs/merge_blocks.err

source slurm/common.sh

SEG_DIR="${OUTPUT_DIR}/segmentation"

echo "=== Merging per-chromosome blocks ==="
zcat ${SEG_DIR}/blocks_chr*.bed.gz | sort -k1,1 -k2,2n | gzip > "${SEG_DIR}/blocks.bed.gz"

N_BLOCKS=$(zcat "${SEG_DIR}/blocks.bed.gz" | wc -l)
echo "  Total blocks (>=4 CpGs, autosomes): ${N_BLOCKS}"
echo "  Output: ${SEG_DIR}/blocks.bed.gz"
