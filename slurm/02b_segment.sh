#!/bin/bash
#SBATCH --job-name=segment
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/segment_%a.out
#SBATCH --error=logs/segment_%a.err

source slurm/common.sh

BETA_DIR="${OUTPUT_DIR}/betas"
SEG_DIR="${OUTPUT_DIR}/segmentation"
mkdir -p "${SEG_DIR}"

CHR="chr${SLURM_ARRAY_TASK_ID}"
BETA_FILES=$(ls ${BETA_DIR}/*.beta 2>/dev/null | tr '\n' ' ')
N_BETAS=$(ls ${BETA_DIR}/*.beta 2>/dev/null | wc -l)

echo "Segmenting ${CHR} with ${N_BETAS} beta files"

${WGBSTOOLS} segment \
    --betas ${BETA_FILES} \
    -r "${CHR}" \
    --max_cpg 1000 \
    --max_bp 2000 \
    --min_cpg 4 \
    -@ 8 \
    -o "${SEG_DIR}/blocks_${CHR}.bed" \
    2>&1

N_BLOCKS=$(wc -l < "${SEG_DIR}/blocks_${CHR}.bed" 2>/dev/null || echo 0)
echo "  ${CHR}: ${N_BLOCKS} blocks"

gzip "${SEG_DIR}/blocks_${CHR}.bed"
