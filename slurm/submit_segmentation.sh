#!/bin/bash
# Submit the segmentation pipeline: pat2beta → segment (parallel) → merge
set -euo pipefail

source slurm/common.sh

# Fix line endings
sed -i 's/\r$//' "${MANIFEST}"

# Create directories
mkdir -p "${OUTPUT_DIR}/betas" "${OUTPUT_DIR}/segmentation" logs

# Submit jobs
BETA_JOB=$(sbatch --parsable slurm/02a_pat2beta.sh)
echo "pat2beta: ${BETA_JOB}"

SEG_JOB=$(sbatch --parsable --dependency=afterok:${BETA_JOB} slurm/02b_segment.sh)
echo "segment (22 chromosomes): ${SEG_JOB}"

MERGE_JOB=$(sbatch --parsable --dependency=afterok:${SEG_JOB} slurm/02c_merge_blocks.sh)
echo "merge blocks: ${MERGE_JOB}"

echo ""
echo "Monitor: squeue -u $(whoami)"
