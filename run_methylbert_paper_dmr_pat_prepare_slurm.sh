#!/bin/bash
#SBATCH --job-name=mbert_dmr_prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/methylbert_dmr_pat_prepare_%j.out
#SBATCH --error=logs/methylbert_dmr_pat_prepare_%j.err

set -euo pipefail

# The DMR pipeline gets its toolchain (python, Rscript, DSS) from conda; keep conda
# active rather than deactivating it (a module stack does not provide DSS here).
export METHYLBERT_STEP=DMR
export METHYLBERT_DEACTIVATE_CONDA=0
source scripts/methylbert_common.sh
bootstrap_methylbert_job

scripts/prepare_methylbert_dmr_pat_inputs.sh
