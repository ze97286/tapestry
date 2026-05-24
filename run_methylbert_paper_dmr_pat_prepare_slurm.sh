#!/bin/bash
#SBATCH --job-name=mbert_dmr_prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/methylbert_dmr_pat_prepare_%j.out
#SBATCH --error=logs/methylbert_dmr_pat_prepare_%j.err

set -euo pipefail

export METHYLBERT_STEP=DMR
source scripts/methylbert_common.sh
bootstrap_methylbert_job

scripts/prepare_methylbert_dmr_pat_inputs.sh
