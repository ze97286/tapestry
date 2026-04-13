#!/bin/bash
#SBATCH --job-name=prepare_index
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prepare_index.out
#SBATCH --error=logs/prepare_index.err

source slurm/common.sh

python scripts/prepare_index.py \
    --config ${CONFIG}
