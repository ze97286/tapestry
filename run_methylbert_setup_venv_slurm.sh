#!/bin/bash
#SBATCH --job-name=mbert_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/methylbert_env_%j.out
#SBATCH --error=logs/methylbert_env_%j.err

set -euo pipefail

mkdir -p logs

if [ -z "${PROJECT_DIR:-}" ]; then
    PROJECT_DIR="$(pwd)"
    export PROJECT_DIR
fi

scripts/setup_methylbert_venv.sh
