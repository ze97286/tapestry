#!/bin/bash
#SBATCH --job-name=rltf_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rltf_env_%j.out
#SBATCH --error=logs/rltf_env_%j.err
#
# Build the rltf venv + pre-fetch the TabICL checkpoint. Submit from the repo
# root. If the short partition has no internet egress, run on a login node:
#     bash readlevel_tf/scripts/setup_env.sh
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh || true
bash "${RLTF_DIR}/scripts/setup_env.sh"
