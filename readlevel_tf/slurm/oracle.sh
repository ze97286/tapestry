#!/bin/bash
#SBATCH --job-name=rltf_oracle
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rltf_oracle_%j.out
#SBATCH --error=logs/rltf_oracle_%j.err
#
# Step 2 — read-level separability oracle (gate). Config-driven. Submit from repo root.
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
CONFIG="${CONFIG:-readlevel_tf/configs/config.toml}"
python "${RLTF_DIR}/scripts/run_oracle.py" --config "${CONFIG}"
