#!/bin/bash
#SBATCH --job-name=rltf_oracle
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rltf_oracle_%j.out
#SBATCH --error=logs/rltf_oracle_%j.err
#
# Step 2 — read-level separability oracle (gate). Submit from repo root.
# Required: MANIFEST, OUT_DIR. Optional: WINDOW TOP_N MIN_TOTAL MIN_EFFECT
#           DIRECTION HOLDOUT_COHORT TEST_FRACTION SEED.
set -euo pipefail
mkdir -p logs
source readlevel_tf/slurm/common.sh
: "${MANIFEST:?set MANIFEST}"
: "${OUT_DIR:?set OUT_DIR}"

EXTRA=()
[ -n "${HOLDOUT_COHORT:-}" ] && EXTRA+=(--holdout-cohort "${HOLDOUT_COHORT}")
[ -n "${HOLDOUT_SAMPLES:-}" ] && EXTRA+=(--holdout-samples "${HOLDOUT_SAMPLES}")

python "${RLTF_DIR}/scripts/run_oracle.py" \
    --manifest "${MANIFEST}" --out-dir "${OUT_DIR}" \
    --window "${WINDOW:-5}" --top-n "${TOP_N:-2000}" \
    --min-total "${MIN_TOTAL:-10}" --min-effect "${MIN_EFFECT:-0.3}" \
    --direction "${DIRECTION:-any}" --test-fraction "${TEST_FRACTION:-0.3}" \
    --seed "${SEED:-0}" "${EXTRA[@]}"
