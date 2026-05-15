#!/bin/bash
#SBATCH --job-name=ctrl_deconv
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/control_deconv_%A_%a.out
#SBATCH --error=logs/control_deconv_%A_%a.err

# Array wrapper: evaluate one hard-control marker panel per array task.
#
# Submit after run_control_constrained_marker_sweep_slurm.sh:
#   N=$(tail -n +2 "${SWEEP_DIR}/sweep_summary.tsv" | wc -l)
#   sbatch --array=1-"${N}" --export=ALL run_control_constrained_deconvolution_sweep_slurm.sh

set -euo pipefail

source slurm/common.sh

COHORT="${COHORT:-AB}"
RUN_LABEL_BASE="${RUN_LABEL:-AB_ctrl56_v04_candidates}"
SWEEP_DIR="${SWEEP_DIR:-${OUTPUT_DIR}/markers_control_constrained_sweep/${RUN_LABEL_BASE}}"
SWEEP_SUMMARY="${SWEEP_SUMMARY:-${SWEEP_DIR}/sweep_summary.tsv}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_DIR}/evaluation_control_constrained_sweep/${RUN_LABEL_BASE}}"
ICHORCNA_FILE="${ICHORCNA_FILE:-${PROJECT_DIR}/data/cfDNA_tumour_fraction_ichorCNA.json}"
MATERIALIZE_LAMBDA="${MATERIALIZE_LAMBDA:-0}"

CFDNA_INPUT_DIR="${CFDNA_INPUT_DIR:-${PROJECT_DIR}/data/${COHORT}}"
CFDNA_EXTRA_CONTROL_DIRS="${CFDNA_EXTRA_CONTROL_DIRS:-${PROJECT_DIR}/data/CD}"
CONTROL_PATTERN="${CONTROL_PATTERN:-Ctrl|healthy|^(GI|SCAN)}"
EXTRA_CONTROL_PATTERN="${EXTRA_CONTROL_PATTERN:-^(GI|SCAN)}"
CONTROL_CROSSFIT_FOLDS="${CONTROL_CROSSFIT_FOLDS:-5}"
UNKNOWN_FIT_EXCLUDE_CELL_TYPES="${UNKNOWN_FIT_EXCLUDE_CELL_TYPES:-OAC}"
ORTHOGONALIZE_TARGET="${ORTHOGONALIZE_TARGET:-}"
FORCE_REFILTER="${FORCE_REFILTER:-1}"
HOMOG_LEN="${HOMOG_LEN:-4}"
LAMBDA_UNKNOWN_GRID="${LAMBDA_UNKNOWN_GRID:-0,0.01,0.1,1,10,100,1000,10000}"
PRIMARY_LAMBDA_UNKNOWN="${PRIMARY_LAMBDA_UNKNOWN:-0}"

if [ ! -f "${SWEEP_SUMMARY}" ]; then
    echo "ERROR: sweep summary not found: ${SWEEP_SUMMARY}"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
PANEL_ROW=$(python - "${SWEEP_SUMMARY}" "${TASK_ID}" <<'PY'
import sys
import pandas as pd

summary_path, task_id = sys.argv[1], int(sys.argv[2])
df = pd.read_csv(summary_path, sep="\t")
idx = task_id - 1
if idx < 0 or idx >= len(df):
    raise SystemExit(f"array task {task_id} outside panel range 1..{len(df)}")
row = df.iloc[idx]
print(row["panel"], row["markers_tsv"], row["markers_bed"], int(row["n_final_markers"]), sep="\t")
PY
)
IFS=$'\t' read -r PANEL MARKERS_TSV MARKERS_BED N_FINAL_MARKERS <<< "${PANEL_ROW}"

if [ "${N_FINAL_MARKERS}" -lt 1 ]; then
    echo "Skipping ${PANEL}: no selected markers"
    exit 0
fi

PANEL_RUN_LABEL="${RUN_LABEL_BASE}_${PANEL}"
PANEL_EVAL_DIR="${EVAL_ROOT}/${PANEL}"
PRED_DIR="${OUTPUT_DIR}/predictions_control_constrained_sweep"
PRED_OUTPUT="${PRED_DIR}/${PANEL_RUN_LABEL}_unknown_robust_nnls.csv"
PATH_OUTPUT="${PRED_DIR}/${PANEL_RUN_LABEL}_unknown_robust_nnls_lambda_path.csv"
MATERIALIZED_OUTPUT="${PRED_DIR}/${PANEL_RUN_LABEL}_lam${MATERIALIZE_LAMBDA}.csv"

mkdir -p "${PRED_DIR}" "${PANEL_EVAL_DIR}" logs

echo "=== control-constrained deconvolution sweep panel ==="
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PANEL=${PANEL}"
echo "MARKERS_TSV=${MARKERS_TSV}"
echo "MARKERS_BED=${MARKERS_BED}"
echo "PANEL_RUN_LABEL=${PANEL_RUN_LABEL}"
echo "PRED_OUTPUT=${PRED_OUTPUT}"
echo "PATH_OUTPUT=${PATH_OUTPUT}"

export COHORT RUN_LABEL="${PANEL_RUN_LABEL}"
export MARKERS_TSV MARKERS_BED MARKERS_DIR
MARKERS_DIR="$(dirname "${MARKERS_TSV}")"
export CFDNA_INPUT_DIR CFDNA_EXTRA_CONTROL_DIRS CONTROL_PATTERN EXTRA_CONTROL_PATTERN
export CONTROL_CROSSFIT_FOLDS UNKNOWN_FIT_EXCLUDE_CELL_TYPES ORTHOGONALIZE_TARGET
export FORCE_REFILTER HOMOG_LEN LAMBDA_UNKNOWN_GRID PRIMARY_LAMBDA_UNKNOWN
export PRED_DIR PRED_OUTPUT PATH_OUTPUT

bash run_deconvolution_slurm.sh

python scripts/evaluate_unknown_lambda_path.py \
    --lambda-path "${PATH_OUTPUT}" \
    --ichorcna-file "${ICHORCNA_FILE}" \
    --output "${PANEL_EVAL_DIR}/lambda_tradeoff.csv"

python scripts/materialize_unknown_lambda.py \
    --lambda-path "${PATH_OUTPUT}" \
    --predictions "${PRED_OUTPUT}" \
    --lambda-unknown "${MATERIALIZE_LAMBDA}" \
    --output "${MATERIALIZED_OUTPUT}"

echo
echo "Done."
echo "Lambda tradeoff: ${PANEL_EVAL_DIR}/lambda_tradeoff.csv"
echo "Materialized lambda ${MATERIALIZE_LAMBDA}: ${MATERIALIZED_OUTPUT}"
