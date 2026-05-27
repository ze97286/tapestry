#!/bin/bash
#SBATCH --job-name=mbert_rcdec
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/methylbert_deconvolute_read_calls_%A_%a.out
#SBATCH --error=logs/methylbert_deconvolute_read_calls_%A_%a.err

set -euo pipefail

export METHYLBERT_STEP=DECONVOLUTE
source scripts/methylbert_common.sh
bootstrap_methylbert_job
source scripts/methylbert_reference.sh

RUN_LABEL="${RUN_LABEL:-oac_methylbert_paper}"
METHYLBERT_DIR="${METHYLBERT_DIR:-${PROJECT_DIR}/external/methylbert}"
METHYLBERT_WORK_DIR="${METHYLBERT_WORK_DIR:-${OUTPUT_DIR}/methylbert/${RUN_LABEL}}"
MODEL_DIR="${MODEL_DIR:-${METHYLBERT_WORK_DIR}/model_taps_read_calls_collapsed_100kb}"
DECONV_DIR="${DECONV_DIR:-${METHYLBERT_WORK_DIR}/deconvolution_taps_read_calls_collapsed_100kb}"
METHYLBERT_DMRS="${METHYLBERT_DMRS:-${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv}"

READ_CALL_MODE="${READ_CALL_MODE:-overlap}"
DMR_START_BASE="${DMR_START_BASE:-1}"
MIN_INFORMATIVE="${MIN_INFORMATIVE:-2}"
READ_CALL_MAX_READS_PER_SAMPLE="${READ_CALL_MAX_READS_PER_SAMPLE:-200000}"
STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE="${STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE:-0}"
DECONV_BATCH_SIZE="${DECONV_BATCH_SIZE:-128}"
ADJUSTMENT="${ADJUSTMENT:-0}"

mkdir -p logs "${DECONV_DIR}"

if [ -z "${METHYLBERT_BULK_READ_CALL_LIST:-}" ] || [ ! -s "${METHYLBERT_BULK_READ_CALL_LIST:-}" ]; then
    echo "Set METHYLBERT_BULK_READ_CALL_LIST to a file listing cfDNA .per-read.bed.gz files to deconvolute" >&2
    exit 1
fi
if [ ! -s "${METHYLBERT_DMRS}" ]; then
    echo "Missing DMRs: ${METHYLBERT_DMRS}" >&2
    exit 1
fi
if [ ! -s "${MODEL_DIR}/train_param.txt" ] || [ ! -d "${MODEL_DIR}/bert.model" ]; then
    echo "Missing fine-tuned model under ${MODEL_DIR}" >&2
    exit 1
fi

METHYLBERT_REF_STAGE_DIR="${METHYLBERT_REF_STAGE_DIR:-${METHYLBERT_WORK_DIR}/reference_stage}"
export METHYLBERT_REF_STAGE_DIR
stage_methylbert_reference

mapfile -t read_calls < <(awk 'NF && $1 !~ /^#/ {print $1}' "${METHYLBERT_BULK_READ_CALL_LIST}")
if [ "${#read_calls[@]}" -eq 0 ]; then
    echo "No read-call files found in ${METHYLBERT_BULK_READ_CALL_LIST}" >&2
    exit 1
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    idx=$((SLURM_ARRAY_TASK_ID - 1))
    if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#read_calls[@]}" ]; then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside read-call list length ${#read_calls[@]}" >&2
        exit 1
    fi
    selected_read_calls=("${read_calls[$idx]}")
else
    selected_read_calls=("${read_calls[@]}")
fi

export PYTHONPATH="${METHYLBERT_DIR}/src:${PYTHONPATH:-}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    echo "python3 or python is required for MethylBERT read-call deconvolution." >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = ["numpy", "pandas", "scipy", "sklearn", "torch", "tqdm", "transformers"]
missing = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name} ({exc})")

if missing:
    sys.stderr.write(
        "Missing Python modules for MethylBERT read-call deconvolution after module load:\n"
        + "\n".join(f"  - {entry}" for entry in missing)
        + "\nSet METHYLBERT_DECONVOLUTE_MODULES or METHYLBERT_FINETUNE_MODULES to a BMRC module stack that provides them.\n"
    )
    raise SystemExit(1)
PY

echo "=== MethylBERT TAPS read-call deconvolution ==="
echo "METHYLBERT_WORK_DIR=${METHYLBERT_WORK_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "METHYLBERT_DMRS=${METHYLBERT_DMRS}"
echo "METHYLBERT_REF_FASTA=${METHYLBERT_REF_FASTA}"
echo "METHYLBERT_BULK_READ_CALL_LIST=${METHYLBERT_BULK_READ_CALL_LIST}"
echo "DECONV_DIR=${DECONV_DIR}"
echo "READ_CALL_MAX_READS_PER_SAMPLE=${READ_CALL_MAX_READS_PER_SAMPLE}"
echo "PYTHON_BIN=${PYTHON_BIN}"

for read_call in "${selected_read_calls[@]}"; do
    if [ ! -s "${read_call}" ]; then
        echo "Missing read-call file: ${read_call}" >&2
        exit 1
    fi

    sample="$(basename "${read_call}")"
    sample="${sample%.per-read.bed.gz}"
    sample="${sample%.bed.gz}"
    sample="${sample%.gz}"
    sample_dir="${DECONV_DIR}/${sample}"
    bulk_preprocess="${sample_dir}/preprocess"
    sample_sheet="${bulk_preprocess}/sample_sheet.tsv"
    mkdir -p "${bulk_preprocess}" "${sample_dir}"

    printf "%s\tN\n" "${read_call}" > "${sample_sheet}"

    echo "Preprocessing ${sample}"
    "${PYTHON_BIN}" scripts/preprocess_methylbert_taps_read_calls.py \
        --sample-sheet "${sample_sheet}" \
        --dmrs "${METHYLBERT_DMRS}" \
        --reference "${METHYLBERT_REF_FASTA}" \
        --output-dir "${bulk_preprocess}" \
        --rows-output "${bulk_preprocess}/data.csv" \
        --summary-output "${bulk_preprocess}/read_call_preprocess_summary.tsv" \
        --mode "${READ_CALL_MODE}" \
        --dmr-start-base "${DMR_START_BASE}" \
        --min-informative "${MIN_INFORMATIVE}" \
        --max-reads-per-sample "${READ_CALL_MAX_READS_PER_SAMPLE}" \
        --stop-after-output-rows-per-sample "${STOP_AFTER_OUTPUT_ROWS_PER_SAMPLE}" \
        --max-reads-per-label 0
    test -s "${bulk_preprocess}/data.csv"

    deconv_args=(
        scripts/run_upstream_methylbert.py deconvolute
        --input_data "${bulk_preprocess}/data.csv"
        --model_dir "${MODEL_DIR}/"
        --output_path "${sample_dir}"
        --batch_size "${DECONV_BATCH_SIZE}"
    )
    if [ "${ADJUSTMENT}" = "1" ]; then
        deconv_args+=(--adjustment)
    fi

    echo "Deconvoluting ${sample}"
    "${PYTHON_BIN}" "${deconv_args[@]}"
    test -s "${sample_dir}/deconvolution.csv"
done

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    "${PYTHON_BIN}" scripts/collect_methylbert_deconvolution.py \
        --deconvolution-dir "${DECONV_DIR}" \
        --output "${DECONV_DIR}/deconvolution_summary.csv"
    echo "Summary: ${DECONV_DIR}/deconvolution_summary.csv"
else
    echo "Array task complete. Run run_methylbert_paper_collect_slurm.sh after all array tasks finish."
fi

echo "Done."
