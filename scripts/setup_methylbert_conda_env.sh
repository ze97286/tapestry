#!/bin/bash
# Create or update the conda prefix used by the MethylBERT DMR/runtime steps.

set -euo pipefail

if [ -z "${PROJECT_DIR:-}" ]; then
    PROJECT_DIR="$(pwd)"
    export PROJECT_DIR
fi

if [ -n "${METHYLBERT_CONFIG:-}" ]; then
    source "${METHYLBERT_CONFIG}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/run_v0.5_methylbert}"
METHYLBERT_CONDA_PREFIX="${METHYLBERT_CONDA_PREFIX:-${OUTPUT_DIR}/envs/methylbert_dmr_conda}"
METHYLBERT_CONDA_ENV_YML="${METHYLBERT_CONDA_ENV_YML:-${PROJECT_DIR}/configs/methylbert_dmr_conda.yml}"

if [ ! -s "${METHYLBERT_CONDA_ENV_YML}" ]; then
    echo "Missing conda environment file: ${METHYLBERT_CONDA_ENV_YML}" >&2
    exit 1
fi

if [ -n "${METHYLBERT_MODULE_INIT:-}" ]; then
    eval "${METHYLBERT_MODULE_INIT}"
fi

if [ -n "${METHYLBERT_MODULE_USE:-}" ]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "METHYLBERT_MODULE_USE is set but the module command is unavailable" >&2
        exit 1
    fi
    old_ifs="${IFS}"
    IFS=":"
    for module_path in ${METHYLBERT_MODULE_USE}; do
        [ -n "${module_path}" ] || continue
        module use "${module_path}"
    done
    IFS="${old_ifs}"
fi

if [ -n "${METHYLBERT_CONDA_MODULES:-}" ]; then
    if command -v module >/dev/null 2>&1; then
        for module_name in ${METHYLBERT_CONDA_MODULES}; do
            module load "${module_name}"
        done
    else
        echo "METHYLBERT_CONDA_MODULES is set but the module command is unavailable" >&2
        exit 1
    fi
fi

if command -v mamba >/dev/null 2>&1; then
    conda_cmd=(mamba)
elif command -v conda >/dev/null 2>&1; then
    conda_cmd=(conda)
elif command -v micromamba >/dev/null 2>&1; then
    conda_cmd=(micromamba)
else
    echo "Could not find mamba, conda, or micromamba on PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "${METHYLBERT_CONDA_PREFIX}")"

echo "Conda command: ${conda_cmd[*]}"
echo "Conda prefix: ${METHYLBERT_CONDA_PREFIX}"
echo "Environment file: ${METHYLBERT_CONDA_ENV_YML}"

if [ -d "${METHYLBERT_CONDA_PREFIX}/conda-meta" ]; then
    echo "Updating existing conda prefix"
    "${conda_cmd[@]}" env update -p "${METHYLBERT_CONDA_PREFIX}" -f "${METHYLBERT_CONDA_ENV_YML}" --prune
else
    echo "Creating conda prefix"
    "${conda_cmd[@]}" env create -p "${METHYLBERT_CONDA_PREFIX}" -f "${METHYLBERT_CONDA_ENV_YML}"
fi

export PATH="${METHYLBERT_CONDA_PREFIX}/bin:${PATH}"

python - <<'PY'
import numpy
import pandas
import pysam
print("python OK")
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("pysam", pysam.__version__)
PY

samtools --version | head -n 1

Rscript -e 'suppressPackageStartupMessages({ library(optparse); library(DSS) }); cat("R/DSS OK\n")'

echo "Done."
echo "METHYLBERT_CONDA_PREFIX=${METHYLBERT_CONDA_PREFIX}"
