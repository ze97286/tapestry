#!/bin/bash
# Create the Python/R environment used by the MethylBERT pipeline.

set -euo pipefail

if [ -z "${PROJECT_DIR:-}" ]; then
    PROJECT_DIR="$(pwd)"
    export PROJECT_DIR
fi

if [ -n "${METHYLBERT_CONFIG:-}" ]; then
    source "${METHYLBERT_CONFIG}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/run_v0.5_methylbert}"
METHYLBERT_DIR="${METHYLBERT_DIR:-${PROJECT_DIR}/external/methylbert}"
METHYLBERT_VENV="${METHYLBERT_VENV:-${OUTPUT_DIR}/envs/methylbert}"
METHYLBERT_R_LIBS="${METHYLBERT_R_LIBS:-${OUTPUT_DIR}/envs/R/methylbert}"
METHYLBERT_VENV_SCOPE="${METHYLBERT_VENV_SCOPE:-dmr}"
METHYLBERT_SETUP_R_DEPS="${METHYLBERT_SETUP_R_DEPS:-1}"
METHYLBERT_VENV_PYTHON="${METHYLBERT_VENV_PYTHON:-python3}"

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

if [ -n "${METHYLBERT_SETUP_MODULES:-}" ]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "METHYLBERT_SETUP_MODULES is set but the module command is unavailable" >&2
        exit 1
    fi
    for module_name in ${METHYLBERT_SETUP_MODULES}; do
        module load "${module_name}"
    done
fi

mkdir -p "$(dirname "${METHYLBERT_VENV}")" "${METHYLBERT_R_LIBS}"

if [ ! -x "${METHYLBERT_VENV}/bin/python" ] || [ "${FORCE_REBUILD_METHYLBERT_VENV:-0}" = "1" ]; then
    echo "Creating Python venv: ${METHYLBERT_VENV}"
    "${METHYLBERT_VENV_PYTHON}" -m venv --clear "${METHYLBERT_VENV}"
fi

source "${METHYLBERT_VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

case "${METHYLBERT_VENV_SCOPE}" in
    dmr)
        python -m pip install pysam pandas numpy
        ;;
    full)
        if [ ! -d "${METHYLBERT_DIR}/src/methylbert" ]; then
            echo "Missing upstream MethylBERT source: ${METHYLBERT_DIR}/src/methylbert" >&2
            exit 1
        fi
        python -m pip install -e "${METHYLBERT_DIR}"
        ;;
    *)
        echo "Unknown METHYLBERT_VENV_SCOPE=${METHYLBERT_VENV_SCOPE}; expected dmr or full" >&2
        exit 1
        ;;
esac

python - <<'PY'
import pysam
print("pysam", pysam.__version__)
PY

if [ "${METHYLBERT_SETUP_R_DEPS}" = "1" ]; then
    if ! command -v Rscript >/dev/null 2>&1; then
        echo "Rscript is required to install/check DSS dependencies" >&2
        exit 1
    fi
    export R_LIBS_USER="${METHYLBERT_R_LIBS}"
    Rscript - <<'RS'
lib <- Sys.getenv("R_LIBS_USER")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))

cran <- "https://cloud.r-project.org"
if (!requireNamespace("optparse", quietly = TRUE)) {
  install.packages("optparse", repos = cran)
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = cran)
}
if (!requireNamespace("DSS", quietly = TRUE)) {
  BiocManager::install("DSS", ask = FALSE, update = FALSE)
}

suppressPackageStartupMessages({
  library(optparse)
  library(DSS)
})
cat("R dependencies OK\n")
RS
fi

echo "Done."
echo "METHYLBERT_VENV=${METHYLBERT_VENV}"
echo "METHYLBERT_R_LIBS=${METHYLBERT_R_LIBS}"
