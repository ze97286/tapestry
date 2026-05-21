#!/bin/bash
# Common bootstrap for MethylBERT Slurm jobs.
#
# This intentionally does not require slurm/common.sh, because that file is
# tied to older cluster paths/module names in some environments.

bootstrap_methylbert_job() {
    if [ -z "${PROJECT_DIR:-}" ]; then
        PROJECT_DIR="$(pwd)"
        export PROJECT_DIR
    fi

    if [ -n "${METHYLBERT_CONFIG:-}" ]; then
        source "${METHYLBERT_CONFIG}"
    fi

    case "${METHYLBERT_STEP:-}" in
        dmr|DMR)
            METHYLBERT_STEP_MODULES="${METHYLBERT_STEP_MODULES:-${METHYLBERT_DMR_MODULES:-}}"
            ;;
    esac

    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/run_v0.5_methylbert}"
    export OUTPUT_DIR

    mkdir -p logs
    cd "${PROJECT_DIR}"
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
    export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

    if [ -n "${METHYLBERT_MODULE_INIT:-}" ]; then
        eval "${METHYLBERT_MODULE_INIT}"
    fi

    local modules_to_load
    modules_to_load="${METHYLBERT_MODULES:-} ${METHYLBERT_STEP_MODULES:-}"

    if [ -n "${modules_to_load// /}" ]; then
        if ! command -v module >/dev/null 2>&1; then
            echo "METHYLBERT_MODULES is set but the module command is unavailable. Set METHYLBERT_MODULE_INIT to source the cluster module init script, or clear METHYLBERT_MODULES and use METHYLBERT_ENV_COMMAND." >&2
            return 1
        fi
        for module_name in ${modules_to_load}; do
            module load "${module_name}"
        done
    fi

    if [ -n "${METHYLBERT_ENV_COMMAND:-}" ]; then
        eval "${METHYLBERT_ENV_COMMAND}"
    fi
}
