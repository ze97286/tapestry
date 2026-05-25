#!/bin/bash
# Common bootstrap for MethylBERT Slurm jobs.
#
# This intentionally does not require slurm/common.sh, because that file is
# tied to older cluster paths/module names in some environments.

_methylbert_prepend_env_path() {
    local var_name="$1"
    local path_value="$2"
    local current_value

    [ -n "${path_value}" ] || return 0
    [ -d "${path_value}" ] || return 0

    current_value="${!var_name:-}"
    case ":${current_value}:" in
        *:"${path_value}":*)
            ;;
        *)
            if [ -n "${current_value}" ]; then
                export "${var_name}=${path_value}:${current_value}"
            else
                export "${var_name}=${path_value}"
            fi
            ;;
    esac
}

_methylbert_remove_env_path() {
    local var_name="$1"
    local path_value="$2"
    local current_value
    local new_value=""
    local path_part
    local old_ifs

    [ -n "${path_value}" ] || return 0

    current_value="${!var_name:-}"
    [ -n "${current_value}" ] || return 0

    old_ifs="${IFS}"
    IFS=":"
    for path_part in ${current_value}; do
        [ "${path_part}" = "${path_value}" ] && continue
        if [ -n "${new_value}" ]; then
            new_value="${new_value}:${path_part}"
        else
            new_value="${path_part}"
        fi
    done
    IFS="${old_ifs}"
    export "${var_name}=${new_value}"
}

_methylbert_deactivate_conda() {
    local conda_prefix
    local conda_exe
    local conda_bin
    local conda_root
    local conda_hook
    local guard=0
    local had_nounset=0

    [ "${METHYLBERT_DEACTIVATE_CONDA:-0}" = "1" ] || return 0

    conda_prefix="${CONDA_PREFIX:-}"
    conda_exe="${CONDA_EXE:-}"

    if [ -z "${conda_prefix}" ] && [ -z "${CONDA_SHLVL:-}" ] && [ -z "${conda_exe}" ]; then
        return 0
    fi

    echo "Deactivating conda before MethylBERT job bootstrap"

    if command -v conda >/dev/null 2>&1; then
        case "$-" in
            *u*)
                had_nounset=1
                set +u
                ;;
        esac
        if conda_hook="$(conda shell.bash hook 2>/dev/null)"; then
            eval "${conda_hook}"
        fi
        while [ "${CONDA_SHLVL:-0}" -gt 0 ] && [ "${guard}" -lt 10 ]; do
            conda deactivate >/dev/null 2>&1 || break
            guard=$((guard + 1))
        done
        if [ "${had_nounset}" = "1" ]; then
            set -u
        fi
    fi

    if [ -n "${conda_prefix}" ]; then
        _methylbert_remove_env_path PATH "${conda_prefix}/bin"
        _methylbert_remove_env_path PATH "${conda_prefix}/condabin"
    fi
    if [ -n "${conda_exe}" ] && [ -x "${conda_exe}" ]; then
        conda_bin="$(dirname "${conda_exe}")"
        conda_root="$(dirname "${conda_bin}")"
        _methylbert_remove_env_path PATH "${conda_bin}"
        _methylbert_remove_env_path PATH "${conda_root}/condabin"
    fi

    unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PROMPT_MODIFIER
    unset CONDA_PYTHON_EXE CONDA_SHLVL _CE_CONDA _CE_M
    unset CC CXX FC F77 CPP LD AR RANLIB CFLAGS CXXFLAGS FFLAGS FCFLAGS CPPFLAGS LDFLAGS
}

_methylbert_add_library_dirs_from_flags() {
    local token
    local libdir

    for token in "$@"; do
        case "${token}" in
            -L*)
                libdir="${token#-L}"
                _methylbert_prepend_env_path LD_LIBRARY_PATH "${libdir}"
                _methylbert_prepend_env_path LIBRARY_PATH "${libdir}"
                ;;
            -Wl,-rpath,*)
                libdir="${token#-Wl,-rpath,}"
                _methylbert_prepend_env_path LD_LIBRARY_PATH "${libdir}"
                ;;
        esac
    done
}

_methylbert_add_library_dirs_from_prefixes() {
    local raw_prefixes
    local prefix
    local root
    local old_ifs

    old_ifs="${IFS}"
    IFS=":"
    for raw_prefixes in "$@"; do
        for prefix in ${raw_prefixes}; do
            [ -n "${prefix}" ] || continue
            root="${prefix%/.}"
            root="${root%/}"
            _methylbert_prepend_env_path LD_LIBRARY_PATH "${root}/lib"
            _methylbert_prepend_env_path LD_LIBRARY_PATH "${root}/lib64"
            _methylbert_prepend_env_path LIBRARY_PATH "${root}/lib"
            _methylbert_prepend_env_path LIBRARY_PATH "${root}/lib64"
        done
    done
    IFS="${old_ifs}"
}

_methylbert_add_library_dirs_from_module_vars() {
    local var_name
    local root

    for var_name in \
        ICU4CDIR PCRE2DIR LIBTIRPCDIR KRB5DIR LIBEDITDIR CURLDIR OPENSSLDIR \
        LIBIDN2DIR LIBUNISTRINGDIR NGHTTP2DIR LIBICONVDIR XZDIR BZIP2DIR \
        ZLIBDIR ZLIB_NGDIR ZSTDDIR NCURSESDIR GCC_RUNTIMEDIR OPENBLASDIR
    do
        root="${!var_name:-}"
        [ -n "${root}" ] || continue
        _methylbert_add_library_dirs_from_prefixes "${root}"
    done
}

_methylbert_get_r_cmd_config() {
    local config_name="$1"
    local config_value

    if config_value="$(R CMD config "${config_name}" 2>/dev/null)"; then
        printf '%s\n' "${config_value}"
    else
        printf '\n'
    fi
}

_methylbert_configure_r_runtime_paths() {
    local r_ldflags
    local r_libs_flags

    command -v R >/dev/null 2>&1 || return 0

    r_ldflags="$(_methylbert_get_r_cmd_config LDFLAGS)"
    r_libs_flags="$(_methylbert_get_r_cmd_config LIBS)"

    # Some R modules expose dependent libraries, for example ICU, only through
    # R's -L flags. Add those paths for configure probes and package loading.
    _methylbert_add_library_dirs_from_module_vars
    _methylbert_add_library_dirs_from_prefixes ${CMAKE_PREFIX_PATH:-}
    _methylbert_add_library_dirs_from_flags ${r_ldflags}
    _methylbert_add_library_dirs_from_flags ${r_libs_flags}
}

bootstrap_methylbert_job() {
    if [ -z "${PROJECT_DIR:-}" ]; then
        PROJECT_DIR="$(pwd)"
        export PROJECT_DIR
    fi

    if [ -n "${METHYLBERT_CONFIG:-}" ]; then
        source "${METHYLBERT_CONFIG}"
    fi

    _methylbert_deactivate_conda

    case "${METHYLBERT_STEP:-}" in
        dmr|DMR)
            METHYLBERT_STEP_MODULES="${METHYLBERT_STEP_MODULES:-${METHYLBERT_DMR_MODULES:-}}"
            ;;
        preprocess_pat|PREPROCESS_PAT)
            METHYLBERT_STEP_MODULES="${METHYLBERT_STEP_MODULES:-${METHYLBERT_PREPROCESS_PAT_MODULES:-}}"
            ;;
        finetune|FINETUNE)
            METHYLBERT_STEP_MODULES="${METHYLBERT_STEP_MODULES:-${METHYLBERT_FINETUNE_MODULES:-}}"
            ;;
        deconvolute|DECONVOLUTE)
            METHYLBERT_STEP_MODULES="${METHYLBERT_STEP_MODULES:-${METHYLBERT_DECONVOLUTE_MODULES:-${METHYLBERT_FINETUNE_MODULES:-}}}"
            ;;
        collect|COLLECT)
            METHYLBERT_STEP_MODULES="${METHYLBERT_STEP_MODULES:-${METHYLBERT_COLLECT_MODULES:-${METHYLBERT_DECONVOLUTE_MODULES:-${METHYLBERT_FINETUNE_MODULES:-}}}}"
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

    if [ -n "${METHYLBERT_MODULE_USE:-}" ]; then
        if ! command -v module >/dev/null 2>&1; then
            echo "METHYLBERT_MODULE_USE is set but the module command is unavailable" >&2
            return 1
        fi
        local module_path
        local old_ifs
        old_ifs="${IFS}"
        IFS=":"
        for module_path in ${METHYLBERT_MODULE_USE}; do
            [ -n "${module_path}" ] || continue
            module use "${module_path}"
        done
        IFS="${old_ifs}"
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

    _methylbert_configure_r_runtime_paths

    if [ -n "${METHYLBERT_ENV_COMMAND:-}" ]; then
        eval "${METHYLBERT_ENV_COMMAND}"
    fi
}
