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
