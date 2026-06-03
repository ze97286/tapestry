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

remove_env_path() {
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

deactivate_conda_for_setup() {
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

    echo "Deactivating conda before MethylBERT setup"

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
        remove_env_path PATH "${conda_prefix}/bin"
        remove_env_path PATH "${conda_prefix}/condabin"
    fi
    if [ -n "${conda_exe}" ] && [ -x "${conda_exe}" ]; then
        conda_bin="$(dirname "${conda_exe}")"
        conda_root="$(dirname "${conda_bin}")"
        remove_env_path PATH "${conda_bin}"
        remove_env_path PATH "${conda_root}/condabin"
    fi

    unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PROMPT_MODIFIER
    unset CONDA_PYTHON_EXE CONDA_SHLVL _CE_CONDA _CE_M
    unset CC CXX FC F77 CPP LD AR RANLIB CFLAGS CXXFLAGS FFLAGS FCFLAGS CPPFLAGS LDFLAGS
}

prepend_env_path() {
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

add_library_dirs_from_flags() {
    local token
    local libdir

    for token in "$@"; do
        case "${token}" in
            -L*)
                libdir="${token#-L}"
                prepend_env_path LD_LIBRARY_PATH "${libdir}"
                prepend_env_path LIBRARY_PATH "${libdir}"
                ;;
            -Wl,-rpath,*)
                libdir="${token#-Wl,-rpath,}"
                prepend_env_path LD_LIBRARY_PATH "${libdir}"
                ;;
        esac
    done
}

add_library_dirs_from_prefixes() {
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
            prepend_env_path LD_LIBRARY_PATH "${root}/lib"
            prepend_env_path LD_LIBRARY_PATH "${root}/lib64"
            prepend_env_path LIBRARY_PATH "${root}/lib"
            prepend_env_path LIBRARY_PATH "${root}/lib64"
        done
    done
    IFS="${old_ifs}"
}

add_library_dirs_from_module_vars() {
    local var_name
    local root

    for var_name in \
        ICU4CDIR PCRE2DIR LIBTIRPCDIR KRB5DIR LIBEDITDIR CURLDIR OPENSSLDIR \
        LIBIDN2DIR LIBUNISTRINGDIR NGHTTP2DIR LIBICONVDIR XZDIR BZIP2DIR \
        ZLIBDIR ZLIB_NGDIR ZSTDDIR NCURSESDIR GCC_RUNTIMEDIR OPENBLASDIR
    do
        root="${!var_name:-}"
        [ -n "${root}" ] || continue
        add_library_dirs_from_prefixes "${root}"
    done
}

get_r_cmd_config() {
    local config_name="$1"
    local config_value

    if config_value="$(R CMD config "${config_name}" 2>/dev/null)"; then
        printf '%s\n' "${config_value}"
    else
        printf '\n'
    fi
}

if [ -n "${METHYLBERT_MODULE_INIT:-}" ]; then
    eval "${METHYLBERT_MODULE_INIT}"
fi

deactivate_conda_for_setup

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

mkdir -p "${METHYLBERT_R_LIBS}"

case "${METHYLBERT_VENV_SCOPE}" in
    none|skip|false|0)
        echo "Skipping Python venv setup because METHYLBERT_VENV_SCOPE=${METHYLBERT_VENV_SCOPE}"
        ;;
    dmr)
        mkdir -p "$(dirname "${METHYLBERT_VENV}")"
        if [ ! -x "${METHYLBERT_VENV}/bin/python" ] || [ "${FORCE_REBUILD_METHYLBERT_VENV:-0}" = "1" ]; then
            echo "Creating Python venv: ${METHYLBERT_VENV}"
            "${METHYLBERT_VENV_PYTHON}" -m venv --clear "${METHYLBERT_VENV}"
        fi
        source "${METHYLBERT_VENV}/bin/activate"
        python -m pip install --upgrade pip setuptools wheel
        python -m pip install pysam pandas numpy
        python - <<'PY'
import pysam
print("pysam", pysam.__version__)
PY
        ;;
    full)
        mkdir -p "$(dirname "${METHYLBERT_VENV}")"
        if [ ! -x "${METHYLBERT_VENV}/bin/python" ] || [ "${FORCE_REBUILD_METHYLBERT_VENV:-0}" = "1" ]; then
            echo "Creating Python venv: ${METHYLBERT_VENV}"
            "${METHYLBERT_VENV_PYTHON}" -m venv --clear "${METHYLBERT_VENV}"
        fi
        source "${METHYLBERT_VENV}/bin/activate"
        python -m pip install --upgrade pip setuptools wheel
        if [ ! -d "${METHYLBERT_DIR}/src/methylbert" ]; then
            echo "Missing upstream MethylBERT source: ${METHYLBERT_DIR}/src/methylbert" >&2
            exit 1
        fi
        python -m pip install -e "${METHYLBERT_DIR}"
        ;;
    *)
        echo "Unknown METHYLBERT_VENV_SCOPE=${METHYLBERT_VENV_SCOPE}; expected none, dmr, or full" >&2
        exit 1
        ;;
esac

if [ "${METHYLBERT_SETUP_R_DEPS}" = "1" ]; then
    if ! command -v Rscript >/dev/null 2>&1; then
        echo "Rscript is required to install/check DSS dependencies; load a BMRC R module first or set METHYLBERT_SETUP_MODULES" >&2
        exit 1
    fi
    if command -v R >/dev/null 2>&1; then
        R_CPPFLAGS="$(get_r_cmd_config CPPFLAGS)"
        R_LDFLAGS="$(get_r_cmd_config LDFLAGS)"
        R_LIBS_FLAGS="$(get_r_cmd_config LIBS)"
        export R_CPPFLAGS R_LDFLAGS R_LIBS_FLAGS
        # Rhdf5lib runs compiled configure probes while building bundled HDF5.
        # R exposes libraries such as ICU through -L flags, but not always
        # through LD_LIBRARY_PATH, so make those runtime paths explicit.
        add_library_dirs_from_module_vars
        add_library_dirs_from_prefixes ${CMAKE_PREFIX_PATH:-}
        add_library_dirs_from_flags ${R_LDFLAGS}
        add_library_dirs_from_flags ${R_LIBS_FLAGS}
        echo "R_CPPFLAGS=${R_CPPFLAGS}"
        echo "R_LDFLAGS=${R_LDFLAGS}"
        echo "R_LIBS_FLAGS=${R_LIBS_FLAGS}"
    fi
    if command -v xml2-config >/dev/null 2>&1; then
        export XML_CONFIG="$(command -v xml2-config)"
        echo "XML_CONFIG=${XML_CONFIG}"
        XML_CFLAGS="$("${XML_CONFIG}" --cflags)"
        XML_LIBS="$("${XML_CONFIG}" --libs)"
        export XML_CFLAGS XML_LIBS
        add_library_dirs_from_flags ${XML_LIBS}
        for token in ${XML_CFLAGS}; do
            case "${token}" in
                -I*/include/libxml2)
                    export LIBXML_INCDIR="${token#-I}"
                    break
                    ;;
            esac
        done
        if [ -z "${LIBXML_INCDIR:-}" ]; then
            for token in ${XML_CFLAGS}; do
                case "${token}" in
                    -I*)
                        export LIBXML_INCDIR="${token#-I}"
                        break
                        ;;
                esac
            done
        fi
        for token in ${XML_LIBS}; do
            case "${token}" in
                -L*)
                    export LIBXML_LIBDIR="${token}"
                    break
                    ;;
            esac
        done
        echo "XML_CFLAGS=${XML_CFLAGS}"
        echo "XML_LIBS=${XML_LIBS}"
        echo "LIBXML_INCDIR=${LIBXML_INCDIR:-}"
        echo "LIBXML_LIBDIR=${LIBXML_LIBDIR:-}"
    else
        echo "xml2-config not found on PATH; XML/R dependency installation may fail" >&2
    fi
    echo "Rscript=$(command -v Rscript)"
    echo "gcc=$(command -v gcc || true)"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    export R_LIBS_USER="${METHYLBERT_R_LIBS}"
    Rscript - <<'RS'
lib <- Sys.getenv("R_LIBS_USER")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))

cran <- "https://cloud.r-project.org"
message("R executable: ", R.home("bin"))
message("R_LIBS_USER: ", Sys.getenv("R_LIBS_USER"))
message("xml2-config: ", Sys.getenv("XML_CONFIG"))
message("LIBXML_INCDIR: ", Sys.getenv("LIBXML_INCDIR"))
message("LIBXML_LIBDIR: ", Sys.getenv("LIBXML_LIBDIR"))
message(".libPaths: ", paste(.libPaths(), collapse = " | "))

if (!requireNamespace("XML", quietly = TRUE)) {
  xml_config <- Sys.getenv("XML_CONFIG")
  if (nzchar(xml_config)) {
    message("Installing XML with --with-xml-config=", xml_config)
    install.packages(
      "XML",
      repos = cran,
      configure.args = c(XML = paste0("--with-xml-config=", xml_config)),
      configure.vars = c(XML = paste0(
        "LIBXML_INCDIR=", Sys.getenv("LIBXML_INCDIR"),
        " LIBXML_LIBDIR=", Sys.getenv("LIBXML_LIBDIR")
      ))
    )
  } else {
    message("Installing XML without XML_CONFIG")
    install.packages("XML", repos = cran)
  }
}
if (!requireNamespace("XML", quietly = TRUE)) {
  stop("XML package installation failed; aborting before DSS dependency installation")
}
if (!requireNamespace("optparse", quietly = TRUE)) {
  install.packages("optparse", repos = cran)
}
if (!requireNamespace("optparse", quietly = TRUE)) {
  stop("optparse package installation failed")
}
if (!requireNamespace("Matrix", quietly = TRUE)) {
  if (getRversion() < "4.4.0") {
    message("Installing archived Matrix compatible with R ", getRversion())
    install.packages(
      "https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-5.tar.gz",
      repos = NULL,
      type = "source"
    )
  } else {
    install.packages("Matrix", repos = cran)
  }
}
if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("Matrix package installation failed")
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = cran)
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  stop("BiocManager package installation failed")
}
if (!requireNamespace("DSS", quietly = TRUE)) {
  BiocManager::install("DSS", ask = FALSE, update = FALSE)
}
if (!requireNamespace("DSS", quietly = TRUE)) {
  stop("DSS package installation failed")
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
