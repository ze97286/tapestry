#!/bin/bash
# Reference FASTA staging helper for MethylBERT jobs.

stage_methylbert_reference() {
    local require_index
    require_index="${METHYLBERT_REF_REQUIRE_INDEX:-1}"

    _methylbert_index_fasta() {
        local fasta_path="$1"
        if command -v samtools >/dev/null 2>&1; then
            samtools faidx "${fasta_path}"
            return 0
        fi
        if command -v python3 >/dev/null 2>&1; then
            python3 scripts/index_fasta.py "${fasta_path}"
            return 0
        fi
        if command -v python >/dev/null 2>&1; then
            python scripts/index_fasta.py "${fasta_path}"
            return 0
        fi
        echo "samtools or python is required to index FASTA ${fasta_path}" >&2
        return 1
    }

    if [ -n "${METHYLBERT_REF_FASTA:-}" ] && [ -s "${METHYLBERT_REF_FASTA}" ] && [[ "${METHYLBERT_REF_FASTA}" != *.gz ]]; then
        if [ "${require_index}" = "1" ] && [ ! -s "${METHYLBERT_REF_FASTA}.fai" ]; then
            _methylbert_index_fasta "${METHYLBERT_REF_FASTA}"
        fi
        return 0
    fi

    if [ -n "${METHYLBERT_REF_FASTA:-}" ] && [[ "${METHYLBERT_REF_FASTA}" == *.gz ]] && [ -z "${METHYLBERT_REF_FASTA_GZ:-}" ]; then
        export METHYLBERT_REF_FASTA_GZ="${METHYLBERT_REF_FASTA}"
        export METHYLBERT_REF_FASTA=""
    fi

    if [ -z "${METHYLBERT_REF_FASTA_GZ:-}" ] || [ ! -s "${METHYLBERT_REF_FASTA_GZ}" ]; then
        echo "Set METHYLBERT_REF_FASTA to an existing .fa, or METHYLBERT_REF_FASTA_GZ to an existing .fa.gz" >&2
        return 1
    fi
    if ! command -v gzip >/dev/null 2>&1; then
        echo "gzip is required to stage METHYLBERT_REF_FASTA_GZ" >&2
        return 1
    fi

    local stage_root
    local ref_name
    local staged_ref
    local lock_dir

    stage_root="${METHYLBERT_REF_STAGE_DIR:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/methylbert_ref_${SLURM_JOB_ID:-$$}}"
    mkdir -p "${stage_root}"

    ref_name="$(basename "${METHYLBERT_REF_FASTA_GZ}")"
    ref_name="${ref_name%.gz}"
    staged_ref="${stage_root}/${ref_name}"
    lock_dir="${staged_ref}.lock"

    if [ -s "${staged_ref}" ] && { [ "${require_index}" != "1" ] || [ -s "${staged_ref}.fai" ]; }; then
        export METHYLBERT_REF_FASTA="${staged_ref}"
        return 0
    fi

    if mkdir "${lock_dir}" 2>/dev/null; then
        trap 'rm -rf "${lock_dir}"' RETURN
        echo "Staging reference FASTA to ${staged_ref}"
        gzip -dc "${METHYLBERT_REF_FASTA_GZ}" > "${staged_ref}"
        if [ "${require_index}" = "1" ]; then
            _methylbert_index_fasta "${staged_ref}"
        fi
        rm -rf "${lock_dir}"
        trap - RETURN
    else
        echo "Waiting for reference staging lock: ${lock_dir}"
        while [ -d "${lock_dir}" ]; do
            sleep 10
        done
        if [ ! -s "${staged_ref}" ] || { [ "${require_index}" = "1" ] && [ ! -s "${staged_ref}.fai" ]; }; then
            echo "Reference staging lock cleared but staged FASTA is missing: ${staged_ref}" >&2
            return 1
        fi
    fi

    export METHYLBERT_REF_FASTA="${staged_ref}"
}
