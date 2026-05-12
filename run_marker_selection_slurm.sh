#!/bin/bash
#SBATCH --job-name=robust_markers
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/robust_markers_%j.out
#SBATCH --error=logs/robust_markers_%j.err

# End-to-end marker selection for coverage-weighted UXM NNLS with an unknown
# component.  By default this script is an orchestrator: if upstream beta,
# segmentation, or homog files are missing, it submits the required SLURM jobs
# and then submits itself as the final marker-selection stage.
#
# Produces:
#   ${MARKERS_DIR}/candidate_markers.tsv
#   ${MARKERS_DIR}/markers.tsv
#   ${MARKERS_DIR}/markers.bed
#   ${MARKERS_DIR}/markers.diagnostics.json
#   ${MARKERS_DIR}/markers.selection_annotations.tsv
#
# Submit:
#   sbatch run_marker_selection_slurm.sh
#
# Common overrides:
#   sbatch --export=ALL,MARKERS_DIR=/path/to/out run_marker_selection_slurm.sh
#   sbatch --export=ALL,TARGET_CELL_TYPE=OAC,TOP_TARGET=150 run_marker_selection_slurm.sh

set -euo pipefail

source slurm/common.sh

PIPELINE_STAGE="${PIPELINE_STAGE:-orchestrate}"
MARKERS_DIR="${MARKERS_DIR:-${OUTPUT_DIR}/markers_unknown_robust}"
HOMOG_DIR="${HOMOG_DIR:-${OUTPUT_DIR}/homog}"
BETA_DIR="${BETA_DIR:-${OUTPUT_DIR}/betas}"
SEG_DIR="${SEG_DIR:-${OUTPUT_DIR}/segmentation}"
BLOCKS_BED="${BLOCKS_BED:-${SEG_DIR}/blocks.bed}"
CANDIDATE_MARKERS_TSV="${CANDIDATE_MARKERS_TSV:-${MARKERS_DIR}/candidate_markers.tsv}"
FINAL_MARKERS_TSV="${FINAL_MARKERS_TSV:-${MARKERS_DIR}/markers.tsv}"
FINAL_MARKERS_BED="${FINAL_MARKERS_BED:-${MARKERS_DIR}/markers.bed}"

TARGET_CELL_TYPE="${TARGET_CELL_TYPE:-OAC}"
EXCLUDE_FROM_BACKBONE="${EXCLUDE_FROM_BACKBONE:-}"
UNKNOWN_BASIS_NPY="${UNKNOWN_BASIS_NPY:-}"

CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-750}"
MIN_SNR="${MIN_SNR:-2.0}"
MIN_SIGNAL="${MIN_SIGNAL:-0.2}"
MAX_BG="${MAX_BG:-0.2}"
MAX_SINGLE_BG="${MAX_SINGLE_BG:-0.35}"
MIN_CONSISTENCY="${MIN_CONSISTENCY:-0.1}"
MIN_COV_PER_SAMPLE="${MIN_COV_PER_SAMPLE:-5}"

TOP_BACKBONE_PER_CELL="${TOP_BACKBONE_PER_CELL:-100}"
TOP_TARGET="${TOP_TARGET:-100}"
TARGET_CANDIDATE_POOL="${TARGET_CANDIDATE_POOL:-300}"
CONDITION_PENALTY="${CONDITION_PENALTY:-0}"
MIN_SINGLE_MARKER_SCORE="${MIN_SINGLE_MARKER_SCORE:-0}"
CONE_RECONSTRUCTION="${CONE_RECONSTRUCTION:-0}"
# Expensive upstream stages are opt-in.  The default is to use existing homog
# outputs and fail if they are incomplete, rather than silently launching
# pat2beta/segmentation/homog again.
RUN_UPSTREAM="${RUN_UPSTREAM:-0}"
FORCE_REBUILD_MARKERS="${FORCE_REBUILD_MARKERS:-0}"

mkdir -p "${MARKERS_DIR}" logs

echo "=== robust marker selection ==="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HOMOG_DIR=${HOMOG_DIR}"
echo "BETA_DIR=${BETA_DIR}"
echo "SEG_DIR=${SEG_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "MARKERS_DIR=${MARKERS_DIR}"
echo "TARGET_CELL_TYPE=${TARGET_CELL_TYPE}"
echo "PIPELINE_STAGE=${PIPELINE_STAGE}"

manifest_count() {
    tail -n +2 "${MANIFEST}" | awk 'NF > 0 {n++} END {print n+0}'
}

count_glob() {
    local pattern="$1"
    find "$(dirname "${pattern}")" -maxdepth 1 -name "$(basename "${pattern}")" 2>/dev/null | wc -l
}

blocks_are_monotonic() {
    local blocks="$1"
    if [ ! -f "${blocks}" ]; then
        return 1
    fi
    awk -F'\t' '
        NR > 1 && $4 < prev { exit 1 }
        { prev = $4 }
    ' "${blocks}"
}

submit_orchestration() {
    local n_ref
    n_ref=$(manifest_count)
    if [ "${n_ref}" -eq 0 ]; then
        echo "ERROR: no reference samples found in MANIFEST=${MANIFEST}"
        exit 1
    fi

    mkdir -p "${BETA_DIR}" "${SEG_DIR}" "${HOMOG_DIR}"

    local beta_count block_count homog_count dependency final_dependency jid
    beta_count=$(count_glob "${BETA_DIR}/*.beta")
    block_count=$(count_glob "${SEG_DIR}/blocks_chr*.bed.gz")
    homog_count=$(count_glob "${HOMOG_DIR}/*.uxm.bed.gz")
    dependency=""

    echo "Reference samples in manifest: ${n_ref}"
    echo "Existing beta files: ${beta_count}"
    echo "Existing per-chromosome block files: ${block_count}"
    echo "Existing homog files: ${homog_count}"

    if [ "${homog_count}" -ge "${n_ref}" ]; then
        echo "Homog files are complete; skipping all upstream beta/segmentation/homog submission."
    elif [ "${RUN_UPSTREAM}" != "1" ]; then
        if [ "${homog_count}" -lt "${n_ref}" ]; then
            echo "ERROR: RUN_UPSTREAM=0 but HOMOG_DIR is incomplete: ${HOMOG_DIR}"
            exit 1
        fi
    else
        if [ "${beta_count}" -lt "${n_ref}" ]; then
            jid=$(sbatch --parsable --array=1-"${n_ref}" slurm/02a_pat2beta.sh)
            echo "Submitted pat2beta array: ${jid}"
            dependency="--dependency=afterok:${jid}"
        else
            echo "Skipping pat2beta submission; beta files already present."
        fi

        if [ "${block_count}" -lt 22 ]; then
            if [ -n "${dependency}" ]; then
                jid=$(sbatch --parsable "${dependency}" --array=1-22 slurm/02b_segment.sh)
            else
                jid=$(sbatch --parsable --array=1-22 slurm/02b_segment.sh)
            fi
            echo "Submitted segmentation array: ${jid}"
            dependency="--dependency=afterok:${jid}"
        else
            echo "Skipping segmentation submission; chromosome block files already present."
        fi

        if ! blocks_are_monotonic "${BLOCKS_BED}"; then
            if [ -n "${dependency}" ]; then
                jid=$(sbatch --parsable "${dependency}" slurm/02c_merge_blocks.sh)
            else
                jid=$(sbatch --parsable slurm/02c_merge_blocks.sh)
            fi
            echo "Submitted block merge job: ${jid}"
            dependency="--dependency=afterok:${jid}"
        else
            echo "Skipping block merge submission; ${BLOCKS_BED} exists and startCpG is monotonic."
        fi

        if [ "${homog_count}" -lt "${n_ref}" ]; then
            if [ -n "${dependency}" ]; then
                jid=$(sbatch --parsable "${dependency}" --array=1-"${n_ref}" slurm/03_homog.sh)
            else
                jid=$(sbatch --parsable --array=1-"${n_ref}" slurm/03_homog.sh)
            fi
            echo "Submitted reference homog array: ${jid}"
            dependency="--dependency=afterok:${jid}"
        else
            echo "Skipping homog submission; homog files already present."
        fi
    fi

    if [ -n "${dependency}" ]; then
        final_dependency="${dependency}"
    else
        final_dependency=""
    fi

    if [ "${FORCE_REBUILD_MARKERS}" != "1" ] && [ -f "${FINAL_MARKERS_TSV}" ] && [ -f "${FINAL_MARKERS_BED}" ]; then
        echo "Final marker outputs already exist:"
        echo "  ${FINAL_MARKERS_TSV}"
        echo "  ${FINAL_MARKERS_BED}"
        echo "Set FORCE_REBUILD_MARKERS=1 to rebuild."
        exit 0
    fi

    if [ -n "${final_dependency}" ]; then
        jid=$(sbatch --parsable "${final_dependency}" --export=ALL,PIPELINE_STAGE=select run_marker_selection_slurm.sh)
    else
        jid=$(sbatch --parsable --export=ALL,PIPELINE_STAGE=select run_marker_selection_slurm.sh)
    fi
    echo "Submitted final robust marker-selection job: ${jid}"
    echo "Pipeline submitted. Final outputs will be written to ${MARKERS_DIR}"
}

if [ "${PIPELINE_STAGE}" = "orchestrate" ]; then
    submit_orchestration
    exit 0
fi

if [ "${PIPELINE_STAGE}" != "select" ]; then
    echo "ERROR: unknown PIPELINE_STAGE=${PIPELINE_STAGE}"
    exit 1
fi

if [ ! -d "${HOMOG_DIR}" ] || [ "$(count_glob "${HOMOG_DIR}/*.uxm.bed.gz")" -eq 0 ]; then
    echo "ERROR: no reference homog files found in ${HOMOG_DIR}"
    echo "Run with PIPELINE_STAGE=orchestrate/RUN_UPSTREAM=1 to build upstream inputs."
    exit 1
fi

echo
echo "Step 1/3: build broad candidate UXM atlas"
python scripts/select_markers.py \
    --homog-dir "${HOMOG_DIR}" \
    --manifest "${MANIFEST}" \
    --output "${CANDIDATE_MARKERS_TSV}" \
    --top-n "${CANDIDATE_TOP_N}" \
    --min-snr "${MIN_SNR}" \
    --min-signal "${MIN_SIGNAL}" \
    --max-bg "${MAX_BG}" \
    --max-single-bg "${MAX_SINGLE_BG}" \
    --min-consistency "${MIN_CONSISTENCY}" \
    --min-cov-per-sample "${MIN_COV_PER_SAMPLE}" \
    --direction U

echo
echo "Step 2/3: select unknown-robust marker subset"
ROBUST_ARGS=(
    --atlas "${CANDIDATE_MARKERS_TSV}"
    --output "${FINAL_MARKERS_TSV}"
    --diagnostics-output "${MARKERS_DIR}/markers.diagnostics.json"
    --annotation-output "${MARKERS_DIR}/markers.selection_annotations.tsv"
    --target-cell-type "${TARGET_CELL_TYPE}"
    --exclude-from-backbone "${EXCLUDE_FROM_BACKBONE}"
    --top-backbone-per-cell "${TOP_BACKBONE_PER_CELL}"
    --top-target "${TOP_TARGET}"
    --target-candidate-pool "${TARGET_CANDIDATE_POOL}"
    --condition-penalty "${CONDITION_PENALTY}"
    --min-single-marker-score "${MIN_SINGLE_MARKER_SCORE}"
)

if [ -n "${UNKNOWN_BASIS_NPY}" ]; then
    ROBUST_ARGS+=(--unknown-basis-npy "${UNKNOWN_BASIS_NPY}")
fi

if [ "${CONE_RECONSTRUCTION}" = "1" ]; then
    ROBUST_ARGS+=(--cone-reconstruction)
fi

python scripts/select_unknown_robust_markers.py "${ROBUST_ARGS[@]}"

echo
echo "Step 3/3: write markers BED for PAT filtering/homog"
python -c '
import sys
import pandas as pd
markers_tsv, markers_bed = sys.argv[1], sys.argv[2]
df = pd.read_csv(markers_tsv, sep="\t")
required = ["chr", "start", "end", "startCpG", "endCpG"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"missing columns for BED: {missing}")
bed = df[required].sort_values(["startCpG", "chr", "start"], kind="mergesort")
if (bed["startCpG"].diff().dropna() < 0).any():
    raise SystemExit("failed to sort markers BED by startCpG")
bed.to_csv(markers_bed, sep="\t", header=False, index=False)
print(f"wrote {len(df)} regions to {markers_bed}")
' "${FINAL_MARKERS_TSV}" "${FINAL_MARKERS_BED}"

echo
echo "Done."
echo "Final atlas: ${FINAL_MARKERS_TSV}"
echo "Final BED:   ${FINAL_MARKERS_BED}"
