#!/bin/bash
#SBATCH --job-name=rl_llr_oracle
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rl_llr_oracle_%j.out
#SBATCH --error=logs/rl_llr_oracle_%j.err
#
# Read-level likelihood-ratio oracle — the decisive test of whether
# tumour-vs-healthy methylation is separable at the single-fragment level on a
# given block panel. See docs/read_level_detection_direction.md.
#
# Streaming, single-threaded, CPU-only: reads marker-filtered PAT files once
# each, accumulates per-CpG reference profiles on the train split, scores
# held-out reads, computes AUC + length/permutation controls. Expected runtime
# is tens of minutes to ~1-2 h on the filtered_pats/ panels; ~a few GB RAM.
# Point it at the marker-FILTERED PATs (runs/.../filtered_pats), not raw
# genome-wide PATs, to keep memory and time bounded.
#
# Required inputs (override via --export=ALL,VAR=...):
#   REGIONS_BED   block panel BED (chrom/start/end in cols 1-3)
#   CPG_INDEX     genome CpG index (wgbs CpG.bed.gz or tapestry tsv.gz)
#   SAMPLES_TSV   header: sample_id<TAB>group<TAB>cohort<TAB>file_path
#                 group in {tumour,healthy}; cohort e.g. OAC_tissue/AB/CD
#   OUT_DIR       output directory for summary.json etc.
#   PANEL_NAME    label recorded in the summary (e.g. tight_blocks / mbert_dmrs)
# Optional:
#   HOLDOUT_COHORT   force a healthy cohort (e.g. CD) into the test split
#   TEST_FRACTION    per-group random test fraction (default 0.3)
#   MIN_CPGS_OVERLAP min CpGs a read must share with a block (default 3)
#   SEED             RNG seed (default 0)

set -euo pipefail
source slurm/common.sh

: "${REGIONS_BED:?set REGIONS_BED}"
: "${CPG_INDEX:?set CPG_INDEX}"
: "${SAMPLES_TSV:?set SAMPLES_TSV}"
: "${OUT_DIR:?set OUT_DIR}"
PANEL_NAME="${PANEL_NAME:-panel}"
TEST_FRACTION="${TEST_FRACTION:-0.3}"
MIN_CPGS_OVERLAP="${MIN_CPGS_OVERLAP:-3}"
SEED="${SEED:-0}"

mkdir -p "${OUT_DIR}" logs

EXTRA_ARGS=()
if [[ -n "${HOLDOUT_COHORT:-}" ]]; then
    EXTRA_ARGS+=(--holdout-cohort "${HOLDOUT_COHORT}")
fi
if [[ -n "${HOLDOUT_SAMPLES:-}" ]]; then
    EXTRA_ARGS+=(--holdout-samples "${HOLDOUT_SAMPLES}")
fi

python scripts/run_read_level_llr_oracle.py \
    --regions-bed "${REGIONS_BED}" \
    --cpg-index "${CPG_INDEX}" \
    --samples-tsv "${SAMPLES_TSV}" \
    --out-dir "${OUT_DIR}" \
    --panel-name "${PANEL_NAME}" \
    --min-cpgs-overlap "${MIN_CPGS_OVERLAP}" \
    --test-fraction "${TEST_FRACTION}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"

# ---------------------------------------------------------------------------
# Example: the two-arm geometry test from the direction doc.
#
# Arm 1 — tight CpG-dense segmentation blocks (expected to SEPARATE if the
# read-level signal exists):
#   sbatch --export=ALL,\
# REGIONS_BED=${OUTPUT_DIR}/segmentation/blocks.bed,\
# CPG_INDEX=/path/to/CpG.bed.gz,\
# SAMPLES_TSV=${PROJECT_DIR}/data/oracle_samples.tsv,\
# OUT_DIR=${OUTPUT_DIR}/readlevel_oracle/tight_blocks,\
# PANEL_NAME=tight_blocks,HOLDOUT_COHORT=CD slurm/oracle_read_level_llr.sh
#
# Arm 2 — broad MethylBERT DSS DMRs (negative geometry control; expected to be
# WEAKER if the user's "regions too broad / too few CpGs per read" hypothesis
# holds):
#   sbatch --export=ALL,\
# REGIONS_BED=/path/to/dmrs_top100.collapsed_100kb.bed,\
# CPG_INDEX=/path/to/CpG.bed.gz,\
# SAMPLES_TSV=${PROJECT_DIR}/data/oracle_samples.tsv,\
# OUT_DIR=${OUTPUT_DIR}/readlevel_oracle/mbert_dmrs,\
# PANEL_NAME=mbert_dmrs,HOLDOUT_COHORT=CD slurm/oracle_read_level_llr.sh
# ---------------------------------------------------------------------------
