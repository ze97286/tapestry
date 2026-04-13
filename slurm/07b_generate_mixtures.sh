#!/bin/bash
#SBATCH --job-name=gen_mix
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_mix_%a.out
#SBATCH --error=logs/gen_mix_%a.err

# Generate synthetic mixtures from filtered reference PATs.
# Array job — each task processes BATCH_SIZE mixtures.
#
# Usage:
#   # For training data (100K samples, 1000 per batch = 100 tasks):
#   sbatch --array=0-99 --export=ALL,DATASET=train slurm/07b_generate_mixtures.sh
#
#   # For evaluation data (20K samples, 1000 per batch = 20 tasks):
#   sbatch --array=0-19 --export=ALL,DATASET=eval slurm/07b_generate_mixtures.sh
#
#   # For OAC dilution series:
#   N_OAC=$(python3 -c "import pandas as pd; print((len(pd.read_csv('${OUTPUT_DIR}/training/oac_dilution_proportions.csv')) + 999) // 1000)")
#   sbatch --array=0-$((N_OAC-1)) --export=ALL,DATASET=oac_dilution slurm/07b_generate_mixtures.sh
#
#   # For T-cell dilution series:
#   N_TC=$(python3 -c "import pandas as pd; print((len(pd.read_csv('${OUTPUT_DIR}/training/tcell_dilution_proportions.csv')) + 999) // 1000)")
#   sbatch --array=0-$((N_TC-1)) --export=ALL,DATASET=tcell_dilution slurm/07b_generate_mixtures.sh

source slurm/common.sh

BATCH_SIZE=1000
TRAINING_DIR="${OUTPUT_DIR}/training"
FILTERED_DIR="${OUTPUT_DIR}/filtered_pats/ref"
MARKERS_BED="${OUTPUT_DIR}/markers/markers.bed"
MARKERS_TSV="${OUTPUT_DIR}/markers/markers.tsv"

if [ -z "${DATASET:-}" ]; then
    echo "ERROR: DATASET not set. Use --export=ALL,DATASET=train (or eval/oac_dilution/tcell_dilution)"
    exit 1
fi

PROPORTIONS="${TRAINING_DIR}/${DATASET}_proportions.csv"
OUT_DIR="${TRAINING_DIR}/${DATASET}"
mkdir -p "${OUT_DIR}"

BATCH_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))

echo "Dataset: ${DATASET}, batch start: ${BATCH_START}, batch size: ${BATCH_SIZE}"

python scripts/generate_mixtures.py \
    --proportions "${PROPORTIONS}" \
    --manifest "${MANIFEST}" \
    --filtered-dir "${FILTERED_DIR}" \
    --markers-bed "${MARKERS_BED}" \
    --atlas "${MARKERS_TSV}" \
    --output-dir "${OUT_DIR}" \
    --pattools "${PATTOOLS}" \
    --wgbstools "${WGBSTOOLS}" \
    --batch-start "${BATCH_START}" \
    --batch-size "${BATCH_SIZE}"
