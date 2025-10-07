#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_array
#$ -t 1-52
#$ -l h_vmem=120G
#$ -l h_rt=2:00:00
#$ -j y
#$ -o /users/zetzioni/sharedscratch/logs/array_$TASK_ID.log

cd /users/zetzioni/sharedscratch/tapestry || exit 1

INPUT_DIR="/mnt/lustre/users/bschuster/OAC_Trial_TAPS_cfDNA_CD/Results/1.1/MethylationCalls/"
# Filter to only GI or SCAN prefixes, excluding lambda
SAMPLE_FILES=($(ls $INPUT_DIR/*.calls.bed.gz | grep -E '(GI|SCAN)' | grep -v lambda | sort))

# Check if array is empty
if [ ${#SAMPLE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No files found matching filter (GI|SCAN, excluding lambda)"
    exit 1
fi

SAMPLE_FILE=${SAMPLE_FILES[$((SGE_TASK_ID - 1))]}

# Check if file exists
if [ ! -f "$SAMPLE_FILE" ]; then
    echo "ERROR: Sample file not found or invalid: $SAMPLE_FILE"
    echo "Total files in array: ${#SAMPLE_FILES[@]}"
    echo "Task ID: $SGE_TASK_ID"
    exit 1
fi

echo "Processing: $(basename $SAMPLE_FILE)"

python src/scripts/prepare_data_single.py \
    --sample-file "$SAMPLE_FILE" \
    --regions-file data/processed/regions.csv \
    --output-dir data/processed/samples \
    --config config/default_config.yaml
