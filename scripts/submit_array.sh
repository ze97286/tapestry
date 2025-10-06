#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_array
#$ -t 1-130
#$ -l h_vmem=50G
#$ -l h_rt=2:00:00
#$ -j y
#$ -o /users/zetzioni/sharedscratch/logs/array_$TASK_ID.log

cd /users/zetzioni/sharedscratch/tapestry || exit 1

INPUT_DIR="/mnt/lustre/users/bschuster/OAC_Trial_TAPS_cfDNA/Results/1.7/MethylationCalls/"
SAMPLE_FILES=($(ls $INPUT_DIR/*.calls.bed.gz | sort))
SAMPLE_FILE=${SAMPLE_FILES[$((SGE_TASK_ID - 1))]}

echo "Processing: $(basename $SAMPLE_FILE)"

python src/scripts/prepare_data_single.py \
    --sample-file "$SAMPLE_FILE" \
    --regions-file data/processed/regions.csv \
    --output-dir data/processed/samples \
    --config config/default_config.yaml
