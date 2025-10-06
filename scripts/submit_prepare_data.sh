#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_prepare
#$ -pe smp 32
#$ -l h_vmem=12G
#$ -l h_rt=12:00:00
#$ -j y
#$ -o logs/prepare_data_$JOB_ID.log

# TAPESTRY Data Preparation - SGE Submission Script
#
# This script submits the data preparation pipeline to SGE with:
# - 32 cores (parallel processing)
# - 12GB RAM per core = 384GB total
# - 12 hour time limit
#
# Usage:
#   qsub scripts/submit_prepare_data.sh

# Change to project directory
cd /users/zetzioni/sharedscratch/tapestry || { echo "ERROR: Cannot cd to /users/zetzioni/sharedscratch/tapestry"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "TAPESTRY Data Preparation"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo "Job Name: $JOB_NAME"
echo "Hostname: $HOSTNAME"
echo "Working Directory: $(pwd)"
echo "Cores: $NSLOTS"
echo "Memory: 12GB per core = $((12 * NSLOTS))GB total"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Default arguments
INPUT_DIR="/mnt/lustre/processed/Lu_lab/OAC_Immuno_Trial/TAPS_cfDNA/Results/1.6.1/MethylationCalls"
OUTPUT_DIR="data/processed"
PATTERN="*.calls.bed.gz"
CONFIG="config/default_config.yaml"

echo "Command: python src/scripts/prepare_data.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR --pattern \"$PATTERN\" --config $CONFIG"
echo ""

# Set number of threads for numpy/pandas
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export OPENBLAS_NUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

# Print system info
echo "System Information:"
free -h
echo ""
lscpu | grep -E "^CPU\(s\)|Model name|Thread|Core"
echo ""

# Run the pipeline
echo "Starting pipeline..."
python src/scripts/prepare_data.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --pattern "$PATTERN" \
    --config "$CONFIG"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Print resource usage
if [ -n "$JOB_ID" ]; then
    echo ""
    echo "Resource Usage:"
    qstat -j $JOB_ID | grep -E "usage|vmem|maxvmem|cpu" || true
fi

exit $EXIT_CODE
