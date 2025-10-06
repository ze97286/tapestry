#!/bin/bash
# TAPESTRY - Complete Pipeline
#
# Usage: bash scripts/submit_pipeline.sh

set -e

cd /users/zetzioni/sharedscratch/tapestry

echo "=========================================="
echo "TAPESTRY Pipeline"
echo "=========================================="

# Step 1: Define regions (run locally, fast)
echo ""
echo "Step 1: Defining regions..."
python src/scripts/prepare_regions.py \
    --output-file /users/zetzioni/sharedscratch/tapestry/data/processed/regions.csv \
    --config config/default_config.yaml
echo "✓ Regions saved"

# Step 2: Submit job array (130 samples in parallel)
echo ""
echo "Step 2: Submitting job array (130 samples)..."

# Create logs directory
mkdir -p /users/zetzioni/sharedscratch/logs

# Submit array job
ARRAY_JOB=$(qsub <<'ARRAY_EOF' | awk '{print $3}')
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_array
#$ -t 1-130
#$ -l h_vmem=32G
#$ -l h_rt=2:00:00
#$ -j y
#$ -o /users/zetzioni/sharedscratch/logs/array_$TASK_ID.log

cd /users/zetzioni/sharedscratch/tapestry || exit 1

INPUT_DIR="/mnt/lustre/processed/Lu_lab/OAC_Immuno_Trial/TAPS_cfDNA/Results/1.6.1/MethylationCalls"
SAMPLE_FILES=($(ls $INPUT_DIR/*.calls.bed.gz | sort))
SAMPLE_FILE=${SAMPLE_FILES[$((SGE_TASK_ID - 1))]}

echo "Processing: $(basename $SAMPLE_FILE)"

python src/scripts/prepare_data_single.py \
    --sample-file "$SAMPLE_FILE" \
    --regions-file data/processed/regions.csv \
    --output-dir data/processed/samples \
    --config config/default_config.yaml
ARRAY_EOF
)

echo "✓ Job array submitted: $ARRAY_JOB"

# Step 3: Submit merge job (waits for array)
echo ""
echo "Step 3: Submitting merge job (waits for array)..."

MERGE_JOB=$(qsub -hold_jid $ARRAY_JOB <<'MERGE_EOF' | awk '{print $3}')
#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_merge
#$ -l h_vmem=200G
#$ -l h_rt=2:00:00
#$ -j y
#$ -o /users/zetzioni/sharedscratch/logs/merge.log

cd /users/zetzioni/sharedscratch/tapestry || exit 1

echo "Merging 130 samples..."

python src/scripts/merge_samples.py \
    --input-dir data/processed/samples \
    --regions-file data/processed/regions.csv \
    --output-dir data/processed \
    --config config/default_config.yaml
MERGE_EOF
)

echo "✓ Merge job submitted: $MERGE_JOB"

echo ""
echo "=========================================="
echo "Pipeline submitted!"
echo "=========================================="
echo ""
echo "Monitor:"
echo "  qstat -u $USER"
echo ""
echo "Array progress:"
echo "  qstat -j $ARRAY_JOB | grep tasks"
echo ""
echo "Results will be in:"
echo "  data/processed/methylation_matrix.npy"
echo "  data/processed/coverage_matrix.npy"
echo "=========================================="
