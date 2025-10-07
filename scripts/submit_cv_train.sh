#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_cv
#$ -pe smp 16
#$ -l h_vmem=20G
#$ -l h_rt=48:00:00
#$ -j y
#$ -o /users/zetzioni/sharedscratch/logs/cv_train.log

cd /users/zetzioni/sharedscratch/tapestry || exit 1

# Set PyTorch to use all allocated cores
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

echo "Starting TAPESTRY Cross-Validation Training..."
echo "Using $NSLOTS cores"
date

python src/scripts/train_cv.py \
    --data-dir /users/zetzioni/sharedscratch/tapestry/data/processed \
    --sample-ids /users/zetzioni/sharedscratch/tapestry/data/processed/sample_ids.csv \
    --ichorcna-file /users/zetzioni/sharedscratch/tapestry/data/ichorcna_tumor_fractions.csv \
    --output-dir /users/zetzioni/sharedscratch/tapestry/results/cv_run_001 \
    --n-folds 5 \
    --epochs 200 \
    --lr 1e-4 \
    --device cpu

echo "Cross-validation complete!"
date
