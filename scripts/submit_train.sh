#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_train
#$ -pe smp 16
#$ -l h_vmem=20G
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /users/zetzioni/sharedscratch/logs/blind_train.log

cd /users/zetzioni/sharedscratch/tapestry || exit 1

# Set PyTorch to use all allocated cores
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

echo "Starting TAPESTRY training..."
echo "Using $NSLOTS cores"
date

python src/scripts/train_model.py \
    --data-dir /users/zetzioni/sharedscratch/tapestry/data/processed \
    --output-dir /users/zetzioni/sharedscratch/tapestry/models/run_004 \
    --config config/sparse_config.yaml \
    --epochs 100000 \
    --device cpu

echo "Training complete!"
date
