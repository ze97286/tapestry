#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_nn
#$ -pe smp 16
#$ -l h_vmem=20G
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /users/zetzioni/sharedscratch/logs/nn_train.log

cd /users/zetzioni/sharedscratch/tapestry || exit 1

# Set PyTorch to use all allocated cores
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

echo "Starting Supervised NN training..."
echo "Using $NSLOTS cores"
date

python src/tapestry/model/nn.py \
    --data-dir /users/zetzioni/sharedscratch/tapestry/data/processed \
    --ichorcna-file /users/zetzioni/sharedscratch/tapestry/data/ichorcna_tumor_fractions.csv \
    --output-dir /users/zetzioni/sharedscratch/tapestry/models/nn_run_001 \
    --epochs 100000 \
    --lr 1e-4 \
    --device cpu

echo "Training complete!"
date
