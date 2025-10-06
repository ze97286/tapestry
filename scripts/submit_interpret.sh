#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_interpret
#$ -l h_vmem=32G
#$ -l h_rt=2:00:00
#$ -j y
#$ -o /users/zetzioni/sharedscratch/logs/interpret.log

cd /users/zetzioni/sharedscratch/tapestry || exit 1

echo "Starting TAPESTRY interpretation..."
date

python src/scripts/interpret.py \
    --model-path /users/zetzioni/sharedscratch/tapestry/models/run_002/best_model.pt \
    --data-dir /users/zetzioni/sharedscratch/tapestry/data/processed \
    --output-dir /users/zetzioni/sharedscratch/tapestry/results/run_002_interpretation \
    --ichorcna-file /users/zetzioni/sharedscratch/tapestry/data/ichorcna_tumor_fractions.csv \
    --config config/default_config.yaml \
    --device cpu

echo "Interpretation complete!"
date
