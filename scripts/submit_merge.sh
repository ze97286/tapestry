#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N tapestry_merge
#$ -l h_vmem=380G
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
