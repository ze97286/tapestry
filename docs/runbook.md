# Tapestry Pipeline Runbook

All commands run from the project root on BMRC.

## Step 1: PAT to beta

```bash
sbatch slurm/02a_pat2beta.sh
```

## Step 2: Segmentation

```bash
# After step 1
sbatch slurm/02b_segment.sh

# After all 22 chromosomes complete
sbatch slurm/02c_merge_blocks.sh
```

## Step 3: UXM homog counting

```bash
# After step 2
sbatch slurm/03_homog.sh
```

## Step 4: Marker selection

```bash
# After step 3
sbatch slurm/04_select_markers.sh
```

## Step 5: Visualise atlas

```bash
# After step 4
sbatch slurm/05_visualise_atlas.sh
```

## Step 6a: Filter reference PATs to marker regions

```bash
# After step 4
sbatch slurm/06a_filter_ref_pats.sh
```

## Step 6b: Filter cfDNA PATs (with TAPS flip)

```bash
# After 6a (needs markers.bed)
N_AB=$(ls /gpfs3/users/ludwig/uii408/sharedscratch/tapestry/data/AB/*.pat.gz | wc -l)
N_CD=$(ls /gpfs3/users/ludwig/uii408/sharedscratch/tapestry/data/CD/*.pat.gz | wc -l)

sbatch --array=1-${N_AB} --export=ALL,COHORT=AB slurm/06b_filter_cfdna_pats.sh
sbatch --array=1-${N_CD} --export=ALL,COHORT=CD slurm/06b_filter_cfdna_pats.sh
```

## Step 7a: Generate proportion tables

```bash
# After step 6a
sbatch slurm/07a_generate_proportions.sh
```

## Step 7b: Generate synthetic mixtures

```bash
# After 7a
sbatch --array=0-99 --export=ALL,DATASET=train slurm/07b_generate_mixtures.sh
sbatch --array=0-19 --export=ALL,DATASET=eval slurm/07b_generate_mixtures.sh
sbatch --array=0-1  --export=ALL,DATASET=oac_dilution slurm/07b_generate_mixtures.sh
sbatch --array=0-1  --export=ALL,DATASET=tcell_dilution slurm/07b_generate_mixtures.sh
```

## Step 7c: Collect into parquets

```bash
# After all 7b jobs complete
sbatch slurm/07c_collect_training_data.sh
```

## Step 7d: Visualise training data

```bash
# After 7c
sbatch slurm/07d_visualise_training_data.sh
```

## Step 8: Train deconvolution model

```bash
# After step 7c
sbatch slurm/08_train_tapestry.sh
```

## Step 9a: Predict on cfDNA cohorts

```bash
# After steps 6b and 8
sbatch --export=ALL,COHORT=AB slurm/09a_predict_cfdna.sh
sbatch --export=ALL,COHORT=CD slurm/09a_predict_cfdna.sh
```

## Step 9b: Clinical evaluation

```bash
# After 9a completes for both cohorts
sbatch slurm/09b_clinical_evaluation.sh
```
