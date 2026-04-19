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

## Step 8b: NNLS baseline on the eval set

```bash
# After step 7c. Independent of step 8 — uses the atlas directly.
# Applies the same NaN-row filter to markers.tsv that tapestry training uses,
# so the two evals are directly comparable.
sbatch slurm/08b_eval_nnls.sh
```

Output to `logs/eval_nnls.out`: per-cell-type MAE + R², overall MAE, log-space
R²/slope/intercept, presence precision/recall/F1.

## Step 8c: Post-hoc calibrated + NNLS-gated predictions

```bash
# After step 8 has written best_model.pt and step 8b has run (optional — NNLS
# is re-run inside the script).
sbatch slurm/08c_predict_calibrated.sh
```

Fits affine log-space recalibration (`log_pred_cal = α·log_pred + β`) per cell
type on the **training set** predictions, then reports four eval variants
side-by-side:

- `raw` — direct model output
- `recalibrated` — log-space affine correction (fixes slope/intercept)
- `nnls_gated` — raw tapestry zeroed where NNLS assigns 0 (borrows NNLS's
  presence precision)
- `recal_gated` — both combined

Outputs land in `${MODEL_DIR}/calibrated/`: `pred_eval_{raw,recalibrated,
nnls_gated,recal_gated}.npy`, `pred_eval_nnls.npy`, `calibration.npz`
(α, β per cell type), `metrics.json`.

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
