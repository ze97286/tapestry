# Tapestry

cfDNA methylation deconvolution pipeline for estimating cell-type proportions from TAPS sequencing data.

## Prerequisites

- **wgbs_tools** installed and compiled on compute nodes: `/users/ludwig/uii408/sharedscratch/wgbs_tools/`
- **Reference genome**: hg38 initialised in wgbs_tools
- **Module**: `GCC/12.3.0` required on BMRC compute nodes (loaded automatically via `slurm/common.sh`)
- **Reference PAT files**: Loyfer et al. bisulfite + TAPS tissue samples in `data/pats/`
- **Manifest**: `data/manifest_atlas.tsv` mapping sample_id → cell_type → file_path

## Pipeline

All SLURM scripts source `slurm/common.sh` which sets paths and loads modules.

### Step 1: Generate beta files from PAT files

Converts each PAT file to a per-CpG methylation summary (`.beta`) needed for segmentation. Array job — one task per sample, 70 in parallel.

```bash
sbatch slurm/02a_pat2beta.sh
```

**Input**: `data/manifest_atlas.tsv` → PAT files listed in column 3  
**Output**: `runs/run_002/betas/*.beta` (70 files)

### Step 2: Joint segmentation

Segments the genome into non-overlapping blocks of homogeneous methylation using dynamic programming across all reference samples. Array job — one task per autosome, 22 in parallel.

```bash
# Wait for step 1 to complete
sbatch slurm/02b_segment.sh
```

**Input**: `runs/run_002/betas/*.beta` (70 files)  
**Output**: `runs/run_002/segmentation/blocks_chr*.bed.gz` (22 files)

Then merge:

```bash
# After all 22 chromosome jobs complete
sbatch slurm/02c_merge_blocks.sh
```

Or manually:

```bash
cd runs/run_002/segmentation
# If files are gzipped but named .bed, fix with:
# for f in blocks_chr*.bed; do mv "$f" "${f}.gz"; gunzip "${f}.gz"; done
cat blocks_chr{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}.bed > blocks.bed
```

**Output**: `runs/run_002/segmentation/blocks.bed` (~2.1M blocks with ≥4 CpGs)

**Parameters**:
- `--min_cpg 4`: blocks must have ≥4 CpGs (for UXM rlen=4 compatibility)
- `--max_cpg 1000`: default, let DP decide block sizes
- `--max_bp 2000`: max 2kb per block
- Autosomes only (chr1-22)

### Step 3: UXM homog counting

Counts U (unmethylated), X (mixed), M (methylated) reads per block per sample. Array job — one task per sample, 70 in parallel.

```bash
# Wait for step 2 to complete
sbatch slurm/03_homog.sh
```

**Input**: `runs/run_002/segmentation/blocks.bed` + PAT files  
**Output**: `runs/run_002/homog/*.uxm.bed.gz` (70 files)

Each file has columns: `chr start end startCpG endCpG U X M`  
UXM classification uses `rlen=4` (reads with <4 CpGs in a block are discarded).

### Step 4: Marker selection

Selects marker blocks where one cell type has strong unmethylated signal that all others lack. Ranks by signal-to-noise ratio.

```bash
# Wait for step 3 to complete
sbatch slurm/04_select_markers.sh
```

**Input**: `runs/run_002/homog/*.uxm.bed.gz` + `data/manifest_atlas.tsv`  
**Output**: `runs/run_002/markers/markers.tsv`

**Filters applied**:
- `--min-snr 3.0`: minimum signal-to-noise ratio (target U-fraction / background U-fraction)
- `--min-signal 0.3`: target cell type must have ≥30% U-reads
- `--max-bg 0.1`: pooled background must have ≤10% U-reads
- `--max-single-bg 0.2`: no individual non-target cell type may have >20% U-reads
- `--min-consistency 0.2`: every sample within the target cell type must individually show ≥20% U-reads
- `--min-cov-per-sample 5`: every target sample must have ≥5 reads at the block
- `--top-n 250`: maximum 250 markers per cell type
- `--direction U`: hypomethylated markers only (standard UXM practice)

**Output columns**: block coordinates, target cell type, SNR, and U-fraction for each cell type.

### Step 5: Visualise atlas

Generates diagnostic plots: SNR distribution, chromosomal spread, block sizes, signal heatmap, target-vs-background scatter, and coverage distribution.

```bash
# Wait for step 4 to complete
sbatch slurm/05_visualise_atlas.sh
```

**Input**: `runs/run_002/markers/markers.tsv`
**Output**: `runs/run_002/markers/plots/*.png` (6 plots)

### Step 6: Filter PAT files to marker regions

Extracts only reads overlapping the ~3,084 marker blocks from each PAT file, producing much smaller files for efficient mixture generation and inference.

#### 6a — Reference samples (bisulfite convention, no flip)

```bash
# Wait for step 4 to complete
sbatch slurm/06a_filter_ref_pats.sh
```

**Input**: `runs/run_002/markers/markers.tsv` + 70 reference PAT files
**Output**: `runs/run_002/filtered_pats/ref/*.markers.pat.gz` (70 files)

Also creates `runs/run_002/markers/markers.bed` (marker regions in BED format).

#### 6b — cfDNA samples (TAPS convention, flipped to bisulfite)

cfDNA PAT files use native TAPS convention (T=methylated, C=unmethylated). The script applies `sed y/CT/TC/` to flip patterns to bisulfite convention before writing. AB and CD cohorts are processed separately into cohort-specific output directories.

```bash
# Count samples in each cohort
N_AB=$(ls /gpfs3/users/ludwig/uii408/sharedscratch/tapestry/data/AB/*.pat.gz | wc -l)
N_CD=$(ls /gpfs3/users/ludwig/uii408/sharedscratch/tapestry/data/CD/*.pat.gz | wc -l)

# Submit separately
sbatch --array=1-${N_AB} --export=ALL,COHORT=AB slurm/06b_filter_cfdna_pats.sh
sbatch --array=1-${N_CD} --export=ALL,COHORT=CD slurm/06b_filter_cfdna_pats.sh
```

**Input**: `data/AB/*.pat.gz`, `data/CD/*.pat.gz` + `markers.bed`
**Output**:
- `runs/run_002/filtered_pats/cfdna/AB/*.markers.pat.gz`
- `runs/run_002/filtered_pats/cfdna/CD/*.markers.pat.gz`

### Step 7: Training data generation

Generates synthetic cfDNA mixtures from filtered reference PAT files for model training and evaluation. Each mixture is created by sampling reads from individual reference samples (one per cell type) at target proportions and coverage, preserving real read-level stochastic noise.

#### 7a — Generate proportion tables

Produces CSV files specifying proportions, target depths, and reference sample assignments for all synthetic mixtures. Five proportion strategies are blended: blood-dominated realistic (40%), rare-type emphasis (20%), broad Dirichlet (20%), zero-forcing (10%), near-pure/edge cases (10%). Target depth is sampled log-uniformly from 5K–200K total reads.

```bash
sbatch slurm/07a_generate_proportions.sh
```

**Output**: `runs/run_002/training/train_proportions.csv` (100K rows), `eval_proportions.csv` (20K rows), `oac_dilution_proportions.csv`, `tcell_dilution_proportions.csv`

#### 7b — Generate mixtures

For each mixture: samples reads from per-cell-type filtered PATs using `pattools sample`, merges into a single mixture PAT, runs `wgbstools homog` to get per-marker U/X/M counts, and extracts U-fraction and coverage. Array job — each task processes 1,000 mixtures.

```bash
# After 07a completes:
sbatch --array=0-99 --export=ALL,DATASET=train slurm/07b_generate_mixtures.sh
sbatch --array=0-19 --export=ALL,DATASET=eval slurm/07b_generate_mixtures.sh
sbatch --array=0-1  --export=ALL,DATASET=oac_dilution slurm/07b_generate_mixtures.sh
sbatch --array=0-1  --export=ALL,DATASET=tcell_dilution slurm/07b_generate_mixtures.sh
```

**Input**: proportion CSVs + `runs/run_002/filtered_pats/ref/*.markers.pat.gz`
**Output**: `runs/run_002/training/{train,eval,oac_dilution,tcell_dilution}/batch_*.npz`

**Key design decisions**:

- One reference sample per cell type per mixture (not merged) — preserves within-type variability, especially important for tumour types with heterogeneous methylation profiles
- Reads are sampled at the PAT level via `pattools sample`, preserving real stochastic noise (binomial sampling, coverage variability, dropout)
- Continuous log-uniform depth distribution rather than discrete tiers — the model learns to handle any coverage level

#### 7c — Collect into parquets

Merges batch npz files into final parquet format for model training.

```bash
# After all 07b jobs complete:
sbatch slurm/07c_collect_training_data.sh
```

**Output** (per dataset):

- `marker_values.parquet` — U-fractions (samples × markers)
- `coverage.parquet` — read counts (samples × markers)
- `ground_truth_y.parquet` — true proportions (samples × cell types)

#### 7d — Visualise training data

Generates comprehensive plots of training and evaluation data distributions for quality control and paper figures. All plots produced as both interactive HTML (plotly) and high-resolution PNG.

```bash
# After 07c completes:
sbatch slurm/07d_visualise_training_data.sh
```

**Output**: `runs/run_002/training/plots/{train,eval,oac_dilution,tcell_dilution}/*.{html,png}`

**Plots generated** (per dataset):

- Proportion distributions (boxplot + log-scale violin)
- Proportion heatmap (samples × cell types)
- Cell type proportion correlations
- Zero-proportion rate per cell type
- Concentration histograms for OAC, T-cells, Hepatocytes, Colon (log scale)
- Per-marker and per-sample coverage distributions
- Zero-coverage marker rate
- Marker value (U-fraction) distribution
- Proportion strategy breakdown
- Target depth distribution
- Dilution series ground truth (for dilution datasets)

### Step 8: Train deconvolution model

Trains the TapestryModel — a two-level hierarchical transformer that takes per-marker U-fraction and coverage as input and predicts cell-type proportions. See `docs/architecture.md` for the full design rationale.

```bash
# After step 7 completes:
sbatch slurm/08_train_tapestry.sh
```

**Input**: `runs/run_002/training/{train,eval}/*.parquet` + `runs/run_002/markers/markers.tsv`
**Output**: `runs/run_002/models/tapestry/`

- `best_model.pt` — best checkpoint by validation metrics
- `final_model.pt` — final epoch checkpoint
- `history.json` — all metrics per epoch (saved incrementally)
- `config.json` — full training configuration
- `training_losses.html` — loss curves (total + per component)
- `validation_metrics.html` — MAE, log-R², slope, presence F1, gradient norm
- `per_celltype_mae.html` / `per_celltype_r2.html` — per-cell-type metric evolution
- `model_internals.html` — mass distribution, detection probabilities
- `pred_vs_true_linear.html` / `pred_vs_true_log.html` — scatter plots at best epoch

**Architecture**: feature_dim=128, Level 1 transformer (2 layers, 4 heads per cell type), Level 2 transformer (1 layer, 4 heads over 13 cell types), ~2-4M parameters.

**Training**: AdamW with linear warmup + ReduceLROnPlateau, gradient accumulation, early stopping on validation loss/log-R²/MAE.

### Step 9: Clinical evaluation

Runs the trained model on real cfDNA samples from the AB and CD cohorts, then generates clinical evaluation plots comparing to ichorCNA and survival data.

#### 9a — Predict on cfDNA cohorts

Runs `wgbstools homog` on each filtered cfDNA PAT file against the marker atlas, then feeds the per-marker U-fractions and coverage through the trained TapestryModel to predict cell-type proportions.

```bash
# After steps 6b and 8 complete:
sbatch --export=ALL,COHORT=AB slurm/09a_predict_cfdna.sh
sbatch --export=ALL,COHORT=CD slurm/09a_predict_cfdna.sh
```

**Input**: `runs/run_002/filtered_pats/cfdna/{AB,CD}/*.markers.pat.gz` + `runs/run_002/models/tapestry/best_model.pt`
**Output**: `runs/run_002/predictions/{AB,CD}_predictions.csv`

Each row contains: sample name, cohort, mean coverage, per-cell-type proportion, per-cell-type detection probability.

#### 9b — Clinical evaluation

Generates clinical evaluation plots and tables. Requires clinical data CSV and optionally ichorCNA tumour fraction data for comparison.

```bash
# After 9a completes:
sbatch slurm/09b_clinical_evaluation.sh
```

**Input**: `runs/run_002/predictions/{AB,CD}_predictions.csv` + clinical data + ichorCNA data
**Output**: `runs/run_002/evaluation/{AB,CD}/`

**Plots generated**:

- Stacked bar chart of all cell-type proportions per sample
- Proportion heatmap across cohort
- OAC waterfall plot (Immonly − ScrBsl delta per patient, coloured by clinical benefit)
- Kaplan-Meier survival curves split by OAC change direction (with log-rank p-value)
- OAC vs ichorCNA tumour fraction scatter per timepoint (with Pearson r)
- Per-cell-type correlation with ichorCNA
- Full deconvolution results table (CSV)
- OAC delta table with clinical annotations (CSV)

### Automated submission (steps 1-2)

```bash
bash slurm/submit_segmentation.sh
```

## Data conventions

- **Reference PAT files** (Loyfer + TAPS tissue): bisulfite convention (C=methylated, T=unmethylated)
- **TAPS cfDNA** (AB cohort): native TAPS convention — must be flipped (`sed y/CT/TC/`) before use with this atlas
- All segmentation and marker selection uses bisulfite-convention data

## Cell types (13)

| Cell type | Samples | Source |
|-----------|---------|--------|
| B-cells | 5 | Loyfer (3 naive + 2 memory) |
| CD34-erythroblasts | 3 | Loyfer |
| CD34-megakaryocytes | 2 | TAPS tissue (flipped to bisulfite) |
| Colon | 8 | Loyfer |
| Esophagus | 2 | Loyfer |
| Gastric | 11 | Loyfer |
| Granulocytes | 3 | Loyfer |
| Hepatocytes | 6 | Loyfer |
| Monocytes | 3 | Loyfer |
| NK-cells | 3 | Loyfer |
| OAC | 5 | TAPS tissue (flipped to bisulfite) |
| Small-intestine | 2 | Loyfer |
| T-cells | 18 | Loyfer (all CD4/CD8 subtypes merged) |
