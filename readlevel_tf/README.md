# rltf — read-level tumour-fraction detection from cfDNA methylation

A **standalone** project: read-level (single-molecule) tumour detection and
tumour-fraction estimation from TAPS/bisulfite cfDNA methylation. It depends on
nothing else in this repository and consumes **no** artifact from the other
attempts (no atlas/segmentation blocks, no DMR panel, no `filtered_pats`, no
ichorCNA json). It owns its full pipeline from raw per-sample PATs: its own
marker discovery, preprocessing (incl. the TAPS↔bisulfite flip), model and
evaluation. Everything works in the PAT **global CpG-index** coordinate, so no
external genomic CpG index is needed either.

## Idea

The discriminative signal at low tumour fraction lives in whole-molecule
methylation patterns, which per-marker averaging (NNLS/atlas) destroys, and
which sample-label classifiers (MethylBERT/deepconv) learn as batch/length
shortcuts. rltf instead scores each **read** by an explicit likelihood ratio
against tumour vs healthy reference profiles, aggregates to coverage-normalised,
reference-defined per-sample features, and feeds them to an in-context tabular
foundation model (TabICL/TabPFN) evaluated leave-one-cohort-out — so an AUC
cannot be batch memorisation, and the head cannot overfit the few labels.

## Layout

```
readlevel_tf/
  rltf/        io, regions, discovery, profiles, llr, features, head, manifest, plots
  scripts/     discover_panel.py, run_oracle.py, run_detector.py, run_clinical_eval.py, setup_env.sh
  slurm/       common.sh, setup_env.sh, discover.sh, oracle.sh, detect.sh, clinical.sh
  tests/       test_pipeline.py, test_clinical.py   (synthetic end-to-end)
  RUNBOOK.md   step-by-step operational runbook (start here to run it)
```

Each step also writes interactive Plotly dashboards under `<out_dir>/plots/`
(self-contained HTML). See `RUNBOOK.md` for the per-step commands and what each
plot shows; the run sequence below is the summary.

## Manifest (one TSV, header row)

```
sample_id  role  group  cohort  convention  file_path  is_cancer  tf
```
- `role` = `reference` (build profiles) | `query` (evaluated).
- `group` = `tumour` (tissue) | `healthy` (cfDNA controls) — reference rows.
- `cohort` = batch label (`OAC_tissue`/`AB`/`CD`); the leave-one-group-out key.
- `convention` = `bisulfite` | `taps` — per sample; the flip is applied internally.
- `is_cancer` = 0/1 for query rows; `tf` = optional ichorCNA TF for query rows.
- **Reference-healthy and query-healthy controls must be disjoint.**

ichorCNA, if used, is for the `tf` validation column only — computed
independently, never read from another project's files.

## Run sequence (submit from the repo root)

**0. Build the env once** (pip + TabICL download need internet; run on a login
node if compute nodes are offline):
```bash
sbatch readlevel_tf/slurm/setup_env.sh
# or: bash readlevel_tf/scripts/setup_env.sh
```

**1. Discover the panel** (reference rows → `blocks.tsv` + `profiles.npz`):
```bash
sbatch --export=ALL,MANIFEST=data/rltf_manifest.tsv,\
OUT_DIR=readlevel_tf/runs/panel,WINDOW=5,TOP_N=2000,MIN_TOTAL=10,MIN_EFFECT=0.3 \
readlevel_tf/slurm/discover.sh
```

**2. Oracle gate** (train/test split, discover-on-train, score held-out reads):
```bash
sbatch --export=ALL,MANIFEST=data/rltf_manifest.tsv,\
OUT_DIR=readlevel_tf/runs/oracle,HOLDOUT_COHORT=CD readlevel_tf/slurm/oracle.sh
```
Read `runs/oracle/summary.json` → `metrics`: `auc` vs `auc_permuted`,
`auc_ncpg_matched`, `separable`, and `per_healthy_cohort`. **If not `separable`,
stop** and adjust discovery (`WINDOW`/`MIN_EFFECT`/`MIN_TOTAL`) — the detector
has no signal otherwise.

**3. Detector** (only if the gate passed):
```bash
sbatch --export=ALL,MANIFEST=data/rltf_manifest.tsv,\
PANEL_DIR=readlevel_tf/runs/panel,OUT_DIR=readlevel_tf/runs/detector,\
BACKEND=tabicl,REGRESSION_MIN_TF=0.03 readlevel_tf/slurm/detect.sh
```
Read `runs/detector/summary.json` → `classification.auc`, `sens_at_spec_0.95`;
`regression.pearson_r`. Per-sample predictions in `classification_oof.tsv` /
`regression_oof.tsv`. Dashboard: `runs/detector/plots/detector.html`.

**4. Clinical evaluation** (ties predictions to patients/ichorCNA/survival):
```bash
sbatch --export=ALL,DETECTOR_DIR=readlevel_tf/runs/detector,\
CLINICAL_TSV=data/rltf_clinical.tsv,OUT_DIR=readlevel_tf/runs/clinical,\
SPECIFICITY=0.95 readlevel_tf/slurm/clinical.sh
```
Clinical TSV: `sample_id` + optional `patient_id timepoint ichorcna_tf
survival_time survival_event stage`. Outputs `runs/clinical/summary.json`
(`detection.sensitivity` at fixed specificity, `tf_vs_ichorcna_pearson_r`,
`km_logrank_p`) and plots: predicted-TF-vs-ichorCNA, per-patient longitudinal TF,
TF-change waterfall, detection-at-specificity, Kaplan–Meier by TF change.

## Backends

- **TabICL** (default; open, no auth, works offline once cached) — classification.
- **TabPFN** — classification + TF regression, but its weights are a gated
  HuggingFace repo (needs `HF_TOKEN`); the head falls back to sklearn if absent.
- **sklearn** — deterministic offline baseline.

## Scale

Discovery holds ~4 float32 global arrays sized to the max CpG index (~0.45 GB for
hg38), one streaming pass over the reference PATs; selection is cheap. Plan ~32 GB
RAM, 1 CPU, minutes-to-hours by reference depth. Oracle/detector stream reads
once per sample. CPU-only; `DEVICE=cuda` optional for TabPFN.

## Test

```bash
PYTHONPATH=readlevel_tf python3 readlevel_tf/tests/test_pipeline.py
```
Synthetic end-to-end: discovery recovers planted markers, oracle AUC > 0.9
(permutation ~0.5), detector AUC > 0.85 with positive TF correlation.
```
