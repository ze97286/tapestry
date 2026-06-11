# rltf runbook

Standalone read-level tumour-fraction pipeline. Submit every job from the
tapestry repo root. Each step writes a `summary.json` and interactive Plotly
dashboards under `<out_dir>/plots/` (self-contained HTML — copy off the cluster
and open in a browser). Disable plotting on any step with `--no-plots`.

Inputs you provide (full column specs in `README.md`):
- **manifest** TSV: `sample_id role group cohort convention file_path is_cancer tf`
  (role=reference/query; group=tumour/healthy; convention=bisulfite/taps).
  Reference-healthy controls must be disjoint from query-healthy controls.
- **clinical** TSV (step 4 only): `sample_id` + optional `patient_id timepoint
  ichorcna_tf survival_time survival_event stage`. ichorCNA is validation-only.

---

## Step 0 — build the environment (once)

```bash
sbatch readlevel_tf/slurm/setup_env.sh        # or, offline nodes: bash readlevel_tf/scripts/setup_env.sh
```
Creates the venv (numpy/pandas/scipy/scikit-learn/plotly + torch/tabicl) and
pre-caches the TabICL checkpoint. Needs internet (login node if `short` has none).

## Step 1 — discover the marker panel

```bash
sbatch --export=ALL,MANIFEST=data/rltf_manifest.tsv,\
OUT_DIR=readlevel_tf/runs/panel,WINDOW=5,TOP_N=2000,MIN_TOTAL=10,MIN_EFFECT=0.3 \
readlevel_tf/slurm/discover.sh
```
Discovers its own discriminative CpG-dense blocks from the reference PATs.
**Outputs:** `panel/blocks.tsv`, `panel/profiles.npz`, `panel/discovery_summary.json`.
**Plot — `panel/plots/discovery.html`:** tumour-vs-healthy per-block methylation
scatter (separation), effect-size histogram, blocks per chromosome, per-block
coverage. *Look for:* blocks sitting off the diagonal with healthy coverage in
both groups.

## Step 2 — oracle gate (read-level separability)

```bash
sbatch --export=ALL,MANIFEST=data/rltf_manifest.tsv,\
OUT_DIR=readlevel_tf/runs/oracle,HOLDOUT_COHORT=CD readlevel_tf/slurm/oracle.sh
```
Splits reference samples, discovers on **train** only, scores held-out reads.
**Outputs:** `oracle/summary.json`, `oracle/per_read_scores.tsv.gz`, `oracle/split.json`.
**Plot — `oracle/plots/oracle.html`:** per-read LLR tumour-vs-healthy histogram,
ROC, AUC by #CpG bucket, per-healthy-cohort mean LLR.
**Decision:** proceed only if `summary.json → metrics.separable == true` (AUC well
above `auc_permuted`, holds up in `auc_ncpg_matched` and across CpG buckets). If
not separable, retune discovery (`WINDOW`/`MIN_EFFECT`/`MIN_TOTAL`) — the detector
has no signal otherwise.

## Step 3 — detector (cancer detection + tumour fraction)

```bash
sbatch --export=ALL,MANIFEST=data/rltf_manifest.tsv,\
PANEL_DIR=readlevel_tf/runs/panel,OUT_DIR=readlevel_tf/runs/detector,\
BACKEND=tabicl,REGRESSION_MIN_TF=0.03 readlevel_tf/slurm/detect.sh
```
Builds per-sample read-level features and evaluates a TabICL head
leave-one-cohort-out. **Outputs:** `detector/summary.json`, `features.tsv`,
`classification_oof.tsv`, `regression_oof.tsv`.
**Plot — `detector/plots/detector.html`:** ROC, cancer-probability by class with
the specificity threshold, predicted-vs-true TF scatter, tumour-pattern read
fraction by class. *Key numbers:* `classification.auc`, `sens_at_spec_0.95`,
`regression.pearson_r`.

## Step 4 — clinical evaluation

```bash
sbatch --export=ALL,DETECTOR_DIR=readlevel_tf/runs/detector,\
CLINICAL_TSV=data/rltf_clinical.tsv,OUT_DIR=readlevel_tf/runs/clinical,\
SPECIFICITY=0.95 readlevel_tf/slurm/clinical.sh
```
Ties predictions to patients/timepoints/ichorCNA/survival.
**Outputs:** `clinical/summary.json`, `clinical/clinical_merged.tsv`.
**Plots — `clinical/plots/`:**
- `clinical_tf_vs_ichorcna.html` — predicted TF vs ichorCNA (independent
  validation), Pearson r.
- `clinical_longitudinal.html` — per-patient TF trajectories across timepoints.
- `clinical_waterfall.html` — first→last TF change per patient.
- `clinical_detection.html` — detection scores with the specificity threshold.
- `clinical_km.html` — Kaplan–Meier by TF-change direction, log-rank p (if
  survival provided).
**Key numbers:** `detection.sensitivity` at `target_specificity`,
`tf_vs_ichorcna_pearson_r`, `km_logrank_p`.

---

Dependencies: 0 before all; 1 and 2 after 0; **gate on step 2** before 3; 4 after 3.
End-to-end on synthetic data: discovery recovers planted markers, oracle AUC ≈ 0.99
(SEPARABLE), detector AUC ≈ 0.99 / TF r ≈ 0.98, clinical TF-vs-ichorCNA r ≈ 0.98.
