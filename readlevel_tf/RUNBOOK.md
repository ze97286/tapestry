# rltf runbook

Standalone read-level tumour-fraction pipeline, **fully config-driven**. All
paths, data sources, conventions, splits and parameters live in one committed
file — `readlevel_tf/configs/config.toml` (the single source of truth). You edit
that once; every step is then a single `sbatch` with **no command-line
parameters**. Each run copies the resolved config and the materialised manifest
into its output dir (`<run_dir>/<step>/{config.toml,manifest.tsv}`) for
provenance. Submit all jobs from the tapestry repo root.

Each step writes a `summary.json` and interactive Plotly dashboards under
`<run_dir>/<step>/plots/` (self-contained HTML — copy off the cluster to view).

---

## Step 0 — configure + build the environment (once)

**a. Edit the config** `readlevel_tf/configs/config.toml` — point it at your
per-read-call directories; the manifest is materialised from the rules. Key
sections:
- `[reference.tumour].files` — the EAC baseline tumour-tissue `.per-read.bed.gz`.
- `[[cohorts]]` AB/CD — `dir`, `control_globs` (AB `*Ctrl*`; CD `GI*`/`SCAN*`),
  `patient_style` (`ab_underscore` / `cd_hyphen`), `controls_role` (`query` =
  held-out negatives; `split` = part builds `p_H`, rest are negatives).
- `[labels]` — `baseline_timepoint = "ScrBsl"`, `disease_filter = "EAC"`.
- `[paths]` — `clinical_csv` (survival/subtype), `ichorcna_json` (clinical-only).
Patients vs controls are decided by the filename rules; positives are EAC baseline
only, one per patient; CD controls split into reference (`p_H`) vs query negatives,
AB controls all held out as negatives (validated at runtime).

**b. Build the venv.** If home is quota'd, point `~/.cache` at scratch first:
```bash
mkdir -p /users/ludwig/uii408/sharedscratch/.cache/{pip,huggingface}
rm -rf ~/.cache/pip ~/.cache/huggingface
ln -s /users/ludwig/uii408/sharedscratch/.cache/pip         ~/.cache/pip
ln -s /users/ludwig/uii408/sharedscratch/.cache/huggingface ~/.cache/huggingface
```
```bash
sbatch readlevel_tf/slurm/setup_env.sh     # offline nodes: bash readlevel_tf/scripts/setup_env.sh
```

## Step 1 — discover the marker panel

**Recommended (sharded, fast + low-risk)** — one chromosome per array task via
tabix region reads, then merge:
```bash
sbatch readlevel_tf/slurm/discover_shard.sh                               # array 1-22 (short, ~30 min each)
sbatch --dependency=afterok:<arrayJobID> readlevel_tf/slurm/discover_merge.sh
```
**Or single job** (genome-wide, `long`, memory-heavy, hours): `sbatch readlevel_tf/slurm/discover.sh`.

Discovers discriminative CpG-dense blocks from the reference per-read calls
(genomic CpG space). **Outputs** under `<run_dir>/panel/`: `cpgs.tsv` (per-CpG
`p_tumour`/`p_healthy`), `blocks.tsv`, `meta.json`. (Sharded run also leaves
per-chromosome partials under `panel/partials/`.)

## Step 2 — oracle gate (read-level separability)
```bash
sbatch readlevel_tf/slurm/oracle.sh
```
Splits reference samples, discovers on **train** only, scores held-out fragments
by the n_cpg-conditioned `z`. **Outputs** under `<run_dir>/oracle/`: `summary.json`,
`per_fragment_scores.tsv.gz`.
**Gate:** continue only if `metrics.separable == true`. Also check the
length-leakage gates: `corr_z_read_length` ≈ 0 and `null_mean_z_by_n_cpg` ≈ 0 in
every bucket. If not separable, retune `[discovery]` (`window`/`min_effect`/`min_total`).

## Step 3 — detector (cancer detection)
```bash
sbatch readlevel_tf/slurm/detect.sh
```
Per-sample features → TabICL head, **leave-one-cohort-out** (detection only — no
ichorCNA). **Outputs** under `<run_dir>/detector/`: `summary.json`
(`classification.auc`, `sens_at_spec_0.95`/`_0.99`), `features.tsv`,
`classification_oof.tsv`.

## Step 4 — clinical evaluation
```bash
sbatch readlevel_tf/slurm/clinical.sh
```
Joins the detector's out-of-fold scores to the clinical CSV (survival/benefit) and
ichorCNA (sanity only). **Outputs** under `<run_dir>/clinical/`: `summary.json`
(`detection.sensitivity` at target specificity, `ichorcna_sanity.pearson_r` where
ichorCNA tf ≥ `clinical.ichorcna_min_tf`, `km_logrank_p` by detection-score split),
`clinical_merged.tsv`. (Longitudinal on-treatment TF trajectories need a separate
scoring pass over the on-treatment timepoints.)

---

Dependencies: 0 → 1 → 2 → (gate) → 3 → 4. To run a variant, copy the config and
pass `CONFIG=...`:
`sbatch --export=ALL,CONFIG=readlevel_tf/configs/config_v2.toml readlevel_tf/slurm/discover.sh`.
Validated end-to-end on synthetic per-read calls: discovery recovers planted
markers, oracle SEPARABLE with healthy `z` ⟂ length, detector AUC under
leave-one-cohort-out, clinical detection + ichorCNA-sanity + KM.
