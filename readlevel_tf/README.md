# rltf — read-level cancer detection from cfDNA methylation

A **standalone** project (depends on nothing else in this repo, consumes no other
attempt's artifacts). It detects cancer at low tumour fraction from **per-read
methylation calls**, owning its full pipeline: its own marker discovery,
preprocessing, model and evaluation.

## Substrate and idea

Input is **per-read calls** (`.per-read.bed.gz`: `chr start end read_id mapq
orientation insert_size read_length flag num_cpg num_mod mod_cpgs unmod_cpgs
snp_cpgs`), read at the **fragment** level — mates collapsed by `read_id` (so a
molecule is counted once), SNP CpGs masked, MAPQ-filtered. CpG identity is the
**genomic position** (`start + offset`), so tumour tissue / AB / CD align on the
genome with no shared-index assumption and no methylation convention (the calls
are explicit mod/unmod).

The signal at low TF lives in whole-molecule patterns, which per-marker averaging
(NNLS/atlas) destroys and which sample-label classifiers (MethylBERT/deepconv)
learn as a batch/length shortcut. rltf scores each **fragment** by a per-read
log-likelihood ratio against tumour vs healthy reference profiles, **standardised
against its own healthy null** (`z = (LLR − μ₀)/σ₀`) so that the number of CpGs a
read covers — hence read length — carries no information about the label (proven
in the tests: healthy `z` ⟂ n_cpg and read length). Per-sample features
(coverage-normalised) feed an in-context tabular model (TabICL/TabPFN), evaluated
**leave-one-cohort-out**, so an AUC cannot be batch memorisation.

**Detection-first.** ichorCNA tumour fraction is mostly <1% in this cohort and
unreliable there, so it is used **only for clinical validation/sanity**, never as
a model input or target.

## Layout

```
readlevel_tf/
  configs/config.toml   single source of truth — data sources, naming rules, params
  rltf/    io, regions, discovery, profiles, llr, features, head, config, manifest, plots
  scripts/ discover_panel, run_oracle, run_detector, run_clinical_eval, setup_env.sh
  slurm/   common, setup_env, discover, oracle, detect, clinical
  tests/   test_io, test_pipeline, test_config, test_clinical
  RUNBOOK.md   step-by-step operational runbook (start here to run it)
```

Config-driven: each step is a single `sbatch` (no CLI args), reads
`configs/config.toml`, and copies the resolved config + materialised manifest into
its output dir for provenance.

## Configuration (`configs/config.toml`)

The manifest is **materialised from the config**, never hand-authored:
- `[reference.tumour].files` — the EAC baseline tumour-tissue per-read calls (build `p_T`).
- `[[cohorts]]` (AB, CD) — `dir`, `control_globs` (AB `*Ctrl*`; CD `GI*`/`SCAN*`),
  `patient_style` (`ab_underscore` / `cd_hyphen`), and `controls_role`
  (`query` = held-out negatives, `split` = part builds `p_H`).
- `[labels]` — `baseline_timepoint = "ScrBsl"` (positives are baseline only),
  `disease_filter = "EAC"` (patients restricted via the clinical CSV subtype).
- `[paths]` — `clinical_csv`, `ichorcna_json` (clinical-only), `run_dir`.
- `[discovery]/[scoring]/[oracle]/[detector]/[clinical]` — parameters.

Patients vs controls by filename rule; baseline-only, EAC-only, one row per
patient; CD controls split into reference (`p_H`) vs query negatives; AB controls
all held out as negatives.

## Run sequence

Edit `configs/config.toml`, then from the repo root:
```bash
sbatch readlevel_tf/slurm/setup_env.sh   # 0. once
sbatch readlevel_tf/slurm/discover.sh    # 1. panel
sbatch readlevel_tf/slurm/oracle.sh      # 2. gate — proceed only if metrics.separable
sbatch readlevel_tf/slurm/detect.sh      # 3. detection (leave-one-cohort-out)
sbatch readlevel_tf/slurm/clinical.sh    # 4. clinical: detection + ichorCNA sanity + survival
```
See `RUNBOOK.md` for per-step outputs and the gate criterion.

## Backends

TabICL (default; open, offline-capable), TabPFN (gated HuggingFace weights — needs
`HF_TOKEN`; falls back to sklearn), sklearn (deterministic baseline).

## Scale

Discovery is a per-CpG tally over the reference samples (genome-wide, memory-heavy
— ~96 GB, or shard by chromosome via `--array=1-22`). Scoring uses **tabix
region-queries** over the panel intervals (pysam), so the multi-GB per-read files
are read only near panel CpGs — minutes per sample.

## Test

```bash
PYTHONPATH=readlevel_tf python3 -m pytest readlevel_tf/tests -q
```
Synthetic end-to-end (reads spanning 2–14 CpGs so length varies): reader semantics
(mate-dedup, SNP-mask, MAPQ), discovery recovers planted markers, oracle separates
with **healthy `z` ⟂ n_cpg and read length**, detector AUC under leave-one-cohort-out,
and the label-builder rules (EAC/baseline/control-split).
```
