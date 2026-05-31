# MethylBERT shortcut diagnostics

This document describes the controls added to the MethylBERT read-call workflow to test
**what the fine-tuned read classifier actually learns** — genuine tumour-vs-normal
methylation, or a trivial shortcut (fragment length, sequencing/library batch, source,
genomic locus, CpG count).

## Why this matters

The pipeline faithfully replicates the MethylBERT paper's ctDNA design: the positive
class is OAC tumour **tissue** and the negative class is healthy **plasma cfDNA**, and the
fine-tuned classifier is deployed on patient plasma. That design is the paper's, but it
confounds the class label with molecule type, library/sequencing batch, cohort and
fragment-length distribution. Because the model is the authors' own code (an unmodified
vendored clone of `CompEpigen/methylbert`), it inherits the paper's shortcut exposure:

- the read classifier is a dense layer over **every** padded position, and upstream calls
  `forward()` with no `attention_mask`, so fragment length is directly readable;
- the DMR id is an explicit channel and `dna_seq` is the reference context, so genomic
  locus is available;
- the held-out test reads share the same source/batch as training, so a high held-out
  accuracy is consistent with a shortcut and is **not** evidence of tumour discrimination.

The paper protected its headline numbers with same-source pseudo-bulks (where length is
independent of the target) and explicit read-length robustness checks. This project can do
something the paper could not: validate against **ichorCNA** tumour fraction, which is
CNA-based and independent of both methylation and fragment length.

Ranked most-to-least likely, the classifier is learning: (1) tissue-vs-plasma
source/batch/length, (2) DMR/locus identity, (3) CpG count/coverage, then (4) genuine
per-CpG tumour methylation.

## Runtime patches (the vendored model stays pristine)

`external/methylbert` is left byte-identical to upstream HEAD `f82f83b`. Two behaviours are
applied at runtime by `scripts/methylbert_patches.py` — `apply_patches()` is called by the
finetune, eval and deconvolution runners — so the vendored model code is never edited:

- `MethylBertEmbeddedDMR.forward` defaults the `attention_mask` from the pad token
  (`pad_index == 0`) so BERT no longer attends to padding, and zeroes padded positions
  before the flatten classifier. This reduces, but does not fully remove, the read-length
  channel: the EOS-at-read-end plus the flatten head still encode some length. Fully closing
  it would require re-architecting the read classifier head (e.g. a masked mean pool) and
  retraining.
- `deconvolute()` accepts `margins_override` so the deploy prior can be set explicitly
  (`--prior-t` / `DECONV_PRIOR_T`) instead of always using the training class frequencies.

Both patches mirror the upstream bodies with only those additions; the maths is otherwise
upstream.

## Reproducibility fix: the `collapsed_100kb` DMR panel

`select_methylbert_dmr_variants.py` now emits the `collapsed_100kb` panel by default
(`--collapse-distances 100000 500000 1000000`). Previously only 500 kb/1000 kb variants
were produced, so the `dmrs_top100.collapsed_100kb.tsv` actually consumed by the workflow
could not be regenerated from committed tooling. Regenerate it from the merged DSS calls
and keep it under version control alongside the run.

## New covariates and how a shortcut shows up

`preprocess_methylbert_taps_read_calls.py` now writes `read_length` and `n_cpg` columns.
They pass through the unmodified upstream `dataset.py` and `read_classification`, so they
appear in `test_predictions.tsv` (eval) and `res.csv` (deploy) for free.

`evaluate_methylbert_finetune.py` now writes, alongside `summary.json`:

- `summary_by_sample.tsv`, `summary_by_cohort.tsv`, `summary_by_length_decile.tsv`,
  `summary_by_n_cpg.tsv` — accuracy stratified by covariates that should not, on their own,
  carry tumour signal;
- `corr_prob_vs_read_length` and `corr_prob_vs_n_cpg` in `summary.json` — the per-read
  correlation of `P(tumour)` with read length and CpG count.

**Genuine signal:** accuracy roughly flat across length deciles and CpG buckets;
`corr_prob_vs_*` near zero. **Shortcut:** accuracy tracks length/cohort, or `P(tumour)`
correlates strongly with read length / `n_cpg`.

## The diagnostic battery

Each control is an environment-variable toggle on the existing SLURM wrappers, or a small
helper script. Run each into a **distinct** output directory (override
`READ_CALL_PREPROCESS_DIR`, `MODEL_DIR`, `EVAL_DIR`, `DECONV_DIR`) so the main run is not
overwritten. The decisive control is the same-molecule-type one; the cheapest informative
one is the label permutation.

| Control | Toggle | Re-run from | Genuine → | Shortcut → |
|---|---|---|---|---|
| **Label permutation** | `SHUFFLE_LABELS=1` (merge) | merge → finetune → eval | AUC → ~0.5 | stays high ⇒ per-sample/batch identity is being read |
| **Methylation ablation** | `BLANK_METHYL=1` (preprocess) | preprocess → … → eval | accuracy → ~chance | stays high ⇒ length/locus/CpG-count alone separate the classes |
| **DNA / locus ablation** | `BLANK_DNA=1 COLLAPSE_DMR_LABEL=1` (preprocess) | preprocess → … → eval | accuracy mostly retained | large drop ⇒ leaned on locus identity |
| **Read-length matching** | `LENGTH_MATCH=1` (merge) | merge → finetune → eval | accuracy ~unchanged | collapses after matching ⇒ fragment-length shortcut |
| **Leave-one-sample-out** | `methylbert_loso_folds.py` + `HOLDOUT_SAMPLES` | per-fold merge → finetune → eval | consistent across folds | high variance, one biopsy carries it |
| **Leave-one-batch-out** | `HOLDOUT_COHORT=CD_plasma` (merge) | merge → finetune → eval | specificity stable across cohorts | swings by cohort ⇒ batch signature |
| **Random-region (DMR shuffle)** | `generate_random_dmr_regions.py` → set `METHYLBERT_DMRS` | preprocess → … → eval | accuracy drops markedly | comparable to real DMRs ⇒ generic source/coverage signal |
| **Same molecule type** *(decisive)* | swap the positive list (tumour cfDNA vs healthy cfDNA, or tumour vs normal tissue) and re-call DMRs | full pipeline | accuracy clearly above chance | drops toward chance once tissue-vs-plasma removed |
| **Deploy prior recalibration** | `DECONV_PRIOR_T=0.5` (deconvolute) | deconvolute | θ stable | θ moves a lot ⇒ estimate was prior-driven |
| **θ vs ichorCNA / fragmentomics** | `plot_methylbert_deconvolution_vs_ichorcna.py` | deconvolute → collect → plot | r(θ, ichorCNA) approaches NNLS's ≈0.885 | r(θ, read length / read count) ≫ r(θ, ichorCNA) ⇒ fragment shortcut |

### Worked example — methylation ablation

```bash
# Re-preprocess with the methylation channel blanked, into a separate dir.
export READ_CALL_SHARD_DIR=".../preprocess_taps_read_call_shards_blankmethyl"
export BLANK_METHYL=1
sbatch --array=1-"${N}"%130 --export=ALL run_methylbert_paper_preprocess_read_calls_array_slurm.sh

# Merge, fine-tune and evaluate into matching dirs, then compare eval accuracy
# (and corr_prob_vs_* in summary.json) against the unablated run.
export READ_CALL_PREPROCESS_DIR=".../preprocess_taps_read_calls_blankmethyl"
export MODEL_DIR=".../model_blankmethyl"
export EVAL_DIR="${MODEL_DIR}/heldout_eval"
sbatch --export=ALL run_methylbert_paper_merge_read_call_shards_slurm.sh
sbatch --export=ALL run_methylbert_paper_finetune_slurm.sh
sbatch --export=ALL run_methylbert_paper_eval_slurm.sh
```

If held-out accuracy stays high with the methylation channel blanked, the classifier is
not relying on methylation — it is reading length, locus or CpG count.

### The orthogonal validator

`data/cfDNA_tumour_fraction_ichorCNA.json` holds ichorCNA tumour fractions (CNA-based,
independent of methylation and fragment length; the NNLS baseline reaches r ≈ 0.885 against
it). Running `collect_methylbert_deconvolution.py` now also records per-sample
`mean_read_length`, `mean_n_cpg` and `mean_p_tumour`, and
`plot_methylbert_deconvolution_vs_ichorcna.py` writes `*_theta_vs_fragmentomics.csv`
correlating θ with ichorCNA and with those fragmentomics features. If θ tracks ichorCNA
more strongly than read length / read count, it is measuring tumour content; if the reverse,
it is a fragmentomics shortcut wearing a tumour-fraction costume.

## Regression test

`tests/test_methylbert_preprocess_roundtrip.py` builds a tiny synthetic input, runs the
preprocessor, checks the covariate columns and ablation flags, and confirms the resulting
CSV still parses through the unmodified upstream `dataset.py`. Run it with
`python3 tests/test_methylbert_preprocess_roundtrip.py`.
