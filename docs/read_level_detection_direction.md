# Direction: read-level, reference-anchored, unknown-aware tumour-fraction detection

Date: 2026-06-10

This note records the strategic diagnosis after five attempts (NNLS, augmented-NNLS
with an unknown component, the TapestryModel transformer / "deepconv", the taps-mrd v2
statistical scorer, and the MethylBERT read classifier), and specifies the decisive next
experiment. It is a planning artifact, not an implementation.

## The unifying diagnosis

Every method tried so far sits on one of two horns:

1. **Per-marker (site-level) and reference-anchored, so robust but insensitive.**
   NNLS, UXM, the beta-binomial / NMF / OT benchmarks, and the TapestryModel decoder all
   reduce each marker to a mean U-fraction (β / α) before modelling. Averaging dilutes the
   handful of tumour-derived molecules into an abundant healthy background, so the score
   floors out around a few percent (NNLS estimates ~4% in healthy controls). This is a
   signal-to-noise ceiling, not a tuning problem.

2. **Sample-label-supervised, so sensitive in principle but confounded.**
   MethylBERT and the TapestryModel learn from sample-level "cancer vs healthy" (or
   tissue-vs-plasma) labels. With few biological samples and AB/CD cohorts that differ in
   coverage, batch and molecule type, the learner fits the confound. MethylBERT's
   controlled variants show this cleanly: once read-length and AB/CD batch shortcuts are
   removed, read-level accuracy collapses to near chance
   (`docs/methylbert_experiment_summary.md`).

The augmented-NNLS unknown failure is a third face of the same problem: the unknown basis
`U` (top-K SVD of healthy-control residuals, **free-sign** coefficients;
`tapestry/benchmark/augmented_nnls.py:39-102,189-210`) is an unconstrained linear
direction, so the solver uses it to explain *any* deviation from the atlas — tumour signal
as readily as background. Nothing in the objective distinguishes tumour-like from
background-like deviation, so the unknown eats the tumour.

The two methods that actually worked partially — **taps-mrd v2** and the literature's
**read-level alpha deconvolution** — share the property the failures lack: they are
**reference-anchored** (score against a healthy null, not a sample label) **and they use
read-level concordance** (the fraction of fragments whose CpGs are *jointly* unmethylated
in the tumour direction), not just the mean. taps-mrd v2's specificity guard is exactly a
concordance statistic (`min_tumour_concordant`, `max_healthy_concordant_p95`), and Alpha-NNLS
(Nat Commun-adjacent, 2025) reports read-level α deconvolution beating site-level UXM by
~30× on TF error (MAE 0.004 vs 0.13) and detecting 0.5% in simulation. Read-level beats
site-level because one tumour molecule carries a coherent multi-CpG pattern that is
vanishingly unlikely under the healthy background; the mean throws that coherence away.

**Conclusion.** The next method should be (a) read/molecule-level to recover sub-1%
sensitivity, (b) reference-anchored to dodge the batch confound, (c) explicit about an
unknown component but constrained so it cannot absorb the tumour, (d) coverage-weighted so
CD's <5x does not dominate, and (e) calibrated against an empirical healthy null with
leave-batch-out so a low-TF detection claim is credible.

## Why MethylBERT does not refute the read-level idea

MethylBERT tested "is there read-level signal?" with a confounded, shortcut-prone
transformer over **broad** DSS DMRs (median ~17.7 kb). A 150 bp read overlaps a tiny,
often CpG-sparse slice of such a region, so most reads carry few discriminative CpGs and
the per-read concordance signal is washed out before any model sees it. The geometry was
wrong for read-level work. The right test is an **explicit per-read likelihood ratio on
tight, CpG-dense blocks** — no transformer, no SGD, no length/batch channel to exploit.
This is the "oracle read-level separability test" that the MethylBERT summary itself flags
as recommended next-diagnostic #1, and it has not yet been run.

## The decisive first experiment — read-level likelihood-ratio oracle

**Question.** In clean, length/batch-matched data, can an explicit per-read likelihood
ratio separate tumour-derived reads from healthy-cfDNA reads on tight CpG-dense blocks?

**Statistic.** For a block with reference epiallele profiles — tumour per-CpG methylation
`p_T` and healthy `p_H` (estimated from tumour-tissue PATs and healthy-control cfDNA PATs
respectively) — a read covering CpGs `S` with methylation calls `m_i` scores

```
LLR(read) = sum_{i in S} [ m_i log(p_T,i / p_H,i) + (1 - m_i) log((1-p_T,i)/(1-p_H,i)) ]
```

a per-read log-likelihood-ratio. Aggregate to a sample score by summing read LLRs (or by
counting reads above a per-read LLR threshold — the "tumour-pattern read count", which is
the molecule-level analogue of concordance). This is reference-defined: it has no access to
read length, cohort, or sample identity.

**Design (all length-matched and batch-controlled, learning from MethylBERT's failures):**

- Block panel: **tight, CpG-dense blocks**, not broad DSS DMRs. Two arms to test the
  geometry hypothesis directly — (i) the existing wgbs segmentation blocks (≥4 CpG,
  ~`blocks.bed`), tumour-selected; (ii) the broad MethylBERT DMRs, as a negative geometry
  control. If the oracle separates on (i) but not (ii), the user's "regions don't have
  enough CpGs" hypothesis is confirmed.
- Positives: reads from OAC tumour-tissue PATs. Negatives: reads from held-out healthy
  cfDNA PATs. Estimate `p_T`, `p_H` on training samples; score reads from held-out samples.
- Controls carried over from the diagnostics doc: exact read-length matching between T/N,
  leave-one-sample-out and leave-one-cohort-out (AB vs CD), and a label-permutation null.
- Metric: held-out per-read AUC, plus AUC stratified by length decile and by per-read CpG
  count. Genuine signal ⇒ AUC clearly above chance and roughly flat across length deciles.

**Decision rule.**

- **Separable** ⇒ the read-level signal exists; build the full program below. MethylBERT
  failed on extraction (shortcut + DMR geometry), not on absence of signal.
- **Not separable on tight blocks either** ⇒ accept that single-molecule tumour signal is
  genuinely absent in this OAC/TAPS data; fall back to sample-level concordance statistics
  + a calibrated head, and invest in marker geometry (dynamic CpG-dense segmentation à la
  Alpha-NNLS) before anything deeper.

**Scale.** Reuses `wgbstools`/`pattools` PAT extraction already in the pipeline. Reads over
~10^5–10^6 blocks across a few tens of PAT files is a single-node, streaming job — order
tens of minutes to ~1–2 h wall, a few GB RAM. No GPU, no training. To be run via a SLURM
wrapper (`slurm/`), not a direct python invocation.

**Implementation status (built, 2026-06-10).** The oracle is implemented and unit-tested:

- `tapestry/readlevel/llr_oracle.py` — reference-profile estimation, per-read LLR scoring,
  and the verdict metrics (overall weighted AUC; AUC stratified by per-read CpG count;
  label-permutation null; CpG-count-matched AUC; per-healthy-cohort AUC-vs-tumour).
- `scripts/run_read_level_llr_oracle.py` — CLI with train/test splitting
  (random, leave-cohort-out via `--holdout-cohort`, or explicit `--holdout-samples`).
- `slurm/oracle_read_level_llr.sh` — SLURM wrapper with the two-arm geometry example
  (tight blocks vs broad MethylBERT DMRs) baked in as comments.
- `tests/test_read_level_llr_oracle.py` — synthetic-PAT end-to-end test (passes: a planted
  tumour/healthy difference gives AUC > 0.9 with permutation null ~0.5; identical
  distributions give AUC ~0.5, not declared separable).

The 0/1 methylation encoding is taken from `tapestry.core.io.load_reads` and applied
identically to profile estimation and scoring, so the AUC is invariant to the
bisulfite-vs-TAPS convention. What remains is to build the `oracle_samples.tsv` manifest
(tumour-tissue PATs + healthy-cfDNA PATs) and run it on BMRC against `blocks.bed` and the
`collapsed_100kb` DMR panel.

## The full program (only if the oracle gate passes)

1. **Read-level LR scorer** (`tapestry/` module + SLURM wrapper): per-block `p_T`/`p_H`
   reference profiles, per-read LLR, sample aggregation to (tumour-pattern read count,
   summed LLR), each with coverage-derived uncertainty (beta-binomial on `p`, Poisson on
   counts). This is the sensitivity engine the per-marker benchmarks lack.

2. **Unknown as the healthy manifold, not a free axis.** Replace the free-sign SVD unknown
   with a constraint: the unknown is confined to the convex hull / learned manifold of
   observed healthy-cfDNA backgrounds (PCA subspace, or a small normalizing flow over the
   54 CD + 4 AB controls), and the tumour contrast is **projected out** of the unknown
   subspace — the repo already has `project_basis_orthogonal_to()`
   (`tapestry/benchmark/augmented_nnls.py`), but baked into the objective, not post-hoc.
   The unknown can then absorb "more healthy-like background" without absorbing tumour.

3. **Empirical, leave-batch-out null.** Characterise the score distribution in true
   negatives using the healthy controls with batch-aware cross-validation; report
   sensitivity at a fixed false-positive rate, and a per-sample detection limit from
   coverage × effect size. This is what licenses a <1% claim.

4. **TabPFN / TabICL as the calibrated decision head (the user's tabular idea).** Feed the
   engineered, batch-robust, read-level features per sample — summed LLR, tumour-pattern
   read count, per-cell-type fractions, unknown mass, coverage, fragmentomics-corrected
   variants — into an in-context tabular foundation model for the final cancer/no-cancer
   probability and (where ichorCNA exists, >3% TF) a TF regression, validated
   leave-one-batch-out. See the verdict below.

## Verdict on tabular foundation models (TabPFN / TabICL)

**Use them — but as the head, not the method.**

- *Why they fit here.* The sample-level decision is a small-N, many-engineered-feature
  problem, which is exactly TabPFN's regime. TabPFN-2.5 handles ≤50k samples and ≤2000
  features (TabPFN-3 more), so feature count is not a blocker. Being **in-context**, it does
  no SGD training on our data, so it cannot overfit the way the bespoke TapestryModel MLP /
  transformer did — this directly answers the "deepconv introduces training bias / overfits
  / breaks under domain shift" complaint, and it returns calibrated uncertainty.
- *What they do not fix.* They are a classifier/regressor over the features you hand them,
  not a feature extractor. Feed them raw per-marker U-fractions and they relearn the AB/CD
  batch split exactly like everything else. Their value is strictly conditional on the
  features being read-level, reference-anchored and batch-robust. And the <1% detection
  claim still rests on the empirical leave-batch-out null, which TabPFN's tiny in-context
  healthy set cannot densely characterise on its own.
- *Net.* TabPFN is a principled, overfit-resistant replacement for the deepconv decision
  head — adopt it at step 4 — but the novelty and the sensitivity have to come from the
  read-level LR + manifold-constrained unknown upstream of it.

## How this respects each prior failure

| Prior failure | What killed it | How this program answers it |
|---|---|---|
| NNLS / UXM | site-level averaging → ~4% floor, no unknown | read-level LR recovers sub-1% sensitivity; coverage-weighted; explicit unknown |
| Augmented-NNLS + unknown | free-sign SVD unknown absorbs tumour | unknown constrained to healthy manifold, tumour contrast projected out |
| TapestryModel / deepconv | SGD-trained on synthetic mixtures → domain-shift bias, no unknown | in-context TabPFN head (no SGD overfit) + explicit unknown; reference-anchored features |
| taps-mrd v2 | hand-tuned z-score + linear log(TF) extrapolates badly <1%; overfit risk | keeps its working concordance+null core; replaces hand-tuning with per-read LR + calibrated head + leave-batch-out null |
| MethylBERT | learned length/batch shortcut on broad DMRs | explicit per-read LR (no shortcut channel) on tight CpG-dense blocks; oracle gate decides viability first |
