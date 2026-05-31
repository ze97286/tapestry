# MethylBERT read-call workflow — runbook

Self-contained BMRC submit scripts for the read-call MethylBERT pipeline, in run order.
Each script sets its environment and submits the matching `run_methylbert_paper_*_slurm.sh`
wrapper. Run them from the repo checkout on BMRC.

## Main sequence

Run in numeric order; wait for each step to finish before the next (later steps `test`
their inputs and will exit if they are missing).

| # | Script | Does | Submits |
| --- | --- | --- | --- |
| 00 | `00_prepare_inputs.sh` | Build the reproducible `collapsed_100kb` DMR panel (and, with `BUILD_READ_CALL_LISTS=1`, the read-call sample sheet + bulk deconvolution sheet) | `run_methylbert_paper_prepare_inputs_slurm.sh` |
| 01 | `01_preprocess_read_call_shards.sh` | Preprocess per-sample read-call shards (array job) | `run_methylbert_paper_preprocess_read_calls_array_slurm.sh` |
| 02 | `02_merge_read_call_shards.sh` | Merge shards → `train_seq.csv` / `test_seq.csv` | `run_methylbert_paper_merge_read_call_shards_slurm.sh` |
| 03 | `03_finetune_read_classifier.sh` | Fine-tune the read classifier (GPU) | `run_methylbert_paper_finetune_slurm.sh` |
| 04 | `04_eval_heldout_reads.sh` | Evaluate on the held-out test set (+ shortcut strata) | `run_methylbert_paper_eval_slurm.sh` |
| 05 | `05_deconvolute_read_calls.sh` | Deconvolute cfDNA bulks (array job) | `run_methylbert_paper_deconvolute_read_calls_slurm.sh` |
| 06 | `06_collect_deconvolution.sh` | Collect per-sample results → summary | `run_methylbert_paper_collect_slurm.sh` |
| 07 | `07_validate_vs_ichorcna.sh` | θ vs ichorCNA + θ vs fragmentomics | `run_methylbert_paper_validate_slurm.sh` |

```bash
cd "${PROJECT_DIR}"   # the repo checkout on BMRC
./scripts/methylbert_bmrc_standalone/00_prepare_inputs.sh
./scripts/methylbert_bmrc_standalone/01_preprocess_read_call_shards.sh   # wait for the array to finish
./scripts/methylbert_bmrc_standalone/02_merge_read_call_shards.sh
./scripts/methylbert_bmrc_standalone/03_finetune_read_classifier.sh
./scripts/methylbert_bmrc_standalone/04_eval_heldout_reads.sh
./scripts/methylbert_bmrc_standalone/05_deconvolute_read_calls.sh        # wait for the array to finish
./scripts/methylbert_bmrc_standalone/06_collect_deconvolution.sh
./scripts/methylbert_bmrc_standalone/07_validate_vs_ichorcna.sh
```

Each script prints its Slurm job id (`--parsable`); to chain automatically, pass the prior
id to the next with `sbatch --dependency=afterok:<jobid>`.

## Sanity checks (success criteria after each step)

Run the check after each step's job completes. Set the work dir and variant first; array
steps (01, 05) should also show no `FAILED`/`TIMEOUT` tasks in `sacct -j <jobid>`.

```bash
WORK=/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.5_methylbert_bmrc/methylbert/OAC_methylbert_paper_bmrc
VARIANT=contained_sample_split   # match the run; each diagnostic uses its own VARIANT
```

**00 — DMR panel built (~100 regions):**

```bash
test -s "$WORK/dmrs_top100.collapsed_100kb.tsv" && wc -l "$WORK/dmrs_top100.collapsed_100kb.tsv"
# PASS: file exists, ~101 lines (header + 100). With BUILD_READ_CALL_LISTS=1 also:
cut -f2 "$WORK/read_call_lists/oac_dmr_read_calls.sample_sheet.tsv" | sort | uniq -c
# PASS: both T and N rows present; oac_bulk_read_calls.sample_sheet.tsv non-empty.
```

**01 — one rows.tsv per sample, new schema present:**

```bash
SHARD="$WORK/preprocess_taps_read_call_shards_${VARIANT}"
ls "$SHARD"/*/rows.tsv | wc -l          # PASS: == sample-sheet row count
find "$SHARD" -name rows.tsv -empty     # PASS: prints nothing (none empty)
head -1 "$(ls "$SHARD"/*/rows.tsv | head -1)" | tr '\t' '\n' | grep -E 'read_length|n_cpg'
# PASS: read_length and n_cpg appear in the header.
```

**02 — train/test built, balanced labels, no sample leakage:**

```bash
PRE="$WORK/preprocess_taps_read_calls_${VARIANT}"
test -s "$PRE/train_seq.csv" && test -s "$PRE/test_seq.csv"
cut -f5 "$PRE/train_seq.csv" | tail -n +2 | sort | uniq -c    # col5 = ctype; PASS: both T and N
comm -12 <(cut -f2 "$PRE/train_seq.csv" | tail -n +2 | sort -u) \
         <(cut -f2 "$PRE/test_seq.csv"  | tail -n +2 | sort -u)
# PASS: empty output — no sample (col2 = filename) is in both train and test.
```

**03 — model produced, runtime patch applied:**

```bash
MODEL="$WORK/model_taps_read_calls_${VARIANT}"
test -d "$MODEL/bert.model" && test -s "$MODEL/train_param.txt"
grep -h "methylbert_patches: applied" logs/methylbert_finetune_*.out   # PASS: attention_mask patch ran
grep -hE "step|loss|acc" logs/methylbert_finetune_*.out | tail         # loss should fall over steps
# PASS: bert.model + train_param.txt exist; patch line present; training ran to STEPS.
```

**04 — metrics + shortcut strata written:**

```bash
EVAL="$WORK/model_taps_read_calls_${VARIANT}/heldout_eval"
cat "$EVAL/summary.json"                 # accuracy, roc_auc, corr_prob_vs_read_length, corr_prob_vs_n_cpg
ls "$EVAL"/summary_by_{sample,cohort,length_decile,n_cpg}.tsv
# PASS (ran): files exist; test_predictions.tsv carries read_length/n_cpg.
# PASS (no shortcut): |corr_prob_vs_read_length| small AND accuracy ~flat across length deciles.
#   A high accuracy together with a high length/cohort correlation is the shortcut signature.
```

**05 — per-sample theta produced, covariates present:**

```bash
DEC="$WORK/deconvolution_taps_read_calls_${VARIANT}_adjusted"
ls "$DEC"/*/deconvolution.csv | wc -l    # PASS: == number of bulk samples
head -1 "$(ls "$DEC"/*/res.csv | head -1)" | tr '\t' '\n' | grep -E 'read_length|n_cpg|P_ctype'
# PASS: each sample dir has deconvolution.csv (T/N rows summing to 1) and res.csv with covariates.
```

**06 — summary has the fragmentomics columns:**

```bash
head -1 "$DEC/deconvolution_summary.csv" | tr ',' '\n' | grep -E 'sample|^T$|mean_read_length|mean_p_tumour'
wc -l "$DEC/deconvolution_summary.csv"   # PASS: one row per sample + header
```

**07 — theta tracks ichorCNA, not fragment length:**

```bash
cat "$DEC/validation/methylbert_theta_vs_fragmentomics.csv"
# PASS criterion: r(theta, ichor_tf) is the dominant correlation (approaching the NNLS
# benchmark ~0.885) and clearly exceeds r(theta, mean_read_length) and r(theta, n_reads_classified).
# If theta correlates with read length / read count more than with ichorCNA, that is the
# fragmentomics shortcut — investigate with the diagnostics below before trusting the estimate.
```

## Prerequisites for 00

- The DSS DMR calls at `${METHYLBERT_WORK_DIR}/dmr_pat/dss_dmrs.tsv` (from the DMR pipeline;
  see `docs/methylbert_paper_replication.md`). 00 collapses them into the `collapsed_100kb`
  panel and copies it to `dmrs_top100.collapsed_100kb.tsv`.
- To (re)build the read-call sample sheet and the bulk deconvolution sheet, export
  `BUILD_READ_CALL_LISTS=1` plus `TUMOUR_SAMPLE_LIST`, `NORMAL_SAMPLE_LIST`,
  `TUMOUR_READ_CALL_DIR`, `AB_READ_CALL_DIR`, `CD_READ_CALL_DIR`, and (for deconvolution)
  `BULK_SAMPLE_LIST` / `BULK_READ_CALL_DIR` before running 00.

## Diagnostics (shortcut controls)

Each control sets `VARIANT` (so it lands in its own output dirs and does not clobber the
main run) plus a toggle, then re-runs the relevant numbered steps. Full readouts and the
decisive same-molecule-type control are in `docs/methylbert_shortcut_diagnostics.md`.

| Control | Export, then run |
| --- | --- |
| Label permutation | `VARIANT=shuffle SHUFFLE_LABELS=1` → 02, 03, 04 |
| Methylation ablation | `VARIANT=blankmethyl BLANK_METHYL=1` → 01, 02, 03, 04 |
| DNA / locus ablation | `VARIANT=blankdna BLANK_DNA=1 COLLAPSE_DMR_LABEL=1` → 01, 02, 03, 04 |
| Read-length matching | `VARIANT=lenmatch LENGTH_MATCH=1` → 02, 03, 04 |
| Leave-one-batch-out | `VARIANT=lobo_cd HOLDOUT_COHORT=CD_plasma` → 02, 03, 04 |
| Leave-one-sample-out | `scripts/methylbert_loso_folds.py` to emit per-fold merge commands, then 03, 04 per fold |
| Random-region (DMR shuffle) | `scripts/generate_random_dmr_regions.py` → `METHYLBERT_DMRS=<random.tsv> VARIANT=randomregion` → 01, 02, 03, 04 |
| Deploy prior recalibration | `DECONV_PRIOR_T=0.5` → 05 |

The label permutation, length-match and leave-one-batch-out controls reuse the existing
preprocess shards (start from 02). The ablation and random-region controls change the read
content, so they start from 01.

## Runtime patches

`external/methylbert` is kept byte-identical to upstream. `scripts/methylbert_patches.py`
applies the `attention_mask` default and the deploy `margins_override` at runtime (called by
the finetune, eval and deconvolution runners). See `docs/methylbert_shortcut_diagnostics.md`.
