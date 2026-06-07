# MethylBERT BMRC v0.6 runbook

This directory defines clean MethylBERT experiments. They write to separate output
directories and do not require manual exports.

| Experiment | Region-selection controls | Training controls | Merge-time length control | Work dir |
| --- | --- | --- | --- | --- |
| `06_AB` | AB controls only | AB controls only | none | `/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_AB` |
| `06_AB_lenmatch_exact` | AB controls only | AB controls only | exact 1 bp T/N length matching, then label balance | `/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_AB_lenmatch_exact` |
| `06_AB_full_length_only` | AB controls only | AB controls only | keep reads with `read_length >= 150`, then label balance | `/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_AB_full_length_only` |
| `06_CD_full_length_only` | CD controls only | CD controls only | keep reads with `read_length >= 150`, then label balance | `/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_CD_full_length_only` |
| `06_CD_clip150` | CD controls only | CD controls only | keep reads with `read_length >= 150`; crop reads >150 bp to 150 bp before tokenization; then label balance | `/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_CD_clip150` |
| `06_ABCD` | 4 AB + 4 CD controls | 4 AB + 4 CD controls | none | `/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_ABCD` |

Scientific intent:

- The previous DMR panel was vulnerable to the 4 AB / 51 CD imbalance.
- `06_AB` tests the strict no-CD-background hypothesis: select regions and train controls
  only against the AB control batch.
- `06_AB_lenmatch_exact` asks whether the AB-only read classifier survives after exact
  T/N read-length matching and T/N label balancing.
- `06_AB_full_length_only` asks whether the AB-only read classifier survives when the
  short-read cue is removed by keeping only reads with `read_length >= 150`, with T/N
  label balancing after filtering.
- `06_CD_full_length_only` repeats the full-length-only test using the CD control cohort
  only. This gives many more control samples, at the cost of lower per-sample coverage.
- `06_CD_clip150` is the corrected CD-only diagnostic after discovering that raw CD
  controls are capped at 150 bp while tumour tissue contains 151 bp reads. It keeps
  tumour 151 bp reads but clips them to a 150 bp representation before 3-mer tokenization,
  so the model cannot use EOS/padding position as a tumour shortcut.
- `06_ABCD` tests the balanced-background hypothesis: select regions and train controls
  against equal AB and CD representation.
- Do not mix files between these experiments. Every script derives its output directory from
  its own parent folder.

## Run Order

Run each numbered step manually after the previous Slurm job or array has finished. The
experiments can be run in parallel at the same step.

```bash
cd /gpfs3/well/ludwig/users/uii408/tapestry

# DMR region selection
./scripts/methylbert_bmrc_v06/06_AB/00a_dmr_prepare.sh
./scripts/methylbert_bmrc_v06/06_AB_lenmatch_exact/00a_dmr_prepare.sh
./scripts/methylbert_bmrc_v06/06_AB_full_length_only/00a_dmr_prepare.sh
./scripts/methylbert_bmrc_v06/06_CD_full_length_only/00a_dmr_prepare.sh
./scripts/methylbert_bmrc_v06/06_CD_clip150/00a_dmr_prepare.sh
./scripts/methylbert_bmrc_v06/06_ABCD/00a_dmr_prepare.sh

# wait for prepare jobs, then:
./scripts/methylbert_bmrc_v06/06_AB/00b_dmr_chr_array.sh
./scripts/methylbert_bmrc_v06/06_AB_lenmatch_exact/00b_dmr_chr_array.sh
./scripts/methylbert_bmrc_v06/06_AB_full_length_only/00b_dmr_chr_array.sh
./scripts/methylbert_bmrc_v06/06_CD_full_length_only/00b_dmr_chr_array.sh
./scripts/methylbert_bmrc_v06/06_CD_clip150/00b_dmr_chr_array.sh
./scripts/methylbert_bmrc_v06/06_ABCD/00b_dmr_chr_array.sh

# wait for arrays, then:
./scripts/methylbert_bmrc_v06/06_AB/00c_dmr_merge.sh
./scripts/methylbert_bmrc_v06/06_AB_lenmatch_exact/00c_dmr_merge.sh
./scripts/methylbert_bmrc_v06/06_AB_full_length_only/00c_dmr_merge.sh
./scripts/methylbert_bmrc_v06/06_CD_full_length_only/00c_dmr_merge.sh
./scripts/methylbert_bmrc_v06/06_CD_clip150/00c_dmr_merge.sh
./scripts/methylbert_bmrc_v06/06_ABCD/00c_dmr_merge.sh

# collapse DMRs and build read-call sample sheets
./scripts/methylbert_bmrc_v06/06_AB/00d_prepare_inputs.sh
./scripts/methylbert_bmrc_v06/06_AB_lenmatch_exact/00d_prepare_inputs.sh
./scripts/methylbert_bmrc_v06/06_AB_full_length_only/00d_prepare_inputs.sh
./scripts/methylbert_bmrc_v06/06_CD_full_length_only/00d_prepare_inputs.sh
./scripts/methylbert_bmrc_v06/06_CD_clip150/00d_prepare_inputs.sh
./scripts/methylbert_bmrc_v06/06_ABCD/00d_prepare_inputs.sh

# read-call preprocessing
./scripts/methylbert_bmrc_v06/06_AB/01_preprocess_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_AB_lenmatch_exact/01_preprocess_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_AB_full_length_only/01_preprocess_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_CD_full_length_only/01_preprocess_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_CD_clip150/01_preprocess_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_ABCD/01_preprocess_read_call_shards.sh

# wait for arrays, then:
./scripts/methylbert_bmrc_v06/06_AB/02_merge_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_AB_lenmatch_exact/02_merge_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_AB_full_length_only/02_merge_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_CD_full_length_only/02_merge_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_CD_clip150/02_merge_read_call_shards.sh
./scripts/methylbert_bmrc_v06/06_ABCD/02_merge_read_call_shards.sh

# wait, then fine-tune:
./scripts/methylbert_bmrc_v06/06_AB/03_finetune_read_classifier.sh
./scripts/methylbert_bmrc_v06/06_AB_lenmatch_exact/03_finetune_read_classifier.sh
./scripts/methylbert_bmrc_v06/06_AB_full_length_only/03_finetune_read_classifier.sh
./scripts/methylbert_bmrc_v06/06_CD_full_length_only/03_finetune_read_classifier.sh
./scripts/methylbert_bmrc_v06/06_CD_clip150/03_finetune_read_classifier.sh
./scripts/methylbert_bmrc_v06/06_ABCD/03_finetune_read_classifier.sh

# wait, then held-out read evaluation:
./scripts/methylbert_bmrc_v06/06_AB/04_eval_heldout_reads.sh
./scripts/methylbert_bmrc_v06/06_AB_lenmatch_exact/04_eval_heldout_reads.sh
./scripts/methylbert_bmrc_v06/06_AB_full_length_only/04_eval_heldout_reads.sh
./scripts/methylbert_bmrc_v06/06_CD_full_length_only/04_eval_heldout_reads.sh
./scripts/methylbert_bmrc_v06/06_CD_clip150/04_eval_heldout_reads.sh
./scripts/methylbert_bmrc_v06/06_ABCD/04_eval_heldout_reads.sh
```

Clinical deconvolution remains optional until the held-out diagnostics look sane:

```bash
./scripts/methylbert_bmrc_v06/06_AB/05_deconvolute_read_calls.sh
./scripts/methylbert_bmrc_v06/06_ABCD/05_deconvolute_read_calls.sh

./scripts/methylbert_bmrc_v06/06_AB/06_collect_deconvolution.sh
./scripts/methylbert_bmrc_v06/06_ABCD/06_collect_deconvolution.sh

./scripts/methylbert_bmrc_v06/06_AB/07_validate_vs_ichorcna.sh
./scripts/methylbert_bmrc_v06/06_ABCD/07_validate_vs_ichorcna.sh
```

## Sanity Checks

Set:

```bash
AB=/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_AB
ABCD=/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_ABCD
CDCLIP=/gpfs3/well/ludwig/users/uii408/tapestry/runs/run_v0.6_methylbert_bmrc/methylbert/06_CD_clip150
```

After `00a`:

```bash
wc -l "$AB/dmr_pat/selected_background_pats.list"
cat "$AB/dmr_pat/selected_background_pats.list"
# PASS: 4 AB control PATs, no CD.

wc -l "$ABCD/dmr_pat/selected_background_pats.list"
cat "$ABCD/dmr_pat/selected_background_pats.list"
# PASS: 8 control PATs = 4 AB + 4 CD.
```

After `00c`:

```bash
test -s "$AB/dmr_pat/dss_dmrs.tsv" && wc -l "$AB/dmr_pat/dss_dmrs.tsv"
test -s "$ABCD/dmr_pat/dss_dmrs.tsv" && wc -l "$ABCD/dmr_pat/dss_dmrs.tsv"
```

After `00d`:

```bash
test -s "$AB/dmrs_top100.collapsed_100kb.tsv" && wc -l "$AB/dmrs_top100.collapsed_100kb.tsv"
cut -f2 "$AB/read_call_lists/oac_dmr_read_calls.sample_sheet.tsv" | sort | uniq -c
# PASS: 4 N, 5 T.

test -s "$ABCD/dmrs_top100.collapsed_100kb.tsv" && wc -l "$ABCD/dmrs_top100.collapsed_100kb.tsv"
cut -f2 "$ABCD/read_call_lists/oac_dmr_read_calls.sample_sheet.tsv" | sort | uniq -c
# PASS: 8 N, 5 T.

test -s "$CDCLIP/dmrs_top100.collapsed_100kb.tsv" && wc -l "$CDCLIP/dmrs_top100.collapsed_100kb.tsv"
cut -f2 "$CDCLIP/read_call_lists/oac_dmr_read_calls.sample_sheet.tsv" | sort | uniq -c
# PASS: many CD N, 5 T.
```

After `01`:

```bash
find "$AB/preprocess_taps_read_call_shards_06_AB" -mindepth 2 -maxdepth 2 -name rows.tsv | wc -l
# PASS: 9.

find "$ABCD/preprocess_taps_read_call_shards_06_ABCD" -mindepth 2 -maxdepth 2 -name rows.tsv | wc -l
# PASS: 13.

find "$CDCLIP/preprocess_taps_read_call_shards_06_CD_clip150" -mindepth 2 -maxdepth 2 -name rows.tsv | wc -l
# PASS: CD N sample count + 5 T.
```

After `02`:

```bash
for W in "$AB" "$ABCD"; do
  V="$(basename "$W")"
  PRE="$W/preprocess_taps_read_calls_${V}"
  test -s "$PRE/train_seq.csv" && test -s "$PRE/test_seq.csv"
  cut -f5 "$PRE/train_seq.csv" | tail -n +2 | sort | uniq -c
  cut -f5 "$PRE/test_seq.csv" | tail -n +2 | sort | uniq -c
done

PRE="$CDCLIP/preprocess_taps_read_calls_06_CD_clip150"
test -s "$PRE/train_seq.csv" && test -s "$PRE/test_seq.csv"
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next} {n[$h["ctype"] "\t" $h["read_length"]]++} END{for(k in n) print k "\t" n[k]}' "$PRE/train_seq.csv" "$PRE/test_seq.csv" | sort
# PASS: all model-facing read_length values are 150.

awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next} {n[$h["ctype"] "\t" $h["original_read_length"]]++} END{for(k in n) print k "\t" n[k]}' "$PRE/train_seq.csv" "$PRE/test_seq.csv" | sort
# PASS: tumour rows include original_read_length 151, showing they were clipped rather than discarded.
```

After `04`, inspect:

```bash
cat "$AB/model_taps_read_calls_06_AB/heldout_eval/summary.json"
column -t "$AB/model_taps_read_calls_06_AB/heldout_eval/summary_by_cohort.tsv"

cat "$ABCD/model_taps_read_calls_06_ABCD/heldout_eval/summary.json"
column -t "$ABCD/model_taps_read_calls_06_ABCD/heldout_eval/summary_by_cohort.tsv"

cat "$CDCLIP/model_taps_read_calls_06_CD_clip150/heldout_eval/summary.json"
column -t "$CDCLIP/model_taps_read_calls_06_CD_clip150/heldout_eval/summary_by_cohort.tsv"
```

Primary decision point:

- If AB-only fixes control specificity but ABCD does not, CD-driven region selection remains
  the suspect.
- If both fail, the problem is downstream of region selection or intrinsic to the read
  representation / paper-method transfer.
- Do not run or interpret clinical deconvolution unless held-out control specificity is
  acceptable.
