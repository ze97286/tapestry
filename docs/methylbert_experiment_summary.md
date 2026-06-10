# MethylBERT OAC/TAPS Experiment Summary

Date: 2026-06-10

This document summarises the MethylBERT work performed for the OAC/TAPS read-level tumour classifier, including the major run variants, what each was intended to test, the observed results, and the current interpretation.

## Executive Summary

We attempted to reproduce the MethylBERT paper-style workflow on OAC TAPS data:

1. Select tumour-specific DMRs using tumour tissue versus healthy cfDNA controls.
2. Extract reads overlapping those DMRs.
3. Fine-tune MethylBERT to classify reads as tumour-like (`T`) or control-like (`N`).
4. Apply the classifier to clinical cfDNA and estimate tumour fraction using the MethylBERT MLE/deconvolution step.

The project did not produce a reliable clinical tumour-fraction estimator. Apparent read-level signal disappeared when obvious batch and read-length shortcuts were removed.

The key finding is:

> In this OAC/TAPS setup, after controlling AB/CD batch composition and read-length artifacts, the MethylBERT read classifier shows little or no robust tumour-vs-control read-level signal.

This is not a small performance regression relative to the paper. It is a qualitative failure of the read-level separability assumption in the current data/implementation combination.

## Paper-Style Methodology Implemented

The workflow was designed to follow the MethylBERT paper as closely as possible:

- DMRs called using DSS from tumour tissue versus healthy/control samples.
- Top DMRs selected by `areaStat`.
- Reads overlapping selected DMRs converted into upstream-compatible `train_seq.csv` and `test_seq.csv`.
- Input representation:
  - reference DNA 3-mers;
  - methylation-state sequence;
  - DMR label;
  - read label `T` / `N`.
- Fine-tuning used the upstream MethylBERT model and pretrained 12-layer checkpoint.
- Evaluation used held-out read classification, then optional MLE tumour-fraction estimation.

Important implementation checks:

- The upstream MethylBERT 3-mer generator uses `range(len(seq)-k)`, and our preprocessor matches that convention.
- The upstream code keeps reads fully contained within the DMR interval; our default `contained` extraction matches that.
- The methylation-state mapping matches the paper convention:
  - `0` = unmethylated;
  - `1` = methylated;
  - `2` = non-CpG / uninformative.
- TAPS values were encoded into this same 0/1/2 state space.

## Data and Batch Structure

Healthy controls were split into two major groups:

- **AB controls**
  - 4 healthy plasma controls.
  - Higher coverage, roughly 30x.
  - Reads include lengths above 150 bp in some contexts.

- **CD controls**
  - Approximately 53-54 healthy plasma controls.
  - Lower coverage, roughly 3-6x.
  - Raw maximum read length observed was 150 bp.

Tumour data:

- OAC tumour tissue reads.
- Five tumour tissue samples were available for the main read-classifier training setup.
- Tumour reads included both 150 bp and 151 bp reads in the relevant full-length workflows.

This created two major risks:

- **Batch/control composition bias**: CD controls dominated by sample count in the original mixed-control setup.
- **Read-length shortcut**: CD controls maxed at 150 bp while tumour tissue contained 151 bp reads.

## DMR Discovery and Region Selection

### Original DMR Discovery

The PAT/DSS route generated a large merged DMR table:

- `dss_dmrs.tsv`: 366,305 DMR rows.
- Selected top 100 DMRs by absolute `areaStat`.

The literal top-100 DMRs were very large and hotspot-heavy:

- `n = 100`
- median length: ~17.7 kb
- mean length: ~20.1 kb
- maximum length: ~64.1 kb
- heavy enrichment on chromosomes 19 and 20.

Example top DMRs had tens of kilobases and hundreds to thousands of CpGs.

### DMR Variants

To reduce hotspot domination, several post-DSS variants were generated:

- `literal_top100`
- `capped10_top100`
- `collapsed_100kb_top100`
- `collapsed_500kb_top100`
- `collapsed_1000kb_top100`

The later v0.6 workflows used `collapsed_100kb` style panels so region selection was not dominated by a single local DMR cluster.

This is a deviation from a literal top-100 paper implementation, but it was introduced because the literal OAC DSS output was dominated by broad neighbouring regions.

## Main Experimental Variants

### 1. Initial Mixed-Control Run: AB + CD Controls

**Purpose:**  
Run the paper-style workflow with all available healthy controls.

**Setup:**

- Healthy controls: 4 AB + ~53/54 CD.
- Tumour tissue positives: 5 OAC tumour samples.
- Read examples were balanced at the label level:
  - train: 400,000 `N`, 400,000 `T`;
  - test: 100,000 `N`, 100,000 `T`.

**Held-out read classifier results:**

- accuracy: 0.7657
- ROC-AUC: 0.8100
- average precision: 0.8092
- label-level performance:
  - `N`: accuracy/specificity 0.889
  - `T`: sensitivity 0.642

At first glance this looked like a meaningful classifier.

**Problem found later:**

The errors were highly structured by control cohort:

- AB high-coverage controls:
  - mean `P(tumour)` ~0.594
  - fraction called tumour at 0.5 ~59.1%

- Other controls:
  - mean `P(tumour)` ~0.292
  - fraction called tumour at 0.5 ~2.1%

- Tumour tissue:
  - mean `P(tumour)` ~0.650
  - fraction called tumour at 0.5 ~64.2%

So the AB controls behaved almost tumour-like, while CD controls behaved cleanly healthy.

**Interpretation:**

This run was confounded. The model was not learning a stable tumour-vs-control rule. It was strongly affected by AB/CD control structure, and the control batch used in training/evaluation mattered as much as the biological label.

### 2. Initial Clinical MLE / ichorCNA Validation

**Purpose:**  
Apply the trained read classifier to clinical cfDNA and estimate tumour fraction using the MethylBERT MLE/deconvolution step.

**Observed result:**

The classifier/MLE tumour fraction versus ichorCNA plot was effectively unusable:

- matched samples: `n = 126`
- all-sample Pearson correlation: approximately `r = -0.433`
- timepoint-specific correlations were also poor or negative.

The plot showed many samples forced toward near-zero or high tumour fractions with no sensible relationship to ichorCNA.

**Interpretation:**

These clinical results were discarded. They were not treated as a valid negative biological result because the read classifier feeding the MLE was already confounded.

### 3. Balanced DMR Discovery: 4 AB + 4 CD

**Purpose:**  
Correct the original DMR selection imbalance, where CD controls dominated by sample count.

**Setup:**

- DSS background capped to:
  - 4 AB controls;
  - 4 CD controls.
- This was controlled with `DMR_MAX_BACKGROUND_PER_COHORT=4`.

**Net effect:**

This did rebalance DMR discovery at the control-sample level. It did not by itself solve the classifier problem, because downstream training still had to be tested under cohort and length controls.

### 4. `06_AB`: AB-Only DMR Selection and AB-Only Training

**Purpose:**  
Remove AB/CD control batch effects entirely by using only AB controls for both DMR selection and read-classifier training.

**Setup:**

- Region-selection controls: AB only.
- Training controls: AB only.
- Effectively, this left very few control samples:
  - several AB samples in training;
  - one AB sample in held-out validation.

**Held-out evaluation:**

- `n_reads`: 209,662
- accuracy: 0.5689
- ROC-AUC: 0.5727
- average precision: 0.5148
- mean probability matching true label:
  - `N`: 0.4952
  - `T`: 0.5472
- correlation of tumour probability with read length: 0.8051
- held-out samples:
  - `X2881_Ctrl_plasma_md` as `N`;
  - `071-021_ScrBsl_tumour_md` as `T`.

**Interpretation:**

AB-only removed the obvious AB/CD control-batch conflict, but did not produce a useful classifier. The read-length correlation remained high, and the tumour-control probability separation was weak.

### 5. `06_AB_lenmatch_exact`: AB-Only With Exact Length Matching

**Purpose:**  
Ask whether AB-only performance survives exact 1 bp T/N length matching.

**Setup:**

Merged training/test data after length matching:

- train:
  - `N`: 224,631
  - `T`: 253,636
- test:
  - `N`: 92,605
  - `T`: 63,600

**Held-out evaluation:**

- `n_reads`: 156,205
- accuracy: 0.5794
- ROC-AUC: 0.5274
- average precision: 0.4277
- mean probability matching true label:
  - `N`: 0.4678
  - `T`: 0.4703
- correlation of tumour probability with read length: -0.0162
- label-level threshold performance:
  - `N` accuracy: 0.8948
  - `T` accuracy: 0.1202

**Interpretation:**

Once exact length matching removed the read-length cue, AB-only signal collapsed to near chance. The model became conservative at threshold 0.5 and failed to call most tumour reads.

### 6. `06_AB_full_length_only`: AB-Only, Reads >=150 bp

**Purpose:**  
Remove short-read effects by keeping only full-length reads.

**Setup:**

- Keep reads with `read_length >= 150`.
- Balance labels after filtering.

Merged data:

- train:
  - `N`: 209,984
  - `T`: 220,615
- test:
  - `N`: 66,052
  - `T`: 55,421

**Held-out evaluation:**

- `n_reads`: 121,473
- accuracy: 0.6993
- ROC-AUC: 0.7457
- average precision: 0.7111
- mean probability matching true label:
  - `N`: 0.3776
  - `T`: 0.5612
- correlation of tumour probability with read length: -0.0414
- correlation with `n_cpg`: -0.0804

**Interpretation:**

This was the best artifact-controlled AB result, but it was still not strong enough to justify clinical deployment:

- only one held-out AB control sample and one held-out tumour sample;
- sensitivity remained modest;
- the signal was much weaker than the paper-style expectation;
- not enough evidence for robust generalisation.

### 7. `06_CD_full_length_only`: CD-Only, Reads >=150 bp

**Purpose:**  
Use the larger CD control cohort only. This avoids AB/CD batch mixing and gives many more healthy-control samples than AB-only.

**Setup:**

- Region-selection controls: CD only.
- Training controls: CD only.
- Keep reads with `read_length >= 150`.

**Held-out evaluation:**

- `n_reads`: 124,253
- accuracy: 0.8628
- ROC-AUC: 0.8740
- average precision: 0.9134
- mean probability matching true label:
  - `N`: 0.2009
  - `T`: 0.7750
- correlation of tumour probability with read length: 0.9975

Label-level:

- `N` accuracy: 0.9999
- `T` accuracy: 0.7218

At first glance this looked like the best result.

**Critical diagnostic:**

Read-length stratification showed the result was a shortcut:

- CD control reads were all 150 bp.
- Tumour reads were split between 150 bp and 151 bp.
- Tumour reads of length 150 looked like controls:
  - mean probability ~0.206
  - almost never called tumour.
- Tumour reads of length 151 were called tumour:
  - mean probability ~0.994
  - essentially all called tumour.

Raw control check confirmed CD controls had maximum read length 150.

**Interpretation:**

This run was not biological. It learned `151 bp => tumour`.

### 8. `06_CD_clip150`: CD-Only With Tumour Reads Clipped to 150 bp

**Purpose:**  
Keep the CD-only setup but remove the 151 bp tumour shortcut by clipping all model-facing reads to 150 bp.

This was preferred over requiring tumour reads to already be 150 bp, because otherwise too much tumour data would be discarded and a new bias could be introduced.

**Sanity check after merge:**

Model-facing read lengths:

- `N`, 150 bp: 306,078
- `T`, 150 bp: 306,078

Original read lengths still tracked provenance:

- `N`, original 150 bp: 306,078
- `T`, original 150 bp: 84,115
- `T`, original 151 bp: 221,963

So tumour 151 bp reads were retained but clipped to a 150 bp representation.

**Held-out evaluation:**

- `n_reads`: 124,253
- accuracy: 0.5535
- ROC-AUC: 0.5763
- average precision: 0.5599
- mean probability matching true label:
  - `N`: 0.4818
  - `T`: 0.4937
- correlation with read length: `NaN` because all model-facing reads had length 150.
- correlation with `n_cpg`: 0.1393

Label-level:

- `N` accuracy: 0.6535
- `T` accuracy: 0.4507

Sample-level control means were nearly identical across CD controls:

- most CD controls had mean probability ~0.481-0.482.
- held-out tumour was only slightly higher at ~0.494.

**Interpretation:**

After removing the length shortcut, the CD-only signal collapsed. This was the decisive check showing that the strong CD-only result was not a methylation classifier.

### 9. `06_ABCD`: Balanced AB + CD Setup

**Purpose:**  
Test a cleaner mixed-control setup:

- 4 AB controls;
- 4 CD controls;
- no domination of DMR discovery by CD sample count.

**Fine-tuning log excerpt:**

- train sequences: 664,426
  - `N`: 264,257
  - `T`: 400,169
- test sequences: 197,930
  - `N`: 98,099
  - `T`: 99,831
- validation loss improved only modestly:
  - step 0: ~0.713
  - step 109: ~0.710
  - step 149: ~0.690

**Interpretation:**

The balanced AB+CD setup was scientifically cleaner than the initial mixed run, but it did not become the main positive result. The available logs did not show a compelling classifier, and the later length-controlled AB/CD and CD-only diagnostics indicated that obvious apparent signal was not robust.

## Summary Table

| Variant | Main Question | Apparent Result | Final Interpretation |
| --- | --- | --- | --- |
| Initial AB+CD mixed controls | Does paper-style workflow work out of the box? | AUC ~0.81 | Confounded by AB/CD control structure; AB controls looked tumour-like |
| Initial MLE vs ichorCNA | Does classifier-derived MLE track clinical tumour fraction? | Pearson r ~-0.43 | Invalid downstream result; classifier already confounded |
| Balanced DMR 4 AB + 4 CD | Does balanced DMR discovery fix control imbalance? | DMR imbalance fixed | Not sufficient by itself |
| `06_AB` | AB-only, no CD batch | AUC ~0.57 | No useful signal; length correlation remained high |
| `06_AB_lenmatch_exact` | AB-only with exact length matching | AUC ~0.53 | Near chance |
| `06_AB_full_length_only` | AB-only full-length reads | AUC ~0.75 | Weak/moderate signal, but too limited and not clinically convincing |
| `06_CD_full_length_only` | CD-only with many controls | AUC ~0.87 | Pure read-length shortcut |
| `06_CD_clip150` | CD-only after clipping length shortcut | AUC ~0.58 | Near chance |
| `06_ABCD` | Balanced AB+CD controls | No compelling positive result retained | Cleaner setup but no resolved signal |

## What We Learned

### The initial signal was not trustworthy

The first apparently useful classifier was dominated by structured differences among the controls. AB high-coverage controls were scored tumour-like, while CD controls were scored healthy-like.

This means the model was sensitive to cohort/batch/source structure rather than only tumour biology.

### The CD-only strong result was not biology

The CD-only full-length classifier looked strong until read length was inspected. The model had learned that 151 bp reads were tumour. CD controls had maximum read length 150, while tumour tissue contained 151 bp reads.

After clipping tumour reads to 150 bp, the signal disappeared.

### AB-only did not rescue the method

AB-only avoided the AB/CD batch conflict, but the available control sample count was very small. Exact length matching reduced performance to near chance. Full-length-only AB showed a weak/moderate signal, but not enough to justify downstream clinical use.

### DMR-level signal does not guarantee read-level classifier signal

The DMRs can be strongly different in aggregate methylation space while individual 150 bp reads inside those regions are not reliably classifiable.

This is not a proof that the DMRs are wrong. It means that the read-level supervised task may not be separable in this OAC/TAPS data after technical shortcuts are removed.

### The reason for failure remains unresolved

We have ruled out several superficial explanations:

- The TAPS 0/1/2 methylation-state mapping appears consistent with the paper.
- The 3-mer tokenisation convention matches upstream MethylBERT.
- The fully-contained-read behaviour matches upstream MethylBERT.
- The CD-only positive result was explained by read length.

But we have not yet proven whether the remaining failure is caused by:

- true absence of single-read tumour/control separability in these OAC/TAPS DMRs;
- TAPS/per-read-call semantics not matching what the paper's bisulfite-derived input represents;
- DMRs that are valid in bulk but too broad/noisy for read-level classification;
- insufficient tumour/control diversity after clean cohort control;
- some remaining implementation mismatch not yet identified.

## Current Scientific Conclusion

The MethylBERT read-classifier route should not be used for clinical tumour-fraction estimation in the current OAC/TAPS workflow.

Every robust-looking signal so far has either:

- been confounded by AB/CD control structure; or
- been explained by read length.

When those shortcuts are removed, classifier performance is near chance or only weakly above chance.

## Recommended Next Diagnostics

### 1. Oracle read-level separability test

Before any further MethylBERT training, test whether the clipped reads are separable by a simple DMR-aware statistical classifier:

- Use the same train/test split.
- Ignore transformer architecture.
- For each read, score observed methylation states against tumour/control methylation profiles for that DMR.
- Evaluate AUC on held-out reads.

Interpretation:

- If the oracle cannot separate reads, the signal is genuinely absent at the read level.
- If the oracle separates reads but MethylBERT does not, the issue is model/preprocessing/implementation.

### 2. Paper positive-control replication

Run the pipeline on a paper-like public dataset or as close as possible to the paper's CRC/PDAC setup.

Interpretation:

- If our pipeline reproduces the paper-like result, the OAC/TAPS data are the issue.
- If it fails, our implementation still differs from the paper in an important way.

### 3. TAPS-specific input audit

Audit the per-read TAPS conversion directly:

- methylated/unmethylated offset semantics;
- strand/orientation;
- handling of SNP CpGs;
- missing CpGs;
- reference reconstruction;
- whether a TAPS read with known calls produces the expected `dna_seq` and `methyl_seq`.

### 4. DMR tightness / informative-CpG audit

For each selected DMR:

- locate which subregions actually drive tumour/control methylation difference;
- quantify per-read informative CpG count;
- compare per-read methylation-pattern distributions, not only mean methylation;
- test shorter/tighter sub-DMRs rather than broad DSS intervals.

## Bottom Line

The experiment did not fail because of one bad Slurm run or one bad hyperparameter. The controlled variants show a consistent pattern:

> Once batch and length shortcuts are removed, the read-level tumour signal is weak or absent in the current OAC/TAPS MethylBERT setup.

Further downstream MLE/clinical validation is not scientifically meaningful until a clean held-out read classifier shows robust tumour-control separation that is not explained by read length, cohort, coverage, or DMR/locus artifacts.
