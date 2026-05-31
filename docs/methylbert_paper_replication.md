# MethylBERT Paper-Style BAM Workflow

This scaffold reproduces the paper-style flow from methylation evidence through tumour fraction estimation. It uses DSS for DMR discovery and the upstream `CompEpigen/methylbert` implementation for read preprocessing, fine-tuning, read classification, and MLE deconvolution. The original upstream MethylBERT path reads methylation-tagged BAMs; for OAC TAPS inputs without BAM methylation tags, the DMR step can read PAT files instead.

For the current OAC TAPS BAMs, the BAM-based extraction route is not usable
as-is: tag inspection shows no Bismark `XM` tags and no Dorado `MM/ML` tags, so
upstream MethylBERT extracts zero methylation-bearing reads.  The closest DMR
analogue to the paper is the PAT/DSS DMR job:
`run_methylbert_paper_dmr_pat_slurm.sh`.

## Inputs

- Tumour-tissue BAMs labelled `T`.
- Non-tumour/control BAMs for DMR calling labelled `N`.
- Healthy-control cfDNA BAMs for MethylBERT fine-tuning labelled `N`.
- Test cfDNA BAMs to deconvolute.
- A reference FASTA matching the BAM alignments. The config can point at the shared `.fa.gz`; Slurm jobs stage a temporary uncompressed `.fa` under local scratch for tools that require plain FASTA.

The upstream code requires BAM methylation tags compatible with `--methylcaller bismark` or `--methylcaller dorado`.

## Steps

### One-command submission

Fill `configs/methylbert_oac_paper.env` and the list files in `data/methylbert/`, then validate the inputs:

```bash
export PROJECT_DIR="$(pwd)"
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
python scripts/validate_methylbert_inputs.py --config "${METHYLBERT_CONFIG}" --check-tags
```

Then submit the full dependency chain:

```bash
bash run_methylbert_paper_e2e_slurm.sh
```

The submitted jobs are:

1. `run_methylbert_paper_dmr_slurm.sh`
2. `run_methylbert_paper_preprocess_slurm.sh`
3. `run_methylbert_paper_finetune_slurm.sh`
4. `run_methylbert_paper_deconvolute_slurm.sh` as a Slurm array
5. `run_methylbert_paper_collect_slurm.sh`

The config must provide:

- `METHYLBERT_REF_FASTA_GZ`: shared compressed reference FASTA used for BAM alignment.
- `METHYLBERT_REF_FASTA`: optional existing uncompressed FASTA, with `.fai`; leave empty to use local scratch staging from `METHYLBERT_REF_FASTA_GZ`.
- `METHYLBERT_REF_STAGE_DIR`: optional shared staging directory; leave empty to use per-job local scratch.
- `METHYLBERT_METHYLCALLER`: `bismark` for `XM` tags or `dorado` for `MM/ML` tags.
- `METHYLBERT_DMR_TUMOUR_BAM_LIST`: tumour tissue BAMs for DSS DMR calling.
- `METHYLBERT_DMR_NORMAL_BAM_LIST`: non-tumour/background BAMs for DSS DMR calling.
- `METHYLBERT_TUMOUR_BAM_LIST`: tumour tissue BAMs for MethylBERT positive read examples.
- `METHYLBERT_CONTROL_BAM_LIST`: healthy-control cfDNA BAMs for MethylBERT negative read examples.
- `METHYLBERT_BULK_BAM_LIST`: cfDNA BAMs to estimate tumour fraction for.
- `METHYLBERT_ENV_COMMAND`: optional activation command for an environment with Python MethylBERT dependencies, `pysam`, R, `DSS`, and `optparse`.

### Manual step-by-step

1. Call tumour-specific DMRs from BAMs using DSS:

```bash
export PROJECT_DIR="$(pwd)"
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
sbatch --export=ALL run_methylbert_paper_dmr_slurm.sh
```

If the BAMs are untagged but PAT files exist, run the PAT/DSS DMR path instead:

```bash
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
sbatch --export=ALL run_methylbert_paper_dmr_pat_slurm.sh
```

On BMRC, use the PAT-only config so the job does not try to load cortex3
modules:

```bash
export PROJECT_DIR="$(pwd)"
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper_bmrc.env"
# Pick the BMRC module names from `module avail R` / `module avail libxml2`.
export METHYLBERT_SETUP_MODULES="<BMRC_R_module> <BMRC_libxml2_module_if_needed>"
export METHYLBERT_DMR_MODULES="${METHYLBERT_SETUP_MODULES}"
# The BMRC config deactivates conda before resolving Rscript/gcc/xml2-config,
# then loads the BMRC modules above.
scripts/setup_methylbert_venv.sh
./run_methylbert_paper_dmr_pat_submit.sh
```

The BAM path extracts DSS `chr,pos,N,X` count tables from each tagged BAM. The PAT path extracts the same DSS count tables from `.pat.gz` files. Both then run `DMLtest`, call DMRs with `delta=0.2`, `p=0.05`, `minCG=4`, `minlen=50`, `dis.merge=50`, and export the top 100 DMRs by absolute `areaStat`. The PAT path also writes a BED file next to `dmrs_top100.tsv` for region filtering.

The BMRC PAT path is submitted as three Slurm phases: prepare count tables and
chromosome splits, run one DSS job per chromosome, then merge chromosome-level
DMRs and select the global top 100. Override chromosome job memory with
`METHYLBERT_DMR_CHR_MEM`, for example `export METHYLBERT_DMR_CHR_MEM=512G`.

The checked-in PAT lists define the same biological contrast as the paper-style
BAM DMR list: OAC tumour tissue (`T`) versus cfDNA controls (`N`). They do not
use the atlas cell-type reference panel as the DSS background.

The checked-in PAT lists and `data/CpG.bed.gz` convention are hg38. Keep the
DMR BED in hg38 for the current hg38-aligned OAC inputs; only lift over if a
later downstream step is run against hg19-aligned reads.

After DMR discovery, create quick SVG/HTML visual summaries:

```bash
python scripts/plot_methylbert_dmrs.py \
  --bed "${METHYLBERT_WORK_DIR}/dmrs_top100.bed" \
  --dmr-tsv "${METHYLBERT_WORK_DIR}/dmrs_top100.tsv" \
  --output-dir "${METHYLBERT_WORK_DIR}/dmr_plots" \
  --prefix oac_methylbert_top100
```

If the literal top 100 is dominated by one chromosome or locus, create
post-DSS alternatives before choosing the BED for MethylBERT filtering:

```bash
python scripts/select_methylbert_dmr_variants.py \
  --input "${METHYLBERT_WORK_DIR}/dmr_pat/dss_dmrs.tsv" \
  --output-dir "${METHYLBERT_WORK_DIR}/dmr_variants" \
  --top-n 100 \
  --collapse-distances 100000 500000 1000000 \
  --max-per-chrom 10

for variant in literal_top100 capped10_top100 collapsed_100kb_top100 collapsed_500kb_top100 collapsed_1000kb_top100; do
  python scripts/plot_methylbert_dmrs.py \
    --bed "${METHYLBERT_WORK_DIR}/dmr_variants/${variant}.bed" \
    --dmr-tsv "${METHYLBERT_WORK_DIR}/dmr_variants/${variant}.tsv" \
    --output-dir "${METHYLBERT_WORK_DIR}/dmr_variants/${variant}_plots" \
    --prefix "${variant}"
done
```

This does not rerun DSS. It only reranks the merged DSS calls into a literal
top set, a chromosome-capped set, and locus-collapsed sets for inspection.

The read-call workflow consumes `dmrs_top100.collapsed_100kb.tsv`. That panel is
the `collapsed_100kb_top100.tsv` variant produced above (100000 is now a default
collapse distance); copy or symlink it to
`${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv` and keep it under version
control so the regions actually used are reproducible. Note this locus collapse is a
deviation from the paper's literal top-100 by `areaStat`; see
`docs/methylbert_shortcut_diagnostics.md`.

2. Prepare fine-tuning reads.

For the OAC PAT/BMRC path, use the PAT preprocessor and point it at the
selected DMR set:

```bash
export METHYLBERT_DMRS="${METHYLBERT_WORK_DIR}/dmrs_top100.collapsed_100kb.tsv"
export PREPROCESS_DIR="${METHYLBERT_WORK_DIR}/preprocess_pat_collapsed_100kb"
export PAT_MAX_READS_PER_SAMPLE=200000
export PAT_MAX_READS_PER_LABEL=500000
sbatch --export=ALL run_methylbert_paper_preprocess_pat_slurm.sh
```

This writes upstream-compatible `train_seq.csv` and `test_seq.csv` from PAT
read patterns. Use this path when BAMs do not carry methylation tags.

For methylation-tagged BAMs, use the original upstream BAM preprocessor:

```bash
export PROJECT_DIR="$(pwd)"
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
sbatch --export=ALL run_methylbert_paper_preprocess_slurm.sh
```

3. Fine-tune upstream MethylBERT:

```bash
export PROJECT_DIR="$(pwd)"
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
sbatch --export=ALL run_methylbert_paper_finetune_slurm.sh
```

4. Apply to cfDNA BAMs and run the upstream MLE deconvolution:

```bash
export PROJECT_DIR="$(pwd)"
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
sbatch --export=ALL run_methylbert_paper_deconvolute_slurm.sh
```

For array execution, submit the deconvolution script with `--array=1-N`, where `N` is the number of lines in `METHYLBERT_BULK_BAM_LIST`.

## Outputs

- `${OUTPUT_DIR}/methylbert/${RUN_LABEL}/preprocess/train_seq.csv`
- `${OUTPUT_DIR}/methylbert/${RUN_LABEL}/preprocess/test_seq.csv`
- `${OUTPUT_DIR}/methylbert/${RUN_LABEL}/preprocess/dmrs.csv`
- `${OUTPUT_DIR}/methylbert/${RUN_LABEL}/model/bert.model/`
- `${OUTPUT_DIR}/methylbert/${RUN_LABEL}/deconvolution/<sample>/deconvolution.csv`
- `${OUTPUT_DIR}/methylbert/${RUN_LABEL}/deconvolution/deconvolution_summary.csv`

## What This Tests

This is the literal paper-style experiment on our data: classify reads overlapping tumour-specific DMRs as tumour-like versus normal-like, then estimate tumour fraction from all cfDNA reads using the upstream Bayes/MLE deconvolution.
