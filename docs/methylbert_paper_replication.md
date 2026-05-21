# MethylBERT Paper-Style BAM Workflow

This scaffold reproduces the paper-style flow from BAMs through tumour fraction estimation. It uses DSS for DMR discovery and the upstream `CompEpigen/methylbert` implementation for read preprocessing, fine-tuning, read classification, and MLE deconvolution. It does not convert through PAT.

## Inputs

- Tumour-tissue BAMs labelled `T`.
- Non-tumour/control BAMs for DMR calling labelled `N`.
- Healthy-control cfDNA BAMs for MethylBERT fine-tuning labelled `N`.
- Test cfDNA BAMs to deconvolute.
- A reference FASTA matching the BAM alignments.

The upstream code requires BAM methylation tags compatible with `--methylcaller bismark` or `--methylcaller dorado`.

## Steps

### One-command submission

Fill `configs/methylbert_oac_paper.env` and the list files in `data/methylbert/`, then validate the inputs:

```bash
source slurm/common.sh
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

- `METHYLBERT_REF_FASTA`: reference FASTA used for BAM alignment, with `.fai`.
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
source slurm/common.sh
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
sbatch --export=ALL run_methylbert_paper_dmr_slurm.sh
```

This extracts DSS `chr,pos,N,X` count tables from each BAM, runs `DMLtest`, calls DMRs with `delta=0.2`, `p=0.05`, `minCG=4`, `minlen=50`, `dis.merge=50`, and exports the top 100 DMRs by absolute `areaStat`.

2. Prepare fine-tuning reads from tumour and healthy-control BAMs:

```bash
source slurm/common.sh
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
sbatch --export=ALL run_methylbert_paper_preprocess_slurm.sh
```

3. Fine-tune upstream MethylBERT:

```bash
source slurm/common.sh
export METHYLBERT_CONFIG="${PROJECT_DIR}/configs/methylbert_oac_paper.env"
sbatch --export=ALL run_methylbert_paper_finetune_slurm.sh
```

4. Apply to cfDNA BAMs and run the upstream MLE deconvolution:

```bash
source slurm/common.sh
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
