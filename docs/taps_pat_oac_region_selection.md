# OAC TAPS/PAT Region Selection

The current OAC BAMs are aligned reads without Bismark `XM` or Dorado `MM/ML`
methylation tags.  Upstream MethylBERT's BAM preprocessing cannot recover
methylation from those files, so region selection should use the existing
TAPS/PAT/wgbstools atlas path instead.

This path stays entirely in hg38:

- hg38 CpG index: `/users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz`
- hg38 PAT index: `/users/zetzioni/sharedscratch/atlas/pat_index/cell_type_pat_index_l4.csv.gz`
- hg38 output regions: `/users/zetzioni/sharedscratch/atlas/markers/dmr_by_read.blood+gi+tum.100.l4.bed`

The public MethylBERT checkpoints are named for hg19 pretraining.  That does
not make hg19 coordinates valid for these BAMs.  If we later reuse a MethylBERT
checkpoint, treat it only as sequence-model initialization; preprocessing,
region coordinates, CpG indices, and reference FASTA must remain hg38 unless
the BAMs and all region files are explicitly lifted over and regenerated.

Run from the cluster checkout:

```bash
cd /mnt/scratch/users/zetzioni/tapestry
git pull

export PROJECT_DIR="$(pwd)"
export TAPS_PAT_REGION_CONFIG="${PROJECT_DIR}/configs/oac_taps_pat_regions.env"
sbatch --export=ALL run_oac_taps_pat_region_selection_slurm.sh
```

The Slurm job wraps the original deepconv command:

```bash
R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/generate_atlas.R \
  --cpg_file /users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz \
  --map_file /users/zetzioni/sharedscratch/atlas/classes/per-read-bed2class.csv \
  --base_dir /users/zetzioni/sharedscratch/atlas \
  --out_file /users/zetzioni/sharedscratch/atlas/markers/dmr_by_read.blood+gi+tum.100.l4.bed \
  --index_file /users/zetzioni/sharedscratch/atlas/pat_index/cell_type_pat_index_l4.csv.gz \
  --top_n 100 \
  --min_cpgs 4 \
  --threads 32 \
  --verbose
```

Optionally, after region selection, write an OAC-only hg38 DMR table for later
experiments by setting `TAPS_PAT_WRITE_METHYLBERT_DMRS=1` before submission.
That table lands at:

```text
runs/run_v0.5_taps_pat/methylbert/OAC_taps_pat_hg38_regions/dmrs_oac_hg38.tsv
```

That file is only a coordinate handoff for future work and requires Python with
pandas.  It does not make the upstream MethylBERT BAM preprocessor usable on
untagged BAMs.
