#!/bin/bash
# Submit PAT/DSS DMR discovery as prepare -> chromosome array -> merge.

set -euo pipefail

mkdir -p logs

array_range="${METHYLBERT_DMR_CHROM_ARRAY:-1-22}"
chr_mem="${METHYLBERT_DMR_CHR_MEM:-256G}"
chr_time="${METHYLBERT_DMR_CHR_TIME:-24:00:00}"

prepare_jid="$(sbatch --parsable --export=ALL run_methylbert_paper_dmr_pat_prepare_slurm.sh)"
prepare_dep="${prepare_jid%%;*}"

chrom_jid="$(
    sbatch \
        --parsable \
        --dependency="afterok:${prepare_dep}" \
        --array="${array_range}" \
        --mem="${chr_mem}" \
        --time="${chr_time}" \
        --export=ALL \
        run_methylbert_paper_dmr_pat_chr_slurm.sh
)"
chrom_dep="${chrom_jid%%;*}"

merge_jid="$(
    sbatch \
        --parsable \
        --dependency="afterok:${chrom_dep}" \
        --export=ALL \
        run_methylbert_paper_dmr_pat_merge_slurm.sh
)"

echo "prepare_jid=${prepare_jid}"
echo "chrom_array_jid=${chrom_jid}"
echo "merge_jid=${merge_jid}"
echo "chrom_array=${array_range}"
echo "chrom_mem=${chr_mem}"
echo "chrom_time=${chr_time}"
