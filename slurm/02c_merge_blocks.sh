#!/bin/bash
#SBATCH --job-name=merge_blocks
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_blocks.out
#SBATCH --error=logs/merge_blocks.err

set -euo pipefail

source slurm/common.sh

SEG_DIR="${OUTPUT_DIR}/segmentation"

echo "=== Merging per-chromosome blocks ==="
mkdir -p "${SEG_DIR}"

N_CHR_BLOCKS=$(find "${SEG_DIR}" -maxdepth 1 -name 'blocks_chr*.bed.gz' | wc -l)
if [ "${N_CHR_BLOCKS}" -eq 0 ]; then
    echo "ERROR: no per-chromosome block files found in ${SEG_DIR}"
    exit 1
fi

# wgbstools homog validates that startCpG is monotonically increasing.  The
# CpG index is global across chromosomes, so sorting by genomic start can break
# this invariant (for example chr10 before chr2 in lexical order).  Sort by the
# global startCpG column instead and keep both uncompressed and gzipped outputs:
# homog reads blocks.bed, downstream scripts often inspect blocks.bed.gz.
zcat "${SEG_DIR}"/blocks_chr*.bed.gz \
    | sort -k4,4n \
    > "${SEG_DIR}/blocks.bed.tmp"

awk -F'\t' '
    NR > 1 && $4 < prev {
        printf("ERROR: startCpG decreased at line %d: %s < %s\n", NR, $4, prev) > "/dev/stderr";
        exit 1
    }
    { prev = $4 }
' "${SEG_DIR}/blocks.bed.tmp"

mv "${SEG_DIR}/blocks.bed.tmp" "${SEG_DIR}/blocks.bed"
gzip -c "${SEG_DIR}/blocks.bed" > "${SEG_DIR}/blocks.bed.gz.tmp"
mv "${SEG_DIR}/blocks.bed.gz.tmp" "${SEG_DIR}/blocks.bed.gz"

N_BLOCKS=$(wc -l < "${SEG_DIR}/blocks.bed")
echo "  Total blocks (>=4 CpGs, autosomes): ${N_BLOCKS}"
echo "  Output: ${SEG_DIR}/blocks.bed"
echo "  Output: ${SEG_DIR}/blocks.bed.gz"
