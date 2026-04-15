#!/bin/bash
# Convert a single cfDNA BAM to PAT with fragment-length-dependent clipping.
#
# Designed to run as a SLURM array job — one task per BAM file.
#
# Pipeline per BAM:
#   1. vclip with rules → clipped BAM
#   2. samtools sort (clipping changes coordinates)
#   3. wgbstools bam2pat
#   4. Flip TAPS → bisulfite (sed y/TC/CT/)
#   5. bgzip + tabix index
#
# Usage (via SLURM array):
#   See slurm/10_bam2pat_cfdna.sh

set -euo pipefail

BAM=$1
OUTDIR=$2
WGBSTOOLS=${3:-/users/ludwig/uii408/sharedscratch/wgbs_tools/wgbstools}
VCLIP=${4:-/users/ludwig/uii408/sharedscratch/vclip/target/release/vclip}
RULES=${5:-/users/ludwig/uii408/sharedscratch/tapestry/data/vclip_rules.tsv}
THREADS=${6:-4}

module load GCC/12.3.0 2>/dev/null || true

name=$(basename "$BAM" .bam)
pat_out="${OUTDIR}/${name}.pat.gz"

if [ -f "${pat_out}" ] && [ -f "${pat_out}.tbi" -o -f "${pat_out}.csi" ]; then
    echo "Skip (exists): ${name}"
    exit 0
fi

echo "=== ${name} ==="
tmpdir=$(mktemp -d -p "${OUTDIR}" "tmp_${name}_XXXX")

# Step 1: vclip
echo "  [1/5] vclip..."
${VCLIP} "$BAM" --config "${RULES}" --output "${tmpdir}/${name}.clipped.bam" --threads ${THREADS} --min-length 20

# Step 2: sort
echo "  [2/5] sort..."
samtools sort -@ ${THREADS} -o "${tmpdir}/${name}.sorted.bam" "${tmpdir}/${name}.clipped.bam"
samtools index "${tmpdir}/${name}.sorted.bam"
rm -f "${tmpdir}/${name}.clipped.bam"

# Step 3: bam2pat
echo "  [3/5] bam2pat..."
${WGBSTOOLS} bam2pat \
    --no_beta -F 3848 \
    -@ ${THREADS} \
    -o "${tmpdir}" --genome GENOME_NAME "${tmpdir}/${name}.sorted.bam"

# Step 4: flip TAPS → bisulfite
raw_pat=$(ls "${tmpdir}"/*.pat.gz 2>/dev/null | head -1)

if [ -n "${raw_pat}" ] && [ -f "${raw_pat}" ]; then
    echo "  [4/5] flip TAPS→bisulfite..."
    zcat "${raw_pat}" | sed 'y/TC/CT/' | bgzip -c > "${pat_out}"
    echo "  [5/5] index..."
    tabix -f -s 1 -b 2 -e 2 -C "${pat_out}"
else
    echo "  WARNING: bam2pat did not produce output for ${name}"
fi

rm -rf "${tmpdir}"
echo "  Done: ${pat_out}"
