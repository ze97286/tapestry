#!/bin/bash
# Convert cfDNA BAMs to PAT files with fragment-length-dependent clipping.
#
# Uses vclip (Ben's tool) to apply graduated soft-clipping rules based on
# insert size, removing artefactual hypomethylation from:
#   - Very short/long fragments (0-100bp, 211-290bp, 361+bp): fully clipped
#   - Medium fragments (101-210bp): graduated 5'/3' clipping
#   - Long R2 reads (291-360bp): R2 fully clipped, R1 partially clipped
#
# Pipeline per BAM:
#   1. vclip with rules → clipped BAM
#   2. samtools sort (clipping changes coordinates)
#   3. wgbstools bam2pat
#   4. Flip TAPS → bisulfite (sed y/TC/CT/)
#   5. bgzip + tabix index
#
# Usage:
#   bash scripts/bam2pat_cfdna.sh [AB|CD]

set -euo pipefail

COHORT=${1:-AB}
WGBSTOOLS="/users/ludwig/uii408/sharedscratch/wgbs_tools/wgbstools"
VCLIP="/users/ludwig/uii408/sharedscratch/vclip/target/release/vclip"
RULES="/users/ludwig/uii408/sharedscratch/tapestry/data/vclip_rules.tsv"
OUTDIR="/users/ludwig/uii408/sharedscratch/tapestry/data/cfdna_pats/${COHORT}"
TMPDIR_BASE="/users/ludwig/uii408/sharedscratch/tapestry/data/cfdna_pats/tmp"
THREADS=8

# BAM locations
if [ "${COHORT}" = "AB" ]; then
    BAM_GLOB="/well/ludwig/processed/Lu_lab/OAC_immuno_Trial/TAPS_cfDNA/Results/1.6.1/Alignments/*_md.bam"
elif [ "${COHORT}" = "CD" ]; then
    BAM_GLOB="/well/ludwig/processed/Lu_lab/OAC_immuno_Trial_CD/TAPS_cfDNA/Results/1.6.1/Alignments/*_md.bam"
else
    echo "ERROR: Unknown cohort ${COHORT}. Use AB or CD."
    exit 1
fi

module load GCC/12.3.0 2>/dev/null || true

mkdir -p "${OUTDIR}" "${TMPDIR_BASE}"

echo "============================================"
echo "cfDNA BAM → PAT (vclip + bam2pat)"
echo "Cohort: ${COHORT}"
echo "Output: ${OUTDIR}"
echo "============================================"

for bam in ${BAM_GLOB}; do
    name=$(basename "$bam" .bam)
    pat_out="${OUTDIR}/${name}.pat.gz"

    if [ -f "${pat_out}" ] && [ -f "${pat_out}.tbi" -o -f "${pat_out}.csi" ]; then
        echo "Skip (exists): ${name}"
        continue
    fi

    echo ""
    echo "=== ${name} ==="
    tmpdir="${TMPDIR_BASE}/${name}"
    mkdir -p "${tmpdir}"

    # Step 1: vclip
    clipped_bam="${tmpdir}/${name}.clipped.bam"
    echo "  [1/5] vclip..."
    ${VCLIP} "$bam" --config "${RULES}" --output "${clipped_bam}" --threads ${THREADS} --min-length 20

    # Step 2: sort (clipping changes coordinates)
    sorted_bam="${tmpdir}/${name}.sorted.bam"
    echo "  [2/5] sort..."
    samtools sort -@ ${THREADS} -o "${sorted_bam}" "${clipped_bam}"
    samtools index "${sorted_bam}"
    rm -f "${clipped_bam}"

    # Step 3: bam2pat
    echo "  [3/5] bam2pat..."
    ${WGBSTOOLS} bam2pat \
        --no_beta -F 3848 \
        -@ ${THREADS} \
        -o "${tmpdir}" --genome hg38 "${sorted_bam}"

    # Step 4: flip TAPS → bisulfite
    raw_pat="${tmpdir}/${name}.sorted.pat.gz"
    if [ ! -f "${raw_pat}" ]; then
        # bam2pat may use original filename
        raw_pat=$(ls "${tmpdir}"/*.pat.gz 2>/dev/null | head -1)
    fi

    if [ -n "${raw_pat}" ] && [ -f "${raw_pat}" ]; then
        echo "  [4/5] flip TAPS→bisulfite..."
        zcat "${raw_pat}" | sed 'y/TC/CT/' | bgzip -c > "${pat_out}"

        # Step 5: index
        echo "  [5/5] index..."
        tabix -f -s 1 -b 2 -e 2 -C "${pat_out}"
    else
        echo "  WARNING: bam2pat did not produce output for ${name}"
    fi

    # Cleanup
    rm -rf "${tmpdir}"

    echo "  Done: ${pat_out}"
done

rm -rf "${TMPDIR_BASE}"

echo ""
echo "============================================"
echo "Complete. PAT files in: ${OUTDIR}"
ls "${OUTDIR}"/*.pat.gz 2>/dev/null | wc -l
echo " PAT files generated"
echo "============================================"
