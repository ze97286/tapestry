#!/usr/bin/env python3
"""Extract per-CpG methylation counts from tagged BAMs for DSS.

Output format is the four-column DSS input table:

    chr    pos    N    X

where `pos` is 1-based, `N` is total informative reads, and `X` is methylated
reads. The methylation parsing follows the same caller choices as upstream
MethylBERT: Bismark `XM` tags or Dorado modified-base tags.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pysam


def cpg_c_position(ref: pysam.FastaFile, chrom: str, ref_pos0: int) -> int | None:
    """Return the 0-based cytosine coordinate for a CpG touched at ref_pos0."""
    if ref_pos0 is None or ref_pos0 < 0:
        return None

    try:
        here = ref.fetch(chrom, ref_pos0, ref_pos0 + 2).upper()
        if here == "CG":
            return ref_pos0
        prev = ref.fetch(chrom, max(0, ref_pos0 - 1), ref_pos0 + 1).upper()
        if prev == "CG":
            return ref_pos0 - 1
    except (KeyError, ValueError):
        return None

    return None


def add_count(
    counts: dict[tuple[str, int], list[int]],
    chrom: str,
    c_pos0: int,
    methylated: bool,
) -> None:
    key = (chrom, c_pos0)
    counts[key][0] += 1
    counts[key][1] += int(methylated)


def extract_bismark(
    bam: pysam.AlignmentFile,
    ref: pysam.FastaFile,
    min_mapq: int,
) -> dict[tuple[str, int], list[int]]:
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])

    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_duplicate or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        try:
            xm = read.get_tag("XM")
        except KeyError:
            continue

        chrom = bam.get_reference_name(read.reference_id)
        for qpos, rpos in read.get_aligned_pairs(matches_only=True):
            if qpos is None or rpos is None or qpos >= len(xm):
                continue
            state = xm[qpos]
            if state not in ("z", "Z"):
                continue
            c_pos0 = cpg_c_position(ref, chrom, rpos)
            if c_pos0 is None:
                continue
            add_count(counts, chrom, c_pos0, state == "Z")

    return counts


def modification_lookup(read: pysam.AlignedSegment) -> dict[int, bool]:
    """Return query-position methylation calls using upstream Dorado semantics."""
    modified = read.modified_bases or {}
    strand = int(read.is_reverse)
    ch_key = ("C", strand, "h")
    cm_key = ("C", strand, "m")

    ch = dict(modified.get(ch_key, []))
    cm = dict(modified.get(cm_key, []))
    positions = sorted(set(ch) | set(cm))
    return {pos: (ch.get(pos, 0) + cm.get(pos, 0)) >= 178 for pos in positions}


def extract_dorado(
    bam: pysam.AlignmentFile,
    ref: pysam.FastaFile,
    min_mapq: int,
) -> dict[tuple[str, int], list[int]]:
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])

    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_duplicate or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        calls = modification_lookup(read)
        if not calls:
            continue

        chrom = bam.get_reference_name(read.reference_id)
        for qpos, rpos in read.get_aligned_pairs(matches_only=True):
            if qpos is None or rpos is None or qpos not in calls:
                continue
            c_pos0 = cpg_c_position(ref, chrom, rpos)
            if c_pos0 is None:
                continue
            add_count(counts, chrom, c_pos0, calls[qpos])

    return counts


def chrom_sort_key(item: tuple[str, int]) -> tuple[int, object, int]:
    chrom, pos = item
    clean = chrom.removeprefix("chr")
    if clean.isdigit():
        return (0, int(clean), pos)
    return (1, clean, pos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methylcaller", choices=["bismark", "dorado"], default="bismark")
    parser.add_argument("--min-mapq", type=int, default=10)
    parser.add_argument("--min-coverage", type=int, default=1)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty counts table instead of failing when no CpGs are extracted.",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pysam.AlignmentFile(args.bam, "rb", reference_filename=args.reference) as bam, pysam.FastaFile(args.reference) as ref:
        if args.methylcaller == "bismark":
            counts = extract_bismark(bam, ref, args.min_mapq)
        else:
            counts = extract_dorado(bam, ref, args.min_mapq)

    with out.open("w") as handle:
        handle.write("chr\tpos\tN\tX\n")
        n_rows = 0
        for chrom, pos0 in sorted(counts.keys(), key=chrom_sort_key):
            total, methylated = counts[(chrom, pos0)]
            if total < args.min_coverage:
                continue
            handle.write(f"{chrom}\t{pos0 + 1}\t{total}\t{methylated}\n")
            n_rows += 1

    print(f"wrote {n_rows} CpGs to {out}")
    if n_rows == 0 and not args.allow_empty:
        raise SystemExit(
            "No CpG counts were extracted. Check that --methylcaller matches the BAM methylation tags "
            "(Bismark XM vs Dorado MM/ML), that the reference contig names match the BAM, and that "
            "--min-mapq is not filtering all reads."
        )


if __name__ == "__main__":
    main()
