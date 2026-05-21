#!/usr/bin/env python3
"""Inspect BAM/CRAM files for methylation tags used by MethylBERT."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pysam


def read_paths(path: str | None, bam: str | None) -> list[str]:
    paths: list[str] = []
    if path:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(line.split()[0])
    if bam:
        paths.append(bam)
    return paths


def inspect_bam(path: str, max_reads: int, reference: str | None) -> dict[str, object]:
    tag_counts: Counter[str] = Counter()
    n_aligned = 0
    n_with_xm = 0
    n_with_mm = 0
    n_with_ml = 0
    n_with_any_methyl = 0

    kwargs = {"reference_filename": reference} if reference else {}
    with pysam.AlignmentFile(path, "rb", **kwargs) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            n_aligned += 1
            tags = {tag: value for tag, value in read.get_tags()}
            tag_counts.update(tags.keys())
            has_xm = "XM" in tags
            has_mm = "MM" in tags or "Mm" in tags
            has_ml = "ML" in tags or "Ml" in tags
            n_with_xm += int(has_xm)
            n_with_mm += int(has_mm)
            n_with_ml += int(has_ml)
            n_with_any_methyl += int(has_xm or has_mm or has_ml)
            if n_aligned >= max_reads:
                break

    mode = "none"
    if n_with_xm:
        mode = "bismark"
    elif n_with_mm and n_with_ml:
        mode = "dorado"
    elif n_with_mm or n_with_ml:
        mode = "modified-base-partial"

    return {
        "path": path,
        "aligned_reads_checked": n_aligned,
        "reads_with_XM": n_with_xm,
        "reads_with_MM_or_Mm": n_with_mm,
        "reads_with_ML_or_Ml": n_with_ml,
        "reads_with_any_methyl_tag": n_with_any_methyl,
        "inferred_methylcaller": mode,
        "top_tags": ",".join(f"{tag}:{count}" for tag, count in tag_counts.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", help="Single BAM/CRAM path")
    parser.add_argument("--list", help="File containing BAM/CRAM paths")
    parser.add_argument("--reference", help="Reference FASTA, needed for some CRAMs")
    parser.add_argument("--max-reads", type=int, default=1000)
    args = parser.parse_args()

    paths = read_paths(args.list, args.bam)
    if not paths:
        raise SystemExit("provide --bam or --list")

    rows = [inspect_bam(path, args.max_reads, args.reference) for path in paths]
    columns = [
        "path",
        "aligned_reads_checked",
        "reads_with_XM",
        "reads_with_MM_or_Mm",
        "reads_with_ML_or_Ml",
        "reads_with_any_methyl_tag",
        "inferred_methylcaller",
        "top_tags",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row[col]) for col in columns))


if __name__ == "__main__":
    main()
