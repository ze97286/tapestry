#!/usr/bin/env python3
"""Split a DSS per-CpG count table into one file per chromosome."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TextIO


def close_all(handles: dict[str, TextIO]) -> None:
    for handle in handles.values():
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input DSS counts TSV: chr,pos,N,X")
    parser.add_argument("--sample", required=True, help="Sample name for the manifest")
    parser.add_argument("--group", required=True, help="Sample group for the manifest")
    parser.add_argument("--output-dir", required=True, help="Directory for per-chromosome count TSVs")
    parser.add_argument("--manifest", required=True, help="Output manifest TSV")
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise SystemExit(f"missing input count file: {input_path}")
    if manifest_path.is_file() and manifest_path.stat().st_size > 0 and not args.force:
        print(f"Skipping existing chromosome split manifest: {manifest_path}")
        return

    handles: dict[str, TextIO] = {}
    paths: dict[str, Path] = {}
    rows: dict[str, int] = {}

    try:
        with input_path.open("rt") as handle:
            header = handle.readline().rstrip("\n")
            if header.split("\t")[:4] != ["chr", "pos", "N", "X"]:
                raise SystemExit(f"unexpected DSS count header in {input_path}: {header!r}")
            for line in handle:
                if not line.strip():
                    continue
                chrom = line.split("\t", 1)[0]
                if chrom not in handles:
                    chrom_path = output_dir / f"{args.sample}.{chrom}.dss_counts.tsv"
                    if chrom_path.exists() and args.force:
                        chrom_path.unlink()
                    out = chrom_path.open("wt")
                    out.write("chr\tpos\tN\tX\n")
                    handles[chrom] = out
                    paths[chrom] = chrom_path
                    rows[chrom] = 0
                handles[chrom].write(line)
                rows[chrom] += 1
    finally:
        close_all(handles)

    with manifest_path.open("wt") as manifest:
        manifest.write("sample\tgroup\tchrom\tcounts_path\n")
        for chrom in sorted(paths):
            if rows[chrom] > 0:
                manifest.write(f"{args.sample}\t{args.group}\t{chrom}\t{paths[chrom]}\n")

    total = sum(rows.values())
    print(f"split {total} rows from {input_path} across {len(paths)} chromosomes")


if __name__ == "__main__":
    main()
