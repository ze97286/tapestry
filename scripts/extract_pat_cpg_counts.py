#!/usr/bin/env python3
"""Extract per-CpG methylation counts from a PAT file for DSS.

PAT rows are expected to contain at least:

    chr    startCpG    pattern    count

where `startCpG` is the CpG-index coordinate used by wgbstools and `pattern`
uses consecutive CpG states.  By default this script assumes PATs have already
been converted to bisulfite convention, where `C` is methylated and `T` is
unmethylated.  For raw unflipped TAPS PATs, swap the methylated and
unmethylated characters on the command line.

The output is the DSS input table:

    chr    pos    N    X

where `pos` comes from the supplied CpG index file and `X` is methylated count.
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path
from typing import IO


Counts = dict[tuple[str, int], list[int]]


def open_text(path: str | Path) -> IO[str]:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def add_pat_counts(
    counts: Counts,
    chrom: str,
    start_cpg: int,
    pattern: str,
    count: int,
    methylated_char: str,
    unmethylated_char: str,
) -> None:
    for offset, state in enumerate(pattern):
        if state == methylated_char:
            bucket = counts[(chrom, start_cpg + offset)]
            bucket[0] += count
            bucket[1] += count
        elif state == unmethylated_char:
            counts[(chrom, start_cpg + offset)][0] += count


def read_pat_counts(
    pat_path: Path,
    methylated_char: str,
    unmethylated_char: str,
) -> Counts:
    counts: Counts = defaultdict(lambda: [0, 0])
    rows = 0
    informative_rows = 0

    with open_text(pat_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"PAT row has fewer than 4 columns in {pat_path}: {line[:120]!r}")
            chrom, start_raw, pattern, count_raw = parts[:4]
            start_cpg = int(start_raw)
            count = int(count_raw)
            if count <= 0:
                continue
            rows += 1
            before = len(counts)
            add_pat_counts(
                counts,
                chrom,
                start_cpg,
                pattern,
                count,
                methylated_char,
                unmethylated_char,
            )
            informative_rows += int(len(counts) > before or any(c in pattern for c in (methylated_char, unmethylated_char)))

    print(f"read {rows} PAT rows from {pat_path}; informative rows={informative_rows}; covered CpGs={len(counts)}")
    return counts


def write_dss_counts(
    counts: Counts,
    cpg_path: Path,
    out_path: Path,
    min_coverage: int,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open_text(cpg_path) as cpg_handle, out_path.open("w") as out:
        out.write("chr\tpos\tN\tX\n")
        for line in cpg_handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"CpG index row has fewer than 3 columns in {cpg_path}: {line[:120]!r}")
            chrom = parts[0]
            pos = int(parts[1])
            cpg_index = int(parts[2])
            total, methylated = counts.get((chrom, cpg_index), (0, 0))
            if total < min_coverage:
                continue
            out.write(f"{chrom}\t{pos}\t{total}\t{methylated}\n")
            written += 1

    return written


def one_char(value: str, label: str) -> str:
    if len(value) != 1:
        raise argparse.ArgumentTypeError(f"{label} must be exactly one character")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pat", required=True, help="Input .pat or .pat.gz")
    parser.add_argument("--cpg-file", required=True, help="CpG index TSV/GZ: chr, position, cpg_index")
    parser.add_argument("--output", required=True, help="Output DSS count TSV")
    parser.add_argument("--min-coverage", type=int, default=1)
    parser.add_argument("--methylated-char", default="C", type=lambda v: one_char(v, "--methylated-char"))
    parser.add_argument("--unmethylated-char", default="T", type=lambda v: one_char(v, "--unmethylated-char"))
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty DSS table instead of failing when no CpGs pass filters.",
    )
    args = parser.parse_args()

    counts = read_pat_counts(
        Path(args.pat),
        methylated_char=args.methylated_char,
        unmethylated_char=args.unmethylated_char,
    )
    n_rows = write_dss_counts(
        counts,
        Path(args.cpg_file),
        Path(args.output),
        min_coverage=args.min_coverage,
    )
    print(f"wrote {n_rows} CpGs to {args.output}")
    if n_rows == 0 and not args.allow_empty:
        raise SystemExit(
            "No CpG counts were written. Check PAT/CpG index compatibility, "
            "PAT methylation convention, and --min-coverage."
        )


if __name__ == "__main__":
    main()
