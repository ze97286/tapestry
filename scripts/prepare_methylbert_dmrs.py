#!/usr/bin/env python3
"""Prepare a DMR table for upstream MethylBERT preprocessing.

The paper workflow fine-tunes on the top tumour-specific DMRs. Upstream
MethylBERT expects a tab-delimited file with at least chr/start/end and a
`ctype` column naming the positive class. This script normalises common BED
or DSS-style inputs into that shape without changing the model logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BED_COLUMNS = [
    "chr",
    "start",
    "end",
    "name",
    "score",
    "strand",
    "areaStat",
    "diff.Methy",
]


def read_regions(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    with path.open() as handle:
        first = handle.readline()

    if first.startswith("#"):
        names = first.lstrip("#").rstrip("\n").split("\t")
        return pd.read_csv(path, sep="\t", comment="#", names=names)

    header = pd.read_csv(path, sep="\t", nrows=0)
    if {"chr", "start", "end"}.issubset(header.columns):
        return pd.read_csv(path, sep="\t")

    n_cols = len(pd.read_csv(path, sep="\t", header=None, nrows=1).columns)
    names = BED_COLUMNS[:n_cols]
    names.extend(f"extra_{i}" for i in range(n_cols - len(names)))
    return pd.read_csv(path, sep="\t", header=None, names=names)


def normalise_chromosome(series: pd.Series) -> pd.Series:
    values = series.astype(str)
    return values.where(values.str.startswith("chr"), "chr" + values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="DMR BED/TSV/parquet")
    parser.add_argument("--output", required=True, help="Prepared MethylBERT DMR TSV")
    parser.add_argument("--top-n", type=int, default=100, help="Number of DMRs to keep")
    parser.add_argument("--target-ctype", default="T", help="Positive class label for DMRs")
    parser.add_argument(
        "--sort-column",
        default=None,
        help="Statistic column to sort by absolute value. Defaults to areaStat, then diff.Methy.",
    )
    args = parser.parse_args()

    df = read_regions(Path(args.input))
    rename = {"#chr": "chr", "chrom": "chr", "chromosome": "chr"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    required = {"chr", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"DMR input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["chr"] = normalise_chromosome(df["chr"])
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(int)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(int)
    df = df[df["end"] > df["start"]].copy()

    sort_column = args.sort_column
    if sort_column is None:
        for candidate in ("areaStat", "diff.Methy", "ttest", "score"):
            if candidate in df.columns:
                sort_column = candidate
                break

    if sort_column:
        if sort_column not in df.columns:
            raise SystemExit(f"sort column not found: {sort_column}")
        df["_sort_abs"] = pd.to_numeric(df[sort_column], errors="coerce").abs()
        df = df.sort_values("_sort_abs", ascending=False, na_position="last")

    if args.top_n > 0:
        df = df.head(args.top_n).copy()

    df["ctype"] = args.target_ctype
    df["dmr_id"] = range(len(df))

    first_cols = ["chr", "start", "end", "ctype", "dmr_id"]
    other_cols = [c for c in df.columns if c not in first_cols and c != "_sort_abs"]
    out = df[first_cols + other_cols]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, sep="\t", index=False)
    print(f"wrote {len(out)} DMRs to {output}")


if __name__ == "__main__":
    main()
