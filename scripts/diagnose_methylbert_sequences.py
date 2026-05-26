#!/usr/bin/env python3
"""Diagnose generated MethylBERT fine-tuning sequence tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def quantiles(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {}
    qs = values.quantile([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    return {f"q{int(q * 100):02d}": float(v) for q, v in qs.items()}


def read_table(path: Path, split: str, max_rows: int) -> pd.DataFrame:
    kwargs = {"sep": "\t"}
    if max_rows > 0:
        kwargs["nrows"] = max_rows
    df = pd.read_csv(path, **kwargs)
    required = {"dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    df["split"] = split
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    methyl = df["methyl_seq"].astype(str)
    df["n_tokens"] = df["dna_seq"].astype(str).str.count(" ") + 1
    df["n_methyl_chars"] = methyl.str.len()
    df["n_methylated"] = methyl.str.count("1")
    df["n_unmethylated"] = methyl.str.count("0")
    df["n_unknown_methyl"] = methyl.str.count("2")
    df["n_informative"] = df["n_methylated"] + df["n_unmethylated"]
    df["frac_informative"] = df["n_informative"] / df["n_methyl_chars"].replace(0, np.nan)
    df["frac_methylated"] = df["n_methylated"] / df["n_informative"].replace(0, np.nan)
    df["matches_dmr_ctype"] = df["ctype"].astype(str) == df["dmr_ctype"].astype(str)
    return df


def write_group_summary(df: pd.DataFrame, out_path: Path, group_cols: list[str]) -> None:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "n_reads": int(len(group)),
                "n_samples": int(group["filename"].nunique()) if "filename" in group else 0,
                "n_dmrs": int(group["dmr_label"].nunique()),
                "mean_tokens": float(group["n_tokens"].mean()),
                "median_tokens": float(group["n_tokens"].median()),
                "mean_informative": float(group["n_informative"].mean()),
                "median_informative": float(group["n_informative"].median()),
                "mean_frac_informative": float(group["frac_informative"].mean()),
                "median_frac_informative": float(group["frac_informative"].median()),
                "mean_frac_methylated": float(group["frac_methylated"].mean()),
                "median_frac_methylated": float(group["frac_methylated"].median()),
            }
        )
        rows.append(row)
    pd.DataFrame(rows).sort_values(group_cols).to_csv(out_path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="train_seq.csv")
    parser.add_argument("--test", required=True, help="test_seq.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="debug limit per split; 0 means all rows")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = read_table(Path(args.train), "train", args.max_rows)
    test = read_table(Path(args.test), "test", args.max_rows)
    df = add_features(pd.concat([train, test], ignore_index=True))

    df[
        [
            "split",
            "name",
            "filename",
            "ctype",
            "dmr_ctype",
            "dmr_label",
            "n_tokens",
            "n_methyl_chars",
            "n_informative",
            "frac_informative",
            "frac_methylated",
            "matches_dmr_ctype",
        ]
    ].to_csv(output_dir / "sequence_diagnostics_per_read.tsv", sep="\t", index=False)

    write_group_summary(df, output_dir / "sequence_diagnostics_by_split_label.tsv", ["split", "ctype"])
    write_group_summary(df, output_dir / "sequence_diagnostics_by_label.tsv", ["ctype"])
    write_group_summary(df, output_dir / "sequence_diagnostics_by_dmr_ctype.tsv", ["dmr_ctype", "ctype"])

    dmr_summary = (
        df.groupby(["dmr_label", "dmr_ctype", "ctype"], dropna=False)
        .agg(
            n_reads=("name", "size"),
            n_samples=("filename", "nunique"),
            mean_informative=("n_informative", "mean"),
            median_informative=("n_informative", "median"),
            mean_frac_methylated=("frac_methylated", "mean"),
            median_frac_methylated=("frac_methylated", "median"),
        )
        .reset_index()
        .sort_values(["dmr_label", "ctype"])
    )
    dmr_summary.to_csv(output_dir / "sequence_diagnostics_by_dmr.tsv", sep="\t", index=False)

    overview = {
        "n_reads": int(len(df)),
        "splits": df["split"].value_counts().to_dict(),
        "labels": df["ctype"].value_counts().to_dict(),
        "n_samples": int(df["filename"].nunique()) if "filename" in df else 0,
        "n_dmrs": int(df["dmr_label"].nunique()),
        "n_tokens": quantiles(df["n_tokens"]),
        "n_informative": quantiles(df["n_informative"]),
        "frac_informative": quantiles(df["frac_informative"].dropna()),
        "frac_methylated": quantiles(df["frac_methylated"].dropna()),
        "reads_with_one_or_fewer_informative_cpgs": int((df["n_informative"] <= 1).sum()),
        "reads_with_two_or_fewer_informative_cpgs": int((df["n_informative"] <= 2).sum()),
        "reads_with_five_or_more_informative_cpgs": int((df["n_informative"] >= 5).sum()),
    }
    with (output_dir / "sequence_diagnostics_summary.json").open("w") as handle:
        json.dump(overview, handle, indent=2)

    print(json.dumps(overview, indent=2))
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
