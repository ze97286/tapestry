#!/usr/bin/env python3
"""Collect per-sample upstream MethylBERT deconvolution outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_sample_result(sample_dir: Path) -> dict[str, object] | None:
    deconv = sample_dir / "deconvolution.csv"
    if not deconv.exists():
        return None

    df = pd.read_csv(deconv, sep="\t")
    row: dict[str, object] = {"sample": sample_dir.name, "deconvolution_path": str(deconv)}
    for _, rec in df.iterrows():
        row[str(rec["cell_type"])] = float(rec["pred"])

    res = sample_dir / "res.csv"
    if res.exists():
        row["n_reads_classified"] = sum(1 for _ in res.open()) - 1
        row["read_classification_path"] = str(res)

    fi = sample_dir / "FI.csv"
    if fi.exists():
        row["fisher_info_path"] = str(fi)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deconvolution-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.deconvolution_dir)
    rows = [read_sample_result(path) for path in sorted(root.iterdir()) if path.is_dir()]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise SystemExit(f"no deconvolution.csv files found below {root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} sample summaries to {output}")


if __name__ == "__main__":
    main()
