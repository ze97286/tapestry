#!/usr/bin/env python3
"""Collect hard-control marker sweep deconvolution results into one table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-summary", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda-unknown", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()

    summary = pd.read_csv(args.sweep_summary, sep="\t")
    rows = []
    for _, panel_row in summary.iterrows():
        panel = str(panel_row["panel"])
        tradeoff_path = Path(args.evaluation_dir) / panel / "lambda_tradeoff.csv"
        out = panel_row.to_dict()
        out["lambda_tradeoff_path"] = str(tradeoff_path)
        if not tradeoff_path.exists():
            out["evaluation_status"] = "missing"
            rows.append(out)
            continue

        tradeoff = pd.read_csv(tradeoff_path)
        delta = (tradeoff["lambda_unknown"] - args.lambda_unknown).abs()
        matched = tradeoff[delta <= args.atol]
        if matched.empty:
            matched = tradeoff.iloc[[int(delta.idxmin())]]
            out["evaluation_status"] = "nearest_lambda"
        else:
            out["evaluation_status"] = "ok"
        metrics = matched.iloc[0].to_dict()
        for key, value in metrics.items():
            out[f"eval_{key}"] = value
        rows.append(out)

    result = pd.DataFrame(rows)
    sort_cols = [
        c for c in [
            "eval_control_oac_mean",
            "eval_control_oac_p95",
            "eval_ichor_r_all",
        ]
        if c in result.columns
    ]
    if sort_cols:
        ascending = [True, True, False][: len(sort_cols)]
        result = result.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
