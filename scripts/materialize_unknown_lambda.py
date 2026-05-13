#!/usr/bin/env python3
"""Materialise one augmented-NNLS lambda as a standard prediction CSV.

The lambda-path CSV has one row per (sample, lambda) and contains only the
augmented cell-type columns for that lambda.  The clinical evaluator expects
one row per sample and, for comparison plots, optional `{cell_type}_nnls`
columns.  This helper selects one lambda from the path and merges the NNLS
columns from the primary prediction CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


META_COLS = {
    "sample",
    "cohort",
    "is_control",
    "unknown_basis_mode",
    "control_crossfit_fold",
    "unknown_n_components",
    "unknown_fit_excluded_cell_types",
    "lambda_unknown",
    "unknown_mag",
    "residual_norm",
}


def infer_cell_types(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-path", required=True)
    parser.add_argument("--predictions", required=True,
                        help="Primary predictions CSV containing `{ct}_nnls` columns.")
    parser.add_argument("--lambda-unknown", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    path_df = pd.read_csv(args.lambda_path)
    pred_df = pd.read_csv(args.predictions)

    selected = path_df[
        (path_df["lambda_unknown"] - args.lambda_unknown).abs() <= args.atol
    ].copy()
    if selected.empty:
        available = ", ".join(f"{v:g}" for v in sorted(path_df["lambda_unknown"].unique()))
        raise SystemExit(
            f"No rows matched --lambda-unknown {args.lambda_unknown:g}. "
            f"Available lambdas: {available}"
        )

    cell_types = infer_cell_types(selected)
    nnls_cols = [f"{ct}_nnls" for ct in cell_types if f"{ct}_nnls" in pred_df.columns]
    merge_cols = [
        c for c in [
            "sample", "mean_coverage", "n_markers_with_coverage", *nnls_cols
        ]
        if c in pred_df.columns
    ]
    selected = selected.merge(pred_df[merge_cols], on="sample", how="left")

    selected = selected.rename(columns={
        "lambda_unknown": "unknown_lambda",
        "residual_norm": "unknown_residual_norm",
    })
    selected["unknown_lambda"] = float(args.lambda_unknown)

    front = [
        c for c in [
            "sample", "cohort", "mean_coverage", "n_markers_with_coverage",
            "is_control", "unknown_mag", "unknown_lambda",
            "unknown_residual_norm", "unknown_basis_mode",
            "control_crossfit_fold", "unknown_n_components",
            "unknown_fit_excluded_cell_types",
        ]
        if c in selected.columns
    ]
    ordered = front + [
        c for ct in cell_types for c in (ct, f"{ct}_nnls") if c in selected.columns
    ]
    remainder = [c for c in selected.columns if c not in ordered]
    selected = selected[ordered + remainder]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(f"Saved {len(selected)} rows at lambda={args.lambda_unknown:g} to {output}")


if __name__ == "__main__":
    main()
