#!/usr/bin/env python3
"""Evaluate augmented-NNLS lambda path against controls and ichorCNA.

This script is intentionally table-first.  The clinical plots are useful once
we pick a primary lambda, but lambda tuning needs a compact trade-off table:

  * healthy-control OAC distribution at each lambda
  * ichorCNA Pearson correlation at each lambda
  * sensitivity at a control-calibrated threshold
  * paired Immonly-ScrBsl OAC delta summaries

Example:
    python scripts/evaluate_unknown_lambda_path.py \
        --lambda-path runs/run_v0.3/predictions_unknown_robust/AB_ctrl56_l4_oac_excluded_unknown_unknown_robust_nnls_lambda_path.csv \
        --ichorcna-file data/cfDNA_tumour_fraction_ichorCNA.json \
        --output runs/run_v0.3/evaluation_unknown_robust/AB_ctrl56_l4_oac_excluded_unknown/lambda_tradeoff.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


TIMEPOINT_TAGS = ["ScrBsl", "Immonly", "C1W3", "C6D22", "PT", "LTSR"]


def parse_timepoint(sample_name: str) -> str:
    for tag in TIMEPOINT_TAGS:
        if f"_{tag}_" in sample_name:
            return tag
    return "Unknown"


def parse_patient(sample_name: str) -> str:
    first = str(sample_name).split("_", maxsplit=1)[0]
    return first if "-" in first else str(sample_name)


def load_ichor(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        with open(path) as handle:
            data = json.load(handle)
        rows = []
        for sample, entry in data.items():
            tf = entry.get("tumour_fraction", entry.get("tf"))
            if tf is not None:
                rows.append({"ichor_sample": str(sample), "ichor_tf": float(tf)})
        return pd.DataFrame(rows)

    df = pd.read_csv(path)
    sample_col = df.columns[0]
    if "tumour_fraction" in df.columns:
        tf_col = "tumour_fraction"
    elif "tf" in df.columns:
        tf_col = "tf"
    else:
        tf_col = df.columns[1]
    return df[[sample_col, tf_col]].rename(
        columns={sample_col: "ichor_sample", tf_col: "ichor_tf"}
    )


def attach_ichor(df: pd.DataFrame, ichor: pd.DataFrame) -> pd.DataFrame:
    """Attach ichorCNA by exact or containment-based sample matching."""
    ichor_rows = ichor.to_dict("records")
    matched = []
    for _, row in df.iterrows():
        sample = str(row["sample"])
        hit = None
        for entry in ichor_rows:
            key = str(entry["ichor_sample"])
            if sample == key or sample in key or key in sample:
                hit = entry
                break
        if hit is None:
            continue
        out = row.to_dict()
        out["ichor_sample"] = hit["ichor_sample"]
        out["ichor_tf"] = hit["ichor_tf"]
        matched.append(out)
    return pd.DataFrame(matched)


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    r, _ = pearsonr(x, y)
    return float(r)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def sensitivity(df: pd.DataFrame, threshold: float, mask: pd.Series, target: str) -> float:
    denom = int(mask.sum())
    if denom == 0:
        return np.nan
    return float((df.loc[mask, target] > threshold).mean())


def paired_delta_summary(df_lam: pd.DataFrame, target: str) -> dict[str, float]:
    rows = []
    for patient_id, group in df_lam.groupby("patient_id"):
        scr = group[group["timepoint"] == "ScrBsl"]
        imm = group[group["timepoint"] == "Immonly"]
        if scr.empty or imm.empty:
            continue
        rows.append(float(imm.iloc[0][target] - scr.iloc[0][target]))
    if not rows:
        return {
            "paired_delta_n": 0,
            "paired_delta_mean": np.nan,
            "paired_delta_median": np.nan,
            "paired_delta_pos_frac": np.nan,
        }
    values = np.asarray(rows, dtype=float)
    return {
        "paired_delta_n": int(len(values)),
        "paired_delta_mean": float(np.mean(values)),
        "paired_delta_median": float(np.median(values)),
        "paired_delta_pos_frac": float(np.mean(values > 0)),
    }


def summarise_lambda_path(
    lambda_path: pd.DataFrame,
    ichor: pd.DataFrame | None,
    target: str,
    control_quantile: float,
    low_ichor_min: float,
    low_ichor_max: float,
) -> pd.DataFrame:
    df = lambda_path.copy()
    df["sample"] = df["sample"].astype(str)
    df["is_control"] = parse_bool_series(df["is_control"])
    df["patient_id"] = df["sample"].apply(parse_patient)
    df["timepoint"] = df["sample"].apply(parse_timepoint)

    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found in lambda path")

    ichor_df = attach_ichor(df, ichor) if ichor is not None else pd.DataFrame()
    rows = []
    for lam, sub in df.groupby("lambda_unknown", sort=True):
        controls = sub[sub["is_control"]]
        noncontrols = sub[~sub["is_control"]]

        if controls.empty:
            threshold = np.nan
        else:
            threshold = float(controls[target].quantile(control_quantile))

        row = {
            "lambda_unknown": float(lam),
            "n_samples": int(len(sub)),
            "n_controls": int(len(controls)),
            "control_threshold_q": float(control_quantile),
            "control_oac_threshold": threshold,
            "control_oac_mean": float(controls[target].mean()) if len(controls) else np.nan,
            "control_oac_median": float(controls[target].median()) if len(controls) else np.nan,
            "control_oac_p95": float(controls[target].quantile(0.95)) if len(controls) else np.nan,
            "control_oac_max": float(controls[target].max()) if len(controls) else np.nan,
            "control_frac_gt_1pct": float((controls[target] > 0.01).mean()) if len(controls) else np.nan,
            "control_frac_gt_2pct": float((controls[target] > 0.02).mean()) if len(controls) else np.nan,
            "control_frac_gt_5pct": float((controls[target] > 0.05).mean()) if len(controls) else np.nan,
            "noncontrol_oac_mean": float(noncontrols[target].mean()) if len(noncontrols) else np.nan,
            "noncontrol_oac_median": float(noncontrols[target].median()) if len(noncontrols) else np.nan,
            "noncontrol_oac_max": float(noncontrols[target].max()) if len(noncontrols) else np.nan,
            "control_unknown_mag_mean": (
                float(controls["unknown_mag"].mean())
                if len(controls) and "unknown_mag" in controls
                else np.nan
            ),
            "noncontrol_unknown_mag_mean": (
                float(noncontrols["unknown_mag"].mean())
                if len(noncontrols) and "unknown_mag" in noncontrols
                else np.nan
            ),
        }
        row.update(paired_delta_summary(sub, target))

        if not ichor_df.empty:
            matched = ichor_df[ichor_df["lambda_unknown"] == lam].copy()
            low_mask = (
                (matched["ichor_tf"] >= low_ichor_min)
                & (matched["ichor_tf"] <= low_ichor_max)
            )
            row.update({
                "ichor_matched_n": int(len(matched)),
                "ichor_r_all": safe_pearson(matched[target], matched["ichor_tf"]),
                "ichor_r_scrbsl": safe_pearson(
                    matched.loc[matched["timepoint"] == "ScrBsl", target],
                    matched.loc[matched["timepoint"] == "ScrBsl", "ichor_tf"],
                ),
                "ichor_r_immonly": safe_pearson(
                    matched.loc[matched["timepoint"] == "Immonly", target],
                    matched.loc[matched["timepoint"] == "Immonly", "ichor_tf"],
                ),
                "sens_ichor_ge_1pct_at_control_q": sensitivity(
                    matched, threshold, matched["ichor_tf"] >= 0.01, target
                ),
                "sens_low_ichor_at_control_q": sensitivity(
                    matched, threshold, low_mask, target
                ),
                "sens_ichor_ge_5pct_at_control_q": sensitivity(
                    matched, threshold, matched["ichor_tf"] >= 0.05, target
                ),
            })
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-path", required=True)
    parser.add_argument("--ichorcna-file", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="OAC")
    parser.add_argument("--control-quantile", type=float, default=0.95)
    parser.add_argument("--low-ichor-min", type=float, default=0.01)
    parser.add_argument("--low-ichor-max", type=float, default=0.05)
    args = parser.parse_args()

    lambda_path = pd.read_csv(args.lambda_path)
    ichor = load_ichor(Path(args.ichorcna_file)) if args.ichorcna_file else None
    summary = summarise_lambda_path(
        lambda_path=lambda_path,
        ichor=ichor,
        target=args.target,
        control_quantile=args.control_quantile,
        low_ichor_min=args.low_ichor_min,
        low_ichor_max=args.low_ichor_max,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    display_cols = [
        "lambda_unknown",
        "control_oac_mean",
        "control_oac_p95",
        "control_oac_max",
        "noncontrol_oac_mean",
        "ichor_r_all",
        "ichor_r_scrbsl",
        "ichor_r_immonly",
        "sens_ichor_ge_1pct_at_control_q",
        "sens_low_ichor_at_control_q",
        "sens_ichor_ge_5pct_at_control_q",
        "paired_delta_median",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
