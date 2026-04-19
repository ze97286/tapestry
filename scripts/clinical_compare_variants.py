#!/usr/bin/env python3
"""Diagnose whether the NNLS gate helps or hurts clinical performance.

Reads the combined predictions CSV from predict_cfdna.py (which contains the
raw tapestry output, the NNLS output, and the NNLS-gated tapestry production
value in separate columns per cell type) and reports OAC-vs-ichorCNA Pearson
correlation for each variant — both overall and per timepoint — so we can
tell which variant is strongest on real cfDNA.

Usage:
    python scripts/clinical_compare_variants.py \\
        --predictions runs/run_v0.3/predictions/AB_predictions.csv \\
        --ichorcna-file data/cfDNA_tumour_fraction_ichorCNA.json \\
        --cohort AB
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


TIMEPOINT_TAGS = ["ScrBsl", "Immonly", "C1W3", "C6D22", "PT", "LTSR"]


def parse_timepoint(sample_name: str) -> str:
    for tag in TIMEPOINT_TAGS:
        if f"_{tag}_" in sample_name:
            return tag
    return "Unknown"


def load_ichor(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        rows = []
        for sample, entry in data.items():
            tf = float(entry.get("tumour_fraction", entry.get("tf", 0)))
            rows.append({"sample": sample, "ichor_tf": tf})
        return pd.DataFrame(rows)
    df = pd.read_csv(path)
    sample_col = df.columns[0]
    tf_col = "tumour_fraction" if "tumour_fraction" in df.columns else ("tf" if "tf" in df.columns else df.columns[1])
    return df[[sample_col, tf_col]].rename(columns={sample_col: "sample", tf_col: "ichor_tf"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ichorcna-file", required=True)
    parser.add_argument("--cohort", default="")
    parser.add_argument("--target", default="OAC",
                        help="Cell type to correlate against ichorCNA (default: OAC)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_csv(args.predictions)
    ichor = load_ichor(Path(args.ichorcna_file))

    df["timepoint"] = df["sample"].apply(parse_timepoint)
    df = df.merge(ichor, on="sample", how="inner")
    if df.empty:
        logger.warning("No overlap between predictions and ichorCNA — nothing to compare")
        return

    target = args.target
    variants = [
        (f"{target}",          "gated (prod)"),
        (f"{target}_raw",      "raw tapestry"),
        (f"{target}_nnls",     "nnls alone"),
    ]
    missing = [col for col, _ in variants if col not in df.columns]
    if missing:
        logger.error("Missing columns: %s — was this CSV produced by the updated predict_cfdna.py?", missing)
        return

    timepoints = sorted(df["timepoint"].unique())

    # Per-timepoint table
    print(f"\nCohort {args.cohort}: {target} vs ichorCNA Pearson r, per timepoint")
    print(f"{'timepoint':10s} {'n':>4s} " + "  ".join(f"{label:>15s}" for _, label in variants))
    print("-" * (18 + 17 * len(variants)))
    for tp in timepoints:
        sub = df[df["timepoint"] == tp]
        if len(sub) < 3:
            continue
        row = f"{tp:10s} {len(sub):4d}"
        for col, _ in variants:
            if sub[col].nunique() < 2:
                row += f"  {'const':>15s}"
            else:
                r, _ = pearsonr(sub[col], sub["ichor_tf"])
                row += f"  {r:15.3f}"
        print(row)

    # All-timepoints combined
    print(f"\nCohort {args.cohort}: all timepoints combined (n={len(df)})")
    print(f"{'variant':20s} {'r':>8s} {'p':>10s}")
    print("-" * 40)
    for col, label in variants:
        r, p = pearsonr(df[col], df["ichor_tf"])
        print(f"{label + ' (' + col + ')':20s} {r:8.4f} {p:10.2e}")


if __name__ == "__main__":
    main()
