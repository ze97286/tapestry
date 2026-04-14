#!/usr/bin/env python3
"""Analyse healthy control signal at atlas marker regions.

For each marker in the atlas, computes U-fraction and coverage from
healthy control homog files. Outputs a CSV showing which markers have
signal in controls, to inform atlas refinement.

Usage:
    python scripts/analyse_control_markers.py \
        --atlas runs/run_v0.1/markers/markers.tsv \
        --control-homog-dir runs/run_v0.1/homog_controls \
        --output runs/run_v0.1/markers/control_analysis.csv
"""

import argparse
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_homog_at_markers(homog_path: str, marker_coords: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Load a homog file and extract U-fraction + coverage at marker coordinates."""
    homog_data = {}
    opener = gzip.open if str(homog_path).endswith(".gz") else open
    with opener(homog_path, "rt") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) >= 8:
                key = (parts[0], int(parts[1]))
                u = int(parts[5])
                m = int(parts[7])
                homog_data[key] = (u, m)

    M = len(marker_coords)
    u_frac = np.zeros(M)
    coverage = np.zeros(M)
    for i, (chrom, start) in enumerate(marker_coords):
        if (chrom, start) in homog_data:
            u, m = homog_data[(chrom, start)]
            total = u + m
            coverage[i] = total
            if total > 0:
                u_frac[i] = u / total

    return u_frac, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", required=True, help="Path to markers.tsv")
    parser.add_argument("--control-homog-dir", required=True, help="Directory with control homog files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load atlas
    atlas = pd.read_csv(args.atlas, sep="\t")
    marker_coords = list(zip(atlas["chr"], atlas["start"]))
    logger.info("Atlas: %d markers", len(atlas))

    # Find control homog files
    homog_dir = Path(args.control_homog_dir)
    homog_files = sorted(homog_dir.glob("*.uxm.bed.gz"))
    if not homog_files:
        homog_files = sorted(homog_dir.glob("*.uxm.bed"))
    logger.info("Found %d control homog files", len(homog_files))

    # Load all controls
    control_ufracs = {}
    control_coverages = {}
    for hf in homog_files:
        name = hf.stem.replace(".uxm.bed", "").replace(".uxm", "")
        uf, cov = load_homog_at_markers(str(hf), marker_coords)
        control_ufracs[name] = uf
        control_coverages[name] = cov
        logger.info("  %s: mean_cov=%.1f, markers_with_cov=%d, mean_ufrac=%.3f",
                     name, cov[cov > 0].mean() if (cov > 0).any() else 0,
                     (cov > 0).sum(), uf[cov > 0].mean() if (cov > 0).any() else 0)

    # Build analysis DataFrame
    result = atlas[["chr", "start", "end", "target", "snr", "target_signal", "bg_signal"]].copy()

    control_names = sorted(control_ufracs.keys())
    for name in control_names:
        result[f"{name}_ufrac"] = control_ufracs[name]
        result[f"{name}_cov"] = control_coverages[name]

    # Summary columns
    ufrac_cols = [f"{n}_ufrac" for n in control_names]
    cov_cols = [f"{n}_cov" for n in control_names]

    result["ctrl_max_ufrac"] = result[ufrac_cols].max(axis=1)
    result["ctrl_mean_ufrac"] = result[ufrac_cols].mean(axis=1)
    result["ctrl_mean_cov"] = result[cov_cols].mean(axis=1)
    result["ctrl_n_with_signal"] = (result[ufrac_cols] > 0.05).sum(axis=1)

    # Save
    result.to_csv(args.output, index=False)
    logger.info("Saved analysis to %s", args.output)

    # Summary by cell type
    logger.info("\n=== Control signal by cell type ===")
    for ct in sorted(atlas["target"].unique()):
        ct_mask = atlas["target"] == ct
        ct_result = result[ct_mask]
        n_markers = len(ct_result)
        n_with_signal = (ct_result["ctrl_max_ufrac"] > 0.1).sum()
        n_high_signal = (ct_result["ctrl_max_ufrac"] > 0.2).sum()
        mean_ctrl = ct_result["ctrl_mean_ufrac"].mean()
        mean_snr = ct_result["snr"].mean()

        logger.info("  %s: %d markers, %d (%.0f%%) with ctrl U-frac>0.1, %d (%.0f%%) >0.2, "
                     "mean_ctrl=%.3f, mean_snr=%.1f",
                     ct, n_markers, n_with_signal, 100 * n_with_signal / n_markers,
                     n_high_signal, 100 * n_high_signal / n_markers, mean_ctrl, mean_snr)

    # How many OAC markers would survive different thresholds?
    logger.info("\n=== OAC markers surviving control filter ===")
    oac_mask = atlas["target"] == "OAC"
    oac_result = result[oac_mask]
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        surviving = (oac_result["ctrl_max_ufrac"] <= thresh).sum()
        logger.info("  max_ctrl_ufrac <= %.2f: %d / %d OAC markers survive",
                     thresh, surviving, len(oac_result))


if __name__ == "__main__":
    main()
