#!/usr/bin/env python3
"""Run NNLS on cfDNA samples for comparison with TapestryModel.

Reuses the same homog + marker extraction pipeline as predict_cfdna.py
but applies coverage-weighted NNLS instead of the neural network.

Usage:
    python scripts/predict_cfdna_nnls.py \
        --cfdna-dir runs/run_002/filtered_pats/cfdna/AB \
        --markers-bed runs/run_002/markers/markers.bed \
        --atlas runs/run_002/markers/markers.tsv \
        --output runs/run_002/predictions/AB_nnls_predictions.csv \
        --cohort AB
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from scripts.predict_cfdna import (
    build_atlas_coord_index, process_cfdna_sample,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NNLS predictions on cfDNA.")
    parser.add_argument("--cfdna-dir", required=True)
    parser.add_argument("--markers-bed", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wgbstools", default="wgbstools")
    parser.add_argument("--cohort", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load atlas
    atlas_df = pd.read_csv(args.atlas, sep="\t")
    meta_cols = ["chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
                 "target", "direction", "target_signal", "bg_signal", "snr",
                 "target_total", "bg_total"]
    cell_types = sorted([c for c in atlas_df.columns if c not in meta_cols])

    # Reference profiles (C, M)
    ref = np.zeros((len(cell_types), len(atlas_df)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        ref[i] = atlas_df[ct].values.astype(np.float32)
    ref = np.nan_to_num(ref, nan=0.5)

    atlas_coords = build_atlas_coord_index(args.atlas)

    # Find PAT files
    cfdna_dir = Path(args.cfdna_dir)
    pat_files = sorted(cfdna_dir.glob("*.markers.pat.gz"))
    if not pat_files:
        pat_files = sorted(cfdna_dir.glob("*.pat.gz"))
    logger.info("Found %d cfDNA PAT files", len(pat_files))

    # Process
    results = []
    with tempfile.TemporaryDirectory(prefix="tapestry_nnls_") as tmp_dir:
        for i, pat_path in enumerate(pat_files):
            sample_name = pat_path.stem.replace(".markers.pat", "").replace(".pat", "")
            logger.info("[%d/%d] %s", i + 1, len(pat_files), sample_name)

            result = process_cfdna_sample(
                str(pat_path), args.markers_bed, args.wgbstools,
                atlas_coords, tmp_dir,
            )
            if result is None:
                continue

            u_fraction, coverage = result

            # NNLS
            A = ref.T
            w = coverage
            Aw = A * w[:, np.newaxis]
            bw = u_fraction * w
            Aw[w == 0] = 0
            bw[w == 0] = 0
            x, _ = nnls(Aw, bw)
            if x.sum() > 0:
                x /= x.sum()

            row = {"sample": sample_name, "cohort": args.cohort}
            row["mean_coverage"] = float(coverage[coverage > 0].mean()) if (coverage > 0).any() else 0
            row["n_markers_with_coverage"] = int((coverage > 0).sum())
            for j, ct in enumerate(cell_types):
                row[ct] = float(x[j])
            results.append(row)

            for f in Path(tmp_dir).glob("*.uxm.bed.gz"):
                f.unlink()

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved NNLS predictions for %d samples to %s", len(df), args.output)


if __name__ == "__main__":
    main()
