#!/usr/bin/env python3
"""Run TapestryModel on cfDNA samples and output per-sample proportions.

Takes filtered+flipped cfDNA PAT files, runs wgbstools homog to get per-marker
U/X/M counts, then runs the trained model to predict cell-type proportions.

Usage:
    python scripts/predict_cfdna.py \
        --cfdna-dir runs/run_002/filtered_pats/cfdna/AB \
        --markers-bed runs/run_002/markers/markers.bed \
        --atlas runs/run_002/markers/markers.tsv \
        --model runs/run_002/models/tapestry/best_model.pt \
        --output runs/run_002/predictions/AB_predictions.csv \
        --cohort AB
"""

import argparse
import gzip
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tapestry.models.deconvolution import TapestryModel

logger = logging.getLogger(__name__)


def build_atlas_coord_index(atlas_path: str) -> list[tuple[str, int]]:
    """Build (chr, start) coordinate list from atlas TSV, preserving row order."""
    coords = []
    with open(atlas_path) as f:
        next(f)
        for line in f:
            parts = line.rstrip().split("\t")
            coords.append((parts[0], int(parts[1])))
    return coords


def extract_marker_values(
    homog_path: str, atlas_coords: list[tuple[str, int]]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract U-fraction and coverage from homog output, aligned to atlas order."""
    try:
        opener = gzip.open if homog_path.endswith(".gz") else open
        homog_data = {}
        with opener(homog_path, "rt") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) >= 8:
                    key = (parts[0], int(parts[1]))
                    u = int(parts[5])
                    m_count = int(parts[7])
                    homog_data[key] = (u, m_count)

        M = len(atlas_coords)
        u_fraction = np.zeros(M, dtype=np.float64)
        coverage = np.zeros(M, dtype=np.float64)

        for i, (chrom, start) in enumerate(atlas_coords):
            if (chrom, start) in homog_data:
                u, m_count = homog_data[(chrom, start)]
                total = u + m_count
                coverage[i] = total
                if total > 0:
                    u_fraction[i] = u / total

        return u_fraction, coverage
    except Exception as e:
        logger.error("Failed to parse %s: %s", homog_path, e)
        return None


def process_cfdna_sample(
    pat_path: str,
    markers_bed: str,
    wgbstools: str,
    atlas_coords: list[tuple[str, int]],
    tmp_dir: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run homog on a single cfDNA PAT file and extract marker values."""
    sample_name = Path(pat_path).stem.replace(".markers.pat", "").replace(".pat", "")

    # Run homog
    homog_out = os.path.join(tmp_dir, f"{sample_name}.uxm.bed.gz")
    cmd = f'{wgbstools} homog -b {markers_bed} -l 4 -o {tmp_dir} {pat_path}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("homog failed for %s: %s", sample_name, result.stderr)
        return None

    # Find the output file (wgbstools names it by input filename)
    pat_basename = Path(pat_path).name.replace(".pat.gz", "")
    expected = os.path.join(tmp_dir, f"{pat_basename}.uxm.bed.gz")
    if os.path.exists(expected):
        homog_out = expected

    return extract_marker_values(homog_out, atlas_coords)


def load_model(model_path: str, atlas_path: str, cell_types: list[str], device: torch.device):
    """Load trained TapestryModel from checkpoint."""
    atlas_df = pd.read_csv(atlas_path, sep="\t")

    target_ids = atlas_df["target"].map(lambda x: cell_types.index(x)).values

    meta_cols = ["chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
                 "target", "name", "direction", "target_signal", "bg_signal", "snr",
                 "target_total", "bg_total"]
    atlas_ct_cols = [c for c in atlas_df.columns if c not in meta_cols]

    atlas_matrix = np.zeros((len(atlas_df), len(cell_types)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        if ct in atlas_ct_cols:
            atlas_matrix[:, i] = atlas_df[ct].values.astype(np.float32)
    atlas_matrix = np.nan_to_num(atlas_matrix, nan=0.5)

    num_markers = len(atlas_df)
    num_cell_types = len(cell_types)

    # Load checkpoint to get config
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = TapestryModel(
        num_markers=num_markers,
        num_cell_types=num_cell_types,
        target_ids=target_ids,
        atlas=atlas_matrix,
        feature_dim=64,
        l1_num_heads=4,
        l1_num_layers=2,
        l2_num_heads=4,
        l2_num_layers=1,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded model from %s (epoch %d)", model_path, checkpoint.get("epoch", -1))

    return model


def main():
    parser = argparse.ArgumentParser(description="Predict cell-type proportions for cfDNA samples.")
    parser.add_argument("--cfdna-dir", required=True, help="Directory with filtered cfDNA PAT files")
    parser.add_argument("--markers-bed", required=True)
    parser.add_argument("--atlas", required=True, help="Path to markers.tsv")
    parser.add_argument("--model", required=True, help="Path to best_model.pt")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--wgbstools", default="wgbstools")
    parser.add_argument("--cohort", default="", help="Cohort name (for logging)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cell types (sorted, matching training)
    atlas_df = pd.read_csv(args.atlas, sep="\t")
    meta_cols = ["chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
                 "target", "name", "direction", "target_signal", "bg_signal", "snr",
                 "target_total", "bg_total"]
    cell_types = sorted([c for c in atlas_df.columns if c not in meta_cols])
    logger.info("Cell types: %s", cell_types)

    # Load model
    model = load_model(args.model, args.atlas, cell_types, device)

    # Build atlas coordinate index
    atlas_coords = build_atlas_coord_index(args.atlas)

    # Find PAT files
    cfdna_dir = Path(args.cfdna_dir)
    pat_files = sorted(cfdna_dir.glob("*.markers.pat.gz"))
    if not pat_files:
        pat_files = sorted(cfdna_dir.glob("*.pat.gz"))
    logger.info("Found %d cfDNA PAT files in %s", len(pat_files), cfdna_dir)

    # Process each sample
    results = []
    with tempfile.TemporaryDirectory(prefix="tapestry_predict_") as tmp_dir:
        for i, pat_path in enumerate(pat_files):
            sample_name = pat_path.stem.replace(".markers.pat", "").replace(".pat", "")
            logger.info("[%d/%d] Processing %s", i + 1, len(pat_files), sample_name)

            result = process_cfdna_sample(
                str(pat_path), args.markers_bed, args.wgbstools,
                atlas_coords, tmp_dir,
            )
            if result is None:
                logger.warning("Skipping %s", sample_name)
                continue

            u_fraction, coverage = result

            # Run model
            fraction_t = torch.tensor(u_fraction, dtype=torch.float32).unsqueeze(0).to(device)
            coverage_t = torch.tensor(coverage, dtype=torch.float32).unsqueeze(0).to(device)
            u_t = fraction_t * coverage_t
            m_t = (1 - fraction_t) * coverage_t

            with torch.no_grad():
                output = model(u_t, m_t, coverage_t, phase="full")

            proportions = output["proportions"].cpu().numpy()[0]
            detection = output["detection"].cpu().numpy()[0]

            row = {"sample": sample_name, "cohort": args.cohort}
            row["mean_coverage"] = float(coverage[coverage > 0].mean()) if (coverage > 0).any() else 0
            row["n_markers_with_coverage"] = int((coverage > 0).sum())
            for j, ct in enumerate(cell_types):
                row[ct] = float(proportions[j])
                row[f"{ct}_detection"] = float(detection[j])
            results.append(row)

            # Clean up homog output
            for f in Path(tmp_dir).glob("*.uxm.bed.gz"):
                f.unlink()

    # Save
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved predictions for %d samples to %s", len(df), args.output)

    # Summary
    logger.info("\nPrediction summary:")
    for ct in cell_types:
        vals = df[ct].values
        logger.info("  %s: mean=%.4f median=%.4f max=%.4f", ct, vals.mean(), np.median(vals), vals.max())


if __name__ == "__main__":
    main()
