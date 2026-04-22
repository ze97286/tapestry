#!/usr/bin/env python3
"""Run TapestryModel + NNLS-gated ensemble on cfDNA samples.

Takes filtered+flipped cfDNA PAT files, runs ``wgbstools homog`` to get
per-marker U/X/M counts, then emits three proportion estimates per sample:

* raw tapestry output
* coverage-weighted NNLS output
* NNLS-gated tapestry — tapestry output zeroed where NNLS assigned exactly 0
  and renormalised. This is the production prediction (best on eval metrics).

The output CSV has one row per sample with columns:

    sample, cohort, mean_coverage, n_markers_with_coverage,
    {ct}              — NNLS-gated tapestry (production)
    {ct}_raw          — raw tapestry
    {ct}_nnls         — NNLS
    {ct}_detection    — tapestry auxiliary detection head output

Applies the same atlas NaN-row filter that training uses so the checkpoint,
atlas matrix, and per-sample marker vectors are all column-aligned.

Usage:
    python scripts/predict_cfdna.py \\
        --cfdna-dir runs/run_v0.3/filtered_pats/cfdna/AB \\
        --markers-bed runs/run_v0.3/markers/markers.bed \\
        --atlas runs/run_v0.3/markers/markers.tsv \\
        --model runs/run_v0.3/models/tapestry/best_model.pt \\
        --output runs/run_v0.3/predictions/AB_predictions.csv \\
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
from tapestry.benchmark.nnls import run_weighted_nnls
from tapestry.benchmark.binomial_mle import run_binomial_mle

logger = logging.getLogger(__name__)


META_COLS = [
    "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
    "target", "name", "direction",
    "target_signal", "bg_signal", "snr", "target_total", "bg_total",
]


def load_atlas(atlas_path: str, cell_types: list[str]):
    """Load atlas, drop rows with any NaN in cell-type columns.

    Returns
    -------
    target_ids : (M',) int array into ``cell_types``.
    atlas_matrix : (M', C) reference U-fractions.
    valid_indices : (M',) int array into the original atlas rows.
    atlas_coords : list of (chr, start) tuples aligned to ``atlas_matrix`` rows.
    total_rows : int — row count before filtering (for logging).
    """
    atlas_df = pd.read_csv(atlas_path, sep="\t")
    atlas_ct_cols = [c for c in atlas_df.columns if c not in META_COLS]

    atlas_matrix_full = np.zeros((len(atlas_df), len(cell_types)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        if ct in atlas_ct_cols:
            atlas_matrix_full[:, i] = atlas_df[ct].values.astype(np.float32)

    any_nan = atlas_df[atlas_ct_cols].isna().any(axis=1).values
    valid_indices = np.where(~any_nan)[0]
    atlas_matrix = atlas_matrix_full[valid_indices]
    target_ids = atlas_df["target"].iloc[valid_indices].map(
        lambda x: cell_types.index(x)
    ).values

    atlas_coords = [
        (str(atlas_df.iloc[i]["chr"]), int(atlas_df.iloc[i]["start"]))
        for i in valid_indices
    ]
    return target_ids, atlas_matrix, valid_indices, atlas_coords, len(atlas_df)


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

    homog_out = os.path.join(tmp_dir, f"{sample_name}.uxm.bed.gz")
    cmd = f"{wgbstools} homog -b {markers_bed} -l 4 -o {tmp_dir} {pat_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("homog failed for %s: %s", sample_name, result.stderr)
        return None

    pat_basename = Path(pat_path).name.replace(".pat.gz", "")
    expected = os.path.join(tmp_dir, f"{pat_basename}.uxm.bed.gz")
    if os.path.exists(expected):
        homog_out = expected

    return extract_marker_values(homog_out, atlas_coords)


def load_model(
    model_path: str,
    num_markers: int,
    num_cell_types: int,
    target_ids: np.ndarray,
    atlas_matrix: np.ndarray,
    device: torch.device,
) -> TapestryModel:
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
    logger.info(
        "Loaded model from %s (epoch %d)", model_path, checkpoint.get("epoch", -1)
    )
    return model


def apply_nnls_gate(raw: np.ndarray, nnls_pred: np.ndarray) -> np.ndarray:
    """Zero raw positions where NNLS assigned 0; renormalise. Falls back to raw
    if NNLS zeroed every position (degenerate sample)."""
    mask = (nnls_pred > 0).astype(np.float32)
    gated = raw * mask
    total = gated.sum()
    if total > 0:
        return gated / total
    return raw


def main():
    parser = argparse.ArgumentParser(
        description="Predict cell-type proportions for cfDNA samples."
    )
    parser.add_argument("--cfdna-dir", required=True,
                        help="Directory with filtered cfDNA PAT files")
    parser.add_argument("--markers-bed", required=True)
    parser.add_argument("--atlas", required=True, help="Path to markers.tsv")
    parser.add_argument("--model", required=True, help="Path to best_model.pt")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--wgbstools", default="wgbstools")
    parser.add_argument("--cohort", default="", help="Cohort name (for logging)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cell types (sorted, matching training convention)
    atlas_df_head = pd.read_csv(args.atlas, sep="\t", nrows=0)
    cell_types = sorted([c for c in atlas_df_head.columns if c not in META_COLS])
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)

    # Load + filter atlas (NaN drop matches training)
    target_ids, atlas_matrix, valid_indices, atlas_coords, total_rows = load_atlas(
        args.atlas, cell_types
    )
    num_markers = atlas_matrix.shape[0]
    num_cell_types = len(cell_types)
    if num_markers < total_rows:
        logger.info(
            "Atlas filter: kept %d / %d rows (dropped %d with NaN in any cell-type column)",
            num_markers, total_rows, total_rows - num_markers,
        )

    # Load model (built against filtered atlas — matches training checkpoint shape)
    model = load_model(
        args.model, num_markers, num_cell_types, target_ids, atlas_matrix, device,
    )

    # NNLS reference profiles: (C, M)
    ref_profiles = atlas_matrix.T.astype(np.float32)

    cfdna_dir = Path(args.cfdna_dir)
    pat_files = sorted(cfdna_dir.glob("*.markers.pat.gz"))
    if not pat_files:
        pat_files = sorted(cfdna_dir.glob("*.pat.gz"))
    logger.info("Found %d cfDNA PAT files in %s", len(pat_files), cfdna_dir)

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

            u_fraction, coverage = result  # aligned to filtered atlas_coords

            # Tapestry forward
            fraction_t = torch.tensor(u_fraction, dtype=torch.float32).unsqueeze(0).to(device)
            coverage_t = torch.tensor(coverage, dtype=torch.float32).unsqueeze(0).to(device)
            u_t = fraction_t * coverage_t
            m_t = (1 - fraction_t) * coverage_t

            with torch.no_grad():
                output = model(u_t, m_t, coverage_t)

            raw = output["proportions"].cpu().numpy()[0]
            detection = output["detection"].cpu().numpy()[0]

            # Coverage-weighted NNLS + Binomial MLE on the same filtered atlas
            X = u_fraction[np.newaxis, :].astype(np.float32)
            cov = coverage[np.newaxis, :].astype(np.float32)
            nnls_pred = run_weighted_nnls(X, cov, ref_profiles)[0]
            binomial_pred = run_binomial_mle(X, cov, ref_profiles)[0]

            # NNLS-gated production prediction (kept as legacy `{ct}` column)
            gated = apply_nnls_gate(raw, nnls_pred)

            row = {
                "sample": sample_name,
                "cohort": args.cohort,
                "mean_coverage": float(coverage[coverage > 0].mean()) if (coverage > 0).any() else 0.0,
                "n_markers_with_coverage": int((coverage > 0).sum()),
            }
            for j, ct in enumerate(cell_types):
                row[ct] = float(gated[j])
                row[f"{ct}_raw"] = float(raw[j])
                row[f"{ct}_nnls"] = float(nnls_pred[j])
                row[f"{ct}_binomial"] = float(binomial_pred[j])
                row[f"{ct}_detection"] = float(detection[j])
            results.append(row)

            for f in Path(tmp_dir).glob("*.uxm.bed.gz"):
                f.unlink()

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved predictions for %d samples to %s", len(df), args.output)

    # Summary on the production (gated) columns
    logger.info("\nPrediction summary (NNLS-gated):")
    for ct in cell_types:
        vals = df[ct].values
        logger.info(
            "  %-25s mean=%.4f median=%.4f max=%.4f",
            ct, vals.mean(), np.median(vals), vals.max(),
        )


if __name__ == "__main__":
    main()
