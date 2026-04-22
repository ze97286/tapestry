#!/usr/bin/env python3
"""Augmented-basis NNLS deconvolution on a cfDNA cohort.

Two-pass pipeline:
  1. Run ``wgbstools homog`` on every cfDNA PAT file; extract per-marker
     U-fraction and coverage, aligned to the atlas (NaN-filtered).
  2. Identify healthy controls by sample-name regex; fit NNLS on those and
     SVD their residuals to build a K-dimensional unknown-tissue basis.
  3. Solve augmented-basis NNLS for every sample using that basis alongside
     the atlas. The augmented coefficients (free sign) absorb methylation
     signal not explained by the atlas — preventing NNLS from misattributing
     non-atlas signal to atlas cell types.

Output CSV has the same format as ``predict_cfdna.py`` so
``clinical_compare_variants.py`` / ``clinical_evaluation.py`` pick it up
directly. Columns:

  ``{ct}``        — augmented-NNLS proportion (production).
  ``{ct}_nnls``   — plain coverage-weighted NNLS for comparison.
  ``unknown_mag`` — sum of absolute unknown-basis coefficients; larger
                    means more of the sample's signal wasn't explained by
                    the atlas.

Usage:
    sbatch --export=ALL,COHORT=AB slurm/09d_predict_cfdna_augmented.sh
"""

import argparse
import gzip
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.benchmark.nnls import run_weighted_nnls
from tapestry.benchmark.augmented_nnls import (
    build_unknown_basis,
    run_augmented_nnls,
)

logger = logging.getLogger(__name__)


META_COLS = [
    "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
    "target", "name", "direction",
    "target_signal", "bg_signal", "snr", "target_total", "bg_total",
]


def load_atlas(atlas_path: str, cell_types: list[str]):
    """Load atlas, drop rows with any NaN in cell-type columns, return
    target_ids, matrix, valid_indices, coords, total_rows."""
    atlas_df = pd.read_csv(atlas_path, sep="\t")
    atlas_ct_cols = [c for c in atlas_df.columns if c not in META_COLS]

    atlas_matrix_full = np.zeros((len(atlas_df), len(cell_types)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        if ct in atlas_ct_cols:
            atlas_matrix_full[:, i] = atlas_df[ct].values.astype(np.float32)

    any_nan = atlas_df[atlas_ct_cols].isna().any(axis=1).values
    valid_indices = np.where(~any_nan)[0]
    atlas_matrix = atlas_matrix_full[valid_indices]
    atlas_coords = [
        (str(atlas_df.iloc[i]["chr"]), int(atlas_df.iloc[i]["start"]))
        for i in valid_indices
    ]
    return atlas_matrix, valid_indices, atlas_coords, len(atlas_df)


def extract_marker_values(homog_path, atlas_coords):
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


def process_cfdna_sample(pat_path, markers_bed, wgbstools, atlas_coords, tmp_dir):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfdna-dir", required=True)
    parser.add_argument("--markers-bed", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wgbstools", default="wgbstools")
    parser.add_argument("--cohort", default="")
    parser.add_argument("--control-pattern", default=r"_Ctrl_|^Ctrl_|_healthy_",
                        help="Regex matching sample names of healthy controls.")
    parser.add_argument("--n-components", type=int, default=3,
                        help="Number of unknown-tissue basis components.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    atlas_df_head = pd.read_csv(args.atlas, sep="\t", nrows=0)
    cell_types = sorted([c for c in atlas_df_head.columns if c not in META_COLS])
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)

    atlas_matrix, valid_indices, atlas_coords, total_rows = load_atlas(args.atlas, cell_types)
    if len(valid_indices) < total_rows:
        logger.info("Atlas filter: kept %d / %d rows", atlas_matrix.shape[0], total_rows)

    cfdna_dir = Path(args.cfdna_dir)
    pat_files = sorted(cfdna_dir.glob("*.markers.pat.gz"))
    if not pat_files:
        pat_files = sorted(cfdna_dir.glob("*.pat.gz"))
    logger.info("Found %d cfDNA PAT files in %s", len(pat_files), cfdna_dir)

    # Pass 1: homog + marker extraction, collect all samples' (u_frac, coverage)
    sample_names = []
    X_all, cov_all = [], []
    with tempfile.TemporaryDirectory(prefix="tapestry_augmented_") as tmp_dir:
        for i, pat_path in enumerate(pat_files):
            sample_name = pat_path.stem.replace(".markers.pat", "").replace(".pat", "")
            logger.info("[%d/%d] homog %s", i + 1, len(pat_files), sample_name)
            result = process_cfdna_sample(
                str(pat_path), args.markers_bed, args.wgbstools, atlas_coords, tmp_dir,
            )
            if result is None:
                continue
            u_fraction, coverage = result
            sample_names.append(sample_name)
            X_all.append(u_fraction)
            cov_all.append(coverage)
            for f in Path(tmp_dir).glob("*.uxm.bed.gz"):
                f.unlink()

    X = np.array(X_all, dtype=np.float32)
    coverage = np.array(cov_all, dtype=np.float32)
    logger.info("Extracted marker values for %d samples", len(sample_names))

    # Identify healthy controls
    ctrl_re = re.compile(args.control_pattern)
    is_control = np.array([bool(ctrl_re.search(s)) for s in sample_names])
    n_ctrl = int(is_control.sum())
    logger.info("Healthy controls matched by %r: %d / %d samples",
                args.control_pattern, n_ctrl, len(sample_names))

    if n_ctrl < 5:
        logger.error("Need at least 5 control samples to build a stable unknown-tissue basis. "
                     "Got %d. Adjust --control-pattern or provide more controls.", n_ctrl)
        return
    if n_ctrl < args.n_components * 3:
        logger.warning("Only %d controls for %d components — basis may be unstable.",
                       n_ctrl, args.n_components)

    # Build unknown-tissue basis from control residuals
    logger.info("Building unknown-tissue basis (K=%d) from %d controls...",
                args.n_components, n_ctrl)
    U_basis, var_explained = build_unknown_basis(
        X[is_control], coverage[is_control], atlas_matrix,
        n_components=args.n_components,
    )
    for k, v in enumerate(var_explained):
        logger.info("  unknown component %d: explains %.2f%% of control residual variance",
                    k + 1, v * 100)

    # Augmented NNLS on all samples
    logger.info("Running augmented NNLS on all %d samples...", len(sample_names))
    aug_props, y_coef, y_mag = run_augmented_nnls(X, coverage, atlas_matrix, U_basis)

    # Plain NNLS for comparison column
    logger.info("Running plain NNLS for comparison column...")
    nnls_props = run_weighted_nnls(X, coverage, atlas_matrix)

    # Assemble output
    results = []
    for i, name in enumerate(sample_names):
        cov_i = coverage[i]
        mean_cov = float(cov_i[cov_i > 0].mean()) if (cov_i > 0).any() else 0.0
        n_with_cov = int((cov_i > 0).sum())
        row = {
            "sample": name,
            "cohort": args.cohort,
            "mean_coverage": mean_cov,
            "n_markers_with_coverage": n_with_cov,
            "is_control": bool(is_control[i]),
            "unknown_mag": float(y_mag[i]),
        }
        for j, ct in enumerate(cell_types):
            row[ct] = float(aug_props[i, j])
            row[f"{ct}_nnls"] = float(nnls_props[i, j])
        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved predictions for %d samples to %s", len(df), args.output)

    # Diagnostic summary
    logger.info("\nProduction (augmented) OAC stats:")
    if "OAC" in cell_types and is_control.any():
        oac_ctrl = df.loc[df["is_control"], "OAC"].values
        oac_cancer = df.loc[~df["is_control"], "OAC"].values
        logger.info("  controls (n=%d):   mean=%.4f  max=%.4f  95pct=%.4f",
                    len(oac_ctrl), oac_ctrl.mean(), oac_ctrl.max(), np.percentile(oac_ctrl, 95))
        logger.info("  non-controls (n=%d): mean=%.4f  max=%.4f  95pct=%.4f",
                    len(oac_cancer), oac_cancer.mean(), oac_cancer.max(),
                    np.percentile(oac_cancer, 95))
    logger.info("\nNNLS comparison OAC stats:")
    if "OAC_nnls" in df.columns and is_control.any():
        oac_ctrl = df.loc[df["is_control"], "OAC_nnls"].values
        oac_cancer = df.loc[~df["is_control"], "OAC_nnls"].values
        logger.info("  controls (n=%d):   mean=%.4f  max=%.4f  95pct=%.4f",
                    len(oac_ctrl), oac_ctrl.mean(), oac_ctrl.max(), np.percentile(oac_ctrl, 95))
        logger.info("  non-controls (n=%d): mean=%.4f  max=%.4f  95pct=%.4f",
                    len(oac_cancer), oac_cancer.mean(), oac_cancer.max(),
                    np.percentile(oac_cancer, 95))


if __name__ == "__main__":
    main()
