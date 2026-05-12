#!/usr/bin/env python3
"""Augmented-basis NNLS deconvolution on a cfDNA cohort.

Two-pass pipeline:
  1. Run ``wgbstools homog`` on every cfDNA PAT file; extract per-marker
     U-fraction and coverage, aligned to the atlas (NaN-filtered).
  2. Identify healthy controls by sample-name regex; fit NNLS on those and
     SVD their residuals to build a K-dimensional unknown-tissue basis.
  3. Optionally project the target-cell-type contrast out of the unknown
     basis, then solve augmented-basis NNLS over a lambda path. The unknown
     coefficients are free sign but ridge-penalised, so non-atlas signal can
     be absorbed without making the nuisance channel arbitrary.

Output CSV has the same format as ``predict_cfdna.py`` so
``clinical_compare_variants.py`` / ``clinical_evaluation.py`` pick it up
directly. Columns:

  ``{ct}``        — augmented-NNLS proportion (production).
  ``{ct}_nnls``   — plain coverage-weighted NNLS for comparison.
  ``unknown_mag`` — sum of absolute unknown-basis coefficients; larger
                    means more of the sample's signal wasn't explained by
                    the atlas.
  ``*_aug_lam_*`` — lambda-path diagnostics for the target cell type.

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
    project_basis_orthogonal_to,
    run_augmented_nnls_path,
    target_contrast,
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
    else:
        matches = sorted(Path(tmp_dir).glob(f"{sample_name}*.uxm.bed.gz"))
        if len(matches) == 1:
            homog_out = str(matches[0])
        elif matches:
            logger.error(
                "Multiple homog outputs matched %s in %s: %s",
                sample_name, tmp_dir, ", ".join(str(p) for p in matches),
            )
            return None
        else:
            logger.error(
                "homog completed for %s but no output was found in %s",
                sample_name, tmp_dir,
            )
            return None
    return extract_marker_values(homog_out, atlas_coords)


def parse_lambda_grid(value: str) -> np.ndarray:
    values = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("--lambda-unknown-grid must contain at least one value")
    values = np.array(values, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("--lambda-unknown-grid values must be non-negative")
    return np.array(sorted(set(float(v) for v in values)), dtype=np.float64)


def lambda_label(value: float) -> str:
    label = f"{value:g}"
    return label.replace("-", "m").replace(".", "p").replace("+", "")


def add_primary_lambda(lambda_grid: np.ndarray, primary_lambda: float) -> np.ndarray:
    if primary_lambda < 0:
        raise ValueError("--primary-lambda-unknown must be non-negative")
    if np.any(np.isclose(lambda_grid, primary_lambda, rtol=0, atol=1e-12)):
        return lambda_grid
    return np.array(sorted([*lambda_grid.tolist(), float(primary_lambda)]), dtype=np.float64)


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
    parser.add_argument("--lambda-unknown-grid",
                        default="0,0.01,0.1,1,10,100,1000,10000",
                        help="Comma-separated ridge penalties for unknown coefficients.")
    parser.add_argument("--primary-lambda-unknown", type=float, default=10.0,
                        help="Lambda used for production columns in the main output CSV.")
    parser.add_argument("--orthogonalize-target", default="OAC",
                        help="Cell type whose atlas contrast is removed from the unknown basis. "
                             "Use an empty string to disable.")
    parser.add_argument("--path-output", default=None,
                        help="Optional long-form lambda-path CSV. Defaults to OUTPUT stem "
                             "with '_lambda_path.csv'.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    atlas_df_head = pd.read_csv(args.atlas, sep="\t", nrows=0)
    cell_types = sorted([c for c in atlas_df_head.columns if c not in META_COLS])
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)
    lambda_grid = parse_lambda_grid(args.lambda_unknown_grid)
    lambda_grid = add_primary_lambda(lambda_grid, args.primary_lambda_unknown)
    primary_lambda_idx = int(np.argmin(np.abs(lambda_grid - args.primary_lambda_unknown)))
    primary_lambda = float(lambda_grid[primary_lambda_idx])
    logger.info("Unknown lambda grid: %s", ", ".join(f"{v:g}" for v in lambda_grid))
    logger.info("Primary unknown lambda: %g", primary_lambda)

    atlas_matrix, valid_indices, atlas_coords, total_rows = load_atlas(args.atlas, cell_types)
    if len(valid_indices) < total_rows:
        logger.info("Atlas filter: kept %d / %d rows", atlas_matrix.shape[0], total_rows)
    # Benchmark utilities use (C, M) convention; load_atlas returns (M, C).
    reference_profiles = atlas_matrix.T.astype(np.float32)

    cfdna_dir = Path(args.cfdna_dir)
    pat_files = sorted(cfdna_dir.glob("*.markers.pat.gz"))
    if not pat_files:
        pat_files = sorted(cfdna_dir.glob("*.pat.gz"))
    logger.info("Found %d cfDNA PAT files in %s", len(pat_files), cfdna_dir)
    if not pat_files:
        logger.error("No PAT files found in %s", cfdna_dir)
        raise SystemExit(2)

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
    if len(sample_names) == 0:
        logger.error("No samples were successfully processed")
        raise SystemExit(2)

    # Identify healthy controls
    ctrl_re = re.compile(args.control_pattern)
    is_control = np.array([bool(ctrl_re.search(s)) for s in sample_names])
    n_ctrl = int(is_control.sum())
    logger.info("Healthy controls matched by %r: %d / %d samples",
                args.control_pattern, n_ctrl, len(sample_names))

    if n_ctrl < 2:
        logger.error("Need at least 2 control samples to build an unknown-tissue basis. "
                     "Got %d. Adjust --control-pattern or provide more controls.", n_ctrl)
        raise SystemExit(2)

    # Cap components at n_ctrl - 1 (SVD rank limit for a centered basis) or
    # n_ctrl (uncentered). Without centering we can go up to n_ctrl but the
    # last singular vector is typically near-zero; cap at min(K, n_ctrl).
    k_effective = min(args.n_components, n_ctrl)
    if k_effective < args.n_components:
        logger.warning(
            "Requested K=%d but only %d controls — reducing to K=%d.",
            args.n_components, n_ctrl, k_effective,
        )
    if n_ctrl < k_effective * 3:
        logger.warning(
            "Only %d controls for K=%d components — basis may be noisy. "
            "Consider --n-components %d for stability.",
            n_ctrl, k_effective, max(1, n_ctrl // 3),
        )

    # Build unknown-tissue basis from control residuals
    logger.info("Building unknown-tissue basis (K=%d) from %d controls...",
                k_effective, n_ctrl)
    U_basis, var_explained = build_unknown_basis(
        X[is_control], coverage[is_control], reference_profiles,
        n_components=k_effective,
    )
    for k, v in enumerate(var_explained):
        logger.info("  unknown component %d: explains %.2f%% of control residual variance",
                    k + 1, v * 100)

    target_name = args.orthogonalize_target.strip()
    target_idx = cell_types.index(target_name) if target_name in cell_types else None
    if target_name and target_idx is None:
        logger.warning("Requested --orthogonalize-target %r, but it is not an atlas cell type.",
                       target_name)
    if target_idx is not None and U_basis.shape[1] > 0:
        before_k = U_basis.shape[1]
        contrast = target_contrast(reference_profiles, target_idx)
        U_basis = project_basis_orthogonal_to(U_basis, contrast)
        after_k = U_basis.shape[1]
        logger.info("Projected unknown basis orthogonal to %s contrast: K %d -> %d",
                    target_name, before_k, after_k)
        if after_k == 0:
            logger.warning("Unknown basis vanished after target-contrast projection. "
                           "Augmented model will reduce to atlas-only NNLS.")

    # Augmented NNLS on all samples
    logger.info("Running regularised augmented NNLS path on all %d samples...", len(sample_names))
    path = run_augmented_nnls_path(
        X, coverage, reference_profiles, U_basis, lambda_values=lambda_grid,
    )
    aug_props = path["proportions"][primary_lambda_idx]
    y_mag = path["unknown_mag"][primary_lambda_idx]
    residual_norm = path["residual_norm"][primary_lambda_idx]

    # Plain NNLS for comparison column
    logger.info("Running plain NNLS for comparison column...")
    nnls_props = run_weighted_nnls(X, coverage, reference_profiles)

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
            "unknown_lambda": primary_lambda,
            "unknown_residual_norm": float(residual_norm[i]),
            "unknown_n_components": int(U_basis.shape[1]),
        }
        if target_idx is not None:
            for l_idx, lam in enumerate(lambda_grid):
                lam_label = lambda_label(float(lam))
                row[f"{target_name}_aug_lam_{lam_label}"] = float(
                    path["proportions"][l_idx, i, target_idx]
                )
                row[f"unknown_mag_lam_{lam_label}"] = float(path["unknown_mag"][l_idx, i])
                row[f"residual_norm_lam_{lam_label}"] = float(
                    path["residual_norm"][l_idx, i]
                )
        for j, ct in enumerate(cell_types):
            row[ct] = float(aug_props[i, j])
            row[f"{ct}_nnls"] = float(nnls_props[i, j])
        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved predictions for %d samples to %s", len(df), args.output)

    path_output = args.path_output
    if path_output is None:
        output_path = Path(args.output)
        path_output = str(output_path.with_name(f"{output_path.stem}_lambda_path.csv"))
    path_rows = []
    for l_idx, lam in enumerate(lambda_grid):
        for i, name in enumerate(sample_names):
            row = {
                "sample": name,
                "cohort": args.cohort,
                "is_control": bool(is_control[i]),
                "lambda_unknown": float(lam),
                "unknown_mag": float(path["unknown_mag"][l_idx, i]),
                "residual_norm": float(path["residual_norm"][l_idx, i]),
            }
            for j, ct in enumerate(cell_types):
                row[ct] = float(path["proportions"][l_idx, i, j])
            path_rows.append(row)
    path_df = pd.DataFrame(path_rows)
    os.makedirs(os.path.dirname(path_output) or ".", exist_ok=True)
    path_df.to_csv(path_output, index=False)
    logger.info("Saved lambda-path diagnostics to %s", path_output)

    # Diagnostic summary
    logger.info("\nProduction (augmented, lambda=%g) OAC stats:", primary_lambda)
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
