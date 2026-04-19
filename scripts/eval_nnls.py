#!/usr/bin/env python3
"""Run NNLS on the eval dataset and report metrics.

Usage:
    python scripts/eval_nnls.py \
        --data-dir runs/run_002/training/eval \
        --atlas runs/run_002/markers/markers.tsv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def run_weighted_nnls(X, coverage, reference_profiles):
    from scipy.optimize import nnls
    n_samples = X.shape[0]
    n_cell_types = reference_profiles.shape[0]
    estimated = np.zeros((n_samples, n_cell_types))
    A = reference_profiles.T
    for i in range(n_samples):
        w = coverage[i]
        A_w = A * w[:, np.newaxis]
        b_w = X[i] * w
        zero_mask = w == 0
        if np.any(zero_mask):
            A_w[zero_mask, :] = 0
            b_w[zero_mask] = 0
        x, _ = nnls(A_w, b_w)
        total = x.sum()
        if total > 0:
            x /= total
        estimated[i] = x
    return estimated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--atlas", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    mv_df = pd.read_parquet(data_dir / "marker_values.parquet")
    cov_df = pd.read_parquet(data_dir / "coverage.parquet")
    y_df = pd.read_parquet(data_dir / "ground_truth_y.parquet")

    id_cols = ["sample_id", "name", "direction"]
    mv_cols = [c for c in mv_df.columns if c not in id_cols]
    cov_cols = [c for c in cov_df.columns if c not in id_cols]
    ct_cols = [c for c in y_df.columns if c not in id_cols]

    X = np.nan_to_num(mv_df[mv_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    coverage = np.nan_to_num(cov_df[cov_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    true = np.nan_to_num(y_df[ct_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Build atlas reference profiles (C, M). Drop rows with any NaN in the
    # cell-type columns — same filter as tapestry training — so NNLS sees the
    # same clean atlas rather than 0.5 noise-fills for uninformative markers.
    atlas_df = pd.read_csv(args.atlas, sep="\t")
    meta_cols = ["chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
                 "target", "name", "direction", "target_signal", "bg_signal", "snr",
                 "target_total", "bg_total"]
    atlas_ct_cols = [c for c in atlas_df.columns if c not in meta_cols]

    any_nan = atlas_df[atlas_ct_cols].isna().any(axis=1).values
    valid_indices = np.where(~any_nan)[0]
    if len(valid_indices) < len(atlas_df):
        logger.info(
            "Atlas filter: dropped %d / %d rows with NaN in any cell-type column",
            len(atlas_df) - len(valid_indices), len(atlas_df),
        )
        atlas_df = atlas_df.iloc[valid_indices].reset_index(drop=True)
        X = X[:, valid_indices]
        coverage = coverage[:, valid_indices]

    ref = np.zeros((len(ct_cols), len(atlas_df)), dtype=np.float32)
    for i, ct in enumerate(ct_cols):
        if ct in atlas_ct_cols:
            ref[i] = atlas_df[ct].values.astype(np.float32)

    logger.info("Running NNLS on %d samples, %d markers, %d cell types", X.shape[0], X.shape[1], len(ct_cols))
    preds = run_weighted_nnls(X, coverage, ref)

    # Metrics
    eps = 1e-7
    print(f"\n{'Cell Type':25s} {'MAE':>8s} {'R²':>8s}")
    print("-" * 43)
    for i, ct in enumerate(ct_cols):
        p, t = preds[:, i], true[:, i]
        mae = np.mean(np.abs(p - t))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + eps) if ss_tot > eps else 0.0
        print(f"{ct:25s} {mae:8.4f} {r2:8.4f}")

    overall_mae = np.mean(np.abs(preds - true))
    print(f"\n{'Overall':25s} {overall_mae:8.4f}")

    # Log-space
    mask = (true > eps) & (preds > eps)
    if mask.sum() > 10:
        log_true = np.log10(true[mask])
        log_pred = np.log10(preds[mask])
        ss_res = np.sum((log_true - log_pred) ** 2)
        ss_tot = np.sum((log_true - log_true.mean()) ** 2)
        log_r2 = 1 - ss_res / (ss_tot + eps)
        coeffs = np.polyfit(log_true, log_pred, 1)
        print(f"\nLog-space: R²={log_r2:.4f} slope={coeffs[0]:.4f} intercept={coeffs[1]:.4f}")

    # Presence
    threshold = 0.001
    pred_present = preds > threshold
    true_present = true > threshold
    tp = (pred_present & true_present).sum()
    fp = (pred_present & ~true_present).sum()
    fn = (~pred_present & true_present).sum()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    print(f"Presence: precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()
