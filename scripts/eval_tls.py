#!/usr/bin/env python3
"""Run scaled Total Least Squares deconvolution on the eval dataset.

Same metric surface as ``eval_nnls.py`` — per-cell-type MAE + R², overall MAE,
log-space R²/slope/intercept, presence precision/recall/F1 — so the output
drops straight into a side-by-side with NNLS.

The novelty here: TLS accounts for both observation noise (via per-marker
binomial variance from coverage) AND atlas noise (via ``--atlas-sigma``). NNLS
ignores the second, which is why it over-weights unreliable atlas entries.

Usage:
    python scripts/eval_tls.py \\
        --data-dir runs/run_v0.3/training/eval \\
        --atlas runs/run_v0.3/markers/markers.tsv \\
        --atlas-sigma 0.05
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.benchmark.tls import run_tls_deconvolution
from tapestry.benchmark.nnls import run_weighted_nnls

logger = logging.getLogger(__name__)


META_COLS = [
    "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
    "target", "name", "direction",
    "target_signal", "bg_signal", "snr", "target_total", "bg_total",
]
ID_COLS = ["sample_id", "name", "direction"]


def load_eval(data_dir: Path, atlas_path: Path):
    mv_df = pd.read_parquet(data_dir / "marker_values.parquet")
    cov_df = pd.read_parquet(data_dir / "coverage.parquet")
    y_df = pd.read_parquet(data_dir / "ground_truth_y.parquet")

    mv_cols = [c for c in mv_df.columns if c not in ID_COLS]
    cov_cols = [c for c in cov_df.columns if c not in ID_COLS]
    ct_cols = [c for c in y_df.columns if c not in ID_COLS]

    X = np.nan_to_num(mv_df[mv_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    coverage = np.nan_to_num(cov_df[cov_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    true = np.nan_to_num(y_df[ct_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    atlas_df = pd.read_csv(atlas_path, sep="\t")
    atlas_ct_cols = [c for c in atlas_df.columns if c not in META_COLS]

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

    return X, coverage, true, ref, ct_cols


def report(preds: np.ndarray, true: np.ndarray, ct_cols: list[str], method_name: str):
    eps = 1e-7
    print(f"\n===== {method_name} =====")
    print(f"{'Cell Type':25s} {'MAE':>8s} {'R²':>8s}")
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

    mask = (true > eps) & (preds > eps)
    if mask.sum() > 10:
        log_true = np.log10(true[mask])
        log_pred = np.log10(preds[mask])
        ss_res = np.sum((log_true - log_pred) ** 2)
        ss_tot = np.sum((log_true - log_true.mean()) ** 2)
        log_r2 = 1 - ss_res / (ss_tot + eps)
        coeffs = np.polyfit(log_true, log_pred, 1)
        print(f"\nLog-space: R²={log_r2:.4f} slope={coeffs[0]:.4f} intercept={coeffs[1]:.4f}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--atlas-sigma", type=float, default=0.05,
                        help="Per-entry atlas noise std (default 0.05, i.e. 5 points)")
    parser.add_argument("--also-nnls", action="store_true",
                        help="Also run coverage-weighted NNLS for side-by-side comparison")
    parser.add_argument("--max-iter", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, coverage, true, ref, ct_cols = load_eval(Path(args.data_dir), Path(args.atlas))
    logger.info(
        "Running TLS on %d samples, %d markers, %d cell types (atlas_sigma=%.3f)",
        X.shape[0], X.shape[1], len(ct_cols), args.atlas_sigma,
    )

    tls_preds = run_tls_deconvolution(
        X, coverage, ref,
        atlas_sigma=args.atlas_sigma,
        max_iter=args.max_iter,
    )
    report(tls_preds, true, ct_cols, f"TLS (atlas_sigma={args.atlas_sigma})")

    if args.also_nnls:
        logger.info("Running coverage-weighted NNLS for comparison")
        nnls_preds = run_weighted_nnls(X, coverage, ref)
        report(nnls_preds, true, ct_cols, "NNLS (reference)")


if __name__ == "__main__":
    main()
