#!/usr/bin/env python3
"""Sweep ``atlas_sigma`` for the TLS solver on the eval set and summarise.

Runs the TLS solver once per sigma value and reports a compact table of the
metrics that matter (overall MAE, log R², log slope, OAC R², presence F1) so
we can eyeball whether any sigma beats NNLS in a single log.

Usage:
    python scripts/sweep_tls_sigma.py \\
        --data-dir runs/run_v0.3/training/eval \\
        --atlas runs/run_v0.3/markers/markers.tsv \\
        --sigmas 0.0 0.01 0.02 0.05 0.10 0.20
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
        atlas_df = atlas_df.iloc[valid_indices].reset_index(drop=True)
        X = X[:, valid_indices]
        coverage = coverage[:, valid_indices]

    ref = np.zeros((len(ct_cols), len(atlas_df)), dtype=np.float32)
    for i, ct in enumerate(ct_cols):
        if ct in atlas_ct_cols:
            ref[i] = atlas_df[ct].values.astype(np.float32)

    return X, coverage, true, ref, ct_cols


def metrics(preds: np.ndarray, true: np.ndarray, ct_cols: list[str]) -> dict:
    eps = 1e-7
    out = {"mae": float(np.mean(np.abs(preds - true)))}

    mask = (true > eps) & (preds > eps)
    if mask.sum() > 10:
        log_t = np.log10(true[mask])
        log_p = np.log10(preds[mask])
        ss_res = np.sum((log_t - log_p) ** 2)
        ss_tot = np.sum((log_t - log_t.mean()) ** 2)
        out["log_r2"] = float(1 - ss_res / (ss_tot + eps))
        coeffs = np.polyfit(log_t, log_p, 1)
        out["log_slope"] = float(coeffs[0])
    else:
        out["log_r2"] = 0.0
        out["log_slope"] = 0.0

    oac_idx = ct_cols.index("OAC") if "OAC" in ct_cols else None
    if oac_idx is not None:
        t, p = true[:, oac_idx], preds[:, oac_idx]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        out["oac_r2"] = float(1 - ss_res / (ss_tot + eps)) if ss_tot > eps else 0.0
    else:
        out["oac_r2"] = float("nan")

    thr = 0.001
    tp = float(((preds > thr) & (true > thr)).sum())
    fp = float(((preds > thr) & ~(true > thr)).sum())
    fn = float((~(preds > thr) & (true > thr)).sum())
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    out["pres_f1"] = 2 * prec * rec / (prec + rec + eps)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--max-iter", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, coverage, true, ref, ct_cols = load_eval(Path(args.data_dir), Path(args.atlas))
    logger.info("Eval set: %d samples, %d markers, %d cell types", X.shape[0], X.shape[1], len(ct_cols))

    logger.info("Running NNLS reference...")
    nnls_preds = run_weighted_nnls(X, coverage, ref)
    nnls_m = metrics(nnls_preds, true, ct_cols)

    rows = []
    rows.append({"method": "NNLS", "sigma": float("nan"), **nnls_m})

    for sigma in args.sigmas:
        logger.info("Running TLS with atlas_sigma=%.4f...", sigma)
        tls_preds = run_tls_deconvolution(
            X, coverage, ref,
            atlas_sigma=max(sigma, 1e-12),  # 0 leads to divide-by-zero in IRLS
            max_iter=args.max_iter,
        )
        m = metrics(tls_preds, true, ct_cols)
        rows.append({"method": "TLS", "sigma": sigma, **m})

    # Print a compact side-by-side
    print(f"\n{'method':8s} {'sigma':>8s}  {'MAE':>8s}  {'log_R²':>8s}  {'slope':>7s}  {'OAC R²':>8s}  {'pres_F1':>8s}")
    print("-" * 65)
    for r in rows:
        sig = f"{r['sigma']:.4f}" if not np.isnan(r['sigma']) else "    —"
        print(
            f"{r['method']:8s} {sig:>8s}  "
            f"{r['mae']:8.4f}  {r['log_r2']:8.4f}  {r['log_slope']:7.3f}  "
            f"{r['oac_r2']:8.4f}  {r['pres_f1']:8.4f}"
        )


if __name__ == "__main__":
    main()
