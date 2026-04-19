#!/usr/bin/env python3
"""Post-hoc calibrated + NNLS-gated predictions for the TapestryModel.

Loads a trained checkpoint, runs inference on the eval set, and produces four
prediction variants plus a side-by-side metrics report:

  1. raw          — direct model output
  2. recalibrated — log-space affine recalibration per cell type
                    (α, β fit on the training set: log_true = α·log_pred + β)
  3. nnls_gated   — raw model output zeroed where NNLS says absent, renormalised
  4. recal_gated  — (2) then (3)

Usage:
    python scripts/predict_calibrated.py \\
        --checkpoint runs/run_v0.3/models/tapestry/best_model.pt \\
        --train-dir runs/run_v0.3/training/train \\
        --eval-dir runs/run_v0.3/training/eval \\
        --atlas runs/run_v0.3/markers/markers.tsv \\
        --output-dir runs/run_v0.3/models/tapestry/calibrated
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from tapestry.models.deconvolution import TapestryModel
from tapestry.benchmark.nnls import run_weighted_nnls

logger = logging.getLogger(__name__)


ID_COLS = ["sample_id", "name", "direction"]
META_COLS = ["chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
             "target", "name", "direction", "target_signal", "bg_signal", "snr",
             "target_total", "bg_total"]


def load_parquet_data(data_dir: Path) -> dict:
    mv_df = pd.read_parquet(data_dir / "marker_values.parquet")
    cov_df = pd.read_parquet(data_dir / "coverage.parquet")
    y_df = pd.read_parquet(data_dir / "ground_truth_y.parquet")
    mv_cols = [c for c in mv_df.columns if c not in ID_COLS]
    cov_cols = [c for c in cov_df.columns if c not in ID_COLS]
    ct_cols = [c for c in y_df.columns if c not in ID_COLS]
    fraction = np.nan_to_num(
        mv_df[mv_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    coverage = np.nan_to_num(
        cov_df[cov_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    proportions = np.nan_to_num(
        y_df[ct_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    return {
        "fraction": fraction, "coverage": coverage,
        "proportions": proportions, "cell_types": ct_cols,
    }


def load_atlas(atlas_path: str, cell_types: list[str]):
    atlas_df = pd.read_csv(atlas_path, sep="\t")
    ct_cols_in_atlas = [c for c in atlas_df.columns if c not in META_COLS]
    atlas_matrix_full = np.zeros((len(atlas_df), len(cell_types)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        if ct in ct_cols_in_atlas:
            atlas_matrix_full[:, i] = atlas_df[ct].values.astype(np.float32)
    any_nan = atlas_df[ct_cols_in_atlas].isna().any(axis=1).values
    valid_indices = np.where(~any_nan)[0]
    atlas_matrix = atlas_matrix_full[valid_indices]
    target_ids = atlas_df["target"].iloc[valid_indices].map(
        lambda x: cell_types.index(x)
    ).values
    return target_ids, atlas_matrix, valid_indices


@torch.no_grad()
def run_tapestry(model, data, batch_size, device) -> np.ndarray:
    fraction = torch.from_numpy(data["fraction"])
    coverage = torch.from_numpy(data["coverage"])
    u = fraction * coverage
    m = (1 - fraction) * coverage
    ds = TensorDataset(u, m, coverage)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    for u_b, m_b, c_b in dl:
        u_b, m_b, c_b = u_b.to(device), m_b.to(device), c_b.to(device)
        out = model(u_b, m_b, c_b)
        preds.append(out["proportions"].cpu().numpy())
    return np.concatenate(preds)


def fit_log_recalibration(
    pred: np.ndarray,
    true: np.ndarray,
    cell_types: list[str],
    threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit α, β per cell type by regressing log10(true) on log10(pred)."""
    alphas = np.ones(len(cell_types), dtype=np.float32)
    betas = np.zeros(len(cell_types), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        t = true[:, i]
        p = pred[:, i]
        mask = (t > threshold) & (p > 1e-8)
        if mask.sum() < 10:
            logger.warning(
                "Not enough samples for %s (%d) — leaving uncalibrated", ct, int(mask.sum())
            )
            continue
        log_t = np.log10(t[mask])
        log_p = np.log10(p[mask])
        coeffs = np.polyfit(log_p, log_t, 1)
        alphas[i] = float(coeffs[0])
        betas[i] = float(coeffs[1])
    return alphas, betas


def apply_recalibration(pred: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    eps = 1e-8
    log_p = np.log10(pred + eps)
    log_p_cal = alphas[np.newaxis, :] * log_p + betas[np.newaxis, :]
    p_cal = np.power(10.0, log_p_cal).astype(np.float32)
    total = p_cal.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return p_cal / total


def apply_nnls_gate(pred: np.ndarray, nnls_pred: np.ndarray) -> np.ndarray:
    """Zero out pred positions where NNLS assigned exactly 0; renormalise."""
    mask = (nnls_pred > 0).astype(np.float32)
    gated = pred * mask
    total = gated.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return gated / total


def compute_metrics(
    pred: np.ndarray, true: np.ndarray, cell_types: list[str], threshold: float = 0.001,
) -> dict:
    eps = 1e-7
    metrics: dict = {
        "overall_mae": float(np.mean(np.abs(pred - true))),
    }
    per_ct: dict = {}
    for i, ct in enumerate(cell_types):
        p, t = pred[:, i], true[:, i]
        mae = float(np.mean(np.abs(p - t)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + eps)) if ss_tot > eps else 0.0
        per_ct[ct] = {"mae": mae, "r2": r2}
    metrics["per_cell_type"] = per_ct

    mask = (true > eps) & (pred > eps)
    if mask.sum() > 10:
        log_t = np.log10(true[mask])
        log_p = np.log10(pred[mask])
        ss_res = np.sum((log_t - log_p) ** 2)
        ss_tot = np.sum((log_t - log_t.mean()) ** 2)
        metrics["log_r2"] = float(1 - ss_res / (ss_tot + eps))
        coeffs = np.polyfit(log_t, log_p, 1)
        metrics["log_slope"] = float(coeffs[0])
        metrics["log_intercept"] = float(coeffs[1])
    else:
        metrics["log_r2"] = 0.0
        metrics["log_slope"] = 0.0
        metrics["log_intercept"] = 0.0

    pred_present = pred > threshold
    true_present = true > threshold
    tp = float((pred_present & true_present).sum())
    fp = float((pred_present & ~true_present).sum())
    fn = float((~pred_present & true_present).sum())
    metrics["pres_precision"] = tp / (tp + fp + eps)
    metrics["pres_recall"] = tp / (tp + fn + eps)
    metrics["pres_f1"] = (
        2 * metrics["pres_precision"] * metrics["pres_recall"]
        / (metrics["pres_precision"] + metrics["pres_recall"] + eps)
    )
    return metrics


def print_side_by_side(results: dict, cell_types: list[str]) -> None:
    variants = list(results.keys())
    col_hdr = " ".join(f"{v:>14s}" for v in variants)
    print(f"\n{'Metric':22s} {col_hdr}")
    print("-" * (22 + 15 * len(variants)))
    for key, label in [
        ("overall_mae",    "Overall MAE"),
        ("log_r2",         "Log R²"),
        ("log_slope",      "Log slope"),
        ("log_intercept",  "Log intercept"),
        ("pres_precision", "Presence prec"),
        ("pres_recall",    "Presence rec"),
        ("pres_f1",        "Presence F1"),
    ]:
        row = " ".join(f"{results[v][key]:14.4f}" for v in variants)
        print(f"{label:22s} {row}")

    print(f"\n{'Per cell-type MAE':22s} {col_hdr}")
    print("-" * (22 + 15 * len(variants)))
    for ct in cell_types:
        row = " ".join(f"{results[v]['per_cell_type'][ct]['mae']:14.4f}" for v in variants)
        print(f"{ct:22s} {row}")

    print(f"\n{'Per cell-type R²':22s} {col_hdr}")
    print("-" * (22 + 15 * len(variants)))
    for ct in cell_types:
        row = " ".join(f"{results[v]['per_cell_type'][ct]['r2']:14.4f}" for v in variants)
        print(f"{ct:22s} {row}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-dir", required=True,
                        help="Used to fit recalibration coefficients.")
    parser.add_argument("--eval-dir", required=True,
                        help="Held-out split for reporting metrics.")
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--presence-threshold", type=float, default=0.001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load data
    train_data = load_parquet_data(Path(args.train_dir))
    eval_data = load_parquet_data(Path(args.eval_dir))
    cell_types = train_data["cell_types"]
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)

    # Atlas (with NaN-row filter — keeps data and atlas column-aligned)
    target_ids, atlas_matrix, valid_indices = load_atlas(args.atlas, cell_types)
    data_markers = train_data["fraction"].shape[1]
    if len(valid_indices) < data_markers:
        logger.info(
            "Atlas filter: dropped %d / %d rows with NaN in any cell-type column",
            data_markers - len(valid_indices), data_markers,
        )
        for split in (train_data, eval_data):
            split["fraction"] = split["fraction"][:, valid_indices]
            split["coverage"] = split["coverage"][:, valid_indices]

    num_markers = atlas_matrix.shape[0]
    num_cell_types = len(cell_types)
    logger.info("Markers after filter: %d", num_markers)

    # Checkpoint — try to rebuild model with matching config
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = TapestryModel(
        num_markers=num_markers,
        num_cell_types=num_cell_types,
        target_ids=target_ids,
        atlas=atlas_matrix,
        feature_dim=int(config.get("feature_dim", 64)),
        l1_num_heads=int(config.get("l1_heads", 4)),
        l1_num_layers=int(config.get("l1_layers", 2)),
        l2_num_heads=int(config.get("l2_heads", 4)),
        l2_num_layers=int(config.get("l2_layers", 1)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    # Tapestry inference on train (for recalibration fit) and eval (for metrics)
    logger.info("Running tapestry on train set (%d samples)", train_data["fraction"].shape[0])
    pred_train_raw = run_tapestry(model, train_data, args.batch_size, device)
    logger.info("Running tapestry on eval set (%d samples)", eval_data["fraction"].shape[0])
    pred_eval_raw = run_tapestry(model, eval_data, args.batch_size, device)

    # NNLS on eval (uses atlas as reference_profiles; shape (C, M))
    logger.info("Running NNLS on eval set")
    ref = atlas_matrix.T  # (C, M)
    pred_eval_nnls = run_weighted_nnls(eval_data["fraction"], eval_data["coverage"], ref)

    # Fit affine log-space recalibration on train set
    logger.info("Fitting affine log-space recalibration on train")
    alphas, betas = fit_log_recalibration(pred_train_raw, train_data["proportions"], cell_types)
    for i, ct in enumerate(cell_types):
        logger.info("  %-25s α=%6.3f β=%6.3f", ct, alphas[i], betas[i])

    # Build variants on eval
    true_eval = eval_data["proportions"]
    preds = {
        "raw":           pred_eval_raw,
        "recalibrated":  apply_recalibration(pred_eval_raw, alphas, betas),
        "nnls_gated":    apply_nnls_gate(pred_eval_raw, pred_eval_nnls),
    }
    preds["recal_gated"] = apply_nnls_gate(preds["recalibrated"], pred_eval_nnls)

    # Metrics
    results = {
        name: compute_metrics(p, true_eval, cell_types, args.presence_threshold)
        for name, p in preds.items()
    }
    print_side_by_side(results, cell_types)

    # Save predictions + calibration coefficients + metrics
    for name, p in preds.items():
        np.save(out_dir / f"pred_eval_{name}.npy", p)
    np.savez(
        out_dir / "calibration.npz",
        alphas=alphas, betas=betas, cell_types=np.array(cell_types),
    )
    np.save(out_dir / "pred_eval_nnls.npy", pred_eval_nnls)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved predictions and metrics to %s", out_dir)


if __name__ == "__main__":
    main()
