#!/usr/bin/env python3
"""Train the TapestryModel for cfDNA deconvolution.

Usage:
    python scripts/train_tapestry.py \
        --train-dir runs/run_002/training/train \
        --eval-dir runs/run_002/training/eval \
        --atlas runs/run_002/markers/markers.tsv \
        --output-dir runs/run_002/models/tapestry \
        --epochs 500 \
        --batch-size 64 \
        --lr 3e-4
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tapestry.models.deconvolution import TapestryModel
from tapestry.losses import tapestry_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet_data(data_dir: str) -> dict:
    """Load marker_values, coverage, and ground_truth_y from parquets."""
    data_dir = Path(data_dir)

    mv_df = pd.read_parquet(data_dir / "marker_values.parquet")
    cov_df = pd.read_parquet(data_dir / "coverage.parquet")
    y_df = pd.read_parquet(data_dir / "ground_truth_y.parquet")

    id_cols = ["sample_id", "name", "direction"]
    mv_cols = [c for c in mv_df.columns if c not in id_cols]
    cov_cols = [c for c in cov_df.columns if c not in id_cols]
    ct_cols = [c for c in y_df.columns if c not in id_cols]

    fraction = mv_df[mv_cols].values.astype(np.float32)
    coverage = cov_df[cov_cols].values.astype(np.float32)
    proportions = y_df[ct_cols].values.astype(np.float32)

    fraction = np.nan_to_num(fraction, nan=0.0)

    return {
        "fraction": fraction,
        "coverage": coverage,
        "proportions": proportions,
        "cell_types": ct_cols,
    }


def make_dataloader(data: dict, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader. Derives u, m counts from fraction and coverage."""
    fraction = torch.from_numpy(data["fraction"])
    coverage = torch.from_numpy(data["coverage"])
    proportions = torch.from_numpy(data["proportions"])

    u = fraction * coverage
    m = (1 - fraction) * coverage

    dataset = TensorDataset(u, m, coverage, proportions)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


def load_atlas(atlas_path: str, cell_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load atlas and extract target_ids and reference U-fractions.

    Returns target_ids (M,) and atlas_matrix (M, C).
    """
    atlas_df = pd.read_csv(atlas_path, sep="\t")

    target_ids = atlas_df["target"].map(lambda x: cell_types.index(x)).values

    meta_cols = ["chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
                 "target", "direction", "target_signal", "bg_signal", "snr",
                 "target_total", "bg_total"]
    ct_cols_in_atlas = [c for c in atlas_df.columns if c not in meta_cols]

    atlas_matrix = np.zeros((len(atlas_df), len(cell_types)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        if ct in ct_cols_in_atlas:
            atlas_matrix[:, i] = atlas_df[ct].values.astype(np.float32)

    atlas_matrix = np.nan_to_num(atlas_matrix, nan=0.5)

    return target_ids, atlas_matrix


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cell_types: list[str],
    phi: float = 50.0,
) -> dict:
    """Run validation and compute comprehensive metrics."""
    model.eval()
    all_preds = []
    all_true = []
    all_logits = []
    all_detection = []
    total_loss = 0
    total_details = defaultdict(float)
    n_batches = 0

    for u, m, c, y_true in val_loader:
        u, m, c, y_true = u.to(device), m.to(device), c.to(device), y_true.to(device)

        output = model(u, m, c)
        loss, details = tapestry_loss(output, y_true, u, m, c, phi=phi)

        total_loss += loss.item()
        for k, v in details.items():
            total_details[k] += v
        n_batches += 1

        all_preds.append(output["proportions"].cpu().numpy())
        all_true.append(y_true.cpu().numpy())
        all_logits.append(output.get("logits", output.get("masses", torch.zeros(1))).cpu().numpy())
        all_detection.append(output["detection"].cpu().numpy())

    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    masses = np.concatenate(all_logits)
    detection = np.concatenate(all_detection)

    avg_loss = total_loss / n_batches
    avg_details = {k: v / n_batches for k, v in total_details.items()}

    metrics = {"loss": avg_loss, "details": avg_details}
    eps = 1e-7

    # Overall MAE
    metrics["mae"] = float(np.mean(np.abs(preds - true)))

    # Per-cell-type MAE and R²
    per_ct = {}
    for i, ct in enumerate(cell_types):
        p, t = preds[:, i], true[:, i]
        ct_mae = float(np.mean(np.abs(p - t)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + eps)) if ss_tot > eps else 0.0
        per_ct[ct] = {"mae": ct_mae, "r2": r2}
    metrics["per_cell_type"] = per_ct

    # Log-space metrics (non-zero entries)
    mask = (true > eps) & (preds > eps)
    if mask.sum() > 10:
        log_true = np.log10(true[mask])
        log_pred = np.log10(preds[mask])
        ss_res = np.sum((log_true - log_pred) ** 2)
        ss_tot = np.sum((log_true - log_true.mean()) ** 2)
        metrics["log_r2"] = float(1 - ss_res / (ss_tot + eps)) if ss_tot > eps else 0.0
        coeffs = np.polyfit(log_true, log_pred, 1)
        metrics["log_slope"] = float(coeffs[0])
        metrics["log_intercept"] = float(coeffs[1])
    else:
        metrics["log_r2"] = 0.0
        metrics["log_slope"] = 0.0
        metrics["log_intercept"] = 0.0

    # Presence detection
    threshold = 0.001
    pred_present = preds > threshold
    true_present = true > threshold
    tp = float((pred_present & true_present).sum())
    fp = float((pred_present & ~true_present).sum())
    fn = float((~pred_present & true_present).sum())
    metrics["presence_precision"] = tp / (tp + fp + eps)
    metrics["presence_recall"] = tp / (tp + fn + eps)
    metrics["presence_f1"] = (
        2 * metrics["presence_precision"] * metrics["presence_recall"]
        / (metrics["presence_precision"] + metrics["presence_recall"] + eps)
    )

    # Model internals
    logits = np.concatenate(all_logits)
    metrics["logit_mean"] = float(logits.mean())
    metrics["logit_std"] = float(logits.std())
    metrics["detection_mean"] = float(detection.mean())
    metrics["prop_min"] = float(preds.min())
    metrics["prop_max"] = float(preds.max())
    metrics["prop_mean"] = float(preds.mean())
    metrics["n_nonzero_per_sample"] = float((preds > threshold).sum(axis=1).mean())

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(fig, path: str):
    """Save plotly figure as lightweight CDN-linked HTML."""
    fig.write_html(path, include_plotlyjs="cdn")


def plot_training_history(history: dict, output_dir: str, cell_types: list[str]):
    """Generate comprehensive training plots from history dict."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    out = Path(output_dir)

    # 1. Loss curves (total + per component)
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Total Loss", "Proportion + NLL Loss", "Detection + Sparsity", "Learning Rate"
    ))
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["proportion_loss"], name="Proportion", line=dict(width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["nll"], name="NLL", line=dict(width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["detection_loss"], name="Detection", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["sparsity"], name="Sparsity", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["lr"], name="LR", line=dict(width=1)), row=2, col=2)
    fig.update_layout(height=800, title="Training Loss Components")
    fig.update_yaxes(type="log", row=2, col=2)
    save_plot(fig, str(out / "training_losses.html"))

    # 2. Key validation metrics
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "MAE", "Log-R² / Slope", "Presence F1", "Gradient Norm"
    ))
    fig.add_trace(go.Scatter(x=epochs, y=history["mae"], name="MAE", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["log_r2"], name="Log-R²", line=dict(width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["log_slope"], name="Log-slope", line=dict(width=1, dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=[1.0]*len(epochs), name="Target slope", line=dict(width=1, dash="dot", color="grey")), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["presence_f1"], name="F1", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["grad_norm"], name="Grad norm", line=dict(width=1)), row=2, col=2)
    fig.update_layout(height=800, title="Validation Metrics")
    save_plot(fig, str(out / "validation_metrics.html"))

    # 3. Per-cell-type MAE evolution
    fig = go.Figure()
    for ct in cell_types:
        key = f"ct_mae_{ct}"
        if key in history:
            fig.add_trace(go.Scatter(x=epochs, y=history[key], name=ct, line=dict(width=1)))
    fig.update_layout(height=600, title="Per-Cell-Type MAE", xaxis_title="Epoch", yaxis_title="MAE")
    save_plot(fig, str(out / "per_celltype_mae.html"))

    # 4. Per-cell-type R² evolution
    fig = go.Figure()
    for ct in cell_types:
        key = f"ct_r2_{ct}"
        if key in history:
            fig.add_trace(go.Scatter(x=epochs, y=history[key], name=ct, line=dict(width=1)))
    fig.update_layout(height=600, title="Per-Cell-Type R²", xaxis_title="Epoch", yaxis_title="R²")
    save_plot(fig, str(out / "per_celltype_r2.html"))

    # 5. Model internals
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Mass Distribution", "Detection Probability", "Non-Zero Types per Sample"
    ))
    fig.add_trace(go.Scatter(x=epochs, y=history["logit_mean"], name="Mean", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["logit_std"], name="Std", line=dict(width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["detection_mean"], name="Mean det prob", line=dict(width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["n_nonzero_per_sample"], name="N>0.001", line=dict(width=1)), row=1, col=3)
    fig.update_layout(height=400, title="Model Internals")
    save_plot(fig, str(out / "model_internals.html"))

    logger.info("Saved training plots to %s", out)


def plot_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cell_types: list[str],
    output_dir: str,
    max_samples: int = 5000,
):
    """Generate predicted vs true scatter plots at the current model state."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for u, m, c, y_true in val_loader:
            u, m, c, y_true = u.to(device), m.to(device), c.to(device), y_true.to(device)
            output = model(u, m, c)
            all_preds.append(output["proportions"].cpu().numpy())
            all_true.append(y_true.cpu().numpy())

    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)

    if len(preds) > max_samples:
        idx = np.random.default_rng(42).choice(len(preds), max_samples, replace=False)
        preds, true = preds[idx], true[idx]

    C = len(cell_types)
    cols = 4
    rows = (C + cols - 1) // cols

    # Linear scale
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=cell_types)
    for i, ct in enumerate(cell_types):
        r, c = i // cols + 1, i % cols + 1
        fig.add_trace(go.Scatter(
            x=true[:, i], y=preds[:, i], mode="markers",
            marker=dict(size=2, opacity=0.3), name=ct, showlegend=False,
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="red", dash="dash", width=1), showlegend=False,
        ), row=r, col=c)
        fig.update_xaxes(title_text="True", range=[0, max(0.5, true[:, i].max() * 1.1)], row=r, col=c)
        fig.update_yaxes(title_text="Pred", range=[0, max(0.5, preds[:, i].max() * 1.1)], row=r, col=c)

    fig.update_layout(height=300 * rows, width=1200, title="Predicted vs True (Linear)")
    save_plot(fig, str(Path(output_dir) / "pred_vs_true_linear.html"))

    # Log scale
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=cell_types)
    eps = 1e-6
    for i, ct in enumerate(cell_types):
        r, c = i // cols + 1, i % cols + 1
        mask = (true[:, i] > eps) & (preds[:, i] > eps)
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=np.log10(true[:, i][mask] + eps), y=np.log10(preds[:, i][mask] + eps),
                mode="markers", marker=dict(size=2, opacity=0.3), name=ct, showlegend=False,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=[-5, 0], y=[-5, 0], mode="lines",
                line=dict(color="red", dash="dash", width=1), showlegend=False,
            ), row=r, col=c)
        fig.update_xaxes(title_text="log₁₀(true)", row=r, col=c)
        fig.update_yaxes(title_text="log₁₀(pred)", row=r, col=c)

    fig.update_layout(height=300 * rows, width=1200, title="Predicted vs True (Log)")
    save_plot(fig, str(Path(output_dir) / "pred_vs_true_log.html"))

    logger.info("Saved prediction scatter plots to %s", output_dir)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    logger.info("Loading training data from %s", args.train_dir)
    train_data = load_parquet_data(args.train_dir)
    logger.info("Loading validation data from %s", args.eval_dir)
    val_data = load_parquet_data(args.eval_dir)

    cell_types = train_data["cell_types"]
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)
    logger.info("Training: %d samples, %d markers", *train_data["fraction"].shape)
    logger.info("Validation: %d samples, %d markers", *val_data["fraction"].shape)

    # Load atlas
    target_ids, atlas_matrix = load_atlas(args.atlas, cell_types)
    num_markers = train_data["fraction"].shape[1]
    num_cell_types = len(cell_types)

    assert num_markers == atlas_matrix.shape[0], (
        f"Marker count mismatch: data has {num_markers}, atlas has {atlas_matrix.shape[0]}"
    )

    # Create model
    model = TapestryModel(
        num_markers=num_markers,
        num_cell_types=num_cell_types,
        target_ids=target_ids,
        atlas=atlas_matrix,
        feature_dim=args.feature_dim,
        l1_num_heads=args.l1_heads,
        l1_num_layers=args.l1_layers,
        l2_num_heads=args.l2_heads,
        l2_num_layers=args.l2_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    # Data loaders
    train_loader = make_dataloader(train_data, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_data, args.batch_size, shuffle=False)

    # Optimiser with cosine warmup then ReduceLROnPlateau
    optimiser = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Warmup for first 5% of epochs, then ReduceLROnPlateau
    warmup_epochs = 10
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimiser, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=args.patience // 3,
        min_lr=1e-6, verbose=True,
    )

    # Training state
    best_val_loss = float("inf")
    best_log_r2 = float("-inf")
    best_mae = float("inf")
    best_state = None
    patience_counter = 0
    history = defaultdict(list)

    # Save config
    config = vars(args)
    config["cell_types"] = list(cell_types)
    config["num_markers"] = num_markers
    config["total_params"] = total_params
    config["warmup_epochs"] = warmup_epochs
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_details = defaultdict(float)
        grad_norms = []
        n_batches = 0
        epoch_start = time.time()

        for batch_idx, (u, m, c, y_true) in enumerate(train_loader):
            u, m, c, y_true = u.to(device), m.to(device), c.to(device), y_true.to(device)

            output = model(u, m, c)
            loss, details = tapestry_loss(
                output, y_true, u, m, c,
                phi=args.phi,
                concentration_weighting=True,
            )

            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_norms.append(grad_norm.item())
                optimiser.step()
                optimiser.zero_grad()

            train_loss += loss.item()
            for k, v in details.items():
                train_details[k] += v
            n_batches += 1

        train_loss /= n_batches
        for k in train_details:
            train_details[k] /= n_batches
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        epoch_time = time.time() - epoch_start

        # LR scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        # plateau_scheduler steps after validation (below)

        # Validation
        val_metrics = validate(model, val_loader, device, cell_types, phi=args.phi)

        # Step plateau scheduler on val loss (after warmup)
        if epoch >= warmup_epochs:
            plateau_scheduler.step(val_metrics["loss"])

        lr = optimiser.param_groups[0]["lr"]

        # --- Logging ---
        logger.info(
            "Epoch %d/%d (%.0fs) lr=%.2e grad=%.3f",
            epoch + 1, args.epochs, epoch_time, lr, avg_grad_norm,
        )
        logger.info(
            "  TRAIN total=%.4f prop=%.4f log_prop=%.4f nll=%.4f det=%.4f sparse=%.4f",
            train_loss,
            train_details.get("proportion_loss", 0),
            train_details.get("log_proportion_loss", 0),
            train_details.get("nll", 0),
            train_details.get("detection_loss", 0),
            train_details.get("sparsity", 0),
        )
        logger.info(
            "  VAL   total=%.4f mae=%.4f log_r2=%.4f slope=%.3f intercept=%.3f pres_f1=%.3f",
            val_metrics["loss"], val_metrics["mae"],
            val_metrics["log_r2"], val_metrics["log_slope"], val_metrics["log_intercept"],
            val_metrics["presence_f1"],
        )
        logger.info(
            "  MODEL logit=%.3f±%.3f det_prob=%.3f props=[%.4f,%.4f] n_active=%.1f",
            val_metrics["logit_mean"], val_metrics["logit_std"],
            val_metrics["detection_mean"],
            val_metrics["prop_min"], val_metrics["prop_max"],
            val_metrics["n_nonzero_per_sample"],
        )

        # Per-cell-type detail every 10 epochs
        if (epoch + 1) % 10 == 0:
            for ct, ct_m in val_metrics["per_cell_type"].items():
                logger.info("    %s: MAE=%.4f R2=%.4f", ct, ct_m["mae"], ct_m["r2"])

        # --- History ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["mae"].append(val_metrics["mae"])
        history["log_r2"].append(val_metrics["log_r2"])
        history["log_slope"].append(val_metrics["log_slope"])
        history["log_intercept"].append(val_metrics["log_intercept"])
        history["presence_f1"].append(val_metrics["presence_f1"])
        history["presence_precision"].append(val_metrics["presence_precision"])
        history["presence_recall"].append(val_metrics["presence_recall"])
        history["lr"].append(lr)
        history["grad_norm"].append(avg_grad_norm)
        history["proportion_loss"].append(train_details.get("proportion_loss", 0))
        history["log_proportion_loss"].append(train_details.get("log_proportion_loss", 0))
        history["nll"].append(train_details.get("nll", 0))
        history["detection_loss"].append(train_details.get("detection_loss", 0))
        history["sparsity"].append(train_details.get("sparsity", 0))
        history["logit_mean"].append(val_metrics["logit_mean"])
        history["logit_std"].append(val_metrics["logit_std"])
        history["detection_mean"].append(val_metrics["detection_mean"])
        history["n_nonzero_per_sample"].append(val_metrics["n_nonzero_per_sample"])

        for ct, ct_m in val_metrics["per_cell_type"].items():
            history[f"ct_mae_{ct}"].append(ct_m["mae"])
            history[f"ct_r2_{ct}"].append(ct_m["r2"])

        # --- Save history incrementally ---
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

        # --- Model selection ---
        # Improvement on ANY key metric resets patience
        improved = False
        improvement_reasons = []

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            improved = True
            improvement_reasons.append(f"loss={best_val_loss:.4f}")

        if val_metrics["log_r2"] > best_log_r2 + 1e-4:
            best_log_r2 = val_metrics["log_r2"]
            improved = True
            improvement_reasons.append(f"log_r2={best_log_r2:.4f}")

        if val_metrics["mae"] < best_mae - 1e-5:
            best_mae = val_metrics["mae"]
            improved = True
            improvement_reasons.append(f"mae={best_mae:.4f}")

        if improved:
            patience_counter = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "val_metrics": {k: v for k, v in val_metrics.items() if k != "details"},
                "best_val_loss": best_val_loss,
                "best_log_r2": best_log_r2,
                "best_mae": best_mae,
            }
            torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))
            logger.info("  >> New best: %s", ", ".join(improvement_reasons))
        else:
            patience_counter += 1
            logger.info(
                "  No improvement (%d/%d). Best: loss=%.4f log_r2=%.4f mae=%.4f",
                patience_counter, args.patience,
                best_val_loss, best_log_r2, best_mae,
            )

        # Early stopping: only after warmup, and check LR hasn't bottomed out
        if epoch >= warmup_epochs and patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

        # Periodic checkpoint + plots
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "history": dict(history),
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch + 1}.pt"),
            )
            plot_training_history(dict(history), args.output_dir, cell_types)
            plot_predictions(model, val_loader, device, cell_types, args.output_dir)

    # --- Final outputs ---

    # Save final model
    torch.save(
        {"model_state_dict": model.state_dict(), "history": dict(history)},
        os.path.join(args.output_dir, "final_model.pt"),
    )

    # Load best model for plots
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        logger.info(
            "Best model from epoch %d: loss=%.4f mae=%.4f log_r2=%.4f",
            best_state["epoch"] + 1, best_val_loss, best_mae, best_log_r2,
        )

    # Generate plots
    plot_training_history(dict(history), args.output_dir, cell_types)
    plot_predictions(model, val_loader, device, cell_types, args.output_dir)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train TapestryModel")

    # Data
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--atlas", required=True, help="Path to markers.tsv")
    parser.add_argument("--output-dir", required=True)

    # Model architecture
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--l1-heads", type=int, default=4)
    parser.add_argument("--l1-layers", type=int, default=2)
    parser.add_argument("--l2-heads", type=int, default=4)
    parser.add_argument("--l2-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=50)

    # Loss
    parser.add_argument("--phi", type=float, default=50.0,
                        help="Beta-binomial concentration parameter")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
