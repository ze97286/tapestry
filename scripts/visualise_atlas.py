#!/usr/bin/env python
"""Visualise the marker atlas: SNR distribution, marker density, chromosomal
spread, block sizes, and per-cell-type heatmaps."""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)


def plot_snr_distribution(df: pd.DataFrame, out_dir: Path):
    """Box + strip plot of SNR per cell type."""
    fig, ax = plt.subplots(figsize=(14, 6))
    cell_types = sorted(df["target"].unique())
    data = [df[df["target"] == ct]["snr"].values for ct in cell_types]
    counts = [len(d) for d in data]
    labels = [f"{ct}\n(n={n})" for ct, n in zip(cell_types, counts)]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)

    for i, d in enumerate(data):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, alpha=0.3, s=8, c="#DD8452", zorder=3)

    ax.set_ylabel("Signal-to-Noise Ratio")
    ax.set_title("Marker SNR by Cell Type")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / "snr_by_celltype.png", dpi=150)
    plt.close(fig)
    logger.info("Saved snr_by_celltype.png")


def plot_chromosome_distribution(df: pd.DataFrame, out_dir: Path):
    """Stacked bar chart of marker counts per chromosome per cell type."""
    chrom_order = [f"chr{i}" for i in range(1, 23)]
    cell_types = sorted(df["target"].unique())

    counts = df.groupby(["chr", "target"]).size().unstack(fill_value=0)
    counts = counts.reindex(chrom_order).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Number of markers")
    ax.set_title("Marker Distribution Across Chromosomes")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / "markers_by_chromosome.png", dpi=150)
    plt.close(fig)
    logger.info("Saved markers_by_chromosome.png")


def plot_block_sizes(df: pd.DataFrame, out_dir: Path):
    """Histogram of block sizes (number of CpGs and bp length)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["n_cpgs"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Number of CpGs per marker")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"CpG Count Distribution (median={df['n_cpgs'].median():.0f})")
    axes[0].axvline(df["n_cpgs"].median(), color="red", linestyle="--", alpha=0.7)

    bp_length = df["end"] - df["start"]
    axes[1].hist(bp_length, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Block length (bp)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Block Length Distribution (median={bp_length.median():.0f} bp)")
    axes[1].axvline(bp_length.median(), color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_dir / "block_sizes.png", dpi=150)
    plt.close(fig)
    logger.info("Saved block_sizes.png")


def plot_signal_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap of U-fraction for each marker × each cell type."""
    ct_cols = [c for c in df.columns if c not in [
        "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
        "target", "direction", "target_signal", "bg_signal", "snr",
        "target_total", "bg_total",
    ]]

    # Sort markers by target cell type, then by SNR within each
    df_sorted = df.sort_values(["target", "snr"], ascending=[True, False])

    matrix = df_sorted[ct_cols].values
    targets = df_sorted["target"].values

    fig, ax = plt.subplots(figsize=(12, max(10, len(df_sorted) * 0.008)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(ct_cols)))
    ax.set_xticklabels(ct_cols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Markers (grouped by target)")
    ax.set_title("U-fraction Heatmap: Markers × Cell Types")

    # Add target cell type labels on the left
    prev_target = None
    boundaries = []
    for i, t in enumerate(targets):
        if t != prev_target:
            boundaries.append((i, t))
            prev_target = t

    for start_idx, target_name in boundaries:
        ax.axhline(start_idx - 0.5, color="black", linewidth=0.5, alpha=0.5)
        ax.text(-0.5, start_idx + 5, target_name, fontsize=6, ha="right",
                va="top", transform=ax.get_yaxis_transform())

    plt.colorbar(im, ax=ax, label="U-fraction", shrink=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "signal_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved signal_heatmap.png")


def plot_target_vs_background(df: pd.DataFrame, out_dir: Path):
    """Scatter plot of target signal vs max single non-target signal."""
    ct_cols = [c for c in df.columns if c not in [
        "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
        "target", "direction", "target_signal", "bg_signal", "snr",
        "target_total", "bg_total",
    ]]

    cell_types = sorted(df["target"].unique())
    n_ct = len(cell_types)
    cols = 4
    rows = (n_ct + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten()

    for i, target in enumerate(cell_types):
        ax = axes[i]
        sub = df[df["target"] == target]
        non_target = [c for c in ct_cols if c != target]
        max_bg = sub[non_target].max(axis=1)

        ax.scatter(sub["target_signal"], max_bg, alpha=0.4, s=10, c="#4C72B0")
        ax.set_xlabel("Target signal")
        ax.set_ylabel("Max non-target signal")
        ax.set_title(f"{target} (n={len(sub)})", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.02, 0.25)
        ax.axhline(0.2, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Target Signal vs Max Non-Target Signal", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "target_vs_background.png", dpi=150)
    plt.close(fig)
    logger.info("Saved target_vs_background.png")


def plot_coverage_distribution(df: pd.DataFrame, out_dir: Path):
    """Box plot of total coverage in target and background."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cell_types = sorted(df["target"].unique())

    for idx, (col, title) in enumerate([
        ("target_total", "Target Coverage"),
        ("bg_total", "Background Coverage"),
    ]):
        data = [df[df["target"] == ct][col].values for ct in cell_types]
        labels = [ct for ct in cell_types]
        axes[idx].boxplot(data, labels=labels, showfliers=False)
        axes[idx].set_ylabel("Total reads")
        axes[idx].set_title(title)
        axes[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(out_dir / "coverage_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved coverage_distribution.png")


def main():
    parser = argparse.ArgumentParser(description="Visualise marker atlas.")
    parser.add_argument("--markers", required=True, help="Path to markers.tsv")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    df = pd.read_csv(args.markers, sep="\t")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded %d markers across %d cell types", len(df), df["target"].nunique())

    plot_snr_distribution(df, out_dir)
    plot_chromosome_distribution(df, out_dir)
    plot_block_sizes(df, out_dir)
    plot_signal_heatmap(df, out_dir)
    plot_target_vs_background(df, out_dir)
    plot_coverage_distribution(df, out_dir)

    logger.info("All plots saved to %s", out_dir)


if __name__ == "__main__":
    main()
