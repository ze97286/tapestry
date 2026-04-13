#!/usr/bin/env python3
"""Visualise training and evaluation data distributions.

Generates comprehensive plots for paper and interactive exploration:
  - Proportion distributions per cell type (box + violin)
  - Proportion heatmap (samples × cell types)
  - Coverage distributions (per-marker and per-sample)
  - Strategy breakdown
  - Correlation matrix between cell types
  - Concentration range histograms for OAC and T-cells
  - Zero-fraction analysis
  - Depth distribution
  - Marker value (U-fraction) distributions
  - Dilution series ground truth

All plots are saved as both interactive HTML (plotly) and high-resolution PNG.

Usage:
    python scripts/visualise_training_data.py \
        --data-dir runs/run_002/training/train \
        --proportions runs/run_002/training/train_proportions.csv \
        --output-dir runs/run_002/training/plots \
        --dataset-name train
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

PLOTLY_TEMPLATE = "plotly_white"


def save_fig(fig, output_dir: Path, name: str):
    """Save figure as lightweight CDN-linked HTML.

    For paper-quality PNGs, download the HTML files and use plotly's
    write_image locally where kaleido works reliably.
    """
    fig.update_layout(template=PLOTLY_TEMPLATE)
    fig.write_html(str(output_dir / f"{name}.html"), include_plotlyjs="cdn")
    logger.info("Saved %s.html", name)


# ---------------------------------------------------------------------------
# Proportion plots
# ---------------------------------------------------------------------------

def plot_proportion_boxplots(y_df: pd.DataFrame, cell_types: list[str],
                             output_dir: Path, dataset_name: str):
    """Box plot of ground-truth proportions per cell type (pre-computed stats)."""
    fig = go.Figure()
    for ct in cell_types:
        vals = y_df[ct].values
        vals_sorted = np.sort(vals)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lo_fence = max(vals.min(), q1 - 1.5 * iqr)
        hi_fence = min(vals.max(), q3 + 1.5 * iqr)
        fig.add_trace(go.Box(
            lowerfence=[lo_fence], q1=[q1], median=[med], q3=[q3],
            upperfence=[hi_fence], name=ct,
            line=dict(width=1),
        ))

    fig.update_layout(
        title=f"Ground-Truth Proportion Distributions ({dataset_name})",
        yaxis_title="Proportion",
        xaxis_title="Cell Type",
        xaxis_tickangle=-45,
        height=600,
    )
    save_fig(fig, output_dir, f"{dataset_name}_proportion_boxplots")


def plot_proportion_violin(y_df: pd.DataFrame, cell_types: list[str],
                           output_dir: Path, dataset_name: str):
    """Violin-style plot using pre-binned histograms on log scale."""
    fig = make_subplots(rows=1, cols=1)
    for ct in cell_types:
        vals = y_df[ct].values
        vals_nonzero = vals[vals > 0]
        if len(vals_nonzero) > 0:
            log_vals = np.log10(vals_nonzero + 1e-6)
            counts, edges = np.histogram(log_vals, bins=50)
            centres = (edges[:-1] + edges[1:]) / 2
            fig.add_trace(go.Scatter(
                x=centres, y=counts, name=ct, mode="lines",
                line=dict(width=1.5),
            ))

    fig.update_layout(
        title=f"Proportion Distributions — Log Scale ({dataset_name})",
        xaxis_title="log₁₀(proportion)",
        yaxis_title="Count",
        height=600,
    )
    save_fig(fig, output_dir, f"{dataset_name}_proportion_violin_log")


def plot_proportion_heatmap(y_df: pd.DataFrame, cell_types: list[str],
                            output_dir: Path, dataset_name: str,
                            max_samples: int = 2000):
    """Heatmap of proportions (samples × cell types), subsampled if needed."""
    if len(y_df) > max_samples:
        y_sub = y_df.sample(max_samples, random_state=42)
    else:
        y_sub = y_df

    # Sort by dominant cell type for visual clarity
    dominant = y_sub[cell_types].idxmax(axis=1)
    y_sorted = y_sub.assign(_dominant=dominant).sort_values(["_dominant"] + cell_types)
    matrix = y_sorted[cell_types].values

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=cell_types,
        y=list(range(len(matrix))),
        colorscale="RdYlBu_r",
        zmin=0, zmax=0.8,
        colorbar=dict(title="Proportion"),
    ))
    fig.update_layout(
        title=f"Proportion Heatmap ({dataset_name}, n={len(matrix)})",
        xaxis_title="Cell Type",
        yaxis_title="Sample",
        xaxis_tickangle=-45,
        height=max(600, len(matrix) * 0.3),
    )
    save_fig(fig, output_dir, f"{dataset_name}_proportion_heatmap")


# ---------------------------------------------------------------------------
# Coverage plots
# ---------------------------------------------------------------------------

def plot_coverage_distributions(coverage: np.ndarray, output_dir: Path,
                                dataset_name: str):
    """Per-marker and per-sample coverage distributions (pre-binned)."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Per-Marker Coverage", "Per-Sample Mean Coverage"))

    # Per-marker: bin all values with numpy
    flat_nonzero = coverage[coverage > 0].flatten()
    counts, edges = np.histogram(flat_nonzero, bins=100,
                                 range=(0, np.percentile(flat_nonzero, 99)))
    centres = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=centres, y=counts, name="Per-marker",
        marker_color="steelblue", opacity=0.8, width=centres[1] - centres[0],
    ), row=1, col=1)

    # Per-sample mean
    sample_means = coverage.mean(axis=1)
    counts2, edges2 = np.histogram(sample_means, bins=50)
    centres2 = (edges2[:-1] + edges2[1:]) / 2
    fig.add_trace(go.Bar(
        x=centres2, y=counts2, name="Per-sample mean",
        marker_color="coral", opacity=0.8, width=centres2[1] - centres2[0],
    ), row=1, col=2)

    fig.update_xaxes(title_text="Coverage (reads)", row=1, col=1)
    fig.update_xaxes(title_text="Mean Coverage", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(
        title=f"Coverage Distributions ({dataset_name})",
        showlegend=False, height=500,
    )
    save_fig(fig, output_dir, f"{dataset_name}_coverage_distributions")


def plot_marker_value_distributions(marker_values: np.ndarray, output_dir: Path,
                                    dataset_name: str):
    """Distribution of observed U-fractions across all markers (pre-binned)."""
    flat_valid = marker_values[~np.isnan(marker_values)].flatten()
    counts, edges = np.histogram(flat_valid, bins=100, range=(0, 1))
    centres = (edges[:-1] + edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centres, y=counts, name="U-fraction",
        marker_color="mediumpurple", opacity=0.8, width=0.01,
    ))
    fig.update_layout(
        title=f"Marker Value (U-Fraction) Distribution ({dataset_name})",
        xaxis_title="U-Fraction",
        yaxis_title="Count",
        height=500,
    )
    save_fig(fig, output_dir, f"{dataset_name}_marker_value_distribution")


def plot_zero_coverage_rate(coverage: np.ndarray, output_dir: Path,
                            dataset_name: str):
    """Percentage of markers with zero coverage per sample (pre-binned)."""
    zero_pct = (coverage == 0).mean(axis=1) * 100
    counts, edges = np.histogram(zero_pct, bins=50)
    centres = (edges[:-1] + edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centres, y=counts,
        marker_color="indianred", opacity=0.8, width=centres[1] - centres[0] if len(centres) > 1 else 1,
    ))
    fig.update_layout(
        title=f"Zero-Coverage Marker Rate ({dataset_name})",
        xaxis_title="% Markers with Zero Coverage",
        yaxis_title="Count",
        height=500,
    )
    save_fig(fig, output_dir, f"{dataset_name}_zero_coverage_rate")


# ---------------------------------------------------------------------------
# Strategy and depth plots
# ---------------------------------------------------------------------------

def plot_strategy_breakdown(props_df: pd.DataFrame, output_dir: Path,
                            dataset_name: str):
    """Pie chart of proportion strategy distribution."""
    if "strategy" not in props_df.columns:
        logger.info("No strategy column, skipping strategy breakdown")
        return

    counts = props_df["strategy"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        textinfo="label+percent",
        hole=0.3,
    ))
    fig.update_layout(
        title=f"Proportion Strategy Distribution ({dataset_name})",
        height=500,
    )
    save_fig(fig, output_dir, f"{dataset_name}_strategy_breakdown")


def plot_depth_distribution(props_df: pd.DataFrame, output_dir: Path,
                            dataset_name: str):
    """Distribution of target sequencing depths."""
    if "depth" not in props_df.columns:
        logger.info("No depth column, skipping depth distribution")
        return

    log_depths = np.log10(props_df["depth"].values + 1)
    counts, edges = np.histogram(log_depths, bins=50)
    centres = (edges[:-1] + edges[1:]) / 2
    tick_vals = [3, 3.5, 4, 4.5, 5, 5.5]
    tick_text = [f"{10**v:.0f}" for v in tick_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centres, y=counts,
        marker_color="seagreen", opacity=0.8,
        width=centres[1] - centres[0] if len(centres) > 1 else 0.1,
    ))
    fig.update_layout(
        title=f"Target Depth Distribution ({dataset_name})",
        xaxis_title="Target Total Reads",
        yaxis_title="Count",
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text),
        height=500,
    )
    save_fig(fig, output_dir, f"{dataset_name}_depth_distribution")


# ---------------------------------------------------------------------------
# Correlation and zero-fraction
# ---------------------------------------------------------------------------

def plot_cell_type_correlation(y_df: pd.DataFrame, cell_types: list[str],
                               output_dir: Path, dataset_name: str):
    """Correlation matrix between cell type proportions."""
    corr = y_df[cell_types].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=cell_types,
        y=cell_types,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(
        title=f"Cell Type Proportion Correlations ({dataset_name})",
        xaxis_tickangle=-45,
        height=700, width=800,
    )
    save_fig(fig, output_dir, f"{dataset_name}_correlation_matrix")


def plot_zero_fraction_per_type(y_df: pd.DataFrame, cell_types: list[str],
                                output_dir: Path, dataset_name: str):
    """Bar chart: percentage of samples where each cell type is exactly zero."""
    zero_pcts = [(y_df[ct] == 0).mean() * 100 for ct in cell_types]

    fig = go.Figure(go.Bar(
        x=cell_types, y=zero_pcts,
        marker_color="steelblue", opacity=0.8,
        text=[f"{p:.1f}%" for p in zero_pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Zero-Proportion Rate per Cell Type ({dataset_name})",
        xaxis_title="Cell Type",
        yaxis_title="% Samples with Proportion = 0",
        xaxis_tickangle=-45,
        height=500,
    )
    save_fig(fig, output_dir, f"{dataset_name}_zero_fraction_per_type")


# ---------------------------------------------------------------------------
# Low-concentration detail (OAC and T-cells)
# ---------------------------------------------------------------------------

def plot_concentration_histograms(y_df: pd.DataFrame, output_dir: Path,
                                  dataset_name: str,
                                  types_of_interest: list[str] = None):
    """Histograms of concentration for key cell types, with log-scale bins."""
    if types_of_interest is None:
        types_of_interest = ["OAC", "T-cells", "Hepatocytes", "Colon"]

    available = [ct for ct in types_of_interest if ct in y_df.columns]
    n = len(available)
    if n == 0:
        return

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=available)

    for i, ct in enumerate(available):
        r, c = i // cols + 1, i % cols + 1
        vals = y_df[ct].values
        vals_pos = vals[vals > 0]

        if len(vals_pos) > 0:
            log_vals = np.log10(vals_pos + 1e-6)
            counts, edges = np.histogram(log_vals, bins=50)
            centres = (edges[:-1] + edges[1:]) / 2
            fig.add_trace(go.Bar(
                x=centres, y=counts,
                marker_color="steelblue", opacity=0.8,
                name=ct, showlegend=False,
                width=centres[1] - centres[0] if len(centres) > 1 else 0.1,
            ), row=r, col=c)

        fig.update_xaxes(title_text="log₁₀(proportion)", row=r, col=c)
        fig.update_yaxes(title_text="Count", row=r, col=c)

    fig.update_layout(
        title=f"Concentration Distributions — Key Cell Types ({dataset_name})",
        height=400 * rows,
    )
    save_fig(fig, output_dir, f"{dataset_name}_concentration_histograms")


# ---------------------------------------------------------------------------
# Dilution series plots
# ---------------------------------------------------------------------------

def plot_dilution_ground_truth(props_df: pd.DataFrame, target_type: str,
                               output_dir: Path, dataset_name: str):
    """Scatter plot of dilution ground truth: target proportion vs depth."""
    if target_type not in props_df.columns:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=props_df["depth"],
        y=props_df[target_type],
        mode="markers",
        marker=dict(size=4, opacity=0.5, color="steelblue"),
    ))
    fig.update_layout(
        title=f"{target_type} Dilution Series Ground Truth ({dataset_name})",
        xaxis_title="Target Depth",
        yaxis_title=f"{target_type} Proportion",
        xaxis_type="log",
        yaxis_type="log",
        height=500,
    )
    save_fig(fig, output_dir, f"{dataset_name}_{target_type}_dilution_ground_truth")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise training/evaluation data.")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with marker_values.parquet, coverage.parquet, ground_truth_y.parquet")
    parser.add_argument("--proportions", default=None,
                        help="Path to proportions CSV (for strategy/depth plots)")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--dataset-name", default="dataset", help="Name for plot titles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data from %s...", data_dir)
    mv_df = pd.read_parquet(data_dir / "marker_values.parquet")
    cov_df = pd.read_parquet(data_dir / "coverage.parquet")
    y_df = pd.read_parquet(data_dir / "ground_truth_y.parquet")

    # Extract arrays (drop sample_id column if present)
    id_cols = ["sample_id", "name", "direction"]
    mv_cols = [c for c in mv_df.columns if c not in id_cols]
    cov_cols = [c for c in cov_df.columns if c not in id_cols]
    ct_cols = [c for c in y_df.columns if c not in id_cols]

    marker_values = mv_df[mv_cols].values
    coverage = cov_df[cov_cols].values
    cell_types = ct_cols

    logger.info("Loaded %d samples, %d markers, %d cell types",
                len(marker_values), marker_values.shape[1], len(cell_types))

    # --- Proportion plots ---
    plot_proportion_boxplots(y_df, cell_types, output_dir, args.dataset_name)
    plot_proportion_violin(y_df, cell_types, output_dir, args.dataset_name)
    plot_proportion_heatmap(y_df, cell_types, output_dir, args.dataset_name)
    plot_cell_type_correlation(y_df, cell_types, output_dir, args.dataset_name)
    plot_zero_fraction_per_type(y_df, cell_types, output_dir, args.dataset_name)
    plot_concentration_histograms(y_df, output_dir, args.dataset_name)

    # --- Coverage and marker value plots ---
    plot_coverage_distributions(coverage, output_dir, args.dataset_name)
    plot_marker_value_distributions(marker_values, output_dir, args.dataset_name)
    plot_zero_coverage_rate(coverage, output_dir, args.dataset_name)

    # --- Strategy and depth (from proportions CSV) ---
    if args.proportions and Path(args.proportions).exists():
        props_df = pd.read_csv(args.proportions)
        plot_strategy_breakdown(props_df, output_dir, args.dataset_name)
        plot_depth_distribution(props_df, output_dir, args.dataset_name)

        # Dilution plots if this is a dilution dataset
        if "dilution" in args.dataset_name.lower() or "oac" in args.dataset_name.lower():
            plot_dilution_ground_truth(props_df, "OAC", output_dir, args.dataset_name)
        if "tcell" in args.dataset_name.lower():
            plot_dilution_ground_truth(props_df, "T-cells", output_dir, args.dataset_name)

    logger.info("All plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
