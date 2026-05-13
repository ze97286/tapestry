#!/usr/bin/env python3
"""Clinical evaluation of tapestry deconvolution on OAC cohorts.

Generates:
  1. Stacked bar chart of all cell-type proportions per sample
  2. Proportion heatmap across cohort
  3. OAC waterfall plot (Immonly - ScrBsl delta) coloured by clinical benefit
  4. Kaplan-Meier survival curves split by OAC change direction
  5. OAC vs ichorCNA scatter plots per timepoint
  6. ichorCNA calibration (optional)
  7. Per-cell-type correlation with ichorCNA tumour fraction
  8. Full deconvolution table export

Usage:
    python scripts/clinical_evaluation.py \
        --predictions runs/run_002/predictions/AB_predictions.csv \
        --clinical-file data/AB_patient_summary.csv \
        --ichorcna-file data/cfDNA_tumour_fraction_ichorCNA.json \
        --output-dir runs/run_002/evaluation/AB \
        --cohort AB
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

logger = logging.getLogger(__name__)


META_COLS = {
    "sample", "cohort", "mean_coverage", "n_markers_with_coverage",
    "patient_id", "timepoint", "is_control", "unknown_mag",
    "unknown_lambda", "lambda_unknown", "unknown_residual_norm", "unknown_basis_mode",
    "control_crossfit_fold", "unknown_n_components",
    "unknown_fit_excluded_cell_types",
}


def infer_cell_type_columns(df: pd.DataFrame) -> list[str]:
    """Infer production cell-type columns from a combined prediction CSV.

    Prediction files may also contain raw/NNLS columns, lambda-path diagnostics,
    and unknown-channel metadata.  Only bare numeric columns corresponding to
    production cell-type estimates should be used for stacked bars, heatmaps,
    deltas, and clinical plots.
    """
    suffixed = (
        "_detection", "_raw", "_nnls", "_binomial",
        "_residual_norm", "_lambda_path",
    )
    blocked_substrings = ("_aug_lam_", "_lam_")
    cell_types = []
    for column in df.columns:
        if column in META_COLS:
            continue
        if column.endswith(suffixed):
            continue
        if any(token in column for token in blocked_substrings):
            continue
        if column.startswith("unknown_") or column.startswith("residual_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        cell_types.append(column)
    return cell_types

# Optional survival analysis
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


def save_fig(fig, path: str):
    fig.write_html(path + ".html", include_plotlyjs="cdn")
    logger.info("Saved %s.html", path)


# ---------------------------------------------------------------------------
# Sample metadata parsing
# ---------------------------------------------------------------------------

def parse_sample_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract patient_id, timepoint from sample names.

    Expected format: 071-001_ScrBsl_plasma_md or similar.
    """
    df = df.copy()

    def extract_patient(s):
        # Take first two hyphenated parts: 071-001
        parts = s.split("_")
        if len(parts) >= 1 and "-" in parts[0]:
            return parts[0]
        return s

    def extract_timepoint(s):
        s_lower = s.lower()
        if "scrbsl" in s_lower:
            return "ScrBsl"
        elif "immonly" in s_lower:
            return "Immonly"
        elif "c3" in s_lower or "cycle3" in s_lower:
            return "C3"
        return "Unknown"

    df["patient_id"] = df["sample"].apply(extract_patient)
    df["timepoint"] = df["sample"].apply(extract_timepoint)

    return df


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_stacked_bars(df: pd.DataFrame, cell_types: list[str],
                      output_dir: Path, cohort: str):
    """Stacked bar chart of cell-type proportions per sample."""
    fig = go.Figure()
    colours = px.colors.qualitative.Set3[:len(cell_types)]
    if len(cell_types) > 12:
        colours += px.colors.qualitative.Plotly[:(len(cell_types) - 12)]

    for ct, colour in zip(cell_types, colours):
        fig.add_trace(go.Bar(
            name=ct, x=df["sample"], y=df[ct], marker_color=colour,
        ))

    fig.update_layout(
        barmode="stack", title=f"Cell-Type Proportions ({cohort})",
        xaxis_title="Sample", yaxis_title="Proportion",
        xaxis_tickangle=-90, height=600, width=max(800, len(df) * 20),
    )
    save_fig(fig, str(output_dir / f"{cohort}_stacked_bars"))


def plot_proportion_heatmap(df: pd.DataFrame, cell_types: list[str],
                             output_dir: Path, cohort: str):
    """Heatmap of proportions (samples × cell types)."""
    matrix = df[cell_types].values.T

    fig = go.Figure(go.Heatmap(
        z=matrix, x=df["sample"].tolist(), y=cell_types,
        colorscale="RdYlBu_r", zmin=0, zmax=0.6,
        colorbar=dict(title="Proportion"),
    ))
    fig.update_layout(
        title=f"Proportion Heatmap ({cohort})",
        xaxis_tickangle=-90, height=500, width=max(800, len(df) * 20),
    )
    save_fig(fig, str(output_dir / f"{cohort}_heatmap"))


def plot_waterfall(df_delta: pd.DataFrame, output_dir: Path, cohort: str,
                   cell_type: str, delta_col: str):
    """Waterfall plot: cell type proportion change between ScrBsl and Immonly."""
    if df_delta.empty or delta_col not in df_delta.columns:
        logger.warning("No delta data for %s waterfall plot", cell_type)
        return

    df_sorted = df_delta.sort_values(delta_col)

    colours = []
    for _, row in df_sorted.iterrows():
        if row.get("Clinical_Benefit") == "Y":
            colours.append("green")
        elif row.get("Clinical_Benefit") == "N":
            colours.append("red")
        else:
            colours.append("grey")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted["patient_id"].tolist(),
        y=df_sorted[delta_col].tolist(),
        marker_color=colours,
        text=[f"{d:.4f}" for d in df_sorted[delta_col]],
        textposition="outside",
        textfont=dict(size=8),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    ct_safe = cell_type.replace("-", "_")
    fig.update_layout(
        title=f"{cell_type} Change: Immonly - ScrBsl ({cohort})",
        xaxis_title="Patient", yaxis_title=f"Δ{cell_type} Proportion",
        xaxis_tickangle=-90, height=500,
        annotations=[
            dict(text="Green = Clinical Benefit, Red = No Benefit, Grey = Unknown",
                 xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False, font=dict(size=10)),
        ],
    )
    save_fig(fig, str(output_dir / f"{cohort}_{ct_safe}_waterfall"))


def plot_ichorcna_scatter(df: pd.DataFrame, ichorcna_data: dict,
                           output_dir: Path, cohort: str):
    """Scatter: OAC estimate vs ichorCNA tumour fraction per timepoint."""
    rows = []
    for _, row in df.iterrows():
        sample = row["sample"]
        # Try matching ichorCNA key
        ichor_tf = None
        for key, data in ichorcna_data.items():
            if sample in key or key in sample:
                ichor_tf = data.get("tumour_fraction", data.get("tf"))
                break
        if ichor_tf is not None:
            rows.append({
                "sample": sample,
                "patient_id": row.get("patient_id", ""),
                "timepoint": row.get("timepoint", "Unknown"),
                "oac_estimate": row["OAC"],
                "ichor_tf": ichor_tf,
            })

    if not rows:
        logger.warning("No ichorCNA matches found")
        return

    scatter_df = pd.DataFrame(rows)
    logger.info("Matched %d samples with ichorCNA", len(scatter_df))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ScrBsl", "Immonly"))

    for col_idx, tp in enumerate(["ScrBsl", "Immonly"], 1):
        tp_df = scatter_df[scatter_df["timepoint"] == tp]
        if tp_df.empty:
            continue

        fig.add_trace(go.Scatter(
            x=tp_df["ichor_tf"], y=tp_df["oac_estimate"],
            mode="markers+text", text=tp_df["patient_id"],
            textposition="top center", textfont=dict(size=7),
            marker=dict(size=8, opacity=0.7),
            name=tp, showlegend=False,
        ), row=1, col=col_idx)

        # y=x line
        max_val = max(tp_df["ichor_tf"].max(), tp_df["oac_estimate"].max(), 0.1)
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(color="red", dash="dash"), showlegend=False,
        ), row=1, col=col_idx)

        # Correlation
        if len(tp_df) > 3:
            r, p = stats.pearsonr(tp_df["ichor_tf"], tp_df["oac_estimate"])
            fig.add_annotation(
                text=f"r={r:.3f}, p={p:.2e}", xref=f"x{col_idx}", yref=f"y{col_idx}",
                x=0.05, y=0.95, xanchor="left", yanchor="top",
                showarrow=False, font=dict(size=10),
            )

        fig.update_xaxes(title_text="ichorCNA TF", row=1, col=col_idx)
        fig.update_yaxes(title_text="OAC estimate", row=1, col=col_idx)

    fig.update_layout(height=500, width=1000,
                      title=f"OAC estimate vs ichorCNA ({cohort})")
    save_fig(fig, str(output_dir / f"{cohort}_oac_vs_ichorcna"))

    # Save the matched data
    scatter_df.to_csv(output_dir / f"{cohort}_oac_ichorcna_matched.csv", index=False)


def plot_celltype_vs_ichorcna(df: pd.DataFrame, ichorcna_data: dict,
                               cell_types: list[str], output_dir: Path, cohort: str):
    """Correlation of each cell type with ichorCNA tumour fraction."""
    # Build matched data
    rows = []
    for _, row in df.iterrows():
        sample = row["sample"]
        ichor_tf = None
        for key, data in ichorcna_data.items():
            if sample in key or key in sample:
                ichor_tf = data.get("tumour_fraction", data.get("tf"))
                break
        if ichor_tf is not None:
            r = {"ichor_tf": ichor_tf}
            for ct in cell_types:
                r[ct] = row[ct]
            rows.append(r)

    if not rows:
        return

    matched = pd.DataFrame(rows)
    n_ct = len(cell_types)
    cols = min(4, n_ct)
    n_rows = (n_ct + cols - 1) // cols

    fig = make_subplots(rows=n_rows, cols=cols, subplot_titles=cell_types)

    for i, ct in enumerate(cell_types):
        r_idx, c_idx = i // cols + 1, i % cols + 1
        fig.add_trace(go.Scatter(
            x=matched["ichor_tf"], y=matched[ct], mode="markers",
            marker=dict(size=4, opacity=0.6), showlegend=False,
        ), row=r_idx, col=c_idx)

        if len(matched) > 3:
            r, p = stats.pearsonr(matched["ichor_tf"], matched[ct])
            fig.add_annotation(
                text=f"r={r:.2f}", xref=f"x{i+1}", yref=f"y{i+1}",
                x=0.95, y=0.95, xanchor="right", yanchor="top",
                showarrow=False, font=dict(size=8),
            )

    fig.update_layout(height=250 * n_rows, width=1200,
                      title=f"Cell Types vs ichorCNA TF ({cohort})")
    save_fig(fig, str(output_dir / f"{cohort}_celltypes_vs_ichorcna"))


def plot_kaplan_meier(df_delta: pd.DataFrame, output_dir: Path, cohort: str,
                      cell_type: str, delta_col: str, prefix: str = ""):
    """Kaplan-Meier survival curves split by cell type change direction."""
    if not LIFELINES_AVAILABLE:
        logger.warning("lifelines not installed, skipping KM plot")
        return

    required = ["OS_MONTHS", "DEATH_IND", delta_col]
    for col in required:
        if col not in df_delta.columns:
            logger.warning("Missing %s column, skipping KM plot for %s", col, cell_type)
            return

    df_km = df_delta.dropna(subset=["OS_MONTHS", "DEATH_IND"])
    df_km = df_km[df_km["OS_MONTHS"] != "Unknown"]
    df_km["OS_MONTHS"] = pd.to_numeric(df_km["OS_MONTHS"], errors="coerce")
    df_km["DEATH_IND"] = pd.to_numeric(df_km["DEATH_IND"], errors="coerce")
    df_km = df_km.dropna(subset=["OS_MONTHS", "DEATH_IND"])

    if len(df_km) < 5:
        logger.warning("Too few samples for KM plot (%d)", len(df_km))
        return

    down = df_km[df_km[delta_col] <= 0]
    up = df_km[df_km[delta_col] > 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    if len(down) > 1:
        kmf.fit(down["OS_MONTHS"], down["DEATH_IND"], label=f"{cell_type} Down (n={len(down)})")
        kmf.plot_survival_function(ax=ax, color="green")

    if len(up) > 1:
        kmf.fit(up["OS_MONTHS"], up["DEATH_IND"], label=f"{cell_type} Up (n={len(up)})")
        kmf.plot_survival_function(ax=ax, color="red")

    if len(down) > 1 and len(up) > 1:
        lr = logrank_test(down["OS_MONTHS"], up["OS_MONTHS"],
                          down["DEATH_IND"], up["DEATH_IND"])
        ax.text(0.95, 0.05, f"Log-rank p = {lr.p_value:.4f}",
                transform=ax.transAxes, ha="right", fontsize=10)

    ax.set_xlabel("Overall Survival (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"Survival by {cell_type} Change ({prefix}{cohort})")
    ax.legend()

    plt.tight_layout()
    fname = f"{prefix}{cohort}_kaplan_meier_{cell_type.replace('-', '_')}.png"
    fig.savefig(str(output_dir / fname), dpi=150)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clinical evaluation of tapestry predictions.")
    parser.add_argument("--predictions", required=True, help="Predictions CSV from predict_cfdna.py")
    parser.add_argument("--nnls-predictions", default=None, help="NNLS predictions CSV for comparison")
    parser.add_argument("--clinical-file", default=None, help="Clinical data CSV")
    parser.add_argument("--ichorcna-file", default=None, help="ichorCNA JSON or CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cohort", default="")
    parser.add_argument("--method-name", default="tapestry",
                        help="Label/prefix for the primary prediction columns")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    df = pd.read_csv(args.predictions)
    logger.info("Loaded %d predictions", len(df))

    # Detect cell type columns. Combined prediction CSVs can include
    # diagnostic metadata, lambda-path columns, and embedded `{ct}_nnls`
    # comparator columns; only the bare numeric columns are the primary
    # production estimates.
    cell_types = infer_cell_type_columns(df)
    if not cell_types:
        raise ValueError("No cell-type columns detected in predictions CSV")
    logger.info("Cell types: %s", cell_types)

    # Parse sample metadata
    df = parse_sample_metadata(df)

    # Load clinical data
    df_clinical = None
    if args.clinical_file and Path(args.clinical_file).exists():
        df_clinical = pd.read_csv(args.clinical_file)
        logger.info("Loaded clinical data: %d patients", len(df_clinical))

    # Load ichorCNA
    ichorcna_data = None
    if args.ichorcna_file and Path(args.ichorcna_file).exists():
        path = Path(args.ichorcna_file)
        if path.suffix == ".json":
            with open(path) as f:
                ichorcna_data = json.load(f)
        else:
            ichor_df = pd.read_csv(path)
            ichorcna_data = {}
            sample_col = ichor_df.columns[0]
            for _, row in ichor_df.iterrows():
                ichorcna_data[str(row[sample_col])] = {"tumour_fraction": float(row.get("tumour_fraction", row.get("tf", 0)))}
        logger.info("Loaded ichorCNA data: %d samples", len(ichorcna_data))

    # --- Generate plots for each method ---
    methods = [(args.method_name, df)]

    # Load NNLS predictions — either from a separate CSV (legacy) or from
    # the `{ct}_nnls` columns embedded in the production CSV.
    if args.nnls_predictions and Path(args.nnls_predictions).exists():
        df_nnls = pd.read_csv(args.nnls_predictions)
        df_nnls = parse_sample_metadata(df_nnls)
        methods.append(("nnls", df_nnls))
        logger.info("Loaded NNLS predictions: %d samples", len(df_nnls))
    elif all(f"{ct}_nnls" in df.columns for ct in cell_types):
        rename = {f"{ct}_nnls": ct for ct in cell_types}
        keep_meta = [c for c in df.columns if c in META_COLS]
        df_nnls = df[keep_meta + list(rename.keys())].rename(columns=rename).copy()
        methods.append(("nnls", df_nnls))
        logger.info("Built NNLS view from combined predictions CSV: %d samples", len(df_nnls))

    for method_name, method_df in methods:
        prefix = f"{method_name}_"
        logger.info("Generating plots for %s...", method_name)

        # 1. Stacked bars
        plot_stacked_bars(method_df, cell_types, output_dir, f"{prefix}{args.cohort}")

        # 2. Heatmap
        plot_proportion_heatmap(method_df, cell_types, output_dir, f"{prefix}{args.cohort}")

        # 3-4. Waterfall + KM
        df_delta = _compute_deltas(method_df, df_clinical)
        if not df_delta.empty:
            # Compute T-cells delta
            if "T-cells_immonly" in df_delta.columns and "T-cells_scrbsl" in df_delta.columns:
                df_delta["tcells_delta"] = df_delta["T-cells_immonly"] - df_delta["T-cells_scrbsl"]

            # OAC waterfall + KM
            plot_waterfall(df_delta, output_dir, f"{prefix}{args.cohort}",
                           cell_type="OAC", delta_col="oac_delta")
            plot_kaplan_meier(df_delta, output_dir, args.cohort,
                              cell_type="OAC", delta_col="oac_delta", prefix=prefix)

            # T-cells waterfall + KM
            if "tcells_delta" in df_delta.columns:
                plot_waterfall(df_delta, output_dir, f"{prefix}{args.cohort}",
                               cell_type="T-cells", delta_col="tcells_delta")
                plot_kaplan_meier(df_delta, output_dir, args.cohort,
                                  cell_type="T-cells", delta_col="tcells_delta", prefix=prefix)

            df_delta.to_csv(output_dir / f"{prefix}{args.cohort}_deltas.csv", index=False)

        # 5-6. ichorCNA comparison
        if ichorcna_data:
            plot_ichorcna_scatter(method_df, ichorcna_data, output_dir, f"{prefix}{args.cohort}")
            plot_celltype_vs_ichorcna(method_df, ichorcna_data, cell_types, output_dir, f"{prefix}{args.cohort}")

        # 7. Export full table
        method_df.to_csv(output_dir / f"{prefix}{args.cohort}_full_results.csv", index=False)

    logger.info("All outputs saved to %s", output_dir)


def _compute_deltas(df: pd.DataFrame, df_clinical: pd.DataFrame | None) -> pd.DataFrame:
    """Compute OAC change between ScrBsl and Immonly per patient."""
    rows = []
    for patient_id, group in df.groupby("patient_id"):
        scr = group[group["timepoint"] == "ScrBsl"]
        imm = group[group["timepoint"] == "Immonly"]
        if len(scr) == 0 or len(imm) == 0:
            continue

        oac_scr = scr.iloc[0]["OAC"]
        oac_imm = imm.iloc[0]["OAC"]

        row = {
            "patient_id": patient_id,
            "oac_scrbsl": oac_scr,
            "oac_immonly": oac_imm,
            "oac_delta": oac_imm - oac_scr,
        }

        # Add all production cell types for both timepoints.
        cell_types = infer_cell_type_columns(df)
        for ct in cell_types:
            row[f"{ct}_scrbsl"] = scr.iloc[0][ct]
            row[f"{ct}_immonly"] = imm.iloc[0][ct]

        # Clinical data
        if df_clinical is not None:
            for _, clin_row in df_clinical.iterrows():
                subject = str(clin_row.get("subject", ""))
                if patient_id in subject or subject in patient_id:
                    row["Clinical_Benefit"] = clin_row.get("Clinical_Benefit", "NA")
                    row["OS_MONTHS"] = clin_row.get("OS_MONTHS", "Unknown")
                    row["DEATH_IND"] = clin_row.get("OS", "Unknown")
                    row["subject_recode"] = clin_row.get("subject_recode", patient_id)
                    break

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
