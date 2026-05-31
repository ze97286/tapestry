#!/usr/bin/env python3
"""Plot MethylBERT read-classifier deconvolution estimates against ichorCNA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_ichor(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        rows = []
        for sample, entry in payload.items():
            if isinstance(entry, dict):
                tf = entry.get("tumour_fraction", entry.get("tumor_fraction", entry.get("tf")))
            else:
                tf = entry
            if tf is not None:
                rows.append({"ichor_sample": str(sample), "ichor_tf": float(tf)})
        return pd.DataFrame(rows)

    df = pd.read_csv(path)
    sample_col = "sample" if "sample" in df.columns else df.columns[0]
    if "tumour_fraction" in df.columns:
        tf_col = "tumour_fraction"
    elif "tumor_fraction" in df.columns:
        tf_col = "tumor_fraction"
    elif "tf" in df.columns:
        tf_col = "tf"
    else:
        tf_col = df.columns[1]
    return df[[sample_col, tf_col]].rename(columns={sample_col: "ichor_sample", tf_col: "ichor_tf"})


def parse_sample(sample: str) -> tuple[str, str]:
    parts = sample.split("_")
    patient_id = parts[0] if parts else sample
    sample_lower = sample.lower()
    if "scrbsl" in sample_lower:
        timepoint = "ScrBsl"
    elif "immonly" in sample_lower:
        timepoint = "Immonly"
    elif "c1w3" in sample_lower:
        timepoint = "C1W3"
    elif "c6d22" in sample_lower:
        timepoint = "C6D22"
    elif "_pt" in sample_lower or sample_lower.endswith("_pt"):
        timepoint = "PT"
    elif "ltsr" in sample_lower:
        timepoint = "LTSR"
    else:
        timepoint = "Unknown"
    return patient_id, timepoint


def normalise_sample_key(value: str) -> str:
    key = Path(str(value)).name
    for suffix in [
        ".per-read.bed.gz",
        ".markers.pat.gz",
        ".pat.gz",
        ".bed.gz",
        ".bam",
        ".cram",
        ".csv",
        ".tsv",
    ]:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def attach_ichor(deconv: pd.DataFrame, ichor: pd.DataFrame, estimate_col: str) -> pd.DataFrame:
    ichor = ichor.copy()
    ichor["ichor_key"] = ichor["ichor_sample"].map(normalise_sample_key)
    ichor_rows = ichor.to_dict("records")

    rows = []
    for _, rec in deconv.iterrows():
        sample = str(rec["sample"])
        sample_key = normalise_sample_key(sample)
        hit = None
        for candidate in ichor_rows:
            key = str(candidate["ichor_key"])
            if sample_key == key or sample_key in key or key in sample_key:
                hit = candidate
                break
        if hit is None:
            continue

        patient_id, timepoint = parse_sample(sample)
        rows.append(
            {
                "sample": sample,
                "patient_id": patient_id,
                "timepoint": timepoint,
                "methylbert_tf": float(rec[estimate_col]),
                "ichor_sample": hit["ichor_sample"],
                "ichor_tf": float(hit["ichor_tf"]),
                "n_reads_classified": rec.get("n_reads_classified", math.nan),
                "mean_read_length": rec.get("mean_read_length", math.nan),
                "mean_n_cpg": rec.get("mean_n_cpg", math.nan),
                "deconvolution_path": rec.get("deconvolution_path", ""),
            }
        )
    return pd.DataFrame(rows)


def fragmentomics_report(matched: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Correlate the MethylBERT tumour fraction with ichorCNA (orthogonal, CNA-based)
    and with non-methylation features (mean read length, mean n_cpg, read count).
    If theta tracks ichorCNA it is catching real tumour content; if it tracks read
    length / read count more strongly, it is a fragmentomics/batch shortcut."""
    features = ["ichor_tf", "mean_read_length", "mean_n_cpg", "n_reads_classified"]
    records = []
    for feature in features:
        if feature not in matched.columns:
            continue
        paired = matched[[feature, "methylbert_tf"]].apply(pd.to_numeric, errors="coerce").dropna()
        records.append(
            {
                "feature": feature,
                "pearson_r_vs_methylbert_tf": pearson(matched["methylbert_tf"], matched[feature]),
                "n": int(len(paired)),
            }
        )
    report = pd.DataFrame(records)
    report.to_csv(output_path, index=False)
    return report


def pearson(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return math.nan
    return float(x[mask].corr(y[mask], method="pearson"))


def build_plot(matched: pd.DataFrame, output_html: Path, title: str) -> None:
    ordered_timepoints = ["ScrBsl", "Immonly", "C1W3", "C6D22", "PT", "LTSR", "Unknown"]
    timepoints = [tp for tp in ordered_timepoints if (matched["timepoint"] == tp).any()]
    if not timepoints:
        timepoints = ["all"]
        matched = matched.assign(timepoint="all")

    cols = min(3, len(timepoints))
    rows = math.ceil(len(timepoints) / cols)
    subplot_titles = []
    for tp in timepoints:
        sub = matched[matched["timepoint"] == tp]
        r = pearson(sub["ichor_tf"], sub["methylbert_tf"])
        r_txt = "NA" if math.isnan(r) else f"{r:.2f}"
        subplot_titles.append(f"{tp} (n={len(sub)}, r={r_txt})")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    max_val = max(float(matched["ichor_tf"].max()), float(matched["methylbert_tf"].max()), 0.05)
    max_val = min(max_val * 1.08, 1.0)

    for idx, tp in enumerate(timepoints):
        sub = matched[matched["timepoint"] == tp].copy()
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(
            go.Scatter(
                x=sub["ichor_tf"],
                y=sub["methylbert_tf"],
                mode="markers+text",
                text=sub["patient_id"],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(size=9, color="#1b2d5a", opacity=0.78, line=dict(width=0.5, color="white")),
                customdata=sub[["sample", "ichor_sample", "n_reads_classified"]],
                hovertemplate=(
                    "sample=%{customdata[0]}<br>"
                    "ichor sample=%{customdata[1]}<br>"
                    "ichorCNA=%{x:.4f}<br>"
                    "MethylBERT=%{y:.4f}<br>"
                    "reads=%{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="#d95b59", dash="dash", width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="ichorCNA tumour fraction", range=[0, max_val], row=row, col=col)
        fig.update_yaxes(title_text="MethylBERT MLE tumour fraction", range=[0, max_val], row=row, col=col)

    all_r = pearson(matched["ichor_tf"], matched["methylbert_tf"])
    all_r_txt = "NA" if math.isnan(all_r) else f"{all_r:.3f}"
    fig.update_layout(
        title=f"{title}<br><sup>Matched n={len(matched)}; all-sample Pearson r={all_r_txt}</sup>",
        template="plotly_white",
        width=1250,
        height=max(520, 420 * rows),
        font=dict(family="Arial, Helvetica, sans-serif", size=14),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deconvolution-summary", required=True)
    parser.add_argument("--ichorcna-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--estimate-col", default="T")
    parser.add_argument("--prefix", default="methylbert")
    args = parser.parse_args()

    deconv = pd.read_csv(args.deconvolution_summary)
    if args.estimate_col not in deconv.columns:
        raise SystemExit(f"{args.deconvolution_summary} does not contain estimate column {args.estimate_col!r}; columns={list(deconv.columns)}")
    if "sample" not in deconv.columns:
        raise SystemExit(f"{args.deconvolution_summary} does not contain a sample column")

    ichor = load_ichor(Path(args.ichorcna_file))
    matched = attach_ichor(deconv, ichor, args.estimate_col)
    if matched.empty:
        raise SystemExit("No overlap between deconvolution summary samples and ichorCNA entries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_path = output_dir / f"{args.prefix}_methylbert_vs_ichorcna_matched.csv"
    html_path = output_dir / f"{args.prefix}_methylbert_vs_ichorcna.html"
    matched.to_csv(matched_path, index=False)
    build_plot(matched, html_path, "MethylBERT classifier/MLE tumour fraction vs ichorCNA")

    fragmentomics_path = output_dir / f"{args.prefix}_theta_vs_fragmentomics.csv"
    report = fragmentomics_report(matched, fragmentomics_path)

    print(f"matched {len(matched)} samples")
    print(f"wrote {matched_path}")
    print(f"wrote {html_path}")
    print(f"wrote {fragmentomics_path}")
    print("theta correlations (genuine signal => ichor_tf dominates read length / read count):")
    print(report.to_csv(index=False))


if __name__ == "__main__":
    main()
