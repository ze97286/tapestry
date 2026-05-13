#!/usr/bin/env python3
"""Plot diagnostics for unknown-robust weighted NNLS runs.

The script intentionally avoids Python plotting dependencies. It writes an
HTML dashboard that loads Plotly from a CDN, plus CSV summary tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json(obj: Any) -> str:
    return json.dumps(obj, allow_nan=False)


def _series(df: pd.DataFrame, column: str) -> list[float | str | bool | None]:
    values = df[column].tolist()
    out = []
    for value in values:
        if pd.isna(value):
            out.append(None)
        elif isinstance(value, (np.integer, np.floating)):
            out.append(float(value))
        else:
            out.append(value)
    return out


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for record in df.to_dict(orient="records"):
        cleaned = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        rows.append(cleaned)
    return rows


def infer_path_output(predictions: Path) -> Path:
    return predictions.with_name(f"{predictions.stem}_lambda_path.csv")


def add_delta_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    nnls_col = f"{target}_nnls"
    if target in df.columns and nnls_col in df.columns:
        df[f"{target}_delta_vs_nnls"] = df[target] - df[nnls_col]
    return df


def write_tables(df: pd.DataFrame, lp: pd.DataFrame, target: str, output_dir: Path) -> None:
    cols = [target, f"{target}_nnls", "unknown_mag", "mean_coverage", "n_markers_with_coverage"]
    cols = [c for c in cols if c in df.columns]
    summary = df.groupby("is_control")[cols].agg(["count", "mean", "median", "max"])
    summary.to_csv(output_dir / "production_summary_by_control.csv")

    lambda_summary = lp.groupby(["lambda_unknown", "is_control"])[[target, "unknown_mag"]].agg(
        ["count", "mean", "median", "max"]
    )
    lambda_summary.to_csv(output_dir / "lambda_summary_by_control.csv")

    sample_cols = [
        "sample", "is_control", "unknown_basis_mode", "control_crossfit_fold",
        "unknown_fit_excluded_cell_types", target, f"{target}_nnls",
        f"{target}_delta_vs_nnls", "unknown_mag", "mean_coverage",
        "n_markers_with_coverage",
    ]
    sample_cols = [c for c in sample_cols if c in df.columns]
    df[sample_cols].sort_values(["is_control", f"{target}_delta_vs_nnls", "sample"]).to_csv(
        output_dir / "sample_level_diagnostics.csv", index=False
    )


def make_lambda_summary(lp: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for (lam, is_control), sub in lp.groupby(["lambda_unknown", "is_control"]):
        rows.append({
            "lambda_unknown": float(lam),
            "is_control": bool(is_control),
            "target_mean": float(sub[target].mean()),
            "target_median": float(sub[target].median()),
            "target_max": float(sub[target].max()),
            "unknown_mag_mean": float(sub["unknown_mag"].mean()),
            "unknown_mag_median": float(sub["unknown_mag"].median()),
            "unknown_mag_max": float(sub["unknown_mag"].max()),
            "n": int(len(sub)),
        })
    return pd.DataFrame(rows).sort_values(["is_control", "lambda_unknown"])


def load_ichor(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        with open(path) as handle:
            data = json.load(handle)
        rows = []
        for sample, entry in data.items():
            if isinstance(entry, dict):
                tf = entry.get("tumour_fraction", entry.get("tf"))
            else:
                tf = entry
            if tf is not None:
                rows.append({"sample": str(sample), "ichor_tf": float(tf)})
        return pd.DataFrame(rows)

    df = pd.read_csv(path)
    sample_col = df.columns[0]
    if "tumour_fraction" in df.columns:
        tf_col = "tumour_fraction"
    elif "tf" in df.columns:
        tf_col = "tf"
    else:
        tf_col = df.columns[1]
    return df[[sample_col, tf_col]].rename(columns={sample_col: "sample", tf_col: "ichor_tf"})


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_by_lambda(lp: pd.DataFrame, ichor: pd.DataFrame, target: str) -> pd.DataFrame:
    merged = lp.merge(ichor, on="sample", how="inner")
    merged = merged[~merged["is_control"]].copy()
    rows = []
    for lam, sub in merged.groupby("lambda_unknown"):
        rows.append({
            "lambda_unknown": float(lam),
            "n": int(len(sub)),
            "pearson_r": pearson(sub[target].to_numpy(float), sub["ichor_tf"].to_numpy(float)),
        })
    return pd.DataFrame(rows).sort_values("lambda_unknown")


def html_dashboard(
    df: pd.DataFrame,
    lp: pd.DataFrame,
    target: str,
    output_dir: Path,
    ichor: pd.DataFrame | None = None,
) -> None:
    lambda_summary = make_lambda_summary(lp, target)
    nnls_col = f"{target}_nnls"
    has_nnls = nnls_col in df.columns
    delta_col = f"{target}_delta_vs_nnls"

    ctrl = df[df["is_control"]].copy()
    nonctrl = df[~df["is_control"]].copy()

    lambda_target_traces = []
    lambda_unknown_traces = []
    for is_control, label, color in [
        (True, "controls", "#1f77b4"),
        (False, "non-controls", "#d62728"),
    ]:
        sub = lambda_summary[lambda_summary["is_control"] == is_control]
        x_values = [f"{v:g}" for v in sub["lambda_unknown"].tolist()]
        lambda_target_traces.extend([
            {
                "x": x_values,
                "y": _series(sub, "target_mean"),
                "mode": "lines+markers",
                "name": f"{label} mean",
                "line": {"color": color},
            },
            {
                "x": x_values,
                "y": _series(sub, "target_median"),
                "mode": "lines+markers",
                "name": f"{label} median",
                "line": {"color": color, "dash": "dash"},
            },
        ])
        lambda_unknown_traces.append({
            "x": x_values,
            "y": _series(sub, "unknown_mag_mean"),
            "mode": "lines+markers",
            "name": f"{label} mean",
            "line": {"color": color},
        })

    lambda_layout_shapes = []
    if has_nnls:
        for is_control, color in [(True, "#1f77b4"), (False, "#d62728")]:
            mean_value = float(df.loc[df["is_control"] == is_control, nnls_col].mean())
            lambda_layout_shapes.append({
                "type": "line", "xref": "paper", "x0": 0, "x1": 1,
                "yref": "y", "y0": mean_value, "y1": mean_value,
                "line": {"color": color, "dash": "dot", "width": 1},
            })

    box_y = []
    box_x = []
    for label, frame in [("control augmented", ctrl), ("control NNLS", ctrl),
                         ("non-control augmented", nonctrl), ("non-control NNLS", nonctrl)]:
        column = target if "augmented" in label else nnls_col
        if column in frame.columns:
            box_y.extend(frame[column].astype(float).tolist())
            box_x.extend([label] * len(frame))

    scatter_traces = []
    max_scatter_value = 0.0
    if has_nnls:
        for is_control, label, color in [
            (True, "control", "#1f77b4"),
            (False, "non-control", "#d62728"),
        ]:
            sub = df[df["is_control"] == is_control]
            scatter_traces.append({
                "x": _series(sub, nnls_col),
                "y": _series(sub, target),
                "text": _series(sub, "sample"),
                "mode": "markers",
                "type": "scatter",
                "name": label,
                "marker": {"color": color, "size": 7, "opacity": 0.75},
            })
        max_scatter_value = float(max(df[nnls_col].max(), df[target].max(), 0.01))
        scatter_traces.append({
            "x": [0, max_scatter_value],
            "y": [0, max_scatter_value],
            "mode": "lines",
            "name": "y=x",
            "line": {"color": "#555", "dash": "dash"},
        })

    corr_rows = []
    scatter_ichor_rows = []
    if ichor is not None:
        corr = correlation_by_lambda(lp, ichor, target)
        corr.to_csv(output_dir / "ichorcna_correlation_by_lambda.csv", index=False)
        corr_rows = _records(corr)
        primary = df.merge(ichor, on="sample", how="inner")
        primary = primary[~primary["is_control"]].copy()
        scatter_ichor_rows = _records(primary)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Unknown-Robust NNLS Diagnostics</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 17px; margin-top: 28px; }}
    .plot {{ width: 100%; height: 520px; }}
    .note {{ color: #555; max-width: 900px; line-height: 1.35; }}
    code {{ background: #f3f3f3; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Unknown-Robust NNLS Diagnostics: {target}</h1>
  <p class="note">
    Controls are evaluated out-of-fold when <code>unknown_basis_mode=control_crossfit</code>.
    Dotted horizontal lines in the lambda plot are plain NNLS means.
  </p>

  <h2>{target} by lambda</h2>
  <div id="lambda-target" class="plot"></div>

  <h2>{target}: augmented vs NNLS distributions</h2>
  <div id="box-target" class="plot"></div>

  <h2>Augmented {target} vs NNLS {target}</h2>
  <div id="scatter-nnls" class="plot"></div>

  <h2>Unknown magnitude by lambda</h2>
  <div id="lambda-unknown" class="plot"></div>

  <h2>{target} delta vs NNLS</h2>
  <div id="delta-box" class="plot"></div>

  <h2>Optional ichorCNA</h2>
  <div id="ichor-corr" class="plot"></div>
  <div id="ichor-scatter" class="plot"></div>

<script>
const layoutLambdaTarget = {{
  xaxis: {{title: 'lambda_unknown', type: 'category'}},
  yaxis: {{title: '{target}'}},
  shapes: {_json(lambda_layout_shapes)},
  legend: {{orientation: 'h'}},
  margin: {{t: 20}}
}};
Plotly.newPlot('lambda-target', {_json(lambda_target_traces)}, layoutLambdaTarget);

Plotly.newPlot('box-target', [{{
  x: {_json(box_x)},
  y: {_json(box_y)},
  type: 'box',
  boxpoints: 'all',
  jitter: 0.35,
  pointpos: 0,
  marker: {{size: 5, opacity: 0.55}}
}}], {{
  yaxis: {{title: '{target}'}},
  xaxis: {{title: ''}},
  margin: {{t: 20, b: 120}}
}});

Plotly.newPlot('scatter-nnls', {_json(scatter_traces)}, {{
  xaxis: {{title: 'NNLS {target}'}},
  yaxis: {{title: 'Augmented {target}'}},
  margin: {{t: 20}}
}});

Plotly.newPlot('lambda-unknown', {_json(lambda_unknown_traces)}, {{
  xaxis: {{title: 'lambda_unknown', type: 'category'}},
  yaxis: {{title: 'unknown_mag'}},
  legend: {{orientation: 'h'}},
  margin: {{t: 20}}
}});

Plotly.newPlot('delta-box', [{{
  x: {_json(["control" if v else "non-control" for v in df["is_control"]])},
  y: {_json(_series(df, delta_col)) if delta_col in df.columns else "[]"},
  type: 'box',
  boxpoints: 'all',
  jitter: 0.35,
  marker: {{size: 5, opacity: 0.55}}
}}], {{
  yaxis: {{title: 'Augmented {target} - NNLS {target}'}},
  shapes: [{{type:'line', xref:'paper', x0:0, x1:1, yref:'y', y0:0, y1:0,
             line:{{color:'#555', dash:'dash'}}}}],
  margin: {{t: 20}}
}});

const corrRows = {_json(corr_rows)};
if (corrRows.length) {{
  Plotly.newPlot('ichor-corr', [{{
    x: corrRows.map(r => String(r.lambda_unknown)),
    y: corrRows.map(r => r.pearson_r),
    mode: 'lines+markers',
    type: 'scatter'
  }}], {{
    xaxis: {{title: 'lambda_unknown', type: 'category'}},
    yaxis: {{title: 'Pearson r vs ichorCNA'}},
    margin: {{t: 20}}
  }});
}} else {{
  document.getElementById('ichor-corr').innerHTML = '<p class="note">No ichorCNA file supplied.</p>';
}}

const ichorRows = {_json(scatter_ichor_rows)};
if (ichorRows.length) {{
  Plotly.newPlot('ichor-scatter', [{{
    x: ichorRows.map(r => r.ichor_tf),
    y: ichorRows.map(r => r['{target}']),
    text: ichorRows.map(r => r.sample),
    mode: 'markers',
    type: 'scatter',
    marker: {{size: 8, opacity: 0.75}}
  }}], {{
    xaxis: {{title: 'ichorCNA tumour fraction'}},
    yaxis: {{title: 'Production augmented {target}'}},
    margin: {{t: 20}}
  }});
}} else {{
  document.getElementById('ichor-scatter').innerHTML = '';
}}
</script>
</body>
</html>
"""
    (output_dir / "unknown_robust_nnls_diagnostics.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--lambda-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="OAC")
    parser.add_argument("--ichorcna-file", default=None)
    args = parser.parse_args()

    predictions = Path(args.predictions)
    lambda_path = Path(args.lambda_path) if args.lambda_path else infer_path_output(predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions)
    lp = pd.read_csv(lambda_path)
    if args.target not in df.columns or args.target not in lp.columns:
        raise SystemExit(f"target column {args.target!r} not found in predictions/lambda path")
    if "is_control" not in df.columns or "is_control" not in lp.columns:
        raise SystemExit("predictions and lambda path must contain is_control")

    df = add_delta_columns(df, args.target)
    ichor = load_ichor(Path(args.ichorcna_file)) if args.ichorcna_file else None

    write_tables(df, lp, args.target, output_dir)
    html_dashboard(df, lp, args.target, output_dir, ichor=ichor)

    print(f"Wrote plots and tables to {output_dir}")
    print(f"Dashboard: {output_dir / 'unknown_robust_nnls_diagnostics.html'}")


if __name__ == "__main__":
    main()
