"""Plotly visualisations for each pipeline step (interactive, self-contained HTML).

One dashboard per step plus the clinical figures. Each function is defensive
(skips panels with no data) and writes self-contained HTML (``include_plotlyjs``
inlined) so the artifacts open offline after copying off the cluster.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _save(fig, path: Path, title: str | None = None) -> Path:
    if title:
        fig.update_layout(title=title)
    fig.update_layout(template="plotly_white", margin=dict(t=60, l=60, r=30, b=50))
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    logger.info("wrote %s", path)
    return path


def _weighted_hist(values, weights, bins):
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    return centres, (counts / total if total else counts)


# ---------------------------------------------------------------------------
# Step 1 — discovery
# ---------------------------------------------------------------------------

def plot_discovery(blocks_df, out_dir: str | Path) -> list[Path]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = blocks_df
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Per-block methylation: tumour vs healthy",
            "Effect size (|Δ mean methylation|)",
            "Blocks per chromosome",
            "Per-block minimum coverage",
        ),
    )
    fig.add_trace(go.Scatter(
        x=df["mean_p_healthy"], y=df["mean_p_tumour"], mode="markers",
        marker=dict(size=6, color=df["effect"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="effect", x=0.46, y=0.8, len=0.4)),
        text=df["block_id"], name="blocks"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="grey"), showlegend=False), row=1, col=1)
    fig.update_xaxes(title_text="healthy mean meth", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="tumour mean meth", range=[0, 1], row=1, col=1)

    fig.add_trace(go.Histogram(x=df["effect"], nbinsx=40, marker_color="#2a9d8f", showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="effect", row=1, col=2)

    by_chrom = df.groupby("chrom").size().sort_values(ascending=False)
    fig.add_trace(go.Bar(x=by_chrom.index.astype(str), y=by_chrom.values,
                         marker_color="#264653", showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="# blocks", row=2, col=1)

    fig.add_trace(go.Histogram(x=df["min_total_tumour"], nbinsx=40, name="tumour",
                               marker_color="#e76f51", opacity=0.7), row=2, col=2)
    fig.add_trace(go.Histogram(x=df["min_total_healthy"], nbinsx=40, name="healthy",
                               marker_color="#457b9d", opacity=0.7), row=2, col=2)
    fig.update_xaxes(title_text="min reads / CpG in block", row=2, col=2)
    fig.update_layout(barmode="overlay", height=820)
    return [_save(fig, out_dir / "discovery.html", f"Panel discovery — {len(df)} blocks")]


# ---------------------------------------------------------------------------
# Step 2 — oracle
# ---------------------------------------------------------------------------

def plot_oracle(reads_df, metrics: dict, out_dir: str | Path) -> list[Path]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import roc_curve

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = reads_df.dropna(subset=["llr"])
    t = df[df["label"] == 1]
    h = df[df["label"] == 0]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Per-read LLR: tumour vs healthy (weighted)",
            f"ROC — AUC={metrics.get('auc', float('nan')):.3f} (perm {metrics.get('auc_permuted', float('nan')):.3f})",
            "AUC by #CpG scored per read",
            "Per-healthy-cohort mean LLR",
        ),
    )
    lo, hi = np.percentile(df["llr"], [1, 99]) if len(df) else (-1, 1)
    bins = np.linspace(lo, hi, 60)
    for sub, name, colour in ((h, "healthy", "#457b9d"), (t, "tumour", "#e76f51")):
        if len(sub):
            c, p = _weighted_hist(sub["llr"].to_numpy(), sub["weight"].to_numpy(), bins)
            fig.add_trace(go.Bar(x=c, y=p, name=name, marker_color=colour, opacity=0.65), row=1, col=1)
    fig.update_xaxes(title_text="read LLR", row=1, col=1)
    fig.update_yaxes(title_text="density", row=1, col=1)

    if df["label"].nunique() == 2:
        fpr, tpr, _ = roc_curve(df["label"], df["llr"], sample_weight=df["weight"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#2a9d8f"),
                                 name="ROC", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="grey"), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="FPR", row=1, col=2)
    fig.update_yaxes(title_text="TPR", row=1, col=2)

    abn = metrics.get("auc_by_n_cpg", {})
    if abn:
        fig.add_trace(go.Bar(x=list(abn.keys()), y=[v["auc"] for v in abn.values()],
                             marker_color="#264653", showlegend=False), row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="grey", row=2, col=1)
    fig.update_yaxes(title_text="AUC", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="#CpG bucket", row=2, col=1)

    if len(h):
        coh = h.groupby("cohort").apply(
            lambda g: np.average(g["llr"], weights=g["weight"]), include_groups=False
        )
        fig.add_trace(go.Bar(x=coh.index.astype(str), y=coh.values, marker_color="#457b9d",
                             showlegend=False), row=2, col=2)
        if len(t):
            fig.add_hline(y=float(np.average(t["llr"], weights=t["weight"])),
                          line_dash="dash", line_color="#e76f51", row=2, col=2)
    fig.update_yaxes(title_text="mean LLR", row=2, col=2)
    fig.update_layout(barmode="overlay", height=820)
    return [_save(fig, out_dir / "oracle.html", "Read-level separability oracle")]


# ---------------------------------------------------------------------------
# Step 3 — detector
# ---------------------------------------------------------------------------

def plot_detector(features_df, cls_oof, reg_oof, summary: dict, out_dir: str | Path) -> list[Path]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import roc_curve

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_metrics = summary.get("classification", {})
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"ROC — AUC={cls_metrics.get('auc', float('nan')):.3f}",
            "Cancer probability by class",
            "Tumour fraction: predicted vs ichorCNA/true",
            "Tumour-pattern read fraction by class",
        ),
    )
    if cls_oof is not None and cls_oof["is_cancer"].nunique() == 2:
        fpr, tpr, _ = roc_curve(cls_oof["is_cancer"], cls_oof["pred_proba_cancer"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#2a9d8f"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False), row=1, col=1)
        thr = cls_metrics.get("threshold_at_spec_0.95")
        for lab, colour in ((0, "#457b9d"), (1, "#e76f51")):
            sub = cls_oof[cls_oof["is_cancer"] == lab]
            fig.add_trace(go.Box(y=sub["pred_proba_cancer"], name=("healthy" if lab == 0 else "cancer"),
                                 marker_color=colour, boxpoints="all", jitter=0.4), row=1, col=2)
        if thr is not None and np.isfinite(thr):
            fig.add_hline(y=float(thr), line_dash="dash", line_color="black", row=1, col=2,
                          annotation_text="spec 0.95")
    fig.update_xaxes(title_text="FPR", row=1, col=1)
    fig.update_yaxes(title_text="TPR", row=1, col=1)
    fig.update_yaxes(title_text="P(cancer)", row=1, col=2)

    if reg_oof is not None and len(reg_oof) >= 3:
        fig.add_trace(go.Scatter(x=reg_oof["tf_true"], y=reg_oof["tf_pred"], mode="markers",
                                 marker=dict(size=8, color="#264653"), text=reg_oof["sample_id"],
                                 showlegend=False), row=2, col=1)
        m = float(np.nanmax([reg_oof["tf_true"].max(), reg_oof["tf_pred"].max(), 0.01]))
        fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False), row=2, col=1)
        r = summary.get("regression", {}).get("pearson_r")
        if r is not None:
            fig.add_annotation(x=0.05 * m, y=0.95 * m, text=f"r={r:.3f}", showarrow=False, row=2, col=1)
    fig.update_xaxes(title_text="true / ichorCNA TF", row=2, col=1)
    fig.update_yaxes(title_text="predicted TF", row=2, col=1)

    if features_df is not None and "is_cancer" in features_df and "frac_reads_llr_gt_0" in features_df:
        for lab, colour in ((0, "#457b9d"), (1, "#e76f51")):
            sub = features_df[features_df["is_cancer"] == lab]
            fig.add_trace(go.Box(y=sub["frac_reads_llr_gt_0"], name=("healthy" if lab == 0 else "cancer"),
                                 marker_color=colour, boxpoints="all", jitter=0.4, showlegend=False), row=2, col=2)
    fig.update_yaxes(title_text="frac reads LLR>0", row=2, col=2)
    fig.update_layout(height=860, showlegend=False)
    return [_save(fig, out_dir / "detector.html", "Read-level tabular detector")]


# ---------------------------------------------------------------------------
# Step 4 — clinical
# ---------------------------------------------------------------------------

def plot_clinical(merged, summary: dict, out_dir: str | Path, threshold: float | None = None) -> list[Path]:
    import plotly.express as px
    import plotly.graph_objects as go

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    df = merged

    # (a) predicted TF vs ichorCNA validation scatter
    if "ichorcna_tf" in df and df["ichorcna_tf"].notna().sum() >= 3:
        sub = df.dropna(subset=["ichorcna_tf", "tf_pred"])
        fig = px.scatter(sub, x="ichorcna_tf", y="tf_pred", color="patient_id" if "patient_id" in sub else None,
                         hover_data=[c for c in ("sample_id", "timepoint") if c in sub])
        m = float(np.nanmax([sub["ichorcna_tf"].max(), sub["tf_pred"].max(), 0.01]))
        fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False))
        r = summary.get("tf_vs_ichorcna_pearson_r")
        title = "Predicted TF vs ichorCNA" + (f" (r={r:.3f})" if r is not None else "")
        paths.append(_save(fig, out_dir / "clinical_tf_vs_ichorcna.html", title))

    # (b) longitudinal per-patient TF trajectories
    if {"patient_id", "timepoint"}.issubset(df.columns):
        sub = df.dropna(subset=["tf_pred"]).sort_values(["patient_id", "timepoint"])
        fig = px.line(sub, x="timepoint", y="tf_pred", color="patient_id", markers=True,
                      hover_data=[c for c in ("sample_id", "ichorcna_tf") if c in sub])
        fig.update_yaxes(title_text="predicted TF")
        paths.append(_save(fig, out_dir / "clinical_longitudinal.html", "Longitudinal tumour fraction per patient"))

        # (c) waterfall of first→last TF change per patient
        deltas = []
        for pid, g in sub.groupby("patient_id"):
            g = g.dropna(subset=["tf_pred"])
            if len(g) >= 2:
                deltas.append((str(pid), float(g["tf_pred"].iloc[-1] - g["tf_pred"].iloc[0])))
        if deltas:
            deltas.sort(key=lambda x: x[1])
            ids, vals = zip(*deltas)
            colours = ["#2a9d8f" if v <= 0 else "#e76f51" for v in vals]
            fig = go.Figure(go.Bar(x=list(ids), y=list(vals), marker_color=colours))
            fig.update_layout(yaxis_title="Δ predicted TF (last − first)", xaxis_title="patient")
            paths.append(_save(fig, out_dir / "clinical_waterfall.html", "Tumour-fraction change (waterfall)"))

    # (d) detection scores with the specificity threshold
    if "pred_proba_cancer" in df and "is_cancer" in df:
        sub = df.dropna(subset=["pred_proba_cancer"])
        fig = px.strip(sub, x="is_cancer", y="pred_proba_cancer",
                       color="is_cancer", hover_data=[c for c in ("sample_id", "patient_id") if c in sub])
        if threshold is not None and np.isfinite(threshold):
            fig.add_hline(y=float(threshold), line_dash="dash", line_color="black",
                          annotation_text="detection threshold")
        fig.update_layout(xaxis_title="is_cancer (0=healthy,1=cancer)", yaxis_title="P(cancer)")
        paths.append(_save(fig, out_dir / "clinical_detection.html", "Detection calls at controlled specificity"))

    # (e) Kaplan-Meier by TF-change direction (optional)
    if {"survival_time", "survival_event"}.issubset(df.columns):
        paths += _plot_km(df, out_dir, summary)
    return paths


def _km_curve(times, events):
    order = np.argsort(times)
    times, events = np.asarray(times)[order], np.asarray(events)[order]
    n = len(times)
    uniq = np.unique(times[events == 1])
    surv, at_risk, t_steps, s_steps = 1.0, n, [0.0], [1.0]
    for t in uniq:
        d = int(np.sum((times == t) & (events == 1)))
        risk = int(np.sum(times >= t))
        if risk > 0:
            surv *= (1 - d / risk)
        t_steps.append(float(t)); s_steps.append(surv)
    return t_steps, s_steps


def _logrank_p(t1, e1, t2, e2):
    from scipy.stats import chi2

    times = np.concatenate([t1, t2]); events = np.concatenate([e1, e2])
    group = np.concatenate([np.zeros(len(t1)), np.ones(len(t2))])
    uniq = np.unique(times[events == 1])
    O1 = E1 = V = 0.0
    for t in uniq:
        n = np.sum(times >= t); n1 = np.sum((times >= t) & (group == 0))
        d = np.sum((times == t) & (events == 1)); d1 = np.sum((times == t) & (events == 1) & (group == 0))
        if n <= 1:
            continue
        O1 += d1; E1 += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if V == 0:
        return float("nan")
    chi = (O1 - E1) ** 2 / V
    return float(chi2.sf(chi, 1))


def _plot_km(df, out_dir: Path, summary: dict) -> list[Path]:
    import plotly.graph_objects as go

    # Group patients by TF-change direction (needs first/last per patient).
    if not {"patient_id", "timepoint", "tf_pred"}.issubset(df.columns):
        return []
    direction = {}
    for pid, g in df.dropna(subset=["tf_pred"]).sort_values("timepoint").groupby("patient_id"):
        if len(g) >= 2:
            direction[pid] = "down" if (g["tf_pred"].iloc[-1] - g["tf_pred"].iloc[0]) <= 0 else "up"
    surv = df.drop_duplicates("patient_id").dropna(subset=["survival_time", "survival_event"]).copy()
    surv["direction"] = surv["patient_id"].map(direction)
    surv = surv.dropna(subset=["direction"])
    if surv["direction"].nunique() < 2:
        return []
    fig = go.Figure()
    arms = {}
    for d, colour in (("down", "#2a9d8f"), ("up", "#e76f51")):
        g = surv[surv["direction"] == d]
        if len(g) == 0:
            continue
        t, s = _km_curve(g["survival_time"].to_numpy(), g["survival_event"].to_numpy())
        fig.add_trace(go.Scatter(x=t, y=s, mode="lines", line_shape="hv",
                                 line=dict(color=colour), name=f"TF {d} (n={len(g)})"))
        arms[d] = g
    p = (_logrank_p(arms["down"]["survival_time"].to_numpy(), arms["down"]["survival_event"].to_numpy(),
                    arms["up"]["survival_time"].to_numpy(), arms["up"]["survival_event"].to_numpy())
         if {"down", "up"}.issubset(arms) else float("nan"))
    summary["km_logrank_p"] = p
    fig.update_layout(xaxis_title="time", yaxis_title="survival", yaxis_range=[0, 1.02],
                      title=f"Kaplan–Meier by TF change (log-rank p={p:.3g})")
    return [_save(fig, out_dir / "clinical_km.html", None)]
