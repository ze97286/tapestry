"""Plotly dashboards for each pipeline step (self-contained interactive HTML).

Matches the per-read-call substrate: discovery writes per-CpG profiles, the oracle
writes per-fragment calibrated z, the detector writes per-sample features + OOF
detection scores, and clinical writes the merged detection/survival table. Each
function is defensive (skips a panel when its data is missing) and inlines
plotly.js so the artifacts open offline after copying off the cluster.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_TUM, _HEA, _ACC, _DARK = "#e76f51", "#457b9d", "#2a9d8f", "#264653"


def _save(fig, path: Path, title: str | None = None) -> Path:
    if title:
        fig.update_layout(title=title)
    fig.update_layout(template="plotly_white", margin=dict(t=60, l=60, r=30, b=50))
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    logger.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Step 1 — discovery (cpgs.tsv: chrom,pos,meth_*,total_*,p_tumour,p_healthy)
# ---------------------------------------------------------------------------

def plot_discovery(cpgs_df, out_dir: str | Path) -> list[Path]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = cpgs_df.copy()
    df["effect"] = (df["p_tumour"] - df["p_healthy"]).abs()
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Panel CpGs: tumour vs healthy methylation", "Effect size |p_tumour − p_healthy|",
        "Panel CpGs per chromosome", "Per-CpG reference coverage"))

    fig.add_trace(go.Scattergl(
        x=df["p_healthy"], y=df["p_tumour"], mode="markers",
        marker=dict(size=5, color=df["effect"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="effect", x=0.46, y=0.8, len=0.4)),
        hovertext=df["chrom"].astype(str) + ":" + df["pos"].astype(str), name="CpGs"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="grey"), showlegend=False), row=1, col=1)
    fig.update_xaxes(title_text="healthy methylation", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="tumour methylation", range=[0, 1], row=1, col=1)

    fig.add_trace(go.Histogram(x=df["effect"], nbinsx=40, marker_color=_ACC, showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="effect", row=1, col=2)

    by_chrom = df.groupby("chrom").size()
    order = sorted(by_chrom.index, key=lambda c: (len(str(c)), str(c)))
    fig.add_trace(go.Bar(x=[str(c) for c in order], y=[int(by_chrom[c]) for c in order],
                         marker_color=_DARK, showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="# CpGs", row=2, col=1)

    fig.add_trace(go.Histogram(x=df["total_tumour"], nbinsx=40, name="tumour", marker_color=_TUM, opacity=0.65), row=2, col=2)
    fig.add_trace(go.Histogram(x=df["total_healthy"], nbinsx=40, name="healthy", marker_color=_HEA, opacity=0.65), row=2, col=2)
    fig.update_xaxes(title_text="reads / CpG", row=2, col=2)
    fig.update_layout(barmode="overlay", height=820)
    return [_save(fig, out_dir / "discovery.html", f"Panel discovery — {len(df)} CpGs")]


# ---------------------------------------------------------------------------
# Step 2 — oracle (per_fragment: z,n_cpg,read_length,label,cohort,sample_id)
# ---------------------------------------------------------------------------

def plot_oracle(reads_df, metrics: dict, out_dir: str | Path) -> list[Path]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import roc_curve

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = reads_df.dropna(subset=["z"])
    t, h = df[df["label"] == 1], df[df["label"] == 0]
    auc, perm = metrics.get("auc", float("nan")), metrics.get("auc_permuted", float("nan"))
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        f"Per-read z: tumour vs healthy (AUC={auc:.3f}, perm={perm:.3f})", "ROC",
        "Null mean z by #CpG — calibration check (want ≈ 0)", "Mean z per held-out healthy sample"))

    lo, hi = (np.percentile(df["z"], [0.5, 99.5]) if len(df) else (-3, 3))
    bins = np.linspace(lo, hi, 60)
    for sub, name, colour in ((h, "healthy", _HEA), (t, "tumour", _TUM)):
        if len(sub):
            fig.add_trace(go.Histogram(x=sub["z"], xbins=dict(start=lo, end=hi, size=(hi - lo) / 60),
                                       histnorm="probability density", name=name, marker_color=colour, opacity=0.6), row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="grey", row=1, col=1)
    fig.update_xaxes(title_text="calibrated z", row=1, col=1)

    if df["label"].nunique() == 2:
        fpr, tpr, _ = roc_curve(df["label"], df["z"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=_ACC), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="FPR", row=1, col=2); fig.update_yaxes(title_text="TPR", row=1, col=2)

    nmz = metrics.get("null_mean_z_by_n_cpg", {})
    if nmz:
        fig.add_trace(go.Bar(x=list(nmz.keys()), y=list(nmz.values()), marker_color=_HEA, showlegend=False), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.update_xaxes(title_text="#CpG bucket", row=2, col=1); fig.update_yaxes(title_text="mean z (healthy)", row=2, col=1)

    if len(h):
        per = h.groupby("sample_id")["z"].mean().sort_values()
        fig.add_trace(go.Bar(x=[str(s) for s in per.index], y=per.values, marker_color=_HEA, showlegend=False), row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=2)
        if np.isfinite(metrics.get("mean_z_tumour", float("nan"))):
            fig.add_hline(y=metrics["mean_z_tumour"], line_dash="dot", line_color=_TUM, row=2, col=2,
                          annotation_text="tumour mean")
    fig.update_yaxes(title_text="mean z", row=2, col=2); fig.update_xaxes(tickangle=45, row=2, col=2)
    fig.update_layout(barmode="overlay", height=860)
    return [_save(fig, out_dir / "oracle.html", "Read-level separability oracle")]


# ---------------------------------------------------------------------------
# Step 3 — detector (features.tsv + classification_oof.tsv + summary)
# ---------------------------------------------------------------------------

def plot_detector(features_df, cls_oof, reg_oof, summary: dict, out_dir: str | Path) -> list[Path]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import roc_curve

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cm = summary.get("classification", {})
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        f"ROC — AUC={cm.get('auc', float('nan')):.3f}", "Cancer probability by class",
        "Tumour-pattern read fraction (frac_z_gt_2) by class", "frac_z_gt_2 by cohort"))

    thr = cm.get("threshold_at_spec_0.95")
    if cls_oof is not None and cls_oof["is_cancer"].nunique() == 2:
        fpr, tpr, _ = roc_curve(cls_oof["is_cancer"], cls_oof["pred_proba_cancer"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=_ACC), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False), row=1, col=1)
        for lab, colour in ((0, _HEA), (1, _TUM)):
            sub = cls_oof[cls_oof["is_cancer"] == lab]
            fig.add_trace(go.Box(y=sub["pred_proba_cancer"], name=("healthy" if lab == 0 else "cancer"),
                                 marker_color=colour, boxpoints="all", jitter=0.4), row=1, col=2)
        if thr is not None and np.isfinite(thr):
            fig.add_hline(y=float(thr), line_dash="dash", line_color="black", row=1, col=2, annotation_text="spec 0.95")
    fig.update_xaxes(title_text="FPR", row=1, col=1); fig.update_yaxes(title_text="TPR", row=1, col=1)
    fig.update_yaxes(title_text="P(cancer)", row=1, col=2)

    feat_col = "frac_z_gt_2" if (features_df is not None and "frac_z_gt_2" in features_df) else "mean_z"
    if features_df is not None and "is_cancer" in features_df and feat_col in features_df:
        for lab, colour in ((0, _HEA), (1, _TUM)):
            sub = features_df[features_df["is_cancer"] == lab]
            fig.add_trace(go.Box(y=sub[feat_col], name=("healthy" if lab == 0 else "cancer"),
                                 marker_color=colour, boxpoints="all", jitter=0.4, showlegend=False), row=2, col=1)
        fig.update_yaxes(title_text=feat_col, row=2, col=1)
        cohort_col = "cohort" if "cohort" in features_df else None
        if cohort_col:
            for lab, colour in ((0, _HEA), (1, _TUM)):
                sub = features_df[features_df["is_cancer"] == lab]
                fig.add_trace(go.Box(x=sub[cohort_col], y=sub[feat_col], name=("healthy" if lab == 0 else "cancer"),
                                     marker_color=colour, boxpoints="all", jitter=0.3, showlegend=False), row=2, col=2)
            fig.update_yaxes(title_text=feat_col, row=2, col=2); fig.update_xaxes(title_text="cohort", row=2, col=2)
    fig.update_layout(height=860, showlegend=False, boxmode="group")
    return [_save(fig, out_dir / "detector.html", "Read-level cancer detector")]


# ---------------------------------------------------------------------------
# Step 4 — clinical (merged: pred_proba_cancer, is_cancer, ichorcna_tf, os_*, ...)
# ---------------------------------------------------------------------------

def plot_clinical(merged, summary: dict, out_dir: str | Path, threshold: float | None = None) -> list[Path]:
    import plotly.express as px
    import plotly.graph_objects as go

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = merged
    paths: list[Path] = []

    # (a) ichorCNA sanity: detection score vs ichorCNA TF (validation only)
    if {"ichorcna_tf", "pred_proba_cancer", "is_cancer"}.issubset(df.columns):
        sub = df[(df["is_cancer"] == 1)].dropna(subset=["ichorcna_tf", "pred_proba_cancer"])
        if len(sub) >= 3:
            r = summary.get("ichorcna_sanity", {}).get("pearson_r")
            fig = px.scatter(sub, x="ichorcna_tf", y="pred_proba_cancer",
                             color="cohort" if "cohort" in sub else None,
                             hover_data=[c for c in ("sample_id", "patient_id") if c in sub])
            fig.update_xaxes(title_text="ichorCNA tumour fraction")
            fig.update_yaxes(title_text="detection score P(cancer)")
            paths.append(_save(fig, out_dir / "clinical_ichorcna_sanity.html",
                               "Detection score vs ichorCNA" + (f" (r={r:.3f})" if r is not None else "")))

    # (b) detection scores by class with the specificity threshold
    if {"pred_proba_cancer", "is_cancer"}.issubset(df.columns):
        sub = df.dropna(subset=["pred_proba_cancer"])
        fig = px.strip(sub, x="is_cancer", y="pred_proba_cancer", color="is_cancer",
                       hover_data=[c for c in ("sample_id", "patient_id", "cohort") if c in sub])
        if threshold is not None and np.isfinite(threshold):
            fig.add_hline(y=float(threshold), line_dash="dash", line_color="black", annotation_text="detection threshold")
        fig.update_layout(xaxis_title="is_cancer (0=healthy, 1=cancer)", yaxis_title="P(cancer)")
        paths.append(_save(fig, out_dir / "clinical_detection.html", "Detection calls at controlled specificity"))

    # (c) Kaplan–Meier by detection-score split (cancer patients with survival)
    if {"pred_proba_cancer", "os_days", "os_event", "is_cancer"}.issubset(df.columns):
        surv = df[df["is_cancer"] == 1].dropna(subset=["pred_proba_cancer", "os_days", "os_event"])
        if "patient_id" in surv:
            surv = surv.drop_duplicates("patient_id")
        if len(surv) >= 6:
            med = float(surv["pred_proba_cancer"].median())
            fig = go.Figure()
            arms = {}
            for name, colour, mask in (("low score", _HEA, surv["pred_proba_cancer"] < med),
                                       ("high score", _TUM, surv["pred_proba_cancer"] >= med)):
                g = surv[mask]
                if len(g) == 0:
                    continue
                ts, ss = _km_curve(g["os_days"].to_numpy(), g["os_event"].to_numpy())
                fig.add_trace(go.Scatter(x=ts, y=ss, mode="lines", line_shape="hv",
                                         line=dict(color=colour), name=f"{name} (n={len(g)})"))
                arms[name] = g
            p = summary.get("km_logrank_p")
            if p is None and {"low score", "high score"}.issubset(arms):
                p = _logrank_p(arms["low score"]["os_days"].to_numpy(), arms["low score"]["os_event"].to_numpy(),
                               arms["high score"]["os_days"].to_numpy(), arms["high score"]["os_event"].to_numpy())
            fig.update_layout(xaxis_title="overall survival (days)", yaxis_title="survival", yaxis_range=[0, 1.02])
            paths.append(_save(fig, out_dir / "clinical_km.html",
                               f"Kaplan–Meier by detection-score split (log-rank p={p:.3g})" if p is not None else "Kaplan–Meier"))
    return paths


def _km_curve(times, events):
    order = np.argsort(times)
    times, events = np.asarray(times)[order], np.asarray(events)[order]
    n = len(times)
    uniq = np.unique(times[events == 1])
    surv, t_steps, s_steps = 1.0, [0.0], [1.0]
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
