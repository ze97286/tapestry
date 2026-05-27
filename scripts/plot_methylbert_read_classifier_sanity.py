#!/usr/bin/env python3
"""Plot sanity summaries for the MethylBERT read classifier.

The script intentionally uses only the Python standard library so it can run on
BMRC without depending on matplotlib/seaborn. It expects the held-out
`test_predictions.tsv` written by `scripts/evaluate_methylbert_finetune.py`.
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


COLORS = {
    "tumour": "#d95b59",
    "AB_high_cov_N": "#6d5bd0",
    "other_N": "#1aa39b",
    "N": "#1aa39b",
    "T": "#d95b59",
    "specificity": "#1aa39b",
    "sensitivity": "#d95b59",
    "balanced_accuracy": "#1b2d5a",
    "roc": "#1b2d5a",
    "pr": "#6d5bd0",
}


AB_HIGH_COV_DEFAULT = {
    "TP277_Ctrl_plasma_md",
    "X3161_Ctrl_plasma_md",
    "X2881_Ctrl_plasma_md",
    "X3421_Ctrl_plasma_md",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_ab_samples(raw: str) -> set[str]:
    if not raw:
        return set(AB_HIGH_COV_DEFAULT)
    return {part.strip() for part in raw.split(",") if part.strip()}


def read_predictions(path: Path, ab_samples: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"filename", "ctype", "dmr_label", "prob_matching_dmr_ctype"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            label = row["ctype"]
            sample = row["filename"]
            if label == "T":
                group = "tumour"
            elif sample in ab_samples:
                group = "AB_high_cov_N"
            else:
                group = "other_N"
            rows.append(
                {
                    "sample": sample,
                    "label": label,
                    "group": group,
                    "dmr": row["dmr_label"],
                    "prob": float(row["prob_matching_dmr_ctype"]),
                }
            )
    if not rows:
        raise SystemExit(f"no prediction rows found in {path}")
    return rows


def svg_page(width: int, height: int, body: list[str]) -> str:
    style = """
    <style>
      text { font-family: Arial, Helvetica, sans-serif; fill: #111827; letter-spacing: 0; }
      .title { font-size: 24px; font-weight: 800; fill: #1b2d5a; }
      .subtitle { font-size: 15px; fill: #526070; font-weight: 700; }
      .axis { stroke: #1b2d5a; stroke-width: 2; }
      .grid { stroke: #d8deea; stroke-width: 1; }
      .tick { font-size: 12px; fill: #526070; }
      .label { font-size: 14px; fill: #1b2d5a; font-weight: 800; }
      .legend { font-size: 14px; fill: #111827; font-weight: 700; }
    </style>
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"{style}\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.write_text(svg_page(width, height, body))


def add_axes(
    body: list[str],
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    x_label: str,
    y_label: str,
    y_max: float,
    y_label_x: int | None = None,
) -> None:
    if y_label_x is None:
        y_label_x = max(24, left - 60)
    body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        val = y_max * i / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{val:.2f}</text>')
    body.append(f'<text x="{left + plot_w / 2}" y="{top + plot_h + 42}" text-anchor="middle" class="label">{esc(x_label)}</text>')
    body.append(
        f'<text x="{y_label_x}" y="{top + plot_h / 2}" text-anchor="middle" '
        f'transform="rotate(-90 {y_label_x} {top + plot_h / 2})" class="label">{esc(y_label)}</text>'
    )


def classifier_points(rows: list[dict[str, object]]) -> list[tuple[int, float]]:
    return [(1 if row["label"] == "T" else 0, float(row["prob"])) for row in rows]


def auc_trapezoid(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def roc_pr_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    labels_scores = classifier_points(rows)
    positives = sum(label for label, _ in labels_scores)
    negatives = len(labels_scores) - positives
    if positives == 0 or negatives == 0:
        return {
            "roc": [(0.0, 0.0), (1.0, 1.0)],
            "pr": [(0.0, 0.0), (1.0, 0.0)],
            "roc_auc": 0.0,
            "average_precision": 0.0,
        }

    ranked = sorted(labels_scores, key=lambda item: item[1], reverse=True)
    roc = [(0.0, 0.0)]
    pr = [(0.0, 1.0)]
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for label, _score in ranked:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        fpr = fp / negatives
        tpr = recall
        roc.append((fpr, tpr))
        pr.append((recall, precision))
        if label:
            ap += precision * (recall - prev_recall)
            prev_recall = recall
    roc.append((1.0, 1.0))
    return {
        "roc": roc,
        "pr": pr,
        "roc_auc": auc_trapezoid(roc),
        "average_precision": ap,
    }


def threshold_metrics(rows: list[dict[str, object]], thresholds: list[float]) -> list[dict[str, float]]:
    labels_scores = classifier_points(rows)
    positives = sum(label for label, _ in labels_scores)
    negatives = len(labels_scores) - positives
    metrics = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for label, score in labels_scores:
            pred = 1 if score >= threshold else 0
            if label == 1 and pred == 1:
                tp += 1
            elif label == 1:
                fn += 1
            elif pred == 1:
                fp += 1
            else:
                tn += 1
        sensitivity = tp / positives if positives else 0.0
        specificity = tn / negatives if negatives else 0.0
        accuracy = (tp + tn) / max(len(labels_scores), 1)
        metrics.append(
            {
                "threshold": threshold,
                "tp": float(tp),
                "fp": float(fp),
                "tn": float(tn),
                "fn": float(fn),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "balanced_accuracy": (sensitivity + specificity) / 2,
            }
        )
    return metrics


def performance_summary_plot(rows: list[dict[str, object]], output: Path) -> None:
    metrics = roc_pr_metrics(rows)
    at_half = threshold_metrics(rows, [0.5])[0]
    width, height = 1120, 560
    body = [
        '<text x="560" y="34" text-anchor="middle" class="title">Held-out read classifier performance</text>',
        '<text x="560" y="58" text-anchor="middle" class="subtitle">Tumour tissue reads are positives; healthy-control reads are negatives</text>',
    ]

    cards = [
        ("Accuracy", at_half["accuracy"], "#1b2d5a"),
        ("ROC AUC", float(metrics["roc_auc"]), "#1b2d5a"),
        ("Average precision", float(metrics["average_precision"]), "#6d5bd0"),
        ("Sensitivity", at_half["sensitivity"], "#d95b59"),
        ("Specificity", at_half["specificity"], "#1aa39b"),
    ]
    card_w = 196
    for idx, (name, value, color) in enumerate(cards):
        x = 52 + idx * 210
        body.append(f'<rect x="{x}" y="92" width="{card_w}" height="126" rx="10" fill="#f7f9fc" stroke="#cfd7e6" stroke-width="2"/>')
        body.append(f'<text x="{x + 18}" y="126" class="label">{esc(name)}</text>')
        body.append(f'<text x="{x + 18}" y="178" font-size="42" font-weight="800" fill="{color}">{value:.3f}</text>')
        if name in {"Sensitivity", "Specificity", "Accuracy"}:
            body.append(f'<text x="{x + 18}" y="202" class="tick">at threshold 0.5</text>')

    left, top = 110, 282
    cell = 92
    tp = int(at_half["tp"])
    fp = int(at_half["fp"])
    tn = int(at_half["tn"])
    fn = int(at_half["fn"])
    matrix = [
        ("true T", "pred T", tp, "#f8d8d6"),
        ("true T", "pred N", fn, "#fff2cc"),
        ("true N", "pred T", fp, "#fff2cc"),
        ("true N", "pred N", tn, "#d8f0ed"),
    ]
    body.append(f'<text x="{left}" y="{top - 24}" class="label">Confusion matrix at threshold 0.5</text>')
    body.append(f'<text x="{left + cell * 1.5}" y="{top - 4}" text-anchor="middle" class="tick">Predicted</text>')
    body.append(f'<text x="{left - 34}" y="{top + cell}" text-anchor="middle" transform="rotate(-90 {left - 34} {top + cell})" class="tick">True label</text>')
    body.append(f'<text x="{left + cell * 0.5}" y="{top + 18}" text-anchor="middle" class="tick">T</text>')
    body.append(f'<text x="{left + cell * 1.5}" y="{top + 18}" text-anchor="middle" class="tick">N</text>')
    body.append(f'<text x="{left - 10}" y="{top + cell * 0.65}" text-anchor="end" class="tick">T</text>')
    body.append(f'<text x="{left - 10}" y="{top + cell * 1.65}" text-anchor="end" class="tick">N</text>')
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for (_true_label, _pred_label, count, fill), (col, row) in zip(matrix, positions):
        x = left + col * cell
        y = top + 24 + row * cell
        body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#cfd7e6" stroke-width="2"/>')
        body.append(f'<text x="{x + cell / 2}" y="{y + 48}" text-anchor="middle" font-size="24" font-weight="800" fill="#111827">{count:,}</text>')

    write_svg(output, width, height, body)


def roc_pr_plot(rows: list[dict[str, object]], output: Path) -> None:
    metrics = roc_pr_metrics(rows)
    width, height = 1160, 560
    body = [
        '<text x="580" y="34" text-anchor="middle" class="title">Overall discrimination on held-out reads</text>',
        '<text x="580" y="58" text-anchor="middle" class="subtitle">ROC and precision-recall curves from read-level tumour probabilities</text>',
    ]
    panels = [
        ("ROC curve", "False positive rate", "True positive rate", metrics["roc"], f'AUC = {float(metrics["roc_auc"]):.3f}', COLORS["roc"], 78),
        ("Precision-recall curve", "Recall", "Precision", metrics["pr"], f'AP = {float(metrics["average_precision"]):.3f}', COLORS["pr"], 618),
    ]
    for title, x_label, y_label, points, stat_label, color, left in panels:
        top, plot_w, plot_h = 104, 420, 330
        body.append(f'<text x="{left + plot_w / 2}" y="{top - 28}" text-anchor="middle" class="label">{esc(title)}</text>')
        add_axes(body, left, top, plot_w, plot_h, x_label, y_label, 1.0)
        for i in range(6):
            x = left + plot_w * i / 5
            body.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{i / 5:.1f}</text>')
        if title == "ROC curve":
            body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#9aa4b2" stroke-width="2" stroke-dasharray="6 6"/>')
        poly = " ".join(f"{left + plot_w * x:.1f},{top + plot_h - plot_h * y:.1f}" for x, y in points)
        body.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="4"/>')
        body.append(f'<rect x="{left + 18}" y="{top + 18}" width="126" height="42" rx="8" fill="#fff" stroke="#cfd7e6" stroke-width="2"/>')
        body.append(f'<text x="{left + 32}" y="{top + 46}" class="legend">{esc(stat_label)}</text>')
    write_svg(output, width, height, body)


def sensitivity_specificity_plot(rows: list[dict[str, object]], output: Path) -> None:
    thresholds = [i / 100 for i in range(0, 101)]
    metrics = threshold_metrics(rows, thresholds)
    width, height = 1040, 580
    left, top, plot_w, plot_h = 86, 82, 820, 390
    body = [
        '<text x="520" y="34" text-anchor="middle" class="title">Sensitivity/specificity trade-off</text>',
        '<text x="520" y="58" text-anchor="middle" class="subtitle">Performance as the tumour-like probability threshold is varied</text>',
    ]
    add_axes(body, left, top, plot_w, plot_h, "Tumour-like probability threshold", "Metric value", 1.0)
    for i in range(11):
        x = left + plot_w * i / 10
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{i / 10:.1f}</text>')
    for key, label in [
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("balanced_accuracy", "Balanced accuracy"),
    ]:
        points = []
        for row in metrics:
            x = left + plot_w * row["threshold"]
            y = top + plot_h - plot_h * row[key]
            points.append((x, y))
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body.append(f'<polyline points="{poly}" fill="none" stroke="{COLORS[key]}" stroke-width="4"/>')
    threshold_x = left + plot_w * 0.5
    body.append(f'<line x1="{threshold_x:.1f}" y1="{top}" x2="{threshold_x:.1f}" y2="{top + plot_h}" stroke="#111827" stroke-width="2" stroke-dasharray="7 7"/>')
    body.append(f'<text x="{threshold_x + 8:.1f}" y="{top + 18}" class="tick">0.5 threshold</text>')
    legend_x = left + plot_w - 250
    for idx, (key, label) in enumerate(
        [
            ("sensitivity", "Sensitivity"),
            ("specificity", "Specificity"),
            ("balanced_accuracy", "Balanced accuracy"),
        ]
    ):
        y = top + 34 + idx * 26
        body.append(f'<line x1="{legend_x}" y1="{y - 5}" x2="{legend_x + 18}" y2="{y - 5}" stroke="{COLORS[key]}" stroke-width="4"/>')
        body.append(f'<text x="{legend_x + 26}" y="{y}" class="legend">{esc(label)}</text>')
    write_svg(output, width, height, body)


def probability_histogram(rows: list[dict[str, object]], output: Path) -> None:
    groups = ["other_N", "AB_high_cov_N", "tumour"]
    bins = [i / 20 for i in range(21)]
    counts: dict[str, list[int]] = {group: [0] * 20 for group in groups}
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        group = str(row["group"])
        prob = float(row["prob"])
        idx = min(int(prob * 20), 19)
        counts[group][idx] += 1
        totals[group] += 1

    densities: dict[str, list[float]] = {}
    y_max = 0.0
    for group in groups:
        denom = max(totals[group], 1)
        densities[group] = [value / denom for value in counts[group]]
        y_max = max(y_max, max(densities[group], default=0))
    y_max = max(y_max, 0.01)

    width, height = 1120, 620
    left, top, plot_w, plot_h = 82, 86, 940, 420
    body = [
        '<text x="560" y="34" text-anchor="middle" class="title">Held-out read probabilities show a control-domain failure mode</text>',
        '<text x="560" y="58" text-anchor="middle" class="subtitle">Bars show the fraction of reads in each probability bin</text>',
    ]
    add_axes(body, left, top, plot_w, plot_h, "Predicted tumour-like probability", "Fraction of reads", y_max)
    for i in range(11):
        x = left + plot_w * i / 10
        body.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" class="axis"/>')
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{i / 10:.1f}</text>')
    threshold_x = left + plot_w * 0.5
    body.append(f'<line x1="{threshold_x:.1f}" y1="{top}" x2="{threshold_x:.1f}" y2="{top + plot_h}" stroke="#111827" stroke-width="2" stroke-dasharray="7 7"/>')
    body.append(f'<text x="{threshold_x + 8:.1f}" y="{top + 18}" class="tick">0.5 threshold</text>')

    bar_w = plot_w / 20
    offsets = {"other_N": -bar_w * 0.25, "AB_high_cov_N": 0, "tumour": bar_w * 0.25}
    widths = {"other_N": bar_w * 0.28, "AB_high_cov_N": bar_w * 0.28, "tumour": bar_w * 0.28}
    for group in groups:
        color = COLORS[group]
        for idx, value in enumerate(densities[group]):
            x = left + idx * bar_w + bar_w * 0.36 + offsets[group]
            h = plot_h * value / y_max
            y = top + plot_h - h
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{widths[group]:.1f}" '
                f'height="{h:.1f}" fill="{color}" opacity="0.78"/>'
            )
    legend_x = left + plot_w - 260
    for idx, group in enumerate(groups):
        y = top + 34 + idx * 26
        body.append(f'<rect x="{legend_x}" y="{y - 13}" width="16" height="16" fill="{COLORS[group]}"/>')
        label = {"other_N": "other controls", "AB_high_cov_N": "AB high-coverage controls", "tumour": "tumour tissue"}[group]
        body.append(f'<text x="{legend_x + 24}" y="{y}" class="legend">{esc(label)}</text>')
    write_svg(output, width, height, body)


def sample_barplot(rows: list[dict[str, object]], output: Path) -> None:
    by_sample: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "sum": 0.0, "high": 0.0})
    sample_group: dict[str, str] = {}
    for row in rows:
        sample = str(row["sample"])
        prob = float(row["prob"])
        group = str(row["group"])
        by_sample[sample]["n"] += 1
        by_sample[sample]["sum"] += prob
        by_sample[sample]["high"] += 1 if prob >= 0.8 else 0
        sample_group[sample] = group
    samples = sorted(by_sample, key=lambda sample: by_sample[sample]["sum"] / by_sample[sample]["n"], reverse=True)

    width, height = 1320, 620
    left, top, plot_w, plot_h = 90, 82, 1120, 350
    body = [
        '<text x="660" y="34" text-anchor="middle" class="title">False-positive signal is concentrated in four AB controls</text>',
        '<text x="660" y="58" text-anchor="middle" class="subtitle">Sample bars show mean tumour-like probability on held-out reads</text>',
    ]
    add_axes(body, left, top, plot_w, plot_h, "Held-out sample", "Mean probability", 1.0)
    bar_w = plot_w / max(len(samples), 1)
    for idx, sample in enumerate(samples):
        stats = by_sample[sample]
        mean = stats["sum"] / stats["n"]
        x = left + idx * bar_w + 1
        h = plot_h * mean
        y = top + plot_h - h
        group = sample_group[sample]
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 1):.1f}" '
            f'height="{h:.1f}" fill="{COLORS[group]}" opacity="0.86">'
            f'<title>{esc(sample)} mean={mean:.3f}</title></rect>'
        )
        if idx % 3 == 0 or group != "other_N":
            label = sample.replace("_Ctrl_plasma_md", "").replace("_ScrBsl_tumour_md", "")
            body.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 18}" text-anchor="end" '
                f'transform="rotate(-55 {x + bar_w / 2:.1f} {top + plot_h + 18})" class="tick">{esc(label)}</text>'
            )
    threshold_y = top + plot_h * 0.5
    body.append(f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{left + plot_w}" y2="{threshold_y:.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="7 7"/>')
    body.append(f'<text x="{left + plot_w + 8}" y="{threshold_y + 4:.1f}" class="tick">0.5</text>')
    write_svg(output, width, height, body)


def threshold_plot(rows: list[dict[str, object]], output: Path) -> None:
    groups = ["other_N", "AB_high_cov_N", "tumour"]
    thresholds = [i / 10 for i in range(1, 10)]
    frac: dict[str, list[float]] = {group: [] for group in groups}
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        denom = max(len(group_rows), 1)
        for threshold in thresholds:
            frac[group].append(sum(1 for row in group_rows if float(row["prob"]) >= threshold) / denom)

    width, height = 980, 580
    left, top, plot_w, plot_h = 82, 76, 780, 380
    body = [
        '<text x="490" y="34" text-anchor="middle" class="title">Thresholding exposes the AB-control false-positive domain</text>',
        '<text x="490" y="58" text-anchor="middle" class="subtitle">Fraction of reads called tumour-like as the threshold changes</text>',
    ]
    add_axes(body, left, top, plot_w, plot_h, "Tumour-like probability threshold", "Fraction called tumour-like", 1.0)
    for i, threshold in enumerate(thresholds):
        x = left + plot_w * i / (len(thresholds) - 1)
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{threshold:.1f}</text>')
    for group in groups:
        points = []
        for idx, value in enumerate(frac[group]):
            x = left + plot_w * idx / (len(thresholds) - 1)
            y = top + plot_h - plot_h * value
            points.append((x, y))
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body.append(f'<polyline points="{poly}" fill="none" stroke="{COLORS[group]}" stroke-width="4"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[group]}"/>')
    legend_x = left + plot_w - 255
    for idx, group in enumerate(groups):
        y = top + 34 + idx * 26
        body.append(f'<line x1="{legend_x}" y1="{y - 5}" x2="{legend_x + 18}" y2="{y - 5}" stroke="{COLORS[group]}" stroke-width="4"/>')
        label = {"other_N": "other controls", "AB_high_cov_N": "AB high-coverage controls", "tumour": "tumour tissue"}[group]
        body.append(f'<text x="{legend_x + 26}" y="{y}" class="legend">{esc(label)}</text>')
    write_svg(output, width, height, body)


def dmr_group_plot(rows: list[dict[str, object]], output: Path) -> None:
    stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"n": 0, "sum": 0.0})
    for row in rows:
        key = (str(row["dmr"]), str(row["group"]))
        stats[key]["n"] += 1
        stats[key]["sum"] += float(row["prob"])
    dmrs = sorted({str(row["dmr"]) for row in rows}, key=lambda value: int(value) if value.isdigit() else value)
    groups = ["other_N", "AB_high_cov_N", "tumour"]
    means: dict[tuple[str, str], float] = {}
    for dmr in dmrs:
        for group in groups:
            s = stats[(dmr, group)]
            means[(dmr, group)] = s["sum"] / s["n"] if s["n"] else 0.0

    width, height = 1280, 640
    left, top, plot_w, plot_h = 82, 82, 1080, 390
    body = [
        '<text x="640" y="34" text-anchor="middle" class="title">AB controls are tumour-like across the DMR panel, not at one locus</text>',
        '<text x="640" y="58" text-anchor="middle" class="subtitle">Each point is the mean tumour-like probability for one DMR and group</text>',
    ]
    add_axes(body, left, top, plot_w, plot_h, "DMR rank", "Mean probability", 1.0)
    step = plot_w / max(len(dmrs) - 1, 1)
    for idx in range(0, len(dmrs), 10):
        x = left + idx * step
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{esc(dmrs[idx])}</text>')
    for group in groups:
        points = []
        for idx, dmr in enumerate(dmrs):
            x = left + idx * step
            y = top + plot_h - plot_h * means[(dmr, group)]
            points.append((x, y))
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body.append(f'<polyline points="{poly}" fill="none" stroke="{COLORS[group]}" stroke-width="2.5" opacity="0.86"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{COLORS[group]}" opacity="0.86"/>')
    legend_x = left + plot_w - 260
    for idx, group in enumerate(groups):
        y = top + 34 + idx * 26
        body.append(f'<line x1="{legend_x}" y1="{y - 5}" x2="{legend_x + 18}" y2="{y - 5}" stroke="{COLORS[group]}" stroke-width="4"/>')
        label = {"other_N": "other controls", "AB_high_cov_N": "AB high-coverage controls", "tumour": "tumour tissue"}[group]
        body.append(f'<text x="{legend_x + 26}" y="{y}" class="legend">{esc(label)}</text>')
    write_svg(output, width, height, body)


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    groups = ["other_N", "AB_high_cov_N", "tumour"]
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["group", "n", "mean_prob", "median_prob", "frac_ge_0p5", "frac_ge_0p8"])
        for group in groups:
            values = sorted(float(row["prob"]) for row in rows if row["group"] == group)
            if not values:
                continue
            n = len(values)
            median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
            writer.writerow(
                [
                    group,
                    n,
                    f"{sum(values) / n:.6g}",
                    f"{median:.6g}",
                    f"{sum(value >= 0.5 for value in values) / n:.6g}",
                    f"{sum(value >= 0.8 for value in values) / n:.6g}",
                ]
            )


def write_overall_performance(rows: list[dict[str, object]], output: Path) -> None:
    metrics = roc_pr_metrics(rows)
    thresholds = [i / 10 for i in range(0, 11)]
    threshold_rows = threshold_metrics(rows, thresholds)
    at_half = threshold_metrics(rows, [0.5])[0]
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy_at_0p5", f"{at_half['accuracy']:.8g}"])
        writer.writerow(["sensitivity_at_0p5", f"{at_half['sensitivity']:.8g}"])
        writer.writerow(["specificity_at_0p5", f"{at_half['specificity']:.8g}"])
        writer.writerow(["balanced_accuracy_at_0p5", f"{at_half['balanced_accuracy']:.8g}"])
        writer.writerow(["roc_auc", f"{float(metrics['roc_auc']):.8g}"])
        writer.writerow(["average_precision", f"{float(metrics['average_precision']):.8g}"])
        writer.writerow(["true_positive_at_0p5", int(at_half["tp"])])
        writer.writerow(["false_positive_at_0p5", int(at_half["fp"])])
        writer.writerow(["true_negative_at_0p5", int(at_half["tn"])])
        writer.writerow(["false_negative_at_0p5", int(at_half["fn"])])

    threshold_path = output.with_name(output.stem.replace("overall_performance", "threshold_metrics") + output.suffix)
    with threshold_path.open("w", newline="") as handle:
        fieldnames = ["threshold", "accuracy", "sensitivity", "specificity", "balanced_accuracy", "tp", "fp", "tn", "fn"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(threshold_rows)


def write_index(output_dir: Path, prefix: str, files: list[Path]) -> None:
    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{esc(prefix)} MethylBERT sanity plots</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:26px;max-width:1320px}"
        "h1{font-size:26px;color:#1b2d5a}figure{margin:30px 0}img{max-width:100%;border:1px solid #d8deea}</style>",
        "</head><body>",
        f"<h1>{esc(prefix)} MethylBERT read-classifier sanity plots</h1>",
    ]
    for path in files:
        body.append(f"<figure><h2>{esc(path.name)}</h2><img src='{esc(path.name)}' alt='{esc(path.name)}'></figure>")
    body.append("</body></html>\n")
    (output_dir / f"{prefix}_overview.html").write_text("\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Held-out test_predictions.tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="methylbert_read_classifier")
    parser.add_argument(
        "--ab-high-cov-samples",
        default=",".join(sorted(AB_HIGH_COV_DEFAULT)),
        help="Comma-separated sample names to mark as AB high-coverage controls.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_predictions(Path(args.predictions), read_ab_samples(args.ab_high_cov_samples))

    files = [
        output_dir / f"{args.prefix}_overall_performance.svg",
        output_dir / f"{args.prefix}_roc_pr_curves.svg",
        output_dir / f"{args.prefix}_sensitivity_specificity.svg",
        output_dir / f"{args.prefix}_probability_histogram.svg",
        output_dir / f"{args.prefix}_sample_mean_probability.svg",
        output_dir / f"{args.prefix}_threshold_curve.svg",
        output_dir / f"{args.prefix}_dmr_group_means.svg",
    ]
    performance_summary_plot(rows, files[0])
    roc_pr_plot(rows, files[1])
    sensitivity_specificity_plot(rows, files[2])
    probability_histogram(rows, files[3])
    sample_barplot(rows, files[4])
    threshold_plot(rows, files[5])
    dmr_group_plot(rows, files[6])
    write_summary(rows, output_dir / f"{args.prefix}_group_summary.tsv")
    write_overall_performance(rows, output_dir / f"{args.prefix}_overall_performance.tsv")
    write_index(output_dir, args.prefix, files)

    print(f"wrote MethylBERT sanity plots to {output_dir}")
    print(output_dir / f"{args.prefix}_overview.html")


if __name__ == "__main__":
    main()
