#!/usr/bin/env python3
"""Build marker panels under hard healthy-control constraints.

This is the next experiment after the soft control-penalty selector.  It keeps
the candidate universe fixed and emits several marker panels, each selected
under a hard ceiling on healthy-control OAC-direction signal.

The intended workflow is:

  candidate atlas + marker_values.tsv + coverage.tsv
      -> this script
      -> markers__ctrl_p95_le_*.tsv/.bed panels + sweep_summary.tsv

Each panel can then be evaluated with the existing deconvolution runner.  The
key question is whether any panel can make healthy controls essentially
OAC-negative without destroying OAC/ichor signal.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from select_control_calibrated_markers import (
    align_matrix,
    compute_candidate_scores,
    infer_cell_types,
    load_ichor,
    outside_hull_1d,
    parse_csv,
    read_table,
    write_bed,
)


logger = logging.getLogger(__name__)


def parse_thresholds(value: str) -> list[float]:
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("--control-p95-grid must contain at least one threshold")
    return out


def threshold_label(value: float) -> str:
    if math.isinf(value):
        return "inf"
    text = f"{value:g}".replace("-", "m").replace(".", "p").replace("+", "")
    return text


def finite_valid_atlas(atlas_raw: pd.DataFrame, cell_types: list[str]) -> pd.DataFrame:
    finite = np.isfinite(atlas_raw[cell_types].to_numpy(dtype=np.float64)).all(axis=1)
    return atlas_raw.loc[finite].reset_index(drop=True)


def select_panel(
    atlas: pd.DataFrame,
    cell_types: list[str],
    reference: np.ndarray,
    scores: pd.DataFrame,
    threshold: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select one hard-constrained marker panel."""
    target_idx = cell_types.index(args.target_cell_type)
    target_marker_label = "target_hard_constrained"
    selected: set[int] = set()
    roles: dict[int, list[str]] = {}

    eligible = (
        np.isfinite(scores["control_projection_p95"].to_numpy(dtype=np.float64))
        & (scores["control_projection_p95"].to_numpy(dtype=np.float64) <= threshold)
        & (scores["target_abs_delta"].to_numpy(dtype=np.float64) >= args.min_target_delta)
        & (scores["control_observed_frac"].to_numpy(dtype=np.float64) >= args.min_control_observed_frac)
        & (scores["control_median_cov"].fillna(0.0).to_numpy(dtype=np.float64) >= args.min_control_median_cov)
    )
    eligible_indices = set(scores.loc[eligible, "marker_index"].astype(int).tolist())

    if not eligible_indices:
        logger.warning("threshold %.4g: no eligible markers under hard control constraint", threshold)

    # Backbone selection remains per cell type, but it is restricted to the
    # hard-control-eligible markers.  This avoids quietly reintroducing markers
    # that look OAC-like in controls via non-OAC backbone slots.
    backbone_counts: dict[str, int] = {}
    for i, ct in enumerate(cell_types):
        if ct == args.target_cell_type:
            continue
        cell_scores = outside_hull_1d(reference, i)
        ranked = np.argsort(cell_scores)[::-1]
        added = 0
        for marker in ranked:
            marker = int(marker)
            if marker not in eligible_indices:
                continue
            if not np.isfinite(cell_scores[marker]) or cell_scores[marker] <= args.min_backbone_score:
                continue
            selected.add(marker)
            roles.setdefault(marker, []).append(f"backbone:{ct}")
            added += 1
            if added >= args.top_backbone_per_cell:
                break
        backbone_counts[ct] = added

    target_candidates = scores.loc[list(eligible_indices)].copy() if eligible_indices else scores.iloc[0:0].copy()
    target_candidates = target_candidates.sort_values(
        ["ichor_marker_r", "target_outside_hull", "calibrated_score"],
        ascending=[False, False, False],
    )
    added_target = 0
    for marker in target_candidates["marker_index"]:
        marker = int(marker)
        selected.add(marker)
        roles.setdefault(marker, []).append(target_marker_label)
        added_target += 1
        if added_target >= args.top_target:
            break

    # Fill to N using the same hard-eligible pool.  This preserves the hard
    # constraint even when strict panels cannot hit the requested size.
    if args.n_final and len(selected) < args.n_final:
        fill_candidates = scores.loc[list(eligible_indices)].copy() if eligible_indices else scores.iloc[0:0].copy()
        fill_candidates = fill_candidates.sort_values(
            ["calibrated_score", "ichor_marker_r", "target_outside_hull"],
            ascending=[False, False, False],
        )
        for marker in fill_candidates["marker_index"]:
            marker = int(marker)
            selected.add(marker)
            roles.setdefault(marker, []).append("hard_fill")
            if len(selected) >= args.n_final:
                break

    selected_indices = sorted(selected)
    selected_df = atlas.iloc[selected_indices].copy()
    if selected_indices:
        annotation = scores.set_index("marker_index").loc[selected_indices].reset_index()
    else:
        annotation = scores.iloc[0:0].copy()
    annotation["selection_role"] = [
        ",".join(roles.get(int(idx), [])) for idx in annotation.get("marker_index", pd.Series(dtype=int))
    ]

    target_selected = int(annotation["selection_role"].str.contains(target_marker_label).sum()) if len(annotation) else 0
    summary = {
        "control_projection_p95_threshold": float(threshold),
        "n_eligible_markers": int(len(eligible_indices)),
        "n_final_markers": int(len(selected_indices)),
        "n_target_markers": target_selected,
        "top_backbone_per_cell": int(args.top_backbone_per_cell),
        "top_target": int(args.top_target),
        "n_final_requested": int(args.n_final),
        "min_backbone_selected": int(min(backbone_counts.values())) if backbone_counts else 0,
        "max_backbone_selected": int(max(backbone_counts.values())) if backbone_counts else 0,
        "mean_backbone_selected": float(np.mean(list(backbone_counts.values()))) if backbone_counts else 0.0,
        "selected_control_projection_p95_mean": float(annotation["control_projection_p95"].mean()) if len(annotation) else np.nan,
        "selected_control_projection_p95_max": float(annotation["control_projection_p95"].max()) if len(annotation) else np.nan,
        "selected_target_outside_hull_mean": float(annotation["target_outside_hull"].mean()) if len(annotation) else np.nan,
        "selected_ichor_marker_r_mean": float(annotation["ichor_marker_r"].mean()) if len(annotation) else np.nan,
        "panel_complete": bool(args.n_final == 0 or len(selected_indices) >= args.n_final),
    }
    return selected_df, annotation, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep hard healthy-control constraints for marker selection.",
    )
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--marker-values", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ichorcna-file", default=None)
    parser.add_argument("--cell-types", default=None)
    parser.add_argument("--target-cell-type", default="OAC")
    parser.add_argument("--control-pattern", default=r"Ctrl|healthy|^(GI|SCAN)")
    parser.add_argument("--control-p95-grid", default="0,0.1,0.25,0.5,1,2,5,inf")
    parser.add_argument("--top-backbone-per-cell", type=int, default=50)
    parser.add_argument("--top-target", type=int, default=250)
    parser.add_argument("--n-final", type=int, default=1200)
    parser.add_argument("--min-target-delta", type=float, default=0.02)
    parser.add_argument("--min-target-delta-floor", type=float, default=1e-3)
    parser.add_argument("--min-backbone-score", type=float, default=0.0)
    parser.add_argument("--min-control-observed-frac", type=float, default=0.6)
    parser.add_argument("--min-control-median-cov", type=float, default=1.0)
    parser.add_argument("--separation-weight", type=float, default=1.0)
    parser.add_argument("--ichor-weight", type=float, default=0.5)
    parser.add_argument("--coverage-weight", type=float, default=0.25)
    parser.add_argument("--control-penalty-weight", type=float, default=1.0)
    parser.add_argument("--missing-penalty-weight", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas_raw = read_table(args.atlas)
    cell_types = infer_cell_types(atlas_raw, parse_csv(args.cell_types))
    if args.target_cell_type not in cell_types:
        raise ValueError(f"{args.target_cell_type!r} not found in cell types: {cell_types}")
    atlas = finite_valid_atlas(atlas_raw, cell_types)
    logger.info("Atlas rows after NaN filtering: %d / %d", len(atlas), len(atlas_raw))
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)

    marker_values = align_matrix(atlas, read_table(args.marker_values), "marker-values")
    coverage = align_matrix(atlas, read_table(args.coverage), "coverage")
    if marker_values.columns.tolist() != coverage.columns.tolist():
        raise ValueError("marker-values and coverage sample columns are not aligned")

    control_re = re.compile(args.control_pattern)
    n_controls = int(sum(bool(control_re.search(str(s))) for s in marker_values.columns))
    if n_controls == 0:
        raise ValueError(f"no controls matched --control-pattern {args.control_pattern!r}")

    ichor = load_ichor(args.ichorcna_file)
    scores, reference, sample_names = compute_candidate_scores(
        atlas=atlas,
        cell_types=cell_types,
        target_cell_type=args.target_cell_type,
        marker_values=marker_values,
        coverage=coverage,
        control_pattern=args.control_pattern,
        ichor=ichor,
        args=args,
    )
    scores_path = output_dir / "candidate_scores.tsv"
    scores.sort_values("calibrated_score", ascending=False).to_csv(scores_path, sep="\t", index=False)
    logger.info("Saved candidate scores to %s", scores_path)

    summaries = []
    for threshold in parse_thresholds(args.control_p95_grid):
        label = threshold_label(threshold)
        panel_dir = output_dir / f"ctrl_p95_le_{label}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        markers_tsv = panel_dir / "markers.tsv"
        markers_bed = panel_dir / "markers.bed"
        annotation_tsv = panel_dir / "markers.selection_annotations.tsv"
        diagnostics_json = panel_dir / "markers.diagnostics.json"

        selected, annotation, summary = select_panel(
            atlas=atlas,
            cell_types=cell_types,
            reference=reference,
            scores=scores,
            threshold=threshold,
            args=args,
        )
        selected.to_csv(markers_tsv, sep="\t", index=False)
        annotation.to_csv(annotation_tsv, sep="\t", index=False)
        write_bed(selected, markers_bed)

        summary.update({
            "panel": f"ctrl_p95_le_{label}",
            "markers_tsv": str(markers_tsv),
            "markers_bed": str(markers_bed),
            "annotation_tsv": str(annotation_tsv),
            "target_cell_type": args.target_cell_type,
            "cell_types": cell_types,
            "n_samples": len(sample_names),
            "n_controls": n_controls,
        })
        diagnostics_json.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        logger.info(
            "%s: eligible=%d selected=%d target=%d complete=%s",
            summary["panel"],
            summary["n_eligible_markers"],
            summary["n_final_markers"],
            summary["n_target_markers"],
            summary["panel_complete"],
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "sweep_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logger.info("Saved sweep summary to %s", summary_path)


if __name__ == "__main__":
    main()
