#!/usr/bin/env python3
"""Select UXM markers for unknown-robust weighted NNLS deconvolution.

This selector is intentionally separate from ``scripts/select_markers.py``.
The older selector ranks per-cell-type one-vs-all SNR.  This script scores
marker sets by the geometry of the downstream optimization: target columns
should not be reconstructable from other atlas columns plus an optional
unknown basis, and the final atlas should remain reasonably conditioned.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.markers.robust_selection import (
    greedy_target_marker_selection,
    leave_one_out_separability,
    rank_backbone_candidates,
    single_marker_outside_hull_scores,
    target_separability,
    weighted_condition_number,
)


logger = logging.getLogger(__name__)

META_COLS = {
    "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
    "target", "name", "direction",
    "target_signal", "bg_signal", "snr", "target_total", "bg_total",
    "robust_role", "robust_backbone_for", "robust_target_rank",
    "robust_target_single_marker_score", "robust_marker_index",
}


def parse_csv(value: str | None) -> list[str]:
    if value is None or value == "":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def infer_cell_types(atlas_df: pd.DataFrame, explicit: list[str]) -> list[str]:
    if explicit:
        missing = [ct for ct in explicit if ct not in atlas_df.columns]
        if missing:
            raise ValueError(f"cell type columns not found in atlas: {missing}")
        return explicit

    cell_types = []
    for column in atlas_df.columns:
        if column in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(atlas_df[column]):
            cell_types.append(column)
    if not cell_types:
        raise ValueError("no cell-type columns inferred; pass --cell-types")
    return cell_types


def default_diagnostics_path(output: Path) -> Path:
    suffix = "".join(output.suffixes)
    if suffix:
        return output.with_name(output.name[: -len(suffix)] + ".diagnostics.json")
    return output.with_suffix(".diagnostics.json")


def default_annotation_path(output: Path) -> Path:
    suffix = "".join(output.suffixes)
    if suffix:
        return output.with_name(output.name[: -len(suffix)] + ".selection_annotations.tsv")
    return output.with_suffix(".selection_annotations.tsv")


def load_inputs(args: argparse.Namespace):
    atlas_df = pd.read_csv(args.atlas, sep="\t")
    cell_types = infer_cell_types(atlas_df, parse_csv(args.cell_types))
    target_cell_type = args.target_cell_type
    if target_cell_type not in cell_types:
        raise ValueError(f"target cell type {target_cell_type!r} not in cell types")

    cell_matrix = atlas_df[cell_types].to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(cell_matrix).all(axis=1)
    if args.weight_column:
        if args.weight_column not in atlas_df.columns:
            raise ValueError(f"weight column {args.weight_column!r} not found")
        weights_full = atlas_df[args.weight_column].to_numpy(dtype=np.float64)
        valid_mask &= np.isfinite(weights_full)
    else:
        weights_full = np.ones(len(atlas_df), dtype=np.float64)

    if not np.any(valid_mask):
        raise ValueError("no valid atlas rows after filtering NaNs")

    atlas_valid = atlas_df.loc[valid_mask].reset_index(drop=True)
    reference_profiles = atlas_valid[cell_types].to_numpy(dtype=np.float64).T
    weights = weights_full[valid_mask]

    unknown_basis = None
    if args.unknown_basis_npy:
        unknown_full = np.load(args.unknown_basis_npy)
        if unknown_full.ndim != 2 or unknown_full.shape[0] != len(atlas_df):
            raise ValueError(
                "--unknown-basis-npy must have shape (atlas_rows, K)"
            )
        unknown_basis = unknown_full[valid_mask]

    return atlas_valid, cell_types, reference_profiles, weights, unknown_basis


def build_selection(args: argparse.Namespace):
    atlas_df, cell_types, reference_profiles, weights, unknown_basis = load_inputs(args)
    target_index = cell_types.index(args.target_cell_type)
    excluded_from_backbone = set(parse_csv(args.exclude_from_backbone))
    excluded_from_backbone.add(args.target_cell_type)
    backbone_indices = np.array(
        [i for i, ct in enumerate(cell_types) if ct not in excluded_from_backbone],
        dtype=int,
    )
    if backbone_indices.size < 2:
        raise ValueError("need at least two backbone cell types")

    logger.info("Cell types (%d): %s", len(cell_types), cell_types)
    logger.info("Target cell type: %s", args.target_cell_type)
    logger.info(
        "Backbone cell types (%d): %s",
        len(backbone_indices),
        [cell_types[i] for i in backbone_indices],
    )
    logger.info("Atlas rows after NaN filtering: %d", reference_profiles.shape[1])

    # Backbone: pick markers that make non-target cell types separable from one
    # another, independent of the cancer target.
    backbone_rankings = rank_backbone_candidates(reference_profiles, backbone_indices)
    selected_backbone: list[int] = []
    backbone_for: dict[int, list[str]] = {}
    for idx in backbone_indices:
        scores = single_marker_outside_hull_scores(
            reference_profiles,
            int(idx),
            comparison_indices=[int(j) for j in backbone_indices if int(j) != int(idx)],
        )
        added = 0
        for marker in backbone_rankings[int(idx)]:
            if scores[marker] <= args.min_single_marker_score:
                break
            marker = int(marker)
            if marker not in selected_backbone:
                selected_backbone.append(marker)
            backbone_for.setdefault(marker, []).append(cell_types[int(idx)])
            added += 1
            if added >= args.top_backbone_per_cell:
                break
        logger.info("Backbone %s: selected %d markers", cell_types[int(idx)], added)

    # Target: choose a candidate pool by one-marker target outside-hull score,
    # then greedily optimize set-level target separability.
    target_scores = single_marker_outside_hull_scores(
        reference_profiles,
        target_index,
        comparison_indices=backbone_indices,
    )
    positive_target = np.where(target_scores > args.min_single_marker_score)[0]
    if positive_target.size == 0:
        logger.warning(
            "No target markers had positive one-marker outside-hull score; "
            "using all markers as target candidates"
        )
        positive_target = np.arange(reference_profiles.shape[1])
    target_candidates = positive_target[
        np.argsort(target_scores[positive_target])[::-1]
    ][: args.target_candidate_pool]

    logger.info("Target candidate pool: %d markers", len(target_candidates))
    target_selection = greedy_target_marker_selection(
        reference_profiles,
        target_index,
        target_candidates,
        n_select=args.top_target,
        weights=weights,
        unknown_basis=unknown_basis,
        condition_penalty=args.condition_penalty,
        nonnegative_sum_to_one=not args.cone_reconstruction,
    )

    selected_target = target_selection.selected_indices.tolist()
    final_indices = sorted(set(selected_backbone).union(selected_target))
    final_indices_arr = np.asarray(final_indices, dtype=int)

    target_rank = {marker: rank + 1 for rank, marker in enumerate(selected_target)}
    roles = []
    backbone_labels = []
    target_ranks = []
    for marker in final_indices:
        marker_roles = []
        if marker in backbone_for:
            marker_roles.append("backbone")
        if marker in target_rank:
            marker_roles.append("target")
        roles.append(",".join(marker_roles))
        backbone_labels.append(",".join(backbone_for.get(marker, [])))
        target_ranks.append(target_rank.get(marker, np.nan))

    selected_df = atlas_df.iloc[final_indices].copy()
    annotation_df = atlas_df.iloc[final_indices][
        [c for c in ["chr", "start", "end", "startCpG", "endCpG", "target"] if c in atlas_df.columns]
    ].copy()
    annotation_df.insert(0, "robust_marker_index", final_indices_arr)
    annotation_df["robust_role"] = roles
    annotation_df["robust_backbone_for"] = backbone_labels
    annotation_df["robust_target_rank"] = target_ranks
    annotation_df["robust_target_single_marker_score"] = target_scores[final_indices_arr]

    target_result = target_separability(
        reference_profiles,
        target_index,
        final_indices_arr,
        weights=weights,
        unknown_basis=unknown_basis,
        nonnegative_sum_to_one=not args.cone_reconstruction,
    )
    loo_distances = leave_one_out_separability(
        reference_profiles,
        final_indices_arr,
        weights=weights,
        unknown_basis=unknown_basis,
        nonnegative_sum_to_one=not args.cone_reconstruction,
    )
    condition = weighted_condition_number(
        reference_profiles,
        final_indices_arr,
        weights=weights,
    )

    diagnostics = {
        "target_cell_type": args.target_cell_type,
        "cell_types": cell_types,
        "backbone_cell_types": [cell_types[int(i)] for i in backbone_indices],
        "n_atlas_rows_valid": int(reference_profiles.shape[1]),
        "n_backbone_markers": int(len(set(selected_backbone))),
        "n_target_markers": int(len(set(selected_target))),
        "n_final_markers": int(len(final_indices)),
        "target_distance": target_result.distance,
        "target_relative_distance": target_result.relative_distance,
        "target_reconstruction_success": target_result.success,
        "condition_number": condition,
        "leave_one_out_distance": {
            ct: float(loo_distances[i]) for i, ct in enumerate(cell_types)
        },
        "target_objective_trace": target_selection.objective_trace.tolist(),
        "target_distance_trace": target_selection.distance_trace.tolist(),
        "target_condition_trace": target_selection.condition_trace.tolist(),
        "used_unknown_basis": bool(unknown_basis is not None),
        "reconstruction": "cone" if args.cone_reconstruction else "convex_hull",
    }
    return selected_df, annotation_df, diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Select markers aligned to UXM weighted NNLS + unknown geometry.",
    )
    parser.add_argument("--atlas", required=True, help="Atlas/markers TSV")
    parser.add_argument("--output", required=True, help="Output selected markers TSV")
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--annotation-output", default=None,
                        help="Optional marker annotation TSV. Defaults beside output.")
    parser.add_argument("--cell-types", default=None,
                        help="Comma-separated cell-type columns. Inferred if omitted.")
    parser.add_argument("--target-cell-type", default="OAC")
    parser.add_argument("--exclude-from-backbone", default="",
                        help="Comma-separated additional cell types excluded from backbone")
    parser.add_argument("--unknown-basis-npy", default=None,
                        help="Optional unknown basis .npy aligned to atlas rows, shape (M,K)")
    parser.add_argument("--weight-column", default=None,
                        help="Optional atlas column used as direct marker weight")
    parser.add_argument("--top-backbone-per-cell", type=int, default=100)
    parser.add_argument("--top-target", type=int, default=100)
    parser.add_argument("--target-candidate-pool", type=int, default=300)
    parser.add_argument("--condition-penalty", type=float, default=0.0)
    parser.add_argument("--min-single-marker-score", type=float, default=0.0)
    parser.add_argument("--cone-reconstruction", action="store_true",
                        help="Use cone reconstruction instead of convex-hull reconstruction")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    selected_df, annotation_df, diagnostics = build_selection(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output, sep="\t", index=False)

    annotation_output = (
        Path(args.annotation_output)
        if args.annotation_output
        else default_annotation_path(output)
    )
    annotation_output.parent.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(annotation_output, sep="\t", index=False)

    diagnostics_output = (
        Path(args.diagnostics_output)
        if args.diagnostics_output
        else default_diagnostics_path(output)
    )
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.write_text(json.dumps(diagnostics, indent=2) + "\n")

    logger.info("Saved %d selected markers to %s", len(selected_df), output)
    logger.info("Saved marker annotations to %s", annotation_output)
    logger.info("Saved diagnostics to %s", diagnostics_output)
    logger.info(
        "Target distance %.4g, relative %.4g, condition %.4g",
        diagnostics["target_distance"],
        diagnostics["target_relative_distance"],
        diagnostics["condition_number"],
    )


if __name__ == "__main__":
    main()
