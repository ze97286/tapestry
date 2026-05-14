#!/usr/bin/env python3
"""Control-calibrated marker selection for weighted UXM NNLS.

This selector targets the actual failure mode we observed: markers can look
good by atlas geometry while still pushing healthy cfDNA controls into the OAC
column.  It ranks OAC markers by a composite score:

  * OAC atlas separation from the non-OAC hull
  * healthy-control observed signal in the OAC direction (penalty)
  * optional correlation of marker signal with ichorCNA in cancer samples
  * empirical coverage/observability

It also keeps a configurable backbone of non-target markers so the selected
atlas remains a deconvolution atlas rather than an OAC-only detector.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

META_COLS = {
    "#chr", "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
    "target", "name", "direction", "region", "lenCpG", "bp",
    "tg_mean", "bg_mean", "dela_means", "delta_quants",
    "delta_maxmin", "ttest",
}
KEY_CANDIDATES = [
    ["name"],
    ["chr", "start", "end"],
    ["chr", "startCpG", "endCpG"],
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        sep = "," if path.suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    if "#chr" in df.columns and "chr" not in df.columns:
        df = df.rename(columns={"#chr": "chr"})
    return df


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def infer_cell_types(atlas: pd.DataFrame, explicit: list[str]) -> list[str]:
    if explicit:
        missing = [ct for ct in explicit if ct not in atlas.columns]
        if missing:
            raise ValueError(f"cell-type columns not found in atlas: {missing}")
        return explicit

    if "direction" in atlas.columns:
        after_direction = atlas.columns.tolist()[atlas.columns.get_loc("direction") + 1:]
        inferred = [
            c for c in after_direction
            if c not in META_COLS and pd.api.types.is_numeric_dtype(atlas[c])
        ]
        if inferred:
            return inferred

    inferred = [
        c for c in atlas.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(atlas[c])
    ]
    if not inferred:
        raise ValueError("no cell-type columns inferred; pass --cell-types")
    return inferred


def choose_key(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    for key in KEY_CANDIDATES:
        if all(c in left.columns and c in right.columns for c in key):
            return key
    raise ValueError(
        "could not align marker matrix to atlas; expected shared key "
        "one of: name, chr/start/end, chr/startCpG/endCpG"
    )


def align_matrix(atlas: pd.DataFrame, matrix: pd.DataFrame, label: str) -> pd.DataFrame:
    key = choose_key(atlas, matrix)
    sample_cols = [c for c in matrix.columns if c not in META_COLS and c not in key]
    if not sample_cols:
        raise ValueError(f"{label} matrix has no sample columns")
    merged = atlas[key].merge(matrix[key + sample_cols], on=key, how="left", sort=False)
    missing_rows = int(merged[sample_cols].isna().all(axis=1).sum())
    if missing_rows:
        logger.warning("%s matrix is missing %d / %d atlas rows", label, missing_rows, len(atlas))
    return merged[sample_cols]


def load_ichor(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    path_obj = Path(path)
    if path_obj.suffix == ".json":
        with open(path_obj) as handle:
            data = json.load(handle)
        return {
            str(sample): float(entry.get("tumour_fraction", entry.get("tf")))
            for sample, entry in data.items()
            if entry.get("tumour_fraction", entry.get("tf")) is not None
        }
    df = read_table(path_obj)
    sample_col = df.columns[0]
    if "tumour_fraction" in df.columns:
        tf_col = "tumour_fraction"
    elif "tf" in df.columns:
        tf_col = "tf"
    else:
        tf_col = df.columns[1]
    return {str(row[sample_col]): float(row[tf_col]) for _, row in df.iterrows()}


def match_ichor(sample_names: list[str], ichor: dict[str, float]) -> np.ndarray:
    values = np.full(len(sample_names), np.nan, dtype=np.float64)
    items = list(ichor.items())
    for i, sample in enumerate(sample_names):
        for key, value in items:
            if sample == key or sample in key or key in sample:
                values[i] = value
                break
    return values


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    ranks = np.zeros(values.shape, dtype=np.float64)
    if finite.sum() == 0:
        return ranks
    order = np.argsort(values[finite], kind="mergesort")
    ranked = np.empty(order.shape[0], dtype=np.float64)
    if len(order) == 1:
        ranked[order] = 1.0
    else:
        ranked[order] = np.linspace(0.0, 1.0, len(order))
    ranks[finite] = ranked
    return ranks


def safe_corr_matrix(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlation of each marker row with y across samples."""
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    for i in range(values.shape[0]):
        x = values[i]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            continue
        xv = x[valid]
        yv = y[valid]
        if np.nanstd(xv) <= 0 or np.nanstd(yv) <= 0:
            continue
        out[i] = float(np.corrcoef(xv, yv)[0, 1])
    return out


def outside_hull_1d(reference_profiles: np.ndarray, target_idx: int) -> np.ndarray:
    target = reference_profiles[target_idx]
    others = np.delete(reference_profiles, target_idx, axis=0)
    low = np.nanmin(others, axis=0)
    high = np.nanmax(others, axis=0)
    return np.maximum.reduce([target - high, low - target, np.zeros_like(target)])


def compute_candidate_scores(
    atlas: pd.DataFrame,
    cell_types: list[str],
    target_cell_type: str,
    marker_values: pd.DataFrame,
    coverage: pd.DataFrame,
    control_pattern: str,
    ichor: dict[str, float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    reference = atlas[cell_types].to_numpy(dtype=np.float64).T
    target_idx = cell_types.index(target_cell_type)
    target = reference[target_idx]
    others = np.delete(reference, target_idx, axis=0)
    bg_center = np.nanmedian(others, axis=0)
    target_delta = target - bg_center
    target_direction = np.where(target_delta >= 0, 1.0, -1.0)
    target_abs_delta = np.abs(target_delta)
    target_outside_hull = outside_hull_1d(reference, target_idx)

    X = marker_values.to_numpy(dtype=np.float64)
    cov = coverage.to_numpy(dtype=np.float64)
    sample_names = list(marker_values.columns)
    control_re = re.compile(control_pattern)
    is_control = np.array([bool(control_re.search(str(s))) for s in sample_names])
    if is_control.sum() == 0:
        raise ValueError(f"no controls matched --control-pattern {control_pattern!r}")

    observed = (cov > 0) & np.isfinite(X)
    control_obs = observed[:, is_control]
    noncontrol_obs = observed[:, ~is_control]
    control_observed_frac = control_obs.mean(axis=1)
    noncontrol_observed_frac = (
        noncontrol_obs.mean(axis=1) if noncontrol_obs.shape[1] else np.zeros(len(atlas))
    )
    control_median_cov = np.nanmedian(np.where(control_obs, cov[:, is_control], np.nan), axis=1)
    noncontrol_median_cov = (
        np.nanmedian(np.where(noncontrol_obs, cov[:, ~is_control], np.nan), axis=1)
        if noncontrol_obs.shape[1]
        else np.full(len(atlas), np.nan)
    )

    denom = np.maximum(target_abs_delta, args.min_target_delta_floor)
    control_projection = (
        target_direction[:, None] * (X[:, is_control] - bg_center[:, None]) / denom[:, None]
    )
    control_projection = np.where(control_obs, control_projection, np.nan)
    control_proj_mean = np.nanmean(control_projection, axis=1)
    control_proj_p95 = np.nanquantile(control_projection, 0.95, axis=1)
    control_penalty = np.maximum(control_proj_p95, 0.0)

    ichor_values = match_ichor(sample_names, ichor)
    if np.isfinite(ichor_values).sum() >= 3:
        projected = target_direction[:, None] * X
        ichor_r = safe_corr_matrix(projected, ichor_values)
    else:
        ichor_r = np.full(len(atlas), np.nan, dtype=np.float64)

    sep_metric = np.where(target_outside_hull > 0, target_outside_hull, 0.25 * target_abs_delta)
    sep_rank = percentile_rank(sep_metric)
    ichor_rank = percentile_rank(np.maximum(ichor_r, 0.0))
    coverage_rank = percentile_rank(np.log1p(np.nan_to_num(control_median_cov, nan=0.0)))
    control_penalty_rank = percentile_rank(control_penalty)

    missing_penalty = 1.0 - control_observed_frac
    calibrated_score = (
        args.separation_weight * sep_rank
        + args.ichor_weight * ichor_rank
        + args.coverage_weight * coverage_rank
        - args.control_penalty_weight * control_penalty_rank
        - args.missing_penalty_weight * missing_penalty
    )

    diagnostics = pd.DataFrame({
        "marker_index": np.arange(len(atlas)),
        "target_cell_type": target_cell_type,
        "target_value": target,
        "background_median": bg_center,
        "target_delta": target_delta,
        "target_abs_delta": target_abs_delta,
        "target_outside_hull": target_outside_hull,
        "control_projection_mean": control_proj_mean,
        "control_projection_p95": control_proj_p95,
        "control_penalty": control_penalty,
        "control_observed_frac": control_observed_frac,
        "noncontrol_observed_frac": noncontrol_observed_frac,
        "control_median_cov": control_median_cov,
        "noncontrol_median_cov": noncontrol_median_cov,
        "ichor_marker_r": ichor_r,
        "calibrated_score": calibrated_score,
    })
    for col in ["chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction"]:
        if col in atlas.columns:
            diagnostics[col] = atlas[col].values
    return diagnostics, reference, sample_names


def select_markers(
    atlas: pd.DataFrame,
    cell_types: list[str],
    reference: np.ndarray,
    scores: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target_idx = cell_types.index(args.target_cell_type)
    selected: set[int] = set()
    role: dict[int, list[str]] = {}

    for i, ct in enumerate(cell_types):
        if ct == args.target_cell_type:
            continue
        cell_scores = outside_hull_1d(reference, i)
        eligible = np.where(np.isfinite(cell_scores) & (cell_scores > 0))[0]
        ranked = eligible[np.argsort(cell_scores[eligible])[::-1]]
        added = 0
        for marker in ranked:
            marker = int(marker)
            selected.add(marker)
            role.setdefault(marker, []).append(f"backbone:{ct}")
            added += 1
            if added >= args.top_backbone_per_cell:
                break

    target_candidates = scores.copy()
    target_candidates = target_candidates[
        (target_candidates["target_abs_delta"] >= args.min_target_delta)
        & (target_candidates["control_observed_frac"] >= args.min_control_observed_frac)
        & (
            target_candidates["control_median_cov"].fillna(0.0)
            >= args.min_control_median_cov
        )
    ]
    if target_candidates.empty:
        logger.warning("No target candidates passed filters; relaxing all target filters")
        target_candidates = scores.copy()
    target_candidates = target_candidates.sort_values("calibrated_score", ascending=False)
    for marker in target_candidates["marker_index"].head(args.top_target):
        marker = int(marker)
        selected.add(marker)
        role.setdefault(marker, []).append("target_calibrated")

    if args.n_final and len(selected) < args.n_final:
        for marker in target_candidates["marker_index"]:
            marker = int(marker)
            selected.add(marker)
            role.setdefault(marker, []).append("score_fill")
            if len(selected) >= args.n_final:
                break

    selected_indices = sorted(selected)
    selected_df = atlas.iloc[selected_indices].copy()
    annotation = scores.set_index("marker_index").loc[selected_indices].reset_index()
    annotation["selection_role"] = [
        ",".join(role.get(int(idx), [])) for idx in annotation["marker_index"]
    ]

    target_selected = annotation["selection_role"].str.contains("target_calibrated").sum()
    diagnostics = {
        "target_cell_type": args.target_cell_type,
        "cell_types": cell_types,
        "n_atlas_rows_valid": int(len(atlas)),
        "n_final_markers": int(len(selected_indices)),
        "n_target_calibrated_markers": int(target_selected),
        "top_backbone_per_cell": int(args.top_backbone_per_cell),
        "top_target": int(args.top_target),
        "n_final_requested": int(args.n_final) if args.n_final else None,
        "control_pattern": args.control_pattern,
        "score_weights": {
            "separation": args.separation_weight,
            "ichor": args.ichor_weight,
            "coverage": args.coverage_weight,
            "control_penalty": args.control_penalty_weight,
            "missing_penalty": args.missing_penalty_weight,
        },
        "selected_control_projection_p95_mean": float(annotation["control_projection_p95"].mean()),
        "selected_target_outside_hull_mean": float(annotation["target_outside_hull"].mean()),
        "selected_calibrated_score_mean": float(annotation["calibrated_score"].mean()),
    }
    return selected_df, annotation, diagnostics


def write_bed(markers: pd.DataFrame, output: Path) -> None:
    required = ["chr", "start", "end", "startCpG", "endCpG"]
    missing = [c for c in required if c not in markers.columns]
    if missing:
        raise ValueError(f"cannot write BED; missing columns: {missing}")
    bed = markers[required].sort_values(["startCpG", "chr", "start"], kind="mergesort")
    output.parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(output, sep="\t", header=False, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select markers with explicit healthy-control OAC penalty.",
    )
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--marker-values", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bed-output", default=None)
    parser.add_argument("--candidate-output", default=None)
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--cell-types", default=None)
    parser.add_argument("--target-cell-type", default="OAC")
    parser.add_argument("--control-pattern", default=r"Ctrl|healthy|^(GI|SCAN)")
    parser.add_argument("--ichorcna-file", default=None)
    parser.add_argument("--top-backbone-per-cell", type=int, default=50)
    parser.add_argument("--top-target", type=int, default=250)
    parser.add_argument("--n-final", type=int, default=1200)
    parser.add_argument("--min-target-delta", type=float, default=0.02)
    parser.add_argument("--min-target-delta-floor", type=float, default=1e-3)
    parser.add_argument("--min-control-observed-frac", type=float, default=0.6)
    parser.add_argument("--min-control-median-cov", type=float, default=1.0)
    parser.add_argument("--separation-weight", type=float, default=1.0)
    parser.add_argument("--ichor-weight", type=float, default=0.5)
    parser.add_argument("--coverage-weight", type=float, default=0.25)
    parser.add_argument("--control-penalty-weight", type=float, default=1.0)
    parser.add_argument("--missing-penalty-weight", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    atlas_raw = read_table(args.atlas)
    cell_types = infer_cell_types(atlas_raw, parse_csv(args.cell_types))
    if args.target_cell_type not in cell_types:
        raise ValueError(f"{args.target_cell_type!r} not found in cell types: {cell_types}")

    finite_atlas = np.isfinite(atlas_raw[cell_types].to_numpy(dtype=np.float64)).all(axis=1)
    atlas = atlas_raw.loc[finite_atlas].reset_index(drop=True)
    logger.info("Atlas rows after NaN filtering: %d / %d", len(atlas), len(atlas_raw))
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)

    marker_values = align_matrix(atlas, read_table(args.marker_values), "marker-values")
    coverage = align_matrix(atlas, read_table(args.coverage), "coverage")
    if marker_values.columns.tolist() != coverage.columns.tolist():
        raise ValueError("marker-values and coverage sample columns are not aligned")
    logger.info("Aligned marker matrix: %d markers x %d samples", *marker_values.shape)

    ichor = load_ichor(args.ichorcna_file)
    if ichor:
        logger.info("Loaded ichorCNA values for %d samples", len(ichor))

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
    selected, annotation, diagnostics = select_markers(
        atlas=atlas,
        cell_types=cell_types,
        reference=reference,
        scores=scores,
        args=args,
    )
    diagnostics["n_samples"] = len(sample_names)
    diagnostics["n_controls"] = int(
        sum(bool(re.search(args.control_pattern, str(s))) for s in sample_names)
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, sep="\t", index=False)

    candidate_output = (
        Path(args.candidate_output)
        if args.candidate_output
        else output.with_name(output.stem + ".candidate_scores.tsv")
    )
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    scores.sort_values("calibrated_score", ascending=False).to_csv(
        candidate_output, sep="\t", index=False
    )

    annotation_output = output.with_name(output.stem + ".selection_annotations.tsv")
    annotation.to_csv(annotation_output, sep="\t", index=False)

    diagnostics_output = (
        Path(args.diagnostics_output)
        if args.diagnostics_output
        else output.with_name(output.stem + ".diagnostics.json")
    )
    diagnostics_output.write_text(json.dumps(diagnostics, indent=2) + "\n")

    bed_output = Path(args.bed_output) if args.bed_output else output.with_suffix(".bed")
    write_bed(selected, bed_output)

    logger.info("Saved selected markers: %s", output)
    logger.info("Saved marker BED: %s", bed_output)
    logger.info("Saved candidate scores: %s", candidate_output)
    logger.info("Saved annotations: %s", annotation_output)
    logger.info("Saved diagnostics: %s", diagnostics_output)
    logger.info(
        "Selected %d markers; target-calibrated markers=%d; controls=%d",
        diagnostics["n_final_markers"],
        diagnostics["n_target_calibrated_markers"],
        diagnostics["n_controls"],
    )


if __name__ == "__main__":
    main()
