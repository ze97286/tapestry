#!/usr/bin/env python3
"""Robust OAC lower-bound deconvolution for cfDNA cohorts.

This is a detection-oriented prototype. It does not ask for a best point
estimate of OAC; it asks how much OAC is required after allowing residual
variation calibrated from healthy cfDNA controls.
"""

import argparse
import gzip
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.benchmark.nnls import run_weighted_nnls
from tapestry.benchmark.robust_lower_bound import (
    calibrate_robust_lower_bound,
    parse_rank_grid,
    solve_theta_interval,
)

logger = logging.getLogger(__name__)


META_COLS = [
    "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
    "target", "name", "direction",
    "target_signal", "bg_signal", "snr", "target_total", "bg_total",
]


def load_atlas(atlas_path: str, cell_types: list[str]):
    atlas_df = pd.read_csv(atlas_path, sep="\t")
    atlas_ct_cols = [c for c in atlas_df.columns if c not in META_COLS]
    atlas_matrix_full = np.zeros((len(atlas_df), len(cell_types)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        if ct in atlas_ct_cols:
            atlas_matrix_full[:, i] = atlas_df[ct].values.astype(np.float32)

    any_nan = atlas_df[atlas_ct_cols].isna().any(axis=1).values
    valid_indices = np.where(~any_nan)[0]
    atlas_matrix = atlas_matrix_full[valid_indices]
    atlas_coords = [
        (str(atlas_df.iloc[i]["chr"]), int(atlas_df.iloc[i]["start"]))
        for i in valid_indices
    ]
    return atlas_matrix, valid_indices, atlas_coords, len(atlas_df)


def extract_marker_values(homog_path, atlas_coords):
    try:
        opener = gzip.open if homog_path.endswith(".gz") else open
        homog_data = {}
        with opener(homog_path, "rt") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) >= 8:
                    key = (parts[0], int(parts[1]))
                    u = int(parts[5])
                    m_count = int(parts[7])
                    homog_data[key] = (u, m_count)
        M = len(atlas_coords)
        u_fraction = np.zeros(M, dtype=np.float64)
        coverage = np.zeros(M, dtype=np.float64)
        for i, (chrom, start) in enumerate(atlas_coords):
            if (chrom, start) in homog_data:
                u, m_count = homog_data[(chrom, start)]
                total = u + m_count
                coverage[i] = total
                if total > 0:
                    u_fraction[i] = u / total
        return u_fraction, coverage
    except Exception as exc:
        logger.error("Failed to parse %s: %s", homog_path, exc)
        return None


def process_cfdna_sample(pat_path, markers_bed, wgbstools, atlas_coords, tmp_dir):
    sample_name = Path(pat_path).stem.replace(".markers.pat", "").replace(".pat", "")
    cmd = f"{wgbstools} homog -b {markers_bed} -l 4 -o {tmp_dir} {pat_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("homog failed for %s: %s", sample_name, result.stderr)
        return None
    pat_basename = Path(pat_path).name.replace(".pat.gz", "")
    expected = os.path.join(tmp_dir, f"{pat_basename}.uxm.bed.gz")
    return extract_marker_values(expected, atlas_coords)


def write_calibration_outputs(prefix: Path, calibration) -> None:
    summary = {
        "target_cell_type": calibration.target_cell_type,
        "alpha": calibration.alpha,
        "tau": calibration.tau,
        "factor_rank": calibration.factor_rank,
        "n_markers": int(calibration.marker_mask.sum()),
        "n_fit_controls": int(calibration.fit_control_indices.size),
        "n_tau_controls": int(calibration.tau_control_indices.size),
        "n_validation_controls": int(calibration.validation_control_indices.size),
        "validation_containment": calibration.validation_containment,
    }
    with open(prefix.with_suffix(".calibration.json"), "w") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for fold_name, indices, distances in [
        ("tau", calibration.tau_control_indices, calibration.tau_distances),
        ("validation", calibration.validation_control_indices,
         calibration.validation_distances),
    ]:
        for idx, distance in zip(indices, distances):
            rows.append({
                "fold": fold_name,
                "sample_index": int(idx),
                "distance": float(distance),
                "inside_tau": bool(distance <= calibration.tau),
            })
    pd.DataFrame(rows).to_csv(prefix.with_suffix(".calibration_distances.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfdna-dir", required=True)
    parser.add_argument("--markers-bed", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wgbstools", default="wgbstools")
    parser.add_argument("--cohort", default="")
    parser.add_argument("--target-cell-type", default="OAC")
    parser.add_argument("--control-pattern", default=r"_Ctrl_|^Ctrl_|_healthy_")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--factor-rank", default="auto",
                        help="'auto' or an integer factor rank.")
    parser.add_argument("--factor-rank-grid", default="0,3,5,10")
    parser.add_argument("--min-control-coverage-fraction", type=float, default=0.8)
    parser.add_argument("--min-sample-coverage", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    atlas_df_head = pd.read_csv(args.atlas, sep="\t", nrows=0)
    cell_types = sorted([c for c in atlas_df_head.columns if c not in META_COLS])
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)
    if args.target_cell_type not in cell_types:
        raise ValueError(f"target cell type {args.target_cell_type!r} not in atlas")
    target_idx = cell_types.index(args.target_cell_type)

    atlas_matrix, valid_indices, atlas_coords, total_rows = load_atlas(args.atlas, cell_types)
    if len(valid_indices) < total_rows:
        logger.info("Atlas filter: kept %d / %d rows", atlas_matrix.shape[0], total_rows)
    reference_profiles = atlas_matrix.T.astype(np.float64)

    cfdna_dir = Path(args.cfdna_dir)
    pat_files = sorted(cfdna_dir.glob("*.markers.pat.gz"))
    if not pat_files:
        pat_files = sorted(cfdna_dir.glob("*.pat.gz"))
    logger.info("Found %d cfDNA PAT files in %s", len(pat_files), cfdna_dir)

    sample_names = []
    X_all = []
    cov_all = []
    with tempfile.TemporaryDirectory(prefix="tapestry_robust_lb_") as tmp_dir:
        for i, pat_path in enumerate(pat_files):
            sample_name = pat_path.stem.replace(".markers.pat", "").replace(".pat", "")
            logger.info("[%d/%d] homog %s", i + 1, len(pat_files), sample_name)
            result = process_cfdna_sample(
                str(pat_path), args.markers_bed, args.wgbstools, atlas_coords, tmp_dir
            )
            if result is None:
                continue
            u_fraction, coverage = result
            sample_names.append(sample_name)
            X_all.append(u_fraction)
            cov_all.append(coverage)
            for f in Path(tmp_dir).glob("*.uxm.bed.gz"):
                f.unlink()

    X = np.asarray(X_all, dtype=np.float64)
    coverage = np.asarray(cov_all, dtype=np.float64)
    logger.info("Extracted marker values for %d samples", len(sample_names))

    ctrl_re = re.compile(args.control_pattern)
    is_control = np.array([bool(ctrl_re.search(s)) for s in sample_names])
    n_controls = int(is_control.sum())
    logger.info("Healthy controls matched by %r: %d / %d samples",
                args.control_pattern, n_controls, len(sample_names))
    if n_controls < 6:
        logger.warning(
            "Only %d controls. This can smoke-test the solver, but calibration folds "
            "will be reused and FPR estimates are not meaningful.", n_controls,
        )

    factor_rank_grid = parse_rank_grid(args.factor_rank_grid)
    factor_rank = None if args.factor_rank == "auto" else int(args.factor_rank)
    calibration = calibrate_robust_lower_bound(
        X,
        coverage,
        reference_profiles,
        cell_types,
        target_cell_type=args.target_cell_type,
        control_mask=is_control,
        alpha=args.alpha,
        factor_rank=factor_rank,
        factor_rank_grid=factor_rank_grid,
        min_control_coverage_fraction=args.min_control_coverage_fraction,
        min_sample_coverage=args.min_sample_coverage,
        random_seed=args.random_seed,
    )
    logger.info(
        "Calibrated nuisance: markers=%d rank=%d tau=%.4g alpha=%.3g validation_containment=%.3f",
        int(calibration.marker_mask.sum()),
        calibration.factor_rank,
        calibration.tau,
        calibration.alpha,
        calibration.validation_containment,
    )

    nnls_props = run_weighted_nnls(X, coverage, reference_profiles)

    rows = []
    for i, sample_name in enumerate(sample_names):
        interval = solve_theta_interval(
            X[i],
            coverage[i],
            reference_profiles,
            calibration,
            min_sample_coverage=args.min_sample_coverage,
        )
        cov_i = coverage[i]
        mean_cov = float(cov_i[cov_i > 0].mean()) if (cov_i > 0).any() else 0.0
        row = {
            "sample": sample_name,
            "cohort": args.cohort,
            "is_control": bool(is_control[i]),
            "mean_coverage": mean_cov,
            "n_markers_with_coverage": int((cov_i > 0).sum()),
            "n_markers_used": int(interval["n_markers_used"]),
            "target_cell_type": args.target_cell_type,
            "theta_min": float(interval["theta_min"]),
            "theta_max": float(interval["theta_max"]),
            "theta_width": float(interval["theta_max"] - interval["theta_min"]),
            "theta_min_distance": float(interval["theta_min_distance"]),
            "theta_max_distance": float(interval["theta_max_distance"]),
            "theta_min_success": bool(interval["theta_min_success"]),
            "theta_max_success": bool(interval["theta_max_success"]),
            "best_distance": float(interval["best_distance"]),
            "best_distance_theta": float(interval["best_distance_theta"]),
            "best_distance_success": bool(interval["best_distance_success"]),
            "inside_nuisance": bool(interval["inside_nuisance"]),
            f"{args.target_cell_type}_nnls": float(nnls_props[i, target_idx]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved robust lower-bound predictions to %s", args.output)
    write_calibration_outputs(Path(args.output), calibration)

    if is_control.any():
        controls = df[df["is_control"]]
        non_controls = df[~df["is_control"]]
        logger.info(
            "theta_min controls n=%d mean=%.5g max=%.5g positive=%d",
            len(controls),
            controls["theta_min"].mean(),
            controls["theta_min"].max(),
            int((controls["theta_min"] > 1e-8).sum()),
        )
        logger.info(
            "theta_min non-controls n=%d mean=%.5g max=%.5g positive=%d",
            len(non_controls),
            non_controls["theta_min"].mean(),
            non_controls["theta_min"].max(),
            int((non_controls["theta_min"] > 1e-8).sum()),
        )


if __name__ == "__main__":
    main()
