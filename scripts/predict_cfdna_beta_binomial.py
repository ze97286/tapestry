#!/usr/bin/env python3
"""Beta-Binomial deconvolution on a cfDNA cohort with empirical Beta priors.

Pipeline
--------
1. Load atlas means (markers.tsv) and per-(cell type, marker) Beta priors
   (atlas_priors_ben.tsv, produced by scripts/build_atlas_priors.py).
2. Run ``wgbstools homog`` on every cfDNA PAT in --cfdna-dir; extract
   per-marker U-fraction and coverage aligned to the atlas.
3. Solve Beta-Binomial MLE per sample on the simplex
   (tapestry/benchmark/beta_binomial.py).
4. Run plain coverage-weighted NNLS as a comparison column.

Output CSV columns
------------------
* ``{ct}``        — Beta-Binomial proportion (production for this script).
* ``{ct}_nnls``   — coverage-weighted NNLS for comparison.
* ``mean_coverage``, ``n_markers_with_coverage``, ``is_control``.

Compatible with clinical_compare_variants.py / clinical_evaluation.py.

Usage
-----
sbatch --export=ALL,COHORT=AB slurm/11c_predict_cfdna_beta_binomial.sh
"""

import argparse
import gzip
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.benchmark.nnls import run_weighted_nnls
from tapestry.benchmark.beta_binomial import run_beta_binomial

logger = logging.getLogger(__name__)


META_COLS = [
    "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
    "target", "name", "direction",
    "target_signal", "bg_signal", "snr", "target_total", "bg_total",
]


def load_atlas_and_priors(atlas_path: str, priors_path: str):
    """Load atlas + Beta priors, drop NaN-in-atlas rows, align coordinates.

    Returns
    -------
    cell_types : sorted list of cell-type names.
    mu_profiles : (C, M) per (cell type, marker) Beta mean, atlas-derived where
        priors data are absent.
    var_profiles : (C, M) per (cell type, marker) Beta variance.
    n_profiles : (C, M) reference-sample count per (cell type, marker).
    atlas_coords : list of (chr, start) for the M valid rows.
    valid_indices : indices into the original atlas rows.
    total_rows : pre-filter row count.
    """
    atlas_df = pd.read_csv(atlas_path, sep="\t")
    atlas_ct_cols = [c for c in atlas_df.columns if c not in META_COLS]
    cell_types = sorted(atlas_ct_cols)
    C = len(cell_types)

    any_nan = atlas_df[atlas_ct_cols].isna().any(axis=1).values
    valid_indices = np.where(~any_nan)[0]
    atlas_df_v = atlas_df.iloc[valid_indices].reset_index(drop=True)
    M = len(atlas_df_v)
    total_rows = len(atlas_df)

    # Atlas means (used as fallback when priors are missing)
    atlas_mu = np.zeros((C, M), dtype=np.float64)
    for j, ct in enumerate(cell_types):
        atlas_mu[j] = atlas_df_v[ct].values.astype(np.float64)

    # Priors (mu, var, n) per (ct, marker), aligned to atlas coords
    priors_df = pd.read_csv(priors_path, sep="\t")
    coords_atlas = list(zip(atlas_df_v["chr"].astype(str),
                            atlas_df_v["start"].astype(int)))
    coords_priors = list(zip(priors_df["chr"].astype(str),
                             priors_df["start"].astype(int)))
    priors_idx = {key: i for i, key in enumerate(coords_priors)}

    mu_profiles = atlas_mu.copy()
    # Default variance for missing priors: cell-type median var from priors data
    default_var = np.full(C, 0.01, dtype=np.float64)
    var_profiles = np.tile(default_var[:, None], (1, M))
    n_profiles = np.zeros((C, M), dtype=np.int32)

    n_missing = 0
    for m, key in enumerate(coords_atlas):
        if key not in priors_idx:
            n_missing += 1
            continue
        pi = priors_idx[key]
        for j, ct in enumerate(cell_types):
            mu_col = f"{ct}_mu"
            var_col = f"{ct}_var"
            n_col = f"{ct}_n"
            if mu_col in priors_df.columns:
                mu_v = priors_df.iloc[pi][mu_col]
                if pd.notna(mu_v):
                    mu_profiles[j, m] = float(mu_v)
            if var_col in priors_df.columns:
                v_v = priors_df.iloc[pi][var_col]
                if pd.notna(v_v) and v_v > 0:
                    var_profiles[j, m] = float(v_v)
            if n_col in priors_df.columns:
                n_v = priors_df.iloc[pi][n_col]
                if pd.notna(n_v):
                    n_profiles[j, m] = int(n_v)
    if n_missing:
        logger.warning("%d / %d atlas markers absent from priors file — "
                       "using atlas mean and default variance there", n_missing, M)

    # Cell-type-wide median var (from priors only) for downgraded fallback
    for j, ct in enumerate(cell_types):
        col = f"{ct}_var"
        if col in priors_df.columns:
            valid = priors_df[col].dropna()
            if len(valid) > 0:
                ct_median = float(valid.median())
                # Markers where var was unset (still at default 0.01) AND priors missing
                # get the cell-type median if it's looser.
                pass  # keep default_var conservative; do not lower

    atlas_coords = coords_atlas
    return (cell_types, mu_profiles, var_profiles, n_profiles,
            atlas_coords, valid_indices, total_rows)


def extract_marker_values(homog_path: str, atlas_coords: list[tuple[str, int]]):
    try:
        opener = gzip.open if homog_path.endswith(".gz") else open
        homog_data = {}
        with opener(homog_path, "rt") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) < 8:
                    continue
                key = (parts[0], int(parts[1]))
                homog_data[key] = (int(parts[5]), int(parts[7]))
        M = len(atlas_coords)
        u_fraction = np.zeros(M, dtype=np.float64)
        coverage = np.zeros(M, dtype=np.float64)
        for i, key in enumerate(atlas_coords):
            if key in homog_data:
                u, m_count = homog_data[key]
                total = u + m_count
                coverage[i] = total
                if total > 0:
                    u_fraction[i] = u / total
        return u_fraction, coverage
    except Exception as e:
        logger.error("Failed to parse %s: %s", homog_path, e)
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
    if not os.path.exists(expected):
        logger.error("homog produced no output for %s (expected %s)",
                     sample_name, expected)
        return None
    return extract_marker_values(expected, atlas_coords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfdna-dir", required=True)
    parser.add_argument("--markers-bed", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--priors", required=True,
                        help="Per-(cell type, marker) Beta priors TSV "
                             "from build_atlas_priors.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wgbstools", default="wgbstools")
    parser.add_argument("--cohort", default="")
    parser.add_argument("--control-pattern", default=r"_Ctrl_|^Ctrl_|_healthy_",
                        help="Regex matching healthy-control sample names.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    (cell_types, mu_profiles, var_profiles, n_profiles,
     atlas_coords, valid_indices, total_rows) = load_atlas_and_priors(
        args.atlas, args.priors,
    )
    logger.info("Atlas: kept %d / %d rows; %d cell types %s",
                len(valid_indices), total_rows, len(cell_types), cell_types)
    logger.info("Priors n-ref summary per cell type:")
    for j, ct in enumerate(cell_types):
        nz = (n_profiles[j] > 0).sum()
        med_n = int(np.median(n_profiles[j][n_profiles[j] > 0])) if nz > 0 else 0
        med_var = float(np.median(var_profiles[j]))
        logger.info("  %s: %d markers with prior data, median n_ref=%d, "
                    "median var=%.4f", ct, nz, med_n, med_var)

    cfdna_dir = Path(args.cfdna_dir)
    pat_files = sorted(cfdna_dir.glob("*.markers.pat.gz"))
    if not pat_files:
        pat_files = sorted(cfdna_dir.glob("*.pat.gz"))
    logger.info("Found %d cfDNA PAT files in %s", len(pat_files), cfdna_dir)

    sample_names: list[str] = []
    X_all: list[np.ndarray] = []
    cov_all: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="tapestry_betabin_") as tmp_dir:
        for i, pat_path in enumerate(pat_files):
            sample_name = pat_path.stem.replace(".markers.pat", "").replace(".pat", "")
            logger.info("[%d/%d] homog %s", i + 1, len(pat_files), sample_name)
            result = process_cfdna_sample(
                str(pat_path), args.markers_bed, args.wgbstools, atlas_coords, tmp_dir,
            )
            if result is None:
                continue
            u_fraction, coverage = result
            sample_names.append(sample_name)
            X_all.append(u_fraction)
            cov_all.append(coverage)
            for f in Path(tmp_dir).glob("*.uxm.bed.gz"):
                f.unlink()

    X = np.asarray(X_all, dtype=np.float32)
    coverage = np.asarray(cov_all, dtype=np.float32)
    logger.info("Extracted marker values for %d samples", len(sample_names))

    ctrl_re = re.compile(args.control_pattern)
    is_control = np.array([bool(ctrl_re.search(s)) for s in sample_names])
    logger.info("Healthy controls matched by %r: %d / %d",
                args.control_pattern, int(is_control.sum()), len(sample_names))

    logger.info("Solving Beta-Binomial MLE on %d samples...", len(sample_names))
    bb_props = run_beta_binomial(X, coverage, mu_profiles, var_profiles)

    logger.info("Solving plain coverage-weighted NNLS for comparison...")
    nnls_props = run_weighted_nnls(X, coverage, mu_profiles)

    results = []
    for i, name in enumerate(sample_names):
        cov_i = coverage[i]
        mean_cov = float(cov_i[cov_i > 0].mean()) if (cov_i > 0).any() else 0.0
        n_with_cov = int((cov_i > 0).sum())
        row = {
            "sample": name,
            "cohort": args.cohort,
            "mean_coverage": mean_cov,
            "n_markers_with_coverage": n_with_cov,
            "is_control": bool(is_control[i]),
        }
        for j, ct in enumerate(cell_types):
            row[ct] = float(bb_props[i, j])
            row[f"{ct}_nnls"] = float(nnls_props[i, j])
        results.append(row)

    df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved predictions for %d samples to %s", len(df), out_path)

    if "OAC" in cell_types and is_control.any():
        logger.info("\nBeta-Binomial OAC stats:")
        oac_ctrl = df.loc[df["is_control"], "OAC"].values
        oac_cancer = df.loc[~df["is_control"], "OAC"].values
        logger.info("  controls (n=%d):     mean=%.4f  max=%.4f  95pct=%.4f",
                    len(oac_ctrl), oac_ctrl.mean(), oac_ctrl.max(),
                    np.percentile(oac_ctrl, 95))
        logger.info("  non-controls (n=%d): mean=%.4f  max=%.4f  95pct=%.4f",
                    len(oac_cancer), oac_cancer.mean(), oac_cancer.max(),
                    np.percentile(oac_cancer, 95))
        logger.info("\nNNLS comparison OAC stats:")
        oac_ctrl = df.loc[df["is_control"], "OAC_nnls"].values
        oac_cancer = df.loc[~df["is_control"], "OAC_nnls"].values
        logger.info("  controls (n=%d):     mean=%.4f  max=%.4f  95pct=%.4f",
                    len(oac_ctrl), oac_ctrl.mean(), oac_ctrl.max(),
                    np.percentile(oac_ctrl, 95))
        logger.info("  non-controls (n=%d): mean=%.4f  max=%.4f  95pct=%.4f",
                    len(oac_cancer), oac_cancer.mean(), oac_cancer.max(),
                    np.percentile(oac_cancer, 95))


if __name__ == "__main__":
    main()
