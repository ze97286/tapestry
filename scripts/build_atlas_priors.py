#!/usr/bin/env python3
"""Build per-(cell type, marker) Beta priors from reference-sample homog outputs.

Inputs
------
* manifest_ben_atlas.tsv — (sample_id, cell_type, file_path) triples.
* ${OUTPUT_DIR}/atlas_homog/<sample_id>.uxm.bed.gz — produced by
  slurm/11a_homog_atlas_references.sh, one per reference sample.
* ${OUTPUT_DIR}/markers/markers.tsv — atlas TSV; defines marker coordinates,
  cell-type column ordering, and meta columns to carry through.

Algorithm
---------
For each (cell_type c, marker m), collect the per-reference U-fraction
observations u_s = U_s / (U_s + M_s) for every reference sample s of type c
with non-zero coverage at marker m. Compute empirical mean and variance of
u_s across those reference samples. Method-of-moments fit Beta(α, β):

    ν = μ(1 − μ) / v  − 1
    α = μ · ν
    β = (1 − μ) · ν

To guard against zero/near-zero variance from small n_c, we floor v at the
cell-type median variance × ``--var-floor-frac`` (default 0.5). This adds
softness where the empirical estimate is implausibly tight; raw v above the
floor is preserved.

Output
------
``data/atlas_priors_ben.tsv`` — one row per atlas marker, columns

    chr, start, end, n_cpgs, [carry-over meta],
    {ct}_mu     — empirical mean U-fraction across reference samples
    {ct}_var    — variance after flooring
    {ct}_n      — number of reference samples with coverage at this marker

The {ct}_mu column should match the existing atlas {ct} column closely
(they are estimates of the same quantity; small discrepancy expected if
the atlas pipeline uses a slightly different aggregation).

Usage
-----
sbatch slurm/11b_build_atlas_priors.sh
or directly:
    python scripts/build_atlas_priors.py \\
        --manifest data/manifest_ben_atlas.tsv \\
        --homog-dir runs/run_v0.3/atlas_homog \\
        --atlas runs/run_v0.3/markers/markers.tsv \\
        --output data/atlas_priors_ben.tsv
"""

import argparse
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


META_COLS = [
    "chr", "start", "end", "startCpG", "endCpG", "n_cpgs",
    "target", "name", "direction",
    "target_signal", "bg_signal", "snr", "target_total", "bg_total",
]


def parse_homog_file(path: Path) -> dict[tuple[str, int], tuple[int, int]]:
    """Return {(chrom, start): (u_count, m_count)} for one homog output file."""
    out: dict[tuple[str, int], tuple[int, int]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 8:
                continue
            key = (parts[0], int(parts[1]))
            u = int(parts[5])
            m_count = int(parts[7])
            out[key] = (u, m_count)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                        help="Reference-sample manifest TSV.")
    parser.add_argument("--homog-dir", required=True,
                        help="Directory of <sample_id>.uxm.bed.gz files.")
    parser.add_argument("--atlas", required=True,
                        help="Atlas TSV (markers.tsv); defines coordinate order "
                             "and cell-type column set.")
    parser.add_argument("--output", required=True,
                        help="Output TSV path.")
    parser.add_argument("--var-floor-frac", type=float, default=0.5,
                        help="Floor per-marker variance at this fraction of the "
                             "cell-type-wide median variance. Default 0.5.")
    parser.add_argument("--abs-var-floor", type=float, default=1e-5,
                        help="Hard absolute lower bound for variance.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    manifest = pd.read_csv(args.manifest, sep="\t")
    logger.info("Manifest: %d reference samples across %d cell types",
                len(manifest), manifest["cell_type"].nunique())
    for ct, n in manifest["cell_type"].value_counts().sort_index().items():
        logger.info("  %s: %d samples", ct, n)

    atlas_df = pd.read_csv(args.atlas, sep="\t")
    atlas_ct_cols = [c for c in atlas_df.columns if c not in META_COLS]
    cell_types = sorted(atlas_ct_cols)
    M = len(atlas_df)
    logger.info("Atlas: %d markers, %d cell types: %s", M, len(cell_types), cell_types)

    coords = [(str(atlas_df.iloc[i]["chr"]), int(atlas_df.iloc[i]["start"]))
              for i in range(M)]

    homog_dir = Path(args.homog_dir)
    cell_type_samples: dict[str, list[str]] = {}
    for _, row in manifest.iterrows():
        cell_type_samples.setdefault(row["cell_type"], []).append(row["sample_id"])

    # u_obs[c][m] -> list of u_fractions across reference samples of cell type c
    u_obs: dict[str, list[list[float]]] = {
        ct: [[] for _ in range(M)] for ct in cell_types
    }
    missing_files: list[str] = []
    for ct, sids in cell_type_samples.items():
        if ct not in u_obs:
            logger.warning("Manifest cell type %r not in atlas — skipping", ct)
            continue
        for sid in sids:
            f = homog_dir / f"{sid}.uxm.bed.gz"
            if not f.exists():
                missing_files.append(str(f))
                continue
            data = parse_homog_file(f)
            for i, key in enumerate(coords):
                if key in data:
                    u, m_count = data[key]
                    total = u + m_count
                    if total > 0:
                        u_obs[ct][i].append(u / total)
    if missing_files:
        logger.warning("%d homog files missing — first 5: %s",
                       len(missing_files), missing_files[:5])

    # Aggregate per (cell type, marker)
    mu = {ct: np.full(M, np.nan, dtype=np.float64) for ct in cell_types}
    var = {ct: np.full(M, np.nan, dtype=np.float64) for ct in cell_types}
    n = {ct: np.zeros(M, dtype=np.int32) for ct in cell_types}
    for ct in cell_types:
        for i in range(M):
            obs = u_obs[ct][i]
            n[ct][i] = len(obs)
            if len(obs) == 0:
                continue
            arr = np.asarray(obs, dtype=np.float64)
            mu[ct][i] = arr.mean()
            if len(arr) >= 2:
                var[ct][i] = arr.var(ddof=1)
            else:
                var[ct][i] = np.nan  # filled in by floor below

    # Variance shrinkage: floor at a fraction of cell-type-wide median variance
    for ct in cell_types:
        v = var[ct]
        valid_v = v[np.isfinite(v) & (v > 0)]
        if len(valid_v) == 0:
            ct_median = args.abs_var_floor
        else:
            ct_median = float(np.median(valid_v))
        floor = max(ct_median * args.var_floor_frac, args.abs_var_floor)
        # Markers with no observations or n=1 → use floor.
        bad = ~np.isfinite(v) | (v < floor)
        n_floored = int(bad.sum())
        v[bad] = floor
        var[ct] = v
        logger.info("  %s: median var=%.5f, floor=%.5f, %d/%d markers floored",
                    ct, ct_median, floor, n_floored, M)

    # Sanity: empirical mean vs atlas value should agree closely.
    logger.info("Sanity: empirical mu vs atlas column (per cell type):")
    for ct in cell_types:
        if ct not in atlas_df.columns:
            continue
        atlas_vals = atlas_df[ct].values.astype(np.float64)
        empirical = mu[ct]
        both = np.isfinite(atlas_vals) & np.isfinite(empirical)
        if not both.any():
            continue
        diff = np.abs(atlas_vals[both] - empirical[both])
        logger.info("  %s: median |empirical - atlas| = %.4f  max %.4f  (over %d markers)",
                    ct, float(np.median(diff)), float(diff.max()), int(both.sum()))

    # Compose output table
    out = atlas_df[[c for c in atlas_df.columns if c in META_COLS]].copy()
    for ct in cell_types:
        out[f"{ct}_mu"] = mu[ct]
        out[f"{ct}_var"] = var[ct]
        out[f"{ct}_n"] = n[ct]

    n_no_obs = sum(int((n[ct] == 0).sum()) for ct in cell_types)
    logger.info("Cell-type/marker pairs with zero observations: %d", n_no_obs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote %s (%d rows × %d columns)",
                out_path, len(out), out.shape[1])


if __name__ == "__main__":
    main()
