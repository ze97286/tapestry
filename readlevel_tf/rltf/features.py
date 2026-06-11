"""Per-sample read-level features for the tabular head.

Each cfDNA sample's reads are scored by per-read LLR against the project's own
profiles and aggregated into a compact, coverage-normalised, reference-defined
feature vector. Counts become fractions so high- and low-coverage cohorts are on
one scale; ``n_reads_scored`` is kept so the head can learn coverage-dependent
calibration; upper LLR quantiles capture the rare tumour-molecule tail.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np

from rltf.llr import score_reads
from rltf.profiles import ReferenceProfiles

logger = logging.getLogger(__name__)

DEFAULT_LLR_THRESHOLDS: tuple[float, ...] = (0.0, 2.0, 5.0)


def feature_names(llr_thresholds: Sequence[float] = DEFAULT_LLR_THRESHOLDS) -> list[str]:
    names = [
        "n_reads_scored", "log1p_n_reads_scored", "mean_llr", "median_llr",
        "llr_p90", "llr_p99", "mean_n_cpg_per_read",
        "n_blocks_covered", "frac_blocks_with_tumour_read",
    ]
    names += [f"frac_reads_llr_gt_{t:g}" for t in llr_thresholds]
    return names


def _wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order].astype(np.float64)
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, v))


def compute_sample_features(
    profiles: ReferenceProfiles,
    pat_path: str,
    convention: str,
    sample_id: str,
    cohort: str,
    min_ref_obs: float = 5.0,
    prob_clip: float = 1e-3,
    llr_thresholds: Sequence[float] = DEFAULT_LLR_THRESHOLDS,
) -> dict:
    scores = score_reads(
        profiles, [(pat_path, convention, sample_id, cohort)], label=0,
        min_ref_obs=min_ref_obs, prob_clip=prob_clip,
    )
    finite = np.isfinite(scores.llr)
    llr = scores.llr[finite]
    w = scores.weight[finite].astype(np.float64)
    ncpg = scores.n_cpg_scored[finite].astype(np.float64)
    blk = scores.block_id[finite]

    feats: dict = {"sample_id": sample_id, "cohort": cohort}
    n_reads = float(w.sum())
    feats["n_reads_scored"] = n_reads
    feats["log1p_n_reads_scored"] = float(np.log1p(n_reads))
    if n_reads == 0:
        for name in feature_names(llr_thresholds):
            feats.setdefault(name, 0.0)
        return feats

    feats["mean_llr"] = float(np.average(llr, weights=w))
    feats["median_llr"] = _wquantile(llr, w, 0.5)
    feats["llr_p90"] = _wquantile(llr, w, 0.90)
    feats["llr_p99"] = _wquantile(llr, w, 0.99)
    feats["mean_n_cpg_per_read"] = float(np.average(ncpg, weights=w))

    base_t = llr_thresholds[0]
    covered: dict[str, bool] = {}
    for b, pos in zip(blk, llr > base_t):
        covered[b] = covered.get(b, False) or bool(pos)
    feats["n_blocks_covered"] = float(len(covered))
    feats["frac_blocks_with_tumour_read"] = float(sum(covered.values()) / len(covered)) if covered else 0.0
    for t in llr_thresholds:
        feats[f"frac_reads_llr_gt_{t:g}"] = float(w[llr > t].sum() / n_reads)
    return feats


def build_feature_matrix(
    profiles: ReferenceProfiles,
    samples: Iterable[tuple[str, str, str, str]],
    min_ref_obs: float = 5.0,
    prob_clip: float = 1e-3,
    llr_thresholds: Sequence[float] = DEFAULT_LLR_THRESHOLDS,
):
    """Build a ``(samples x features)`` DataFrame from ``(path, convention, sample_id, cohort)``."""
    import pandas as pd

    rows = []
    for pat_path, convention, sample_id, cohort in samples:
        feats = compute_sample_features(
            profiles, pat_path, convention, sample_id, cohort,
            min_ref_obs=min_ref_obs, prob_clip=prob_clip, llr_thresholds=llr_thresholds,
        )
        rows.append(feats)
        logger.info("Features %s (%s): n_reads=%.0f mean_llr=%.3f",
                    sample_id, cohort, feats.get("n_reads_scored", 0.0), feats.get("mean_llr", float("nan")))
    df = pd.DataFrame(rows).set_index("sample_id")
    return df[["cohort"] + feature_names(llr_thresholds)]
