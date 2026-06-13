"""Per-sample features for the tabular detection head.

Aggregates the per-fragment **calibrated** z over a sample. The calibration
(:class:`rltf.llr.NullCalibration`) is fit on a healthy reference set disjoint
from the samples being scored, so healthy reads centre at z≈0 for every n_cpg and
read length carries no label information. All features are coverage-normalised
(fractions / log-counts); ``mean_n_cpg``/``corr_z_read_length`` are QC only.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np

from rltf.llr import NullCalibration, score_fragments
from rltf.profiles import ReferenceProfiles

logger = logging.getLogger(__name__)

DEFAULT_Z_THRESHOLDS: tuple[float, ...] = (1.0, 2.0, 3.0)


def feature_names(z_thresholds: Sequence[float] = DEFAULT_Z_THRESHOLDS) -> list[str]:
    names = ["log1p_n_frags", "mean_z", "median_z", "z_p90", "z_p99"]
    names += [f"frac_z_gt_{t:g}" for t in z_thresholds]
    return names


def _wq(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def compute_sample_features(
    profiles: ReferenceProfiles,
    pat_path: str,
    sample_id: str,
    cohort: str,
    calibration: NullCalibration,
    min_ref_obs: float = 5.0,
    min_mapq: int = 30,
    flank: int = 1000,
    z_thresholds: Sequence[float] = DEFAULT_Z_THRESHOLDS,
) -> dict:
    s = score_fragments(profiles, [(pat_path, sample_id, cohort)], label=0,
                        min_ref_obs=min_ref_obs, min_mapq=min_mapq, flank=flank)
    z = calibration.z(s.llr, s.n_cpg)
    feats: dict = {"sample_id": sample_id, "cohort": cohort}
    n = len(z)
    feats["log1p_n_frags"] = float(np.log1p(n))
    if n == 0:
        for name in feature_names(z_thresholds):
            feats.setdefault(name, 0.0)
        feats["mean_n_cpg"] = 0.0
        feats["corr_z_read_length"] = float("nan")
        return feats
    feats["mean_z"] = float(np.mean(z))
    feats["median_z"] = _wq(z, 0.5)
    feats["z_p90"] = _wq(z, 0.90)
    feats["z_p99"] = _wq(z, 0.99)
    for t in z_thresholds:
        feats[f"frac_z_gt_{t:g}"] = float(np.mean(z > t))
    feats["mean_n_cpg"] = float(np.mean(s.n_cpg))          # QC only
    feats["corr_z_read_length"] = (float(np.corrcoef(z, s.read_length)[0, 1])
                                   if np.std(s.read_length) > 0 and n > 2 else float("nan"))
    return feats


def build_feature_matrix(
    profiles: ReferenceProfiles,
    samples: Iterable[tuple[str, str, str]],
    calibration: NullCalibration,
    min_ref_obs: float = 5.0,
    min_mapq: int = 30,
    flank: int = 1000,
    z_thresholds: Sequence[float] = DEFAULT_Z_THRESHOLDS,
):
    """Build a ``(samples x features)`` DataFrame from ``(path, sample_id, cohort)``."""
    import pandas as pd

    rows = []
    for path, sample_id, cohort in samples:
        feats = compute_sample_features(profiles, path, sample_id, cohort, calibration,
                                        min_ref_obs=min_ref_obs, min_mapq=min_mapq,
                                        flank=flank, z_thresholds=z_thresholds)
        rows.append(feats)
        logger.info("Features %s (%s): n_frags=%.0f mean_z=%.3f",
                    sample_id, cohort, np.expm1(feats["log1p_n_frags"]), feats.get("mean_z", float("nan")))
    df = pd.DataFrame(rows).set_index("sample_id")
    return df[["cohort"] + feature_names(z_thresholds) + ["mean_n_cpg", "corr_z_read_length"]]
