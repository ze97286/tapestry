"""Per-fragment tumour statistic with an empirically-calibrated null.

Each fragment gets a raw per-read log-likelihood ratio against the panel's
tumour/healthy profiles:

    LLR = Σ_i [ m_i log(p_T,i/p_H,i) + (1-m_i) log((1-p_T,i)/(1-p_H,i)) ]

Standardising LLR by an *analytic* null (assuming p_T/p_H are exact) is biased
when the profiles are estimated from finite, low-coverage data: the bias grows
with the number of CpGs a read covers — hence with read length — re-introducing
exactly the length/batch confound we are trying to kill. So instead we calibrate
**empirically** against a healthy reference set: under the null, LLR has mean and
variance that grow ~linearly with the CpG count k, so we fit ``mean ≈ a·k`` and
``var ≈ b·k`` on held-out healthy reads and report

    z = (LLR - a·k) / sqrt(b·k)

By construction healthy reads centre at z≈0 for **every** k (so n_cpg / read
length carry no label information), and tumour reads score as deviations. The
calibration set must be healthy reads disjoint from those being scored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from rltf.io import load_fragments_regions, panel_intervals
from rltf.profiles import ReferenceProfiles

logger = logging.getLogger(__name__)


@dataclass
class FragmentScores:
    llr: np.ndarray          # raw per-read log-likelihood ratio
    n_cpg: np.ndarray        # usable panel CpGs covered
    read_length: np.ndarray  # QC only
    label: np.ndarray
    sample_id: np.ndarray
    cohort: np.ndarray


def _precompute(profiles: ReferenceProfiles, min_ref_obs: float, prob_clip: float):
    lo, hi = prob_clip, 1.0 - prob_clip
    p_t = np.clip(profiles.p_tumour, lo, hi)
    p_h = np.clip(profiles.p_healthy, lo, hi)
    a = np.log(p_t / p_h)
    b = np.log((1.0 - p_t) / (1.0 - p_h))
    usable = (profiles.total_tumour >= min_ref_obs) & (profiles.total_healthy >= min_ref_obs)
    return a, b, usable


def score_fragments(
    profiles: ReferenceProfiles,
    samples: Sequence[tuple[str, str, str]],
    label: int,
    min_ref_obs: float = 5.0,
    prob_clip: float = 1e-3,
    min_mapq: int = 30,
    flank: int = 1000,
) -> FragmentScores:
    """Raw per-read LLR for samples ``(path, sample_id, cohort)`` (tabix region-read)."""
    a, b, usable = _precompute(profiles, min_ref_obs, prob_clip)
    index = profiles.index
    intervals = panel_intervals(profiles.cpg_chrom, profiles.cpg_pos, flank=flank)

    llr_l, n_l, rl_l, s_l, c_l = [], [], [], [], []
    for path, sample_id, cohort in samples:
        for frag in load_fragments_regions(path, intervals, min_mapq=min_mapq):
            chrom = frag.chrom
            idxs, states = [], []
            for pos, st in zip(frag.cpg_pos.tolist(), frag.states.tolist()):
                i = index.get((chrom, pos))
                if i is not None and usable[i]:
                    idxs.append(i)
                    states.append(st)
            if not idxs:
                continue
            idx = np.asarray(idxs)
            st = np.asarray(states, dtype=np.float64)
            llr_l.append(float(np.sum(st * a[idx] + (1.0 - st) * b[idx])))
            n_l.append(len(idx))
            rl_l.append(frag.read_length)
            s_l.append(sample_id)
            c_l.append(cohort)

    return FragmentScores(
        llr=np.asarray(llr_l, dtype=np.float64),
        n_cpg=np.asarray(n_l, dtype=np.int32),
        read_length=np.asarray(rl_l, dtype=np.int32),
        label=np.full(len(llr_l), label, dtype=np.int8),
        sample_id=np.asarray(s_l, dtype=object),
        cohort=np.asarray(c_l, dtype=object),
    )


@dataclass
class NullCalibration:
    """Empirical healthy null: LLR mean ``a·k`` and variance ``b·k`` (k = n_cpg)."""

    a: float
    b: float

    @classmethod
    def fit(cls, llr: np.ndarray, n_cpg: np.ndarray) -> "NullCalibration":
        k = np.asarray(n_cpg, dtype=np.float64)
        l = np.asarray(llr, dtype=np.float64)
        ok = k > 0
        k, l = k[ok], l[ok]
        if len(k) < 10:
            return cls(a=0.0, b=1.0)
        a = float(np.sum(k * l) / np.sum(k * k))      # mean(LLR) ≈ a·k, through origin
        resid = l - a * k
        b = float(np.sum(resid * resid) / np.sum(k))  # var(LLR) ≈ b·k
        return cls(a=a, b=max(b, 1e-9))

    def z(self, llr: np.ndarray, n_cpg: np.ndarray) -> np.ndarray:
        k = np.maximum(np.asarray(n_cpg, dtype=np.float64), 1.0)
        return (np.asarray(llr, dtype=np.float64) - self.a * k) / np.sqrt(self.b * k)


def fit_calibration(scores: FragmentScores) -> NullCalibration:
    return NullCalibration.fit(scores.llr, scores.n_cpg)


def merge_scores(scores: Iterable[FragmentScores]) -> FragmentScores:
    scores = list(scores)
    return FragmentScores(
        llr=np.concatenate([s.llr for s in scores]),
        n_cpg=np.concatenate([s.n_cpg for s in scores]),
        read_length=np.concatenate([s.read_length for s in scores]),
        label=np.concatenate([s.label for s in scores]),
        sample_id=np.concatenate([s.sample_id for s in scores]),
        cohort=np.concatenate([s.cohort for s in scores]),
    )


def _auc(label: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(label)) < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def oracle_metrics(
    tumour_scores: FragmentScores,
    healthy_scores: FragmentScores,
    calibration: NullCalibration,
    n_cpg_buckets: tuple[int, ...] = (1, 2, 4, 6, 8, 12),
    rng_seed: int = 0,
) -> dict:
    """Verdict metrics on the **calibrated** z, including the length-leakage gates."""
    m = merge_scores([tumour_scores, healthy_scores])
    z = calibration.z(m.llr, m.n_cpg)
    label, ncpg, rlen = m.label.astype(int), m.n_cpg, m.read_length

    auc = _auc(label, z)
    edges = list(n_cpg_buckets) + [np.iinfo(np.int32).max]
    auc_by_n_cpg, null_mean_by_n_cpg = {}, {}
    for lo_b, hi_b in zip(edges[:-1], edges[1:]):
        mask = (ncpg >= lo_b) & (ncpg < hi_b)
        if mask.sum() == 0:
            continue
        name = f"{lo_b}" if hi_b == lo_b + 1 else f"{lo_b}-{'inf' if hi_b == edges[-1] else hi_b - 1}"
        auc_by_n_cpg[name] = _auc(label[mask], z[mask])
        hmask = mask & (label == 0)
        if hmask.sum():
            null_mean_by_n_cpg[name] = float(np.mean(z[hmask]))

    rng = np.random.default_rng(rng_seed)
    auc_perm = _auc(rng.permutation(label), z)
    corr_len = float(np.corrcoef(z, rlen)[0, 1]) if np.std(rlen) > 0 and len(z) > 2 else float("nan")

    separable = bool(np.isfinite(auc) and auc >= 0.60 and (auc - 0.5) > 2 * abs(auc_perm - 0.5))
    return {
        "auc": auc,
        "auc_permuted": auc_perm,
        "auc_by_n_cpg": auc_by_n_cpg,
        "null_mean_z_by_n_cpg": null_mean_by_n_cpg,
        "corr_z_read_length": corr_len,
        "calibration": {"a": calibration.a, "b": calibration.b},
        "n_tumour_frags": int((label == 1).sum()),
        "n_healthy_frags": int((label == 0).sum()),
        "mean_z_tumour": float(np.mean(z[label == 1])) if (label == 1).any() else float("nan"),
        "mean_z_healthy": float(np.mean(z[label == 0])) if (label == 0).any() else float("nan"),
        "separable": separable,
        "verdict": ("SEPARABLE at read level — build the detector"
                    if separable else "NOT separable on this panel — adjust discovery"),
    }
