"""Per-fragment, n_cpg-conditioned tumour statistic, and the separability oracle.

For a fragment covering panel CpGs ``S`` with states ``m_i``, the raw
log-likelihood ratio is ``LLR = Σ_i [m_i a_i + (1-m_i) b_i]`` with
``a_i = log(p_T,i/p_H,i)``, ``b_i = log((1-p_T,i)/(1-p_H,i))``. Its mean and
variance **under the healthy null** both scale with the number of covered CpGs,
so raw LLR is confounded with n_cpg — which is correlated with read length and
hence batch. We therefore standardise each fragment against its own null:

    μ0 = Σ_i [p_H,i a_i + (1-p_H,i) b_i]          (expected LLR if healthy)
    σ0² = Σ_i (a_i - b_i)² p_H,i (1-p_H,i)        (variance if healthy)
    z  = (LLR - μ0) / σ0

Under the healthy null E[z]=0, Var[z]=1 **for any CpG set** — so n_cpg (and thus
read length) carries no information about the label. A tumour fragment's pattern
deviates toward p_T → z > 0, with more CpGs giving more power but no inflated
expectation. ``read_length`` is retained only to *verify* orthogonality.
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
    z: np.ndarray
    n_cpg: np.ndarray
    read_length: np.ndarray
    label: np.ndarray
    sample_id: np.ndarray
    cohort: np.ndarray


def _precompute(profiles: ReferenceProfiles, min_ref_obs: float, prob_clip: float):
    lo, hi = prob_clip, 1.0 - prob_clip
    p_t = np.clip(profiles.p_tumour, lo, hi)
    p_h = np.clip(profiles.p_healthy, lo, hi)
    a = np.log(p_t / p_h)
    b = np.log((1.0 - p_t) / (1.0 - p_h))
    mu_i = p_h * a + (1.0 - p_h) * b
    var_i = (a - b) ** 2 * p_h * (1.0 - p_h)
    usable = (profiles.total_tumour >= min_ref_obs) & (profiles.total_healthy >= min_ref_obs)
    return a, b, mu_i, var_i, usable


def score_fragments(
    profiles: ReferenceProfiles,
    samples: Sequence[tuple[str, str, str]],
    label: int,
    min_ref_obs: float = 5.0,
    prob_clip: float = 1e-3,
    min_mapq: int = 30,
    flank: int = 1000,
) -> FragmentScores:
    """Score fragments from samples ``(path, sample_id, cohort)`` by the n_cpg-conditioned z.

    Reads only the panel's intervals (tabix region-query) — the difference
    between minutes and hours on multi-GB per-read files.
    """
    a, b, mu_i, var_i, usable = _precompute(profiles, min_ref_obs, prob_clip)
    index = profiles.index
    intervals = panel_intervals(profiles.cpg_chrom, profiles.cpg_pos, flank=flank)

    z_l, n_l, rl_l, s_l, c_l = [], [], [], [], []
    for path, sample_id, cohort in samples:
        for frag in load_fragments_regions(path, intervals, min_mapq=min_mapq):
            chrom = frag.chrom
            idxs = []
            states = []
            for pos, st in zip(frag.cpg_pos.tolist(), frag.states.tolist()):
                i = index.get((chrom, pos))
                if i is not None and usable[i]:
                    idxs.append(i)
                    states.append(st)
            if not idxs:
                continue
            idx = np.asarray(idxs)
            st = np.asarray(states, dtype=np.float64)
            llr = float(np.sum(st * a[idx] + (1.0 - st) * b[idx]))
            mu = float(mu_i[idx].sum())
            var = float(var_i[idx].sum())
            if var <= 0:
                continue
            z_l.append((llr - mu) / np.sqrt(var))
            n_l.append(len(idx))
            rl_l.append(frag.read_length)
            s_l.append(sample_id)
            c_l.append(cohort)

    return FragmentScores(
        z=np.asarray(z_l, dtype=np.float64),
        n_cpg=np.asarray(n_l, dtype=np.int32),
        read_length=np.asarray(rl_l, dtype=np.int32),
        label=np.full(len(z_l), label, dtype=np.int8),
        sample_id=np.asarray(s_l, dtype=object),
        cohort=np.asarray(c_l, dtype=object),
    )


def merge_scores(scores: Iterable[FragmentScores]) -> FragmentScores:
    scores = list(scores)
    return FragmentScores(
        z=np.concatenate([s.z for s in scores]),
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
    n_cpg_buckets: tuple[int, ...] = (1, 2, 4, 6, 8, 12),
    rng_seed: int = 0,
) -> dict:
    """Verdict metrics, including the length-leakage gates.

    ``null_mean_z_by_n_cpg`` should be ~0 in every bucket (the statistic is
    length-invariant under the null) and ``corr_z_read_length`` ~0; these are the
    checks PAT could not even run.
    """
    m = merge_scores([tumour_scores, healthy_scores])
    z, label, ncpg, rlen = m.z, m.label.astype(int), m.n_cpg, m.read_length

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
        "n_tumour_frags": int((label == 1).sum()),
        "n_healthy_frags": int((label == 0).sum()),
        "mean_z_tumour": float(np.mean(z[label == 1])) if (label == 1).any() else float("nan"),
        "mean_z_healthy": float(np.mean(z[label == 0])) if (label == 0).any() else float("nan"),
        "separable": separable,
        "verdict": ("SEPARABLE at read level — build the detector"
                    if separable else
                    "NOT separable on this panel — adjust discovery"),
    }
