"""Per-read likelihood-ratio scoring and the separability oracle.

For a block with reference per-CpG tumour/healthy methylation ``p_T``/``p_H``, a
read covering CpGs ``S`` with states ``m_i`` scores

    LLR = sum_{i in S} [ m_i log(p_T,i/p_H,i) + (1-m_i) log((1-p_T,i)/(1-p_H,i)) ]

an explicit log-likelihood ratio of tumour vs healthy origin. It depends only on
the read's methylation states and the reference profiles — never on read length,
cohort or sample id — so the oracle AUC cannot be a batch/length shortcut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from rltf.io import load_reads
from rltf.profiles import ReferenceProfiles

logger = logging.getLogger(__name__)


@dataclass
class ReadScores:
    llr: np.ndarray
    n_cpg_scored: np.ndarray
    weight: np.ndarray
    label: np.ndarray
    sample_id: np.ndarray
    cohort: np.ndarray
    block_id: np.ndarray


def score_reads(
    profiles: ReferenceProfiles,
    samples: Sequence[tuple[str, str, str, str]],
    label: int,
    min_ref_obs: float = 5.0,
    prob_clip: float = 1e-3,
    min_cpgs_overlap: int = 1,
) -> ReadScores:
    """Score reads from samples ``(pat_path, convention, sample_id, cohort)``.

    A CpG contributes to a read's LLR only if it has >= *min_ref_obs* observed
    reads in **both** reference groups (so the LLR is never driven by an
    unestimated site). ``label`` (1=tumour, 0=healthy) tags every read.
    """
    lo, hi = prob_clip, 1.0 - prob_clip
    p_t_all = np.clip(profiles.p_tumour, lo, hi)
    p_h_all = np.clip(profiles.p_healthy, lo, hi)
    usable_all = (profiles.total_tumour >= min_ref_obs) & (profiles.total_healthy >= min_ref_obs)
    log_meth_all = np.log(p_t_all / p_h_all)
    log_unmeth_all = np.log((1.0 - p_t_all) / (1.0 - p_h_all))

    llr_l, ncpg_l, w_l, samp_l, coh_l, blk_l = [], [], [], [], [], []

    for pat_path, convention, sample_id, cohort in samples:
        pat_path = Path(pat_path)
        if not pat_path.exists():
            logger.warning("PAT missing, skipping: %s", pat_path)
            continue
        for read in load_reads(pat_path, convention, min_cpgs=min_cpgs_overlap):
            for bi, row in profiles.block_index.assign(read):
                observed = (row != -1) & usable_all[bi]
                if not observed.any():
                    continue
                states = row[observed]
                contrib = np.where(states == 1, log_meth_all[bi][observed], log_unmeth_all[bi][observed])
                llr_l.append(float(contrib.sum()))
                ncpg_l.append(int(observed.sum()))
                w_l.append(int(read.count))
                samp_l.append(sample_id)
                coh_l.append(cohort)
                blk_l.append(profiles.blocks[bi].block_id)

    return ReadScores(
        llr=np.asarray(llr_l, dtype=np.float64),
        n_cpg_scored=np.asarray(ncpg_l, dtype=np.int32),
        weight=np.asarray(w_l, dtype=np.int64),
        label=np.full(len(llr_l), label, dtype=np.int8),
        sample_id=np.asarray(samp_l, dtype=object),
        cohort=np.asarray(coh_l, dtype=object),
        block_id=np.asarray(blk_l, dtype=object),
    )


def merge_scores(scores: Iterable[ReadScores]) -> ReadScores:
    scores = list(scores)
    return ReadScores(
        llr=np.concatenate([s.llr for s in scores]),
        n_cpg_scored=np.concatenate([s.n_cpg_scored for s in scores]),
        weight=np.concatenate([s.weight for s in scores]),
        label=np.concatenate([s.label for s in scores]),
        sample_id=np.concatenate([s.sample_id for s in scores]),
        cohort=np.concatenate([s.cohort for s in scores]),
        block_id=np.concatenate([s.block_id for s in scores]),
    )


def _weighted_auc(label: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(label)) < 2:
        return float("nan")
    return float(roc_auc_score(label, score, sample_weight=weight))


def oracle_metrics(
    tumour_scores: ReadScores,
    healthy_scores: ReadScores,
    n_cpg_buckets: tuple[int, ...] = (1, 2, 4, 6, 8, 12),
    rng_seed: int = 0,
) -> dict:
    """Verdict metrics: AUC, AUC-by-CpG-count, permutation null, CpG-matched AUC."""
    merged = merge_scores([tumour_scores, healthy_scores])
    finite = np.isfinite(merged.llr)
    llr = merged.llr[finite]
    label = merged.label[finite].astype(int)
    weight = merged.weight[finite].astype(float)
    ncpg = merged.n_cpg_scored[finite]

    auc = _weighted_auc(label, llr, weight)
    edges = list(n_cpg_buckets) + [np.iinfo(np.int32).max]
    auc_by_n_cpg: dict[str, dict] = {}
    for lo_b, hi_b in zip(edges[:-1], edges[1:]):
        mask = (ncpg >= lo_b) & (ncpg < hi_b)
        if mask.sum() == 0:
            continue
        name = f"{lo_b}" if hi_b == lo_b + 1 else f"{lo_b}-{'inf' if hi_b == edges[-1] else hi_b - 1}"
        auc_by_n_cpg[name] = {
            "auc": _weighted_auc(label[mask], llr[mask], weight[mask]),
            "n_tumour": int(weight[mask & (label == 1)].sum()),
            "n_healthy": int(weight[mask & (label == 0)].sum()),
        }

    rng = np.random.default_rng(rng_seed)
    auc_permuted = _weighted_auc(rng.permutation(label), llr, weight)

    match_weight = weight.copy()
    for lo_b, hi_b in zip(edges[:-1], edges[1:]):
        mask = (ncpg >= lo_b) & (ncpg < hi_b)
        w_t = weight[mask & (label == 1)].sum()
        w_h = weight[mask & (label == 0)].sum()
        if w_t == 0 or w_h == 0:
            match_weight[mask] = 0.0
            continue
        target = min(w_t, w_h)
        match_weight[mask & (label == 1)] *= target / w_t
        match_weight[mask & (label == 0)] *= target / w_h
    keep = match_weight > 0
    auc_matched = _weighted_auc(label[keep], llr[keep], match_weight[keep]) if keep.any() else float("nan")

    separable = bool(np.isfinite(auc) and auc >= 0.60 and (auc - 0.5) > 2 * abs(auc_permuted - 0.5))
    return {
        "auc": auc,
        "auc_permuted": auc_permuted,
        "auc_ncpg_matched": auc_matched,
        "auc_by_n_cpg": auc_by_n_cpg,
        "n_tumour_reads": int(weight[label == 1].sum()),
        "n_healthy_reads": int(weight[label == 0].sum()),
        "mean_llr_tumour": float(np.average(llr[label == 1], weights=weight[label == 1])) if (label == 1).any() else float("nan"),
        "mean_llr_healthy": float(np.average(llr[label == 0], weights=weight[label == 0])) if (label == 0).any() else float("nan"),
        "separable": separable,
        "verdict": (
            "SEPARABLE at read level — build the detector on this panel"
            if separable else
            "NOT separable at read level on this panel — adjust discovery (window/effect/coverage)"
        ),
    }
