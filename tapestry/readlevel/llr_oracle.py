"""Read-level likelihood-ratio oracle for tumour-vs-healthy separability.

The MethylBERT experiment left one decisive question unanswered
(``docs/methylbert_experiment_summary.md`` — Recommended Next Diagnostics #1):
once read-length and AB/CD batch shortcuts are removed, is there *any*
single-molecule tumour-vs-control methylation signal in the selected regions?
MethylBERT could not answer this because its classifier could (and did) read
fragment length and cohort identity instead of methylation.

This module answers it directly, with an explicit per-read likelihood ratio
that has no access to read length, cohort or sample identity — only the
methylation states of the CpGs a read covers, scored against reference
methylation profiles.

Statistic
---------
For a region with reference per-CpG methylation probabilities ``p_T`` (tumour)
and ``p_H`` (healthy), a read covering CpGs ``S`` with binary methylation
states ``m_i`` scores

    LLR(read) = sum_{i in S} [ m_i * log(p_T,i / p_H,i)
                               + (1 - m_i) * log((1 - p_T,i) / (1 - p_H,i)) ]

This is the log-likelihood ratio of the read under a tumour-origin model
versus a healthy-origin model, assuming CpGs are conditionally independent
within a read given the origin (a deliberately simple, transparent model —
the point of an *oracle* is to be the cleanest possible separability test, not
the final estimator).

Convention note
---------------
Reads are parsed via :func:`tapestry.core.io.load_reads`, so the 0/1
methylation encoding is whatever that loader produces, applied *identically*
to reference-profile estimation and to read scoring. The oracle's AUC is
therefore invariant to the bisulfite-vs-TAPS convention question: it only
requires that tumour references, healthy references and scored reads share one
encoding, which reusing a single loader guarantees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from tapestry.core.datatypes import Region
from tapestry.core.io import load_reads
from tapestry.markers.regions import assign_reads_to_regions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region loading
# ---------------------------------------------------------------------------

def load_regions_from_bed(
    bed_path: str | Path,
    cpg_index: dict[str, np.ndarray],
    min_cpgs: int = 4,
) -> list[Region]:
    """Load genomic blocks from a BED file and attach their CpG positions.

    Each retained region's ``cpg_positions`` are the genomic CpG coordinates
    from *cpg_index* that fall within ``[start, end)``. Regions with fewer than
    *min_cpgs* CpGs are dropped.

    Parameters
    ----------
    bed_path : str or Path
        BED file with at least ``chrom``, ``start``, ``end`` in the first three
        tab-separated columns. Extra columns (e.g. ``startCpG``, ``endCpG``)
        are ignored. ``#``-prefixed and ``track``/``browser`` lines are skipped.
    cpg_index : dict[str, np.ndarray]
        Mapping chromosome -> sorted genomic CpG positions
        (see :func:`tapestry.core.cpg_index.load_cpg_index`).
    min_cpgs : int
        Minimum number of CpGs a region must contain to be kept.

    Returns
    -------
    list[Region]
        Regions sorted by ``(chrom, start)``.
    """
    import gzip

    bed_path = Path(bed_path)
    opener = gzip.open if bed_path.suffix == ".gz" else open

    regions: list[Region] = []
    with opener(bed_path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith(("track", "browser")):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            positions = cpg_index.get(chrom)
            if positions is None or len(positions) == 0:
                continue
            left = int(np.searchsorted(positions, start, side="left"))
            right = int(np.searchsorted(positions, end, side="left"))
            cpg_positions = positions[left:right]
            if len(cpg_positions) < min_cpgs:
                continue

            regions.append(
                Region(
                    region_id=f"{chrom}:{start}-{end}",
                    chrom=chrom,
                    start=start,
                    end=end,
                    cpg_positions=cpg_positions.copy(),
                )
            )

    regions.sort(key=lambda r: (r.chrom, r.start))
    logger.info("Loaded %d regions (>= %d CpGs) from %s", len(regions), min_cpgs, bed_path)
    return regions


# ---------------------------------------------------------------------------
# Reference profiles
# ---------------------------------------------------------------------------

@dataclass
class ReferenceProfiles:
    """Per-CpG tumour/healthy methylation profiles for a panel of regions.

    For every region id, ``meth``/``total`` hold per-CpG methylated and
    observed-read counts (aligned to that region's ``cpg_positions`` order),
    accumulated separately for the tumour and healthy reference groups.

    The smoothed probabilities :meth:`p_tumour` / :meth:`p_healthy` apply a
    Beta(``prior``, ``prior``) (Jeffreys when ``prior == 0.5``) pseudocount.
    """

    region_ids: list[str]
    cpg_positions: dict[str, np.ndarray]
    tumour_meth: dict[str, np.ndarray]
    tumour_total: dict[str, np.ndarray]
    healthy_meth: dict[str, np.ndarray]
    healthy_total: dict[str, np.ndarray]
    prior: float = 0.5
    n_tumour_samples: int = 0
    n_healthy_samples: int = 0

    def p_tumour(self, region_id: str) -> np.ndarray:
        return (self.tumour_meth[region_id] + self.prior) / (
            self.tumour_total[region_id] + 2.0 * self.prior
        )

    def p_healthy(self, region_id: str) -> np.ndarray:
        return (self.healthy_meth[region_id] + self.prior) / (
            self.healthy_total[region_id] + 2.0 * self.prior
        )


def _accumulate(
    pat_paths: Iterable[str | Path],
    regions: list[Region],
    cpg_index: dict[str, np.ndarray],
    meth: dict[str, np.ndarray],
    total: dict[str, np.ndarray],
    min_cpgs_overlap: int,
) -> int:
    """Accumulate per-CpG methylated/observed counts over a group of PAT files.

    Counts are weighted by the PAT ``count`` (collapsed identical patterns), so
    one row representing *k* reads contributes *k*. Returns the number of files
    successfully processed.
    """
    n_files = 0
    for pat_path in pat_paths:
        pat_path = Path(pat_path)
        if not pat_path.exists():
            logger.warning("PAT file missing, skipping: %s", pat_path)
            continue
        reads = load_reads(pat_path, cpg_index, min_cpgs=min_cpgs_overlap)
        reads_by_region = assign_reads_to_regions(
            reads, regions, min_cpgs_overlap=min_cpgs_overlap
        )
        for rid, read_rows in reads_by_region.items():
            m = meth[rid]
            t = total[rid]
            for read, row in read_rows:
                w = int(getattr(read, "count", 1) or 1)
                observed = row != -1
                m[observed] += w * (row[observed] == 1)
                t[observed] += w
        n_files += 1
    return n_files


def estimate_reference_profiles(
    regions: list[Region],
    tumour_pat_paths: Iterable[str | Path],
    healthy_pat_paths: Iterable[str | Path],
    cpg_index: dict[str, np.ndarray],
    prior: float = 0.5,
    min_cpgs_overlap: int = 1,
) -> ReferenceProfiles:
    """Estimate per-CpG tumour and healthy methylation profiles.

    Parameters
    ----------
    regions : list[Region]
        The block panel (e.g. from :func:`load_regions_from_bed`).
    tumour_pat_paths, healthy_pat_paths : iterable of path
        PAT files for the tumour and healthy reference groups. These must be
        the **training** samples — score reads only from held-out samples to
        keep the oracle honest.
    cpg_index : dict[str, np.ndarray]
        Genomic CpG index.
    prior : float
        Beta pseudocount per side (0.5 = Jeffreys).
    min_cpgs_overlap : int
        Minimum CpGs a read must share with a region to contribute.

    Returns
    -------
    ReferenceProfiles
    """
    region_ids = [r.region_id for r in regions]
    cpg_positions = {r.region_id: r.cpg_positions for r in regions}
    tumour_meth = {r.region_id: np.zeros(len(r.cpg_positions), dtype=np.float64) for r in regions}
    tumour_total = {r.region_id: np.zeros(len(r.cpg_positions), dtype=np.float64) for r in regions}
    healthy_meth = {r.region_id: np.zeros(len(r.cpg_positions), dtype=np.float64) for r in regions}
    healthy_total = {r.region_id: np.zeros(len(r.cpg_positions), dtype=np.float64) for r in regions}

    tumour_pat_paths = list(tumour_pat_paths)
    healthy_pat_paths = list(healthy_pat_paths)

    n_t = _accumulate(
        tumour_pat_paths, regions, cpg_index, tumour_meth, tumour_total, min_cpgs_overlap
    )
    n_h = _accumulate(
        healthy_pat_paths, regions, cpg_index, healthy_meth, healthy_total, min_cpgs_overlap
    )
    logger.info(
        "Estimated reference profiles from %d tumour and %d healthy samples over %d regions",
        n_t, n_h, len(regions),
    )

    return ReferenceProfiles(
        region_ids=region_ids,
        cpg_positions=cpg_positions,
        tumour_meth=tumour_meth,
        tumour_total=tumour_total,
        healthy_meth=healthy_meth,
        healthy_total=healthy_total,
        prior=prior,
        n_tumour_samples=n_t,
        n_healthy_samples=n_h,
    )


# ---------------------------------------------------------------------------
# Read scoring
# ---------------------------------------------------------------------------

@dataclass
class ReadScores:
    """Per-read scoring output, ready for AUC / stratified analysis.

    All arrays are aligned row-for-row. ``label`` is 1 for tumour-origin reads
    and 0 for healthy-origin reads. ``weight`` is the PAT ``count`` (number of
    physical reads the row represents). ``llr`` may be NaN for reads that
    covered no usably-estimated CpG; callers should drop NaNs before metrics.
    """

    llr: np.ndarray
    n_cpg_scored: np.ndarray
    weight: np.ndarray
    label: np.ndarray
    sample_id: np.ndarray
    cohort: np.ndarray

    def as_dataframe(self):  # pragma: no cover - thin pandas wrapper
        import pandas as pd

        return pd.DataFrame(
            {
                "llr": self.llr,
                "n_cpg_scored": self.n_cpg_scored,
                "weight": self.weight,
                "label": self.label,
                "sample_id": self.sample_id,
                "cohort": self.cohort,
            }
        )


def score_reads(
    regions: list[Region],
    pat_paths: Iterable[tuple[str | Path, str, str]],
    label: int,
    profiles: ReferenceProfiles,
    cpg_index: dict[str, np.ndarray],
    min_cpgs_overlap: int = 1,
    min_ref_obs: int = 5,
    prob_clip: float = 1e-3,
) -> ReadScores:
    """Score reads from a group of held-out PAT files by per-read LLR.

    Parameters
    ----------
    regions : list[Region]
        Same panel used for the profiles.
    pat_paths : iterable of (path, sample_id, cohort)
        Held-out PAT files to score, each tagged with a sample id and cohort.
    label : int
        Origin label to assign every read in this group (1 = tumour, 0 = healthy).
    profiles : ReferenceProfiles
        Reference profiles estimated on the **training** samples.
    cpg_index : dict[str, np.ndarray]
        Genomic CpG index.
    min_cpgs_overlap : int
        Minimum CpGs a read must share with a region to be scored.
    min_ref_obs : int
        A CpG contributes to a read's LLR only if it has at least this many
        observed reads in **both** the tumour and healthy reference groups, so
        the LLR is never driven by an unestimated site.
    prob_clip : float
        Reference probabilities are clipped to ``[prob_clip, 1 - prob_clip]``
        to bound per-CpG log-ratios.

    Returns
    -------
    ReadScores
    """
    llr_list: list[float] = []
    ncpg_list: list[int] = []
    weight_list: list[int] = []
    sample_list: list[str] = []
    cohort_list: list[str] = []

    lo, hi = prob_clip, 1.0 - prob_clip

    for pat_path, sample_id, cohort in pat_paths:
        pat_path = Path(pat_path)
        if not pat_path.exists():
            logger.warning("PAT file missing, skipping: %s", pat_path)
            continue
        reads = load_reads(pat_path, cpg_index, min_cpgs=min_cpgs_overlap)
        reads_by_region = assign_reads_to_regions(
            reads, regions, min_cpgs_overlap=min_cpgs_overlap
        )
        for rid, read_rows in reads_by_region.items():
            p_t = np.clip(profiles.p_tumour(rid), lo, hi)
            p_h = np.clip(profiles.p_healthy(rid), lo, hi)
            usable = (profiles.tumour_total[rid] >= min_ref_obs) & (
                profiles.healthy_total[rid] >= min_ref_obs
            )
            log_ratio_meth = np.log(p_t / p_h)
            log_ratio_unmeth = np.log((1.0 - p_t) / (1.0 - p_h))

            for read, row in read_rows:
                observed = (row != -1) & usable
                if not observed.any():
                    continue
                states = row[observed]
                contrib = np.where(states == 1, log_ratio_meth[observed], log_ratio_unmeth[observed])
                llr_list.append(float(contrib.sum()))
                ncpg_list.append(int(observed.sum()))
                weight_list.append(int(getattr(read, "count", 1) or 1))
                sample_list.append(sample_id)
                cohort_list.append(cohort)

    return ReadScores(
        llr=np.asarray(llr_list, dtype=np.float64),
        n_cpg_scored=np.asarray(ncpg_list, dtype=np.int32),
        weight=np.asarray(weight_list, dtype=np.int64),
        label=np.full(len(llr_list), label, dtype=np.int8),
        sample_id=np.asarray(sample_list, dtype=object),
        cohort=np.asarray(cohort_list, dtype=object),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _weighted_auc(label: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    """Weighted ROC-AUC; returns NaN if a class is absent."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(label)) < 2:
        return float("nan")
    return float(roc_auc_score(label, score, sample_weight=weight))


def _merge_scores(scores: Iterable[ReadScores]) -> ReadScores:
    scores = list(scores)
    return ReadScores(
        llr=np.concatenate([s.llr for s in scores]),
        n_cpg_scored=np.concatenate([s.n_cpg_scored for s in scores]),
        weight=np.concatenate([s.weight for s in scores]),
        label=np.concatenate([s.label for s in scores]),
        sample_id=np.concatenate([s.sample_id for s in scores]),
        cohort=np.concatenate([s.cohort for s in scores]),
    )


def compute_oracle_metrics(
    tumour_scores: ReadScores,
    healthy_scores: ReadScores,
    n_cpg_buckets: tuple[int, ...] = (1, 2, 4, 6, 8, 12),
    rng_seed: int = 0,
) -> dict:
    """Compute the oracle's verdict metrics from scored reads.

    Returns a dict with:

    * ``auc`` — overall weighted per-read ROC-AUC. The headline number.
    * ``auc_by_n_cpg`` — AUC within strata of the number of CpGs scored per
      read; genuine signal is roughly flat and well above 0.5, whereas a value
      that climbs only with CpG count points at a coverage/length artefact.
    * ``auc_permuted`` — AUC after shuffling the tumour/healthy labels; the
      null. Should sit at ~0.5. If the real AUC is not clearly above this, the
      "signal" is noise.
    * ``auc_ncpg_matched`` — AUC after subsampling so the tumour and healthy
      reads share the same per-read CpG-count distribution; removes the
      crudest read-length/coverage confound at the read level.
    * counts and a one-line ``verdict``.
    """
    merged = _merge_scores([tumour_scores, healthy_scores])
    finite = np.isfinite(merged.llr)
    llr = merged.llr[finite]
    label = merged.label[finite].astype(int)
    weight = merged.weight[finite].astype(float)
    ncpg = merged.n_cpg_scored[finite]

    n_t = int(weight[label == 1].sum())
    n_h = int(weight[label == 0].sum())

    auc = _weighted_auc(label, llr, weight)

    # Stratified by number of CpGs scored.
    edges = list(n_cpg_buckets) + [np.iinfo(np.int32).max]
    auc_by_n_cpg: dict[str, dict] = {}
    for lo_b, hi_b in zip(edges[:-1], edges[1:]):
        mask = (ncpg >= lo_b) & (ncpg < hi_b)
        if mask.sum() == 0:
            continue
        name = f"{lo_b}" if hi_b == lo_b + 1 else f"{lo_b}-{hi_b - 1 if hi_b != edges[-1] else 'inf'}"
        auc_by_n_cpg[name] = {
            "auc": _weighted_auc(label[mask], llr[mask], weight[mask]),
            "n_tumour": int(weight[mask & (label == 1)].sum()),
            "n_healthy": int(weight[mask & (label == 0)].sum()),
        }

    # Label-permutation null.
    rng = np.random.default_rng(rng_seed)
    permuted = rng.permutation(label)
    auc_permuted = _weighted_auc(permuted, llr, weight)

    # CpG-count-matched AUC: equalise the n_cpg histogram across classes by
    # weighting each class to the per-bucket minimum total weight.
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
    auc_ncpg_matched = (
        _weighted_auc(label[keep], llr[keep], match_weight[keep]) if keep.any() else float("nan")
    )

    separable = np.isfinite(auc) and auc >= 0.60 and (auc - 0.5) > 2 * abs(auc_permuted - 0.5)
    verdict = (
        "SEPARABLE at read level — build the read-level deconvolution programme"
        if separable
        else "NOT separable at read level on this panel — fix marker geometry or "
        "fall back to sample-level statistics"
    )

    return {
        "auc": auc,
        "auc_permuted": auc_permuted,
        "auc_ncpg_matched": auc_ncpg_matched,
        "auc_by_n_cpg": auc_by_n_cpg,
        "n_tumour_reads": n_t,
        "n_healthy_reads": n_h,
        "n_reads_scored": int(weight.sum()),
        "mean_llr_tumour": float(np.average(llr[label == 1], weights=weight[label == 1]))
        if (label == 1).any()
        else float("nan"),
        "mean_llr_healthy": float(np.average(llr[label == 0], weights=weight[label == 0]))
        if (label == 0).any()
        else float("nan"),
        "verdict": verdict,
        "separable": bool(separable),
    }
