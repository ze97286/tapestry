"""Marker-panel discovery from raw reference PATs (self-contained).

The project defines its **own** markers — it does not consume any segmentation,
DMR panel or marker table from another method. From tumour-tissue and
healthy-control reference PATs it tallies per-CpG methylation genome-wide (in
global CpG-index space), tiles fixed-width candidate blocks, and selects the
most discriminative, well-covered, CpG-dense blocks. The result is a
:class:`~rltf.profiles.ReferenceProfiles` (panel + per-CpG profiles).

Scale: the global tally is a handful of float32 arrays sized to the max CpG
index (~28.2M for hg38 ⇒ ~0.45 GB for 4 arrays), filled in one streaming pass
over the reference PATs. Selection over the tiling is cheap. Plan ~32 GB RAM,
single CPU, minutes-to-hours depending on reference depth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from rltf.io import load_reads
from rltf.profiles import ReferenceProfiles
from rltf.regions import Block, tile_blocks

logger = logging.getLogger(__name__)


class _GlobalTally:
    """Growable global per-CpG methylated/observed counts for two groups."""

    def __init__(self, init_size: int = 1 << 20) -> None:
        self.size = init_size
        self.meth = {"tumour": np.zeros(init_size, np.float32), "healthy": np.zeros(init_size, np.float32)}
        self.total = {"tumour": np.zeros(init_size, np.float32), "healthy": np.zeros(init_size, np.float32)}
        self.chrom_bounds: dict[str, list[int]] = {}

    def _ensure(self, n: int) -> None:
        if n <= self.size:
            return
        new = self.size
        while new < n:
            new *= 2
        for d in (self.meth, self.total):
            for k in d:
                grown = np.zeros(new, np.float32)
                grown[: self.size] = d[k]
                d[k] = grown
        self.size = new

    def add_reads(self, path: str | Path, convention: str, group: str, min_cpgs: int = 1) -> int:
        n = 0
        for read in load_reads(path, convention, min_cpgs=min_cpgs):
            self._ensure(read.end_cpg)
            idxs = np.arange(read.start_cpg, read.end_cpg)
            obs = read.states != -1
            g = idxs[obs]
            if g.size == 0:
                continue
            st = read.states[obs]
            self.total[group][g] += read.count
            self.meth[group][g] += read.count * (st == 1)
            cb = self.chrom_bounds.get(read.chrom)
            lo, hi = read.start_cpg, read.end_cpg - 1
            if cb is None:
                self.chrom_bounds[read.chrom] = [lo, hi]
            else:
                cb[0] = min(cb[0], lo)
                cb[1] = max(cb[1], hi)
            n += 1
        return n


def discover_panel(
    tumour_samples: Sequence[tuple[str, str]],
    healthy_samples: Sequence[tuple[str, str]],
    window: int = 5,
    top_n: int = 2000,
    min_total: float = 10.0,
    min_effect: float = 0.3,
    direction: str = "any",
    prior: float = 0.5,
) -> ReferenceProfiles:
    """Discover a marker panel and per-CpG profiles from reference PATs.

    Parameters
    ----------
    tumour_samples, healthy_samples : list of (pat_path, convention)
        Reference groups; ``convention`` is ``bisulfite`` or ``taps`` per sample.
    window : int
        Block width in CpGs (a single tumour molecule should span it).
    top_n : int
        Keep the top-N discriminative non-overlapping blocks.
    min_total : float
        Require every CpG in a block to have >= this observed reads in **both**
        groups (CpG-dense and well-estimated).
    min_effect : float
        Require mean per-CpG |p_tumour - p_healthy| >= this.
    direction : str
        ``hypo`` (tumour < healthy), ``hyper`` (tumour > healthy) or ``any``.
    prior : float
        Beta pseudocount used for the stored profiles.
    """
    tally = _GlobalTally()
    n_t = 0
    for path, conv in tumour_samples:
        n_t += 1
        tally.add_reads(path, conv, "tumour")
    n_h = 0
    for path, conv in healthy_samples:
        n_h += 1
        tally.add_reads(path, conv, "healthy")

    bounds = {c: (lo, hi) for c, (lo, hi) in tally.chrom_bounds.items()}
    candidates = tile_blocks(bounds, window=window)
    logger.info("Tiled %d candidate blocks (window=%d) over %d chromosomes",
                len(candidates), window, len(bounds))

    mt, tt = tally.meth["tumour"], tally.total["tumour"]
    mh, th = tally.meth["healthy"], tally.total["healthy"]

    scored: list[tuple[float, Block, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for b in candidates:
        sl = slice(b.start_cpg, b.end_cpg)
        bt_t, bt_total = mt[sl], tt[sl]
        bt_h, bt_htotal = mh[sl], th[sl]
        if bt_total.min() < min_total or bt_htotal.min() < min_total:
            continue
        p_t = (bt_t + prior) / (bt_total + 2 * prior)
        p_h = (bt_h + prior) / (bt_htotal + 2 * prior)
        effect = float(np.mean(np.abs(p_t - p_h)))
        if effect < min_effect:
            continue
        mean_diff = float(p_t.mean() - p_h.mean())
        if direction == "hypo" and mean_diff >= 0:
            continue
        if direction == "hyper" and mean_diff <= 0:
            continue
        score = effect * np.sqrt(min(bt_total.min(), bt_htotal.min()))
        scored.append((score, b, bt_t.copy(), bt_total.copy(), bt_h.copy(), bt_htotal.copy()))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:top_n]
    logger.info("Selected %d / %d blocks passing coverage+effect filters", len(selected), len(scored))
    if not selected:
        raise SystemExit("No blocks passed discovery filters; relax --min-total / --min-effect / --window.")

    blocks = [s[1] for s in selected]
    meth_tumour = np.stack([s[2] for s in selected]).astype(np.float64)
    total_tumour = np.stack([s[3] for s in selected]).astype(np.float64)
    meth_healthy = np.stack([s[4] for s in selected]).astype(np.float64)
    total_healthy = np.stack([s[5] for s in selected]).astype(np.float64)

    return ReferenceProfiles(
        blocks=blocks,
        meth_tumour=meth_tumour, total_tumour=total_tumour,
        meth_healthy=meth_healthy, total_healthy=total_healthy,
        window=window, prior=prior, n_tumour_samples=n_t, n_healthy_samples=n_h,
    )
