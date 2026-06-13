"""Marker-panel discovery from raw per-read calls (genomic CpG space).

Streams tumour-tissue and healthy-control fragments (mate-deduped, SNP-masked,
MAPQ-filtered by the reader), tallies per-CpG methylation by genomic position,
tiles windows of consecutive CpGs, and keeps the discriminative, well-covered
ones. No PAT, no CpG-index, no convention — the calls are explicit mod/unmod.

Scale: the per-CpG tally is a dict keyed by ``(chrom, pos)``. Genome-wide this is
large; for the real cohort run discovery sharded by chromosome (the SLURM job
arrays over chromosomes and merges). Discovery touches only the handful of
reference samples.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from rltf.io import load_fragments, load_fragments_chrom
from rltf.profiles import ReferenceProfiles
from rltf.regions import tile_blocks

logger = logging.getLogger(__name__)

_LOG_EVERY = 5_000_000   # reads


def _split_samples(samples: Sequence[str], seed: int) -> tuple[list, list]:
    """Deterministic ~50/50 split of samples into (selection, estimation) folds."""
    ordered = sorted(samples)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ordered))
    k = len(ordered) // 2
    sel = [ordered[i] for i in idx[:k]]
    est = [ordered[i] for i in idx[k:]]
    return sel, est


def _tally(samples: Sequence[str], tally: dict, slot: int, observed: dict, min_mapq: int,
           chroms: set | None, label: str = "", width: int = 4) -> int:
    """Accumulate methylated/total counts into ``tally[(chrom,pos)][slot:slot+2]``.

    Entries are *width*-long (4 normally; 8 for cross-fit, holding selection- and
    estimation-fold counters). When *chroms* is given, each chromosome is read via
    tabix so a shard touches only ~1/22 of each file. Logs progress.
    """
    n = 0
    nf = len(samples)
    for fi, path in enumerate(samples, 1):
        if chroms is None:
            frag_iter = load_fragments(path, min_mapq=min_mapq)
        else:
            frag_iter = (frag for c in sorted(chroms) for frag in load_fragments_chrom(path, c, min_mapq=min_mapq))
        logger.info("tally %s [%d/%d] %s", label, fi, nf, Path(path).name)
        t0 = time.monotonic()
        reads = 0
        for frag in frag_iter:
            chrom = frag.chrom
            obs = observed[chrom]
            for pos, st in zip(frag.cpg_pos.tolist(), frag.states.tolist()):
                key = (chrom, pos)
                e = tally.get(key)
                if e is None:
                    e = [0] * width
                    tally[key] = e
                    obs.add(pos)
                e[slot] += int(st)      # methylated count
                e[slot + 1] += 1        # observed count
            reads += 1
            if reads % _LOG_EVERY == 0:
                rate = reads / max(time.monotonic() - t0, 1e-6)
                logger.info("  %s [%d/%d]: %d reads (%.0fk reads/s), %d CpGs tallied",
                            Path(path).name, fi, nf, reads, rate / 1000, len(tally))
        logger.info("  %s done: %d reads in %.0fs", Path(path).name, reads, time.monotonic() - t0)
        n += 1
    return n


def discover_panel(
    tumour_samples: Sequence[str],
    healthy_samples: Sequence[str],
    window: int = 5,
    top_n: int | None = 2000,
    min_total: float = 10.0,
    min_effect: float = 0.3,
    direction: str = "any",
    prior: float = 0.5,
    min_mapq: int = 30,
    chroms: set | None = None,
    cross_fit: bool = True,
    cross_fit_seed: int = 0,
) -> ReferenceProfiles:
    """Discover a marker panel + per-CpG profiles from reference per-read calls.

    **Cross-fit (default):** to avoid winner's-curse, the reference samples are
    split into disjoint selection and estimation folds — markers are *chosen* by
    effect size on the selection fold but the stored ``p_T``/``p_H`` profiles are
    *estimated* on the held-out estimation fold. This decouples selection from
    estimation so held-out healthy reads are not systematically tumour-ward
    (``null_mean_z`` ≈ 0). Requires ≥2 samples per group; otherwise falls back to
    the non-cross-fit path. Coverage filters apply to **all** folds used, so
    cross-fit needs more reference coverage to keep the same marker count.

    *chroms* restricts the tally to those chromosomes (sharded discovery).
    """
    do_cf = cross_fit and len(tumour_samples) >= 2 and len(healthy_samples) >= 2
    observed: dict[str, set] = defaultdict(set)
    tally: dict[tuple[str, int], list] = {}

    if do_cf:
        sel_t, est_t = _split_samples(tumour_samples, cross_fit_seed)
        sel_h, est_h = _split_samples(healthy_samples, cross_fit_seed + 1)
        logger.info("Cross-fit: select on %d tumour / %d healthy, estimate on %d tumour / %d healthy",
                    len(sel_t), len(sel_h), len(est_t), len(est_h))
        _tally(sel_t, tally, 0, observed, min_mapq, chroms, label="sel-tumour", width=8)
        _tally(sel_h, tally, 2, observed, min_mapq, chroms, label="sel-healthy", width=8)
        _tally(est_t, tally, 4, observed, min_mapq, chroms, label="est-tumour", width=8)
        _tally(est_h, tally, 6, observed, min_mapq, chroms, label="est-healthy", width=8)
        S_T, S_H, E_T, E_H = 0, 2, 4, 6
    else:
        _tally(tumour_samples, tally, 0, observed, min_mapq, chroms, label="tumour", width=4)
        _tally(healthy_samples, tally, 2, observed, min_mapq, chroms, label="healthy", width=4)
        S_T = E_T = 0
        S_H = E_H = 2
    n_t, n_h = len(tumour_samples), len(healthy_samples)
    logger.info("Tallied %d tumour + %d healthy reference samples over %d CpGs (cross_fit=%s)",
                n_t, n_h, len(tally), do_cf)

    cpg_by_chrom = {c: np.array(sorted(s), dtype=np.int64) for c, s in observed.items()}
    blocks = tile_blocks(cpg_by_chrom, window=window)
    logger.info("Tiled %d candidate blocks (window=%d)", len(blocks), window)

    sel_blocks, sel_chrom, sel_pos = [], [], []
    sel_mt, sel_tt, sel_mh, sel_th = [], [], [], []
    scored = []
    for b in blocks:
        arr = np.array([tally[(b.chrom, int(p))] for p in b.cpg_pos], dtype=float)
        # selection-fold counts (choose markers), estimation-fold counts (store profile)
        s_tt, s_th = arr[:, S_T + 1], arr[:, S_H + 1]
        e_mt, e_tt, e_mh, e_th = arr[:, E_T], arr[:, E_T + 1], arr[:, E_H], arr[:, E_H + 1]
        if s_tt.min() < min_total or s_th.min() < min_total or e_tt.min() < min_total or e_th.min() < min_total:
            continue
        p_t_sel = (arr[:, S_T] + prior) / (s_tt + 2 * prior)
        p_h_sel = (arr[:, S_H] + prior) / (s_th + 2 * prior)
        effect = float(np.mean(np.abs(p_t_sel - p_h_sel)))
        if effect < min_effect:
            continue
        mean_diff = float(p_t_sel.mean() - p_h_sel.mean())
        if direction == "hypo" and mean_diff >= 0:
            continue
        if direction == "hyper" and mean_diff <= 0:
            continue
        score = effect * np.sqrt(min(s_tt.min(), s_th.min()))
        scored.append((score, b, e_mt, e_tt, e_mh, e_th))   # PROFILE from estimation fold

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored if top_n is None else scored[:top_n]
    logger.info("Selected %d / %d blocks passing coverage+effect filters", len(selected), len(scored))
    if not selected:
        raise SystemExit("No blocks passed discovery filters; relax min_total/min_effect/window.")

    for score, b, mt, tt, mh, th in selected:
        b.score = float(score)
        sel_blocks.append(b)
        for k, p in enumerate(b.cpg_pos.tolist()):
            sel_chrom.append(b.chrom); sel_pos.append(p)
            sel_mt.append(mt[k]); sel_tt.append(tt[k]); sel_mh.append(mh[k]); sel_th.append(th[k])

    return ReferenceProfiles(
        blocks=sel_blocks, cpg_chrom=sel_chrom, cpg_pos=np.asarray(sel_pos, dtype=np.int64),
        meth_tumour=np.asarray(sel_mt), total_tumour=np.asarray(sel_tt),
        meth_healthy=np.asarray(sel_mh), total_healthy=np.asarray(sel_th),
        prior=prior, n_tumour_samples=n_t, n_healthy_samples=n_h,
    )


def merge_partials(partial_dirs: Sequence[str], top_n: int) -> ReferenceProfiles:
    """Merge per-chromosome discovery partials into the global top-N panel.

    Each partial is a saved :class:`ReferenceProfiles` of *all* surviving blocks
    on one chromosome (discovered with ``top_n=None``). Blocks are ranked
    globally by their discovery score and the top-N kept, with per-CpG profiles
    rebuilt from the owning partial.
    """
    parts = [ReferenceProfiles.load(d) for d in partial_dirs]
    ranked = sorted(((b.score, pi, b) for pi, p in enumerate(parts) for b in p.blocks),
                    key=lambda x: x[0], reverse=True)[:top_n]
    if not ranked:
        raise SystemExit("No blocks across partials; check the shard outputs.")

    sel_blocks, chrom, pos, mt, tt, mh, th = [], [], [], [], [], [], []
    for _score, pi, b in ranked:
        p = parts[pi]
        sel_blocks.append(b)
        for cp in b.cpg_pos.tolist():
            idx = p.index.get((b.chrom, int(cp)))
            if idx is None:
                continue
            chrom.append(b.chrom); pos.append(int(cp))
            mt.append(p.meth_tumour[idx]); tt.append(p.total_tumour[idx])
            mh.append(p.meth_healthy[idx]); th.append(p.total_healthy[idx])

    return ReferenceProfiles(
        blocks=sel_blocks, cpg_chrom=chrom, cpg_pos=np.asarray(pos, dtype=np.int64),
        meth_tumour=np.asarray(mt), total_tumour=np.asarray(tt),
        meth_healthy=np.asarray(mh), total_healthy=np.asarray(th),
        prior=parts[0].prior, n_tumour_samples=parts[0].n_tumour_samples,
        n_healthy_samples=parts[0].n_healthy_samples,
    )
