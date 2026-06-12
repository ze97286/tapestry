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
from collections import defaultdict
from typing import Sequence

import numpy as np

from rltf.io import load_fragments, load_fragments_chrom
from rltf.profiles import ReferenceProfiles
from rltf.regions import tile_blocks

logger = logging.getLogger(__name__)


def _tally(samples: Sequence[str], tally: dict, slot: int, observed: dict, min_mapq: int,
           chroms: set | None) -> int:
    """Accumulate methylated/total counts into ``tally[(chrom,pos)][slot:slot+2]``.

    When *chroms* is given, each chromosome is read via tabix (load_fragments_chrom),
    so a shard touches only ~1/22 of each file.
    """
    n = 0
    for path in samples:
        if chroms is None:
            frag_iter = load_fragments(path, min_mapq=min_mapq)
        else:
            frag_iter = (frag for c in sorted(chroms) for frag in load_fragments_chrom(path, c, min_mapq=min_mapq))
        for frag in frag_iter:
            chrom = frag.chrom
            obs = observed[chrom]
            for pos, st in zip(frag.cpg_pos.tolist(), frag.states.tolist()):
                key = (chrom, pos)
                e = tally.get(key)
                if e is None:
                    e = [0, 0, 0, 0]
                    tally[key] = e
                    obs.add(pos)
                e[slot] += int(st)      # methylated count
                e[slot + 1] += 1        # observed count
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
) -> ReferenceProfiles:
    """Discover a marker panel + per-CpG profiles from reference per-read calls.

    *chroms* restricts the tally to those chromosomes (for sharded genome-wide
    discovery); None processes all observed chromosomes.
    """
    tally: dict[tuple[str, int], list] = {}
    observed: dict[str, set] = defaultdict(set)
    n_t = _tally(tumour_samples, tally, 0, observed, min_mapq, chroms)
    n_h = _tally(healthy_samples, tally, 2, observed, min_mapq, chroms)
    logger.info("Tallied %d tumour + %d healthy reference samples over %d CpGs",
                n_t, n_h, len(tally))

    cpg_by_chrom = {c: np.array(sorted(s), dtype=np.int64) for c, s in observed.items()}
    blocks = tile_blocks(cpg_by_chrom, window=window)
    logger.info("Tiled %d candidate blocks (window=%d)", len(blocks), window)

    sel_blocks, sel_chrom, sel_pos = [], [], []
    sel_mt, sel_tt, sel_mh, sel_th = [], [], [], []
    scored = []
    for b in blocks:
        rows = [tally[(b.chrom, int(p))] for p in b.cpg_pos]
        mt = np.array([r[0] for r in rows], float); tt = np.array([r[1] for r in rows], float)
        mh = np.array([r[2] for r in rows], float); th = np.array([r[3] for r in rows], float)
        if tt.min() < min_total or th.min() < min_total:
            continue
        p_t = (mt + prior) / (tt + 2 * prior)
        p_h = (mh + prior) / (th + 2 * prior)
        effect = float(np.mean(np.abs(p_t - p_h)))
        if effect < min_effect:
            continue
        mean_diff = float(p_t.mean() - p_h.mean())
        if direction == "hypo" and mean_diff >= 0:
            continue
        if direction == "hyper" and mean_diff <= 0:
            continue
        score = effect * np.sqrt(min(tt.min(), th.min()))
        scored.append((score, b, mt, tt, mh, th))

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
