"""Blocks (candidate markers) as windows of consecutive CpGs in genomic space.

A :class:`Block` is ``window`` genomically-adjacent CpGs on one chromosome,
identified by their genomic positions. Blocks tile the observed-CpG universe;
discovery keeps the discriminative ones and the panel is the union of their CpGs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Block:
    block_id: str
    chrom: str
    cpg_pos: np.ndarray  # int64, sorted genomic positions (length = window)
    score: float = 0.0   # discovery discriminativeness score (for cross-shard ranking)

    @property
    def n_cpg(self) -> int:
        return len(self.cpg_pos)


def tile_blocks(cpg_by_chrom: dict[str, np.ndarray], window: int, step: int | None = None) -> list[Block]:
    """Tile windows of *window* consecutive CpGs across each chromosome.

    *cpg_by_chrom* maps chrom -> sorted array of observed CpG genomic positions.
    """
    step = step or window
    blocks: list[Block] = []
    for chrom in sorted(cpg_by_chrom):
        pos = cpg_by_chrom[chrom]
        i = 0
        while i + window <= len(pos):
            cps = pos[i:i + window]
            blocks.append(Block(f"{chrom}:{int(cps[0])}-{int(cps[-1])}", chrom, np.asarray(cps, dtype=np.int64)))
            i += step
    return blocks
