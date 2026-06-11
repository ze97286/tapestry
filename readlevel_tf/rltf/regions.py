"""Blocks (candidate markers) in global CpG-index space.

A :class:`Block` is a contiguous run of ``n_cpg`` CpGs ``[start_cpg, end_cpg)``
in the PAT global CpG index. Because CpG indices are contiguous integers,
read↔block overlap and per-block read rows are exact integer-slice operations —
no genomic coordinates, no searchsorted on positions.

Blocks are non-overlapping (tiling, or a selected subset of a tiling), which the
:class:`BlockIndex` overlap query relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rltf.io import Read


@dataclass
class Block:
    block_id: str
    chrom: str
    start_cpg: int   # inclusive, global
    end_cpg: int     # exclusive, global

    @property
    def n_cpg(self) -> int:
        return self.end_cpg - self.start_cpg


def tile_blocks(
    chrom_bounds: dict[str, tuple[int, int]],
    window: int,
    step: int | None = None,
) -> list[Block]:
    """Tile fixed-width blocks of *window* CpGs within each chromosome's bounds.

    *chrom_bounds* maps chrom → (min_cpg, max_cpg) inclusive global indices.
    Default *step* = *window* (non-overlapping).
    """
    step = step or window
    blocks: list[Block] = []
    for chrom, (lo, hi) in sorted(chrom_bounds.items()):
        s = lo
        while s + window <= hi + 1:
            blocks.append(Block(f"{chrom}:{s}-{s + window}", chrom, s, s + window))
            s += step
    return blocks


def read_block_row(block: Block, read: Read) -> np.ndarray | None:
    """Row of methylation states for *read* over *block*'s CpGs.

    Length ``block.n_cpg``; entries 1/0 where the read covers that CpG and the
    call is present, ``-1`` otherwise. Returns ``None`` if there is no overlap.
    """
    a = max(block.start_cpg, read.start_cpg)
    z = min(block.end_cpg, read.end_cpg)
    if z <= a:
        return None
    row = np.full(block.n_cpg, -1, dtype=np.int8)
    row[a - block.start_cpg : z - block.start_cpg] = read.states[a - read.start_cpg : z - read.start_cpg]
    return row


class BlockIndex:
    """Fast read→block overlap lookup for a set of non-overlapping blocks."""

    def __init__(self, blocks: list[Block]) -> None:
        self.blocks = blocks
        by_chrom: dict[str, list[int]] = {}
        for i, b in enumerate(blocks):
            by_chrom.setdefault(b.chrom, []).append(i)
        self._idx: dict[str, np.ndarray] = {}
        self._starts: dict[str, np.ndarray] = {}
        self._ends: dict[str, np.ndarray] = {}
        for chrom, idxs in by_chrom.items():
            idxs = sorted(idxs, key=lambda i: blocks[i].start_cpg)
            self._idx[chrom] = np.asarray(idxs, dtype=np.int64)
            self._starts[chrom] = np.asarray([blocks[i].start_cpg for i in idxs], dtype=np.int64)
            self._ends[chrom] = np.asarray([blocks[i].end_cpg for i in idxs], dtype=np.int64)

    def overlapping(self, chrom: str, start_cpg: int, end_cpg: int) -> np.ndarray:
        """Indices into ``self.blocks`` of blocks overlapping ``[start_cpg, end_cpg)``."""
        if chrom not in self._starts:
            return np.empty(0, dtype=np.int64)
        starts = self._starts[chrom]
        ends = self._ends[chrom]
        lo = int(np.searchsorted(ends, start_cpg, side="right"))   # end > start_cpg
        hi = int(np.searchsorted(starts, end_cpg, side="left"))    # start < end_cpg
        return self._idx[chrom][lo:hi]

    def assign(self, read: Read) -> list[tuple[int, np.ndarray]]:
        """Return [(block_index, row)] for every block this read overlaps."""
        out: list[tuple[int, np.ndarray]] = []
        for bi in self.overlapping(read.chrom, read.start_cpg, read.end_cpg):
            row = read_block_row(self.blocks[int(bi)], read)
            if row is not None:
                out.append((int(bi), row))
        return out
