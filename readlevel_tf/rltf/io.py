"""PAT reading in global CpG-index space, with explicit methylation convention.

Standalone: this module vendors a minimal PAT reader so the project depends on
no other code or artifact. PAT lines are ``chrom  start_cpg  pattern  count``
where ``start_cpg`` is the 1-based **global** CpG index of the first CpG and
``pattern`` is a per-CpG string. Everything downstream works purely in this
global CpG-index coordinate, so no external genomic CpG index is needed.

Methylation is normalised to one internal convention — ``1 = methylated``,
``0 = unmethylated``, ``-1 = missing`` — and the per-sample sequencing
convention is applied here so tumour-tissue (bisulfite) and TAPS cfDNA reads end
up on the same scale:

* ``bisulfite``: ``C`` = methylated, ``T`` = unmethylated.
* ``taps`` (native): ``T`` = methylated, ``C`` = unmethylated.

The project owns this flip itself (no reliance on pre-flipped inputs).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

_BISULFITE = {"C": 1, "T": 0}
_TAPS = {"T": 1, "C": 0}


def char_map(convention: str) -> dict:
    """Return the char→state map for a sequencing convention."""
    c = convention.strip().lower()
    if c in ("bisulfite", "bs", "wgbs"):
        return _BISULFITE
    if c in ("taps", "native"):
        return _TAPS
    raise ValueError(f"unknown convention '{convention}' (use 'bisulfite' or 'taps')")


@dataclass
class Read:
    """A PAT row in global CpG-index space."""

    chrom: str
    start_cpg: int           # 1-based global CpG index of the first CpG
    states: np.ndarray       # int8, one per CpG: 1=meth, 0=unmeth, -1=missing
    count: int = 1

    @property
    def end_cpg(self) -> int:
        """Exclusive global CpG index past the last covered CpG."""
        return self.start_cpg + len(self.states)


def load_reads(
    path: str | Path,
    convention: str,
    min_cpgs: int = 1,
) -> Iterator[Read]:
    """Stream :class:`Read` objects from a (optionally gzipped) PAT file.

    Parameters
    ----------
    path : str or Path
        ``.pat`` / ``.pat.gz`` file.
    convention : str
        ``bisulfite`` or ``taps`` — applied to map chars to methylation state.
    min_cpgs : int
        Skip patterns shorter than this.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    cmap = char_map(convention)

    with opener(path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pattern = parts[2]
            if len(pattern) < min_cpgs:
                continue
            states = np.fromiter(
                (cmap.get(ch, -1) for ch in pattern), dtype=np.int8, count=len(pattern)
            )
            count = int(parts[3]) if len(parts) > 3 and parts[3] else 1
            yield Read(chrom=parts[0], start_cpg=int(parts[1]), states=states, count=count)
