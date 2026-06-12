"""Per-read-call (.per-read.bed[.gz]) reading at the fragment level.

Schema (header line, tab-separated; `mod_cps` is a known typo for `mod_cpgs`,
both accepted):

    #chr start end read_id mapq orientation insert_size read_length flag
        num_cpg num_mod mod_cpgs unmod_cpgs snp_cpgs

`mod_cpgs`/`unmod_cpgs`/`snp_cpgs` are comma-separated **offsets from `start`**, so
a CpG's genomic position is ``start + offset`` (the two mates of a pair share
identical offsets). We mask SNP CpGs and keep mod∪unmod.

A sequenced **fragment** is ≥2 lines sharing one `read_id` (the mates); we
collapse by `read_id` into one :class:`Fragment` (the unit of the per-read
statistic), union the mates' CpG calls, drop CpGs whose mates disagree, and
filter low-MAPQ mates. All in genomic coordinates — no shared CpG-index
assumption. `read_length` is retained for QC only.

Two readers:
* :func:`load_fragments` — stream the whole file (windowed mate-collapse,
  memory-bounded); used by discovery.
* :func:`load_fragments_regions` — tabix region-query a panel's intervals via
  pysam (falls back to a filtered full stream without pysam); used by scoring so
  only reads near panel CpGs are touched.
"""

from __future__ import annotations

import gzip
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

FLUSH_MARGIN = 5000


@dataclass
class Fragment:
    chrom: str
    cpg_pos: np.ndarray   # int64, sorted genomic positions of usable CpGs
    states: np.ndarray    # int8, 1=methylated / 0=unmethylated, aligned
    read_length: int      # max over mates (QC only)

    @property
    def n_cpg(self) -> int:
        return len(self.cpg_pos)


def _offsets(field: str) -> list[int]:
    field = field.strip()
    if not field or field == ".":
        return []
    return [int(x) for x in field.split(",") if x]


def _column_index(header: str) -> dict[str, int]:
    names = header.lstrip("#").rstrip("\n").split("\t")
    col = {n: i for i, n in enumerate(names)}
    if "mod_cpgs" not in col and "mod_cps" in col:
        col["mod_cpgs"] = col["mod_cps"]
    return col


def _read_header(path: Path) -> dict[str, int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:  # type: ignore[call-overload]
        return _column_index(fh.readline())


def _ingest(fields: list[str], col: dict, min_mapq: int, buf: dict) -> int | None:
    """Add one line's CpG calls into ``buf[read_id]``. Returns the line's start
    (for windowed flushing), or None if the line was skipped (low MAPQ)."""
    try:
        if int(fields[col["mapq"]]) < min_mapq:
            return None
    except (ValueError, IndexError):
        return None
    chrom = fields[col["chr"]]
    start = int(fields[col["start"]])
    read_id = fields[col["read_id"]]
    snp = set(_offsets(fields[col["snp_cpgs"]]) if col["snp_cpgs"] < len(fields) else [])
    entry = buf.get(read_id)
    if entry is None:
        entry = [chrom, {}, 0]
        buf[read_id] = entry
    posmap = entry[1]
    try:
        entry[2] = max(entry[2], int(fields[col["read_length"]]))
    except (ValueError, IndexError):
        pass
    for off in _offsets(fields[col["mod_cpgs"]]):
        if off in snp:
            continue
        pos = start + off
        posmap[pos] = 1 if pos not in posmap else (1 if posmap[pos] == 1 else -1)
    for off in _offsets(fields[col["unmod_cpgs"]]):
        if off in snp:
            continue
        pos = start + off
        posmap[pos] = 0 if pos not in posmap else (0 if posmap[pos] == 0 else -1)
    return start


def _emit(entry: list) -> Fragment | None:
    chrom, posmap, rlen = entry
    items = [(p, s) for p, s in posmap.items() if s in (0, 1)]
    if not items:
        return None
    items.sort()
    return Fragment(
        chrom=chrom,
        cpg_pos=np.fromiter((p for p, _ in items), dtype=np.int64, count=len(items)),
        states=np.fromiter((s for _, s in items), dtype=np.int8, count=len(items)),
        read_length=rlen,
    )


def _stream_dedup(field_iter, col: dict, min_mapq: int, flush_margin: int) -> Iterator[Fragment]:
    """Collapse mate pairs from a start-sorted iterator of split field-lists.

    Shared by whole-file streaming and per-chromosome tabix streaming; both are
    position-sorted, so a windowed flush keeps memory bounded.
    """
    buf: dict[str, list] = {}
    queue: deque[tuple[int, str]] = deque()
    cur_chrom: str | None = None
    for f in field_iter:
        chrom = f[col["chr"]]
        start = int(f[col["start"]])
        if chrom != cur_chrom:
            while queue:
                rid = queue.popleft()[1]
                if rid in buf:
                    frag = _emit(buf.pop(rid))
                    if frag is not None:
                        yield frag
            cur_chrom = chrom
        while queue and start - queue[0][0] > flush_margin:
            rid = queue.popleft()[1]
            if rid in buf:
                frag = _emit(buf.pop(rid))
                if frag is not None:
                    yield frag
        rid = f[col["read_id"]]
        new = rid not in buf
        s = _ingest(f, col, min_mapq, buf)
        if s is not None and new and rid in buf:
            queue.append((start, rid))
    while queue:
        rid = queue.popleft()[1]
        if rid in buf:
            frag = _emit(buf.pop(rid))
            if frag is not None:
                yield frag


def load_fragments(path: str | Path, min_mapq: int = 30, flush_margin: int = FLUSH_MARGIN) -> Iterator[Fragment]:
    """Stream the whole file, collapsing mate pairs (windowed, memory-bounded)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:  # type: ignore[call-overload]
        header = fh.readline()
        if not header:
            return
        col = _column_index(header)

        def fields():
            for line in fh:
                if line and line[0] != "#":
                    yield line.rstrip("\n").split("\t")

        yield from _stream_dedup(fields(), col, min_mapq, flush_margin)


def load_fragments_chrom(path: str | Path, chrom: str, min_mapq: int = 30,
                         flush_margin: int = FLUSH_MARGIN) -> Iterator[Fragment]:
    """Stream one chromosome via tabix (pysam) with windowed mate-collapse.

    Reads only that chromosome's records — ~1/22 of the file — for sharded
    discovery. Falls back to a chrom-filtered full stream without pysam/index.
    """
    path = Path(path)
    if path.suffix == ".gz" and Path(str(path) + ".tbi").exists():
        try:
            import pysam
        except ImportError:
            pysam = None
        if pysam is not None:
            col = _read_header(path)
            tbx = pysam.TabixFile(str(path))
            try:
                if chrom in set(tbx.contigs):
                    yield from _stream_dedup(
                        (line.rstrip("\n").split("\t") for line in tbx.fetch(chrom)),
                        col, min_mapq, flush_margin)
            finally:
                tbx.close()
            return
    for frag in load_fragments(path, min_mapq=min_mapq, flush_margin=flush_margin):
        if frag.chrom == chrom:
            yield frag


def load_fragments_regions(
    path: str | Path,
    intervals: list[tuple[str, int, int]],
    min_mapq: int = 30,
) -> Iterator[Fragment]:
    """Read only fragments overlapping *intervals* (panel) via tabix.

    Uses pysam if the file is bgzipped+indexed; otherwise falls back to a full
    stream (intended for small/un-indexed local files). Fragments split across
    intervals are merged by `read_id` globally, so a molecule is counted once.
    """
    path = Path(path)
    col = _read_header(path)
    buf: dict[str, list] = {}

    pysam = None
    if path.suffix == ".gz" and Path(str(path) + ".tbi").exists():
        try:
            import pysam as _pysam
            pysam = _pysam
        except ImportError:
            pysam = None

    if pysam is not None:
        tbx = pysam.TabixFile(str(path))
        contigs = set(tbx.contigs)
        for chrom, start, end in intervals:
            if chrom not in contigs:
                continue
            for line in tbx.fetch(chrom, max(0, start), end):
                _ingest(line.rstrip("\n").split("\t"), col, min_mapq, buf)
        tbx.close()
    else:
        # Fallback: stream the file (local/un-indexed). Reads everything; scoring
        # filters to panel CpGs anyway.
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:  # type: ignore[call-overload]
            fh.readline()
            for line in fh:
                if not line or line[0] == "#":
                    continue
                _ingest(line.rstrip("\n").split("\t"), col, min_mapq, buf)

    for entry in buf.values():
        frag = _emit(entry)
        if frag is not None:
            yield frag


def panel_intervals(cpg_chrom: list[str], cpg_pos: np.ndarray, flank: int = 1000) -> list[tuple[str, int, int]]:
    """Merge panel CpG positions into tabix fetch intervals, padded by *flank*."""
    by_chrom: dict[str, list[int]] = defaultdict(list)
    for c, p in zip(cpg_chrom, cpg_pos.tolist()):
        by_chrom[c].append(int(p))
    intervals: list[tuple[str, int, int]] = []
    for chrom, positions in by_chrom.items():
        positions.sort()
        s = positions[0] - flank
        e = positions[0] + flank
        for p in positions[1:]:
            if p - flank <= e:
                e = p + flank
            else:
                intervals.append((chrom, max(0, s), e))
                s, e = p - flank, p + flank
        intervals.append((chrom, max(0, s), e))
    return intervals
