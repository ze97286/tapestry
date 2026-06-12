"""Sharded discovery (per-chromosome partials + merge) must equal genome-wide."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rltf.discovery import discover_panel, merge_partials

CHROMS = ["chr1", "chr2"]
CPGS = [1000 + 20 * i for i in range(40)]
MARK_IDX = set(range(5, 10)) | set(range(25, 30))   # 2 marker windows per chrom
MARK_POS = {CPGS[i] for i in MARK_IDX}
HEADER = ("#chr\tstart\tend\tread_id\tmapq\torientation\tinsert_size\tread_length\t"
          "flag\tnum_cpg\tnum_mod\tmod_cpgs\tunmod_cpgs\tsnp_cpgs\n")


def _p(origin, pos):
    if pos in MARK_POS:
        return 0.95 if origin == "tumour" else 0.05
    return 0.5


def _write(path: Path, origin: str, rng: np.random.Generator, n_per_chrom=1500):
    lines = [HEADER]
    rid = 0
    for chrom in CHROMS:
        for _ in range(n_per_chrom):
            L = int(rng.integers(2, 13))
            i0 = int(rng.integers(0, 40 - L + 1))
            cs = CPGS[i0:i0 + L]
            start = cs[0] - 1
            mod, unmod = [], []
            for pos in cs:
                (mod if rng.random() < _p(origin, pos) else unmod).append(pos - start)
            rid += 1
            lines.append(f"{chrom}\t{start}\t{cs[-1] + 1}\tR{rid}\t50\t+\t150\t150\t99\t{L}\t{len(mod)}\t"
                         f"{','.join(map(str, mod))}\t{','.join(map(str, unmod))}\t\n")
    path.write_text("".join(lines))


def test_sharded_equals_genome_wide():
    with tempfile.TemporaryDirectory() as dd:
        d = Path(dd)
        rng = np.random.default_rng(0)
        tum = [str(d / f"t{k}.per-read.bed") for k in range(2)]
        hea = [str(d / f"h{k}.per-read.bed") for k in range(2)]
        for p in tum:
            _write(Path(p), "tumour", rng)
        for p in hea:
            _write(Path(p), "healthy", rng)

        kw = dict(window=5, min_total=10, min_effect=0.3)
        full = discover_panel(tum, hea, top_n=8, **kw)

        parts = []
        for chrom in CHROMS:
            prof = discover_panel(tum, hea, top_n=None, chroms={chrom}, **kw)
            pdir = d / "partials" / chrom
            prof.save(pdir)
            parts.append(str(pdir))
        merged = merge_partials(parts, top_n=8)

        full_cpgs = {(c, int(p)) for c, p in zip(full.cpg_chrom, full.cpg_pos)}
        merged_cpgs = {(c, int(p)) for c, p in zip(merged.cpg_chrom, merged.cpg_pos)}
        assert full_cpgs == merged_cpgs, (len(full_cpgs), len(merged_cpgs))
        assert {b.block_id for b in full.blocks} == {b.block_id for b in merged.blocks}
        # both find the 4 marker windows (2 per chromosome), on both chromosomes
        assert {b.chrom for b in merged.blocks} == {"chr1", "chr2"}
        assert len(merged.blocks) == 4


if __name__ == "__main__":
    test_sharded_equals_genome_wide()
    print("rltf sharding: per-chrom partials + merge == genome-wide")
