"""The project's marker panel + per-CpG tumour/healthy profiles (genomic space).

Per selected CpG (keyed by genomic ``(chrom, pos)``), holds methylated/observed
counts for the tumour and healthy reference groups. Smoothed probabilities use a
Beta(prior, prior) pseudocount. Persists to ``cpgs.tsv`` + ``blocks.tsv`` +
``meta.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rltf.regions import Block


@dataclass
class ReferenceProfiles:
    blocks: list[Block]
    cpg_chrom: list[str]
    cpg_pos: np.ndarray          # int64, (n_cpg,)
    meth_tumour: np.ndarray      # float, (n_cpg,)
    total_tumour: np.ndarray
    meth_healthy: np.ndarray
    total_healthy: np.ndarray
    prior: float = 0.5
    n_tumour_samples: int = 0
    n_healthy_samples: int = 0
    index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {(c, int(p)): i for i, (c, p) in enumerate(zip(self.cpg_chrom, self.cpg_pos))}

    @property
    def p_tumour(self) -> np.ndarray:
        return (self.meth_tumour + self.prior) / (self.total_tumour + 2 * self.prior)

    @property
    def p_healthy(self) -> np.ndarray:
        return (self.meth_healthy + self.prior) / (self.total_healthy + 2 * self.prior)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pt, ph = self.p_tumour, self.p_healthy
        with open(out_dir / "cpgs.tsv", "w") as fh:
            fh.write("chrom\tpos\tmeth_tumour\ttotal_tumour\tmeth_healthy\ttotal_healthy\tp_tumour\tp_healthy\n")
            for i in range(len(self.cpg_pos)):
                fh.write(f"{self.cpg_chrom[i]}\t{int(self.cpg_pos[i])}\t{self.meth_tumour[i]:.0f}\t"
                         f"{self.total_tumour[i]:.0f}\t{self.meth_healthy[i]:.0f}\t{self.total_healthy[i]:.0f}\t"
                         f"{pt[i]:.5f}\t{ph[i]:.5f}\n")
        with open(out_dir / "blocks.tsv", "w") as fh:
            fh.write("block_id\tchrom\tstart_pos\tend_pos\tn_cpg\tcpg_pos\tscore\n")
            for b in self.blocks:
                cps = ",".join(str(int(p)) for p in b.cpg_pos)
                fh.write(f"{b.block_id}\t{b.chrom}\t{int(b.cpg_pos[0])}\t{int(b.cpg_pos[-1])}\t{b.n_cpg}\t{cps}\t{b.score:.6f}\n")
        (out_dir / "meta.json").write_text(json.dumps(
            {"prior": self.prior, "n_tumour_samples": self.n_tumour_samples,
             "n_healthy_samples": self.n_healthy_samples, "n_cpg": int(len(self.cpg_pos)),
             "n_blocks": len(self.blocks)}, indent=2))

    @classmethod
    def load(cls, in_dir: str | Path) -> "ReferenceProfiles":
        in_dir = Path(in_dir)
        chroms, pos, mt, tt, mh, th = [], [], [], [], [], []
        with open(in_dir / "cpgs.tsv") as fh:
            next(fh)
            for line in fh:
                f = line.rstrip("\n").split("\t")
                chroms.append(f[0]); pos.append(int(f[1]))
                mt.append(float(f[2])); tt.append(float(f[3])); mh.append(float(f[4])); th.append(float(f[5]))
        blocks = []
        with open(in_dir / "blocks.tsv") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            hi = {n: i for i, n in enumerate(header)}
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if "cpg_pos" in hi and f[hi["cpg_pos"]]:
                    cps = np.array([int(x) for x in f[hi["cpg_pos"]].split(",")], dtype=np.int64)
                else:  # backward-compat: only start/end stored
                    cps = np.array([int(f[hi["start_pos"]]), int(f[hi["end_pos"]])], dtype=np.int64)
                score = float(f[hi["score"]]) if "score" in hi and f[hi["score"]] else 0.0
                blocks.append(Block(f[hi["block_id"]], f[hi["chrom"]], cps, score=score))
        meta = json.loads((in_dir / "meta.json").read_text())
        return cls(blocks=blocks, cpg_chrom=chroms, cpg_pos=np.asarray(pos, dtype=np.int64),
                   meth_tumour=np.asarray(mt), total_tumour=np.asarray(tt),
                   meth_healthy=np.asarray(mh), total_healthy=np.asarray(th),
                   prior=meta["prior"], n_tumour_samples=meta["n_tumour_samples"],
                   n_healthy_samples=meta["n_healthy_samples"])
