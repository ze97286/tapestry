"""The project's own marker panel + per-CpG tumour/healthy profiles.

A :class:`ReferenceProfiles` bundles the discovered blocks (fixed window ``W``)
with per-block, per-CpG methylated/observed counts for the tumour and healthy
reference groups. Smoothed probabilities use a Beta(``prior``, ``prior``)
pseudocount. It persists to a ``blocks.tsv`` + ``profiles.npz`` pair so the
panel is reusable across the oracle and detector without re-discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rltf.regions import Block, BlockIndex


@dataclass
class ReferenceProfiles:
    blocks: list[Block]
    meth_tumour: np.ndarray    # (n_blocks, W) float
    total_tumour: np.ndarray   # (n_blocks, W) float
    meth_healthy: np.ndarray   # (n_blocks, W) float
    total_healthy: np.ndarray  # (n_blocks, W) float
    window: int
    prior: float = 0.5
    n_tumour_samples: int = 0
    n_healthy_samples: int = 0

    def __post_init__(self) -> None:
        self.block_index = BlockIndex(self.blocks)
        self._row = {b.block_id: i for i, b in enumerate(self.blocks)}

    def row_of(self, block_id: str) -> int:
        return self._row[block_id]

    @property
    def p_tumour(self) -> np.ndarray:
        return (self.meth_tumour + self.prior) / (self.total_tumour + 2.0 * self.prior)

    @property
    def p_healthy(self) -> np.ndarray:
        return (self.meth_healthy + self.prior) / (self.total_healthy + 2.0 * self.prior)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p_t, p_h = self.p_tumour, self.p_healthy
        with open(out_dir / "blocks.tsv", "w") as fh:
            fh.write("block_id\tchrom\tstart_cpg\tend_cpg\tn_cpg\t"
                     "mean_p_tumour\tmean_p_healthy\teffect\tmin_total_tumour\tmin_total_healthy\n")
            for i, b in enumerate(self.blocks):
                fh.write(
                    f"{b.block_id}\t{b.chrom}\t{b.start_cpg}\t{b.end_cpg}\t{b.n_cpg}\t"
                    f"{p_t[i].mean():.5f}\t{p_h[i].mean():.5f}\t{abs(p_t[i].mean() - p_h[i].mean()):.5f}\t"
                    f"{self.total_tumour[i].min():.0f}\t{self.total_healthy[i].min():.0f}\n"
                )
        np.savez_compressed(
            out_dir / "profiles.npz",
            meth_tumour=self.meth_tumour, total_tumour=self.total_tumour,
            meth_healthy=self.meth_healthy, total_healthy=self.total_healthy,
            window=self.window, prior=self.prior,
            n_tumour_samples=self.n_tumour_samples, n_healthy_samples=self.n_healthy_samples,
        )

    @classmethod
    def load(cls, in_dir: str | Path) -> "ReferenceProfiles":
        in_dir = Path(in_dir)
        blocks: list[Block] = []
        with open(in_dir / "blocks.tsv") as fh:
            next(fh)  # header
            for line in fh:
                f = line.rstrip("\n").split("\t")
                blocks.append(Block(f[0], f[1], int(f[2]), int(f[3])))
        npz = np.load(in_dir / "profiles.npz")
        return cls(
            blocks=blocks,
            meth_tumour=npz["meth_tumour"], total_tumour=npz["total_tumour"],
            meth_healthy=npz["meth_healthy"], total_healthy=npz["total_healthy"],
            window=int(npz["window"]), prior=float(npz["prior"]),
            n_tumour_samples=int(npz["n_tumour_samples"]), n_healthy_samples=int(npz["n_healthy_samples"]),
        )
