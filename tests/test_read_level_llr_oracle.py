"""End-to-end test for the read-level LLR oracle on synthetic PAT data.

Builds a tiny CpG index, one block, and synthetic tumour/healthy PAT files
with a deliberately strong per-CpG methylation difference, then checks that:

* reference profiles recover the planted tumour/healthy probabilities;
* held-out tumour reads score higher LLR than healthy reads (AUC well above
  chance), and the label-permutation null sits near 0.5;
* when tumour and healthy are drawn from the *same* distribution, the oracle
  reports AUC near 0.5 (no false signal).

Run directly: ``python tests/test_read_level_llr_oracle.py`` or via pytest.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import numpy as np

from tapestry.core.cpg_index import load_cpg_index
from tapestry.readlevel.llr_oracle import (
    compute_oracle_metrics,
    estimate_reference_profiles,
    load_regions_from_bed,
    score_reads,
)

N_CPG = 20
SPACING = 100
READ_LEN = 6  # CpGs per synthetic read


def _write_cpg_index(path: Path) -> dict:
    positions = [(i + 1) * SPACING for i in range(N_CPG)]
    with gzip.open(path, "wt") as fh:
        fh.write("#chrom\tposition\n")
        for p in positions:
            fh.write(f"chr1\t{p}\n")
    return {"chr1": np.array(positions, dtype=np.int64)}


def _write_bed(path: Path) -> None:
    # One block spanning all CpGs.
    path.write_text(f"chr1\t0\t{(N_CPG + 1) * SPACING}\n")


def _write_pat(path: Path, p_meth: float, n_reads: int, rng: np.random.Generator) -> None:
    """Write a PAT file of reads, each covering READ_LEN consecutive CpGs.

    ``p_meth`` is the probability of state 1 (encoded 'T' for the io.py loader).
    """
    lines = []
    for _ in range(n_reads):
        start_local = int(rng.integers(0, N_CPG - READ_LEN + 1))
        states = rng.random(READ_LEN) < p_meth
        pattern = "".join("T" if s else "C" for s in states)
        # PAT column 2 is the 1-based GLOBAL CpG index (single chrom -> offset 0).
        lines.append(f"chr1\t{start_local + 1}\t{pattern}\t1")
    path.write_text("\n".join(lines) + "\n")


def _build_samples(workdir: Path, p_tumour: float, p_healthy: float, seed: int):
    rng = np.random.default_rng(seed)
    samples = {"tumour": {"train": [], "test": []}, "healthy": {"train": [], "test": []}}
    for split in ("train", "test"):
        for k in range(2):
            tpath = workdir / f"tumour_{split}_{k}.pat"
            hpath = workdir / f"healthy_{split}_{k}.pat"
            _write_pat(tpath, p_tumour, 400, rng)
            _write_pat(hpath, p_healthy, 400, rng)
            samples["tumour"][split].append((str(tpath), f"tumour_{split}_{k}", "OAC_tissue"))
            samples["healthy"][split].append((str(hpath), f"healthy_{split}_{k}", "AB"))
    return samples


def _run(workdir: Path, p_tumour: float, p_healthy: float, seed: int = 0) -> dict:
    cpg_index = _write_cpg_index(workdir / "cpg.tsv.gz")
    cpg_index = load_cpg_index(workdir / "cpg.tsv.gz")
    _write_bed(workdir / "blocks.bed")
    regions = load_regions_from_bed(workdir / "blocks.bed", cpg_index, min_cpgs=4)
    assert len(regions) == 1
    assert len(regions[0].cpg_positions) == N_CPG

    samples = _build_samples(workdir, p_tumour, p_healthy, seed)
    profiles = estimate_reference_profiles(
        regions,
        tumour_pat_paths=[s[0] for s in samples["tumour"]["train"]],
        healthy_pat_paths=[s[0] for s in samples["healthy"]["train"]],
        cpg_index=cpg_index,
        min_cpgs_overlap=3,
    )
    tumour_scores = score_reads(
        regions, samples["tumour"]["test"], label=1, profiles=profiles,
        cpg_index=cpg_index, min_cpgs_overlap=3, min_ref_obs=3,
    )
    healthy_scores = score_reads(
        regions, samples["healthy"]["test"], label=0, profiles=profiles,
        cpg_index=cpg_index, min_cpgs_overlap=3, min_ref_obs=3,
    )
    metrics = compute_oracle_metrics(tumour_scores, healthy_scores, rng_seed=seed)
    metrics["_p_tumour_est"] = float(np.nanmean(profiles.p_tumour(regions[0].region_id)))
    metrics["_p_healthy_est"] = float(np.nanmean(profiles.p_healthy(regions[0].region_id)))
    return metrics


def test_separable_signal():
    with tempfile.TemporaryDirectory() as d:
        m = _run(Path(d), p_tumour=0.9, p_healthy=0.1, seed=1)
    # Profiles recover the planted probabilities.
    assert m["_p_tumour_est"] > 0.8, m
    assert m["_p_healthy_est"] < 0.2, m
    # Strong, real read-level separation; permutation null near chance.
    assert m["auc"] > 0.9, m
    assert abs(m["auc_permuted"] - 0.5) < 0.1, m
    assert m["auc_ncpg_matched"] > 0.85, m
    assert m["separable"] is True, m
    # Tumour reads score higher than healthy.
    assert m["mean_llr_tumour"] > m["mean_llr_healthy"], m


def test_no_false_signal_when_identical():
    with tempfile.TemporaryDirectory() as d:
        m = _run(Path(d), p_tumour=0.5, p_healthy=0.5, seed=2)
    # No real difference -> AUC near chance and not declared separable.
    assert abs(m["auc"] - 0.5) < 0.1, m
    assert m["separable"] is False, m


if __name__ == "__main__":
    test_separable_signal()
    test_no_false_signal_when_identical()
    print("read-level LLR oracle: all checks passed")
