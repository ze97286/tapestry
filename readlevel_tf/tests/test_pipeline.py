"""End-to-end test of the per-read-call rltf core on synthetic data.

Synthetic genome: chr1 with 60 CpGs every 20 bp; three tile-aligned 5-CpG marker
windows where tumour is methylated (0.95) and healthy unmethylated (0.05),
elsewhere both 0.5. Reads cover a *random number* of CpGs (2-14) so length/n_cpg
varies — which lets us prove the key property: the per-fragment z is
**null-invariant to n_cpg** (healthy z ~ 0 regardless of how many CpGs a read
covers), so read length cannot leak.

Checks: discovery recovers the planted markers; the oracle separates tumour vs
healthy fragments; healthy z is uncorrelated with n_cpg and read length; and the
tabular head separates cancer from healthy under leave-one-cohort-out.

Run: ``PYTHONPATH=readlevel_tf python3 readlevel_tf/tests/test_pipeline.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rltf.discovery import discover_panel
from rltf.features import build_feature_matrix, feature_names
from rltf.head import leave_one_group_out_cv
from rltf.llr import merge_scores, oracle_metrics, score_fragments

CPGS = np.array([1000 + 20 * i for i in range(60)], dtype=np.int64)
MARK_IDX = set(range(5, 10)) | set(range(25, 30)) | set(range(45, 50))
MARK_POS = {int(CPGS[i]) for i in MARK_IDX}

HEADER = ("#chr\tstart\tend\tread_id\tmapq\torientation\tinsert_size\tread_length\t"
          "flag\tnum_cpg\tnum_mod\tmod_cpgs\tunmod_cpgs\tsnp_cpgs\n")


def _p(origin: str, pos: int) -> float:
    if pos in MARK_POS:
        return 0.95 if origin == "tumour" else 0.05
    return 0.5


def _write_perread(path: Path, n_reads: int, origin: str, rng: np.random.Generator, tf: float = 1.0):
    """origin 'tumour'/'healthy', or mixture: each read is tumour w.p. tf else healthy."""
    lines = [HEADER]
    rid = 0
    for _ in range(n_reads):
        L = int(rng.integers(2, 15))            # 2..14 CpGs -> varying length
        i0 = int(rng.integers(0, 60 - L + 1))
        cpgs = CPGS[i0:i0 + L]
        o = origin if (origin != "mix" or rng.random() < tf) else "healthy"
        if origin == "mix":
            o = "tumour" if rng.random() < tf else "healthy"
        start = int(cpgs[0]) - 1
        end = int(cpgs[-1]) + 1
        mod, unmod = [], []
        for pos in cpgs:
            off = int(pos) - start
            (mod if rng.random() < _p(o, int(pos)) else unmod).append(off)
        rid += 1
        read_length = end - start              # correlated with n_cpg by construction
        lines.append(f"chr1\t{start}\t{end}\tR{rid}\t50\t+\t{read_length}\t{read_length}\t99\t"
                     f"{L}\t{len(mod)}\t{','.join(map(str, mod))}\t{','.join(map(str, unmod))}\t\n")
    path.write_text("".join(lines))


def _run(d: Path):
    rng = np.random.default_rng(0)
    ref_t, ref_h = [], []
    for k in range(3):
        tp, hp = d / f"ref_t_{k}.bed", d / f"ref_h_{k}.bed"
        _write_perread(tp, 4000, "tumour", rng)
        _write_perread(hp, 4000, "healthy", rng)
        ref_t.append(str(tp)); ref_h.append(str(hp))

    profiles = discover_panel(ref_t, ref_h, window=5, top_n=10, min_total=10, min_effect=0.3)

    ht, hh = d / "ho_t.bed", d / "ho_h.bed"
    _write_perread(ht, 3000, "tumour", rng)
    _write_perread(hh, 3000, "healthy", rng)
    t_sc = score_fragments(profiles, [(str(ht), "ho_t", "OAC")], label=1, min_ref_obs=5)
    h_sc = score_fragments(profiles, [(str(hh), "ho_h", "AB")], label=0, min_ref_obs=5)
    oracle = oracle_metrics(t_sc, h_sc)

    query, labels, cohorts = [], [], []
    for cohort in ("AB", "CD"):
        for j in range(6):
            p = d / f"q_{cohort}_h_{j}.bed"
            _write_perread(p, 1500, "healthy", rng)
            query.append((str(p), f"{cohort}_h_{j}", cohort)); labels.append(0); cohorts.append(cohort)
        for j, tf in enumerate([0.1, 0.2, 0.3, 0.5, 0.7, 0.9]):
            p = d / f"q_{cohort}_c_{j}.bed"
            _write_perread(p, 1500, "mix", rng, tf=tf)
            query.append((str(p), f"{cohort}_c_{j}", cohort)); labels.append(1); cohorts.append(cohort)

    feat = build_feature_matrix(profiles, query, min_ref_obs=5)
    X = feat[feature_names()].to_numpy(float)
    cls = leave_one_group_out_cv(X, np.array(labels), np.array(cohorts), task="classify", backend="sklearn")
    return profiles, oracle, h_sc, cls, feat


def test_pipeline():
    with tempfile.TemporaryDirectory() as dd:
        profiles, oracle, h_sc, cls, feat = _run(Path(dd))

    # Discovery recovers exactly the three planted marker windows.
    starts = sorted(int(b.cpg_pos[0]) for b in profiles.blocks)
    assert starts == [int(CPGS[5]), int(CPGS[25]), int(CPGS[45])], starts

    # Oracle separates; permutation ~0.5.
    assert oracle["auc"] > 0.9, oracle
    assert abs(oracle["auc_permuted"] - 0.5) < 0.1, oracle
    assert oracle["separable"] is True

    # THE KEY PROPERTY: healthy z is null-invariant to n_cpg and read length.
    assert abs(oracle["mean_z_healthy"]) < 0.15, oracle["mean_z_healthy"]
    assert abs(np.corrcoef(h_sc.z, h_sc.n_cpg)[0, 1]) < 0.15, "healthy z correlates with n_cpg!"
    assert abs(np.corrcoef(h_sc.z, h_sc.read_length)[0, 1]) < 0.15, "healthy z correlates with read length!"
    # null mean z ~0 in every CpG-count bucket
    for bucket, mz in oracle["null_mean_z_by_n_cpg"].items():
        assert abs(mz) < 0.35, (bucket, mz)

    # Detector separates cancer vs healthy under leave-one-cohort-out.
    assert cls.metrics["auc"] > 0.85, cls.metrics


if __name__ == "__main__":
    test_pipeline()
    print("rltf per-read core: all checks passed (incl. z null-invariant to length/n_cpg)")
