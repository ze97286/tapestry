"""End-to-end test of the standalone rltf pipeline on synthetic PATs.

Synthetic genome in global CpG-index space: chr1 CpGs 1..200, with three
tile-aligned 'true marker' windows where tumour is methylated (p=0.95) and
healthy is unmethylated (p=0.05); elsewhere both p=0.5 (no signal). Checks:

* discovery recovers exactly the planted marker blocks (and rejects background);
* the oracle separates held-out tumour vs healthy reads (AUC high, permutation
  null ~0.5);
* the tabular head (sklearn backend, offline) separates cancer vs healthy and
  recovers a positive tumour-fraction correlation under leave-one-cohort-out CV.

Run: ``PYTHONPATH=readlevel_tf python readlevel_tf/tests/test_pipeline.py`` or pytest.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rltf.discovery import discover_panel
from rltf.features import build_feature_matrix, feature_names
from rltf.head import leave_one_group_out_cv
from rltf.llr import oracle_metrics, score_reads

CPG_MIN, CPG_MAX = 1, 200
READ_LEN = 6
MARKERS = set(range(11, 16)) | set(range(51, 56)) | set(range(101, 106))


def _p_tumour(i: int) -> float:
    return 0.95 if i in MARKERS else 0.5


def _p_healthy(i: int) -> float:
    return 0.05 if i in MARKERS else 0.5


def _write_pat(path: Path, p_func, n_reads: int, rng: np.random.Generator, tf: float = 1.0, p_other=None) -> None:
    """Write reads (bisulfite convention: C=methylated). For mixtures, each read
    is tumour-origin with prob *tf* (uses p_func) else healthy-origin (p_other)."""
    lines = []
    for _ in range(n_reads):
        start = int(rng.integers(CPG_MIN, CPG_MAX - READ_LEN + 2))
        src = p_func if (p_other is None or rng.random() < tf) else p_other
        chars = []
        for j in range(READ_LEN):
            meth = rng.random() < src(start + j)
            chars.append("C" if meth else "T")
        lines.append(f"chr1\t{start}\t{''.join(chars)}\t1")
    path.write_text("\n".join(lines) + "\n")


def _run(d: Path):
    rng = np.random.default_rng(11)

    # Reference: tumour tissue + healthy cfDNA (disjoint from query), bisulfite.
    ref_tumour, ref_healthy = [], []
    for k in range(3):
        tp, hp = d / f"ref_t_{k}.pat", d / f"ref_h_{k}.pat"
        _write_pat(tp, _p_tumour, 1500, rng)
        _write_pat(hp, _p_healthy, 1500, rng)
        ref_tumour.append((str(tp), "bisulfite"))
        ref_healthy.append((str(hp), "bisulfite"))

    profiles = discover_panel(
        ref_tumour, ref_healthy, window=5, top_n=10, min_total=10, min_effect=0.3,
    )

    # Held-out reads for the oracle.
    htp, hhp = d / "ho_t.pat", d / "ho_h.pat"
    _write_pat(htp, _p_tumour, 800, rng)
    _write_pat(hhp, _p_healthy, 800, rng)
    t_scores = score_reads(profiles, [(str(htp), "bisulfite", "ho_t", "OAC")], label=1, min_ref_obs=5)
    h_scores = score_reads(profiles, [(str(hhp), "bisulfite", "ho_h", "AB")], label=0, min_ref_obs=5)
    oracle = oracle_metrics(t_scores, h_scores)

    # Query samples for the detector: 2 cohorts, healthy + cancer mixtures.
    query, labels, tfs, cohorts = [], [], [], []
    tf_grid = [0.05, 0.1, 0.2, 0.35, 0.5, 0.8]
    for cohort in ("AB", "CD"):
        for j in range(6):
            p = d / f"q_{cohort}_h_{j}.pat"
            _write_pat(p, _p_healthy, 600, rng)
            query.append((str(p), "bisulfite", f"{cohort}_h_{j}", cohort))
            labels.append(0); tfs.append(0.0); cohorts.append(cohort)
        for j, tf in enumerate(tf_grid):
            p = d / f"q_{cohort}_c_{j}.pat"
            _write_pat(p, _p_tumour, 600, rng, tf=tf, p_other=_p_healthy)
            query.append((str(p), "bisulfite", f"{cohort}_c_{j}", cohort))
            labels.append(1); tfs.append(tf); cohorts.append(cohort)

    feat = build_feature_matrix(profiles, query, min_ref_obs=5)
    X = feat[feature_names()].to_numpy(float)
    groups = np.array(cohorts)
    cls = leave_one_group_out_cv(X, np.array(labels), groups, task="classify", backend="sklearn")
    reg = leave_one_group_out_cv(X, np.array(tfs, float), groups, task="regress", backend="sklearn")
    return profiles, oracle, cls, reg, feat


def test_pipeline():
    with tempfile.TemporaryDirectory() as dd:
        profiles, oracle, cls, reg, feat = _run(Path(dd))

    # Discovery recovers exactly the three planted marker tiles.
    starts = sorted(b.start_cpg for b in profiles.blocks)
    assert starts == [11, 51, 101], starts

    # Oracle separates reads; permutation null ~0.5.
    assert oracle["auc"] > 0.9, oracle
    assert abs(oracle["auc_permuted"] - 0.5) < 0.1, oracle
    assert oracle["separable"] is True, oracle

    # Detector separates cancer/healthy and tracks tumour fraction.
    assert cls.metrics["auc"] > 0.85, cls.metrics
    assert reg.metrics["pearson_r"] > 0.6, reg.metrics

    # Cancer samples carry more tumour-pattern reads than controls.
    cancer = feat.index.str.contains("_c_")
    assert feat.loc[cancer, "frac_reads_llr_gt_0"].mean() > feat.loc[~cancer, "frac_reads_llr_gt_0"].mean()


if __name__ == "__main__":
    test_pipeline()
    print("rltf standalone pipeline: all checks passed")
