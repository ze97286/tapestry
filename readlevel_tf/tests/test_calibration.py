"""Empirical null calibration removes the winner's-curse / estimation-noise drift.

All-null synthetic (tumour and healthy drawn identically, p=0.5) with low coverage,
so finite-sample noise produces spurious 'markers' and a biased analytic null. The
empirical calibration (fit on a disjoint healthy set) must center held-out healthy
reads at z≈0 with no n_cpg drift.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rltf.discovery import discover_panel
from rltf.llr import fit_calibration, score_fragments

NCPG = 200
CPGS = [1000 + 20 * i for i in range(NCPG)]
HEADER = ("#chr\tstart\tend\tread_id\tmapq\torientation\tinsert_size\tread_length\t"
          "flag\tnum_cpg\tnum_mod\tmod_cpgs\tunmod_cpgs\tsnp_cpgs\n")


def _gen(path: Path, n: int, rng: np.random.Generator, p: float = 0.5):
    lines = [HEADER]
    rid = 0
    for _ in range(n):
        w = int(rng.integers(4, 9))
        i0 = int(rng.integers(0, NCPG - w + 1))
        cs = CPGS[i0:i0 + w]
        start = cs[0] - 1
        mod, unmod = [], []
        for pos in cs:
            (mod if rng.random() < p else unmod).append(pos - start)
        rid += 1
        lines.append(f"chr1\t{start}\t{cs[-1] + 1}\tR{rid}\t50\t+\t150\t150\t99\t{w}\t{len(mod)}\t"
                     f"{','.join(map(str, mod))}\t{','.join(map(str, unmod))}\t\n")
    path.write_text("".join(lines))


def test_calibration_removes_null_drift():
    with tempfile.TemporaryDirectory() as dd:
        d = Path(dd)
        rng = np.random.default_rng(1)
        tum = [str(d / f"t{k}.bed") for k in range(4)]
        hea = [str(d / f"h{k}.bed") for k in range(4)]
        for p in tum + hea:
            _gen(Path(p), 500, rng)            # all null
        cal = [str(d / f"cal{k}.bed") for k in range(3)]
        test = [str(d / f"te{k}.bed") for k in range(3)]
        for p in cal + test:
            _gen(Path(p), 1500, rng)

        prof = discover_panel(tum, hea, window=5, top_n=40, min_total=8, min_effect=0.10, cross_fit=True)
        calib = fit_calibration(score_fragments(prof, [(p, f"c{k}", "CD") for k, p in enumerate(cal)],
                                                label=0, min_ref_obs=5))
        ts = score_fragments(prof, [(p, f"t{k}", "CD") for k, p in enumerate(test)], label=0, min_ref_obs=5)
        z = calib.z(ts.llr, ts.n_cpg)

        assert abs(float(np.mean(z))) < 0.1, float(np.mean(z))
        for lo, hi in [(1, 2), (2, 4), (4, 7)]:
            m = (ts.n_cpg >= lo) & (ts.n_cpg < hi)
            if m.sum() > 50:
                assert abs(float(np.mean(z[m]))) < 0.15, (lo, hi, float(np.mean(z[m])))
        # calibrated healthy z is essentially uncorrelated with n_cpg
        assert abs(np.corrcoef(z, ts.n_cpg)[0, 1]) < 0.1


if __name__ == "__main__":
    test_calibration_removes_null_drift()
    print("rltf calibration: held-out healthy z centred with no n_cpg drift")
