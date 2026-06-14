#!/usr/bin/env python3
"""Re-fit the per-k null calibration on saved oracle scores — OFFLINE, no re-discovery.

Reads ``calibration_scores.tsv.gz`` (train-healthy raw LLR) and
``per_fragment_scores.tsv.gz`` (test raw LLR) from a finished oracle run and
recomputes the gate metrics under a new ``--min-per-k``, in seconds. Lets you tune
the calibration without re-running the genome-wide discovery. Requires an oracle
run produced by the current code (which saves raw ``llr``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rltf.llr import FragmentScores, NullCalibration, oracle_metrics


def _fs(sub: pd.DataFrame) -> FragmentScores:
    return FragmentScores(
        llr=sub["llr"].to_numpy(float), n_cpg=sub["n_cpg"].to_numpy(int),
        read_length=sub["read_length"].to_numpy(int) if "read_length" in sub else sub["n_cpg"].to_numpy(int),
        label=sub["label"].to_numpy(int),
        sample_id=sub["sample_id"].to_numpy(object), cohort=sub["cohort"].to_numpy(object))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle-dir", required=True, type=Path)
    ap.add_argument("--min-per-k", type=int, default=100)
    args = ap.parse_args()

    cal = pd.read_csv(args.oracle_dir / "calibration_scores.tsv.gz", sep="\t")
    pf = pd.read_csv(args.oracle_dir / "per_fragment_scores.tsv.gz", sep="\t")
    if "llr" not in pf.columns:
        raise SystemExit("per_fragment_scores.tsv.gz has no raw 'llr' column — re-run the oracle with current code.")

    calib = NullCalibration.fit(cal["llr"].to_numpy(float), cal["n_cpg"].to_numpy(int), min_per_k=args.min_per_k)
    metrics = oracle_metrics(_fs(pf[pf["label"] == 1]), _fs(pf[pf["label"] == 0]), calib)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
