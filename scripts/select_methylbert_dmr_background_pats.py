#!/usr/bin/env python3
"""Select healthy-control PATs for MethylBERT DMR discovery.

This is the region-selection control, not the read-classifier training control:
it decides which healthy cfDNA samples enter the DSS tumour-vs-background contrast.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def read_paths(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                rows.append(line.split()[0])
    return rows


def cohort_of(path: str) -> str:
    name = Path(path).name
    if "_Ctrl_plasma" in name:
        return "AB_plasma"
    if name.startswith(("GI", "SCAN")):
        return "CD_plasma"
    return "other"


def parse_cohorts(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-pat-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--background-cohorts",
        default="AB_plasma,CD_plasma",
        help="Comma-separated cohorts to keep. Use AB_plasma for AB-only DMR discovery.",
    )
    parser.add_argument(
        "--max-per-cohort",
        type=int,
        default=0,
        help="Maximum samples per kept cohort; 0 keeps all samples in each cohort.",
    )
    parser.add_argument("--seed", type=int, default=950410)
    args = parser.parse_args()

    paths = read_paths(Path(args.normal_pat_list))
    keep_cohorts = parse_cohorts(args.background_cohorts)
    by_cohort: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_cohort[cohort_of(path)].append(path)

    rng = random.Random(args.seed)
    selected: list[str] = []
    for cohort in sorted(keep_cohorts):
        rows = list(by_cohort.get(cohort, []))
        if args.max_per_cohort > 0 and len(rows) > args.max_per_cohort:
            rows = sorted(rng.sample(rows, args.max_per_cohort))
        selected.extend(rows)

    if not selected:
        observed = ", ".join(f"{k}={len(v)}" for k, v in sorted(by_cohort.items()))
        raise SystemExit(
            f"no background PATs selected for cohorts {sorted(keep_cohorts)}; observed {observed}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n")

    observed = " ".join(f"{k}={len(v)}" for k, v in sorted(by_cohort.items()))
    kept = " ".join(
        f"{cohort}={sum(cohort_of(path) == cohort for path in selected)}"
        for cohort in sorted(keep_cohorts)
    )
    print(f"background cohorts observed: {observed}")
    print(f"background cohorts kept: {kept}")
    print(f"wrote {len(selected)} background PATs to {output}")


if __name__ == "__main__":
    main()
