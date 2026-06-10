#!/usr/bin/env python3
"""Run the read-level likelihood-ratio oracle.

Decides whether tumour-vs-healthy methylation is separable at the single-read
level on a given block panel, with explicit length/cohort controls. See
``docs/read_level_detection_direction.md``.

Inputs
------
* ``--regions-bed``  block panel (BED; first three columns chrom/start/end).
* ``--cpg-index``    genome CpG index (``CpG.bed.gz`` / tapestry tsv.gz).
* ``--samples-tsv``  one row per PAT file, tab-separated, header columns:
      ``sample_id``  ``group``  ``cohort``  ``file_path``
  where ``group`` is ``tumour`` or ``healthy`` and ``cohort`` is e.g. ``AB`` /
  ``CD`` / ``OAC_tissue`` (used for leave-cohort-out and stratified reporting).

Reference profiles are estimated on the **train** split only; reads from the
held-out **test** split are scored. Outputs go to ``--out-dir``:
``summary.json`` (the verdict + AUCs), ``split.json``, and an optional
down-sampled ``per_read_scores.tsv.gz``.

This is a CPU/streaming job; run it from ``slurm/oracle_read_level_llr.sh``,
not as a bare ``python`` invocation on a login node.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from tapestry.core.cpg_index import load_cpg_index
from tapestry.readlevel.llr_oracle import (
    compute_oracle_metrics,
    estimate_reference_profiles,
    load_regions_from_bed,
    score_reads,
)

logger = logging.getLogger("read_level_llr_oracle")


def _read_samples_tsv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for required in ("sample_id", "group", "cohort", "file_path"):
            if required not in col:
                raise ValueError(f"samples TSV missing required column '{required}'")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            f = line.split("\t")
            group = f[col["group"]].strip().lower()
            if group not in ("tumour", "tumor", "healthy"):
                raise ValueError(f"group must be tumour/healthy, got '{group}'")
            rows.append(
                {
                    "sample_id": f[col["sample_id"]].strip(),
                    "group": "tumour" if group in ("tumour", "tumor") else "healthy",
                    "cohort": f[col["cohort"]].strip(),
                    "file_path": f[col["file_path"]].strip(),
                }
            )
    return rows


def _assign_split(
    samples: list[dict],
    test_fraction: float,
    holdout_cohort: str | None,
    holdout_samples: set[str],
    seed: int,
) -> dict[str, str]:
    """Assign each sample id to 'train' or 'test'.

    A sample is forced to test if its id is in *holdout_samples* or (for healthy
    samples) its cohort equals *holdout_cohort*. Remaining samples are split
    per group at random with probability *test_fraction* assigned to test.
    """
    rng = np.random.default_rng(seed)
    split: dict[str, str] = {}
    by_group: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        if s["sample_id"] in holdout_samples or (
            holdout_cohort is not None and s["group"] == "healthy" and s["cohort"] == holdout_cohort
        ):
            split[s["sample_id"]] = "test"
        else:
            by_group[s["group"]].append(s)

    for group, members in by_group.items():
        members = sorted(members, key=lambda s: s["sample_id"])
        idx = rng.permutation(len(members))
        n_test = max(1, int(round(test_fraction * len(members)))) if len(members) > 1 else 0
        test_ids = {members[i]["sample_id"] for i in idx[:n_test]}
        for s in members:
            split[s["sample_id"]] = "test" if s["sample_id"] in test_ids else "train"
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions-bed", required=True, type=Path)
    ap.add_argument("--cpg-index", required=True, type=Path)
    ap.add_argument("--samples-tsv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--panel-name", default="panel", help="label for this run, recorded in summary.json")
    ap.add_argument("--region-min-cpgs", type=int, default=4)
    ap.add_argument("--min-cpgs-overlap", type=int, default=3,
                    help="min CpGs a read must share with a block to be used")
    ap.add_argument("--min-ref-obs", type=int, default=5,
                    help="min reference reads per CpG (both groups) for it to contribute to an LLR")
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--prob-clip", type=float, default=1e-3)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--holdout-cohort", default=None,
                    help="force healthy samples of this cohort (e.g. CD) into the test split")
    ap.add_argument("--holdout-samples", default="",
                    help="comma-separated sample ids forced into the test split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-per-read", type=int, default=200000,
                    help="down-sample this many scored reads to per_read_scores.tsv.gz (0 = none)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CpG index from %s", args.cpg_index)
    cpg_index = load_cpg_index(args.cpg_index)
    regions = load_regions_from_bed(args.regions_bed, cpg_index, min_cpgs=args.region_min_cpgs)
    if not regions:
        raise SystemExit("No regions with enough CpGs; check --regions-bed and --cpg-index.")

    samples = _read_samples_tsv(args.samples_tsv)
    holdout_samples = {s for s in args.holdout_samples.split(",") if s}
    split = _assign_split(samples, args.test_fraction, args.holdout_cohort, holdout_samples, args.seed)

    train = {"tumour": [], "healthy": []}
    test = {"tumour": [], "healthy": []}
    for s in samples:
        (train if split[s["sample_id"]] == "train" else test)[s["group"]].append(s)

    for grp in ("tumour", "healthy"):
        if not train[grp]:
            raise SystemExit(f"No {grp} samples in the train split — cannot estimate profiles.")
        if not test[grp]:
            raise SystemExit(f"No {grp} samples in the test split — cannot evaluate AUC.")

    logger.info(
        "Split: train tumour=%d healthy=%d | test tumour=%d healthy=%d",
        len(train["tumour"]), len(train["healthy"]), len(test["tumour"]), len(test["healthy"]),
    )

    profiles = estimate_reference_profiles(
        regions,
        tumour_pat_paths=[s["file_path"] for s in train["tumour"]],
        healthy_pat_paths=[s["file_path"] for s in train["healthy"]],
        cpg_index=cpg_index,
        prior=args.prior,
        min_cpgs_overlap=args.min_cpgs_overlap,
    )

    tumour_scores = score_reads(
        regions,
        [(s["file_path"], s["sample_id"], s["cohort"]) for s in test["tumour"]],
        label=1,
        profiles=profiles,
        cpg_index=cpg_index,
        min_cpgs_overlap=args.min_cpgs_overlap,
        min_ref_obs=args.min_ref_obs,
        prob_clip=args.prob_clip,
    )
    healthy_scores = score_reads(
        regions,
        [(s["file_path"], s["sample_id"], s["cohort"]) for s in test["healthy"]],
        label=0,
        profiles=profiles,
        cpg_index=cpg_index,
        min_cpgs_overlap=args.min_cpgs_overlap,
        min_ref_obs=args.min_ref_obs,
        prob_clip=args.prob_clip,
    )

    metrics = compute_oracle_metrics(tumour_scores, healthy_scores, rng_seed=args.seed)

    # Per-cohort healthy AUC against the tumour reads — a leave-cohort-out style
    # specificity read-out (does one healthy cohort score tumour-like?).
    from tapestry.readlevel.llr_oracle import _merge_scores, _weighted_auc  # noqa: E402

    per_cohort = {}
    for cohort in sorted(set(healthy_scores.cohort.tolist())):
        mask = healthy_scores.cohort == cohort
        sub = _merge_scores([
            tumour_scores,
            type(healthy_scores)(
                llr=healthy_scores.llr[mask], n_cpg_scored=healthy_scores.n_cpg_scored[mask],
                weight=healthy_scores.weight[mask], label=healthy_scores.label[mask],
                sample_id=healthy_scores.sample_id[mask], cohort=healthy_scores.cohort[mask],
            ),
        ])
        fin = np.isfinite(sub.llr)
        per_cohort[cohort] = {
            "auc_vs_tumour": _weighted_auc(sub.label[fin].astype(int), sub.llr[fin], sub.weight[fin].astype(float)),
            "n_healthy_reads": int(healthy_scores.weight[mask].sum()),
            "mean_llr": float(np.average(healthy_scores.llr[mask][np.isfinite(healthy_scores.llr[mask])],
                                         weights=healthy_scores.weight[mask][np.isfinite(healthy_scores.llr[mask])]))
            if np.isfinite(healthy_scores.llr[mask]).any() else float("nan"),
        }

    summary = {
        "panel_name": args.panel_name,
        "regions_bed": str(args.regions_bed),
        "n_regions": len(regions),
        "n_tumour_train": len(train["tumour"]),
        "n_healthy_train": len(train["healthy"]),
        "n_tumour_test": len(test["tumour"]),
        "n_healthy_test": len(test["healthy"]),
        "params": {
            "region_min_cpgs": args.region_min_cpgs,
            "min_cpgs_overlap": args.min_cpgs_overlap,
            "min_ref_obs": args.min_ref_obs,
            "prior": args.prior,
            "prob_clip": args.prob_clip,
            "test_fraction": args.test_fraction,
            "holdout_cohort": args.holdout_cohort,
            "seed": args.seed,
        },
        "metrics": metrics,
        "per_healthy_cohort": per_cohort,
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "split.json").write_text(
        json.dumps(
            {
                "train": {g: [s["sample_id"] for s in train[g]] for g in train},
                "test": {g: [s["sample_id"] for s in test[g]] for g in test},
            },
            indent=2,
        )
    )

    if args.write_per_read > 0:
        merged = _merge_scores([tumour_scores, healthy_scores])
        n = len(merged.llr)
        rng = np.random.default_rng(args.seed)
        sel = np.arange(n) if n <= args.write_per_read else rng.choice(n, args.write_per_read, replace=False)
        import pandas as pd

        pd.DataFrame(
            {
                "llr": merged.llr[sel],
                "n_cpg_scored": merged.n_cpg_scored[sel],
                "weight": merged.weight[sel],
                "label": merged.label[sel],
                "sample_id": merged.sample_id[sel],
                "cohort": merged.cohort[sel],
            }
        ).to_csv(args.out_dir / "per_read_scores.tsv.gz", sep="\t", index=False)

    logger.info("AUC=%.4f permuted=%.4f ncpg_matched=%.4f | %s",
                metrics["auc"], metrics["auc_permuted"], metrics["auc_ncpg_matched"], metrics["verdict"])
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
