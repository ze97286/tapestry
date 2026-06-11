#!/usr/bin/env python3
"""Step 2 — read-level separability oracle (the gate).

Splits the reference samples (train/test), discovers a panel on the **train**
split only, and scores held-out **test** reads by per-read LLR. Reports AUC with
length/permutation/CpG-count controls. If reads are not separable here, the
detector has no signal to learn — fix discovery before proceeding.

Discovery on train-only keeps the gate honest (no block-selection leakage).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from rltf.discovery import discover_panel
from rltf.llr import ReadScores, merge_scores, oracle_metrics, score_reads
from rltf.manifest import read_manifest

logger = logging.getLogger("run_oracle")


def _split(ref: list[dict], test_fraction: float, holdout_cohort: str | None,
           holdout_samples: set[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    split: dict[str, str] = {}
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in ref:
        if r["sample_id"] in holdout_samples or (
            holdout_cohort and r["group"] == "healthy" and r["cohort"] == holdout_cohort
        ):
            split[r["sample_id"]] = "test"
        else:
            by_group[r["group"]].append(r)
    for members in by_group.values():
        members = sorted(members, key=lambda s: s["sample_id"])
        idx = rng.permutation(len(members))
        n_test = max(1, int(round(test_fraction * len(members)))) if len(members) > 1 else 0
        test_ids = {members[i]["sample_id"] for i in idx[:n_test]}
        for s in members:
            split[s["sample_id"]] = "test" if s["sample_id"] in test_ids else "train"
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--min-total", type=float, default=10.0)
    ap.add_argument("--min-effect", type=float, default=0.3)
    ap.add_argument("--direction", default="any", choices=["any", "hypo", "hyper"])
    ap.add_argument("--min-ref-obs", type=float, default=5.0)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--holdout-cohort", default=None)
    ap.add_argument("--holdout-samples", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--write-per-read", type=int, default=200000,
                    help="down-sample this many scored reads to per_read_scores.tsv.gz (0 = none)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = [r for r in read_manifest(args.manifest) if r["role"] == "reference"]
    holdout = {s for s in args.holdout_samples.split(",") if s}
    split = _split(ref, args.test_fraction, args.holdout_cohort, holdout, args.seed)
    train = {g: [r for r in ref if r["group"] == g and split[r["sample_id"]] == "train"] for g in ("tumour", "healthy")}
    test = {g: [r for r in ref if r["group"] == g and split[r["sample_id"]] == "test"] for g in ("tumour", "healthy")}
    for g in ("tumour", "healthy"):
        if not train[g]:
            raise SystemExit(f"No {g} in train split.")
        if not test[g]:
            raise SystemExit(f"No {g} in test split.")
    logger.info("Split train t/h=%d/%d  test t/h=%d/%d",
                len(train["tumour"]), len(train["healthy"]), len(test["tumour"]), len(test["healthy"]))

    profiles = discover_panel(
        [(r["file_path"], r["convention"]) for r in train["tumour"]],
        [(r["file_path"], r["convention"]) for r in train["healthy"]],
        window=args.window, top_n=args.top_n, min_total=args.min_total,
        min_effect=args.min_effect, direction=args.direction,
    )

    t_scores = score_reads(profiles, [(r["file_path"], r["convention"], r["sample_id"], r["cohort"]) for r in test["tumour"]],
                           label=1, min_ref_obs=args.min_ref_obs)
    h_scores = score_reads(profiles, [(r["file_path"], r["convention"], r["sample_id"], r["cohort"]) for r in test["healthy"]],
                           label=0, min_ref_obs=args.min_ref_obs)
    metrics = oracle_metrics(t_scores, h_scores, rng_seed=args.seed)

    from rltf.llr import _weighted_auc  # noqa: E402
    per_cohort = {}
    for cohort in sorted(set(h_scores.cohort.tolist())):
        mask = h_scores.cohort == cohort
        sub = merge_scores([t_scores, ReadScores(
            llr=h_scores.llr[mask], n_cpg_scored=h_scores.n_cpg_scored[mask], weight=h_scores.weight[mask],
            label=h_scores.label[mask], sample_id=h_scores.sample_id[mask], cohort=h_scores.cohort[mask],
            block_id=h_scores.block_id[mask],
        )])
        fin = np.isfinite(sub.llr)
        per_cohort[cohort] = {
            "auc_vs_tumour": _weighted_auc(sub.label[fin].astype(int), sub.llr[fin], sub.weight[fin].astype(float)),
            "n_healthy_reads": int(h_scores.weight[mask].sum()),
        }

    summary = {
        "n_blocks": len(profiles.blocks), "window": args.window,
        "n_tumour_train": len(train["tumour"]), "n_healthy_train": len(train["healthy"]),
        "n_tumour_test": len(test["tumour"]), "n_healthy_test": len(test["healthy"]),
        "metrics": metrics, "per_healthy_cohort": per_cohort,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "split.json").write_text(json.dumps(
        {"train": {g: [r["sample_id"] for r in train[g]] for g in train},
         "test": {g: [r["sample_id"] for r in test[g]] for g in test}}, indent=2))

    # Persist down-sampled per-read scores and draw the oracle dashboard.
    if args.write_per_read > 0 or not args.no_plots:
        import pandas as pd

        merged = merge_scores([t_scores, h_scores])
        n = len(merged.llr)
        rng2 = np.random.default_rng(args.seed)
        sel = np.arange(n) if (args.write_per_read <= 0 or n <= args.write_per_read) \
            else rng2.choice(n, args.write_per_read, replace=False)
        reads_df = pd.DataFrame({
            "llr": merged.llr[sel], "n_cpg_scored": merged.n_cpg_scored[sel],
            "weight": merged.weight[sel], "label": merged.label[sel],
            "sample_id": merged.sample_id[sel], "cohort": merged.cohort[sel],
        })
        if args.write_per_read > 0:
            reads_df.to_csv(args.out_dir / "per_read_scores.tsv.gz", sep="\t", index=False)
        if not args.no_plots:
            try:
                from rltf.plots import plot_oracle
                plot_oracle(reads_df, metrics, args.out_dir / "plots")
            except Exception as exc:  # pragma: no cover
                logger.warning("plotting skipped: %r", exc)

    logger.info("AUC=%.4f permuted=%.4f ncpg_matched=%.4f | %s",
                metrics["auc"], metrics["auc_permuted"], metrics["auc_ncpg_matched"], metrics["verdict"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
