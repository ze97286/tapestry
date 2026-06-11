#!/usr/bin/env python3
"""Step 3 — read-level tabular detector / tumour-fraction estimator.

Loads the discovered panel (step 1), builds per-sample read-level features for
the manifest's ``role=query`` rows, and evaluates a tabular foundation-model head
leave-one-cohort-out: cancer-vs-healthy classification and tumour-fraction
regression (where ``tf`` is present). Run only after the oracle gate passes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from rltf.features import build_feature_matrix, feature_names
from rltf.head import leave_one_group_out_cv
from rltf.manifest import read_manifest
from rltf.profiles import ReferenceProfiles

logger = logging.getLogger("run_detector")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--panel-dir", required=True, type=Path, help="output of discover_panel.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backend", default="tabicl", choices=["tabicl", "tabpfn", "sklearn"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--group-col", default="cohort")
    ap.add_argument("--min-ref-obs", type=float, default=5.0)
    ap.add_argument("--regression-min-tf", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    profiles = ReferenceProfiles.load(args.panel_dir)
    query = [r for r in read_manifest(args.manifest) if r["role"] == "query"]
    if not query:
        raise SystemExit("No role=query rows in manifest.")
    logger.info("Scoring %d query samples against %d-block panel", len(query), len(profiles.blocks))

    feat = build_feature_matrix(
        profiles,
        [(r["file_path"], r["convention"], r["sample_id"], r["cohort"]) for r in query],
        min_ref_obs=args.min_ref_obs,
    )
    by_id = {r["sample_id"]: r for r in query}
    feat["is_cancer"] = [by_id[s]["is_cancer"] for s in feat.index]
    feat["tf"] = [by_id[s]["tf"] for s in feat.index]
    feat.to_csv(args.out_dir / "features.tsv", sep="\t")

    X = feat[feature_names()].to_numpy(dtype=np.float64)
    groups = feat[args.group_col].to_numpy()
    summary: dict = {"backend_requested": args.backend, "n_blocks": len(profiles.blocks),
                     "n_query": len(query), "group_col": args.group_col}

    import pandas as pd

    cls_oof_df = reg_oof_df = None
    y_cls = feat["is_cancer"].to_numpy(dtype=float)
    m = np.isfinite(y_cls)
    if m.sum() >= 4 and len(np.unique(y_cls[m])) == 2:
        cv = leave_one_group_out_cv(X[m], y_cls[m].astype(int), groups[m], task="classify",
                                    backend=args.backend, device=args.device, random_state=args.seed)
        summary["classification"] = {"backend_used": cv.backend, **cv.metrics}
        cls_oof_df = pd.DataFrame({"sample_id": feat.index[m], "group": groups[m], "is_cancer": y_cls[m],
                                   "pred_proba_cancer": cv.oof_pred})
        cls_oof_df.to_csv(args.out_dir / "classification_oof.tsv", sep="\t", index=False)
        logger.info("Classification LOGO AUC=%.4f (backend=%s)", cv.metrics.get("auc", float("nan")), cv.backend)
    else:
        summary["classification"] = {"skipped": "need >=4 labelled query samples spanning both classes"}

    y_tf = feat["tf"].to_numpy(dtype=float)
    mt = np.isfinite(y_tf)
    if mt.sum() >= 5:
        cv = leave_one_group_out_cv(X[mt], y_tf[mt], groups[mt], task="regress", backend=args.backend,
                                    device=args.device, random_state=args.seed, regression_min_tf=args.regression_min_tf)
        summary["regression"] = {"backend_used": cv.backend, **cv.metrics}
        reg_oof_df = pd.DataFrame({"sample_id": feat.index[mt], "group": groups[mt], "tf_true": y_tf[mt],
                                   "tf_pred": cv.oof_pred})
        reg_oof_df.to_csv(args.out_dir / "regression_oof.tsv", sep="\t", index=False)
        logger.info("Regression LOGO pearson r=%.4f (backend=%s)", cv.metrics.get("pearson_r", float("nan")), cv.backend)
    else:
        summary["regression"] = {"skipped": "need >=5 query samples with a tf value"}

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_plots:
        try:
            from rltf.plots import plot_detector
            plot_detector(feat, cls_oof_df, reg_oof_df, summary, args.out_dir / "plots")
        except Exception as exc:  # pragma: no cover
            logger.warning("plotting skipped: %r", exc)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
