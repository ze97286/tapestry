#!/usr/bin/env python3
"""Step 4 — clinical evaluation.

Ties the detector's out-of-fold predictions back to patients, timepoints,
ichorCNA and (optionally) survival. Produces clinical metrics and plots:

* predicted tumour fraction vs **ichorCNA** (independent validation), Pearson r;
* per-patient **longitudinal** TF trajectories and a first→last **waterfall**;
* **detection calls** at a specificity threshold learned from the healthy
  controls (sensitivity at fixed specificity), optionally by stage;
* **Kaplan–Meier** by TF-change direction with a log-rank p (if survival given).

Inputs: a detector output dir (``classification_oof.tsv`` / ``regression_oof.tsv``)
and a clinical TSV keyed by ``sample_id`` with optional columns
``patient_id timepoint ichorcna_tf survival_time survival_event stage``.
ichorCNA here is used only as an independent validation target.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("run_clinical_eval")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-dir", required=True, type=Path, help="output dir of run_detector.py")
    ap.add_argument("--clinical-tsv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--specificity", type=float, default=0.95, help="target specificity for the detection threshold")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cls_path = args.detector_dir / "classification_oof.tsv"
    reg_path = args.detector_dir / "regression_oof.tsv"
    if not cls_path.exists() and not reg_path.exists():
        raise SystemExit("detector-dir has neither classification_oof.tsv nor regression_oof.tsv.")

    merged = None
    if cls_path.exists():
        merged = pd.read_csv(cls_path, sep="\t")[["sample_id", "is_cancer", "pred_proba_cancer"]]
    if reg_path.exists():
        reg = pd.read_csv(reg_path, sep="\t")[["sample_id", "tf_pred"]]
        merged = reg if merged is None else merged.merge(reg, on="sample_id", how="outer")

    clin = pd.read_csv(args.clinical_tsv, sep="\t")
    if "sample_id" not in clin.columns:
        raise SystemExit("clinical TSV needs a 'sample_id' column.")
    merged = merged.merge(clin, on="sample_id", how="left")

    summary: dict = {"n_samples": int(len(merged)),
                     "n_patients": int(merged["patient_id"].nunique()) if "patient_id" in merged else None}

    # Detection threshold from healthy controls, then sensitivity.
    threshold = None
    if {"is_cancer", "pred_proba_cancer"}.issubset(merged.columns):
        neg = merged.loc[merged["is_cancer"] == 0, "pred_proba_cancer"].dropna()
        pos = merged.loc[merged["is_cancer"] == 1, "pred_proba_cancer"].dropna()
        if len(neg) and len(pos):
            threshold = float(np.quantile(neg, args.specificity))
            merged["detected"] = (merged["pred_proba_cancer"] >= threshold).astype("Int64")
            summary["detection"] = {
                "target_specificity": args.specificity,
                "threshold": threshold,
                "sensitivity": float(np.mean(pos >= threshold)),
                "achieved_specificity": float(np.mean(neg < threshold)),
                "n_cancer": int(len(pos)), "n_healthy": int(len(neg)),
            }
            if "stage" in merged.columns:
                by_stage = {}
                for stg, g in merged[(merged["is_cancer"] == 1)].dropna(subset=["pred_proba_cancer"]).groupby("stage"):
                    by_stage[str(stg)] = {"n": int(len(g)),
                                          "sensitivity": float(np.mean(g["pred_proba_cancer"] >= threshold))}
                summary["detection"]["sensitivity_by_stage"] = by_stage

    # Predicted TF vs ichorCNA (independent validation).
    if {"tf_pred", "ichorcna_tf"}.issubset(merged.columns):
        sub = merged.dropna(subset=["tf_pred", "ichorcna_tf"])
        if len(sub) >= 3:
            from scipy.stats import pearsonr, spearmanr
            summary["tf_vs_ichorcna_pearson_r"] = float(pearsonr(sub["ichorcna_tf"], sub["tf_pred"])[0])
            summary["tf_vs_ichorcna_spearman_r"] = float(spearmanr(sub["ichorcna_tf"], sub["tf_pred"])[0])
            summary["tf_vs_ichorcna_n"] = int(len(sub))

    merged.to_csv(args.out_dir / "clinical_merged.tsv", sep="\t", index=False)

    if not args.no_plots:
        try:
            from rltf.plots import plot_clinical
            paths = plot_clinical(merged, summary, args.out_dir / "plots", threshold=threshold)
            summary["plots"] = [str(p.name) for p in paths]
        except Exception as exc:  # pragma: no cover
            logger.warning("plotting skipped: %r", exc)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
