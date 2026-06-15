#!/usr/bin/env python3
"""Step 4 — clinical evaluation (detection-first).

Config-driven. Joins the detector's out-of-fold detection scores to the clinical
CSV (survival/benefit) and to ichorCNA (validation only). Reports detection
sensitivity at a controlled specificity, an ichorCNA sanity-correlation where
ichorCNA is meaningful (tf >= clinical.ichorcna_min_tf), and Kaplan-Meier by
detection-score split. ichorCNA is never a model input — only a sanity anchor.

(Longitudinal on-treatment TF trajectories require scoring the on-treatment
timepoints, which are excluded from training/detection here; that is a separate
follow-up pass.)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rltf.config import get, load_config, numeric_prefix

logger = logging.getLogger("run_clinical_eval")
DEFAULT_CONFIG = "readlevel_tf/configs/config.toml"


def _load_clinical(csv_path: str) -> dict[str, dict]:
    out = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            subj = (row.get("subject") or "").strip()
            if not subj:
                continue
            out[numeric_prefix(subj)] = row
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def detection_by_tf(pos_df: pd.DataFrame, threshold: float, edges) -> dict:
    """Detection sensitivity among positives, stratified by ichorCNA tumour fraction.

    `pos_df` needs columns ``tf`` (ichorCNA TF — x-axis only, never a model input) and
    ``pred_proba_cancer`` (the detector OOF score). The threshold is fixed externally by
    the controls at the target specificity. Answers the core question: where is the
    detection floor, and are the confidently-called positives the higher-TF ones?
    Positives with no ichorCNA TF are reported separately, not silently binned.
    """
    edges = sorted({float(e) for e in edges})
    ranges = ([(0.0, edges[0])] + [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
              + [(edges[-1], float("inf"))])
    known = pos_df.dropna(subset=["tf"])
    bins = []
    for lo, hi in ranges:
        sub = known[(known["tf"] >= lo) & (known["tf"] < hi)]
        det = sub["pred_proba_cancer"] >= threshold
        hi_lbl = "inf" if hi == float("inf") else f"{hi:.0%}"
        bins.append({"tf_range": f"[{lo:.0%},{hi_lbl})", "n": int(len(sub)),
                     "n_detected": int(det.sum()),
                     "sensitivity": float(det.mean()) if len(sub) else None})
    return {"threshold": threshold, "bins": bins,
            "n_positives_with_tf": int(len(known)),
            "n_positives_without_tf": int(pos_df["tf"].isna().sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    run_dir = Path(get(cfg, "paths.run_dir", "readlevel_tf/runs"))
    det_dir, out_dir = run_dir / "detector", run_dir / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = get(cfg, "clinical.specificity", 0.95)
    ichor_min = get(cfg, "clinical.ichorcna_min_tf", 0.03)

    oof_path = det_dir / "classification_oof.tsv"
    man_path = det_dir / "manifest.tsv"
    if not oof_path.exists():
        raise SystemExit(f"{oof_path} missing — run the detector (step 3) first.")
    oof = pd.read_csv(oof_path, sep="\t")
    man = pd.read_csv(man_path, sep="\t")[["sample_id", "patient_id", "cohort", "tf"]]
    df = oof.merge(man, on="sample_id", how="left")

    clinical = _load_clinical(get(cfg, "paths.clinical_csv", "")) if get(cfg, "paths.clinical_csv", "") else {}

    def clin(sample_patient, field):
        rec = clinical.get(numeric_prefix(str(sample_patient))) if isinstance(sample_patient, str) else None
        return rec.get(field) if rec else None

    df["os_days"] = [_f(clin(p, "OS_days")) for p in df["patient_id"]]
    df["os_event"] = [_f(clin(p, "OS")) for p in df["patient_id"]]
    df["clinical_benefit"] = [clin(p, "Clinical_Benefit") for p in df["patient_id"]]
    df.to_csv(out_dir / "clinical_merged.tsv", sep="\t", index=False)

    summary: dict = {"n_samples": int(len(df))}

    neg = df.loc[df["is_cancer"] == 0, "pred_proba_cancer"].dropna()
    pos = df.loc[df["is_cancer"] == 1, "pred_proba_cancer"].dropna()
    threshold = None
    if len(neg) and len(pos):
        threshold = float(np.quantile(neg, spec))
        df["detected"] = (df["pred_proba_cancer"] >= threshold).astype("Int64")
        summary["detection"] = {"target_specificity": spec, "threshold": threshold,
                                "sensitivity": float(np.mean(pos >= threshold)),
                                "achieved_specificity": float(np.mean(neg < threshold)),
                                "n_cancer": int(len(pos)), "n_healthy": int(len(neg))}
        # The low-TF question: sensitivity by ichorCNA TF bin at the control-fixed threshold.
        summary["detection_by_tf"] = {"target_specificity": spec,
                                      **detection_by_tf(df[df["is_cancer"] == 1], threshold,
                                                        get(cfg, "clinical.tf_bins", [0.01, 0.03, 0.10]))}

    # ichorCNA sanity: our score vs ichorCNA where ichorCNA is meaningful.
    sane = df[(df["is_cancer"] == 1) & (df["tf"] >= ichor_min)].dropna(subset=["tf", "pred_proba_cancer"])
    if len(sane) >= 3:
        from scipy.stats import pearsonr, spearmanr
        summary["ichorcna_sanity"] = {"n": int(len(sane)), "min_tf": ichor_min,
                                      "pearson_r": float(pearsonr(sane["tf"], sane["pred_proba_cancer"])[0]),
                                      "spearman_r": float(spearmanr(sane["tf"], sane["pred_proba_cancer"])[0])}
    else:
        summary["ichorcna_sanity"] = {"skipped": f"<3 positives with ichorCNA tf >= {ichor_min}"}

    # KM by detection-score split (positives with survival).
    surv = df[(df["is_cancer"] == 1)].dropna(subset=["os_days", "os_event", "pred_proba_cancer"]).drop_duplicates("patient_id")
    if len(surv) >= 6:
        med = float(surv["pred_proba_cancer"].median())
        surv = surv.assign(arm=np.where(surv["pred_proba_cancer"] >= med, "high_score", "low_score"))
        try:
            from rltf.plots import _logrank_p
            a = surv[surv["arm"] == "low_score"]; b = surv[surv["arm"] == "high_score"]
            summary["km_logrank_p"] = _logrank_p(a["os_days"].to_numpy(), a["os_event"].to_numpy(),
                                                 b["os_days"].to_numpy(), b["os_event"].to_numpy())
        except Exception as exc:  # pragma: no cover
            logger.warning("KM skipped: %r", exc)

    if not args.no_plots:
        try:
            from rltf.plots import plot_clinical
            plot_clinical(df.rename(columns={"tf": "ichorcna_tf", "pred_proba_cancer": "pred_proba_cancer"}),
                          summary, out_dir / "plots", threshold=threshold)
        except Exception as exc:  # pragma: no cover
            logger.warning("plotting skipped: %r", exc)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
