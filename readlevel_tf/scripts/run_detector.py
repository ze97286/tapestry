#!/usr/bin/env python3
"""Step 3 — read-level cancer detector (detection only).

Config-driven. Loads the panel, builds per-sample features for the query rows
(EAC baseline patients = positives, held-out controls = negatives), and
evaluates a tabular head leave-one-group-out. ichorCNA is NOT used here — it is
clinical validation only.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from rltf.config import build_manifest_rows, get, load_config, write_manifest_tsv
from rltf.features import build_feature_matrix, feature_names
from rltf.head import leave_one_group_out_cv
from rltf.llr import fit_calibration, score_fragments
from rltf.manifest import read_manifest
from rltf.profiles import ReferenceProfiles

logger = logging.getLogger("run_detector")
DEFAULT_CONFIG = "readlevel_tf/configs/config.toml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--from-features", action="store_true",
                    help="skip scoring; re-run CV/diagnostics on the saved features.tsv (instant)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    run_dir = Path(get(cfg, "paths.run_dir", "readlevel_tf/runs"))
    panel_dir, out_dir = run_dir / "panel", run_dir / "detector"
    out_dir.mkdir(parents=True, exist_ok=True)
    backend, device = get(cfg, "detector.backend", "tabicl"), get(cfg, "detector.device", "cpu")
    group_col, seed = get(cfg, "detector.group_col", "cohort"), get(cfg, "detector.seed", 0)
    cv_mode = get(cfg, "detector.cv", "loso")     # loso (leave-one-patient/control-out) | cohort
    import json as _json

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from rltf.head import TabularTumourHead

    if args.from_features:   # re-run CV/diagnostics on saved features — no scoring
        feat = pd.read_csv(out_dir / "features.tsv", sep="\t", index_col=0)
        # Reconstruct label/group columns if an older features.tsv lacks them.
        if "cohort_lbl" not in feat.columns and "cohort" in feat.columns:
            feat["cohort_lbl"] = feat["cohort"]
        if ("cv_group" not in feat.columns or "is_cancer" not in feat.columns) and (out_dir / "manifest.tsv").exists():
            man = {r["sample_id"]: r for r in read_manifest(out_dir / "manifest.tsv")}
            if "is_cancer" not in feat.columns:
                feat["is_cancer"] = [man.get(str(s), {}).get("is_cancer", np.nan) for s in feat.index]
            if "cv_group" not in feat.columns:
                feat["cv_group"] = [(man.get(str(s), {}).get("patient_id", "") or str(s)) for s in feat.index]
        n_cpg = (int(_json.loads((panel_dir / "meta.json").read_text())["n_cpg"])
                 if (panel_dir / "meta.json").exists() else 0)
        logger.info("Re-running %s CV on %d saved features (no re-scoring)", cv_mode, len(feat))
    else:
        if not (panel_dir / "cpgs.tsv").exists():
            raise SystemExit(f"No panel at {panel_dir}; run discover_panel.py (step 1) first.")
        shutil.copy(args.config, out_dir / "config.toml")
        write_manifest_tsv(build_manifest_rows(cfg), out_dir / "manifest.tsv")
        profiles = ReferenceProfiles.load(panel_dir)
        rows = read_manifest(out_dir / "manifest.tsv")
        query = [r for r in rows if r["role"] == "query"]
        ref_healthy = [r for r in rows if r["role"] == "reference" and r["group"] == "healthy"]
        if not query:
            raise SystemExit("No role=query rows; check sources in the config.")
        if not ref_healthy:
            raise SystemExit("No reference-healthy rows to calibrate the null.")
        mro, mq, flank = get(cfg, "scoring.min_ref_obs", 5), get(cfg, "scoring.min_mapq", 30), get(cfg, "scoring.flank_bp", 1000)
        # Calibrate the empirical healthy null on the reference-healthy controls
        # (disjoint from the query negatives), then score the query against it.
        cal_sc = score_fragments(profiles, [(r["file_path"], r["sample_id"], r["cohort"]) for r in ref_healthy],
                                 label=0, min_ref_obs=mro, min_mapq=mq, flank=flank)
        calibration = fit_calibration(cal_sc, min_per_k=get(cfg, "scoring.calib_min_per_k", 200))
        logger.info("Per-k null calibration on %d reference-healthy frags: %d k-bins (a=%.4f)",
                    len(cal_sc.llr), len(calibration.ks), calibration.a)
        logger.info("Scoring %d query samples against %d-CpG panel", len(query), len(profiles.cpg_pos))
        feat = build_feature_matrix(
            profiles, [(r["file_path"], r["sample_id"], r["cohort"]) for r in query], calibration,
            min_ref_obs=mro, min_mapq=mq, flank=flank)
        by_id = {r["sample_id"]: r for r in query}
        feat["is_cancer"] = [by_id[s]["is_cancer"] for s in feat.index]
        feat["cohort_lbl"] = [by_id[s]["cohort"] for s in feat.index]
        # Robust CV group: one patient (positives) or one control (negatives) per fold, so
        # every fold's training keeps both classes — avoids the degenerate single-class fold
        # that leave-one-COHORT-out hits here (AB ≈ all-cancer, CD ≈ all-control).
        feat["cv_group"] = [(by_id[s]["patient_id"] or s) for s in feat.index]
        feat.to_csv(out_dir / "features.tsv", sep="\t")
        n_cpg = int(len(profiles.cpg_pos))

    X = feat[feature_names()].to_numpy(dtype=np.float64)
    y = feat["is_cancer"].to_numpy(dtype=float)
    cohort_arr = feat["cohort_lbl"].to_numpy()
    groups = (feat["cv_group"].to_numpy() if cv_mode == "loso"
              else feat[group_col if group_col in feat.columns else "cohort_lbl"].to_numpy())
    m = np.isfinite(y)
    summary = {"backend_requested": backend, "n_cpg": n_cpg,
               "n_query": int(len(feat)), "n_positive": int(np.nansum(y == 1)),
               "n_negative": int(np.nansum(y == 0)), "cv": cv_mode,
               "class_by_cohort": {str(c): {"cancer": int(((cohort_arr == c) & (y == 1)).sum()),
                                            "healthy": int(((cohort_arr == c) & (y == 0)).sum())}
                                   for c in sorted(set(cohort_arr.tolist()))}}
    if m.sum() >= 4 and len(np.unique(y[m])) == 2:
        cv = leave_one_group_out_cv(X[m], y[m].astype(int), groups[m], task="classify",
                                    backend=backend, device=device, random_state=seed)
        summary["classification"] = {"backend_used": cv.backend, **cv.metrics}
        pd.DataFrame({"sample_id": feat.index[m], "cohort": cohort_arr[m], "is_cancer": y[m],
                      "pred_proba_cancer": cv.oof_pred}).to_csv(out_dir / "classification_oof.tsv", sep="\t", index=False)
        logger.info("Detection %s-CV AUC=%.4f sens@0.95=%.3f (backend=%s)", cv_mode,
                    cv.metrics.get("auc", float("nan")), cv.metrics.get("sens_at_spec_0.95", float("nan")), cv.backend)

        # Cross-cohort diagnostic: train on the other cohort(s), test each cohort (the honest
        # batch-generalisation check). Gracefully reports degenerate directions.
        cross = {}
        ym, Xm, cm = y[m].astype(int), X[m], cohort_arr[m]
        for held in sorted(set(cm.tolist())):
            tr, te = cm != held, cm == held
            entry = {"n_test_cancer": int((ym[te] == 1).sum()), "n_test_healthy": int((ym[te] == 0).sum())}
            if len(np.unique(ym[tr])) < 2 or len(np.unique(ym[te])) < 2:
                entry["auc"] = None
                entry["note"] = "train or test split is single-class"
            else:
                h = TabularTumourHead(task="classify", backend=backend, device=device, random_state=seed).fit(Xm[tr], ym[tr])
                entry["auc"] = float(roc_auc_score(ym[te], h.predict_proba(Xm[te])))
            cross[str(held)] = entry
        summary["classification"]["cross_cohort"] = cross
    else:
        summary["classification"] = {"skipped": "need >=4 labelled query samples spanning both classes"}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not args.no_plots:
        try:
            from rltf.plots import plot_detector
            import pandas as pd
            cls = pd.read_csv(out_dir / "classification_oof.tsv", sep="\t") if (out_dir / "classification_oof.tsv").exists() else None
            plot_detector(feat, cls, None, summary, out_dir / "plots")
        except Exception as exc:  # pragma: no cover
            logger.warning("plotting skipped: %r", exc)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
