#!/usr/bin/env python3
"""Step 5 — longitudinal monitoring (on-treatment trajectory vs survival).

Scores the on-treatment (followup) timepoints with the baseline-trained detector and
relates the per-patient score trajectory (followup − baseline) to overall survival.

Constraints held throughout:
  * On-treatment samples NEVER enter training. The detector is fit on baseline positives
    vs controls and APPLIED to the followup timepoints out-of-sample — exactly a deployed
    monitor. (Baseline and followup are scored by the SAME final model so the Δ is on one
    scale; the unbiased OOF baseline score is also recorded for reference.)
  * The score is a tumour-burden PROXY (a detection probability), not a calibrated tumour
    fraction. ichorCNA is a sanity anchor only — never an input or a target.
  * Survival schemas differ by cohort (AB event = OS, CD event = OS_ind); the per-cohort
    columns come from the config via load_survival(), so neither cohort is silently dropped.

Requires the detector (step 3) to have run: reuses runs/detector/{panel reference, the
final-model training features, and the OOF scores}.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rltf.config import (build_manifest_rows, get, load_config, load_survival,
                         numeric_prefix, write_manifest_tsv)
from rltf.features import build_feature_matrix, feature_names
from rltf.head import TabularTumourHead
from rltf.llr import fit_calibration, score_fragments
from rltf.manifest import read_manifest
from rltf.profiles import ReferenceProfiles

logger = logging.getLogger("run_monitor")
DEFAULT_CONFIG = "readlevel_tf/configs/config.toml"


def summarise_trajectory(traj: pd.DataFrame, threshold: float | None) -> dict:
    """Pure summary of a baseline→followup score-trajectory table (one row per followup sample)."""
    d = traj.dropna(subset=["delta"])
    out = {
        "n_followup_samples": int(len(traj)),
        "n_with_paired_scores": int(len(d)),
        "n_decreased": int((d["delta"] < 0).sum()),
        "frac_decreased": (float((d["delta"] < 0).mean()) if len(d) else None),
        "median_delta": (float(d["delta"].median()) if len(d) else None),
    }
    if threshold is not None:
        r = traj.dropna(subset=["followup_score"])
        out["residual_threshold"] = float(threshold)
        out["n_residual_at_followup"] = int((r["followup_score"] >= threshold).sum())
        out["n_cleared_at_followup"] = int((r["followup_score"] < threshold).sum())
    return out


def _benefit_vs_delta(traj: pd.DataFrame, col: str = "delta") -> dict:
    """Test the headline hypothesis: a falling burden on treatment (Δ<=0) predicts clinical benefit.

    `col` is the trajectory metric — the saturating classifier Δ ("delta") or the continuous
    Δmean_z ("delta_meanz"). benefit is 1/0; a benefit patient should have a MORE NEGATIVE Δ,
    so −Δ is the score that should rank benefit highest. Reports AUC(−Δ→benefit), per-group
    median Δ, a down/up × benefit 2×2 with Fisher p, and a one-sided Mann–Whitney that benefit
    Δ is stochastically lower.
    """
    b = traj.dropna(subset=[col, "benefit"]).drop_duplicates("patient_id")
    if len(b) < 6 or b["benefit"].nunique() < 2:
        return {"skipped": f"need >=6 patients with {col}+benefit spanning both classes, have {len(b)}"}
    from scipy.stats import fisher_exact, mannwhitneyu
    from sklearn.metrics import roc_auc_score
    y = b["benefit"].astype(int).to_numpy()
    d = b[col].to_numpy()
    down = d <= 0
    tab = [[int((down & (y == 1)).sum()), int((down & (y == 0)).sum())],
           [int((~down & (y == 1)).sum()), int((~down & (y == 0)).sum())]]
    return {
        "n": int(len(b)), "n_benefit": int((y == 1).sum()), "n_no_benefit": int((y == 0).sum()),
        "auc_neg_delta_predicts_benefit": float(roc_auc_score(y, -d)),
        "median_delta_benefit": float(np.median(d[y == 1])),
        "median_delta_no_benefit": float(np.median(d[y == 0])),
        "contingency_downup_x_benefit": tab,
        "fisher_p": float(fisher_exact(tab)[1]),
        "mannwhitney_p_benefit_more_negative": float(mannwhitneyu(d[y == 1], d[y == 0], alternative="less").pvalue),
    }


def _km_split(df: pd.DataFrame, mask, score_cols) -> dict:
    """Log-rank OS comparison of the patients where mask is True vs False (deduped by patient)."""
    from rltf.plots import _logrank_p
    d = df.dropna(subset=list(score_cols) + ["os_days", "os_event"]).drop_duplicates("patient_id")
    if len(d) < 6:
        return {"skipped": f"need >=6 patients with score+survival, have {len(d)}"}
    g = mask(d)
    a, b = d[g], d[~g]
    if not len(a) or not len(b):
        return {"skipped": "one arm empty"}
    return {"n_group_a": int(len(a)), "n_group_b": int(len(b)),
            "logrank_p_os": _logrank_p(a["os_days"].to_numpy(), a["os_event"].to_numpy(),
                                       b["os_days"].to_numpy(), b["os_event"].to_numpy()),
            "median_os_days_a": float(a["os_days"].median()),
            "median_os_days_b": float(b["os_days"].median())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--from-features", action="store_true",
                    help="reuse saved followup_features.tsv; skip scoring (instant re-analysis)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    run_dir = Path(get(cfg, "paths.run_dir", "readlevel_tf/runs"))
    panel_dir, det_dir, out_dir = run_dir / "panel", run_dir / "detector", run_dir / "monitor"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = get(cfg, "clinical.specificity", 0.95)
    mro, mq, flank = get(cfg, "scoring.min_ref_obs", 5), get(cfg, "scoring.min_mapq", 30), get(cfg, "scoring.flank_bp", 1000)
    backend, device, seed = get(cfg, "detector.backend", "tabicl"), get(cfg, "detector.device", "cpu"), get(cfg, "detector.seed", 0)

    for need in ("classification_oof.tsv", "features.tsv"):
        if not (det_dir / need).exists():
            raise SystemExit(f"{det_dir / need} missing — run the detector (step 3) first.")

    write_manifest_tsv(build_manifest_rows(cfg), out_dir / "manifest.tsv")
    rows = read_manifest(out_dir / "manifest.tsv")
    followup = [r for r in rows if r["role"] == "followup"]
    baseline_pos = {r["sample_id"]: r for r in rows if r["role"] == "query" and r["is_cancer"] == 1}
    ref_healthy = [r for r in rows if r["role"] == "reference" and r["group"] == "healthy"]
    if not followup:
        raise SystemExit("No role=followup rows; set labels.followup_timepoints in the config.")

    cache = out_dir / "followup_features.tsv"
    if args.from_features and cache.exists():
        fu_feat = pd.read_csv(cache, sep="\t", index_col=0)
        logger.info("Reusing %d saved followup features (no scoring)", len(fu_feat))
    else:
        if not (panel_dir / "cpgs.tsv").exists():
            raise SystemExit(f"No panel at {panel_dir}; run discover_panel.py (step 1) first.")
        profiles = ReferenceProfiles.load(panel_dir)
        # Same null calibration as the detector: empirical per-k null on the reference-healthy controls.
        cal_sc = score_fragments(profiles, [(r["file_path"], r["sample_id"], r["cohort"]) for r in ref_healthy],
                                 label=0, min_ref_obs=mro, min_mapq=mq, flank=flank)
        calibration = fit_calibration(cal_sc, min_per_k=get(cfg, "scoring.calib_min_per_k", 200))
        logger.info("Scoring %d followup samples against the %d-CpG panel", len(followup), len(profiles.cpg_pos))
        fu_feat = build_feature_matrix(
            profiles, [(r["file_path"], r["sample_id"], r["cohort"]) for r in followup], calibration,
            min_ref_obs=mro, min_mapq=mq, flank=flank)
        fu_feat.to_csv(cache, sep="\t")

    # Final detector: trained on ALL baseline query (positives + controls), applied out-of-sample
    # to followup. Baseline scored by the same model for a single-scale Δ.
    base_feat = pd.read_csv(det_dir / "features.tsv", sep="\t", index_col=0)
    Xb, yb = base_feat[feature_names()].to_numpy(np.float64), base_feat["is_cancer"].to_numpy(float)
    m = np.isfinite(yb)
    head = TabularTumourHead(task="classify", backend=backend, device=device, random_state=seed).fit(Xb[m], yb[m].astype(int))
    base_feat["final_score"] = head.predict_proba(Xb)
    fu_score = head.predict_proba(fu_feat[feature_names()].to_numpy(np.float64))

    # Baseline reference scores per patient: final-model (Δ scale) and unbiased OOF.
    oof = pd.read_csv(det_dir / "classification_oof.tsv", sep="\t")
    base_final = {baseline_pos[str(s)]["patient_id"]: float(v)
                  for s, v in base_feat["final_score"].items() if str(s) in baseline_pos}
    base_oof = {baseline_pos[str(s)]["patient_id"]: float(p)
                for s, p in zip(oof["sample_id"], oof["pred_proba_cancer"]) if str(s) in baseline_pos}
    base_tf = {r["patient_id"]: r["tf"] for r in rows if r["role"] == "query" and r["is_cancer"] == 1}

    # Continuous burden axis: mean_z (already a feature) — a better Δ metric than the
    # saturating classifier probability for trajectories.
    fu_meanz = dict(zip(fu_feat.index.astype(str), fu_feat["mean_z"].astype(float)))
    base_meanz = {baseline_pos[str(s)]["patient_id"]: float(v)
                  for s, v in base_feat["mean_z"].items() if str(s) in baseline_pos}

    # Align by sample_id (not row position) so we never mismatch a score to the wrong sample.
    fu_meta = {r["sample_id"]: r for r in followup}
    fu_score = dict(zip(fu_feat.index.astype(str), fu_score))
    traj = pd.DataFrame([{
        "patient_id": meta["patient_id"], "cohort": meta["cohort"],
        "followup_sample": sid, "timepoint": meta["timepoint"],
        "baseline_score": base_final.get(meta["patient_id"], np.nan),
        "baseline_score_oof": base_oof.get(meta["patient_id"], np.nan),
        "followup_score": fu_score[str(sid)],
        "delta": (fu_score[str(sid)] - base_final[meta["patient_id"]]
                  if meta["patient_id"] in base_final else np.nan),
        "baseline_meanz": base_meanz.get(meta["patient_id"], np.nan),
        "followup_meanz": fu_meanz[str(sid)],
        "delta_meanz": (fu_meanz[str(sid)] - base_meanz[meta["patient_id"]]
                        if meta["patient_id"] in base_meanz else np.nan),
        "baseline_tf": base_tf.get(meta["patient_id"], np.nan), "followup_tf": meta["tf"],
    } for sid in fu_feat.index for meta in [fu_meta[str(sid)]]])

    # Per-feature Δ (followup − baseline) for every read-level feature, so we can ask which
    # feature's trajectory best predicts benefit — not just mean_z. frac_z_gt_* is a
    # tumour-like-read fraction: an ichorCNA-free burden/TF proxy.
    for feat in feature_names():
        bvals = {baseline_pos[str(s)]["patient_id"]: float(base_feat.loc[s, feat])
                 for s in base_feat.index if str(s) in baseline_pos}
        fvals = dict(zip(fu_feat.index.astype(str), fu_feat[feat].astype(float)))
        traj[f"delta_{feat}"] = [fvals[str(sid)] - bvals.get(pid, np.nan)
                                 for sid, pid in zip(traj["followup_sample"], traj["patient_id"])]

    # Control-derived residual threshold (same controls that defined detector specificity).
    neg = oof.loc[oof["is_cancer"] == 0, "pred_proba_cancer"].dropna()
    threshold = float(np.quantile(neg, spec)) if len(neg) else None

    surv = load_survival(cfg)
    keys = list(zip(traj["cohort"], traj["patient_id"]))
    traj["os_days"] = [surv.get((c, numeric_prefix(str(p))), {}).get("os_days", np.nan) for c, p in keys]
    traj["os_event"] = [surv.get((c, numeric_prefix(str(p))), {}).get("os_event", np.nan) for c, p in keys]
    traj["benefit"] = [surv.get((c, numeric_prefix(str(p))), {}).get("benefit", None) for c, p in keys]
    traj.to_csv(out_dir / "trajectories.tsv", sep="\t", index=False)

    dz = traj.dropna(subset=["delta_meanz"])
    summary: dict = {"n_followup": len(followup), "backend_used": head.backend, "target_specificity": spec,
                     "n_with_survival": int(traj["os_days"].notna().sum()),
                     "n_with_benefit": int(traj["benefit"].notna().sum()),
                     "trajectory": summarise_trajectory(traj, threshold),
                     "trajectory_meanz": {
                         "n_with_paired_meanz": int(len(dz)),
                         "n_decreased": int((dz["delta_meanz"] < 0).sum()),
                         "frac_decreased": (float((dz["delta_meanz"] < 0).mean()) if len(dz) else None),
                         "median_delta_meanz": (float(dz["delta_meanz"].median()) if len(dz) else None)}}
    # KM split by Δ direction — v2 convention: down = Δ<=0 (favourable arm) vs up = Δ>0.
    summary["km_by_trajectory"] = _km_split(traj, lambda d: d["delta"] <= 0, ["delta"])
    summary["km_by_trajectory_meanz"] = _km_split(traj, lambda d: d["delta_meanz"] <= 0, ["delta_meanz"])
    if threshold is not None:
        # Does molecular residual at the followup timepoint track worse survival?
        summary["km_by_residual"] = _km_split(traj, lambda d: d["followup_score"] >= threshold, ["followup_score"])
        # Confound check: is the residual signal just baseline stage? Baseline detection alone,
        # and on-treatment clearance WITHIN the baseline-detected (the MRD-beyond-baseline test).
        summary["km_by_baseline_residual"] = _km_split(traj, lambda d: d["baseline_score"] >= threshold, ["baseline_score"])
        det = traj[traj["baseline_score"] >= threshold]
        summary["km_residual_within_baseline_detected"] = {
            "n_baseline_detected": int(traj["baseline_score"].ge(threshold).sum()),
            **_km_split(det, lambda d: d["followup_score"] >= threshold, ["followup_score"])}
    # HEADLINE hypothesis: a falling burden on treatment (Δ<=0) predicts clinical benefit —
    # on both the saturating probability Δ and the continuous Δmean_z.
    summary["benefit_vs_delta"] = _benefit_vs_delta(traj, "delta")
    summary["benefit_vs_delta_meanz"] = _benefit_vs_delta(traj, "delta_meanz")

    # (a) AB-only head-to-head with v2 (AB AUC 0.834) + (b) feature-Δ sweep: which read-level
    # feature's trajectory best predicts benefit, pooled and AB-only. Compact (auc + MW p).
    def _sweep(t: pd.DataFrame) -> dict:
        out = {}
        for feat in feature_names():
            r = _benefit_vs_delta(t, f"delta_{feat}")
            out[feat] = ({"auc": round(r["auc_neg_delta_predicts_benefit"], 4),
                          "mw_p": r["mannwhitney_p_benefit_more_negative"], "n": r["n"],
                          "median_benefit": r["median_delta_benefit"],
                          "median_no_benefit": r["median_delta_no_benefit"]}
                         if "auc_neg_delta_predicts_benefit" in r else r)
        return out

    ab = traj[traj["cohort"] == "AB"]
    summary["feature_delta_sweep"] = {"ALL": _sweep(traj), "AB": _sweep(ab)}
    # AB-only headline (mean_z) to compare like-for-like with v2's AB numbers.
    summary["AB_only"] = {
        "n_followup": int(len(ab)),
        "benefit_vs_delta_meanz": _benefit_vs_delta(ab, "delta_meanz"),
        "km_by_trajectory_meanz": _km_split(ab, lambda d: d["delta_meanz"] <= 0, ["delta_meanz"]),
        "km_by_residual": (_km_split(ab, lambda d: d["followup_score"] >= threshold, ["followup_score"])
                           if threshold is not None else {"skipped": "no threshold"})}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # v2-equivalent figures: Δ waterfall coloured by benefit + KM by Δ direction.
    try:
        from rltf.plots import plot_monitor
        for coh, metric in [("AB", "delta_frac_z_gt_2"), ("AB", "delta_mean_z"), (None, "delta_frac_z_gt_2")]:
            plot_monitor(traj, summary, out_dir / "plots", metric=metric, cohort=coh)
    except Exception as exc:  # pragma: no cover
        logger.warning("monitor plotting skipped: %r", exc)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
