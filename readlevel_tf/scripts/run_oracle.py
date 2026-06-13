#!/usr/bin/env python3
"""Step 2 — read-level separability oracle (the gate).

Config-driven. Splits the reference samples (tumour tissue + reference-healthy
controls) per group into train/test, discovers a panel on **train** only, and
scores held-out **test** reads by the n_cpg-conditioned z. Reports AUC plus the
length-leakage gates (``null_mean_z_by_n_cpg``, ``corr_z_read_length``).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from rltf.config import build_manifest_rows, get, load_config, write_manifest_tsv
from rltf.discovery import discover_panel
from rltf.llr import fit_calibration, merge_scores, oracle_metrics, score_fragments
from rltf.manifest import read_manifest

logger = logging.getLogger("run_oracle")
DEFAULT_CONFIG = "readlevel_tf/configs/config.toml"


def _split(ref, group, test_fraction, seed):
    members = sorted([r for r in ref if r["group"] == group], key=lambda r: r["sample_id"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(members))
    n_test = max(1, int(round(test_fraction * len(members)))) if len(members) > 1 else 0
    test_ids = {members[i]["sample_id"] for i in idx[:n_test]}
    train = [r for r in members if r["sample_id"] not in test_ids]
    test = [r for r in members if r["sample_id"] in test_ids]
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    out_dir = Path(get(cfg, "paths.run_dir", "readlevel_tf/runs")) / "oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_tsv(build_manifest_rows(cfg), out_dir / "manifest.tsv")
    shutil.copy(args.config, out_dir / "config.toml")

    ref = [r for r in read_manifest(out_dir / "manifest.tsv") if r["role"] == "reference"]
    tf, seed = get(cfg, "oracle.test_fraction", 0.3), get(cfg, "oracle.seed", 0)
    tr_t, te_t = _split(ref, "tumour", tf, seed)
    tr_h, te_h = _split(ref, "healthy", tf, seed + 1)
    for nm, grp in (("tumour-train", tr_t), ("tumour-test", te_t), ("healthy-train", tr_h), ("healthy-test", te_h)):
        if not grp:
            raise SystemExit(f"empty split '{nm}' — add reference samples or lower oracle.test_fraction.")
    logger.info("Split train t/h=%d/%d  test t/h=%d/%d", len(tr_t), len(tr_h), len(te_t), len(te_h))

    profiles = discover_panel(
        [r["file_path"] for r in tr_t], [r["file_path"] for r in tr_h],
        window=get(cfg, "discovery.window", 5), top_n=get(cfg, "discovery.top_n", 2000),
        min_total=get(cfg, "discovery.min_total", 10), min_effect=get(cfg, "discovery.min_effect", 0.3),
        direction=get(cfg, "discovery.direction", "any"), min_mapq=get(cfg, "scoring.min_mapq", 30),
        cross_fit=get(cfg, "discovery.cross_fit", True))

    mro, flank, mq = get(cfg, "scoring.min_ref_obs", 5), get(cfg, "scoring.flank_bp", 1000), get(cfg, "scoring.min_mapq", 30)
    # Calibrate the empirical healthy null on TRAIN healthy (disjoint from the test
    # reads being scored), then apply to held-out test reads.
    cal_sc = score_fragments(profiles, [(r["file_path"], r["sample_id"], r["cohort"]) for r in tr_h],
                             label=0, min_ref_obs=mro, flank=flank, min_mapq=mq)
    calibration = fit_calibration(cal_sc)
    logger.info("Null calibration on %d train-healthy frags: a=%.4f b=%.4f",
                len(cal_sc.llr), calibration.a, calibration.b)
    t_sc = score_fragments(profiles, [(r["file_path"], r["sample_id"], r["cohort"]) for r in te_t],
                           label=1, min_ref_obs=mro, flank=flank, min_mapq=mq)
    h_sc = score_fragments(profiles, [(r["file_path"], r["sample_id"], r["cohort"]) for r in te_h],
                           label=0, min_ref_obs=mro, flank=flank, min_mapq=mq)
    metrics = oracle_metrics(t_sc, h_sc, calibration, rng_seed=seed)

    # Per-test-sample mean z (healthy should be ~0; an outlier sample => batch, not bias).
    def _per_sample(sc):
        out = {}
        z = calibration.z(sc.llr, sc.n_cpg)
        for sid in sorted(set(sc.sample_id.tolist())):
            zz = z[sc.sample_id == sid]
            out[sid] = {"mean_z": float(np.mean(zz)), "n_frags": int(len(zz))}
        return out

    summary = {"n_blocks": len(profiles.blocks), "n_cpg": int(len(profiles.cpg_pos)),
               "n_tumour_test": len(te_t), "n_healthy_test": len(te_h), "metrics": metrics,
               "mean_z_by_healthy_sample": _per_sample(h_sc),
               "mean_z_by_tumour_sample": _per_sample(t_sc)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import pandas as pd
    m = merge_scores([t_sc, h_sc])
    n = len(m.llr)
    rng = np.random.default_rng(seed)
    sel = np.arange(n) if n <= 200000 else rng.choice(n, 200000, replace=False)
    reads_df = pd.DataFrame({"z": calibration.z(m.llr, m.n_cpg)[sel], "n_cpg": m.n_cpg[sel],
                             "read_length": m.read_length[sel], "label": m.label[sel],
                             "cohort": m.cohort[sel], "sample_id": m.sample_id[sel]})
    reads_df.to_csv(out_dir / "per_fragment_scores.tsv.gz", sep="\t", index=False)
    if not args.no_plots:
        try:
            from rltf.plots import plot_oracle
            plot_oracle(reads_df, metrics, out_dir / "plots")
        except Exception as exc:  # pragma: no cover
            logger.warning("plotting skipped: %r", exc)

    logger.info("AUC=%.4f permuted=%.4f corr_z_len=%.3f | %s",
                metrics["auc"], metrics["auc_permuted"], metrics["corr_z_read_length"], metrics["verdict"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
