#!/usr/bin/env python3
"""Step 1 — discover the marker panel from reference per-read calls.

Config-driven (``--config``). Materialises the manifest + copies the config into
the panel dir for provenance. ``--chrom`` restricts to one chromosome (for
sharding the genome-wide reference pass; merge the partials separately).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from rltf.config import build_manifest_rows, get, load_config, write_manifest_tsv
from rltf.discovery import discover_panel, merge_partials
from rltf.manifest import read_manifest

logger = logging.getLogger("discover_panel")
DEFAULT_CONFIG = "readlevel_tf/configs/config.toml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--chrom", default=None, help="discover one chromosome -> a partial (sharding)")
    ap.add_argument("--merge", action="store_true", help="merge per-chromosome partials into the final panel")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    base = Path(get(cfg, "paths.run_dir", "readlevel_tf/runs")) / "panel"
    top_n = get(cfg, "discovery.top_n", 2000)

    if args.merge:
        partial_dirs = sorted(str(d) for d in (base / "partials").glob("*") if (d / "cpgs.tsv").exists())
        if not partial_dirs:
            raise SystemExit(f"No partials under {base}/partials — run the per-chromosome shards first.")
        logger.info("Merging %d chromosome partials into top-%d panel", len(partial_dirs), top_n)
        profiles = merge_partials(partial_dirs, top_n=top_n)
        out_dir = base
    else:
        out_dir = (base / "partials" / args.chrom) if args.chrom else base
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest_tsv(build_manifest_rows(cfg), out_dir / "manifest.tsv")
        shutil.copy(args.config, out_dir / "config.toml")
        ref = [r for r in read_manifest(out_dir / "manifest.tsv") if r["role"] == "reference"]
        tumour = [r["file_path"] for r in ref if r["group"] == "tumour"]
        healthy = [r["file_path"] for r in ref if r["group"] == "healthy"]
        logger.info("Discovering from %d tumour + %d healthy reference samples%s",
                    len(tumour), len(healthy), f" (chrom {args.chrom}, partial)" if args.chrom else "")
        profiles = discover_panel(
            tumour, healthy,
            window=get(cfg, "discovery.window", 5),
            top_n=None if args.chrom else top_n,   # shards keep all surviving; merge ranks globally
            min_total=get(cfg, "discovery.min_total", 10), min_effect=get(cfg, "discovery.min_effect", 0.3),
            direction=get(cfg, "discovery.direction", "any"), min_mapq=get(cfg, "scoring.min_mapq", 30),
            cross_fit=get(cfg, "discovery.cross_fit", True),
            chroms={args.chrom} if args.chrom else None,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    profiles.save(out_dir)
    summary = {"n_blocks": len(profiles.blocks), "n_cpg": int(len(profiles.cpg_pos)),
               "n_tumour_samples": profiles.n_tumour_samples, "n_healthy_samples": profiles.n_healthy_samples,
               "chrom": args.chrom, "merged": args.merge, "discovery_params": cfg.get("discovery", {})}
    (out_dir / "discovery_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_plots:
        try:
            import pandas as pd
            from rltf.plots import plot_discovery
            plot_discovery(pd.read_csv(out_dir / "cpgs.tsv", sep="\t"), out_dir / "plots")
        except Exception as exc:  # pragma: no cover
            logger.warning("plotting skipped: %r", exc)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
