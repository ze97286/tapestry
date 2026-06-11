#!/usr/bin/env python3
"""Step 1 — discover the project's own marker panel from reference PATs.

Reads the manifest's ``role=reference`` rows (group=tumour tissue, group=healthy
cfDNA controls), discovers discriminative CpG-dense blocks in global CpG-index
space, and saves the panel + per-CpG profiles to ``--out-dir``
(``blocks.tsv`` + ``profiles.npz``). No other method's artifacts are used.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rltf.discovery import discover_panel
from rltf.manifest import read_manifest

logger = logging.getLogger("discover_panel")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--window", type=int, default=5, help="block width in CpGs")
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--min-total", type=float, default=10.0, help="min reads per CpG in BOTH groups")
    ap.add_argument("--min-effect", type=float, default=0.3, help="min mean per-CpG |p_tumour - p_healthy|")
    ap.add_argument("--direction", default="any", choices=["any", "hypo", "hyper"])
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = read_manifest(args.manifest)
    ref = [r for r in rows if r["role"] == "reference"]
    tumour = [(r["file_path"], r["convention"]) for r in ref if r["group"] == "tumour"]
    healthy = [(r["file_path"], r["convention"]) for r in ref if r["group"] == "healthy"]
    if not tumour or not healthy:
        raise SystemExit("Need reference rows for both group=tumour and group=healthy.")
    logger.info("Discovering from %d tumour + %d healthy reference samples", len(tumour), len(healthy))

    profiles = discover_panel(
        tumour, healthy, window=args.window, top_n=args.top_n,
        min_total=args.min_total, min_effect=args.min_effect, direction=args.direction, prior=args.prior,
    )
    profiles.save(args.out_dir)
    summary = {
        "n_blocks": len(profiles.blocks),
        "window": args.window,
        "n_tumour_samples": profiles.n_tumour_samples,
        "n_healthy_samples": profiles.n_healthy_samples,
        "params": {"top_n": args.top_n, "min_total": args.min_total,
                   "min_effect": args.min_effect, "direction": args.direction},
    }
    (args.out_dir / "discovery_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_plots:
        try:
            import pandas as pd

            from rltf.plots import plot_discovery
            blocks_df = pd.read_csv(args.out_dir / "blocks.tsv", sep="\t")
            plot_discovery(blocks_df, args.out_dir / "plots")
        except Exception as exc:  # pragma: no cover - plotting is non-fatal
            logger.warning("plotting skipped: %r", exc)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
