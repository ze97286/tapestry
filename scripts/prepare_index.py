#!/usr/bin/env python
"""Build the CpG index and candidate regions (run once before the array job).

This must complete before the per-chromosome preprocessing array job starts.
It produces two shared artefacts that all array tasks read:
  - cpg_index.tsv.gz
  - candidate_regions.bed.gz
"""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from tapestry.core.config import Config
from tapestry.core.cpg_index import (
    build_cpg_index_from_fasta,
    build_cpg_index_from_reads,
    load_cpg_index,
    save_cpg_index,
)
from tapestry.core.io import load_manifest
from tapestry.markers.regions import generate_candidate_regions

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build CpG index and candidate regions.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output_dir = Path(config.project.output_dir) / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.project.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config.save(str(output_dir / "config_resolved.yaml"))

    # --- CpG index ---
    cpg_index_path = output_dir / "cpg_index.tsv.gz"
    if config.data.cpg_index_path and os.path.exists(config.data.cpg_index_path):
        logger.info("Loading CpG index from %s", config.data.cpg_index_path)
        cpg_index_full = load_cpg_index(config.data.cpg_index_path)
        # Keep only chromosomes listed in the config.
        chrom_set = set(config.data.chromosomes)
        cpg_index = {c: v for c, v in cpg_index_full.items() if c in chrom_set}
        save_cpg_index(cpg_index, str(cpg_index_path))
    elif config.data.reference_fasta and os.path.exists(config.data.reference_fasta):
        logger.info("Building CpG index from reference FASTA")
        cpg_index = build_cpg_index_from_fasta(
            config.data.reference_fasta, config.data.chromosomes,
        )
        save_cpg_index(cpg_index, str(cpg_index_path))
    else:
        logger.info("Building CpG index from observed reads")
        manifest = load_manifest(config.data.manifest_path)
        cpg_index = build_cpg_index_from_reads(
            manifest, config.data.chromosomes,
            n_workers=config.preprocessing.n_workers,
        )
        save_cpg_index(cpg_index, str(cpg_index_path))

    logger.info("CpG index: %d chromosomes, %d total sites",
                len(cpg_index),
                sum(len(v) for v in cpg_index.values()))

    # --- Candidate regions ---
    regions_bed_path = output_dir / "candidate_regions.bed.gz"
    all_regions = generate_candidate_regions(
        cpg_index,
        window_size=config.regions.window_size,
        step_size=config.regions.step_size,
        min_cpgs=config.regions.min_cpgs_per_window,
        chromosomes=config.data.chromosomes,
    )
    rows = []
    for r in all_regions:
        rows.append({
            "chrom": r.chrom, "start": r.start, "end": r.end,
            "region_id": r.region_id, "n_cpgs": len(r.cpg_positions),
        })
    regions_df = pd.DataFrame(rows)
    regions_df.to_csv(str(regions_bed_path), sep="\t", index=False)

    logger.info("Generated %d candidate regions → %s", len(all_regions), regions_bed_path)


if __name__ == "__main__":
    main()
