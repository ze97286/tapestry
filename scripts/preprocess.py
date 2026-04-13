#!/usr/bin/env python
"""Assign reads to candidate regions for a single chromosome.

Expects prepare_index.py to have already produced:
  - {output_dir}/preprocessing/cpg_index.tsv.gz
  - {output_dir}/preprocessing/candidate_regions.bed.gz

This script loads those shared artefacts, then for the requested chromosome
builds RegionReadMatrix objects and writes them to an HDF5 file.
"""
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tapestry.core.config import Config
from tapestry.core.cpg_index import load_cpg_index
from tapestry.core.datatypes import Region, RegionReadMatrix
from tapestry.core.io import load_manifest, load_reads, save_region_read_matrix
from tapestry.markers.regions import assign_reads_to_regions, build_region_read_matrices

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build RegionReadMatrix HDF5 for one chromosome.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--chrom", required=True, help="Chromosome to process (e.g. chr1)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    prep_dir = Path(config.project.output_dir) / "preprocessing"

    logging.basicConfig(
        level=getattr(logging, config.project.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    chrom = args.chrom

    # --- Load shared artefacts produced by prepare_index.py ---
    cpg_index = load_cpg_index(str(prep_dir / "cpg_index.tsv.gz"))
    regions_df = pd.read_csv(str(prep_dir / "candidate_regions.bed.gz"), sep="\t")

    # Reconstruct Region objects for the requested chromosome only.
    chrom_df = regions_df[regions_df["chrom"] == chrom]
    regions = []
    chrom_cpgs = cpg_index.get(chrom, np.array([], dtype=np.int64))
    for _, row in chrom_df.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        mask = (chrom_cpgs >= start) & (chrom_cpgs < end)
        regions.append(Region(
            region_id=row["region_id"],
            chrom=chrom,
            start=start,
            end=end,
            cpg_positions=chrom_cpgs[mask],
        ))

    if not regions:
        logger.info("No candidate regions on %s, nothing to do", chrom)
        return

    logger.info("%s: %d candidate regions", chrom, len(regions))

    # --- Skip if output already exists ---
    matrices_dir = prep_dir / "region_read_matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    h5_path = matrices_dir / f"{chrom}.h5"
    if h5_path.exists():
        logger.info("%s already exists, skipping", h5_path)
        return

    # --- Process every sample for this chromosome ---
    manifest = load_manifest(config.data.manifest_path)
    ct_to_idx = manifest.cell_type_indices  # derived from manifest

    all_rrms = []
    for sample_idx in range(manifest.n_samples):
        sid = manifest.sample_id[sample_idx]
        ct = manifest.cell_type[sample_idx]
        ct_idx = ct_to_idx.get(ct)
        fpath = manifest.file_path[sample_idx]

        logger.info("  Sample %s (%s)", sid, ct)
        reads = load_reads(
            fpath, cpg_index,
            chrom_filter=chrom,
            min_cpgs=config.preprocessing.min_cpgs_per_read,
        )
        reads_by_region = assign_reads_to_regions(
            reads, regions,
            min_cpgs_overlap=config.markers.min_cpgs_per_read,
        )
        rrms = build_region_read_matrices(
            reads_by_region, regions,
            cell_type_label=ct_idx, sample_id=sid,
        )
        all_rrms.extend(rrms)

    # --- Merge reads across samples per region and write HDF5 ---
    by_region = defaultdict(list)
    for rrm in all_rrms:
        by_region[rrm.region.region_id].append(rrm)

    h5_path = matrices_dir / f"{chrom}.h5"
    with h5py.File(str(h5_path), "w") as f:
        for region_id, rrm_list in by_region.items():
            matrices = [rrm.matrix for rrm in rrm_list]
            counts = [rrm.read_counts for rrm in rrm_list
                      if rrm.read_counts is not None]
            labels = [rrm.read_labels for rrm in rrm_list
                      if rrm.read_labels is not None]
            sids = [rrm.sample_ids for rrm in rrm_list
                    if rrm.sample_ids is not None]

            merged = RegionReadMatrix(
                matrix=np.concatenate(matrices, axis=0),
                cpg_positions=rrm_list[0].cpg_positions,
                read_counts=np.concatenate(counts, axis=0) if counts else None,
                read_labels=np.concatenate(labels, axis=0) if labels else None,
                sample_ids=np.concatenate(sids, axis=0) if sids else None,
                region=rrm_list[0].region,
            )
            grp = f.create_group(region_id)
            save_region_read_matrix(merged, grp)

    logger.info("Saved %s: %d regions", h5_path, len(by_region))


if __name__ == "__main__":
    main()
