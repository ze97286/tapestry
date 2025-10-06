#!/usr/bin/env python3
"""
TAPESTRY Step 1c: Merge Sample Results

Combines individual sample HDF5 files into final matrices.
"""

import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tapestry.data.aggregator import RegionAggregator


def main():
    parser = argparse.ArgumentParser(
        description='TAPESTRY: Merge individual sample results'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing sample HDF5 files'
    )
    parser.add_argument(
        '--regions-file',
        type=Path,
        required=True,
        help='Path to regions CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for final matrices'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/default_config.yaml'),
        help='Configuration file'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("TAPESTRY - Merge Sample Results")
    logger.info("="*60)

    # Load regions
    logger.info("Loading regions...")
    regions = pd.read_csv(args.regions_file)
    n_regions = len(regions)
    logger.info(f"Total regions: {n_regions:,}")

    # Create region ID to index mapping
    region_id_to_idx = {rid: idx for idx, rid in enumerate(regions['region_id'].values)}

    # Find all sample files
    sample_files = sorted(args.input_dir.glob('*.h5'))
    n_samples = len(sample_files)
    logger.info(f"Found {n_samples} sample files")

    if n_samples == 0:
        logger.error(f"No HDF5 files found in {args.input_dir}")
        return

    # Initialize sparse matrices
    logger.info("Loading sample data...")
    methylation_matrix = np.zeros((n_samples, n_regions), dtype=np.float32)
    coverage_matrix = np.zeros((n_samples, n_regions), dtype=np.float32)
    sample_ids = []

    for sample_idx, sample_file in enumerate(tqdm(sample_files, desc="Loading samples")):
        with h5py.File(sample_file, 'r') as f:
            sample_id = f.attrs['sample_id']
            sample_ids.append(sample_id)

            region_ids = f['region_ids'][:].astype(str)
            mod_counts = f['mod_counts'][:]
            coverage = f['coverage'][:]

            # Map to full matrix
            for rid, mod, cov in zip(region_ids, mod_counts, coverage):
                if rid in region_id_to_idx:
                    idx = region_id_to_idx[rid]
                    methylation_matrix[sample_idx, idx] = mod
                    coverage_matrix[sample_idx, idx] = cov

    logger.info(f"Loaded matrix shape: {methylation_matrix.shape}")

    # Filter by coverage
    logger.info("Filtering regions...")
    min_samples_covered = config['data']['min_samples_covered']
    samples_per_region = (coverage_matrix > 0).sum(axis=0)
    kept_mask = samples_per_region >= min_samples_covered

    logger.info(
        f"Keeping {kept_mask.sum():,} / {len(kept_mask):,} regions "
        f"(covered in ≥{min_samples_covered} samples)"
    )

    meth_filtered = methylation_matrix[:, kept_mask]
    cov_filtered = coverage_matrix[:, kept_mask]
    regions_kept = regions[kept_mask].reset_index(drop=True)

    # Filter by variance
    logger.info("Selecting most variable regions...")
    aggregator = RegionAggregator(
        min_cpgs_per_region=config['filtering']['min_cpgs_per_region'],
        min_region_coverage=config['data']['min_region_coverage']
    )

    meth_final, cov_final, regions_final = aggregator.filter_by_variance(
        meth_filtered,
        cov_filtered,
        regions_kept,
        top_k=config['data']['max_regions'],
        min_variance=config['filtering']['min_variance']
    )

    logger.info(f"Final data shape: {meth_final.shape}")

    # Save results
    logger.info("Saving final results...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.output_dir / 'methylation_matrix.npy', meth_final)
    np.save(args.output_dir / 'coverage_matrix.npy', cov_final)
    regions_final.to_csv(args.output_dir / 'regions.csv', index=False)
    pd.DataFrame({'sample_id': sample_ids}).to_csv(
        args.output_dir / 'sample_ids.csv', index=False
    )

    logger.info("="*60)
    logger.info("Complete!")
    logger.info(f"Output saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
