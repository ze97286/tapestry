#!/usr/bin/env python3
"""
TAPESTRY Step 1a: Process Single Sample

Aggregates one sample to regions and saves to HDF5.
This is designed to run as part of an SGE job array.
"""

import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import h5py
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tapestry.data.loader import TAPSLoader
from tapestry.data.regions import RegionDefiner
from tapestry.data.aggregator import RegionAggregator


def setup_logging():
    """Configure logging to stdout only (SGE captures to log file)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description='TAPESTRY: Process single sample to regions'
    )
    parser.add_argument(
        '--sample-file',
        type=Path,
        required=True,
        help='Path to single sample file'
    )
    parser.add_argument(
        '--regions-file',
        type=Path,
        required=True,
        help='Path to regions CSV file (from prepare_regions.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for HDF5 file'
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

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample ID from filename
    sample_id = args.sample_file.stem.replace('.calls.bed', '')

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info(f"Processing sample: {sample_id}")
    logger.info("="*60)

    # Load regions
    logger.info("Loading regions...")
    import pandas as pd
    regions = pd.read_csv(args.regions_file)
    logger.info(f"Loaded {len(regions)} regions")

    # Load sample
    logger.info(f"Loading sample from {args.sample_file}...")
    loader = TAPSLoader(
        min_coverage=config['data']['min_cpg_coverage']
        # Note: context_filter removed - files are pre-filtered to CpG only
    )

    cpg_data = loader.load_sample(args.sample_file)
    logger.info(f"Loaded {len(cpg_data):,} CpG sites")

    # Aggregate to regions
    logger.info("Aggregating to regions...")
    aggregator = RegionAggregator(
        min_cpgs_per_region=config['filtering']['min_cpgs_per_region'],
        min_region_coverage=config['data']['min_region_coverage']
    )

    sample_regions = aggregator.aggregate_sample(cpg_data, regions)
    logger.info(f"Covered {len(sample_regions):,} regions")

    # Save to HDF5
    output_file = args.output_dir / f'{sample_id}.h5'
    logger.info(f"Saving to {output_file}...")

    with h5py.File(output_file, 'w') as f:
        # Save as sparse: only store non-zero regions
        f.create_dataset('region_ids', data=sample_regions['region_id'].values.astype('S50'))
        f.create_dataset('mod_counts', data=sample_regions['mod_count'].values.astype('int32'))
        f.create_dataset('coverage', data=sample_regions['coverage'].values.astype('int32'))
        f.attrs['sample_id'] = sample_id
        f.attrs['n_regions_covered'] = len(sample_regions)

    logger.info("="*60)
    logger.info("Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
