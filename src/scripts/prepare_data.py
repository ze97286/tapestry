#!/usr/bin/env python3
"""
TAPESTRY Step 1: Prepare Data

Load TAPS samples and aggregate to regions.
"""

import argparse
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tapestry.data.loader import TAPSLoader
from tapestry.data.regions import RegionDefiner
from tapestry.data.aggregator import RegionAggregator


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tapestry_prepare.log')
        ]
    )

# python scripts/01_prepare_data.py \
# --input-dir data/raw \
# --output-dir data/processed \
# --pattern "*.bed" \
# --metadata data/metadata.csv
def main():
    parser = argparse.ArgumentParser(
        description='TAPESTRY: Prepare data from TAPS samples'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing TAPS sample files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/default_config.yaml'),
        help='Configuration file'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.calls.bed.gz',
        help='File pattern to match (e.g., *.bed, *.tsv)'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        help='CSV file with sample metadata (sample_id, patient_id, etc.)'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("TAPESTRY - Data Preparation")
    logger.info("="*60)
    
    # Step 1: Load samples
    logger.info("\nStep 1: Loading TAPS samples...")
    
    sample_files = sorted(args.input_dir.glob(args.pattern))
    
    if len(sample_files) == 0:
        logger.error(f"No files found matching {args.pattern} in {args.input_dir}")
        return
    
    logger.info(f"Found {len(sample_files)} sample files")
    
    loader = TAPSLoader(
        min_coverage=config['data']['min_cpg_coverage'],
        context_filter=config['data']['cpg_context_filter']
    )
    
    cohort_data = loader.load_cohort(sample_files)
    
    # Get coverage statistics
    stats = loader.get_coverage_stats(cohort_data)
    stats.to_csv(args.output_dir / 'sample_stats.csv', index=False)
    logger.info(f"Sample statistics saved to sample_stats.csv")
    
    # Step 2: Define regions
    logger.info("\nStep 2: Defining genomic regions...")
    
    region_definer = RegionDefiner(
        region_size=config['data']['region_size'],
        region_step=config['data']['region_step']
    )
    
    regions = region_definer.create_tiling_regions()
    
    logger.info(f"Defined {len(regions)} candidate regions")
    
    # Step 3: Aggregate to regions
    logger.info("\nStep 3: Aggregating CpGs to regions...")
    
    aggregator = RegionAggregator(
        min_cpgs_per_region=config['filtering']['min_cpgs_per_region'],
        min_region_coverage=config['data']['min_region_coverage']
    )
    
    meth_matrix, cov_matrix, regions_kept, sample_ids = aggregator.aggregate_cohort(
        cohort_data,
        regions,
        min_samples_covered=config['data']['min_samples_covered']
    )
    
    logger.info(f"Aggregated data shape: {meth_matrix.shape}")
    
    # Step 4: Filter by variance
    logger.info("\nStep 4: Selecting most variable regions...")
    
    meth_final, cov_final, regions_final = aggregator.filter_by_variance(
        meth_matrix,
        cov_matrix,
        regions_kept,
        top_k=config['data']['max_regions'],
        min_variance=config['filtering']['min_variance']
    )
    
    logger.info(f"Final data shape: {meth_final.shape}")
    
    # Step 5: Save processed data
    logger.info("\nStep 5: Saving processed data...")
    
    # Save matrices
    np.save(args.output_dir / 'methylation_matrix.npy', meth_final)
    np.save(args.output_dir / 'coverage_matrix.npy', cov_final)
    
    # Save regions
    regions_final.to_csv(args.output_dir / 'regions.csv', index=False)
    
    # Save sample IDs
    pd.DataFrame({'sample_id': sample_ids}).to_csv(
        args.output_dir / 'sample_ids.csv', index=False
    )
    
    # Load and save metadata if provided
    if args.metadata:
        metadata = pd.read_csv(args.metadata)
        # Ensure order matches sample_ids
        metadata = metadata.set_index('sample_id').loc[sample_ids].reset_index()
        metadata.to_csv(args.output_dir / 'metadata.csv', index=False)
        logger.info("Metadata saved")
    
    logger.info("\n" + "="*60)
    logger.info("Data preparation complete!")
    logger.info(f"Output saved to: {args.output_dir}")
    logger.info("="*60)
    
    # Print summary
    print("\nSummary:")
    print(f"  Samples: {meth_final.shape[0]}")
    print(f"  Regions: {meth_final.shape[1]}")
    print(f"  Mean coverage per region: {cov_final.mean():.1f}")
    print(f"  Mean methylation rate: {(meth_final.sum() / cov_final.sum()):.3f}")


if __name__ == '__main__':
    main()