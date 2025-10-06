#!/usr/bin/env python3
"""
TAPESTRY Step 1a: Define Regions

Creates the genomic regions once, saves to CSV.
This is run once before the job array.
"""

import argparse
from pathlib import Path
import yaml
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tapestry.data.regions import RegionDefiner


def main():
    parser = argparse.ArgumentParser(
        description='TAPESTRY: Define genomic regions'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('data/processed/regions.csv'),
        help='Output CSV file for regions'
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
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Defining genomic regions...")

    region_definer = RegionDefiner(
        region_size=config['data']['region_size'],
        region_step=config['data']['region_step']
    )

    regions = region_definer.create_tiling_regions()

    logger.info(f"Defined {len(regions)} regions")
    logger.info(f"Saving to {args.output_file}...")

    regions.to_csv(args.output_file, index=False)

    logger.info("Complete!")


if __name__ == '__main__':
    main()
