#!/usr/bin/env python3
"""Collect batch npz files into final training parquet files.

Reads all batch_*.npz files from the output directory and writes:
  - marker_values.parquet  (samples × markers, U-fractions)
  - coverage.parquet        (samples × markers)
  - ground_truth_y.parquet  (samples × cell_types, proportions)

Usage:
    python scripts/collect_training_data.py \
        --input-dir runs/run_002/training/train \
        --output-dir runs/run_002/training/train
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect batch npz into parquets.")
    parser.add_argument("--input-dir", required=True, help="Directory with batch_*.npz files")
    parser.add_argument("--output-dir", required=True, help="Output directory for parquets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and sort batch files
    batch_files = sorted(input_dir.glob("batch_*.npz"))
    if not batch_files:
        logger.error("No batch_*.npz files found in %s", input_dir)
        return

    logger.info("Found %d batch files", len(batch_files))

    # Load and concatenate
    all_marker_values = []
    all_coverage = []
    all_proportions = []
    cell_types = None

    for f in batch_files:
        data = np.load(f)
        all_marker_values.append(data["marker_values"])
        all_coverage.append(data["coverage"])
        all_proportions.append(data["proportions"])
        if cell_types is None:
            cell_types = data["cell_types"].tolist()
        logger.info("  %s: %d samples", f.name, len(data["marker_values"]))

    marker_values = np.concatenate(all_marker_values, axis=0)
    coverage = np.concatenate(all_coverage, axis=0)
    proportions = np.concatenate(all_proportions, axis=0)

    logger.info("Total: %d samples, %d markers, %d cell types",
                marker_values.shape[0], marker_values.shape[1], len(cell_types))

    # Write parquets
    sample_ids = [f"mix_{i}" for i in range(len(marker_values))]

    mv_df = pd.DataFrame(marker_values, columns=[f"marker_{i}" for i in range(marker_values.shape[1])])
    mv_df.insert(0, "sample_id", sample_ids)
    mv_df.to_parquet(output_dir / "marker_values.parquet", index=False)

    cov_df = pd.DataFrame(coverage, columns=[f"marker_{i}" for i in range(coverage.shape[1])])
    cov_df.insert(0, "sample_id", sample_ids)
    cov_df.to_parquet(output_dir / "coverage.parquet", index=False)

    y_df = pd.DataFrame(proportions, columns=cell_types)
    y_df.insert(0, "sample_id", sample_ids)
    y_df.to_parquet(output_dir / "ground_truth_y.parquet", index=False)

    logger.info("Wrote marker_values.parquet, coverage.parquet, ground_truth_y.parquet to %s", output_dir)

    # Summary statistics
    logger.info("Proportion statistics:")
    for i, ct in enumerate(cell_types):
        col = proportions[:, i]
        n_zero = (col == 0).sum()
        logger.info("  %s: mean=%.4f, median=%.4f, zero=%d (%.1f%%)",
                    ct, col.mean(), np.median(col), n_zero, 100 * n_zero / len(col))


if __name__ == "__main__":
    main()
