#!/usr/bin/env python3
"""Generate synthetic cfDNA mixtures from filtered reference PAT files.

For each mixture defined in a proportions CSV:
  1. Sample reads from each cell type's reference PAT using pattools
  2. Merge sampled reads into a single mixture PAT
  3. Run wgbstools homog to get per-marker U/X/M counts
  4. Extract U-fraction and coverage per marker

Designed to run as a SLURM array job processing batches of mixtures.

Usage:
    python scripts/generate_mixtures.py \
        --proportions runs/run_002/training/train_proportions.csv \
        --filtered-dir runs/run_002/filtered_pats/ref \
        --markers-bed runs/run_002/markers/markers.bed \
        --output-dir runs/run_002/training/train \
        --batch-start 0 \
        --batch-size 1000 \
        --threads 4
"""

import argparse
import gzip
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def count_reads_in_pat(pat_path: str) -> int:
    """Count total reads (sum of count column) in a PAT file."""
    total = 0
    opener = gzip.open if pat_path.endswith(".gz") else open
    with opener(pat_path, "rt") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) >= 4:
                total += int(parts[3])
    return total


def sample_and_merge(
    proportions: dict[str, float],
    ref_samples: dict[str, str],
    target_depth: int,
    filtered_dir: Path,
    read_counts: dict[str, int],
    tmp_dir: str,
    pattools: str,
    mixture_id: str,
) -> str | None:
    """Sample reads from reference PATs and merge into a mixture.

    Returns path to the merged PAT file, or None if no reads sampled.
    """
    sampled_files = []

    for cell_type, proportion in proportions.items():
        if proportion <= 0:
            continue

        ref_sample = ref_samples[cell_type]
        ref_path = filtered_dir / f"{ref_sample}.markers.pat.gz"

        if not ref_path.exists():
            logger.warning("Reference PAT not found: %s", ref_path)
            continue

        # Compute sampling fraction
        ref_reads = read_counts.get(ref_sample, 0)
        if ref_reads == 0:
            logger.warning("Zero reads in reference %s", ref_sample)
            continue

        target_reads = proportion * target_depth
        sampling_fraction = min(target_reads / ref_reads, 1.0)

        if sampling_fraction <= 0:
            continue

        # Sample using pattools
        sampled_path = os.path.join(tmp_dir, f"{mixture_id}_{cell_type}.pat.gz")
        cmd = f'{pattools} sample -s {sampling_fraction:.8f} {ref_path} | gzip > {sampled_path}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("pattools sample failed for %s: %s", cell_type, result.stderr)
            continue

        sampled_files.append(sampled_path)

    if not sampled_files:
        return None

    # Merge: zcat all | sort | bgzip, then tabix index
    merged_path = os.path.join(tmp_dir, f"{mixture_id}.pat.gz")
    cmd = (
        f'zcat {" ".join(sampled_files)} '
        f'| sort -k1,1V -k2,2n -k3,3 '
        f'| bgzip > {merged_path} '
        f'&& tabix -s 1 -b 2 -e 2 {merged_path}'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Merge failed for %s: %s", mixture_id, result.stderr)
        return None

    return merged_path


def run_homog(
    pat_path: str,
    markers_bed: str,
    wgbstools: str,
    output_path: str,
) -> bool:
    """Run wgbstools homog on a mixture PAT to get per-marker U/X/M counts."""
    out_dir = os.path.dirname(output_path)
    cmd = f'{wgbstools} homog -b {markers_bed} -l 4 -o {out_dir} {pat_path}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("homog failed: %s", result.stderr)
        return False

    # wgbstools names output by input filename; find and rename
    pat_basename = os.path.basename(pat_path).replace(".pat.gz", "")
    expected = os.path.join(out_dir, f"{pat_basename}.uxm.bed.gz")
    if os.path.exists(expected) and expected != output_path:
        os.rename(expected, output_path)
    return True


def extract_marker_values(
    homog_path: str, atlas_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract U-fraction and coverage from a homog UXM file, aligned to atlas order.

    The homog output is in genomic order, but the atlas (markers.tsv) is sorted
    by cell type. This function matches homog rows to atlas rows by (chr, start)
    coordinates to ensure correct alignment.

    Parameters
    ----------
    homog_path : str
        Path to the .uxm.bed.gz file from wgbstools homog.
    atlas_coords : ndarray of shape (M, 2)
        Pre-built lookup: atlas_coords[i] = (chr_hash, start) for atlas row i.
        Use ``build_atlas_coord_index()`` to create this.

    Returns
    -------
    (u_fraction, coverage) arrays of length M aligned to atlas row order,
    or None on failure.
    """
    try:
        opener = gzip.open if homog_path.endswith(".gz") else open
        # Read homog output keyed by (chr, start)
        homog_data = {}
        with opener(homog_path, "rt") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) >= 8:
                    key = (parts[0], int(parts[1]))
                    u = int(parts[5])
                    m_count = int(parts[7])
                    homog_data[key] = (u, m_count)

        if not homog_data:
            return None

        M = len(atlas_coords)
        u_fraction = np.zeros(M, dtype=np.float64)
        coverage = np.zeros(M, dtype=np.float64)

        for i, (chrom, start) in enumerate(atlas_coords):
            key = (chrom, start)
            if key in homog_data:
                u, m_count = homog_data[key]
                total = u + m_count
                coverage[i] = total
                if total > 0:
                    u_fraction[i] = u / total

        return u_fraction, coverage

    except Exception as e:
        logger.error("Failed to parse %s: %s", homog_path, e)
        return None


def build_atlas_coord_index(atlas_path: str) -> list[tuple[str, int]]:
    """Build (chr, start) coordinate list from the atlas TSV, preserving its row order.

    The atlas (markers.tsv) is sorted by cell type, not genomic position.
    The homog output is in genomic order. This index allows us to map
    homog rows back to atlas rows by coordinate matching.
    """
    coords = []
    with open(atlas_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip().split("\t")
            coords.append((parts[0], int(parts[1])))
    return coords


def process_batch(
    proportions_df: pd.DataFrame,
    cell_types: list[str],
    filtered_dir: Path,
    markers_bed: str,
    atlas_path: str,
    output_dir: Path,
    pattools: str,
    wgbstools: str,
    read_counts: dict[str, int],
    batch_id: int,
):
    """Process a batch of mixtures. Writes results to a single npz file."""
    n = len(proportions_df)
    all_fractions = []
    all_coverages = []
    all_proportions = []
    valid_indices = []

    # Build coordinate index for aligning homog output to atlas row order
    atlas_coords = build_atlas_coord_index(atlas_path)

    with tempfile.TemporaryDirectory(prefix=f"tapestry_mix_{batch_id}_") as tmp_dir:
        for i, (idx, row) in enumerate(proportions_df.iterrows()):
            mixture_id = f"b{batch_id}_s{i}"

            # Extract proportions and reference samples
            props = {ct: row[ct] for ct in cell_types}
            refs = {ct: row[f"ref_{ct}"] for ct in cell_types}
            depth = int(row["depth"])

            # Sample and merge
            merged = sample_and_merge(
                props, refs, depth, filtered_dir, read_counts,
                tmp_dir, pattools, mixture_id,
            )
            if merged is None:
                logger.warning("Skipping mixture %s: no reads sampled", mixture_id)
                continue

            # Run homog
            homog_path = os.path.join(tmp_dir, f"{mixture_id}.uxm.bed.gz")
            if not run_homog(merged, markers_bed, wgbstools, homog_path):
                logger.warning("Skipping mixture %s: homog failed", mixture_id)
                continue

            # Extract marker values, aligned to atlas row order
            result = extract_marker_values(homog_path, atlas_coords)
            if result is None:
                logger.warning("Skipping mixture %s: extraction failed", mixture_id)
                continue

            u_fraction, coverage = result
            all_fractions.append(u_fraction)
            all_coverages.append(coverage)
            all_proportions.append([row[ct] for ct in cell_types])
            valid_indices.append(idx)

            if (i + 1) % 100 == 0:
                logger.info("Batch %d: processed %d/%d mixtures", batch_id, i + 1, n)

    if not all_fractions:
        logger.error("Batch %d: no valid mixtures produced", batch_id)
        return

    # Save batch results
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"batch_{batch_id}.npz",
        marker_values=np.array(all_fractions),
        coverage=np.array(all_coverages),
        proportions=np.array(all_proportions),
        cell_types=np.array(cell_types),
        indices=np.array(valid_indices),
    )
    logger.info("Batch %d: saved %d mixtures to batch_%d.npz",
                batch_id, len(all_fractions), batch_id)


def build_read_count_cache(filtered_dir: Path, manifest: pd.DataFrame) -> dict[str, int]:
    """Count total reads in each filtered reference PAT file.

    Results are cached to a TSV file to avoid re-counting on subsequent runs.
    """
    cache_path = filtered_dir / "read_counts.tsv"
    if cache_path.exists():
        logger.info("Loading cached read counts from %s", cache_path)
        df = pd.read_csv(cache_path, sep="\t")
        return dict(zip(df["sample_id"], df["total_reads"]))

    logger.info("Counting reads in filtered PAT files...")
    counts = {}
    for _, row in manifest.iterrows():
        sid = row["sample_id"]
        pat_path = filtered_dir / f"{sid}.markers.pat.gz"
        if pat_path.exists():
            n = count_reads_in_pat(str(pat_path))
            counts[sid] = n
            logger.info("  %s: %d reads", sid, n)
        else:
            logger.warning("  %s: file not found", sid)
            counts[sid] = 0

    # Cache
    pd.DataFrame([
        {"sample_id": k, "total_reads": v} for k, v in counts.items()
    ]).to_csv(cache_path, sep="\t", index=False)
    logger.info("Cached read counts to %s", cache_path)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic mixtures.")
    parser.add_argument("--proportions", required=True, help="Path to proportions CSV")
    parser.add_argument("--manifest", required=True, help="Path to manifest_atlas.tsv")
    parser.add_argument("--filtered-dir", required=True, help="Directory with filtered ref PATs")
    parser.add_argument("--markers-bed", required=True, help="Path to markers.bed")
    parser.add_argument("--atlas", required=True, help="Path to markers.tsv (atlas)")
    parser.add_argument("--output-dir", required=True, help="Output directory for batch npz files")
    parser.add_argument("--pattools", default="pattools", help="Path to pattools binary")
    parser.add_argument("--wgbstools", default="wgbstools", help="Path to wgbstools binary")
    parser.add_argument("--batch-start", type=int, default=0, help="Starting row index")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per batch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load proportions
    proportions_df = pd.read_csv(args.proportions)
    manifest = pd.read_csv(args.manifest, sep="\t")
    cell_types = sorted(manifest["cell_type"].unique())

    # Slice to this batch
    batch_df = proportions_df.iloc[args.batch_start : args.batch_start + args.batch_size]
    if batch_df.empty:
        logger.info("No rows in this batch range, exiting")
        return

    batch_id = args.batch_start // args.batch_size
    logger.info("Processing batch %d: rows %d-%d (%d mixtures)",
                batch_id, args.batch_start, args.batch_start + len(batch_df) - 1, len(batch_df))

    # Build read count cache
    filtered_dir = Path(args.filtered_dir)
    read_counts = build_read_count_cache(filtered_dir, manifest)

    process_batch(
        batch_df,
        cell_types,
        filtered_dir,
        args.markers_bed,
        args.atlas,
        Path(args.output_dir),
        args.pattools,
        args.wgbstools,
        read_counts,
        batch_id,
    )


if __name__ == "__main__":
    main()
