#!/usr/bin/env python
"""Select marker regions from UXM homog counts using 1-vs-all SNR ranking.

Memory-efficient: loads one UXM file at a time and accumulates counts
per block per cell type, rather than loading all 70 files into memory.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BLOCK_COLS = ["chr", "start", "end", "startCpG", "endCpG"]


def load_and_aggregate(homog_dir: Path, manifest_path: Path):
    """Load UXM files one at a time, accumulate per-block per-cell-type counts.

    Returns
    -------
    ct_counts : dict[str, DataFrame]
        Per cell type: DataFrame indexed by block with columns [U, X, M, total, n_samples]
    sample_counts : dict[str, dict[str, DataFrame]]
        Per cell type → per sample: DataFrame indexed by block with columns [U, X, M, total]
    """
    manifest = pd.read_csv(manifest_path, sep="\t")
    sid_to_ct = dict(zip(manifest["sample_id"], manifest["cell_type"]))

    # Accumulate per cell type (summed) and per sample (individual)
    ct_counts = {}      # cell_type → DataFrame
    sample_counts = {}  # cell_type → {sample_id → DataFrame}

    for uxm_file in sorted(homog_dir.glob("*.uxm.bed.gz")):
        sid = uxm_file.name.replace(".uxm.bed.gz", "")
        if sid not in sid_to_ct:
            logger.warning("Skipping %s (not in manifest)", sid)
            continue

        ct = sid_to_ct[sid]
        logger.info("  Loading %s (%s)", sid, ct)

        df = pd.read_csv(
            uxm_file, sep="\t", header=None,
            names=BLOCK_COLS + ["U", "X", "M"],
            dtype={"chr": str, "start": np.int64, "end": np.int64,
                   "startCpG": np.int32, "endCpG": np.int32,
                   "U": np.int32, "X": np.int32, "M": np.int32},
        )
        df["total"] = df["U"] + df["X"] + df["M"]
        df = df.set_index(BLOCK_COLS)

        # Accumulate per cell type
        if ct not in ct_counts:
            ct_counts[ct] = df[["U", "X", "M", "total"]].copy()
            ct_counts[ct]["n_samples"] = 1
        else:
            ct_counts[ct][["U", "X", "M", "total"]] += df[["U", "X", "M", "total"]]
            ct_counts[ct]["n_samples"] += 1

        # Store per-sample for consistency check
        if ct not in sample_counts:
            sample_counts[ct] = {}
        sample_counts[ct][sid] = df[["U", "X", "M", "total"]].copy()

        del df  # free memory

    logger.info("Loaded %d cell types", len(ct_counts))
    for ct, counts in ct_counts.items():
        logger.info("  %s: %d samples", ct, int(counts["n_samples"].iloc[0]))

    return ct_counts, sample_counts


def select_markers(
    ct_counts: dict,
    sample_counts: dict,
    direction: str = "U",
    top_n: int = 250,
    min_snr: float = 3.0,
    min_target_signal: float = 0.3,
    max_bg_signal: float = 0.1,
    min_coverage_per_sample: int = 5,
    min_sample_consistency: float = 0.2,
    max_single_bg: float = 0.2,
) -> pd.DataFrame:
    """Select markers with per-sample consistency checks."""

    cell_types = sorted(ct_counts.keys())
    noise_floor = 0.01
    sig_col = direction  # "U" or "M"

    all_markers = []

    for target_ct in cell_types:
        logger.info("Processing %s...", target_ct)

        target = ct_counts[target_ct].copy()
        target["target_signal"] = target[sig_col] / target["total"].clip(lower=1)

        # Background: sum across all non-target cell types
        bg_parts = [ct_counts[ct][["U", "X", "M", "total"]] for ct in cell_types if ct != target_ct]
        bg = bg_parts[0].copy()
        for part in bg_parts[1:]:
            bg += part
        bg["bg_signal"] = bg[sig_col] / bg["total"].clip(lower=1)

        # Per-sample consistency check
        target_samples = sample_counts[target_ct]
        n_target = len(target_samples)

        # For each block, check all samples have sufficient coverage and signal
        consistent = pd.Series(True, index=target.index)
        for sid, sdf in target_samples.items():
            sample_signal = sdf[sig_col] / sdf["total"].clip(lower=1)
            consistent &= (sdf["total"] >= min_coverage_per_sample)
            consistent &= (sample_signal >= min_sample_consistency)

        # Per-cell-type background check: no single non-target type above threshold
        max_single_bg_signal = pd.Series(0.0, index=target.index)
        for other_ct in cell_types:
            if other_ct == target_ct:
                continue
            other = ct_counts[other_ct]
            other_signal = other[sig_col] / other["total"].clip(lower=1)
            max_single_bg_signal = pd.concat(
                [max_single_bg_signal, other_signal], axis=1,
            ).max(axis=1)

        # Compute SNR
        snr = target["target_signal"] / (bg["bg_signal"] + noise_floor)

        # Build result DataFrame
        result = pd.DataFrame({
            "target_signal": target["target_signal"],
            "bg_signal": bg["bg_signal"],
            "snr": snr,
            "target_total": target["total"],
            "bg_total": bg["total"],
            "consistent": consistent,
            "max_single_bg": max_single_bg_signal,
        })

        # Filter
        mask = (
            result["consistent"]
            & (result["snr"] >= min_snr)
            & (result["target_signal"] >= min_target_signal)
            & (result["bg_signal"] <= max_bg_signal)
            & (result["max_single_bg"] <= max_single_bg)
        )
        filtered = result[mask].copy()

        # Select top N
        selected = filtered.nlargest(top_n, "snr")
        selected["target"] = target_ct
        selected["direction"] = direction

        # Add n_cpgs
        idx = selected.index.to_frame()
        selected["n_cpgs"] = idx["endCpG"].values - idx["startCpG"].values

        n_consistent = int(consistent.sum())
        n_filtered = len(filtered)
        n_selected = len(selected)

        logger.info(
            "  %s: %d consistent → %d after filters → %d selected (SNR %.1f–%.1f)",
            target_ct, n_consistent, n_filtered, n_selected,
            selected["snr"].min() if n_selected > 0 else 0,
            selected["snr"].max() if n_selected > 0 else 0,
        )

        all_markers.append(selected)

    markers = pd.concat(all_markers)
    markers = markers.reset_index()

    # Add per-cell-type signal columns (U-fraction for each cell type)
    sig_col = direction
    for ct in cell_types:
        counts = ct_counts[ct]
        ct_signal = counts[sig_col] / counts["total"].clip(lower=1)
        ct_signal.name = ct
        markers = markers.merge(
            ct_signal.reset_index(),
            on=BLOCK_COLS, how="left",
        )

    out_cols = BLOCK_COLS + [
        "n_cpgs", "target", "direction",
        "target_signal", "bg_signal", "snr",
        "target_total", "bg_total",
    ] + cell_types
    markers = markers[out_cols].sort_values(
        ["target", "snr"], ascending=[True, False],
    )

    return markers


def main():
    parser = argparse.ArgumentParser(
        description="Select marker regions from UXM homog counts.",
    )
    parser.add_argument("--homog-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=250)
    parser.add_argument("--min-snr", type=float, default=3.0)
    parser.add_argument("--min-signal", type=float, default=0.3)
    parser.add_argument("--max-bg", type=float, default=0.1)
    parser.add_argument("--max-single-bg", type=float, default=0.2,
                        help="Max signal in any single non-target cell type")
    parser.add_argument("--min-consistency", type=float, default=0.2)
    parser.add_argument("--min-cov-per-sample", type=int, default=5)
    parser.add_argument("--direction", default="U", choices=["U", "M"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Loading UXM files")
    ct_counts, sample_counts = load_and_aggregate(
        Path(args.homog_dir), Path(args.manifest),
    )

    logger.info("Selecting markers")
    markers = select_markers(
        ct_counts, sample_counts,
        direction=args.direction,
        top_n=args.top_n,
        min_snr=args.min_snr,
        min_target_signal=args.min_signal,
        max_bg_signal=args.max_bg,
        min_coverage_per_sample=args.min_cov_per_sample,
        min_sample_consistency=args.min_consistency,
        max_single_bg=args.max_single_bg,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markers.to_csv(output_path, sep="\t", index=False)

    logger.info("Saved %d markers across %d cell types", len(markers), markers["target"].nunique())


if __name__ == "__main__":
    main()
