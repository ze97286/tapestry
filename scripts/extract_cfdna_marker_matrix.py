#!/usr/bin/env python3
"""Extract cfDNA UXM marker-value and coverage matrices.

This is the extraction-only front end for marker selection.  It takes a
candidate atlas and cfDNA PAT files, applies the same marker/read filtering
path used by deconvolution, runs wgbstools homog, and writes:

  * marker_values.tsv: marker rows x sample columns, UXM U-fraction
  * coverage.tsv: marker rows x sample columns, U+M coverage

No NNLS or unknown-channel fitting is run here.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from predict_cfdna_augmented import (
    META_COLS,
    load_atlas,
    process_cfdna_sample,
    validate_markers_bed,
    write_marker_matrix,
)


logger = logging.getLogger(__name__)


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def infer_cell_types(atlas_path: str, explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    header = pd.read_csv(atlas_path, sep="\t", nrows=0)
    return sorted([col for col in header.columns if col not in META_COLS])


def sample_id(path: Path) -> str:
    return path.name.removesuffix(".pat.gz").removesuffix(".markers")


def write_markers_bed(atlas_df: pd.DataFrame, markers_bed: str) -> None:
    required = ["chr", "start", "end", "startCpG", "endCpG"]
    missing = [col for col in required if col not in atlas_df.columns]
    if missing:
        raise ValueError(f"missing columns for marker BED: {missing}")
    bed = atlas_df[required].sort_values(["startCpG", "chr", "start"], kind="mergesort")
    output = Path(markers_bed)
    output.parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(output, sep="\t", header=False, index=False)
    logger.info("Wrote %d candidate regions to %s", len(bed), output)


def collect_pat_files(args: argparse.Namespace) -> list[Path]:
    paths = sorted(Path(args.cfdna_dir).glob("*.pat.gz"))
    extra_pattern = args.extra_control_pattern
    for extra_dir in args.extra_control_dir:
        for path in sorted(Path(extra_dir).glob("*.pat.gz")):
            sid = sample_id(path)
            if extra_pattern and not args.extra_control_re.search(sid):
                continue
            paths.append(path)

    unique = sorted(set(paths))
    ids = [sample_id(path) for path in unique]
    duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
    if duplicates:
        raise ValueError(f"duplicate sample IDs in input PAT list: {', '.join(duplicates)}")
    return unique


def filter_pat_to_markers(
    pat_path: Path,
    output_path: Path,
    markers_bed: str,
    wgbstools: str,
    cview_args: str,
    force: bool,
) -> Path:
    if output_path.exists() and not force:
        logger.info("Skipping existing %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    cmd = (
        f"{shlex.quote(wgbstools)} cview {shlex.quote(str(pat_path))} "
        f"-L {shlex.quote(markers_bed)} {cview_args} "
        f"| bgzip > {shlex.quote(str(tmp_path))}"
    )
    logger.info("Filtering %s", sample_id(pat_path))
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", cmd],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cview failed for {pat_path}: stdout={result.stdout[-1000:]!r} "
            f"stderr={result.stderr[-1000:]!r}"
        )
    tmp_path.replace(output_path)
    subprocess.run(
        ["tabix", "-f", "-s", "1", "-b", "2", "-e", "2", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract UXM marker values and coverage from cfDNA PAT files.",
    )
    parser.add_argument("--cfdna-dir", required=True)
    parser.add_argument("--extra-control-dir", action="append", default=[])
    parser.add_argument("--extra-control-pattern", default="")
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--markers-bed", required=True)
    parser.add_argument("--filtered-dir", required=True)
    parser.add_argument("--marker-values-output", required=True)
    parser.add_argument("--coverage-output", required=True)
    parser.add_argument("--sample-manifest-output", default=None)
    parser.add_argument("--wgbstools", default="wgbstools")
    parser.add_argument("--cview-args", default="")
    parser.add_argument("--homog-len", type=int, default=4)
    parser.add_argument("--force-refilter", action="store_true")
    parser.add_argument("--cell-types", default=None)
    args = parser.parse_args()

    import re

    args.extra_control_re = re.compile(args.extra_control_pattern) if args.extra_control_pattern else None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cell_types = infer_cell_types(args.atlas, parse_csv(args.cell_types))
    logger.info("Cell types (%d): %s", len(cell_types), cell_types)

    atlas_matrix, valid_indices, atlas_coords, total_rows, atlas_valid_df = load_atlas(
        args.atlas, cell_types
    )
    if len(valid_indices) < total_rows:
        logger.info("Atlas filter: kept %d / %d rows", atlas_matrix.shape[0], total_rows)

    write_markers_bed(atlas_valid_df, args.markers_bed)
    validate_markers_bed(args.markers_bed)

    pat_files = collect_pat_files(args)
    logger.info("Found %d cfDNA PAT files", len(pat_files))
    if not pat_files:
        raise SystemExit("No PAT files found")

    filtered_dir = Path(args.filtered_dir)
    filtered_paths = []
    manifest_rows = []
    for pat_path in pat_files:
        sid = sample_id(pat_path)
        out_path = filtered_dir / f"{sid}.markers.pat.gz"
        filtered_path = filter_pat_to_markers(
            pat_path=pat_path,
            output_path=out_path,
            markers_bed=args.markers_bed,
            wgbstools=args.wgbstools,
            cview_args=args.cview_args,
            force=args.force_refilter,
        )
        filtered_paths.append(filtered_path)
        manifest_rows.append({
            "sample": sid,
            "source_pat": str(pat_path),
            "filtered_pat": str(filtered_path),
        })

    manifest_output = (
        Path(args.sample_manifest_output)
        if args.sample_manifest_output
        else Path(args.marker_values_output).with_name("sample_manifest.tsv")
    )
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_output, sep="\t", index=False)
    logger.info("Saved sample manifest to %s", manifest_output)

    tmp_parent = Path(args.marker_values_output).resolve().parent / ".tmp_homog"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    sample_names: list[str] = []
    marker_values: list[np.ndarray] = []
    coverages: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="tapestry_extract_", dir=tmp_parent) as tmp_dir:
        for i, filtered_path in enumerate(filtered_paths):
            sid = sample_id(filtered_path)
            logger.info("[%d/%d] homog %s", i + 1, len(filtered_paths), sid)
            result = process_cfdna_sample(
                str(filtered_path),
                args.markers_bed,
                args.wgbstools,
                atlas_coords,
                tmp_dir,
                args.homog_len,
            )
            if result is None:
                continue
            u_fraction, coverage = result
            sample_names.append(sid)
            marker_values.append(u_fraction)
            coverages.append(coverage)
            for path in Path(tmp_dir).glob("*.uxm.bed.gz"):
                path.unlink()

    if not sample_names:
        raise SystemExit("No samples were successfully processed by homog")

    X = np.asarray(marker_values, dtype=np.float32)
    coverage = np.asarray(coverages, dtype=np.float32)
    write_marker_matrix(args.marker_values_output, atlas_valid_df, sample_names, X)
    write_marker_matrix(args.coverage_output, atlas_valid_df, sample_names, coverage)
    logger.info(
        "Done. Wrote %d markers x %d samples",
        X.shape[1],
        X.shape[0],
    )


if __name__ == "__main__":
    main()
