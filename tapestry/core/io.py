"""
I/O utilities for the Tapestry cfDNA methylation deconvolution pipeline.

Provides streaming read loading from gzipped TSV, manifest loading,
validation helpers, and HDF5 serialisation of :class:`RegionReadMatrix`.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np

from tapestry.core.datatypes import (
    Read,
    Region,
    RegionReadMatrix,
    SampleManifest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read loading (PAT format)
# ---------------------------------------------------------------------------

def load_reads(
    path: str | Path,
    cpg_index: dict[str, np.ndarray],
    chrom_filter: Optional[str] = None,
    min_cpgs: int = 1,
) -> Iterator[Read]:
    """Stream reads from a PAT file.

    PAT format (tab-separated, no header or ``#``-prefixed header):
        1. chrom              – chromosome name
        2. start_cpg_index    – 1-based global CpG index of the first CpG
        3. pattern            – string of ``C`` (unmethylated) / ``T`` (methylated)
        4. count              – number of reads with this pattern

    Genomic positions are resolved via *cpg_index* (from ``CpG.bed.gz``).

    Parameters
    ----------
    path : str or Path
        Path to the ``.pat.gz`` or ``.tsv.gz`` PAT file.
    cpg_index : dict[str, np.ndarray]
        Mapping from chromosome name to sorted int64 array of genomic CpG
        positions, as returned by :func:`load_cpg_index`.
    chrom_filter : str, optional
        If provided, only yield reads on this chromosome.
    min_cpgs : int
        Minimum pattern length to be yielded.

    Yields
    ------
    Read
        One :class:`Read` per qualifying line.  Collapsed patterns with
        ``count > 1`` are yielded once; the caller should use ``read.count``
        for weighting or expansion.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    # Build a per-chromosome lookup: global CpG index → position offset.
    # The CpG.bed.gz lists CpGs genome-wide with a 1-based global index in
    # column 3.  We need to map (chrom, global_idx) → genomic position.
    # However, we only receive the per-chrom position arrays here, so we
    # need to know where each chromosome's CpG indices start.
    #
    # Strategy: accumulate chromosome offsets so that
    #   global_idx - chrom_offset = local index into cpg_index[chrom]
    chrom_offsets: dict[str, int] = {}
    running = 0
    for chrom_name in sorted(cpg_index.keys(), key=_chrom_sort_key):
        chrom_offsets[chrom_name] = running
        running += len(cpg_index[chrom_name])

    with opener(path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            chrom = parts[0]
            start_global = int(parts[1])   # 1-based global CpG index
            pattern = parts[2]
            count = int(parts[3]) if len(parts) > 3 else 1

            n_cpgs = len(pattern)
            if n_cpgs < min_cpgs:
                continue
            if chrom_filter is not None and chrom != chrom_filter:
                continue

            chrom_positions = cpg_index.get(chrom)
            if chrom_positions is None:
                continue

            # Convert global CpG index to local index within this chromosome.
            offset = chrom_offsets.get(chrom, 0)
            local_start = start_global - 1 - offset  # 0-based local index
            local_end = local_start + n_cpgs

            if local_start < 0 or local_end > len(chrom_positions):
                continue

            cpg_positions = chrom_positions[local_start:local_end]
            meth_states = np.array(
                [1 if c == "T" else 0 for c in pattern], dtype=np.int8,
            )

            yield Read(
                chrom=chrom,
                cpg_positions=cpg_positions,
                meth_states=meth_states,
                count=count,
            )


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    """Sort chromosomes numerically: chr1, chr2, ..., chr22, chrX, chrY."""
    name = chrom.replace("chr", "")
    try:
        return (0, f"{int(name):03d}")
    except ValueError:
        return (1, name)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> SampleManifest:
    """Load a sample manifest from a TSV file.

    This is a thin wrapper around :meth:`SampleManifest.from_tsv`.

    Parameters
    ----------
    path : str or Path
        Path to the manifest TSV (may be gzipped).

    Returns
    -------
    SampleManifest
    """
    return SampleManifest.from_tsv(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pat_file(path: str | Path, n_lines: int = 1000) -> dict:
    """Spot-check a PAT file for format compliance.

    Inspects up to *n_lines* data lines and returns a summary dictionary with
    keys:

    * ``valid`` – bool, ``True`` if all checks passed.
    * ``errors`` – list of human-readable error strings.
    * ``lines_checked`` – number of data lines inspected.

    Checks performed:
        - Each line has 4 tab-separated columns.
        - Column 2 is a positive integer (global CpG index).
        - Column 3 contains only ``C`` and ``T`` characters.
        - Column 4 is a positive integer (count).
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    errors: list[str] = []
    checked = 0

    with opener(path, "rt") as fh:  # type: ignore[call-overload]
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            checked += 1
            if checked > n_lines:
                break

            parts = line.split("\t")
            if len(parts) != 4:
                errors.append(
                    f"Line {checked}: expected 4 columns, found {len(parts)}."
                )
                continue

            # Global CpG index (1-based, positive).
            try:
                idx = int(parts[1])
                if idx < 1:
                    errors.append(f"Line {checked}: CpG index must be >= 1, got {idx}.")
            except ValueError:
                errors.append(f"Line {checked}: cannot parse CpG index '{parts[1]}'.")

            # Pattern: only C and T.
            pattern = parts[2]
            if not pattern or not all(c in "CT" for c in pattern):
                errors.append(
                    f"Line {checked}: pattern contains characters other than C/T."
                )

            # Count (positive integer).
            try:
                count = int(parts[3])
                if count < 1:
                    errors.append(f"Line {checked}: count must be >= 1, got {count}.")
            except ValueError:
                errors.append(f"Line {checked}: cannot parse count '{parts[3]}'.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "lines_checked": checked,
    }


def validate_manifest(manifest: SampleManifest) -> list[str]:
    """Validate a :class:`SampleManifest`, returning a list of warnings/errors.

    Checks:
        - All referenced files exist on disc.
        - Every cell type has at least 2 samples.
        - No duplicate sample IDs.

    Returns
    -------
    list[str]
        Empty list if the manifest passes all checks; otherwise each element
        is a human-readable description of a problem found.
    """
    issues: list[str] = []

    # Duplicate IDs.
    seen: set[str] = set()
    for sid in manifest.sample_id:
        if sid in seen:
            issues.append(f"Duplicate sample ID: {sid}")
        seen.add(sid)

    # File existence.
    for i, fp in enumerate(manifest.file_path):
        if not Path(fp).exists():
            issues.append(
                f"File not found for sample '{manifest.sample_id[i]}': {fp}"
            )

    # Minimum samples per cell type.
    from collections import Counter

    counts = Counter(manifest.cell_type)
    for ct, n in counts.items():
        if n < 2:
            issues.append(
                f"Cell type '{ct}' has only {n} sample(s); at least 2 required."
            )

    return issues


# ---------------------------------------------------------------------------
# HDF5 serialisation for RegionReadMatrix
# ---------------------------------------------------------------------------

def save_region_read_matrix(rrm: RegionReadMatrix, h5_group: h5py.Group) -> None:
    """Persist a :class:`RegionReadMatrix` into an HDF5 group.

    Parameters
    ----------
    rrm : RegionReadMatrix
        The matrix to save.
    h5_group : h5py.Group
        Target HDF5 group (will be populated with datasets and attributes).
    """
    h5_group.create_dataset("matrix", data=rrm.matrix, compression="gzip")
    h5_group.create_dataset("cpg_positions", data=rrm.cpg_positions)

    if rrm.read_counts is not None:
        h5_group.create_dataset("read_counts", data=rrm.read_counts)

    if rrm.read_labels is not None:
        h5_group.create_dataset("read_labels", data=rrm.read_labels)

    if rrm.sample_ids is not None:
        # Store variable-length strings via special dtype.
        dt = h5py.string_dtype()
        ds = h5_group.create_dataset(
            "sample_ids",
            shape=(len(rrm.sample_ids),),
            dtype=dt,
        )
        ds[:] = rrm.sample_ids.astype(str)

    if rrm.region is not None:
        rg = h5_group.create_group("region")
        rg.attrs["region_id"] = rrm.region.region_id
        rg.attrs["chrom"] = rrm.region.chrom
        rg.attrs["start"] = rrm.region.start
        rg.attrs["end"] = rrm.region.end
        rg.create_dataset("cpg_positions", data=rrm.region.cpg_positions)


def load_region_read_matrix(h5_group: h5py.Group) -> RegionReadMatrix:
    """Reconstruct a :class:`RegionReadMatrix` from an HDF5 group.

    Parameters
    ----------
    h5_group : h5py.Group
        Source HDF5 group previously written by :func:`save_region_read_matrix`.

    Returns
    -------
    RegionReadMatrix
    """
    matrix = np.asarray(h5_group["matrix"], dtype=np.int8)
    cpg_positions = np.asarray(h5_group["cpg_positions"], dtype=np.int64)

    read_counts = None
    if "read_counts" in h5_group:
        read_counts = np.asarray(h5_group["read_counts"], dtype=np.int32)

    read_labels = None
    if "read_labels" in h5_group:
        read_labels = np.asarray(h5_group["read_labels"], dtype=np.int32)

    sample_ids = None
    if "sample_ids" in h5_group:
        sample_ids = np.asarray(h5_group["sample_ids"], dtype=object)

    region = None
    if "region" in h5_group:
        rg = h5_group["region"]
        region = Region(
            region_id=str(rg.attrs["region_id"]),
            chrom=str(rg.attrs["chrom"]),
            start=int(rg.attrs["start"]),
            end=int(rg.attrs["end"]),
            cpg_positions=np.asarray(rg["cpg_positions"], dtype=np.int64),
        )

    return RegionReadMatrix(
        matrix=matrix,
        cpg_positions=cpg_positions,
        read_counts=read_counts,
        read_labels=read_labels,
        sample_ids=sample_ids,
        region=region,
    )
