"""Sliding window candidate region generation.

Provides utilities for generating candidate genomic regions from CpG indices,
assigning sequencing reads to those regions, and building the read matrices
used downstream for marker scoring.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from tapestry.core.datatypes import Read, Region, RegionReadMatrix


def generate_candidate_regions(
    cpg_index: dict[str, np.ndarray],
    window_size: int = 500,
    step_size: int = 250,
    min_cpgs: int = 5,
    chromosomes: list[str] | None = None,
) -> list[Region]:
    """Generate candidate regions by sliding a window across CpG positions.

    For each chromosome a window of *window_size* bp is advanced by
    *step_size* bp.  Windows that contain at least *min_cpgs* CpG sites
    are retained.

    Parameters
    ----------
    cpg_index:
        Mapping of chromosome name to a **sorted** array of CpG positions.
    window_size:
        Width of the sliding window in base pairs.
    step_size:
        Step between consecutive window starts in base pairs.
    min_cpgs:
        Minimum number of CpG sites required within a window.
    chromosomes:
        Subset of chromosomes to process.  When ``None``, every chromosome
        present in *cpg_index* is used.

    Returns
    -------
    list[Region]
        Regions sorted by ``(chrom, start)``.
    """
    if chromosomes is None:
        chromosomes = sorted(cpg_index.keys())

    regions: list[Region] = []

    for chrom in chromosomes:
        if chrom not in cpg_index:
            continue

        positions = cpg_index[chrom]
        if len(positions) == 0:
            continue

        # Ensure positions are sorted
        positions = np.sort(positions)

        chrom_start = int(positions[0])
        chrom_end = int(positions[-1]) + 1  # inclusive upper bound for last CpG

        start = chrom_start
        while start < chrom_end:
            end = start + window_size

            # Use searchsorted to find CpGs falling within [start, end)
            left_idx = np.searchsorted(positions, start, side="left")
            right_idx = np.searchsorted(positions, end, side="left")

            cpg_positions = positions[left_idx:right_idx]

            if len(cpg_positions) >= min_cpgs:
                region_id = f"{chrom}_{start}_{end}"
                regions.append(
                    Region(
                        region_id=region_id,
                        chrom=chrom,
                        start=start,
                        end=end,
                        cpg_positions=cpg_positions.copy(),
                    )
                )

            start += step_size

    # Sort by (chrom, start) – already chromosome-sorted but ensure stability
    regions.sort(key=lambda r: (r.chrom, r.start))
    return regions


def assign_reads_to_regions(
    reads: Iterator[Read],
    regions: list[Region],
    min_cpgs_overlap: int = 3,
) -> dict[str, list[tuple[Read, np.ndarray]]]:
    """Assign reads to overlapping regions and build per-read row vectors.

    Each row vector has one entry per CpG in the region:

    *  ``1``  – methylated
    *  ``0``  – unmethylated
    * ``-1``  – not covered by this read

    The function exploits the fact that both *reads* and *regions* are sorted
    by ``(chrom, start)`` and uses a two-pointer approach to avoid scanning
    every region for every read.

    Parameters
    ----------
    reads:
        An iterator of :class:`Read` objects sorted by ``(chrom, start)``.
    regions:
        Regions sorted by ``(chrom, start)``.
    min_cpgs_overlap:
        Minimum number of the region's CpGs that a read must cover.

    Returns
    -------
    dict[str, list[tuple[Read, np.ndarray]]]
        Mapping of ``region_id`` to a list of ``(Read, row_vector)`` tuples.
    """
    result: dict[str, list[tuple[Read, np.ndarray]]] = {}

    if not regions:
        return result

    # Build a quick lookup: chrom -> list of (index into regions)
    chrom_region_indices: dict[str, list[int]] = {}
    for idx, region in enumerate(regions):
        chrom_region_indices.setdefault(region.chrom, []).append(idx)

    # Two-pointer per chromosome: track where we left off
    chrom_pointer: dict[str, int] = {ch: 0 for ch in chrom_region_indices}

    for read in reads:
        chrom = read.chrom
        if chrom not in chrom_region_indices:
            continue

        region_indices = chrom_region_indices[chrom]
        ptr = chrom_pointer[chrom]

        # Advance pointer past regions whose end <= read.start
        while ptr < len(region_indices) and regions[region_indices[ptr]].end <= read.start:
            ptr += 1
        chrom_pointer[chrom] = ptr

        # Scan forward through regions that could overlap this read
        j = ptr
        while j < len(region_indices):
            region = regions[region_indices[j]]

            # If the region starts beyond the read's end, no further overlap
            if region.start >= read.end:
                break

            # Compute overlap of CpG positions
            read_positions = read.cpg_positions
            region_positions = region.cpg_positions

            # Fast intersection using searchsorted
            indices_in_read = np.searchsorted(read_positions, region_positions)
            # Clamp indices and check for actual matches
            indices_in_read = np.clip(indices_in_read, 0, len(read_positions) - 1)
            matched_mask = read_positions[indices_in_read] == region_positions

            n_overlap = int(np.sum(matched_mask))

            if n_overlap >= min_cpgs_overlap:
                # Build the row vector for this region
                row = np.full(len(region_positions), -1, dtype=np.int8)

                for k, (is_match, idx_in_read) in enumerate(
                    zip(matched_mask, indices_in_read)
                ):
                    if is_match:
                        row[k] = read.meth_states[idx_in_read]

                rid = region.region_id
                if rid not in result:
                    result[rid] = []
                result[rid].append((read, row))

            j += 1

    return result


def build_region_read_matrices(
    reads_by_region: dict[str, list[tuple[Read, np.ndarray]]],
    regions: list[Region],
    cell_type_label: int | None = None,
    sample_id: str | None = None,
) -> list[RegionReadMatrix]:
    """Convert the output of :func:`assign_reads_to_regions` into matrices.

    Parameters
    ----------
    reads_by_region:
        Mapping produced by :func:`assign_reads_to_regions`.
    regions:
        The full list of regions (used to look up metadata).
    cell_type_label:
        If provided, every read in the resulting matrices receives this
        label (useful when processing a single cell-type BAM).
    sample_id:
        If provided, every read is tagged with this sample identifier.

    Returns
    -------
    list[RegionReadMatrix]
        One :class:`RegionReadMatrix` per region that has at least one read.
    """
    region_lookup = {r.region_id: r for r in regions}
    matrices: list[RegionReadMatrix] = []

    for rid, read_rows in reads_by_region.items():
        if not read_rows:
            continue

        region = region_lookup[rid]

        # Store one row per unique pattern, with counts from PAT.
        rows: list[np.ndarray] = []
        counts: list[int] = []
        for read, row in read_rows:
            rows.append(row)
            counts.append(getattr(read, "count", 1) or 1)

        matrix = np.stack(rows, axis=0)
        n_rows = len(rows)
        read_counts = np.array(counts, dtype=np.int32)

        read_labels: np.ndarray | None = None
        if cell_type_label is not None:
            read_labels = np.full(n_rows, cell_type_label, dtype=np.int32)

        sample_ids_arr: np.ndarray | None = None
        if sample_id is not None:
            sample_ids_arr = np.array([sample_id] * n_rows, dtype=object)

        matrices.append(
            RegionReadMatrix(
                matrix=matrix,
                cpg_positions=region.cpg_positions,
                read_counts=read_counts,
                read_labels=read_labels,
                sample_ids=sample_ids_arr,
                region=region,
            )
        )

    # Return in the same order as the input regions
    region_order = {r.region_id: i for i, r in enumerate(regions)}
    matrices.sort(key=lambda m: region_order.get(m.region.region_id, 0))
    return matrices
