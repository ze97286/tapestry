"""
CpG index construction and persistence for the Tapestry pipeline.

A CpG index is a mapping from chromosome name to a sorted array of genomic
positions where CpG dinucleotides occur.  It can be built from a reference
FASTA or from the union of observed sites across sample reads files.
"""

from __future__ import annotations

import gzip
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from tapestry.core.datatypes import SampleManifest
from tapestry.core.io import load_reads

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building from reference FASTA
# ---------------------------------------------------------------------------

def build_cpg_index_from_fasta(
    fasta_path: str | Path,
    chromosomes: list[str],
) -> dict[str, np.ndarray]:
    """Parse a reference FASTA and locate all CpG positions.

    Scans for occurrences of the dinucleotide ``CG`` (case-insensitive) on
    the forward strand.  The reported position is the 0-based coordinate of
    the cytosine.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the reference genome FASTA (may be uncompressed or gzipped).
    chromosomes : list[str]
        Chromosome names to include.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from chromosome name to a sorted int64 array of CpG positions.
    """
    fasta_path = Path(fasta_path)
    chrom_set = set(chromosomes)
    index: dict[str, list[int]] = {c: [] for c in chromosomes}

    opener = gzip.open if fasta_path.suffix == ".gz" else open
    current_chrom: Optional[str] = None
    offset: int = 0
    prev_char: str = ""

    with opener(fasta_path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                # Header line – extract chromosome name (first whitespace-
                # delimited token after '>').
                current_chrom = line[1:].split()[0]
                if current_chrom not in chrom_set:
                    current_chrom = None
                offset = 0
                prev_char = ""
                continue

            if current_chrom is None:
                continue

            seq = line.upper()
            for ch in seq:
                if prev_char == "C" and ch == "G":
                    # Record the position of the C in the CpG.
                    index[current_chrom].append(offset - 1)
                prev_char = ch
                offset += 1

    result: dict[str, np.ndarray] = {}
    for chrom in chromosomes:
        positions = index.get(chrom, [])
        result[chrom] = np.array(positions, dtype=np.int64)
    return result


# ---------------------------------------------------------------------------
# Building from observed reads
# ---------------------------------------------------------------------------

def _collect_cpgs_from_file(
    file_path: str,
    chromosomes: list[str],
) -> dict[str, set[int]]:
    """Worker: collect unique CpG positions from a single reads file."""
    chrom_set = set(chromosomes)
    cpgs: dict[str, set[int]] = {c: set() for c in chromosomes}
    for read in load_reads(file_path):
        if read.chrom in chrom_set:
            cpgs[read.chrom].update(read.cpg_positions.tolist())
    return cpgs


def build_cpg_index_from_reads(
    manifest: SampleManifest,
    chromosomes: list[str],
    n_workers: int = 4,
) -> dict[str, np.ndarray]:
    """Build a CpG index from the union of observed sites across all samples.

    Parameters
    ----------
    manifest : SampleManifest
        Sample manifest listing the reads files.
    chromosomes : list[str]
        Chromosomes to include.
    n_workers : int
        Number of parallel worker processes.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from chromosome name to a sorted int64 array of CpG positions.
    """
    merged: dict[str, set[int]] = {c: set() for c in chromosomes}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_collect_cpgs_from_file, fp, chromosomes): fp
            for fp in manifest.file_path
        }
        for future in as_completed(futures):
            fp = futures[future]
            try:
                per_file = future.result()
            except Exception:
                logger.exception("Failed to process %s", fp)
                continue
            for chrom in chromosomes:
                merged[chrom].update(per_file.get(chrom, set()))

    result: dict[str, np.ndarray] = {}
    for chrom in chromosomes:
        arr = np.array(sorted(merged[chrom]), dtype=np.int64)
        result[chrom] = arr
        logger.info("Chromosome %s: %d CpG sites", chrom, len(arr))
    return result


# ---------------------------------------------------------------------------
# Persistence (gzipped TSV)
# ---------------------------------------------------------------------------

def save_cpg_index(index: dict[str, np.ndarray], path: str | Path) -> None:
    """Save a CpG index to a gzipped two-column TSV file.

    Columns are ``chrom`` and ``position`` (0-based).  Rows are sorted by
    chromosome (in the order of the dict) then by position.

    Parameters
    ----------
    index : dict[str, np.ndarray]
        CpG index to persist.
    path : str or Path
        Destination file path (should end with ``.tsv.gz``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt") as fh:
        fh.write("#chrom\tposition\n")
        for chrom, positions in index.items():
            for pos in positions:
                fh.write(f"{chrom}\t{pos}\n")


def load_cpg_index(path: str | Path) -> dict[str, np.ndarray]:
    """Load a CpG index from a gzipped TSV file.

    Expects at least two columns: ``chrom`` and ``position``.  Any additional
    columns (e.g. a global CpG index) are ignored.

    Parameters
    ----------
    path : str or Path
        Path to the ``.tsv.gz`` (or plain ``.tsv`` / ``.bed.gz``) file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from chromosome name to a sorted int64 array of CpG positions.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    data: dict[str, list[int]] = {}

    with opener(path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            chrom = parts[0]
            pos = int(parts[1])
            data.setdefault(chrom, []).append(pos)

    return {
        chrom: np.array(positions, dtype=np.int64)
        for chrom, positions in data.items()
    }
