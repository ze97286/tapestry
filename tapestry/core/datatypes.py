"""
Core data structures for the Tapestry cfDNA methylation deconvolution pipeline.

Defines the canonical cell-type labels, read and region representations,
the region-read matrix used throughout marker selection and encoding, and
the sample manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cell-type labels — no hard-coded list.
# Cell types are discovered from the sample manifest at runtime.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

@dataclass
class Read:
    """A single sequencing read (or collapsed PAT pattern) with CpG methylation.

    Attributes
    ----------
    chrom : str
        Chromosome name (e.g. ``"chr1"``).
    cpg_positions : np.ndarray
        Sorted genomic positions of CpGs covered by this read (int64).
    meth_states : np.ndarray
        Methylation state at each CpG position: 0 = unmethylated,
        1 = methylated (int8).
    count : int
        Number of reads with this exact pattern (from PAT file).
    """

    chrom: str
    cpg_positions: np.ndarray  # int64
    meth_states: np.ndarray    # int8
    count: int = 1

    @property
    def num_cpgs(self) -> int:
        """Number of CpGs covered by this read."""
        return len(self.cpg_positions)

    @property
    def start(self) -> int:
        """Genomic position of the first CpG (for region overlap checks)."""
        return int(self.cpg_positions[0]) if len(self.cpg_positions) > 0 else 0

    @property
    def end(self) -> int:
        """Genomic position past the last CpG (for region overlap checks)."""
        return int(self.cpg_positions[-1]) + 1 if len(self.cpg_positions) > 0 else 0


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """A genomic window that contains CpG sites.

    Attributes
    ----------
    region_id : str
        Unique identifier, typically ``"chrom:start-end"``.
    chrom : str
        Chromosome name.
    start : int
        0-based start position.
    end : int
        0-based exclusive end position.
    cpg_positions : np.ndarray
        Sorted CpG positions within this region (int64).
    """

    region_id: str
    chrom: str
    start: int
    end: int
    cpg_positions: np.ndarray  # int64


# ---------------------------------------------------------------------------
# RegionReadMatrix
# ---------------------------------------------------------------------------

@dataclass
class RegionReadMatrix:
    """Methylation matrix for reads overlapping a single region.

    The matrix has shape ``(N, C)`` where *N* is the number of reads and *C*
    is the number of CpG sites in the region.  Values are encoded as:
    ``0`` = unmethylated, ``1`` = methylated, ``-1`` = missing / no coverage.

    Attributes
    ----------
    matrix : np.ndarray
        Read-by-CpG methylation matrix, dtype int8, shape (N, C).
    cpg_positions : np.ndarray
        Genomic positions of the C CpG columns (int64, shape C).
    read_counts : np.ndarray or None
        Number of original reads each row represents (int32, shape N).
        From collapsed PAT patterns where count > 1.  ``None`` means
        every row has count 1.
    read_labels : np.ndarray or None
        Cell-type index for each read (int32, shape N).  ``None`` when
        labels are unavailable (e.g. cfDNA inference).
    sample_ids : np.ndarray or None
        Sample identifier for each read (object dtype, shape N).
    region : Region
        The genomic region that this matrix corresponds to.
    """

    matrix: np.ndarray          # int8,   (N, C)
    cpg_positions: np.ndarray   # int64,  (C,)
    read_counts: Optional[np.ndarray] = None   # int32,  (N,)
    read_labels: Optional[np.ndarray] = None   # int32,  (N,)
    sample_ids: Optional[np.ndarray] = None    # object, (N,)
    region: Optional[Region] = None

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def n_reads(self) -> int:
        """Number of reads (rows) in the matrix."""
        return self.matrix.shape[0]

    @property
    def n_cpgs(self) -> int:
        """Number of CpG columns in the matrix."""
        return self.matrix.shape[1]

    # -----------------------------------------------------------------
    # Filtering
    # -----------------------------------------------------------------

    def filter_by_coverage(self, min_cpgs: int) -> RegionReadMatrix:
        """Return a new matrix keeping only reads with >= *min_cpgs* observed sites.

        A site is considered observed when its value is not ``-1``.
        """
        observed = (self.matrix != -1).sum(axis=1)
        mask = observed >= min_cpgs
        return self._apply_row_mask(mask)

    def filter_by_cell_type(self, cell_type_idx: int) -> RegionReadMatrix:
        """Return a new matrix keeping only reads belonging to *cell_type_idx*.

        Raises
        ------
        ValueError
            If ``read_labels`` is ``None``.
        """
        if self.read_labels is None:
            raise ValueError("Cannot filter by cell type: read_labels is None.")
        mask = self.read_labels == cell_type_idx
        return self._apply_row_mask(mask)

    def _apply_row_mask(self, mask: np.ndarray) -> RegionReadMatrix:
        """Return a copy of this matrix with only rows where *mask* is True."""
        return RegionReadMatrix(
            matrix=self.matrix[mask],
            cpg_positions=self.cpg_positions.copy(),
            read_counts=self.read_counts[mask] if self.read_counts is not None else None,
            read_labels=self.read_labels[mask] if self.read_labels is not None else None,
            sample_ids=self.sample_ids[mask] if self.sample_ids is not None else None,
            region=self.region,
        )

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (arrays become lists)."""
        d: dict = {
            "matrix": self.matrix.tolist(),
            "cpg_positions": self.cpg_positions.tolist(),
        }
        if self.read_counts is not None:
            d["read_counts"] = self.read_counts.tolist()
        if self.read_labels is not None:
            d["read_labels"] = self.read_labels.tolist()
        if self.sample_ids is not None:
            d["sample_ids"] = self.sample_ids.tolist()
        if self.region is not None:
            d["region"] = {
                "region_id": self.region.region_id,
                "chrom": self.region.chrom,
                "start": self.region.start,
                "end": self.region.end,
                "cpg_positions": self.region.cpg_positions.tolist(),
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RegionReadMatrix:
        """Reconstruct a ``RegionReadMatrix`` from a dictionary."""
        region = None
        if "region" in d:
            rd = d["region"]
            region = Region(
                region_id=rd["region_id"],
                chrom=rd["chrom"],
                start=rd["start"],
                end=rd["end"],
                cpg_positions=np.asarray(rd["cpg_positions"], dtype=np.int64),
            )

        read_counts = None
        if "read_counts" in d:
            read_counts = np.asarray(d["read_counts"], dtype=np.int32)

        read_labels = None
        if "read_labels" in d:
            read_labels = np.asarray(d["read_labels"], dtype=np.int32)

        sample_ids = None
        if "sample_ids" in d:
            sample_ids = np.asarray(d["sample_ids"], dtype=object)

        return cls(
            matrix=np.asarray(d["matrix"], dtype=np.int8),
            cpg_positions=np.asarray(d["cpg_positions"], dtype=np.int64),
            read_counts=read_counts,
            read_labels=read_labels,
            sample_ids=sample_ids,
            region=region,
        )


# ---------------------------------------------------------------------------
# SampleManifest
# ---------------------------------------------------------------------------

@dataclass
class SampleManifest:
    """Tabular manifest describing all training / reference samples.

    Each row corresponds to a single sample file.

    Attributes
    ----------
    sample_id : list[str]
        Unique identifiers.
    cell_type : list[str]
        Cell-type label for each sample.
    file_path : list[str]
        Path to the reads file for each sample.
    coverage : list[float] or None
        Optional mean coverage per sample.
    batch : list[str] or None
        Optional batch identifier per sample.
    """

    sample_id: list[str] = field(default_factory=list)
    cell_type: list[str] = field(default_factory=list)
    file_path: list[str] = field(default_factory=list)
    coverage: Optional[list[float]] = None
    batch: Optional[list[str]] = None

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return len(self.sample_id)

    @property
    def n_cell_types(self) -> int:
        """Number of unique cell types."""
        return len(self.cell_types)

    @property
    def cell_types(self) -> list[str]:
        """Sorted list of unique cell types."""
        return sorted(set(self.cell_type))

    @property
    def cell_type_indices(self) -> dict[str, int]:
        """Mapping from cell-type label to integer index (sorted order)."""
        return {ct: i for i, ct in enumerate(self.cell_types)}

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def from_tsv(cls, path: str | Path) -> SampleManifest:
        """Load a manifest from a tab-separated file.

        Expected columns: ``sample_id``, ``cell_type``, ``file_path``, and
        optionally ``coverage`` and ``batch``.

        Parameters
        ----------
        path : str or Path
            Path to the TSV file (may be gzipped).
        """
        import gzip

        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:  # type: ignore[call-overload]
            header_line = fh.readline().rstrip("\n")
            # Detect delimiter: tab if present, otherwise whitespace.
            if "\t" in header_line:
                headers = header_line.split("\t")
                split_fn = lambda line: line.rstrip("\n").split("\t")
            else:
                headers = header_line.split()
                split_fn = lambda line: line.rstrip("\n").split()

            col = {name: i for i, name in enumerate(headers)}

            sample_ids: list[str] = []
            cell_types: list[str] = []
            file_paths: list[str] = []
            coverages: list[float] = []
            batches: list[str] = []
            has_coverage = "coverage" in col
            has_batch = "batch" in col

            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                fields = split_fn(line)
                sample_ids.append(fields[col["sample_id"]])
                cell_types.append(fields[col["cell_type"]])
                file_paths.append(fields[col["file_path"]])
                if has_coverage:
                    coverages.append(float(fields[col["coverage"]]))
                if has_batch:
                    batches.append(fields[col["batch"]])

        return cls(
            sample_id=sample_ids,
            cell_type=cell_types,
            file_path=file_paths,
            coverage=coverages if has_coverage else None,
            batch=batches if has_batch else None,
        )

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def samples_for_cell_type(self, ct: str) -> list[int]:
        """Return row indices for all samples belonging to cell type *ct*."""
        return [i for i, c in enumerate(self.cell_type) if c == ct]
