"""Tests for I/O operations: PAT loading, manifest parsing, HDF5 serialisation, validation."""

import gzip
import os
import tempfile

import h5py
import numpy as np
import pytest

from tapestry.core.io import (
    load_manifest,
    load_reads,
    load_region_read_matrix,
    save_region_read_matrix,
    validate_manifest,
    validate_pat_file,
)


class TestLoadReads:
    """Tests for loading reads from PAT files."""

    def test_loads_reads(self, synthetic_reads_dir, synthetic_cpg_index):
        """Should successfully load reads from a synthetic PAT file."""
        reads_dir, sample_info = synthetic_reads_dir
        sid, ct, fpath = sample_info[0]
        reads = list(load_reads(fpath, synthetic_cpg_index))
        assert len(reads) > 0
        for r in reads:
            assert r.num_cpgs >= 1
            assert len(r.cpg_positions) == len(r.meth_states)
            assert all(m in (0, 1) for m in r.meth_states)

    def test_chrom_filter(self, synthetic_reads_dir, synthetic_cpg_index):
        """Filtering by chromosome should return only reads on that chromosome."""
        _, sample_info = synthetic_reads_dir
        fpath = sample_info[0][2]
        reads_chr1 = list(load_reads(fpath, synthetic_cpg_index, chrom_filter="chr1"))
        reads_all = list(load_reads(fpath, synthetic_cpg_index))
        assert len(reads_chr1) < len(reads_all)
        assert all(r.chrom == "chr1" for r in reads_chr1)

    def test_min_cpgs_filter(self, synthetic_reads_dir, synthetic_cpg_index):
        """Filtering by minimum CpGs should exclude short reads."""
        _, sample_info = synthetic_reads_dir
        fpath = sample_info[0][2]
        reads = list(load_reads(fpath, synthetic_cpg_index, min_cpgs=5))
        assert all(r.num_cpgs >= 5 for r in reads)

    def test_genomic_positions_resolved(self, synthetic_reads_dir, synthetic_cpg_index):
        """CpG positions should be genomic coordinates, not global indices."""
        _, sample_info = synthetic_reads_dir
        fpath = sample_info[0][2]
        reads = list(load_reads(fpath, synthetic_cpg_index, chrom_filter="chr1"))
        # Positions should fall within the chr1 CpG range
        chr1_positions = synthetic_cpg_index["chr1"]
        for r in reads:
            assert all(pos in chr1_positions for pos in r.cpg_positions)


class TestValidation:
    """Tests for file and manifest validation routines."""

    def test_validate_good_pat(self, synthetic_reads_dir):
        """A well-formed PAT file should pass validation."""
        _, sample_info = synthetic_reads_dir
        result = validate_pat_file(sample_info[0][2])
        assert result["valid"]
        assert result["lines_checked"] > 0

    def test_validate_manifest(self, synthetic_manifest_path):
        """A manifest pointing to existing files should produce no file-not-found issues."""
        m = load_manifest(synthetic_manifest_path)
        issues = validate_manifest(m)
        file_issues = [i for i in issues if "File not found" in i]
        assert len(file_issues) == 0


class TestHDF5Serialisation:
    """Tests for saving and loading RegionReadMatrix objects via HDF5."""

    def test_roundtrip(self, sample_rrm, tmp_path):
        """Saving and loading a RegionReadMatrix should preserve all data."""
        h5_path = str(tmp_path / "test.h5")
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("test_region")
            save_region_read_matrix(sample_rrm, grp)

        with h5py.File(h5_path, "r") as f:
            loaded = load_region_read_matrix(f["test_region"])

        np.testing.assert_array_equal(loaded.matrix, sample_rrm.matrix)
        np.testing.assert_array_equal(loaded.cpg_positions, sample_rrm.cpg_positions)
        np.testing.assert_array_equal(loaded.read_labels, sample_rrm.read_labels)
        assert loaded.region.region_id == sample_rrm.region.region_id
