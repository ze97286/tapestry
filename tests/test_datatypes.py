"""Tests for core data types: cell types, RegionReadMatrix, and SampleManifest."""

import numpy as np
import pytest

from tapestry.core.datatypes import (
    Read,
    Region,
    RegionReadMatrix,
    SampleManifest,
)


class TestCellTypesFromManifest:
    """Cell types are discovered from the manifest, not hard-coded."""

    def test_cell_types_from_manifest(self, synthetic_manifest_path):
        m = SampleManifest.from_tsv(synthetic_manifest_path)
        assert m.n_cell_types == 6
        # Indices should be consistent
        for ct in m.cell_types:
            assert ct in m.cell_type_indices


class TestRegionReadMatrix:
    """Tests for RegionReadMatrix construction, filtering, and serialisation."""

    def test_properties(self, sample_rrm):
        """Basic shape properties should be correct."""
        assert sample_rrm.n_reads == 30
        assert sample_rrm.n_cpgs == 8

    def test_filter_by_coverage(self, sample_rrm):
        """Filtering by coverage should remove reads with too many missing CpGs."""
        filtered = sample_rrm.filter_by_coverage(min_cpgs=7)
        # All retained reads must have at least 7 non-missing CpGs
        for i in range(filtered.n_reads):
            assert (filtered.matrix[i] != -1).sum() >= 7

    def test_filter_by_cell_type(self, sample_rrm):
        """Filtering by cell type should return only reads with the matching label."""
        ct0 = sample_rrm.filter_by_cell_type(0)
        assert ct0.n_reads == 10
        assert all(ct0.read_labels == 0)

    def test_filter_by_cell_type_no_labels(self, sample_region):
        """Filtering by cell type should raise when labels are absent."""
        rrm = RegionReadMatrix(
            matrix=np.zeros((5, 8), dtype=np.int8),
            cpg_positions=sample_region.cpg_positions,
        )
        with pytest.raises(ValueError):
            rrm.filter_by_cell_type(0)

    def test_serialisation_roundtrip(self, sample_rrm):
        """Serialising to dict and back should preserve all data."""
        d = sample_rrm.to_dict()
        restored = RegionReadMatrix.from_dict(d)
        np.testing.assert_array_equal(restored.matrix, sample_rrm.matrix)
        np.testing.assert_array_equal(restored.cpg_positions, sample_rrm.cpg_positions)
        np.testing.assert_array_equal(restored.read_labels, sample_rrm.read_labels)
        assert restored.region.region_id == sample_rrm.region.region_id


class TestSampleManifest:
    """Tests for loading and querying the sample manifest."""

    def test_from_tsv(self, synthetic_manifest_path):
        """Loading a manifest should yield the expected sample count."""
        m = SampleManifest.from_tsv(synthetic_manifest_path)
        assert m.n_samples == 18  # 6 cell types * 3 samples
        assert m.n_cell_types == 6

    def test_samples_for_cell_type(self, synthetic_manifest_path):
        """Querying by cell type should return the correct number of samples."""
        m = SampleManifest.from_tsv(synthetic_manifest_path)
        cd4 = m.samples_for_cell_type("CD4_T")
        assert len(cd4) == 3
