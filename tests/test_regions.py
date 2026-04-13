"""Tests for region generation and read assignment to regions."""

import numpy as np

from tapestry.core.datatypes import Read, Region
from tapestry.markers.regions import (
    assign_reads_to_regions,
    build_region_read_matrices,
    generate_candidate_regions,
)


class TestCandidateRegions:
    """Tests for sliding-window region generation."""

    def test_generates_regions(self, synthetic_cpg_index):
        """Should produce regions across both chromosomes."""
        regions = generate_candidate_regions(
            synthetic_cpg_index,
            window_size=500,
            step_size=250,
            min_cpgs=3,
        )
        assert len(regions) > 0
        for r in regions:
            assert len(r.cpg_positions) >= 3
            assert r.end - r.start == 500

    def test_respects_min_cpgs(self, synthetic_cpg_index):
        """Raising the minimum CpG threshold should produce fewer (or equal) regions."""
        regions_3 = generate_candidate_regions(
            synthetic_cpg_index,
            window_size=500,
            step_size=250,
            min_cpgs=3,
        )
        regions_10 = generate_candidate_regions(
            synthetic_cpg_index,
            window_size=500,
            step_size=250,
            min_cpgs=10,
        )
        assert len(regions_10) <= len(regions_3)


class TestReadAssignment:
    """Tests for assigning reads to candidate regions."""

    def test_assigns_reads(self, synthetic_cpg_index):
        """Reads overlapping a region should be assigned to it."""
        regions = generate_candidate_regions(
            synthetic_cpg_index,
            window_size=2000,
            step_size=1000,
            min_cpgs=3,
            chromosomes=["chr1"],
        )
        # Create some synthetic reads on chr1
        cpgs = synthetic_cpg_index["chr1"]
        reads = []
        for i in range(0, min(20, len(cpgs) - 3)):
            read_cpgs = cpgs[i : i + 4]
            reads.append(
                Read(
                    chrom="chr1",
                    cpg_positions=read_cpgs,
                    meth_states=np.ones(len(read_cpgs), dtype=np.int8),
                )
            )

        result = assign_reads_to_regions(iter(reads), regions, min_cpgs_overlap=3)
        # At least some reads should be assigned
        total_assigned = sum(len(v) for v in result.values())
        assert total_assigned > 0
