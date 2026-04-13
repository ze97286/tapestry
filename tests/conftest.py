"""Shared fixtures for all Tapestry tests.

Creates a small synthetic dataset with known properties for deterministic testing.
PAT format: chrom, global_cpg_index (1-based), pattern (C/T), count.
"""

import gzip
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Provide a reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def tmp_dir():
    """Provide a session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope="session")
def synthetic_cell_types():
    """6 cell types for testing."""
    return ("CD4_T", "CD8_T", "B_cell", "NK", "monocyte", "neutrophil")


@pytest.fixture(scope="session")
def synthetic_cpg_index():
    """Small CpG index for chr1 and chr2."""
    rng = np.random.default_rng(42)
    # ~100 CpGs per chromosome spread over 0-50000
    chr1 = np.sort(rng.choice(50000, size=100, replace=False)).astype(np.int64)
    chr2 = np.sort(rng.choice(50000, size=80, replace=False)).astype(np.int64)
    return {"chr1": chr1, "chr2": chr2}


@pytest.fixture(scope="session")
def synthetic_cpg_bed(tmp_dir, synthetic_cpg_index):
    """Write a CpG.bed.gz file matching the synthetic index.

    Format: chrom  position  global_index (1-based)
    """
    bed_path = os.path.join(tmp_dir, "CpG.bed.gz")
    global_idx = 1
    with gzip.open(bed_path, "wt") as f:
        # Write in chromosome order matching _chrom_sort_key
        for chrom in ["chr1", "chr2"]:
            for pos in synthetic_cpg_index[chrom]:
                f.write(f"{chrom}\t{pos}\t{global_idx}\n")
                global_idx += 1
    return bed_path


@pytest.fixture(scope="session")
def synthetic_reads_dir(tmp_dir, synthetic_cpg_index, synthetic_cell_types):
    """Create synthetic PAT files for 6 cell types, 3 samples each.

    PAT format: chrom  global_cpg_index  pattern  count
    """
    rng = np.random.default_rng(123)
    reads_dir = os.path.join(tmp_dir, "reads")
    os.makedirs(reads_dir, exist_ok=True)

    # Build chrom offsets for global CpG index (1-based)
    chrom_order = ["chr1", "chr2"]
    chrom_offsets = {}
    running = 0
    for chrom in chrom_order:
        chrom_offsets[chrom] = running
        running += len(synthetic_cpg_index[chrom])

    sample_info = []
    for ct_idx, ct in enumerate(synthetic_cell_types):
        for sample_num in range(3):
            sample_id = f"{ct}_s{sample_num}"
            path = os.path.join(reads_dir, f"{sample_id}.pat.gz")

            # Each cell type has a characteristic methylation rate.
            meth_rate = 0.1 + ct_idx * 0.15  # ranges from 0.1 to 0.85

            with gzip.open(path, "wt") as f:
                for chrom in chrom_order:
                    cpgs = synthetic_cpg_index[chrom]
                    offset = chrom_offsets[chrom]
                    # Generate ~50 reads per chromosome
                    for _ in range(50):
                        idx = rng.integers(0, max(1, len(cpgs) - 5))
                        n_cpgs = rng.integers(3, min(8, len(cpgs) - idx) + 1)
                        meth_states = rng.random(n_cpgs) < meth_rate
                        pattern = "".join(
                            "T" if m else "C" for m in meth_states
                        )
                        # Global index is 1-based
                        global_idx = offset + idx + 1
                        count = 1
                        f.write(f"{chrom}\t{global_idx}\t{pattern}\t{count}\n")

            sample_info.append((sample_id, ct, path))

    return reads_dir, sample_info


@pytest.fixture(scope="session")
def synthetic_manifest_path(tmp_dir, synthetic_reads_dir):
    """Create a manifest TSV file."""
    _, sample_info = synthetic_reads_dir
    manifest_path = os.path.join(tmp_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("sample_id\tcell_type\tfile_path\n")
        for sid, ct, fpath in sample_info:
            f.write(f"{sid}\t{ct}\t{fpath}\n")
    return manifest_path


@pytest.fixture(scope="session")
def test_config(tmp_dir, synthetic_manifest_path, synthetic_cpg_bed):
    """Create a test config from the test YAML with overrides."""
    from tapestry.core.config import Config

    return Config.from_yaml(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "configs", "test.yaml"
        ),
        overrides={
            "project": {"output_dir": os.path.join(tmp_dir, "output")},
            "data": {
                "manifest_path": synthetic_manifest_path,
                "cpg_index_path": synthetic_cpg_bed,
                "chromosomes": ["chr1", "chr2"],
            },
        },
    )


@pytest.fixture
def sample_region():
    """A simple test Region."""
    from tapestry.core.datatypes import Region

    return Region(
        region_id="chr1_1000_1500",
        chrom="chr1",
        start=1000,
        end=1500,
        cpg_positions=np.array(
            [1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400], dtype=np.int64
        ),
    )


@pytest.fixture
def sample_rrm(sample_region):
    """A RegionReadMatrix with known content."""
    from tapestry.core.datatypes import RegionReadMatrix

    rng = np.random.default_rng(99)
    n_reads = 30
    n_cpgs = len(sample_region.cpg_positions)
    matrix = rng.choice(
        [-1, 0, 1], size=(n_reads, n_cpgs), p=[0.1, 0.45, 0.45]
    ).astype(np.int8)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int32)
    sids = np.array(
        ["s0"] * 5 + ["s1"] * 5 + ["s0"] * 5 + ["s1"] * 5 + ["s0"] * 5 + ["s1"] * 5,
        dtype=object,
    )
    return RegionReadMatrix(
        matrix=matrix,
        cpg_positions=sample_region.cpg_positions,
        read_labels=labels,
        sample_ids=sids,
        region=sample_region,
    )
