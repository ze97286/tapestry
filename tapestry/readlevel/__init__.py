"""Read-level (single-molecule) tumour-signal tooling.

This subpackage implements the read-level likelihood-ratio *oracle* — the
decisive test of whether tumour-vs-healthy methylation is separable at the
single-fragment level on a given block panel, with no transformer, no
gradient training, and no read-length / cohort channel for a classifier to
exploit.

See ``docs/read_level_detection_direction.md`` for the rationale and how this
gates the broader read-level deconvolution programme.
"""

from tapestry.readlevel.llr_oracle import (
    ReferenceProfiles,
    compute_oracle_metrics,
    estimate_reference_profiles,
    load_regions_from_bed,
    score_reads,
)

__all__ = [
    "ReferenceProfiles",
    "compute_oracle_metrics",
    "estimate_reference_profiles",
    "load_regions_from_bed",
    "score_reads",
]
