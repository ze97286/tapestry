"""rltf — standalone read-level tumour-fraction detection from cfDNA methylation.

Self-contained: depends only on numpy/pandas/scipy/scikit-learn (+ optional
tabicl/tabpfn). It imports nothing from any other project and consumes no other
method's artifacts. Everything works in the PAT global CpG-index coordinate.

Pipeline: discover own panel (discovery) → profiles → per-read LLR / oracle
(llr) → sample features (features) → tabular head (head).
"""

from rltf.discovery import discover_panel
from rltf.features import build_feature_matrix, compute_sample_features, feature_names
from rltf.head import CVResult, TabularTumourHead, leave_one_group_out_cv
from rltf.io import Read, load_reads
from rltf.llr import ReadScores, merge_scores, oracle_metrics, score_reads
from rltf.profiles import ReferenceProfiles
from rltf.regions import Block, BlockIndex, read_block_row, tile_blocks

__all__ = [
    "discover_panel",
    "ReferenceProfiles",
    "Read", "load_reads",
    "Block", "BlockIndex", "read_block_row", "tile_blocks",
    "ReadScores", "score_reads", "merge_scores", "oracle_metrics",
    "build_feature_matrix", "compute_sample_features", "feature_names",
    "TabularTumourHead", "leave_one_group_out_cv", "CVResult",
]
