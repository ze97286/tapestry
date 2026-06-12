"""rltf — standalone read-level tumour detection from cfDNA methylation.

Per-read-call substrate (genomic CpG coordinates), mate-deduped, SNP-masked,
MAPQ-filtered. Detection via an n_cpg-conditioned per-fragment statistic (so read
length carries no label information) fed to a tabular head, evaluated
leave-one-group-out. ichorCNA is used only for clinical validation, never as a
model input. Depends only on numpy/pandas/scipy/scikit-learn (+ optional
tabicl/tabpfn, pysam for tabix region queries).
"""

from rltf.discovery import discover_panel
from rltf.features import build_feature_matrix, compute_sample_features, feature_names
from rltf.head import CVResult, TabularTumourHead, leave_one_group_out_cv
from rltf.io import Fragment, load_fragments
from rltf.llr import FragmentScores, merge_scores, oracle_metrics, score_fragments
from rltf.profiles import ReferenceProfiles
from rltf.regions import Block, tile_blocks

__all__ = [
    "load_fragments", "Fragment",
    "discover_panel", "ReferenceProfiles", "Block", "tile_blocks",
    "FragmentScores", "score_fragments", "merge_scores", "oracle_metrics",
    "build_feature_matrix", "compute_sample_features", "feature_names",
    "TabularTumourHead", "leave_one_group_out_cv", "CVResult",
]
