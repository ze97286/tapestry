"""PyTorch Dataset classes for all tapestry models."""

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Detector / Aggregator datasets (single-cell-type concentration)
# ---------------------------------------------------------------------------

class DetectorDataset(Dataset):
    """Dataset for single-cell-type concentration estimation.

    Each sample contains marker methylation values, coverage, the target
    cell type's true concentration, and an optional control flag.

    Parameters
    ----------
    marker_values : np.ndarray
        Shape (N, M) — methylation fractions per marker.
    coverage : np.ndarray
        Shape (N, M) — read counts per marker.
    y_true : np.ndarray
        Shape (N,) — true concentration of the target cell type.
    control_mask : np.ndarray or None
        Shape (N,) — boolean mask for known-negative control samples.
    """

    def __init__(self, marker_values, coverage, y_true, control_mask=None):
        self.marker_values = torch.tensor(marker_values, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.y_true = torch.tensor(y_true, dtype=torch.float32).unsqueeze(1)
        self.control_mask = (
            torch.tensor(control_mask, dtype=torch.bool)
            if control_mask is not None
            else torch.zeros(len(y_true), dtype=torch.bool)
        )

    def __len__(self):
        return len(self.y_true)

    def __getitem__(self, idx):
        return (
            self.marker_values[idx],
            self.coverage[idx],
            self.y_true[idx],
            self.control_mask[idx],
        )


# ---------------------------------------------------------------------------
# Deconvolution datasets (all cell types)
# ---------------------------------------------------------------------------

class DeconvolutionDataset(Dataset):
    """Dataset for full multi-cell-type deconvolution.

    Each sample includes marker fractions, coverage, ground-truth proportions
    for all cell types, and optional NNLS predictions.

    Parameters
    ----------
    fraction : array-like
        Shape (N, M) — methylation fractions per marker.
    coverage : array-like
        Shape (N, M) — read counts per marker.
    atlas : array-like
        Reference atlas data (M, C) — stored for NNLS computation.
    y : array-like or None
        Shape (N, C) — ground-truth proportions. None for inference.
    x_nnls : array-like or None
        Shape (N, C) — pre-computed NNLS predictions.
    """

    def __init__(self, fraction, coverage, atlas, y=None, x_nnls=None):
        self.fraction = torch.tensor(fraction, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.atlas = torch.tensor(atlas, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.x_nnls = torch.tensor(x_nnls, dtype=torch.float32) if x_nnls is not None else None

    def __len__(self):
        return self.fraction.size(0)

    def __getitem__(self, idx):
        item = {"X": self.fraction[idx], "coverage": self.coverage[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        if self.x_nnls is not None:
            item["x_nnls"] = self.x_nnls[idx]
        return item


# ---------------------------------------------------------------------------
# Presence dataset (binary classification per cell type)
# ---------------------------------------------------------------------------

class PresenceDataset(Dataset):
    """Dataset for binary cell-type presence classification.

    Parameters
    ----------
    fraction : array-like
        Shape (N, M) — methylation fractions for the target cell type's markers.
    coverage : array-like
        Shape (N, M) — read counts for those markers.
    y : array-like or None
        Shape (N, C) — full proportion vector (used to derive binary labels).
    target_cell_type : int
        Index of the target cell type.
    target_ids : array-like or None
        Marker-to-cell-type mapping (for extracting cell-specific markers).
    presence_threshold : float
        Threshold above which a cell type is considered present.
    """

    def __init__(
        self,
        fraction,
        coverage,
        y=None,
        target_cell_type: int = 0,
        target_ids=None,
        presence_threshold: float = 0.0005,
    ):
        self.fraction = torch.tensor(fraction, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.target_cell_type = target_cell_type

        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32)
            self.target_prop = y_tensor[:, target_cell_type]
            self.label = (self.target_prop > presence_threshold).float()
        else:
            self.target_prop = None
            self.label = None

        if target_ids is not None:
            ids_t = torch.tensor(target_ids, dtype=torch.long)
            self.target_markers_mask = ids_t == target_cell_type
        else:
            self.target_markers_mask = None

    def __len__(self):
        return self.fraction.size(0)

    def __getitem__(self, idx):
        item = {"X": self.fraction[idx], "coverage": self.coverage[idx]}
        if self.label is not None:
            item["label"] = self.label[idx]
        if self.target_prop is not None:
            item["concentration"] = self.target_prop[idx]
        if self.target_markers_mask is not None:
            item["target_markers_mask"] = self.target_markers_mask
        return item
