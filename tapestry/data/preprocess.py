"""Data loading and preprocessing for training and evaluation.

Handles loading parquet files (marker_values, coverage, ground_truth_y),
extracting target cell type markers from the atlas, and creating DataLoaders.
"""

import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tapestry.data.datasets import DetectorDataset, DeconvolutionDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detector / Aggregator data loading
# ---------------------------------------------------------------------------

def load_detector_data(
    marker_values_path: str,
    coverage_path: str,
    ground_truth_path: str,
    atlas_path: str,
    target_cell_type: str,
    target_cell_idx: int,
) -> tuple[DataLoader, int]:
    """Load parquet data for single-cell-type detector training.

    Extracts only the markers belonging to ``target_cell_type`` from the atlas,
    then builds a DataLoader.

    Returns
    -------
    loader : DataLoader
    num_markers : int
    """
    marker_values_df = pd.read_parquet(marker_values_path)
    coverage_df = pd.read_parquet(coverage_path)
    ground_truth_df = pd.read_parquet(ground_truth_path)

    y_true = ground_truth_df.iloc[:, target_cell_idx].values

    atlas = pd.read_csv(atlas_path, sep="\t")
    target_markers = atlas[atlas.target == target_cell_type]
    target_indices = target_markers.index.values
    logger.info("Using %d markers for %s", len(target_indices), target_cell_type)

    marker_values = marker_values_df.iloc[target_indices][marker_values_df.columns[2:]].values.T
    coverage = coverage_df.iloc[target_indices][coverage_df.columns[2:]].values.T

    logger.info("Marker values shape: %s, coverage shape: %s", marker_values.shape, coverage.shape)

    dataset = DetectorDataset(marker_values, coverage, y_true)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    return loader, marker_values.shape[1]


def prepare_detector_splits(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    atlas_path: str,
    target_cell_type: str,
    target_cell_idx: int,
    use_raw: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load train/val/test splits for detector training.

    Returns
    -------
    train_loader, val_loader, test_loader, num_markers
    """
    prefix = "raw_" if use_raw else ""
    loaders = []
    for data_dir in [train_dir, val_dir, test_dir]:
        loader, num_markers = load_detector_data(
            marker_values_path=os.path.join(data_dir, f"{prefix}marker_values.parquet"),
            coverage_path=os.path.join(data_dir, f"{prefix}coverage.parquet"),
            ground_truth_path=os.path.join(data_dir, f"{prefix}ground_truth_y.parquet"),
            atlas_path=atlas_path,
            target_cell_type=target_cell_type,
            target_cell_idx=target_cell_idx,
        )
        loaders.append(loader)
    return loaders[0], loaders[1], loaders[2], num_markers


def load_control_data(
    data_dir: str,
    atlas_path: str,
    target_cell_type: str,
) -> tuple[torch.Tensor, torch.Tensor, pd.Index]:
    """Load known-negative control samples for contrastive training.

    Returns marker_values, coverage (as tensors) and sample_ids.
    """
    marker_values_df = pd.read_parquet(os.path.join(data_dir, "marker_values.parquet"))
    coverage_df = pd.read_parquet(os.path.join(data_dir, "coverage.parquet"))

    atlas = pd.read_csv(atlas_path, sep="\t")
    target_indices = atlas[atlas.target == target_cell_type].index.values
    logger.info("Using %d markers for %s in control data", len(target_indices), target_cell_type)

    sample_ids = marker_values_df.columns[2:]
    marker_values = marker_values_df.iloc[target_indices][marker_values_df.columns[2:]].values.T
    coverage = coverage_df.iloc[target_indices][coverage_df.columns[2:]].values.T

    return (
        torch.tensor(marker_values, dtype=torch.float32),
        torch.tensor(coverage, dtype=torch.float32),
        sample_ids,
    )


def create_mixed_dataset(
    train_marker_values: torch.Tensor,
    train_coverage: torch.Tensor,
    train_y_true: torch.Tensor,
    control_marker_values: torch.Tensor,
    control_coverage: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge training samples with known-negative controls.

    Returns combined arrays and a boolean control mask.
    """
    train_size = train_marker_values.shape[0]
    control_size = control_marker_values.shape[0]

    control_mask = np.concatenate([
        np.zeros(train_size, dtype=bool),
        np.ones(control_size, dtype=bool),
    ])

    combined_values = np.concatenate([
        train_marker_values.numpy(),
        control_marker_values.numpy(),
    ], axis=0)
    combined_coverage = np.concatenate([
        train_coverage.numpy(),
        control_coverage.numpy(),
    ], axis=0)
    combined_y = np.concatenate([
        train_y_true.squeeze().numpy(),
        np.zeros(control_size),
    ])

    return combined_values, combined_coverage, combined_y, control_mask


def add_control_data_to_loader(
    train_loader: DataLoader,
    control_data_dir: str,
    atlas_path: str,
    target_cell_type: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Add control samples to training data and create a control validation loader.

    Returns
    -------
    train_loader : DataLoader — training data with controls mixed in
    control_val_loader : DataLoader — held-out controls for calibration
    """
    control_mv, control_cov, _ = load_control_data(
        control_data_dir, atlas_path, target_cell_type
    )

    control_size = len(control_mv)
    val_size = min(int(control_size * 0.2), 100)

    # Validation controls
    control_val_dataset = DetectorDataset(
        control_mv[:val_size].numpy(),
        control_cov[:val_size].numpy(),
        np.zeros(val_size),
        np.ones(val_size, dtype=bool),
    )
    control_val_loader = DataLoader(
        control_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Mixed training
    combined_values, combined_coverage, combined_y, control_mask = create_mixed_dataset(
        train_loader.dataset.marker_values,
        train_loader.dataset.coverage,
        train_loader.dataset.y_true,
        control_mv[val_size:],
        control_cov[val_size:],
    )
    mixed_dataset = DetectorDataset(combined_values, combined_coverage, combined_y, control_mask)
    mixed_loader = DataLoader(
        mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    logger.info(
        "Mixed dataset: %d samples (%d controls)",
        len(mixed_dataset), control_mask.sum(),
    )
    return mixed_loader, control_val_loader


# ---------------------------------------------------------------------------
# Deconvolution data loading
# ---------------------------------------------------------------------------

def load_deconv_parquets(
    data_dir: str,
    atlas: pd.DataFrame,
    marker_names: set,
    num_files: int = 1,
    prefix: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load deconvolution training data from parquet files.

    Returns marker_values (N, M), coverage (N, M), y (N, C) as numpy arrays.
    """
    markers_list, coverage_list, y_list = [], [], []
    for i in range(1, num_files + 1):
        p = f"{prefix}{i}_" if prefix else ""
        markers_list.append(pd.read_parquet(os.path.join(data_dir, f"{p}marker_values.parquet")))
        coverage_list.append(pd.read_parquet(os.path.join(data_dir, f"{p}coverage.parquet")))
        y_list.append(pd.read_parquet(os.path.join(data_dir, f"{p}ground_truth_y.parquet")))

    # Merge batches
    merged_markers = markers_list[0]
    merged_coverage = coverage_list[0]
    for i, (m, c) in enumerate(zip(markers_list[1:], coverage_list[1:])):
        suffix = f"_batch{i + 1}"
        merged_markers = merged_markers.merge(m, on=["name", "direction"], how="outer", suffixes=("", suffix))
        merged_coverage = merged_coverage.merge(c, on=["name", "direction"], how="outer", suffixes=("", suffix))

    y = pd.concat(y_list, ignore_index=True).fillna(0)

    # Filter to atlas markers
    X = merged_markers[merged_markers.name.isin(marker_names)]
    cov = merged_coverage[merged_coverage.name.isin(marker_names)]

    X_np = X.drop(columns=["name", "direction"]).T.to_numpy()
    cov_np = cov.drop(columns=["name", "direction"]).T.to_numpy()
    y_np = y.to_numpy()

    return X_np, cov_np, y_np


def prepare_deconv_input(
    atlas_path: str,
    eval_pat_dir: str,
    dilutions: list[float],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load evaluation data for deconvolution and attach dilution metadata.

    Returns X_val, coverage_val, y_true_df, y_dilutions.
    """
    atlas = pd.read_csv(atlas_path, sep="\t").dropna()
    names = set(atlas.name.unique())

    X_val = pd.read_parquet(os.path.join(eval_pat_dir, "marker_values.parquet"))
    coverage_val = pd.read_parquet(os.path.join(eval_pat_dir, "coverage.parquet"))
    y_val = pd.read_parquet(os.path.join(eval_pat_dir, "ground_truth_y.parquet"))

    y_val["sample"] = list(X_val.columns[2:])
    y_val["dilution"] = y_val["sample"].apply(
        lambda s: dilutions[int(s.split("_")[1][3:]) - 1]
    )
    y_dilutions = pd.DataFrame(y_val["dilution"].values, columns=["dilution"])
    y_val = y_val.drop(columns=["dilution", "sample"]).to_numpy()

    X_val = X_val[X_val.name.isin(names)].drop(columns=["name", "direction"]).T.to_numpy()
    coverage_val = coverage_val[coverage_val.name.isin(names)].drop(columns=["name", "direction"]).T.to_numpy()

    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    y_val_t = y_val_t / y_val_t.sum(dim=1, keepdim=True)
    y_true_df = pd.DataFrame(y_val_t.numpy(), columns=list(atlas.columns[8:]))

    return X_val, coverage_val, y_true_df, y_dilutions
