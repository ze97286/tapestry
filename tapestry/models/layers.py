"""Shared neural network layers used across detector and deconvolution models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMarkerPruning(nn.Module):
    """Zeros out marker values where coverage falls below a threshold."""

    def __init__(self, low_coverage_threshold: float = 3.0):
        super().__init__()
        self.low_coverage_threshold = low_coverage_threshold

    def forward(self, marker_values: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        marker_values_pruned = marker_values.clone()
        marker_values_pruned[coverage < self.low_coverage_threshold] = 0.0
        return marker_values_pruned


class ZeroAnchoringLayer(nn.Module):
    """Learns to distinguish true-zero samples from trace concentrations.

    Applies exponential dampening to the predicted concentration based on a
    learned zero-probability, pushing likely-absent predictions towards zero.
    """

    def __init__(self, feature_dim: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.1))
        layers += [
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.1))
        layers += [
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid(),
        ]
        self.zero_detector = nn.Sequential(*layers)
        self.sharpness = nn.Parameter(torch.tensor(12.0))

    def forward(
        self, features: torch.Tensor, concentration: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features_flat = features.reshape(-1, features.size(-1))
        zero_prob_flat = self.zero_detector(features_flat)
        zero_prob = zero_prob_flat.reshape(concentration.shape)
        zero_factor = torch.exp(-self.sharpness * zero_prob)
        anchored = concentration * zero_factor
        return anchored, zero_prob


class ResidualBiasCorrectionLayer(nn.Module):
    """Small multiplicative correction to reduce systematic bias.

    Operates in log-space to produce a correction factor around 1.0, scaled
    by a learnable (initially small) parameter.
    """

    def __init__(self, feature_dim: int, use_dropout: bool = False):
        super().__init__()
        layers = [nn.Linear(feature_dim + 1, feature_dim // 2), nn.GELU()]
        if use_dropout:
            layers.append(nn.Dropout(0.1))
        layers += [
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Tanh(),
        ]
        self.correction_network = nn.Sequential(*layers)
        self.correction_scale = nn.Parameter(torch.tensor(0.05))

    def forward(
        self, features: torch.Tensor, initial_pred: torch.Tensor
    ) -> torch.Tensor:
        log_pred = torch.log10(torch.clamp(initial_pred, 1e-6, 1.0))
        input_features = torch.cat([features, log_pred], dim=1)
        correction = self.correction_network(input_features) * self.correction_scale
        corrected = initial_pred * torch.exp(correction)
        return torch.clamp(corrected, 0.0, 1.0)


class DeepSetsMarkerProcessor(nn.Module):
    """Deep Sets phi-function: processes each marker independently.

    f(markers) = rho(sum(phi(marker_i)))
    This module implements phi; the rho aggregation is done via attention
    in the parent model.
    """

    def __init__(self, feature_dim: int = 16, hidden_dim: int = 64,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.marker_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self, features: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, num_markers, feat_dim = features.shape
        encoded = self.marker_encoder(features.reshape(-1, feat_dim))
        encoded = encoded.reshape(batch_size, num_markers, self.hidden_dim)
        output = self.output_projection(encoded)
        if key_padding_mask is not None:
            output = output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return output
