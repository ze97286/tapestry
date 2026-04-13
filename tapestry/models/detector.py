"""Transformer-based single-cell-type concentration detector.

Estimates the concentration of a single target cell type (e.g. OAC) from
per-marker methylation values and coverage. Uses multi-modal feature embedding,
a transformer encoder for cross-marker attention, and mixture-of-experts
concentration heads for different concentration ranges.

Trained via ``tapestry.training.detector``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tapestry.models.layers import (
    DynamicMarkerPruning,
    ResidualBiasCorrectionLayer,
    ZeroAnchoringLayer,
)


class TransformerDetector(nn.Module):
    """Transformer-based OAC concentration detector.

    Architecture
    ------------
    1. Multi-modal embedding: value + log-value + log-coverage -> feature_dim
    2. Learnable marker identity embedding (positional)
    3. Transformer encoder (cross-marker self-attention)
    4. Attention-weighted aggregation to sample-level representation
    5. Mixture-of-experts concentration heads (standard / low / ultra-low)
    6. Zero anchoring + bias correction
    7. Uncertainty estimation

    Parameters
    ----------
    num_markers : int
        Number of input markers.
    feature_dim : int
        Internal feature dimension. Must be divisible by ``num_heads``.
    num_heads : int
        Number of transformer attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dropout_rate : float
        Dropout rate for regularisation.
    min_reliable_coverage : float
        Coverage below this is treated as unreliable.
    """

    def __init__(
        self,
        num_markers: int,
        feature_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        min_reliable_coverage: float = 5.0,
    ):
        super().__init__()
        self.min_reliable_coverage = min_reliable_coverage
        self.num_markers = num_markers

        # Marker pruning
        self.marker_pruning = DynamicMarkerPruning(min_reliable_coverage)

        # Multi-modal feature embedding
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.log_value_embedding = nn.Linear(1, feature_dim // 2)
        self.value_bn = nn.BatchNorm1d(feature_dim // 2)
        self.coverage_bn = nn.BatchNorm1d(feature_dim // 2)
        self.log_value_bn = nn.BatchNorm1d(feature_dim // 2)

        # Projection and positional embedding
        self.feature_projection = nn.Linear(feature_dim * 3 // 2, feature_dim)
        self.marker_identity_embedding = nn.Parameter(
            torch.randn(1, num_markers, feature_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 3,
            dropout=dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Attention aggregation
        self.attention = nn.Linear(feature_dim, 1)

        # Mixture-of-experts concentration heads
        self.concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2), nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
        )
        self.low_concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim, feature_dim // 2), nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim // 2, 1),
        )
        self.ultra_low_concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim, feature_dim // 2), nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
        )
        self.concentration_gate = nn.Sequential(
            nn.Linear(feature_dim, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 2), nn.Softmax(dim=1),
        )

        # Uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1), nn.Softplus(),
        )

        # Post-processing
        self.bias_correction = ResidualBiasCorrectionLayer(feature_dim)
        self.zero_anchoring = ZeroAnchoringLayer(feature_dim)

        # Calibration buffers
        self.register_buffer("calibration", torch.ones(1))
        self.register_buffer("clinical_threshold", torch.tensor(0.001))

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed_features(
        self, marker_values: torch.Tensor, coverage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce multi-modal embeddings and masks."""
        marker_values_pruned = self.marker_pruning(marker_values, coverage)
        coverage_reliability = 1.0 - torch.exp(
            -coverage / self.min_reliable_coverage
        )
        coverage_reliability = torch.clamp(coverage_reliability, 0.01, 1.0)
        marker_values_weighted = marker_values_pruned * coverage_reliability
        marker_values_weighted = torch.nan_to_num(marker_values_weighted, nan=0.0)

        missing_mask = coverage == 0
        unreliable_mask = (coverage < self.min_reliable_coverage) & ~missing_mask
        combined_mask = missing_mask | unreliable_mask

        B, M = marker_values_weighted.shape

        # Value features
        val_feat = self.value_embedding(marker_values_weighted.unsqueeze(-1))
        val_feat = self.value_bn(val_feat.reshape(B * M, -1)).reshape(B, M, -1)

        # Log-value features
        log_vals = torch.log1p(marker_values_weighted * 100)
        log_feat = self.log_value_embedding(log_vals.unsqueeze(-1))
        log_feat = self.log_value_bn(log_feat.reshape(B * M, -1)).reshape(B, M, -1)

        # Coverage features
        log_cov = torch.log1p(coverage).unsqueeze(-1)
        cov_feat = self.coverage_embedding(log_cov)
        cov_feat = self.coverage_bn(cov_feat.reshape(B * M, -1)).reshape(B, M, -1)

        features = torch.cat([val_feat, cov_feat, log_feat], dim=-1)
        features = self.feature_projection(features)
        features = features + self.marker_identity_embedding

        return features, combined_mask, coverage_reliability

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, marker_values: torch.Tensor, coverage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        concentration : (B, 1)
        uncertainty : (B, 1)
        attention_weights : (B, M)
        zero_prob : (B, 1)
        """
        features, combined_mask, coverage_reliability = self._embed_features(
            marker_values, coverage
        )

        # Transformer
        transformer_out = self.transformer_encoder(
            features, src_key_padding_mask=combined_mask
        )

        # Attention aggregation
        reliability = coverage_reliability.unsqueeze(-1)
        attn_scores = self.attention(transformer_out).squeeze(-1)
        attn_scores = attn_scores * reliability.squeeze(-1)
        attn_scores = attn_scores.masked_fill(coverage == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        aggregated = torch.sum(attn_weights.unsqueeze(-1) * transformer_out, dim=1)

        # Mixture-of-experts
        standard = F.softplus(self.concentration_head(aggregated)) * 0.2
        low = F.softplus(self.low_concentration_head(aggregated)) * 0.01
        ultra_low = F.softplus(self.ultra_low_concentration_head(aggregated)) * 0.001

        gates = self.concentration_gate(aggregated)
        ultra_gate = 1.0 - gates.sum(dim=1, keepdim=True)
        concentration = (
            gates[:, 0:1] * standard
            + gates[:, 1:2] * low
            + ultra_gate * ultra_low
        )

        # Post-processing
        concentration = self.bias_correction(aggregated, concentration)
        concentration, zero_prob = self.zero_anchoring(aggregated, concentration)

        mean_cov = coverage.mean(dim=1, keepdim=True)
        cov_factor = torch.clamp(
            mean_cov / (self.min_reliable_coverage * 2.0), 0.2, 1.0
        )
        concentration = torch.clamp(concentration * cov_factor, 0.0, 1.0)

        uncertainty = self.uncertainty_head(aggregated)

        return concentration, uncertainty, attn_weights, zero_prob

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def get_estimate_and_ci(
        self,
        mu: torch.Tensor,
        uncertainty: torch.Tensor,
        ci_level: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_thresh = torch.where(
            mu >= self.clinical_threshold, mu, torch.zeros_like(mu)
        )
        scaled = uncertainty * self.calibration
        z = 1.96 if ci_level == 0.95 else float(
            torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + ci_level) / 2))
        )
        lower = torch.clamp(mu_thresh - z * scaled, min=0.0)
        upper = torch.clamp(mu_thresh + z * scaled, max=1.0)
        return mu_thresh, torch.cat([lower, upper], dim=1), scaled

    def predict_with_clinical_threshold(
        self, marker_values: torch.Tensor, coverage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        concentration, uncertainty, attn_weights, zero_prob = self.forward(
            marker_values, coverage
        )
        mean_cov = coverage.mean(dim=1, keepdim=True)
        low_cov_mask = mean_cov < (self.min_reliable_coverage * 1.5)
        dynamic_thresh = torch.where(
            low_cov_mask,
            self.clinical_threshold * 2.0,
            self.clinical_threshold,
        )
        adjusted = torch.where(
            zero_prob > 0.7, torch.zeros_like(concentration), concentration
        )
        detected = adjusted >= dynamic_thresh
        thresholded = torch.where(detected, adjusted, torch.zeros_like(adjusted))
        return thresholded, uncertainty, attn_weights, detected
