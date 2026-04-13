"""DeepSets-based single-cell-type concentration detector (simplified variant).

A lighter alternative to the transformer detector. Uses a DeepSets
architecture (per-marker MLP + attention-weighted aggregation) instead of
cross-marker self-attention. No zero anchoring or bias correction —
trades some expressiveness for training stability.

Trained via ``tapestry.training.aggregator``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tapestry.models.layers import DynamicMarkerPruning, DeepSetsMarkerProcessor


class DeepSetsDetector(nn.Module):
    """DeepSets-based OAC concentration detector.

    Architecture
    ------------
    1. Multi-modal embedding: value + log-value + log-coverage -> feature_dim
    2. DeepSets phi-function (per-marker MLP)
    3. Attention-weighted aggregation to sample-level representation
    4. Single concentration head (sigmoid-scaled to max 10%)
    5. Uncertainty estimation

    Parameters
    ----------
    num_markers : int
        Number of input markers.
    feature_dim : int
        Internal feature dimension.
    dropout_rate : float
        Dropout rate for regularisation.
    min_reliable_coverage : float
        Coverage below this is treated as unreliable.
    """

    def __init__(
        self,
        num_markers: int,
        feature_dim: int = 16,
        dropout_rate: float = 0.2,
        min_reliable_coverage: float = 5.0,
    ):
        super().__init__()
        self.min_reliable_coverage = min_reliable_coverage
        self.num_markers = num_markers

        self.marker_pruning = DynamicMarkerPruning(min_reliable_coverage)

        # Multi-modal feature embedding
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.log_value_embedding = nn.Linear(1, feature_dim // 2)
        self.value_bn = nn.BatchNorm1d(feature_dim // 2)
        self.coverage_bn = nn.BatchNorm1d(feature_dim // 2)
        self.log_value_bn = nn.BatchNorm1d(feature_dim // 2)

        self.feature_projection = nn.Linear(feature_dim * 3 // 2, feature_dim)

        # DeepSets marker processor (phi function)
        self.marker_processor = DeepSetsMarkerProcessor(
            feature_dim=feature_dim, hidden_dim=64, dropout_rate=dropout_rate
        )

        # Attention aggregation (rho function)
        self.attention = nn.Linear(feature_dim, 1)

        # Single concentration head
        self.concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim // 4, 1),
        )

        # Uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus(),
        )

        # Calibration buffers
        self.register_buffer("calibration", torch.ones(1))
        self.register_buffer("clinical_threshold", torch.tensor(0.001))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

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
        zero_prob : (B, 1)  — always zeros (no zero-anchoring in this model)
        """
        marker_values = torch.clamp(marker_values, 0.0, 1.0)
        coverage = torch.clamp(coverage, 0.0, 1000.0)

        pruned = self.marker_pruning(marker_values, coverage)
        cov_reliability = torch.clamp(
            coverage / self.min_reliable_coverage, 0.1, 1.0
        )
        weighted = torch.nan_to_num(pruned * cov_reliability, nan=0.0)

        missing_mask = coverage == 0
        unreliable_mask = (coverage < self.min_reliable_coverage) & ~missing_mask
        combined_mask = missing_mask | unreliable_mask

        B, M = weighted.shape

        # Embeddings
        val_feat = self.value_embedding(weighted.unsqueeze(-1))
        val_feat = self.value_bn(val_feat.reshape(B * M, -1)).reshape(B, M, -1)

        log_vals = torch.log1p(weighted * 100)
        log_feat = self.log_value_embedding(log_vals.unsqueeze(-1))
        log_feat = self.log_value_bn(log_feat.reshape(B * M, -1)).reshape(B, M, -1)

        log_cov = torch.log1p(coverage).unsqueeze(-1)
        cov_feat = self.coverage_embedding(log_cov)
        cov_feat = self.coverage_bn(cov_feat.reshape(B * M, -1)).reshape(B, M, -1)

        features = self.feature_projection(
            torch.cat([val_feat, cov_feat, log_feat], dim=-1)
        )

        # DeepSets phi
        processed = self.marker_processor(features, key_padding_mask=combined_mask)

        # Attention-weighted aggregation (rho)
        reliability = cov_reliability.unsqueeze(-1)
        attn_scores = self.attention(processed).squeeze(-1)
        attn_scores = attn_scores * reliability.squeeze(-1)
        attn_scores = attn_scores.masked_fill(missing_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        aggregated = torch.sum(attn_weights.unsqueeze(-1) * processed, dim=1)

        # Concentration (capped at 10%)
        raw = self.concentration_head(aggregated)
        concentration = torch.sigmoid(raw) * 0.1

        # Coverage dampening
        mean_cov = coverage.mean(dim=1, keepdim=True)
        cov_factor = torch.clamp(
            mean_cov / (self.min_reliable_coverage * 2.0), 0.3, 1.0
        )
        concentration = torch.clamp(concentration * cov_factor, 0.0, 1.0)

        uncertainty = self.uncertainty_head(aggregated)
        zero_prob = torch.zeros_like(concentration)

        return concentration, uncertainty, attn_weights, zero_prob

    # ------------------------------------------------------------------
    # Inference helpers (same interface as TransformerDetector)
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
        detected = concentration >= dynamic_thresh
        thresholded = torch.where(
            detected, concentration, torch.zeros_like(concentration)
        )
        return thresholded, uncertainty, attn_weights, detected
