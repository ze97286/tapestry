"""Per-cell-type binary presence classifier.

Predicts whether a specific cell type is present (above a threshold) in a
cfDNA sample. One model is trained per cell type. Used by the full
deconvolution model for presence gating.

Architecture: multi-resolution feature extraction (pooling at k=1,3,7),
attention-weighted aggregation, coverage-adaptive thresholding.

Trained via ``tapestry.training.presence``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleCellTypePresenceModel(nn.Module):
    """Binary classifier for cell type presence detection.

    Parameters
    ----------
    feature_dim : int
        Internal feature dimension.
    dropout_rate : float
        Dropout rate for regularisation.
    """

    def __init__(self, feature_dim: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.specificity_threshold = 0.5

        self.input_norm = nn.BatchNorm1d(2)

        self.feature_extractor = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.multi_res_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.attention = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 1, 64),  # +1 for missing rate
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(self.classifier[-1], "bias"):
            self.classifier[-1].bias.data.fill_(0.0)

    def _extract_features(self, markers: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        features_input = torch.cat([markers, coverage], dim=1)
        features_norm = self.input_norm(features_input)
        return self.feature_extractor(features_norm)

    def forward(
        self, marker_values: torch.Tensor, coverage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        logits : (B, 1)
        attention_weights : (B, M) — normalised attention per marker
        missing_rate : (B, 1)
        """
        B, M = marker_values.shape
        valid_mask = coverage > 0
        missing_rate = 1.0 - valid_mask.float().mean(dim=1, keepdim=True)
        safe_values = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))

        coverage_safe = coverage.clone() + 1e-10
        confidence = torch.clamp(
            torch.where(
                coverage_safe < 5.0,
                coverage_safe / (coverage_safe + 15.0),
                coverage_safe / (coverage_safe + 10.0),
            ),
            0.2, 1.0,
        )
        normalised = safe_values * confidence

        # Multi-resolution analysis
        markers_flat = normalised.reshape(-1, 1)
        cov_flat = torch.log1p(coverage_safe).reshape(-1, 1)
        feat_orig = self._extract_features(markers_flat, cov_flat)

        markers_med = F.avg_pool1d(normalised.view(B, 1, M), 3, 1, 1).view(B, M)
        cov_med = F.avg_pool1d(coverage_safe.view(B, 1, M), 3, 1, 1).view(B, M)
        feat_med = self._extract_features(markers_med.reshape(-1, 1), torch.log1p(cov_med).reshape(-1, 1))

        markers_low = F.avg_pool1d(normalised.view(B, 1, M), 7, 1, 3).view(B, M)
        cov_low = F.avg_pool1d(coverage_safe.view(B, 1, M), 7, 1, 3).view(B, M)
        feat_low = self._extract_features(markers_low.reshape(-1, 1), torch.log1p(cov_low).reshape(-1, 1))

        features = self.multi_res_fusion(torch.cat([feat_orig, feat_med, feat_low], dim=1))
        features = features + self.feature_transform(features)

        # Attention
        attn_flat = self.attention(features).reshape(B, M)
        masked_attn = attn_flat * valid_mask.float() * confidence
        attn_sum = masked_attn.sum(dim=1, keepdim=True)
        attn_sum = torch.where(attn_sum > 0, attn_sum, torch.ones_like(attn_sum))
        norm_attn = masked_attn / attn_sum

        # Aggregate
        feat_reshaped = features.reshape(B, M, self.feature_dim)
        weighted = feat_reshaped * norm_attn.unsqueeze(-1)
        aggregated = weighted.sum(dim=1)

        enhanced = torch.cat([aggregated, missing_rate], dim=1)
        logits = self.classifier(enhanced)

        return logits, norm_attn, missing_rate

    def load_threshold(self, checkpoint: dict):
        if "specificity_threshold" in checkpoint:
            self.specificity_threshold = checkpoint["specificity_threshold"]
        else:
            self.specificity_threshold = 0.5

    def predict(
        self, marker_values: torch.Tensor, coverage: torch.Tensor,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _, _ = self.forward(marker_values, coverage)
        probs = torch.sigmoid(logits).squeeze(-1)
        thresh = threshold if threshold is not None else self.specificity_threshold
        preds = (probs >= thresh).float()
        return preds, probs

    def adaptive_predict(
        self, marker_values: torch.Tensor, coverage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _, missing_rate = self.forward(marker_values, coverage)
        probs = torch.sigmoid(logits).squeeze(-1)
        mean_cov = coverage.mean(dim=1)
        cov_adj = torch.clamp(0.02 - 0.001 * mean_cov, 0.0, 0.02)
        miss_adj = torch.clamp(0.02 * missing_rate.squeeze(), 0.0, 0.02)
        total_adj = torch.clamp(cov_adj + miss_adj, 0.0, 0.03)
        adaptive_thresh = self.specificity_threshold + total_adj
        preds = (probs >= adaptive_thresh).float()
        return preds, probs, adaptive_thresh
