"""Tapestry deconvolution model.

Two-level hierarchical transformer for amortised inference of cell-type
proportions from cfDNA methylation count data. See docs/architecture.md
for the full design rationale.

Architecture:
    1. Per-marker embedding (counts + coverage -> feature_dim)
    2. Level 1: within-cell-type transformer (batched across cell types)
    3. Masked mean pooling -> one summary per cell type
    4. Level 2: cross-cell-type transformer (13 tokens)
    5. Proportion head: softplus masses -> normalise
    6. Detection head: per-cell-type presence probability (auxiliary)
    7. Decoder: proportions -> expected marker probabilities (for beta-binomial NLL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class MarkerEmbedding(nn.Module):
    """Embeds per-marker count observations into feature vectors."""

    def __init__(self, num_markers: int, feature_dim: int):
        super().__init__()
        self.projection = nn.Linear(4, feature_dim)
        self.marker_identity = nn.Parameter(
            torch.randn(1, num_markers, feature_dim) * 0.02
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, u: torch.Tensor, m: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        fraction = u / torch.clamp(c, min=1)
        channels = torch.stack([
            fraction, torch.log1p(u), torch.log1p(m), torch.log1p(c),
        ], dim=-1)
        return self.norm(self.projection(channels) + self.marker_identity)


class WithinCellTypeTransformer(nn.Module):
    """Level 1: shared transformer applied to all cell types in one batched call.

    Instead of looping over 13 cell types, pads all groups to the same size
    and processes them as a single (B*C, max_markers, D) batch.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4,
            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor,
        target_ids: torch.Tensor, num_cell_types: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process all cell types' markers through the shared transformer.

        Returns
        -------
        summaries : (B, C, D) — masked mean pooled per cell type.
        valid_counts : (B, C) — number of valid markers per cell type.
        """
        B, M, D = features.shape
        device = features.device

        # Build per-cell-type index lists
        group_indices = []
        max_group_size = 0
        for k in range(num_cell_types):
            idx = (target_ids == k).nonzero(as_tuple=True)[0]
            group_indices.append(idx)
            max_group_size = max(max_group_size, len(idx))

        # Pad all groups to max_group_size and stack into (B*C, max_group_size, D)
        padded_features = torch.zeros(B * num_cell_types, max_group_size, D, device=device)
        padded_mask = torch.ones(B * num_cell_types, max_group_size, dtype=torch.bool, device=device)

        for k, idx in enumerate(group_indices):
            n_k = len(idx)
            if n_k == 0:
                continue
            # features[:, idx] -> (B, n_k, D)
            padded_features[k * B : (k + 1) * B, :n_k] = features[:, idx]
            padded_mask[k * B : (k + 1) * B, :n_k] = mask[:, idx]

        # If a row has no valid keys (sample with zero coverage on every marker
        # of a cell type), softmax over an all-masked row returns NaN. Unmask
        # position 0 for those rows so the encoder runs cleanly; the original
        # padded_mask is used during pooling so the summary still collapses to 0.
        attn_mask = padded_mask.clone()
        all_masked = padded_mask.all(dim=1)
        if all_masked.any():
            attn_mask[all_masked, 0] = False

        # One transformer call for all cell types
        encoded = self.encoder(padded_features, src_key_padding_mask=attn_mask)

        # Masked mean pooling per cell type
        summaries = torch.zeros(B, num_cell_types, D, device=device)
        valid_counts = torch.zeros(B, num_cell_types, device=device)

        for k, idx in enumerate(group_indices):
            n_k = len(idx)
            if n_k == 0:
                continue
            group_encoded = encoded[k * B : (k + 1) * B, :n_k]  # (B, n_k, D)
            group_valid = ~padded_mask[k * B : (k + 1) * B, :n_k]  # (B, n_k)
            counts = group_valid.float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            summaries[:, k] = (group_encoded * group_valid.unsqueeze(-1).float()).sum(dim=1) / counts
            valid_counts[:, k] = counts.squeeze(1)

        return summaries, valid_counts


class CrossCellTypeTransformer(nn.Module):
    """Level 2: shallow transformer over C cell-type summary vectors."""

    def __init__(self, feature_dim: int, num_heads: int = 4, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4,
            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, cell_type_summaries: torch.Tensor) -> torch.Tensor:
        return self.encoder(cell_type_summaries)


class TapestryModel(nn.Module):
    """Two-level hierarchical transformer for cfDNA deconvolution.

    Parameters
    ----------
    num_markers, num_cell_types, target_ids, atlas : see docs/architecture.md
    feature_dim : int — internal feature dimension.
    l1_num_heads, l1_num_layers : Level 1 (within-cell-type) config.
    l2_num_heads, l2_num_layers : Level 2 (cross-cell-type) config.
    dropout : float
    """

    def __init__(
        self, num_markers: int, num_cell_types: int, target_ids, atlas,
        feature_dim: int = 64, l1_num_heads: int = 4, l1_num_layers: int = 2,
        l2_num_heads: int = 4, l2_num_layers: int = 1, dropout: float = 0.1,
    ):
        super().__init__()
        self.num_markers = num_markers
        self.num_cell_types = num_cell_types
        self.feature_dim = feature_dim

        self.register_buffer("target_ids", torch.as_tensor(target_ids, dtype=torch.long))
        self.register_buffer("atlas", torch.as_tensor(atlas, dtype=torch.float32))

        self.embedding = MarkerEmbedding(num_markers, feature_dim)
        self.level1 = WithinCellTypeTransformer(feature_dim, l1_num_heads, l1_num_layers, dropout)
        self.level2 = CrossCellTypeTransformer(feature_dim, l2_num_heads, l2_num_layers, dropout)

        self.proportion_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
        )
        self.detection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, u: torch.Tensor, m: torch.Tensor, c: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = c == 0
        embedded = self.embedding(u, m, c)
        # Zero out embeddings for markers with no coverage so they can't
        # leak signal through any path that bypasses the attention mask.
        embedded = embedded * (~mask).unsqueeze(-1).float()
        summaries, _ = self.level1(embedded, mask, self.target_ids, self.num_cell_types)
        refined = self.level2(summaries)

        logits = self.proportion_head(refined).squeeze(-1)  # (B, C)
        masses = F.softplus(logits)
        # Soft gating: multiply each mass by detection probability.
        # This allows the model to push absent types to near-zero through
        # the detection head, solving the sparsity problem without entmax.
        gates = torch.sigmoid(self.detection_head(refined).squeeze(-1))
        gated_masses = masses * gates
        proportions = gated_masses / gated_masses.sum(dim=1, keepdim=True).clamp(min=1e-8)
        expected_q = torch.matmul(proportions, self.atlas.T).clamp(1e-6, 1 - 1e-6)

        return {
            "proportions": proportions,
            "logits": logits,
            "gates": gates,
            "detection": gates,  # gates ARE the detection probabilities
            "expected_q": expected_q,
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self, u: torch.Tensor, m: torch.Tensor, c: torch.Tensor,
        n_samples: int = 50,
    ) -> dict[str, torch.Tensor]:
        """MC dropout inference: run forward pass n_samples times with dropout on.

        Returns
        -------
        dict with:
            proportions : (B, C) — mean prediction
            uncertainty : (B, C) — std across MC samples
            lower : (B, C) — 5th percentile
            upper : (B, C) — 95th percentile
            detection : (B, C) — mean detection probability
            quality : (B,) — reconstruction quality score (lower = more suspicious)
        """
        self.train()  # enable dropout
        all_props = []
        all_det = []
        for _ in range(n_samples):
            out = self.forward(u, m, c)
            all_props.append(out["proportions"])
            all_det.append(out["detection"])

        stacked = torch.stack(all_props)  # (n_samples, B, C)
        mean_props = stacked.mean(dim=0)
        std_props = stacked.std(dim=0)
        lower = torch.quantile(stacked, 0.05, dim=0)
        upper = torch.quantile(stacked, 0.95, dim=0)
        mean_det = torch.stack(all_det).mean(dim=0)

        # Reconstruction quality: how well do the predicted proportions
        # explain the observed marker values? Low residual = good fit.
        expected_q = torch.matmul(mean_props, self.atlas.T).clamp(1e-6, 1 - 1e-6)
        observed_frac = u / c.clamp(min=1)
        valid = c > 0
        residual = ((expected_q - observed_frac) ** 2 * valid.float()).sum(dim=1)
        n_valid = valid.float().sum(dim=1).clamp(min=1)
        quality = 1.0 - (residual / n_valid).clamp(max=1.0)  # 1.0 = perfect, 0.0 = terrible

        self.eval()
        return {
            "proportions": mean_props,
            "uncertainty": std_props,
            "lower": lower,
            "upper": upper,
            "detection": mean_det,
            "quality": quality,
        }
