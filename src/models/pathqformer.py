"""PathQ-Former: Q-Former-based multimodal fusion for computational pathology survival prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from .query_block import ModalityQueryBlock
from .fusion_block import CrossModalFusionBlock
from .survival_head import SurvivalHead
from .null_tokens import NullTokenModule


class PathQFormer(nn.Module):
    """Two modality query blocks -> learned null codes for absent modalities -> fusion -> hazards.

    Missing modalities are handled uniformly at train and test time: an absent modality's K
    query codes are replaced by a learned null code. During training each *sample*
    independently drops each present modality with probability ``modality_dropout`` (never
    both), so the fusion block learns to read null codes as "no information".
    """

    def __init__(
        self,
        wsi_input_dim: int = 1024,
        genomic_input_dim: int = 256,
        hidden_dim: int = 256,
        num_queries: int = 32,
        num_heads: int = 8,
        query_layers: int = 2,
        fusion_layers: int = 2,
        num_bins: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.15,
        norm_first: bool = True,
    ):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.num_queries = num_queries

        self.histology_query_block = ModalityQueryBlock(
            num_queries=num_queries, hidden_dim=hidden_dim, input_dim=wsi_input_dim,
            num_heads=num_heads, num_layers=query_layers, dropout=dropout, norm_first=norm_first,
        )
        self.genomic_query_block = ModalityQueryBlock(
            num_queries=num_queries, hidden_dim=hidden_dim, input_dim=genomic_input_dim,
            num_heads=num_heads, num_layers=query_layers, dropout=dropout, norm_first=norm_first,
        )
        self.fusion_block = CrossModalFusionBlock(
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=fusion_layers, dropout=dropout,
        )
        self.null_tokens = NullTokenModule(num_queries=num_queries, hidden_dim=hidden_dim)
        self.survival_head = SurvivalHead(hidden_dim=hidden_dim, num_bins=num_bins, dropout=dropout)

    # ------------------------------------------------------------------------------------
    def forward(
        self,
        wsi_features: torch.Tensor | None = None,
        genomic_features: torch.Tensor | None = None,
        wsi_mask: torch.Tensor | None = None,
        genomic_mask: torch.Tensor | None = None,
        drop_wsi: torch.Tensor | None = None,
        drop_genomic: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """
        Args:
            wsi_features: (B, N, wsi_input_dim) or None when the modality is absent for the batch
            genomic_features: (B, P, genomic_input_dim) or None
            wsi_mask / genomic_mask: (B, N) / (B, P) True for valid tokens
            drop_wsi / drop_genomic: optional (B,) bool masks forcing per-sample absence
                (used for missing-modality evaluation; sampled automatically in training)
            return_attention: also return last-layer cross-attention maps per modality
        Returns:
            hazard_logits (B, num_bins) [, {"histology": (B,K,N)|None, "genomic": (B,K,P)|None}]
        """
        ref = wsi_features if wsi_features is not None else genomic_features
        if ref is None:
            raise ValueError("At least one modality must be provided")
        B, device = ref.shape[0], ref.device
        has_wsi, has_gen = wsi_features is not None, genomic_features is not None

        if self.training and self.modality_dropout > 0 and drop_wsi is None and drop_genomic is None:
            drop_wsi, drop_genomic = self._sample_modality_dropout(B, has_wsi, has_gen, device)

        attn_h = attn_g = None
        if has_wsi:
            z_h, attn_h = self.histology_query_block(wsi_features, mask=wsi_mask, return_attention=return_attention)
            if drop_wsi is not None and bool(drop_wsi.any()):
                z_h = torch.where(drop_wsi.view(B, 1, 1), self.null_tokens.get_null("histology", B), z_h)
        else:
            z_h = self.null_tokens.get_null("histology", B)

        if has_gen:
            z_g, attn_g = self.genomic_query_block(genomic_features, mask=genomic_mask, return_attention=return_attention)
            if drop_genomic is not None and bool(drop_genomic.any()):
                z_g = torch.where(drop_genomic.view(B, 1, 1), self.null_tokens.get_null("genomic", B), z_g)
        else:
            z_g = self.null_tokens.get_null("genomic", B)

        z_fused = self.fusion_block(z_h, z_g)
        logits = self.survival_head(z_fused)
        if return_attention:
            return logits, {"histology": attn_h, "genomic": attn_g}
        return logits

    def _sample_modality_dropout(
        self, B: int, has_wsi: bool, has_gen: bool, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample Bernoulli drops. A sample never loses both modalities, and a modality that
        is the only one present is never dropped."""
        zeros = torch.zeros(B, dtype=torch.bool, device=device)
        if not (has_wsi and has_gen):
            return zeros, zeros.clone()
        p = self.modality_dropout
        d_w = torch.rand(B, device=device) < p
        d_g = torch.rand(B, device=device) < p
        both = d_w & d_g
        if bool(both.any()):
            keep_wsi = torch.rand(B, device=device) < 0.5
            d_w = d_w & ~(both & keep_wsi)
            d_g = d_g & ~(both & ~keep_wsi)
        return d_w, d_g

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
