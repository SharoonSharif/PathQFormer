"""Cross-modal fusion block: self-attention over concatenated modality query outputs."""

import torch
import torch.nn as nn


class CrossModalFusionBlock(nn.Module):
    """
    Standard Transformer encoder that fuses concatenated query outputs [Z_h; Z_g].
    Enables bidirectional interaction: histology queries attend to genomic queries
    and vice versa.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_h: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_h: (B, K, d) — histology query outputs
            z_g: (B, K, d) — genomic query outputs
        Returns:
            (B, 2K, d) — fused representation
        """
        z = torch.cat([z_h, z_g], dim=1)
        z = self.encoder(z)
        return self.norm(z)
