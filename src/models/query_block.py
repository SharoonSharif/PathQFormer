"""Modality-specific Q-Former query block: learnable queries cross-attend into a token sequence."""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityQueryBlock(nn.Module):
    """K learnable queries compress a variable-length token sequence into a fixed (K, d) code.

    Each layer: CrossAttention(Q, KV=input) -> SelfAttention(Q) -> FFN(Q).
    ``norm_first=True`` (default) uses pre-norm residual blocks, which are markedly more stable
    for training from scratch on a few hundred patients; ``False`` reproduces the original
    post-norm variant. Projected inputs are LayerNormed once so attention logits do not scale
    with the (arbitrary) norm of the frozen encoder features.
    """

    def __init__(
        self,
        num_queries: int = 32,
        hidden_dim: int = 256,
        input_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList(
            [QueryBlockLayer(hidden_dim, num_heads, dropout, norm_first=norm_first) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (B, N, input_dim) token sequence (patches or pathways)
            mask: (B, N) True for valid tokens, False for padding
            return_attention: also return the last layer's head-averaged cross-attention (B, K, N)
        Returns:
            (B, K, hidden_dim) query codes, and attention weights or None
        """
        B = x.shape[0]
        x = self.input_norm(self.input_proj(x))
        q = self.queries.expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None

        attn = None
        for i, layer in enumerate(self.layers):
            want = return_attention and i == len(self.layers) - 1
            q, attn = layer(q, x, key_padding_mask=key_padding_mask, need_weights=want)
        return self.norm(q), attn


class QueryBlockLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, norm_first: bool = True):
        super().__init__()
        self.norm_first = norm_first
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.norm_first:
            out, attn = self.cross_attn(
                self.cross_norm(q), kv, kv, key_padding_mask=key_padding_mask, need_weights=need_weights
            )
            q = q + out
            h = self.self_norm(q)
            out, _ = self.self_attn(h, h, h, need_weights=False)
            q = q + out
            q = q + self.ffn(self.ffn_norm(q))
            return q, attn

        out, attn = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=need_weights)
        q = self.cross_norm(q + out)
        out, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.self_norm(q + out)
        q = self.ffn_norm(q + self.ffn(q))
        return q, attn
