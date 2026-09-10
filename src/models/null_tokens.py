"""Learned null token module for missing-modality robustness."""

import torch
import torch.nn as nn


class NullTokenModule(nn.Module):
    """
    Maintains learned null embeddings for each modality. When a modality is absent,
    its query output is replaced with the corresponding null token set.
    Initialized near zero so the fusion block learns to interpret them as 'no information'.
    """

    def __init__(self, num_queries: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.null_histology = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.01)
        self.null_genomic = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.01)

    def get_null(self, modality: str, batch_size: int) -> torch.Tensor:
        if modality == "histology":
            return self.null_histology.expand(batch_size, -1, -1)
        elif modality == "genomic":
            return self.null_genomic.expand(batch_size, -1, -1)
        else:
            raise ValueError(f"Unknown modality: {modality}")
