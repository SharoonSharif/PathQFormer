"""Data augmentation transforms for WSI feature bags."""

import torch
import numpy as np


class FeatureBagAugmentation:
    """Lightweight augmentations for pre-extracted WSI feature bags."""

    def __init__(self, noise_std: float = 0.0, dropout_rate: float = 0.0):
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            features = features + torch.randn_like(features) * self.noise_std

        if self.dropout_rate > 0:
            mask = torch.rand(features.shape[0]) > self.dropout_rate
            features = features[mask]
            if features.shape[0] == 0:
                features = torch.zeros(1, features.shape[1])

        return features
