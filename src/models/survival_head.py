"""Survival prediction head and NLL-Survival loss for discrete-time hazard prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurvivalHead(nn.Module):
    """Attention-pooled aggregation followed by discrete hazard prediction over ``num_bins``."""

    def __init__(self, hidden_dim: int = 256, num_bins: int = 4, dropout: float = 0.25):
        super().__init__()
        self.attention_pool = AttentionPool(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 2K, d) fused tokens -> (B, num_bins) hazard logits."""
        return self.classifier(self.attention_pool(z))


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attn(x), dim=1)  # (B, N, 1)
        return (weights * x).sum(dim=1)


def hazards_to_survival(hazard_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """logits (B, T) -> (hazards h_t, survival S_t = prod_{k<=t}(1-h_k))."""
    hazards = torch.sigmoid(hazard_logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return hazards, survival


def risk_from_logits(hazard_logits: torch.Tensor) -> torch.Tensor:
    """Standard MCAT/SurvPath risk: negative expected survival, -sum_t S_t (higher = worse)."""
    _, survival = hazards_to_survival(hazard_logits)
    return -survival.sum(dim=1)


class NLLSurvivalLoss(nn.Module):
    """Negative log-likelihood for discrete-time survival (Zadeh & Schmid 2020), as in MCAT/SurvPath.

    With hazards h_k = sigmoid(logit_k) and S_j = prod_{k<=j}(1 - h_k):
        uncensored (event in bin t):  -log S_{t-1} - log h_t
        censored at bin t:            -log S_t
    ``alpha`` mixes in the uncensored term once more, exactly as MCAT's ``nll_loss``:
        loss = (1 - alpha) * (uncensored + censored) + alpha * uncensored
    """

    def __init__(self, alpha: float = 0.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        hazard_logits: torch.Tensor,
        survival_time_bin: torch.Tensor,
        censorship: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hazard_logits: (B, num_bins)
            survival_time_bin: (B,) 0-indexed bin of the event / censoring time
            censorship: (B,) 1 if censored, 0 if the event was observed
        """
        hazards, S = hazards_to_survival(hazard_logits)
        S_padded = torch.cat([torch.ones(S.shape[0], 1, device=S.device, dtype=S.dtype), S], dim=1)

        idx = torch.arange(hazard_logits.shape[0], device=hazard_logits.device)
        t = survival_time_bin.long().clamp(min=0, max=hazards.shape[1] - 1)
        c = censorship.float()

        s_prev = S_padded[idx, t].clamp(min=self.eps)  # S_{t-1}
        s_curr = S[idx, t].clamp(min=self.eps)         # S_t
        h_curr = hazards[idx, t].clamp(min=self.eps)   # h_t

        uncensored = -(1.0 - c) * (torch.log(s_prev) + torch.log(h_curr))
        censored = -c * torch.log(s_curr)
        loss = (1.0 - self.alpha) * (uncensored + censored) + self.alpha * uncensored
        return loss.mean()
