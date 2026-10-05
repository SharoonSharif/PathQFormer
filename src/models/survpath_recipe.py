"""SurvPath + PathQ-Former's missing-modality "recipe" (null tokens, modality dropout, auxiliary heads).

The robustness comparison in the paper contrasts PathQ-Former (trained with modality dropout and learned
null codes) against the official SurvPath, which has neither and therefore has to be evaluated with mean
imputation. That confounds the architecture with the training recipe. ``SurvPathRecipe`` isolates the
recipe: it reuses the official SurvPath submodules untouched (per-pathway SNNs, WSI projection,
co-attention, feed-forward, logits) and only changes *what the co-attention sees* when a modality is absent:

* histology  -> a learned set of ``num_null_patches`` null patch tokens in the projected WSI space
* transcriptomics -> one learned null pathway token per pathway, in the pathway-token space

Both are inserted after the respective encoders and before co-attention, so the official fusion code runs
unchanged. Training uses PathQ-Former's per-sample modality-dropout sampler (never both modalities), and
optional auxiliary unimodal heads score the mean-pooled pre-fusion tokens of each branch, returned in the
``return_aux`` format ``src/training/train.py`` expects (``aux_unimodal_weight`` works unchanged).

Usage (config): ``model_type: survpath_recipe`` plus ``modality_dropout`` / ``aux_unimodal_weight`` /
``eval_missing`` as for PathQ-Former; see ``configs/protocol_fixed/survpath_recipe_e20.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .pathqformer import PathQFormer


class SurvPathRecipe(nn.Module):
    """Official SurvPath submodules + learned null tokens + modality dropout + optional aux heads (batch 1).

    The pathway groups are the same filtered composition that ``SurvPathBaseline`` and PathQ-Former's
    tokenizer use (``min_genes`` / ``max_genes``), so all models see identical genomic inputs.
    """

    supports_missing = True
    modality = "both"

    def __init__(
        self,
        pathway_composition: dict[str, list[int]],
        wsi_input_dim: int = 1536,
        num_bins: int = 4,
        dropout: float = 0.1,
        wsi_projection_dim: int = 256,
        min_genes: int = 3,
        max_genes: int = 300,
        survpath_dir: str | Path = "data/survpath_repo",
        modality_dropout: float = 0.15,
        aux_heads: bool = False,
        num_null_patches: int = 32,
    ):
        super().__init__()
        repo = str(Path(survpath_dir).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from models.model_SurvPath import SurvPath  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(f"Could not import the official SurvPath model from {repo} (pip install einops?)") from exc

        self.modality_dropout = float(modality_dropout)
        self.aux_heads = bool(aux_heads)
        self.num_null_patches = int(num_null_patches)

        self.pathway_names: list[str] = []
        index: list[torch.Tensor] = []
        for name, idx in sorted(pathway_composition.items()):
            if len(idx) < min_genes:
                continue
            self.pathway_names.append(name)
            index.append(torch.tensor(idx[:max_genes], dtype=torch.long))
        for i, t in enumerate(index):
            self.register_buffer(f"_idx{i}", t, persistent=False)
        self.num_pathways = len(index)
        self.net = SurvPath(
            omic_sizes=[len(t) for t in index],
            wsi_embedding_dim=wsi_input_dim,
            dropout=dropout,
            num_classes=num_bins,
            wsi_projection_dim=wsi_projection_dim,
        )
        # the official SNN blocks emit 256-d pathway tokens; co-attention requires them to match the WSI projection
        token_dim = self.net.sig_networks[0][-1][0].out_features if self.num_pathways else wsi_projection_dim
        if token_dim != wsi_projection_dim:
            raise ValueError(f"pathway token dim {token_dim} != wsi_projection_dim {wsi_projection_dim}")
        self.token_dim = token_dim

        # learned null tokens, initialised near zero like PathQ-Former's NullTokenModule
        self.null_wsi = nn.Parameter(torch.randn(1, self.num_null_patches, token_dim) * 0.01)
        self.null_pathways = nn.Parameter(torch.randn(1, self.num_pathways, token_dim) * 0.01)

        if self.aux_heads:
            # unimodal heads on the mean-pooled pre-fusion tokens, same shape as SurvPath's ``to_logits``
            self.aux_head_h = self._make_head(token_dim, num_bins)
            self.aux_head_g = self._make_head(token_dim, num_bins)

    @staticmethod
    def _make_head(dim: int, num_bins: int) -> nn.Module:
        return nn.Sequential(nn.Linear(dim, dim // 4), nn.ReLU(), nn.Linear(dim // 4, num_bins))

    def _index(self, i: int) -> torch.Tensor:
        return getattr(self, f"_idx{i}")

    # ------------------------------------------------------------------------------------
    def encode_wsi(self, wsi_features: torch.Tensor, wsi_mask: torch.Tensor | None) -> torch.Tensor:
        """(1, N, D_in) -> (1, N_valid, D) with the official projection; padding is removed as in the baseline."""
        x = wsi_features if wsi_mask is None else wsi_features[:, wsi_mask[0]]
        return self.net.wsi_projection_net(x)

    def encode_pathways(self, genomic_features: torch.Tensor) -> torch.Tensor:
        """(1, G) scaled gene vector -> (1, P, D) pathway tokens through the official per-pathway SNNs."""
        h = [self.net.sig_networks[i](genomic_features[0, self._index(i)].float()) for i in range(self.num_pathways)]
        return torch.stack(h).unsqueeze(0)

    def fuse(self, h_pathways: torch.Tensor, h_wsi: torch.Tensor) -> torch.Tensor:
        """The official SurvPath forward from the token level on (co-attention -> FFN -> LN -> pooling -> logits)."""
        net = self.net
        tokens = net.identity(torch.cat([h_pathways, h_wsi], dim=1))
        mm = net.cross_attender(x=tokens, mask=None, return_attention=False)
        mm = net.layer_norm(net.feed_forward(mm))
        paths = mm[:, : self.num_pathways, :].mean(dim=1)
        wsi = mm[:, self.num_pathways :, :].mean(dim=1)
        return net.to_logits(torch.cat([paths, wsi], dim=1))

    def forward(
        self,
        wsi_features: torch.Tensor | None = None,
        genomic_features: torch.Tensor | None = None,
        wsi_mask: torch.Tensor | None = None,
        drop_wsi: torch.Tensor | None = None,
        drop_genomic: torch.Tensor | None = None,
        return_attention: bool = False,
        return_aux: bool = False,
    ):
        """
        Args:
            wsi_features: (1, N, wsi_input_dim) or None when histology is absent
            genomic_features: (1, G) scaled gene vector (``GenePassthrough`` tokenizer) or None when absent
            wsi_mask: (1, N) True for valid patches
            drop_wsi / drop_genomic: optional (1,) bool masks forcing absence (sampled in training when None)
        Returns:
            hazard_logits (1, num_bins) [, attention dict] [, aux dict with h / g logits and *_present flags]
        """
        ref = wsi_features if wsi_features is not None else genomic_features
        if ref is None:
            raise ValueError("At least one modality must be provided")
        if ref.shape[0] != 1:
            raise ValueError("SurvPathRecipe processes one patient at a time (batch_size=1), like the official SurvPath")
        B, device = 1, ref.device
        has_wsi, has_gen = wsi_features is not None, genomic_features is not None

        if self.training and self.modality_dropout > 0 and drop_wsi is None and drop_genomic is None:
            drop_wsi, drop_genomic = self._sample_modality_dropout(B, has_wsi, has_gen, device)

        aux: dict = {}
        present_h = torch.full((B,), has_wsi, dtype=torch.bool, device=device)
        present_g = torch.full((B,), has_gen, dtype=torch.bool, device=device)
        if has_wsi:
            h_wsi = self.encode_wsi(wsi_features, wsi_mask)
            if return_aux and self.aux_heads:
                aux["h"] = self.aux_head_h(h_wsi.mean(dim=1))
            if drop_wsi is not None and bool(drop_wsi.any()):
                present_h = present_h & ~drop_wsi
                h_wsi = self.null_wsi
        else:
            h_wsi = self.null_wsi

        if has_gen:
            h_pw = self.encode_pathways(genomic_features)
            if return_aux and self.aux_heads:
                aux["g"] = self.aux_head_g(h_pw.mean(dim=1))
            if drop_genomic is not None and bool(drop_genomic.any()):
                present_g = present_g & ~drop_genomic
                h_pw = self.null_pathways
        else:
            h_pw = self.null_pathways

        logits = self.fuse(h_pw, h_wsi)
        attn = {"histology": None, "genomic": None}
        if return_aux:
            aux.update(h_present=present_h, g_present=present_g)
            return (logits, attn, aux) if return_attention else (logits, aux)
        return (logits, attn) if return_attention else logits

    # same sampling rule and probability semantics as PathQ-Former (per-sample Bernoulli, never both)
    _sample_modality_dropout = PathQFormer._sample_modality_dropout

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
