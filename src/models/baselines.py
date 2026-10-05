"""Baselines trained under the identical v2 protocol (same patients, splits, bins, scaler, selection, metrics).

* ``ABMIL``      WSI-only gated attention MIL (Ilse et al., 2018) with the layer sizes used in the SurvPath
                 repository ([D, 256, 256], dropout 0.25); re-implemented because the reference code
                 hard-codes a 1024-d input.
* ``SNNOmics``   genomics-only self-normalising network (Klambauer et al., 2017) as in MCAT / SurvPath:
                 Linear -> ELU -> AlphaDropout blocks [G, 256, 256].
* ``MLPOmics``   genomics-only MLP as in the SurvPath repository (projection 512 -> two 256 layers).
* ``SurvPathBaseline``  the **official** SurvPath model (Jaume et al., CVPR 2024) imported from the cloned
                 repository, fed the same 275 pathway groupings that PathQ-Former's tokenizer uses.

Every baseline exposes PathQ-Former's forward signature
``forward(wsi_features, genomic_features, wsi_mask, drop_wsi, drop_genomic, return_attention)`` and
receives the scaled (B, G) gene vector as ``genomic_features`` (the "tokenizer" is ``GenePassthrough``).
None of them can run with a missing modality, so ``supports_missing = False``. ``SurvPathRecipe``
(``src/models/survpath_recipe.py``, registered as ``"survpath_recipe"``) is the exception: the official
SurvPath submodules plus PathQ-Former's null tokens / modality dropout / auxiliary heads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


class GenePassthrough(nn.Module):
    """Stands in for PathwayTokenizer: returns the scaled gene vector unchanged."""

    num_pathways = 0

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        return gene_expression

    def get_pathway_names(self) -> list[str]:
        return []


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    return torch.softmax(scores, dim=1)


class ABMIL(nn.Module):
    """Gated attention-based MIL over patch embeddings (WSI only)."""

    supports_missing = False
    modality = "wsi"

    def __init__(self, wsi_input_dim: int = 1536, hidden_dim: int = 256, attn_dim: int = 256, dropout: float = 0.25, num_bins: int = 4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(wsi_input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attn_v = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attn_u = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attn_w = nn.Linear(attn_dim, 1)
        self.rho = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, num_bins)

    def forward(self, wsi_features=None, genomic_features=None, wsi_mask=None, drop_wsi=None, drop_genomic=None, return_attention=False):
        if wsi_features is None:
            raise ValueError("ABMIL needs WSI features")
        h = self.fc(wsi_features)                                   # (B, N, d)
        a = _masked_softmax(self.attn_w(self.attn_v(h) * self.attn_u(h)), wsi_mask)  # (B, N, 1)
        z = self.rho((a * h).sum(dim=1))                            # (B, d)
        logits = self.classifier(z)
        if return_attention:
            return logits, {"histology": a.squeeze(-1).unsqueeze(1), "genomic": None}
        return logits


class SNNOmics(nn.Module):
    """Self-normalising network on the full gene vector (genomics only)."""

    supports_missing = False
    modality = "genomic"

    def __init__(self, num_genes: int, hidden: tuple[int, ...] = (256, 256), dropout: float = 0.25, num_bins: int = 4):
        super().__init__()
        layers, d = [], num_genes
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU(), nn.AlphaDropout(p=dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.classifier = nn.Linear(d, num_bins)

    def forward(self, wsi_features=None, genomic_features=None, wsi_mask=None, drop_wsi=None, drop_genomic=None, return_attention=False):
        if genomic_features is None:
            raise ValueError("SNNOmics needs the gene vector")
        logits = self.classifier(self.net(genomic_features))
        return (logits, {"histology": None, "genomic": None}) if return_attention else logits


class MLPOmics(nn.Module):
    """Two-layer MLP on the full gene vector (genomics only), as in the SurvPath repository."""

    supports_missing = False
    modality = "genomic"

    def __init__(self, num_genes: int, projection_dim: int = 512, dropout: float = 0.1, num_bins: int = 4):
        super().__init__()
        h = projection_dim // 2
        self.net = nn.Sequential(
            nn.Linear(num_genes, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(h, num_bins)

    def forward(self, wsi_features=None, genomic_features=None, wsi_mask=None, drop_wsi=None, drop_genomic=None, return_attention=False):
        if genomic_features is None:
            raise ValueError("MLPOmics needs the gene vector")
        logits = self.classifier(self.net(genomic_features))
        return (logits, {"histology": None, "genomic": None}) if return_attention else logits


class SurvPathBaseline(nn.Module):
    """Official SurvPath (Jaume et al., CVPR 2024) from the cloned repository, batch size 1.

    The pathway groups are the same filtered composition used by PathQ-Former's tokenizer
    (``min_genes`` / ``max_genes``), so both models see identical genomic inputs.
    """

    supports_missing = False
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
    ):
        super().__init__()
        repo = str(Path(survpath_dir).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from models.model_SurvPath import SurvPath  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(f"Could not import the official SurvPath model from {repo} (pip install einops?)") from exc

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

    def _index(self, i: int) -> torch.Tensor:
        return getattr(self, f"_idx{i}")

    def forward(self, wsi_features=None, genomic_features=None, wsi_mask=None, drop_wsi=None, drop_genomic=None, return_attention=False):
        if wsi_features is None or genomic_features is None:
            raise ValueError("SurvPath needs both modalities")
        if wsi_features.shape[0] != 1:
            raise ValueError("The official SurvPath implementation processes one patient at a time (batch_size=1)")
        x = wsi_features if wsi_mask is None else wsi_features[:, wsi_mask[0]]
        kwargs = {"x_path": x, "return_attn": False}
        for i in range(self.num_pathways):
            kwargs[f"x_omic{i + 1}"] = genomic_features[0, self._index(i)]
        logits = self.net(**kwargs)
        return (logits, {"histology": None, "genomic": None}) if return_attention else logits


from .survpath_recipe import SurvPathRecipe  # noqa: E402  (imports nothing from this module)

BASELINES = {
    "abmil": ABMIL,
    "snn": SNNOmics,
    "mlp_omics": MLPOmics,
    "survpath": SurvPathBaseline,
    "survpath_recipe": SurvPathRecipe,
}
