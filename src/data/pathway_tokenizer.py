"""Genomic pathway tokenizer: groups genes into biological pathways and embeds them.

Supports two input formats:
  1. SurvPath composition CSV (binary gene × pathway matrix) — preferred
  2. JSON mapping pathway_name -> [gene_symbols]
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class PathwayTokenizer(nn.Module):
    """
    Converts raw gene expression vectors into pathway-level token embeddings.
    Each pathway gets a lightweight MLP that maps its member gene expression values
    to a fixed-dim embedding.
    """

    def __init__(
        self,
        pathway_composition: dict[str, list[int]] | None = None,
        pathway_file: str | Path | None = None,
        gene_list: list[str] | None = None,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        min_genes: int = 3,
        max_genes: int = 300,
    ):
        """
        Provide either:
          (a) pathway_composition: pre-built dict of pathway_name -> list of gene indices
          (b) pathway_file + gene_list: JSON file + ordered gene list for index lookup
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        if pathway_composition is not None:
            indexed_pathways = pathway_composition
        elif pathway_file is not None and gene_list is not None:
            indexed_pathways = self._load_from_json(pathway_file, gene_list)
        else:
            raise ValueError("Provide either pathway_composition or (pathway_file, gene_list)")

        self.pathway_names = []
        self.pathway_gene_indices = []
        pathway_mlps = []

        for name, indices in sorted(indexed_pathways.items()):
            if len(indices) < min_genes:
                continue
            indices = indices[:max_genes]

            self.pathway_names.append(name)
            self.pathway_gene_indices.append(torch.tensor(indices, dtype=torch.long))

            n_genes = len(indices)
            pathway_mlps.append(nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            ))

        self.pathway_mlps = nn.ModuleList(pathway_mlps)
        self.num_pathways = len(self.pathway_names)

    @staticmethod
    def _load_from_json(pathway_file: str | Path, gene_list: list[str]) -> dict[str, list[int]]:
        gene_to_idx = {g: i for i, g in enumerate(gene_list)}
        with open(pathway_file) as f:
            raw = json.load(f)
        return {
            name: [gene_to_idx[g] for g in genes if g in gene_to_idx]
            for name, genes in raw.items()
        }

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_expression: (B, num_genes) — raw expression values
        Returns:
            (B, P, embedding_dim) — pathway token embeddings
        """
        device = gene_expression.device
        tokens = []

        for indices, mlp in zip(self.pathway_gene_indices, self.pathway_mlps):
            pathway_genes = gene_expression[:, indices.to(device)]
            tokens.append(mlp(pathway_genes))

        return torch.stack(tokens, dim=1)

    def get_pathway_names(self) -> list[str]:
        return list(self.pathway_names)


def load_survpath_compositions(
    composition_csv: str | Path,
    gene_list: list[str],
) -> dict[str, list[int]]:
    """
    Load SurvPath's binary pathway composition matrix and convert to
    pathway_name -> list of gene column indices (into gene_list).

    Args:
        composition_csv: path to combine_comps.csv / hallmarks_comps.csv
        gene_list: ordered gene symbols matching the RNA expression columns
    Returns:
        dict mapping pathway name to list of integer indices into gene_list
    """
    comp_df = pd.read_csv(composition_csv)
    comp_df = comp_df.rename(columns={comp_df.columns[0]: "gene"})

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    pathway_names = [c for c in comp_df.columns if c != "gene"]

    result = {}
    for pw in pathway_names:
        member_genes = comp_df.loc[comp_df[pw] == 1, "gene"].tolist()
        indices = [gene_to_idx[g] for g in member_genes if g in gene_to_idx]
        if indices:
            result[pw] = indices

    return result


def build_pathways_from_gmt(gmt_path: str | Path, output_json_path: str | Path) -> None:
    """Parse a GMT file (Reactome or MSigDB) into JSON: pathway_name -> [gene_symbols]."""
    pathways = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathways[parts[0]] = parts[2:]

    with open(output_json_path, "w") as f:
        json.dump(pathways, f, indent=2)
