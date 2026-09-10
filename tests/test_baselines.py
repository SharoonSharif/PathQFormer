"""Baselines expose PathQ-Former's forward contract and run on CPU."""

from pathlib import Path

import pytest
import torch

from src.models.baselines import ABMIL, MLPOmics, SNNOmics, SurvPathBaseline

ROOT = Path(__file__).resolve().parents[1]
SURVPATH = ROOT / "data" / "survpath_repo"


def _inputs(B=2, N=30, D=64, G=50):
    x = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 20:] = False
    g = torch.randn(B, G)
    return x, mask, g


def test_abmil_masks_padding_and_shapes():
    x, mask, g = _inputs()
    m = ABMIL(wsi_input_dim=64, num_bins=4).eval()
    out, attn = m(x, g, mask, return_attention=True)
    assert out.shape == (2, 4)
    assert float(attn["histology"][1, 0, 20:].max()) == 0.0
    # padded patches must not influence the output
    x2 = x.clone()
    x2[1, 20:] = 99.0
    assert torch.allclose(m(x2, g, mask), out, atol=1e-5)
    assert m.supports_missing is False and m.modality == "wsi"


def test_omics_baselines():
    _, _, g = _inputs()
    for cls in (SNNOmics, MLPOmics):
        m = cls(num_genes=50, num_bins=4).eval()
        assert m(None, g).shape == (2, 4)
        assert m.modality == "genomic"
        with pytest.raises(ValueError):
            m(torch.randn(2, 5, 64), None)


@pytest.mark.skipif(not SURVPATH.exists(), reason="SurvPath repository not cloned")
def test_official_survpath_wrapper():
    pytest.importorskip("einops")
    comp = {f"pw{i}": list(range(i, i + 6)) for i in range(0, 40, 4)}  # 10 pathways of 6 genes
    m = SurvPathBaseline(pathway_composition=comp, wsi_input_dim=64, num_bins=4, survpath_dir=SURVPATH).eval()
    x = torch.randn(1, 30, 64)
    mask = torch.ones(1, 30, dtype=torch.bool)
    mask[0, 25:] = False
    g = torch.randn(1, 50)
    out = m(x, g, mask)
    assert out.shape == (1, 4) and m.num_pathways == 10
    # padding is removed before the official forward, so padded rows cannot leak in
    x2 = x.clone()
    x2[0, 25:] = 99.0
    assert torch.allclose(m(x2, g, mask), out, atol=1e-5)
    with pytest.raises(ValueError):
        m(torch.randn(2, 30, 64), torch.randn(2, 50))
