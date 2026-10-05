"""SurvPathRecipe: official SurvPath submodules + null tokens / modality dropout / aux heads, on CPU."""

from pathlib import Path

import pytest
import torch

from src.models.baselines import BASELINES, SurvPathBaseline
from src.models.survpath_recipe import SurvPathRecipe
from src.training.train import with_defaults

ROOT = Path(__file__).resolve().parents[1]
SURVPATH = ROOT / "data" / "survpath_repo"
pytestmark = pytest.mark.skipif(not SURVPATH.exists(), reason="SurvPath repository not cloned")

COMP = {f"pw{i}": list(range(i, i + 6)) for i in range(0, 40, 4)}  # 10 pathways of 6 genes over 50 genes


def _model(**kw) -> SurvPathRecipe:
    pytest.importorskip("einops")
    torch.manual_seed(0)
    return SurvPathRecipe(pathway_composition=COMP, wsi_input_dim=64, num_bins=4, survpath_dir=SURVPATH, **kw).eval()


def _inputs(N: int = 30):
    torch.manual_seed(1)
    x = torch.randn(1, N, 64)
    mask = torch.ones(1, N, dtype=torch.bool)
    mask[0, 25:] = False
    g = torch.randn(1, 50)
    return x, mask, g


def test_registry_and_contract():
    assert BASELINES["survpath_recipe"] is SurvPathRecipe
    m = _model()
    assert m.supports_missing is True and m.modality == "both" and m.num_pathways == 10
    assert m.null_wsi.shape == (1, 32, 256) and m.null_pathways.shape == (1, 10, 256)
    cfg = with_defaults({"model_type": "survpath_recipe"})
    assert cfg["train_modalities"] == "both" and cfg["eval_missing"] is True  # unlike the other baselines
    assert with_defaults({"model_type": "survpath"})["eval_missing"] is False


def test_shapes_and_padding_removed():
    m = _model()
    x, mask, g = _inputs()
    out = m(x, g, mask)
    assert out.shape == (1, 4) and torch.isfinite(out).all()
    x2 = x.clone()
    x2[0, 25:] = 99.0
    assert torch.allclose(m(x2, g, mask), out, atol=1e-5)
    logits, attn = m(x, g, mask, return_attention=True)
    assert torch.allclose(logits, out) and set(attn) == {"histology", "genomic"}
    with pytest.raises(ValueError):
        m(torch.randn(2, 30, 64), torch.randn(2, 50))
    with pytest.raises(ValueError):
        m(None, None)


def test_drop_flags_change_output_and_equal_absent_modality():
    m = _model()
    x, mask, g = _inputs()
    t = torch.tensor([True])
    full = m(x, g, mask)
    no_wsi = m(x, g, mask, drop_wsi=t)
    no_gen = m(x, g, mask, drop_genomic=t)
    assert not torch.allclose(full, no_wsi, atol=1e-4)
    assert not torch.allclose(full, no_gen, atol=1e-4)
    assert not torch.allclose(no_wsi, no_gen, atol=1e-4)
    # a dropped modality and an absent modality both route through the same null tokens
    assert torch.allclose(m(None, g), no_wsi, atol=1e-6)
    assert torch.allclose(m(x, None, mask), no_gen, atol=1e-6)
    # the drop flag False is a no-op
    assert torch.allclose(m(x, g, mask, drop_wsi=torch.tensor([False]), drop_genomic=torch.tensor([False])), full)


def test_absent_modality_is_finite_and_trainable():
    m = _model()
    x, mask, g = _inputs()
    for out in (m(None, g), m(x, None, mask)):
        assert out.shape == (1, 4) and torch.isfinite(out).all()
    # gradients reach the null tokens of the absent modality
    m.train()
    m.modality_dropout = 0.0
    m(None, g).sum().backward()
    assert m.null_wsi.grad is not None and float(m.null_wsi.grad.abs().sum()) > 0
    assert m.null_pathways.grad is None


def test_aux_outputs_in_train_py_format():
    m = _model(aux_heads=True)
    x, mask, g = _inputs()
    logits, aux = m(x, g, mask, return_aux=True)
    assert logits.shape == (1, 4)
    assert set(aux) == {"h", "g", "h_present", "g_present"}
    assert aux["h"].shape == (1, 4) and aux["g"].shape == (1, 4)
    assert bool(aux["h_present"].all()) and bool(aux["g_present"].all())
    assert aux["h_present"].dtype == torch.bool and aux["h_present"].shape == (1,)
    # absent genomics: no "g" head output, flag False; dropped WSI: head computed on the real tokens, flag False
    _, aux = m(x, None, mask, return_aux=True)
    assert "g" not in aux and "h" in aux and not bool(aux["g_present"].any())
    _, aux = m(x, g, mask, drop_wsi=torch.tensor([True]), return_aux=True)
    assert "h" in aux and not bool(aux["h_present"].any()) and bool(aux["g_present"].all())
    logits2, attn, aux2 = m(x, g, mask, return_aux=True, return_attention=True)
    assert torch.allclose(logits2, logits) and set(attn) == {"histology", "genomic"} and "h" in aux2
    # no aux heads: return_aux still yields the dict train.py reads (only the presence flags)
    m0 = _model()
    _, aux0 = m0(x, g, mask, return_aux=True)
    assert set(aux0) == {"h_present", "g_present"} and not hasattr(m0, "aux_head_h")


def test_modality_dropout_sampler_matches_pathqformer_rule():
    m = _model(modality_dropout=1.0)
    d_w, d_g = m._sample_modality_dropout(64, True, True, torch.device("cpu"))
    assert d_w.shape == (64,) and d_g.shape == (64,)
    assert not bool((d_w & d_g).any())  # never both
    assert bool((d_w | d_g).all())  # p = 1: exactly one modality dropped per sample
    assert 0 < int(d_w.sum()) < 64  # the coin flip keeps each modality sometimes
    z_w, z_g = m._sample_modality_dropout(8, True, False, torch.device("cpu"))
    assert not bool(z_w.any()) and not bool(z_g.any())  # a lone modality is never dropped
    # training mode samples drops only when no flags are given; eval never samples
    x, mask, g = _inputs()
    m.train()
    torch.manual_seed(0)
    _, aux = m(x, g, mask, return_aux=True)
    assert bool(aux["h_present"].any()) != bool(aux["g_present"].any())
    m.eval()
    _, aux = m(x, g, mask, return_aux=True)
    assert bool(aux["h_present"].all()) and bool(aux["g_present"].all())
    m0 = _model(modality_dropout=0.0).train()
    _, aux = m0(x, g, mask, return_aux=True)
    assert bool(aux["h_present"].all()) and bool(aux["g_present"].all())


def test_parameter_count_close_to_official_baseline():
    pytest.importorskip("einops")
    base = SurvPathBaseline(pathway_composition=COMP, wsi_input_dim=64, num_bins=4, survpath_dir=SURVPATH)
    n_base = sum(p.numel() for p in base.parameters())
    m = _model()
    n_recipe = m.num_parameters()
    assert n_recipe - n_base == 32 * 256 + 10 * 256  # only the null tokens are added
    assert n_recipe / n_base < 1.05
    m_aux = _model(aux_heads=True)
    assert m_aux.num_parameters() / n_base < 1.10
    # the official submodules are the same objects the baseline uses: identical state-dict keys under ``net.``
    assert {k for k in m.state_dict() if k.startswith("net.")} == {k for k in base.state_dict() if k.startswith("net.")}


@pytest.mark.skipif(not (ROOT / "data" / "embeddings" / "uni2h_dummy").exists(), reason="dummy embeddings not present")
@pytest.mark.slow
def test_one_fold_on_dummy_data_reports_missing_modality_conditions(tmp_path):
    pytest.importorskip("einops")
    from src.training.train import train_fold

    cfg = with_defaults({
        "survpath_dir": str(SURVPATH), "embeddings_dir": str(ROOT / "data" / "embeddings" / "uni2h_dummy"),
        "output_dir": str(tmp_path), "cancer_type": "blca", "model_type": "survpath_recipe",
        "wsi_input_dim": 1536, "num_bins": 4, "dropout": 0.1, "modality_dropout": 0.15, "aux_unimodal_weight": 0.5,
        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 1, "epochs": 1, "patience": 99, "num_folds": 5,
        "num_workers": 0, "num_threads": 4, "max_patches": 32, "bootstrap": 10, "missing_repeats": 1,
        "missing_rates": [0.5], "selection_metric": "last", "device": "cpu",
    })
    run_dir = tmp_path / "blca"
    run_dir.mkdir()
    fr = train_fold(cfg, 0, torch.device("cpu"), run_dir)
    assert set(fr["metrics"]) == {"both", "wsi_only", "genomic_only"}
    assert [pm["rate"] for pm in fr["partial_missing"]] == [0.5]
    assert all(torch.isfinite(torch.tensor(fr["metrics"][k]["c_index"])) for k in fr["metrics"])
