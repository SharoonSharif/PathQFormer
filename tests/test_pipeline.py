"""Fast CPU tests for the pieces whose silent failure would corrupt a result table."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import OmicsScaler, SurvivalBins, TCGAMultimodalDataset, collate_multimodal, slide_stem
from src.models import NLLSurvivalLoss, PathQFormer, hazards_to_survival, risk_from_logits
from src.training.evaluate import bootstrap_cindex_ci, concordance, mean_ci95, survival_metrics

ROOT = Path(__file__).resolve().parents[1]
SURVPATH = ROOT / "data" / "survpath_repo"
DUMMY = ROOT / "data" / "embeddings" / "uni2h_dummy"


# ------------------------------------------------------------------------------ bins / scaler
def test_bins_from_uncensored_training_patients_are_balanced_and_reused():
    rng = np.random.default_rng(0)
    times = rng.exponential(30, 400)
    cens = (rng.random(400) < 0.6).astype(float)
    bins = SurvivalBins.from_training(times, cens, num_bins=4)
    assert bins.num_bins == 4
    assert bins.edges[0] == -np.inf and bins.edges[-1] == np.inf
    unc = bins.assign(times[cens < 1])
    counts = np.bincount(unc, minlength=4)
    assert counts.max() - counts.min() <= 1  # quantiles of the uncensored subset
    # left-closed: a time exactly on an interior edge goes to the upper bin
    assert bins.assign([bins.edges[1]])[0] == 1
    # unseen times far outside the training range land in the outer bins, never crash
    assert bins.assign([1e9])[0] == 3 and bins.assign([-5.0])[0] == 0
    assert SurvivalBins.from_dict(bins.to_dict()).assign(times).tolist() == bins.assign(times).tolist()


def test_scaler_fits_on_train_only():
    tr = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]], dtype=np.float32)
    va = np.array([[6.0, 11.0]], dtype=np.float32)
    s = OmicsScaler("minmax").fit(tr)
    assert np.allclose(s.transform(tr)[:, 0], [-1, 0, 1])
    assert np.allclose(s.transform(tr)[:, 1], -1)  # constant gene: no division by zero
    assert s.transform(va)[0, 0] == pytest.approx(2.0)  # val may exceed the train range
    s2 = OmicsScaler.from_dict(s.to_dict())
    assert np.allclose(s2.transform(va), s.transform(va))


def test_slide_stem():
    assert slide_stem("TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739.svs") == "TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739"
    assert slide_stem("TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739.h5") == "TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739"


# ------------------------------------------------------------------------------ loss / risk
def test_nll_matches_hand_computation():
    logits = torch.tensor([[0.2, -0.5, 1.0, 0.3]])
    h, S = hazards_to_survival(logits)
    loss = NLLSurvivalLoss()
    # event in bin 2:  -log S_1 - log h_2
    expected_unc = -(torch.log(S[0, 1]) + torch.log(h[0, 2]))
    assert torch.isclose(loss(logits, torch.tensor([2]), torch.tensor([0.0])), expected_unc)
    # censored in bin 1: -log S_1
    assert torch.isclose(loss(logits, torch.tensor([1]), torch.tensor([1.0])), -torch.log(S[0, 1]))
    # event in bin 0: S_{-1} = 1
    assert torch.isclose(loss(logits, torch.tensor([0]), torch.tensor([0.0])), -torch.log(h[0, 0]))


def test_risk_is_monotone_in_hazard():
    low = torch.tensor([[-3.0, -3.0, -3.0, -3.0]])
    high = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    assert risk_from_logits(high) > risk_from_logits(low)


# ------------------------------------------------------------------------------ model
@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return PathQFormer(wsi_input_dim=64, genomic_input_dim=32, hidden_dim=32, num_queries=4, num_heads=4,
                       query_layers=1, fusion_layers=1, num_bins=4, dropout=0.0, modality_dropout=0.5).eval()


def test_forward_shapes_and_missing_modalities(model):
    x, g = torch.randn(3, 20, 64), torch.randn(3, 7, 32)
    mask = torch.ones(3, 20, dtype=torch.bool)
    mask[1, 10:] = False
    assert model(x, g, mask).shape == (3, 4)
    assert model(x, None, mask).shape == (3, 4)
    assert model(None, g).shape == (3, 4)
    _, attn = model(x, g, mask, return_attention=True)
    assert attn["histology"].shape == (3, 4, 20) and attn["genomic"].shape == (3, 4, 7)
    assert float(attn["histology"][1, :, 10:].abs().max()) == 0.0  # padded keys get no attention


def test_per_sample_drop_equals_batch_missing(model):
    x, g = torch.randn(3, 20, 64), torch.randn(3, 7, 32)
    drop = torch.tensor([True, False, True])
    out = model(x, g, drop_wsi=drop)
    ref_missing = model(None, g)
    ref_full = model(x, g)
    assert torch.allclose(out[0], ref_missing[0], atol=1e-5)
    assert torch.allclose(out[2], ref_missing[2], atol=1e-5)
    assert torch.allclose(out[1], ref_full[1], atol=1e-5)


def test_aux_unimodal_heads_and_checkpoint_compatibility():
    torch.manual_seed(0)
    m = PathQFormer(wsi_input_dim=64, genomic_input_dim=32, hidden_dim=32, num_queries=4, num_heads=4,
                    query_layers=1, fusion_layers=1, dropout=0.0, aux_heads=True).eval()
    x, g = torch.randn(3, 20, 64), torch.randn(3, 7, 32)
    logits, aux = m(x, g, drop_wsi=torch.tensor([True, False, False]), return_aux=True)
    assert logits.shape == (3, 4) and aux["h"].shape == (3, 4) and aux["g"].shape == (3, 4)
    assert aux["h_present"].tolist() == [False, True, True] and aux["g_present"].tolist() == [True, True, True]
    assert m(x, g).shape == (3, 4)  # plain forward unchanged
    plain = PathQFormer(wsi_input_dim=64, genomic_input_dim=32, hidden_dim=32, num_queries=4, num_heads=4,
                        query_layers=1, fusion_layers=1)
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("aux_head")})


def test_attention_maps_are_distributions_over_valid_keys(model):
    x, g = torch.randn(2, 20, 64), torch.randn(2, 7, 32)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 12:] = False
    _, attn = model(x, g, mask, return_attention=True)
    assert torch.allclose(attn["histology"].sum(-1), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(attn["genomic"].sum(-1), torch.ones(2, 4), atol=1e-5)
    assert float(attn["histology"][1, :, 12:].abs().max()) == 0.0


def test_post_norm_variant_and_null_tokens_are_learnable():
    m = PathQFormer(wsi_input_dim=16, genomic_input_dim=8, hidden_dim=16, num_queries=2, num_heads=2,
                    query_layers=1, fusion_layers=1, norm_first=False).eval()
    out = m(torch.randn(2, 5, 16), torch.randn(2, 3, 8))
    assert out.shape == (2, 4)
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert "null_tokens.null_histology" in names and "null_tokens.null_genomic" in names
    # gradient reaches the null code when a modality is absent
    m.train()
    loss = m(None, torch.randn(2, 3, 8)).sum()
    loss.backward()
    assert m.null_tokens.null_histology.grad is not None and float(m.null_tokens.null_histology.grad.abs().sum()) > 0


def test_batched_modality_dropout_is_per_sample(model):
    model.train()
    torch.manual_seed(3)
    drops = torch.stack([torch.stack(model._sample_modality_dropout(256, True, True, torch.device("cpu"))) for _ in range(4)])
    d_w, d_g = drops[:, 0].float(), drops[:, 1].float()
    assert 0.3 < d_w.mean() < 0.6 and 0.3 < d_g.mean() < 0.6  # p = 0.5 per modality, minus the never-both rule
    assert not bool((drops[:, 0] & drops[:, 1]).any())
    model.eval()


def test_training_dropout_never_removes_both_modalities(model):
    model.train()
    torch.manual_seed(1)
    for _ in range(50):
        d_w, d_g = model._sample_modality_dropout(64, True, True, torch.device("cpu"))
        assert not bool((d_w & d_g).any())
    d_w, d_g = model._sample_modality_dropout(64, True, False, torch.device("cpu"))
    assert not bool(d_w.any())  # the only present modality is never dropped
    model.eval()


# ------------------------------------------------------------------------------ metrics
def test_concordance_extremes_and_bootstrap():
    time = np.array([1, 2, 3, 4, 5, 6], float)
    event = np.array([1, 1, 1, 1, 1, 1], bool)
    perfect = -time  # higher risk = shorter time
    assert concordance(event, time, perfect) == pytest.approx(1.0)
    assert concordance(event, time, time) == pytest.approx(0.0)
    lo, hi, _ = bootstrap_cindex_ci(event, time, perfect, n_boot=50, seed=0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_concordance_returns_nan_without_comparable_pairs():
    # a single event at the latest time: no pair is comparable (COADREAD bootstrap crash, 2026-09-11)
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([False, False, False, True])
    assert np.isnan(concordance(event, time, np.array([0.1, 0.2, 0.3, 0.4])))
    lo, hi, _ = bootstrap_cindex_ci(np.array([True, False, True]), np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), n_boot=20, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_survival_metrics_run_end_to_end():
    rng = np.random.default_rng(0)
    n = 120
    t_train = rng.exponential(30, 300)
    e_train = rng.random(300) < 0.5
    time = rng.exponential(30, n)
    event = rng.random(n) < 0.5
    bins = SurvivalBins.from_training(t_train, ~e_train, 4)
    logits = torch.randn(n, 4) - torch.tensor(time / 30.0).float().unsqueeze(1)  # longer time -> lower hazard
    _, S = hazards_to_survival(logits)
    m = survival_metrics(e_train, t_train, event, time, risk_from_logits(logits).numpy(), S.numpy(), bins.edges)
    assert 0.5 < m["c_index"] <= 1.0
    for key in ("c_index_ipcw", "ibs", "iauc"):
        assert key in m


def test_eval_grid_and_ipcw_horizon_follow_the_training_quartiles():
    from src.training.evaluate import eval_times, ipcw_horizon

    edges = np.array([-np.inf, 8.0, 14.0, 21.0, np.inf])
    train_time = np.linspace(1, 100, 200)
    val_time = np.linspace(2, 60, 50)
    assert eval_times(train_time, val_time, edges).tolist() == [8.0, 14.0, 21.0]
    assert ipcw_horizon(train_time, edges) == 21.0
    # an interior edge beyond the validation follow-up is dropped; the horizon never exceeds the training follow-up
    assert eval_times(train_time, np.linspace(2, 15, 50), edges).tolist() == [8.0, 14.0]
    assert ipcw_horizon(np.linspace(1, 10, 20), edges) == 10.0


def test_survival_metrics_degrade_to_nan_not_crash():
    rng = np.random.default_rng(1)
    t_train, e_train = rng.exponential(30, 100), rng.random(100) < 0.5
    time, event = rng.exponential(30, 30), rng.random(30) < 0.5
    S = np.clip(rng.random((30, 4)), 0.01, 0.99)
    S.sort(axis=1)
    S = S[:, ::-1].copy()
    # a grid with no valid interior edges: Brier/IBS/AUC are NaN or empty, C-index still computed
    m = survival_metrics(e_train, t_train, event, time, rng.random(30), S, np.array([-np.inf, np.inf]))
    assert np.isfinite(m["c_index"]) and m["eval_times"] == [] and np.isnan(m["ibs"]) and np.isnan(m["iauc"])


def test_collate_pads_and_masks_multiple_patients():
    a = {"case_id": "a", "slide_ids": ["a"], "wsi_features": torch.ones(3, 4), "gene_expression": torch.zeros(5),
         "survival_time_bin": torch.tensor(1), "censorship": torch.tensor(0.0), "survival_time": torch.tensor(2.0)}
    b = {**a, "case_id": "b", "wsi_features": torch.ones(7, 4) * 2}
    batch = collate_multimodal([a, b])
    assert batch["wsi_features"].shape == (2, 7, 4) and batch["wsi_mask"].sum(1).tolist() == [3, 7]
    assert float(batch["wsi_features"][0, 3:].abs().sum()) == 0.0 and batch["case_ids"] == ["a", "b"]


def test_mean_ci95():
    s = mean_ci95([0.6, 0.62, 0.64, 0.58, 0.61])
    assert s["n"] == 5 and s["ci95"][0] < s["mean"] < s["ci95"][1]


# ------------------------------------------------------------------------------ dataset (needs repo data)
needs_data = pytest.mark.skipif(not (SURVPATH.exists() and DUMMY.exists()), reason="SurvPath CSVs / dummy embeddings not present")


@needs_data
def test_dataset_is_patient_level_and_matches_survpath_split():
    kw = dict(
        metadata_csv=SURVPATH / "datasets_csv/metadata/tcga_blca.csv",
        rna_csv=SURVPATH / "datasets_csv/raw_rna_data/combine/blca/rna_clean.csv",
        split_csv=SURVPATH / "splits/5foldcv/tcga_blca/splits_0.csv",
        embeddings_dir=DUMMY,
        num_bins=4,
    )
    tr = TCGAMultimodalDataset(split="train", **kw)
    va = TCGAMultimodalDataset(split="val", bins=tr.bins, scaler=tr.scaler, **kw)
    assert len(tr) == 289 and len(va) == 70  # SurvPath fold-0 patient counts
    assert tr.patients["case_id"].is_unique and va.patients["case_id"].is_unique
    unc = tr.patients[tr.patients["censorship"] < 1]["survival_time_bin"].value_counts()
    assert unc.max() - unc.min() <= 1
    multi = tr.patients.index[tr.patients["n_slides"] > 1][0]
    item = tr[multi]
    assert len(item["slide_ids"]) > 1 and item["wsi_features"].shape[1] == 1536
    batch = collate_multimodal([tr[0], item])
    assert batch["wsi_mask"].sum(1).tolist() == [tr[0]["wsi_features"].shape[0], item["wsi_features"].shape[0]]
    assert va.gene_matrix.dtype == np.float32 and tr.gene_matrix.min() >= -1.0 and tr.gene_matrix.max() <= 1.0
    with pytest.raises(ValueError):
        TCGAMultimodalDataset(split="val", **kw)  # val must receive train bins/scaler
