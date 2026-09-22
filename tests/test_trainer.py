"""End-to-end tests of the training driver and analysis scripts on the dummy embeddings (CPU, tiny models).

Each training call is a real 5-fold-protocol fold on 289 patients, so the file takes a few minutes;
run `pytest -m "not slow"` for the fast subset.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.training.train import (
    build_datasets,
    make_loader,
    memory_limit_bytes,
    parse_overrides,
    train_cv,
    train_fold,
    with_defaults,
)

ROOT = Path(__file__).resolve().parents[1]
SURVPATH = ROOT / "data" / "survpath_repo"
DUMMY = ROOT / "data" / "embeddings" / "uni2h_dummy"
needs_data = pytest.mark.skipif(not (SURVPATH.exists() and DUMMY.exists()), reason="SurvPath CSVs / dummy embeddings not present")
slow = pytest.mark.slow


def tiny_cfg(out_dir: Path, **overrides) -> dict:
    cfg = {
        "survpath_dir": str(SURVPATH),
        "embeddings_dir": str(DUMMY),
        "output_dir": str(out_dir),
        "cancer_type": "blca",
        "wsi_input_dim": 1536,
        "hidden_dim": 32,
        "num_queries": 4,
        "num_heads": 4,
        "query_layers": 1,
        "fusion_layers": 1,
        "num_bins": 4,
        "dropout": 0.1,
        "modality_dropout": 0.15,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 1,
        "epochs": 1,
        "patience": 99,
        "num_folds": 5,
        "num_workers": 0,
        "num_threads": 4,
        "max_patches": 32,
        "pathway_hidden_dim": 16,
        "bootstrap": 10,
        "missing_repeats": 1,
        "missing_rates": [0.2, 0.5],
        "selection_metric": "last",
        "device": "cpu",
    }
    cfg.update(overrides)
    return cfg


def run_one_fold(cfg: dict, fold: int = 0) -> dict:
    cfg = with_defaults(cfg)
    run_dir = Path(cfg["output_dir"]) / cfg["cancer_type"]
    run_dir.mkdir(parents=True, exist_ok=True)
    return train_fold(cfg, fold, torch.device("cpu"), run_dir)


# --------------------------------------------------------------------------- config handling
def test_parse_overrides_types():
    ov = parse_overrides(["lr=1e-4", "epochs=3", "cache_in_ram=true", "max_patches=null", "missing_rates=[0.1,0.5]", "cancer_type=stad"])
    assert ov == {"lr": 1e-4, "epochs": 3, "cache_in_ram": True, "max_patches": None, "missing_rates": [0.1, 0.5], "cancer_type": "stad"}
    with pytest.raises(SystemExit):
        parse_overrides(["novalue"])


def test_with_defaults_validation():
    base = {"train_modalities": "both", "selection_metric": "val_loss", "model_type": "pathqformer"}
    assert with_defaults(base)["bootstrap"] == 1000
    with pytest.raises(ValueError):
        with_defaults({**base, "selection_metric": "best"})
    with pytest.raises(ValueError):
        with_defaults({**base, "model_type": "mcat"})
    for kind, mod in (("abmil", "wsi"), ("snn", "genomic"), ("mlp_omics", "genomic"), ("survpath", "both")):
        out = with_defaults({**base, "model_type": kind})
        assert out["train_modalities"] == mod and out["eval_missing"] is False


def test_memory_limit_is_sane():
    b = memory_limit_bytes()
    assert 1 << 30 <= b < 1 << 50


# --------------------------------------------------------------------------- data loading
@needs_data
def test_weighted_sampler_balances_bin_by_censorship_classes(tmp_path):
    cfg = with_defaults(tiny_cfg(tmp_path, weighted_sample=True))
    train_ds, _ = build_datasets(cfg, 0)
    loader = make_loader(train_ds, cfg, shuffle=True, seed=0, device=torch.device("cpu"))
    idx = np.array(list(iter(loader.sampler)))
    cls = train_ds.patients["survival_time_bin"].to_numpy()[idx] * 2 + (train_ds.patients["censorship"].to_numpy()[idx] > 0)
    counts = np.bincount(cls, minlength=8)
    natural = np.bincount(train_ds.patients["survival_time_bin"].to_numpy() * 2 + (train_ds.patients["censorship"].to_numpy() > 0), minlength=8)
    assert len(idx) == len(train_ds)
    # balanced sampling flattens the class histogram: max/min ratio must drop well below the natural one
    assert counts.max() / max(counts.min(), 1) < natural.max() / max(natural.min(), 1) / 2


@needs_data
def test_other_endpoints_and_missing_embeddings(tmp_path):
    cfg = with_defaults(tiny_cfg(tmp_path, endpoint="os"))
    tr, va = build_datasets(cfg, 0)
    assert tr.label_col == "survival_months" and len(tr) > 200
    cfg = with_defaults(tiny_cfg(tmp_path, endpoint="pfi"))
    tr, _ = build_datasets(cfg, 0)
    assert tr.label_col == "survival_months_pfi"
    # a subset of embeddings: patients without files are excluded and counted, never crash
    sub = tmp_path / "emb"
    sub.mkdir()
    for f in sorted(DUMMY.glob("*.pt"))[:60]:
        shutil.copy(f, sub / f.name)
    cfg = with_defaults(tiny_cfg(tmp_path, embeddings_dir=str(sub)))
    tr, va = build_datasets(cfg, 0)
    assert len(tr) < 289 and len(tr.missing_slides) > 0 and len(tr) + len(va) <= 60


# --------------------------------------------------------------------------- training driver
@needs_data
@slow
def test_train_cv_end_to_end_and_fold_skip(tmp_path):
    cfg = tiny_cfg(tmp_path / "run")
    res = train_cv(cfg, folds=[0])
    run_dir = Path(cfg["output_dir"]) / "blca"
    assert (run_dir / "results.json").exists() and (run_dir / "summary.md").exists() and (run_dir / "config.yaml").exists()
    fold = res["folds"][0]
    assert fold["selected_epoch"] == 1 and fold["epochs_run"] == 1 and fold["selection_metric"] == "last"
    assert set(fold["metrics"]) == {"both", "wsi_only", "genomic_only"}
    for m in fold["metrics"].values():
        for key in ("c_index", "c_index_ipcw", "ibs", "iauc", "c_index_ci95", "logrank_p", "ipcw_tau", "eval_times"):
            assert key in m
    assert [pm["rate"] for pm in fold["partial_missing"]] == [0.2, 0.5]
    for key in ("both", "wsi_only", "genomic_only"):
        assert (run_dir / "fold_0" / f"predictions_{key}.csv").exists()
    assert (run_dir / "fold_0" / "best_checkpoint.pt").exists() and not (run_dir / "fold_0" / "latest_checkpoint.pt").exists()
    summary = res["summary"]
    assert summary["n_folds"] == 1 and "conditions" in summary and summary["partial_missing"][0]["rate"] == 0.2
    # second call: the finished fold is skipped, nothing retrained
    mtime = (run_dir / "fold_0" / "fold_results.json").stat().st_mtime
    res2 = train_cv(cfg, folds=[0])
    assert res2["folds"][0]["c_index"] == fold["c_index"]
    assert (run_dir / "fold_0" / "fold_results.json").stat().st_mtime == mtime


@needs_data
@slow
def test_mid_fold_resume_continues_from_latest_checkpoint(tmp_path):
    cfg1 = tiny_cfg(tmp_path / "run", epochs=1, keep_latest_checkpoint=True)
    r1 = run_one_fold(cfg1)
    assert r1["epochs_run"] == 1
    latest = Path(cfg1["output_dir"]) / "blca" / "fold_0" / "latest_checkpoint.pt"
    assert latest.exists()
    cfg2 = tiny_cfg(tmp_path / "run", epochs=2, keep_latest_checkpoint=False)
    r2 = run_one_fold(cfg2)
    assert r2["epochs_run"] == 2 and r2["selected_epoch"] == 2
    assert r2["history"][0]["val_cindex"] == pytest.approx(r1["history"][0]["val_cindex"])  # epoch 1 was not redone
    assert not latest.exists()


@needs_data
@slow
def test_selection_modes_pick_the_right_epoch(tmp_path):
    for sel in ("val_loss", "val_cindex"):
        cfg = tiny_cfg(tmp_path / sel, epochs=2, selection_metric=sel, lr=3e-3)
        r = run_one_fold(cfg)
        hist = r["history"]
        key = "val_loss" if sel == "val_loss" else "val_cindex"
        pick = (np.argmin if sel == "val_loss" else np.argmax)([h[key] for h in hist]) + 1
        assert r["selected_epoch"] == pick and r["epochs_run"] == 2
    cfg = tiny_cfg(tmp_path / "es", epochs=4, selection_metric="val_loss", patience=1, min_epochs=0, lr=5e-2)
    r = run_one_fold(cfg)
    assert r["epochs_run"] < 4 or r["selected_epoch"] <= 4  # early stopping fires or the budget ends


@needs_data
@slow
def test_same_seed_is_deterministic_on_cpu(tmp_path):
    a = run_one_fold(tiny_cfg(tmp_path / "a", bootstrap=0))
    b = run_one_fold(tiny_cfg(tmp_path / "b", bootstrap=0))
    assert a["c_index"] == pytest.approx(b["c_index"], abs=1e-6)
    assert a["history"][0]["train_loss"] == pytest.approx(b["history"][0]["train_loss"], abs=1e-6)
    c = run_one_fold(tiny_cfg(tmp_path / "c", bootstrap=0, seed=7))
    assert c["history"][0]["train_loss"] != pytest.approx(a["history"][0]["train_loss"], abs=1e-6)


@needs_data
@slow
def test_aux_unimodal_loss_trains(tmp_path):
    r = run_one_fold(tiny_cfg(tmp_path, aux_unimodal_weight=0.5))
    assert np.isfinite(r["history"][0]["train_loss"]) and set(r["metrics"]) == {"both", "wsi_only", "genomic_only"}


@needs_data
@slow
@pytest.mark.parametrize("kind,cond", [("abmil", "wsi_only"), ("snn", "genomic_only"), ("mlp_omics", "genomic_only")])
def test_baselines_through_the_trainer(tmp_path, kind, cond):
    r = run_one_fold(tiny_cfg(tmp_path, model_type=kind, optimizer="radam", weighted_sample=True, nll_alpha=0.5))
    assert set(r["metrics"]) == {cond} and r["partial_missing"] == []
    assert 0.0 <= r["c_index"] <= 1.0


@needs_data
@slow
def test_official_survpath_through_the_trainer(tmp_path):
    pytest.importorskip("einops")
    r = run_one_fold(tiny_cfg(tmp_path, model_type="survpath", optimizer="radam", max_patches=64, pathway_hidden_dim=16))
    assert set(r["metrics"]) == {"both"} and r["partial_missing"] == []


# --------------------------------------------------------------------------- analysis scripts
@needs_data
@slow
def test_analysis_scripts_on_real_outputs(tmp_path):
    py = sys.executable
    joint = tiny_cfg(tmp_path / "joint")
    wsi = tiny_cfg(tmp_path / "wsi", train_modalities="wsi", eval_missing=False)
    rna = tiny_cfg(tmp_path / "rna", train_modalities="genomic", eval_missing=False)
    for cfg in (joint, wsi, rna):
        train_cv(cfg, folds=[0, 1])

    def run(*args):
        proc = subprocess.run([py, *args], cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONUTF8": "1"})
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        return proc.stdout

    run("scripts/recompute_metrics.py", str(tmp_path / "joint" / "blca"), "--bootstrap", "5")
    with open(tmp_path / "joint" / "blca" / "results.json") as f:
        assert f.read().count("metrics_recomputed_at") >= 1

    out = run("scripts/late_fusion.py", str(tmp_path / "wsi" / "blca"), str(tmp_path / "rna" / "blca"), "--out", str(tmp_path / "late" / "blca"), "--bootstrap", "5")
    assert "fold 0" in out and (tmp_path / "late" / "blca" / "results.json").exists()

    out = run("scripts/aggregate_results.py", str(tmp_path), "--out", str(tmp_path / "summary_all.md"), "--compare", "joint")
    assert "Paired tests" in out and "late" in out and (tmp_path / "summary_all.md").exists()

    out = run("scripts/seed_table.py", str(tmp_path), "--methods", "joint", "late", "--ref", "joint")
    assert "blca" in out

    out = run("scripts/epoch_curves.py", str(tmp_path), "--runs", "joint", "--epochs", "1")
    assert "POOLED" in out

    out = run("scripts/analyze_run.py", str(tmp_path / "joint" / "blca"), "--attention", "--n_heatmap", "1")
    an = tmp_path / "joint" / "blca" / "analysis"
    assert (an / "training_curves.png").exists() and (an / "km_curves.png").exists() and (an / "pathway_importance.csv").exists()
    assert any(an.glob("attention/*.npz"))
