"""PathQ-Former training with 5-fold cross-validation on SurvPath's splits.

Rigor built in:
  * every fold is seeded (``seed + fold``); config, seed and git commit are stored with results
  * model selection on validation **loss** by default (selecting on validation C-index and
    reporting that same C-index is optimistically biased; both are still logged)
  * C-index on continuous survival time with the MCAT/SurvPath risk score (-sum_t S_t)
  * per-epoch history, per-patient predictions and full survival metrics (IPCW C-index,
    IBS, td-AUC, bootstrap CI, KM log-rank) for the selected checkpoint
  * missing-modality evaluation of the *same* checkpoint: WSI-only, genomics-only and
    randomly missing modalities for 10-50% of patients
  * two-level resume: finished folds (``cv_progress.json``) and mid-fold (``latest_checkpoint.pt``)

Usage:
  python -m src.training.train --config configs/blca_hybrid_v2.yaml
  python -m src.training.train --config configs/blca_hybrid_v2.yaml --set num_queries=64 output_dir=outputs_k64
  python -m src.training.train --config configs/blca_hybrid_v2.yaml --smoke     # dummy embeddings, 2 epochs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import PathwayTokenizer, TCGAMultimodalDataset, collate_multimodal, load_survpath_compositions
from src.models import NLLSurvivalLoss, PathQFormer, hazards_to_survival, risk_from_logits
from src.models.baselines import BASELINES, GenePassthrough
from src.training.evaluate import (
    HAS_SKSURV,
    bootstrap_cindex_ci,
    concordance,
    km_logrank_split,
    mean_ci95,
    survival_metrics,
)
from src.utils.repro import git_commit, physical_cores, set_seed

MODALITIES = ("both", "wsi", "genomic")

DEFAULTS: dict = {
    "model_type": "pathqformer",  # pathqformer | survpath | abmil | snn | mlp_omics
    "optimizer": "adamw",  # adamw | radam | adam
    "weighted_sample": False,  # SurvPath-style class-balanced sampling over (bin, censorship)
    "cache_dtype": "float32",
    "endpoint": "dss",
    "pathway_type": "combine",
    "scaler": "minmax",
    "seed": 0,
    "num_workers": 4,
    "cache_in_ram": False,
    "num_threads": None,
    "selection_metric": "val_loss",  # "val_cindex", or "last" = fixed epoch budget (MCAT/SurvPath style)
    "patience": 5,
    "min_epochs": 0,
    "warmup_epochs": 0,
    "grad_accum_steps": 1,
    "clip_grad_norm": 1.0,
    "nll_alpha": 0.0,
    "pathway_hidden_dim": 128,
    "min_genes": 3,
    "max_genes": 300,
    "max_patches": None,
    "norm_first": True,
    "aux_unimodal_weight": 0.0,  # >0 adds unimodal survival heads on each branch (loss += w * (L_wsi + L_rna))
    "train_modalities": "both",  # "both" | "wsi" | "genomic"  (component ablations)
    "eval_missing": True,
    "missing_rates": [0.1, 0.2, 0.3, 0.5],
    "missing_repeats": 5,
    "bootstrap": 1000,
    "save_predictions": True,
    "keep_latest_checkpoint": False,
}


def with_defaults(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update(cfg)
    if out["train_modalities"] not in MODALITIES:
        raise ValueError(f"train_modalities must be one of {MODALITIES}")
    if out["selection_metric"] not in ("val_loss", "val_cindex", "last"):
        raise ValueError("selection_metric must be 'val_loss', 'val_cindex' or 'last' (fixed epoch budget, no early stopping)")
    if out["model_type"] != "pathqformer":
        forced = {"survpath": "both", "abmil": "wsi", "snn": "genomic", "mlp_omics": "genomic"}
        if out["model_type"] not in forced:
            raise ValueError(f"unknown model_type {out['model_type']!r}; choose pathqformer or {sorted(forced)}")
        out["train_modalities"] = forced[out["model_type"]]
        out["eval_missing"] = False
    return out


# ---------------------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------------------
def get_device(cfg: dict) -> torch.device:
    if cfg.get("device"):
        return torch.device(cfg["device"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        return xm.xla_device()
    except ImportError:
        return torch.device("cpu")


def data_paths(cfg: dict, fold: int) -> dict:
    base = Path(cfg["survpath_dir"])
    cancer, pathway = cfg["cancer_type"], cfg["pathway_type"]
    return {
        "metadata": base / "datasets_csv" / "metadata" / f"tcga_{cancer}.csv",
        "rna": base / "datasets_csv" / "raw_rna_data" / pathway / cancer / "rna_clean.csv",
        "split": base / "splits" / "5foldcv" / f"tcga_{cancer}" / f"splits_{fold}.csv",
        "composition": base / "datasets_csv" / "pathway_compositions" / f"{pathway}_comps.csv",
    }


def build_datasets(cfg: dict, fold: int) -> tuple[TCGAMultimodalDataset, TCGAMultimodalDataset]:
    p = data_paths(cfg, fold)
    common = dict(
        metadata_csv=p["metadata"],
        rna_csv=p["rna"],
        embeddings_dir=cfg["embeddings_dir"],
        split_csv=p["split"],
        num_bins=cfg["num_bins"],
        endpoint=cfg["endpoint"],
        cache_in_ram=cfg["cache_in_ram"],
        cache_dtype=getattr(torch, str(cfg["cache_dtype"])),
    )
    train = TCGAMultimodalDataset(split="train", scaler_kind=cfg["scaler"], max_patches=cfg.get("max_patches"), **common)
    val = TCGAMultimodalDataset(split="val", bins=train.bins, scaler=train.scaler, **common)
    return train, val


def build_model(cfg: dict, gene_columns: list[str], composition_csv: Path, device):
    """Returns (model, tokenizer). Baselines get a pass-through tokenizer and see the scaled gene vector."""
    comp = load_survpath_compositions(composition_csv, gene_columns)
    kind = cfg["model_type"]
    if kind != "pathqformer":
        if kind == "survpath":
            model = BASELINES[kind](
                pathway_composition=comp, wsi_input_dim=cfg["wsi_input_dim"], num_bins=cfg["num_bins"],
                dropout=cfg["dropout"], min_genes=cfg["min_genes"], max_genes=cfg["max_genes"],
                survpath_dir=cfg["survpath_dir"],
            )
        elif kind == "abmil":
            model = BASELINES[kind](wsi_input_dim=cfg["wsi_input_dim"], dropout=cfg["dropout"], num_bins=cfg["num_bins"])
        else:  # snn / mlp_omics
            model = BASELINES[kind](num_genes=len(gene_columns), dropout=cfg["dropout"], num_bins=cfg["num_bins"])
        return model.to(device), GenePassthrough().to(device)

    tokenizer = PathwayTokenizer(
        pathway_composition=comp,
        embedding_dim=cfg["hidden_dim"],
        hidden_dim=cfg["pathway_hidden_dim"],
        min_genes=cfg["min_genes"],
        max_genes=cfg["max_genes"],
    ).to(device)
    model = PathQFormer(
        wsi_input_dim=cfg["wsi_input_dim"],
        genomic_input_dim=cfg["hidden_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_queries=cfg["num_queries"],
        num_heads=cfg["num_heads"],
        query_layers=cfg["query_layers"],
        fusion_layers=cfg["fusion_layers"],
        num_bins=cfg["num_bins"],
        dropout=cfg["dropout"],
        modality_dropout=cfg["modality_dropout"],
        norm_first=cfg["norm_first"],
        aux_heads=float(cfg["aux_unimodal_weight"]) > 0,
    ).to(device)
    return model, tokenizer


def make_loader(ds, cfg: dict, shuffle: bool, seed: int, device) -> DataLoader:
    workers = 0 if cfg["cache_in_ram"] else int(cfg["num_workers"])
    gen = torch.Generator()
    gen.manual_seed(seed)
    sampler = None
    if shuffle and cfg["weighted_sample"]:
        # SurvPath / MCAT: balance the (time bin, censorship) classes, sampling with replacement
        cls = ds.patients["survival_time_bin"].to_numpy() * 2 + (ds.patients["censorship"].to_numpy() > 0).astype(int)
        counts = np.bincount(cls, minlength=int(cls.max()) + 1).astype(float)
        weights = torch.as_tensor(1.0 / counts[cls], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=gen)
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_multimodal,
        pin_memory=(device.type == "cuda" and workers > 0),  # pinning in the main thread is slower than the copy it saves
        generator=gen if (shuffle and sampler is None) else None,
        persistent_workers=workers > 0,
    )


def build_optimizer(params, cfg: dict):
    name = str(cfg["optimizer"]).lower()
    lr, wd = float(cfg["lr"]), float(cfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")


def build_scheduler(optimizer, cfg: dict):
    warmup = int(cfg["warmup_epochs"])
    if warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg["epochs"] - warmup))
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cosine], milestones=[warmup])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])


# ---------------------------------------------------------------------------------------
# Forward / train / predict
# ---------------------------------------------------------------------------------------
def forward_batch(model, tokenizer, batch, device, modality="both", drop_wsi=None, drop_genomic=None, return_aux=False):
    wsi = batch["wsi_features"].to(device, non_blocking=True).float() if modality in ("both", "wsi") else None
    mask = batch["wsi_mask"].to(device, non_blocking=True) if wsi is not None else None
    gen = tokenizer(batch["gene_expression"].to(device, non_blocking=True)) if modality in ("both", "genomic") else None
    kwargs = {"return_aux": True} if return_aux else {}
    return model(wsi_features=wsi, genomic_features=gen, wsi_mask=mask, drop_wsi=drop_wsi, drop_genomic=drop_genomic, **kwargs)


def train_one_epoch(model, tokenizer, loader, criterion, optimizer, device, cfg) -> float:
    model.train()
    tokenizer.train()
    accum = int(cfg["grad_accum_steps"])
    params = list(model.parameters()) + list(tokenizer.parameters())
    optimizer.zero_grad(set_to_none=True)
    total, n = 0.0, 0
    aux_w = float(cfg["aux_unimodal_weight"]) if bool(getattr(model, "aux_heads", False)) else 0.0
    for step, batch in enumerate(loader):
        bins = batch["survival_time_bin"].to(device)
        cens = batch["censorship"].to(device)
        if aux_w > 0:
            logits, aux = forward_batch(model, tokenizer, batch, device, cfg["train_modalities"], return_aux=True)
            loss = criterion(logits, bins, cens)
            for key in ("h", "g"):  # unimodal heads, only on samples whose modality was really present
                if key in aux:
                    present = aux[f"{key}_present"]
                    if bool(present.any()):
                        loss = loss + aux_w * criterion(aux[key][present], bins[present], cens[present])
        else:
            logits = forward_batch(model, tokenizer, batch, device, cfg["train_modalities"])
            loss = criterion(logits, bins, cens)
        (loss / accum).backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            if cfg["clip_grad_norm"]:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg["clip_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        bs = batch["censorship"].shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def predict(model, tokenizer, loader, criterion, device, modality="both", drop_plan: dict | None = None) -> dict:
    """Run the model over a loader. ``drop_plan`` maps case_id -> 'wsi' | 'genomic' to force that
    modality absent for that patient (partial-missing simulation)."""
    model.eval()
    tokenizer.eval()
    logits_all, times, events, bins, case_ids = [], [], [], [], []
    total, n = 0.0, 0
    for batch in loader:
        dw = dg = None
        if drop_plan:
            dw = torch.tensor([drop_plan.get(c) == "wsi" for c in batch["case_ids"]], device=device)
            dg = torch.tensor([drop_plan.get(c) == "genomic" for c in batch["case_ids"]], device=device)
        logits = forward_batch(model, tokenizer, batch, device, modality, dw, dg)
        loss = criterion(logits, batch["survival_time_bin"].to(device), batch["censorship"].to(device))
        bs = logits.shape[0]
        total += loss.item() * bs
        n += bs
        logits_all.append(logits.float().cpu())
        times.append(batch["survival_time"])
        events.append(1.0 - batch["censorship"])
        bins.append(batch["survival_time_bin"])
        case_ids.extend(batch["case_ids"])
    logits = torch.cat(logits_all)
    _, S = hazards_to_survival(logits)
    return {
        "loss": total / max(n, 1),
        "logits": logits.numpy(),
        "survival": S.numpy(),
        "risk": risk_from_logits(logits).numpy(),
        "time": torch.cat(times).numpy().astype(float),
        "event": torch.cat(events).numpy().astype(bool),
        "bin": torch.cat(bins).numpy(),
        "case_id": case_ids,
    }


def metrics_from_prediction(pred: dict, train_ds, val_ds, cfg: dict, seed: int) -> dict:
    tr_event, tr_time = train_ds.surv_arrays()
    m = survival_metrics(tr_event, tr_time, pred["event"], pred["time"], pred["risk"], pred["survival"], val_ds.bins.edges)
    m["loss"] = float(pred["loss"])
    if int(cfg["bootstrap"]) > 0:
        lo, hi, se = bootstrap_cindex_ci(pred["event"], pred["time"], pred["risk"], n_boot=int(cfg["bootstrap"]), seed=seed)
        m["c_index_ci95"] = [lo, hi]
        m["c_index_boot_se"] = se
    m.update(km_logrank_split(pred["event"], pred["time"], pred["risk"]))
    return m


def final_evaluation(model, tokenizer, val_loader, train_ds, val_ds, criterion, device, cfg, seed):
    """Full metrics for the selected checkpoint under every test-time modality condition."""
    trained = cfg["train_modalities"]
    can_miss = bool(getattr(model, "supports_missing", True))
    modes = list(MODALITIES) if (trained == "both" and can_miss) else [trained]
    results, preds = {}, {}
    for mode in modes:
        p = predict(model, tokenizer, val_loader, criterion, device, modality=mode)
        key = "both" if mode == "both" else f"{mode}_only"
        preds[key] = p
        results[key] = metrics_from_prediction(p, train_ds, val_ds, cfg, seed)

    partial = []
    if trained == "both" and can_miss and cfg["eval_missing"] and cfg["missing_rates"]:
        case_ids = val_ds.patients["case_id"].tolist()
        rng = np.random.default_rng(seed + 1000)
        for rate in cfg["missing_rates"]:
            cis = []
            for _ in range(int(cfg["missing_repeats"])):
                k = int(round(float(rate) * len(case_ids)))
                chosen = rng.choice(len(case_ids), size=k, replace=False)
                plan = {case_ids[i]: ("wsi" if rng.random() < 0.5 else "genomic") for i in chosen}
                p = predict(model, tokenizer, val_loader, criterion, device, "both", drop_plan=plan)
                cis.append(concordance(p["event"], p["time"], p["risk"]))
            partial.append({
                "rate": float(rate),
                "c_index_mean": float(np.nanmean(cis)),
                "c_index_std": float(np.nanstd(cis)),
                "repeats": len(cis),
            })
    return results, partial, preds


def save_predictions(preds: dict, fold_dir: Path) -> None:
    import pandas as pd

    for key, p in preds.items():
        df = pd.DataFrame({"case_id": p["case_id"], "risk": p["risk"], "time": p["time"], "event": p["event"].astype(int), "bin": p["bin"]})
        for t in range(p["survival"].shape[1]):
            df[f"S{t}"] = p["survival"][:, t]
        df.to_csv(fold_dir / f"predictions_{key}.csv", index=False)


# ---------------------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------------------
def train_fold(cfg: dict, fold: int, device, run_dir: Path) -> dict:
    seed = int(cfg["seed"]) + fold
    set_seed(seed)
    t0 = time.time()
    print(f"\n{'=' * 64}\n  Fold {fold + 1}/{cfg['num_folds']}   (seed {seed})\n{'=' * 64}", flush=True)

    train_ds, val_ds = build_datasets(cfg, fold)
    print(f"  {train_ds.summary()}\n  {val_ds.summary()}", flush=True)
    print(f"  Bins (months): {[round(float(e), 1) for e in train_ds.bins.interior_edges()]}", flush=True)

    paths = data_paths(cfg, fold)
    model, tokenizer = build_model(cfg, train_ds.gene_columns, paths["composition"], device)
    n_tok = sum(p.numel() for p in tokenizer.parameters())
    n_model = sum(p.numel() for p in model.parameters())
    n_paths = getattr(tokenizer, "num_pathways", 0) or getattr(model, "num_pathways", 0)
    print(f"  Model: {cfg['model_type']} | pathways: {n_paths} | model params: {n_model:,} | tokenizer params: {n_tok:,}", flush=True)

    train_loader = make_loader(train_ds, cfg, shuffle=True, seed=seed, device=device)
    val_loader = make_loader(val_ds, cfg, shuffle=False, seed=seed, device=device)

    criterion = NLLSurvivalLoss(alpha=float(cfg["nll_alpha"]))
    params = list(model.parameters()) + list(tokenizer.parameters())
    optimizer = build_optimizer(params, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    fold_dir = run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = fold_dir / "best_checkpoint.pt", fold_dir / "latest_checkpoint.pt"

    sel = cfg["selection_metric"]
    better = (lambda a, b: a < b) if sel == "val_loss" else (lambda a, b: a > b)
    state = {"epoch": 0, "best": None, "best_epoch": -1, "patience": 0, "history": []}

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        tokenizer.load_state_dict(ckpt["pathway_tokenizer"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        print(f"  Resumed at epoch {state['epoch'] + 1} (best {sel} {state['best']:.4f} @ epoch {state['best_epoch']})", flush=True)

    for epoch in range(state["epoch"], int(cfg["epochs"])):
        te = time.time()
        train_loss = train_one_epoch(model, tokenizer, train_loader, criterion, optimizer, device, cfg)
        pred = predict(model, tokenizer, val_loader, criterion, device, cfg["train_modalities"])
        val_loss = float(pred["loss"])
        val_ci = concordance(pred["event"], pred["time"], pred["risk"])
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        rec = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_cindex": val_ci, "lr": lr, "seconds": time.time() - te}
        state["history"].append(rec)
        if sel == "last":  # fixed budget: every epoch replaces the checkpoint, no early stopping
            metric, improved = val_loss, True
        else:
            metric = val_loss if sel == "val_loss" else val_ci
            improved = state["best"] is None or (np.isfinite(metric) and better(metric, state["best"]))
        if improved:
            state.update(best=metric, best_epoch=epoch + 1, patience=0)
            torch.save({
                "model": model.state_dict(),
                "pathway_tokenizer": tokenizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_cindex": val_ci,
                "config": cfg,
                "bins": train_ds.bins.to_dict(),
                "scaler": train_ds.scaler.to_dict(),
                "gene_columns": train_ds.gene_columns,
                "pathway_names": tokenizer.get_pathway_names(),
            }, best_path)
        else:
            state["patience"] += 1
        state["epoch"] = epoch + 1

        print(
            f"  Epoch {epoch + 1:3d} | train {train_loss:.4f} | val {val_loss:.4f} | C-index {val_ci:.4f} "
            f"| lr {lr:.2e} | {rec['seconds']:5.0f}s{'  *' if improved else ''}",
            flush=True,
        )

        torch.save({
            "model": model.state_dict(),
            "pathway_tokenizer": tokenizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
        }, latest_path)

        if sel != "last" and state["patience"] >= int(cfg["patience"]) and epoch + 1 >= int(cfg["min_epochs"]):
            print(f"  Early stopping at epoch {epoch + 1} (no {sel} improvement for {cfg['patience']} epochs)", flush=True)
            break

    # ---- final evaluation on the selected checkpoint --------------------------------------
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    tokenizer.load_state_dict(ckpt["pathway_tokenizer"])
    results, partial, preds = final_evaluation(model, tokenizer, val_loader, train_ds, val_ds, criterion, device, cfg, seed)
    if cfg["save_predictions"]:
        save_predictions(preds, fold_dir)
    if not cfg["keep_latest_checkpoint"] and latest_path.exists():
        latest_path.unlink()

    hist = state["history"]
    primary = "both" if cfg["train_modalities"] == "both" else f"{cfg['train_modalities']}_only"
    fold_result = {
        "fold": fold,
        "seed": seed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "events_train": int(train_ds.surv_arrays()[0].sum()),
        "events_val": int(val_ds.surv_arrays()[0].sum()),
        "epochs_run": len(hist),
        "selected_epoch": state["best_epoch"],
        "selection_metric": sel,
        "c_index": results[primary]["c_index"],
        "best_val_cindex_any_epoch": float(np.nanmax([h["val_cindex"] for h in hist])),
        "last_epoch_val_cindex": hist[-1]["val_cindex"],
        "metrics": results,
        "partial_missing": partial,
        "bins": train_ds.bins.to_dict(),
        "history": hist,
        "seconds": time.time() - t0,
    }
    with open(fold_dir / "fold_results.json", "w") as f:
        json.dump(fold_result, f, indent=2)

    r = results[primary]
    ci = r.get("c_index_ci95", [float("nan"), float("nan")])
    print(
        f"  Selected epoch {state['best_epoch']} ({sel}) -> C-index {r['c_index']:.4f} "
        f"[95% CI {ci[0]:.3f}, {ci[1]:.3f}] | IPCW {r.get('c_index_ipcw', float('nan')):.4f} "
        f"| IBS {r.get('ibs', float('nan')):.4f} | td-AUC {r.get('iauc', float('nan')):.4f} "
        f"| log-rank p {r.get('logrank_p', float('nan')):.3g}",
        flush=True,
    )
    if "wsi_only" in results and "genomic_only" in results:
        print(f"  Missing-modality: WSI-only {results['wsi_only']['c_index']:.4f} | genomics-only {results['genomic_only']['c_index']:.4f}", flush=True)
    print(f"  Fold time: {fold_result['seconds'] / 60:.1f} min", flush=True)
    return fold_result


# ---------------------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------------------
def _progress_path(run_dir: Path) -> Path:
    return run_dir / "cv_progress.json"


def load_progress(run_dir: Path) -> dict:
    p = _progress_path(run_dir)
    if p.exists():
        with open(p) as f:
            prog = json.load(f)
        prog.setdefault("fold_results", {})
        prog.setdefault("fold_details", {})
        return prog
    return {"fold_results": {}, "fold_details": {}}


def save_progress(run_dir: Path, progress: dict) -> None:
    with open(_progress_path(run_dir), "w") as f:
        json.dump(progress, f, indent=2)


def aggregate(fold_details: list[dict]) -> dict:
    summary: dict = {"n_folds": len(fold_details)}
    if not fold_details:
        return summary
    summary["c_index_per_fold"] = [d["c_index"] for d in fold_details]
    summary["c_index"] = mean_ci95([d["c_index"] for d in fold_details])
    summary["best_val_cindex_any_epoch"] = mean_ci95([d["best_val_cindex_any_epoch"] for d in fold_details])
    summary["last_epoch_val_cindex"] = mean_ci95([d["last_epoch_val_cindex"] for d in fold_details])
    summary["selected_epochs"] = [d["selected_epoch"] for d in fold_details]

    conditions = sorted({k for d in fold_details for k in d["metrics"]})
    summary["conditions"] = {}
    for cond in conditions:
        block = {}
        for metric in ("c_index", "c_index_ipcw", "ibs", "iauc", "loss", "logrank_p"):
            vals = [d["metrics"][cond].get(metric) for d in fold_details if cond in d["metrics"]]
            block[metric] = mean_ci95(vals)
        summary["conditions"][cond] = block

    rates = sorted({pm["rate"] for d in fold_details for pm in d.get("partial_missing", [])})
    summary["partial_missing"] = []
    for rate in rates:
        vals = [pm["c_index_mean"] for d in fold_details for pm in d.get("partial_missing", []) if pm["rate"] == rate]
        summary["partial_missing"].append({"rate": rate, **mean_ci95(vals)})
    return summary


def summary_markdown(cfg: dict, summary: dict, fold_details: list[dict]) -> str:
    def f(x, nd=4):
        return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    lines = [
        f"# {cfg['cancer_type'].upper()} — {summary.get('n_folds', 0)}-fold CV ({cfg['endpoint'].upper()}, selection: {cfg['selection_metric']})",
        "",
        "| Test-time modalities | C-index (mean ± std) | 95% CI | IPCW C-index | IBS | td-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for cond, block in summary.get("conditions", {}).items():
        c = block["c_index"]
        lines.append(
            f"| {cond} | {f(c['mean'])} ± {f(c['std'])} | [{f(c['ci95'][0], 3)}, {f(c['ci95'][1], 3)}] "
            f"| {f(block['c_index_ipcw']['mean'])} | {f(block['ibs']['mean'])} | {f(block['iauc']['mean'])} |"
        )
    if summary.get("partial_missing"):
        lines += ["", "| Patients missing one modality | C-index |", "|---|---|"]
        for pm in summary["partial_missing"]:
            lines.append(f"| {int(pm['rate'] * 100)}% | {f(pm['mean'])} ± {f(pm['std'])} |")
    lines += ["", "| Fold | C-index (selected) | best any epoch | last epoch | selected ep / run | val events |", "|---|---|---|---|---|---|"]
    for d in fold_details:
        lines.append(
            f"| {d['fold']} | {f(d['c_index'])} | {f(d['best_val_cindex_any_epoch'])} | {f(d['last_epoch_val_cindex'])} "
            f"| {d['selected_epoch']} / {d['epochs_run']} | {d['events_val']} / {d['n_val']} |"
        )
    opt = summary.get("best_val_cindex_any_epoch", {})
    lines += ["", f"Optimistic (best epoch by val C-index, as in earlier logs): {f(opt.get('mean'))} ± {f(opt.get('std'))}"]
    return "\n".join(lines) + "\n"


def train_cv(cfg: dict, folds: list[int] | None = None) -> dict:
    cfg = with_defaults(cfg)
    device = get_device(cfg)
    threads = int(cfg["num_threads"] or physical_cores())
    if device.type == "cpu":
        torch.set_num_threads(threads)
    run_dir = Path(cfg["output_dir"]) / cfg["cancer_type"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Device: {device} ({threads} threads) | cancer: {cfg['cancer_type']} | endpoint: {cfg['endpoint']} "
          f"| sksurv: {HAS_SKSURV} | git: {git_commit()} | out: {run_dir}", flush=True)

    progress = load_progress(run_dir)
    folds = list(folds) if folds is not None else list(range(int(cfg["num_folds"])))
    for fold in folds:
        if str(fold) in progress["fold_details"]:
            print(f"\n  Fold {fold + 1}/{cfg['num_folds']} already done (C-index {progress['fold_results'][str(fold)]:.4f}), skipping", flush=True)
            continue
        fr = train_fold(cfg, fold, device, run_dir)
        progress["fold_results"][str(fold)] = fr["c_index"]
        progress["fold_details"][str(fold)] = {k: v for k, v in fr.items() if k != "history"}
        save_progress(run_dir, progress)

    details = [progress["fold_details"][k] for k in sorted(progress["fold_details"], key=int)]
    summary = aggregate(details)
    results = {
        "cancer_type": cfg["cancer_type"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "device": str(device),
        "sksurv": HAS_SKSURV,
        "config": cfg,
        "summary": summary,
        "folds": details,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    md = summary_markdown(cfg, summary, details)
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    return results


# ---------------------------------------------------------------------------------------
def parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PathQ-Former with k-fold CV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cancer_type", default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:N")
    parser.add_argument("--folds", default=None, help="comma-separated fold ids, e.g. 0,1")
    parser.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE", help="override config entries")
    parser.add_argument("--reset", action="store_true", help="ignore saved progress and start fresh")
    parser.add_argument("--smoke", action="store_true", help="tiny run on dummy embeddings to validate the pipeline")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    folds = [int(x) for x in args.folds.split(",")] if args.folds else None

    if args.smoke:  # tiny run on dummy embeddings; explicit --set overrides still win
        cfg.update({
            "embeddings_dir": "data/embeddings/uni2h_dummy",
            "output_dir": "outputs_smoke",
            "epochs": 2,
            "patience": 99,
            "bootstrap": 50,
            "missing_repeats": 1,
            "num_workers": 0,
        })
        folds = folds or [0]

    cfg.update(parse_overrides(args.set))
    if args.cancer_type:
        cfg["cancer_type"] = args.cancer_type
    if args.device:
        cfg["device"] = args.device

    run_dir = Path(cfg["output_dir"]) / cfg["cancer_type"]
    if args.reset and _progress_path(run_dir).exists():
        _progress_path(run_dir).unlink()
        print("Progress reset — starting fresh")

    train_cv(cfg, folds=folds)


if __name__ == "__main__":
    main()
