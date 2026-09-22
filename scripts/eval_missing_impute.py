"""Missing-modality evaluation with mean imputation for finished runs of ANY model.

    python scripts/eval_missing_impute.py outputs_e20/survpath_e20/blca outputs_e20/pathq_fast_e20_aux/blca

Baselines such as SurvPath cannot run without a modality, so the fair comparison is not "they crash" but
"they receive the training-set mean instead": the absent gene vector is replaced by the mean scaled gene
vector of the training split, the absent slide by a single patch equal to the mean training patch
embedding. Each fold's selected checkpoint is re-evaluated under both conditions; the metrics are stored
in fold_results.json under ``wsi_only_impute`` / ``genomic_only_impute`` and rolled up into results.json
and summary.md, so aggregate_results.py shows them as extra test-time conditions.

``--permute`` adds a permutation test: the validation patients' gene vectors (``rna_permuted``) or slides
(``wsi_permuted``) are shuffled across patients, so each patient is scored with someone else's modality. A model
that ignores a modality is unaffected by permuting it; the drop measures how much of the score that modality
carries in the presence of the other. One permutation per fold (seeded), 15 per cohort over 3 seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import hazards_to_survival, risk_from_logits  # noqa: E402
from src.training.train import (  # noqa: E402
    NLLSurvivalLoss,
    aggregate,
    build_datasets,
    build_model,
    data_paths,
    load_progress,
    make_loader,
    metrics_from_prediction,
    save_progress,
    summary_markdown,
    with_defaults,
)


def training_means(train_ds, n_slides: int = 60) -> tuple[torch.Tensor, torch.Tensor]:
    gene_mean = torch.from_numpy(train_ds.gene_matrix.mean(0))
    sums, count = None, 0
    for idx in range(min(n_slides, len(train_ds))):
        feats = train_ds[idx]["wsi_features"].float()
        sums = feats.sum(0) if sums is None else sums + feats.sum(0)
        count += feats.shape[0]
    return gene_mean, sums / max(count, 1)


@torch.no_grad()
def predict_imputed(model, tokenizer, loader, criterion, device, mode: str, gene_mean, patch_mean) -> dict:
    model.eval()
    tokenizer.eval()
    logits_all, times, events, total, n = [], [], [], 0.0, 0
    for batch in loader:
        B = batch["gene_expression"].shape[0]
        if mode == "genomic_only":  # WSI absent -> one mean patch per patient
            wsi = patch_mean.view(1, 1, -1).expand(B, 1, -1).contiguous().to(device)
            mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            gene = batch["gene_expression"].to(device)
        else:  # WSI only: RNA absent -> mean gene vector
            wsi = batch["wsi_features"].to(device).float()
            mask = batch["wsi_mask"].to(device)
            gene = gene_mean.view(1, -1).expand(B, -1).contiguous().to(device)
        logits = model(wsi_features=wsi, genomic_features=tokenizer(gene), wsi_mask=mask)
        loss = criterion(logits, batch["survival_time_bin"].to(device), batch["censorship"].to(device))
        total += loss.item() * B
        n += B
        logits_all.append(logits.float().cpu())
        times.append(batch["survival_time"])
        events.append(1.0 - batch["censorship"])
    logits = torch.cat(logits_all)
    _, S = hazards_to_survival(logits)
    return {"loss": total / max(n, 1), "risk": risk_from_logits(logits).numpy(), "survival": S.numpy(),
            "time": torch.cat(times).numpy().astype(float), "event": torch.cat(events).numpy().astype(bool)}


@torch.no_grad()
def predict_permuted(model, tokenizer, val_ds, loader, criterion, device, mode: str, seed: int) -> dict:
    """Score patient i with patient perm[i]'s RNA (rna_permuted) or slides (wsi_permuted)."""
    model.eval()
    tokenizer.eval()
    import numpy as np

    perm = np.random.default_rng(seed).permutation(len(val_ds))
    gene_matrix = getattr(val_ds, "gene_matrix", None)
    logits_all, times, events, total, n = [], [], [], 0.0, 0
    for i, batch in enumerate(loader):
        assert batch["gene_expression"].shape[0] == 1, "permutation eval expects batch_size 1"
        j = int(perm[i])
        if mode == "rna_permuted":
            wsi = batch["wsi_features"].to(device).float()
            mask = batch["wsi_mask"].to(device)
            gene = (torch.as_tensor(gene_matrix[j]).float().view(1, -1) if gene_matrix is not None
                    else val_ds[j]["gene_expression"].view(1, -1)).to(device)
        else:
            other = val_ds[j]["wsi_features"].float().unsqueeze(0)
            wsi = other.to(device)
            mask = torch.ones(1, other.shape[1], dtype=torch.bool, device=device)
            gene = batch["gene_expression"].to(device)
        logits = model(wsi_features=wsi, genomic_features=tokenizer(gene), wsi_mask=mask)
        loss = criterion(logits, batch["survival_time_bin"].to(device), batch["censorship"].to(device))
        total += loss.item()
        n += 1
        logits_all.append(logits.float().cpu())
        times.append(batch["survival_time"])
        events.append(1.0 - batch["censorship"])
    logits = torch.cat(logits_all)
    _, S = hazards_to_survival(logits)
    return {"loss": total / max(n, 1), "risk": risk_from_logits(logits).numpy(), "survival": S.numpy(),
            "time": torch.cat(times).numpy().astype(float), "event": torch.cat(events).numpy().astype(bool)}


def process_run(run_dir: Path, device, bootstrap: int | None, permute: bool = False, impute: bool = True) -> None:
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    cfg = with_defaults(results["config"])
    if bootstrap is not None:
        cfg["bootstrap"] = bootstrap
    cfg["cache_in_ram"] = False
    if cfg["train_modalities"] != "both":
        print(f"{run_dir}: single-modality model, nothing to impute")
        return
    criterion = NLLSurvivalLoss(alpha=float(cfg["nll_alpha"]))
    progress = load_progress(run_dir)
    for fr_path in sorted(run_dir.glob("fold_*/fold_results.json")):
        ckpt_path = fr_path.parent / "best_checkpoint.pt"
        if not ckpt_path.exists():
            print(f"  {fr_path.parent.name}: no checkpoint, skipped")
            continue
        with open(fr_path) as f:
            fr = json.load(f)
        fold, seed = fr["fold"], fr["seed"]
        train_ds, val_ds = build_datasets(cfg, fold)
        model, tok = build_model(cfg, train_ds.gene_columns, data_paths(cfg, fold)["composition"], device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        tok.load_state_dict(ckpt["pathway_tokenizer"])
        gene_mean, patch_mean = training_means(train_ds)
        loader = make_loader(val_ds, {**cfg, "num_workers": 0, "batch_size": 1}, shuffle=False, seed=seed, device=device)
        if impute:
            for mode in ("wsi_only", "genomic_only"):
                pred = predict_imputed(model, tok, loader, criterion, device, mode, gene_mean, patch_mean)
                fr["metrics"][f"{mode}_impute"] = metrics_from_prediction(pred, train_ds, val_ds, cfg, seed)
            fr["imputation_evaluated_at"] = datetime.now().isoformat(timespec="seconds")
        if permute:
            for mode in ("rna_permuted", "wsi_permuted"):
                pred = predict_permuted(model, tok, val_ds, loader, criterion, device, mode, seed)
                fr["metrics"][mode] = metrics_from_prediction(pred, train_ds, val_ds, cfg, seed)
            fr["permutation_evaluated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(fr_path, "w") as f:
            json.dump(fr, f, indent=2)
        progress["fold_details"][str(fold)] = {k: v for k, v in fr.items() if k != "history"}
        m = fr["metrics"]
        parts = [f"both {m['both']['c_index']:.3f}"]
        for k, label in (("wsi_only_impute", "WSI-only(impute)"), ("genomic_only_impute", "RNA-only(impute)"),
                         ("rna_permuted", "RNA permuted"), ("wsi_permuted", "WSI permuted")):
            if k in m:
                parts.append(f"{label} {m[k]['c_index']:.3f}")
        print(f"  fold {fold}: " + " | ".join(parts), flush=True)
    save_progress(run_dir, progress)
    details = [progress["fold_details"][k] for k in sorted(progress["fold_details"], key=int)]
    results["summary"] = aggregate(details)
    results["folds"] = details
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    (run_dir / "summary.md").write_text(summary_markdown(cfg, results["summary"], details), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--permute", action="store_true", help="also run the RNA / WSI permutation test")
    ap.add_argument("--no-impute", action="store_true", help="skip the mean-imputation conditions (e.g. permutation only)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for rd in args.run_dirs:
        run_dir = Path(rd)
        if not (run_dir / "results.json").exists():
            print(f"{run_dir}: not finished, skipped")
            continue
        print(f"== {run_dir}")
        process_run(run_dir, device, args.bootstrap, permute=args.permute, impute=not args.no_impute)


if __name__ == "__main__":
    main()
