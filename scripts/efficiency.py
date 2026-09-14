"""Compute cost of PathQ-Former vs the official SurvPath on real validation patients.

    python scripts/efficiency.py --config configs/protocol_fixed/pathq_fast_e20_aux.yaml \
        --baseline configs/protocol_fixed/survpath_e20.yaml --n 40 --out results/efficiency.md

Reports trainable parameters, median forward latency per patient, median forward+backward latency, and
peak GPU memory, each model on the input it was trained with (PathQ-Former: every patch; SurvPath: its
4096-patch subsample) and PathQ-Former additionally on a 4096-patch subsample for a like-for-like point.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import NLLSurvivalLoss  # noqa: E402
from src.training.train import build_datasets, build_model, data_paths, with_defaults  # noqa: E402


def timeit(fn, n_warmup=3, n=20, device=None):
    for _ in range(n_warmup):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def profile(cfg_path: str, label: str, device, n_patients: int, max_patches: int | None):
    cfg = with_defaults(yaml.safe_load(open(cfg_path)))
    cfg.update(cache_in_ram=False, num_workers=0)
    if not Path(cfg["embeddings_dir"]).exists():
        cfg["embeddings_dir"] = "data/embeddings/uni2h_dummy"
    train_ds, val_ds = build_datasets(cfg, 0)
    model, tok = build_model(cfg, train_ds.gene_columns, data_paths(cfg, 0)["composition"], device)
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in tok.parameters())
    crit = NLLSurvivalLoss()
    fwd, fb, patches = [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(min(n_patients, len(val_ds))):
        item = val_ds[i]
        x = item["wsi_features"].float()
        if max_patches and x.shape[0] > max_patches:
            x = x[torch.randperm(x.shape[0])[:max_patches]]
        patches.append(x.shape[0])
        x = x.unsqueeze(0).to(device)
        g = item["gene_expression"].unsqueeze(0).to(device)
        mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
        bins = item["survival_time_bin"].view(1).to(device)
        cens = item["censorship"].view(1).to(device)
        model.eval()
        with torch.no_grad():
            fwd.append(timeit(lambda: model(wsi_features=x, genomic_features=tok(g), wsi_mask=mask), n=5, device=device))
        model.train()

        def step():
            loss = crit(model(wsi_features=x, genomic_features=tok(g), wsi_mask=mask), bins, cens)
            loss.backward()
            model.zero_grad(set_to_none=True)
            tok.zero_grad(set_to_none=True)

        fb.append(timeit(step, n=3, device=device))
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    return {"label": label, "params_M": n_params / 1e6, "patches_median": float(np.median(patches)), "fwd_ms": float(np.median(fwd)),
            "fwd_bwd_ms": float(np.median(fb)), "peak_gib": peak}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_fixed/pathq_fast_e20_aux.yaml")
    ap.add_argument("--baseline", default="configs/protocol_fixed/survpath_e20.yaml")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="results/efficiency.md")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        profile(args.config, "PathQ-Former (all patches)", device, args.n, None),
        profile(args.config, "PathQ-Former (4096 patches)", device, args.n, 4096),
        profile(args.baseline, "SurvPath (4096 patches, as trained)", device, args.n, 4096),
        profile(args.baseline, "SurvPath (all patches)", device, args.n, None),
    ]
    lines = [f"# Compute cost per patient ({device}, BLCA fold-0 validation, {args.n} patients, medians)", "",
             "| Model | Params (M) | Patches | Forward (ms) | Forward+backward (ms) | Peak GPU mem (GiB) |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['params_M']:.1f} | {r['patches_median']:.0f} | {r['fwd_ms']:.1f} | {r['fwd_bwd_ms']:.1f} | {r['peak_gib']:.2f} |")
    md = "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
