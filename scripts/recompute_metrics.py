"""Recompute every metric block of a finished run from its saved per-patient predictions with the
*current* ``src/training/evaluate.py`` and rewrite fold_results.json, cv_progress.json, results.json
and summary.md. Training is untouched; this only keeps runs produced by older evaluation code
consistent with the current metric definitions (grid, IPCW horizon, bootstrap).

    python scripts/recompute_metrics.py outputs_v2/hybrid/blca [outputs_v2/wsi_only/blca ...] [--bootstrap 1000]

Run it only on runs that have finished (a live run would overwrite the recomputed files at its end).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import (  # noqa: E402
    aggregate,
    build_datasets,
    load_progress,
    metrics_from_prediction,
    save_progress,
    summary_markdown,
    with_defaults,
)


def recompute_run(run_dir: Path, bootstrap: int | None) -> None:
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    cfg = with_defaults(results["config"])
    if bootstrap is not None:
        cfg["bootstrap"] = bootstrap
    if not Path(cfg["embeddings_dir"]).exists():  # labels only are needed here
        cfg["embeddings_dir"] = "data/embeddings/uni2h_dummy"
    primary = "both" if cfg["train_modalities"] == "both" else f"{cfg['train_modalities']}_only"

    progress = load_progress(run_dir)
    for fr_path in sorted(run_dir.glob("fold_*/fold_results.json")):
        with open(fr_path) as f:
            fr = json.load(f)
        fold, seed = fr["fold"], fr["seed"]
        train_ds, val_ds = build_datasets(cfg, fold)
        for key in list(fr["metrics"]):
            pred_csv = fr_path.parent / f"predictions_{key}.csv"
            if not pred_csv.exists():
                print(f"  fold {fold}: no {pred_csv.name}, keeping stored metrics for {key}")
                continue
            df = pd.read_csv(pred_csv)
            pred = {
                "loss": fr["metrics"][key].get("loss", float("nan")),
                "event": df["event"].to_numpy(bool),
                "time": df["time"].to_numpy(float),
                "risk": df["risk"].to_numpy(float),
                "survival": df[[c for c in df.columns if c.startswith("S")]].to_numpy(float),
            }
            fr["metrics"][key] = metrics_from_prediction(pred, train_ds, val_ds, cfg, seed)
        fr["c_index"] = fr["metrics"][primary]["c_index"]
        fr["metrics_recomputed_at"] = datetime.now().isoformat(timespec="seconds")
        with open(fr_path, "w") as f:
            json.dump(fr, f, indent=2)
        progress["fold_results"][str(fold)] = fr["c_index"]
        progress["fold_details"][str(fold)] = {k: v for k, v in fr.items() if k != "history"}
        m = fr["metrics"][primary]
        print(f"  fold {fold}: C={m['c_index']:.4f} IPCW(tau={m.get('ipcw_tau', float('nan')):.1f})={m.get('c_index_ipcw', float('nan')):.4f} "
              f"IBS={m.get('ibs', float('nan')):.4f} td-AUC={m.get('iauc', float('nan')):.4f}")

    save_progress(run_dir, progress)
    details = [progress["fold_details"][k] for k in sorted(progress["fold_details"], key=int)]
    results["summary"] = aggregate(details)
    results["folds"] = details
    results["metrics_recomputed_at"] = datetime.now().isoformat(timespec="seconds")
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    (run_dir / "summary.md").write_text(summary_markdown(cfg, results["summary"], details), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--bootstrap", type=int, default=None, help="override the bootstrap sample count")
    args = ap.parse_args()
    for rd in args.run_dirs:
        run_dir = Path(rd)
        if not (run_dir / "results.json").exists():
            print(f"{run_dir}: no results.json (run not finished?) - skipped")
            continue
        print(f"== {run_dir}")
        recompute_run(run_dir, args.bootstrap)


if __name__ == "__main__":
    main()
