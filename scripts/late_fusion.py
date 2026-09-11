"""Late-fusion baseline: combine two finished single-modality runs fold by fold.

    python scripts/late_fusion.py outputs_v2/pathq_fast_e10_wsi_only/blca outputs_v2/pathq_fast_e10_genomic_only/blca \
        --out outputs_v2/late_fusion_fast_e10/blca

Per fold, each model's risk score is z-scored over the validation patients and the two are averaged
(equal weights by default); survival curves are averaged directly. The result is written in the same
layout as a trained run (fold_results.json, predictions_both.csv, results.json, summary.md), so
scripts/aggregate_results.py treats it like any other method. If the joint model does not beat this
baseline, the fusion block is not extracting complementary signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import aggregate, build_datasets, metrics_from_prediction, summary_markdown, with_defaults  # noqa: E402


def load_run(run_dir: Path) -> tuple[dict, dict[int, Path]]:
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    folds = {}
    for fr in run_dir.glob("fold_*/fold_results.json"):
        with open(fr) as f:
            d = json.load(f)
        key = next(iter(d["metrics"]))
        folds[d["fold"]] = fr.parent / f"predictions_{key}.csv"
    return results, folds


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs=2, type=float, default=(0.5, 0.5))
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    res_a, folds_a = load_run(Path(args.run_a))
    res_b, folds_b = load_run(Path(args.run_b))
    cfg = with_defaults(res_a["config"])
    cfg.update(train_modalities="both", selection_metric="late_fusion", bootstrap=args.bootstrap)
    if not Path(cfg["embeddings_dir"]).exists():
        cfg["embeddings_dir"] = "data/embeddings/uni2h_dummy"  # labels only
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wa, wb = args.weights

    details = []
    for fold in sorted(set(folds_a) & set(folds_b)):
        a = pd.read_csv(folds_a[fold]).set_index("case_id")
        b = pd.read_csv(folds_b[fold]).set_index("case_id")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        s_cols = [c for c in a.columns if c.startswith("S")]
        risk = wa * zscore(a["risk"].to_numpy(float)) + wb * zscore(b["risk"].to_numpy(float))
        S = wa * a[s_cols].to_numpy(float) + wb * b[s_cols].to_numpy(float)
        pred = {"loss": float("nan"), "event": a["event"].to_numpy(bool), "time": a["time"].to_numpy(float), "risk": risk, "survival": S}
        train_ds, val_ds = build_datasets(cfg, fold)
        seed = int(cfg["seed"]) + fold
        m = metrics_from_prediction(pred, train_ds, val_ds, cfg, seed)

        fold_dir = out / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        df = pd.DataFrame({"case_id": common, "risk": risk, "time": pred["time"], "event": pred["event"].astype(int), "bin": a["bin"].to_numpy()})
        for t in range(S.shape[1]):
            df[f"S{t}"] = S[:, t]
        df.to_csv(fold_dir / "predictions_both.csv", index=False)
        fr = {
            "fold": fold, "seed": seed, "n_train": len(train_ds), "n_val": int(len(common)),
            "events_train": int(train_ds.surv_arrays()[0].sum()), "events_val": int(pred["event"].sum()),
            "epochs_run": 0, "selected_epoch": -1, "selection_metric": "late_fusion",
            "c_index": m["c_index"], "best_val_cindex_any_epoch": float("nan"), "last_epoch_val_cindex": float("nan"),
            "metrics": {"both": m}, "partial_missing": [], "bins": train_ds.bins.to_dict(), "history": [],
            "sources": {"a": str(folds_a[fold]), "b": str(folds_b[fold]), "weights": [wa, wb]},
        }
        with open(fold_dir / "fold_results.json", "w") as f:
            json.dump(fr, f, indent=2)
        details.append({k: v for k, v in fr.items() if k != "history"})
        print(f"fold {fold}: n={len(common)} C={m['c_index']:.4f} (A={res_a['folds'][fold]['c_index']:.4f}, B={res_b['folds'][fold]['c_index']:.4f})")

    summary = aggregate(details)
    results = {
        "cancer_type": cfg["cancer_type"], "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": "late_fusion", "sources": [args.run_a, args.run_b], "weights": [wa, wb],
        "config": cfg, "summary": summary, "folds": details,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out / "cv_progress.json", "w") as f:
        json.dump({"fold_results": {str(d["fold"]): d["c_index"] for d in details}, "fold_details": {str(d["fold"]): d for d in details}}, f, indent=2)
    md = summary_markdown(cfg, summary, details)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
