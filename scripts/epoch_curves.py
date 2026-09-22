"""Mean validation C-index per epoch across folds (and cohorts) for fixed-budget runs.

    python scripts/epoch_curves.py pod_results/outputs_v2 pod_results/outputs_ablate --runs pathq_fast_e10 survpath_e10

Every fold stores its per-epoch history, so the C-index that a *shorter* fixed budget would have
reported can be read off without retraining: the value at epoch k is what "train k epochs and
report the final checkpoint" would give. Choosing k from the curve pooled over all cohorts and
seeds, and applying the same k to every method, keeps the protocol symmetric.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect(roots: list[str], run_prefixes: list[str]) -> dict:
    """{method: {cohort: [history, ...]}} where history is a list of val_cindex per epoch."""
    out: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for root in roots:
        for fr in Path(root).glob("*/*/fold_*/fold_results.json"):
            run, cohort = fr.parts[-4], fr.parts[-3]
            method = next((p for p in run_prefixes if run == p or run.startswith(p + "_seed")), None)
            if method is None:
                continue
            with open(fr) as f:
                d = json.load(f)
            hist = [h["val_cindex"] for h in d.get("history", [])]
            if hist:
                out[method][cohort].append(hist)
    return out


def curve(histories: list[list[float]], max_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    m = np.full((len(histories), max_epochs), np.nan)
    for i, h in enumerate(histories):
        n = min(len(h), max_epochs)
        m[i, :n] = h[:n]
    return np.nanmean(m, axis=0), np.sum(~np.isnan(m), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--runs", nargs="+", default=["pathq_fast_e10", "survpath_e10"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--plot", default=None, help="save the pooled curves as a figure (png/pdf)")
    ap.add_argument("--labels", nargs="+", default=None, help="legend labels, one per --runs entry")
    args = ap.parse_args()
    pooled_curves = {}

    data = collect(args.roots, args.runs)
    for method, by_cohort in data.items():
        print(f"\n== {method} ==")
        header = "cohort      n_folds  " + " ".join(f"ep{e + 1:<4d}" for e in range(args.epochs))
        print(header)
        pooled = []
        for cohort, hists in sorted(by_cohort.items()):
            mean, n = curve(hists, args.epochs)
            pooled += hists
            print(f"{cohort:10s} {len(hists):6d}   " + " ".join(f"{v:.3f}" if np.isfinite(v) else "  -  " for v in mean))
        mean, n = curve(pooled, args.epochs)
        pooled_curves[method] = mean
        best = int(np.nanargmax(mean)) + 1
        print(f"{'POOLED':10s} {len(pooled):6d}   " + " ".join(f"{v:.3f}" for v in mean) + f"   <- best budget: {best} epochs ({mean[best - 1]:.3f}); at 10: {mean[-1]:.3f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = dict(zip(args.runs, args.labels or args.runs))
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for method, mean in pooled_curves.items():
            ax.plot(np.arange(1, len(mean) + 1), mean, marker="o", ms=3, label=labels.get(method, method))
        ax.set_xlabel("training epochs (fixed budget, final checkpoint)")
        ax.set_ylabel("pooled validation C-index")
        ax.set_xticks(range(1, args.epochs + 1, max(1, args.epochs // 10)))
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=200)
        print(f"figure -> {args.plot}")


if __name__ == "__main__":
    main()
