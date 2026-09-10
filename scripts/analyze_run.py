"""Post-hoc analysis and paper figures for one finished run directory.

    python scripts/analyze_run.py outputs_v2/hybrid/blca                  # curves, KM, missing-modality plot
    python scripts/analyze_run.py outputs_v2/hybrid/blca --attention      # + pathway importance and patch attention
    python scripts/analyze_run.py outputs_v2/hybrid/blca --figdir paper/figures/blca_hybrid

Writes to <run_dir>/analysis/ (and copies figures to --figdir if given):
  training_curves.png        train/val loss and val C-index per epoch, one panel per fold
  km_curves.png              pooled Kaplan-Meier curves, high vs low risk (fold-wise median split), log-rank p
  missing_modality.png       C-index vs share of patients missing one modality; WSI-only / genomics-only endpoints
  pathway_importance.csv/png mean genomic cross-attention per pathway (needs --attention)
  attention/<case>.npz       per-query patch attention + coordinates for the top/bottom-risk patients (needs --attention)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import build_datasets, build_model, data_paths, make_loader, with_defaults  # noqa: E402


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    folds = []
    for fr in sorted(run_dir.glob("fold_*/fold_results.json")):
        with open(fr) as f:
            folds.append(json.load(f))
    return results, folds


# --------------------------------------------------------------------------- figures
def plot_training_curves(folds: list[dict], out: Path) -> None:
    n = len(folds)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for ax, fr in zip(axes[0], folds):
        h = pd.DataFrame(fr["history"])
        ax.plot(h["epoch"], h["train_loss"], label="train loss", color="tab:gray")
        ax.plot(h["epoch"], h["val_loss"], label="val loss", color="tab:blue")
        ax.axvline(fr["selected_epoch"], color="k", ls=":", lw=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("NLL")
        ax2 = ax.twinx()
        ax2.plot(h["epoch"], h["val_cindex"], color="tab:red", label="val C-index")
        ax2.set_ylim(0.3, 0.9)
        ax2.axhline(0.5, color="tab:red", lw=0.5, alpha=0.4)
        ax.set_title(f"fold {fr['fold']}  (selected ep {fr['selected_epoch']}, C={fr['c_index']:.3f})", fontsize=9)
        if ax is axes[0][0]:
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_km(run_dir: Path, out: Path, condition: str = "both") -> dict:
    """Pool validation patients over folds; split high/low at each fold's median risk."""
    frames = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        p = fold_dir / f"predictions_{condition}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["high"] = df["risk"] > df["risk"].median()
        frames.append(df)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    kmf = KaplanMeierFitter()
    for flag, label, color in ((False, "low risk", "tab:blue"), (True, "high risk", "tab:red")):
        sub = df[df["high"] == flag]
        kmf.fit(sub["time"], sub["event"], label=f"{label} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
    res = logrank_test(df[df.high]["time"], df[~df.high]["time"], df[df.high]["event"], df[~df.high]["event"])
    ax.set_xlabel("months")
    ax.set_ylabel("disease-specific survival")
    ax.set_title(f"pooled validation folds, log-rank p = {res.p_value:.2e}", fontsize=9)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"logrank_p_pooled": float(res.p_value), "n": int(len(df))}


def plot_missing(results: dict, out: Path) -> None:
    s = results["summary"]
    conds = s.get("conditions", {})
    if "both" not in conds or not s.get("partial_missing"):
        return
    rates = [0.0] + [pm["rate"] for pm in s["partial_missing"]]
    means = [conds["both"]["c_index"]["mean"]] + [pm["mean"] for pm in s["partial_missing"]]
    stds = [conds["both"]["c_index"]["std"]] + [pm["std"] for pm in s["partial_missing"]]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.errorbar([r * 100 for r in rates], means, yerr=stds, marker="o", capsize=3, label="random missing (one modality)")
    for key, label, color in (("wsi_only", "WSI only (all patients)", "tab:green"), ("genomic_only", "genomics only (all patients)", "tab:purple")):
        if key in conds:
            ax.axhline(conds[key]["c_index"]["mean"], color=color, ls="--", label=label)
    ax.axhline(0.5, color="gray", lw=0.6)
    ax.set_xlabel("% of validation patients missing one modality at test time")
    ax.set_ylabel("C-index (mean ± std over folds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- attention
@torch.no_grad()
def attention_analysis(run_dir: Path, results: dict, out_dir: Path, n_heatmap: int, device) -> pd.DataFrame | None:
    cfg = with_defaults(results["config"])
    pathway_scores: dict[str, list[float]] = {}
    attn_dir = out_dir / "attention"
    attn_dir.mkdir(exist_ok=True)
    for fold_dir in sorted(run_dir.glob("fold_*")):
        ckpt_path = fold_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            continue
        fold = int(fold_dir.name.split("_")[1])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        train_ds, val_ds = build_datasets(cfg, fold)
        model, tok = build_model(cfg, train_ds.gene_columns, data_paths(cfg, fold)["composition"], device)
        model.load_state_dict(ckpt["model"])
        tok.load_state_dict(ckpt["pathway_tokenizer"])
        model.eval()
        tok.eval()
        names = tok.get_pathway_names()
        loader = make_loader(val_ds, {**cfg, "num_workers": 0, "batch_size": 1}, shuffle=False, seed=0, device=device)

        preds = pd.read_csv(fold_dir / "predictions_both.csv").set_index("case_id")
        order = preds["risk"].sort_values()
        heat_cases = set(order.index[:n_heatmap]) | set(order.index[-n_heatmap:])

        gen_attn_sum = np.zeros(len(names))
        n_pat = 0
        for batch in loader:
            case = batch["case_ids"][0]
            wsi = batch["wsi_features"].to(device)
            mask = batch["wsi_mask"].to(device)
            gen = tok(batch["gene_expression"].to(device))
            logits, attn = model(wsi, gen, mask, return_attention=True)
            g = attn["genomic"][0].mean(0).cpu().numpy()  # (P,) mean over queries
            gen_attn_sum += g
            n_pat += 1
            if case in heat_cases:
                h = attn["histology"][0].cpu().numpy()  # (K, N)
                coords = np.concatenate([c for c in (val_ds.load_coords(s) for s in batch["slide_ids"][0]) if c is not None]) \
                    if all(val_ds.load_coords(s) is not None for s in batch["slide_ids"][0]) else None
                np.savez_compressed(
                    attn_dir / f"fold{fold}_{case}.npz",
                    attention=h.astype(np.float32),
                    coords=coords if coords is not None else np.zeros((0, 2)),
                    slide_ids=np.array(batch["slide_ids"][0]),
                    risk=float(preds.loc[case, "risk"]),
                    time=float(preds.loc[case, "time"]),
                    event=int(preds.loc[case, "event"]),
                )
        for name, v in zip(names, gen_attn_sum / max(n_pat, 1)):
            pathway_scores.setdefault(name, []).append(float(v))
        print(f"  fold {fold}: attention over {n_pat} validation patients", flush=True)

    if not pathway_scores:
        return None
    df = pd.DataFrame({
        "pathway": list(pathway_scores),
        "attention_mean": [np.mean(v) for v in pathway_scores.values()],
        "attention_std": [np.std(v) for v in pathway_scores.values()],
        "n_folds": [len(v) for v in pathway_scores.values()],
    }).sort_values("attention_mean", ascending=False)
    df["attention_x_uniform"] = df["attention_mean"] * len(df)  # 1.0 = uniform attention
    df.to_csv(out_dir / "pathway_importance.csv", index=False)

    top = df.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.barh(top["pathway"].str.slice(0, 48), top["attention_x_uniform"], xerr=top["attention_std"] * len(df), color="tab:purple", alpha=0.85)
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("mean genomic cross-attention (× uniform)")
    ax.set_title("Top-30 pathways attended by the genomic queries (validation patients, all folds)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pathway_importance.png", dpi=150)
    plt.close(fig)
    return df


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--n_heatmap", type=int, default=4, help="patients per extreme (highest/lowest risk) per fold")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results, folds = load_run(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    plot_training_curves(folds, out / "training_curves.png")
    km = plot_km(run_dir, out / "km_curves.png")
    plot_missing(results, out / "missing_modality.png")
    print(f"figures -> {out}  {km}")

    if args.attention:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df = attention_analysis(run_dir, results, out, args.n_heatmap, device)
        if df is not None:
            print(df.head(15).to_string(index=False))

    if args.figdir:
        fig_dir = Path(args.figdir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.png"):
            shutil.copy(f, fig_dir / f.name)
        print(f"copied figures -> {fig_dir}")


if __name__ == "__main__":
    main()
