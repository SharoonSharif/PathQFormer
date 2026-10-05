"""Paper figures for the TMLR revision: Kaplan-Meier grid and pathway importance (v2).

    .venv/Scripts/python.exe scripts/paper_figures_km.py --outdir <paper>/figures [--preview_dir <dir>]

Writes vector PDFs (serif, 9 pt, sized for the 5.5 in TMLR text width) into --outdir:

  km_grid.pdf                2 x 5 Kaplan-Meier grid. Rows: PathQ-Former (pathq_fast_e20_aux) and
                             SurvPath (survpath_e20); columns: BLCA, BRCA, COADREAD, HNSC, STAD.
                             High- vs low-risk halves of the pooled out-of-fold validation
                             predictions of seed 0, 95% confidence bands, log-rank p per panel.
  pathway_importance_v2.pdf  Top-k pathway tokens by mean genomic cross-attention per cohort for
                             PathQ-Former, one panel per cohort, asymmetric error bars.

Optional support files (not paper figures) go to --preview_dir: a PNG preview of each figure and a
JSON file with every number behind the panels.

How the KM split and pooling are done (identical to scripts/analyze_run.py::plot_km):
  * for every fold k = 0..4 read <run>/<cohort>/fold_<k>/predictions_both.csv, one row per
    validation patient with its predicted risk, disease-specific survival time (months) and event
    indicator written by the training script at the final (20th) epoch;
  * within each fold, flag a patient "high risk" when risk > median(risk) of that fold's
    validation set (fold-wise median split, so each fold model is only compared with itself);
  * concatenate the five folds (every patient appears exactly once, as a validation patient of
    one fold) and fit a Kaplan-Meier curve per group on the pooled rows; the log-rank test is the
    two-sample test between the pooled high- and low-risk groups (lifelines.logrank_test).

Pathway importance:
  * reads <run>[_seed1|_seed2]/<cohort>/analysis/pathway_importance.csv (written by
    scripts/analyze_run.py --attention: mean over validation patients and folds of the genomic
    cross-attention per pathway token, plus its std over folds). Seeds that have no CSV are skipped
    and reported;
  * the point estimate is the seed-average of attention_mean, multiplied by the number of pathway
    tokens so that 1.0 is uniform attention;
  * error bars: if per-(fold, seed) values are available (columns attention_fold<k> in the CSV or
    a long-format pathway_importance_folds.csv with columns pathway, fold, attention next to it),
    the bars span the 25-75% quantiles over folds x seeds, clipped at 0. Otherwise (the stored
    CSVs carry only mean and std) the bars are mean +- std over folds with the lower end clipped
    at 0, and the JSON/console output says so.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "pod_results" / "outputs_e20"
COHORTS = ["blca", "brca", "coadread", "hnsc", "stad"]
COHORT_LABEL = {c: c.upper() for c in COHORTS}
KM_METHODS = [("PathQ-Former", "pathq_fast_e20_aux"), ("SurvPath", "survpath_e20")]
PATHWAY_RUN = "pathq_fast_e20_aux"
SEED_SUFFIX = {0: "", 1: "_seed1", 2: "_seed2"}

# Colour-blind-safe categorical slots 1 and 2 of the dataviz reference palette (validated:
# adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on white) plus its ink/chrome tokens.
C_LOW, C_HIGH = "#2a78d6", "#eb6834"  # KM: low-risk half, high-risk half
C_REACTOME, C_HALLMARK = "#2a78d6", "#eb6834"  # pathways: Reactome, MSigDB Hallmark
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def check_out(path: Path, force: bool) -> None:
    if path.exists() and not force:
        sys.exit(f"refusing to overwrite existing {path} (pass --force if it is this script's own output)")


def save(fig: plt.Figure, pdf: Path, preview_dir: Path | None) -> list[str]:
    written = []
    fig.savefig(pdf)
    written.append(str(pdf))
    if preview_dir is not None:
        png = preview_dir / (pdf.stem + ".png")
        fig.savefig(png, dpi=200)
        written.append(str(png))
    plt.close(fig)
    return written


def finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


# --------------------------------------------------------------------------- Kaplan-Meier grid
def pooled_predictions(run_dir: Path, condition: str = "both") -> pd.DataFrame:
    """Pool validation patients over folds; fold-wise median split (scripts/analyze_run.py::plot_km)."""
    frames = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        p = fold_dir / f"predictions_{condition}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["fold"] = int(fold_dir.name.split("_")[1])
        df["high"] = df["risk"] > df["risk"].median()
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no fold_*/predictions_{condition}.csv under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def fmt_p(p: float) -> str:
    if p < 1e-3:
        exp = int(math.floor(math.log10(p)))
        mant = p / 10 ** exp
        if round(mant) >= 10:
            mant, exp = 1.0, exp + 1
        return rf"$p$ = {mant:.0f}$\times$10$^{{{exp}}}$"
    if p < 0.1:
        return rf"$p$ = {p:.3f}"
    return rf"$p$ = {p:.2f}"


def km_panel(ax: plt.Axes, df: pd.DataFrame) -> dict:
    kmf = KaplanMeierFitter(alpha=0.05)
    stats: dict = {}
    for flag, label, color in ((False, "low", C_LOW), (True, "high", C_HIGH)):
        sub = df[df["high"] == flag]
        kmf.fit(sub["time"], sub["event"], label=label)
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        t = sf.index.to_numpy()
        ax.fill_between(t, ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy(), step="post",
                        color=color, alpha=0.15, lw=0)
        ax.step(t, sf.iloc[:, 0].to_numpy(), where="post", color=color, lw=1.0, solid_joinstyle="round")
        stats[label] = {
            "n": int(len(sub)),
            "events": int(sub["event"].sum()),
            "median_survival_months": finite(kmf.median_survival_time_),
            "max_follow_up_months": float(sub["time"].max()),
        }
    res = logrank_test(df[df.high]["time"], df[~df.high]["time"], df[df.high]["event"], df[~df.high]["event"])
    stats["logrank_p"] = float(res.p_value)
    stats["logrank_statistic"] = float(res.test_statistic)
    stats["n_pooled"] = int(len(df))
    stats["n_per_fold"] = [int(v) for v in df.groupby("fold").size().sort_index()]
    return stats


def figure_km(out_pdf: Path, preview_dir: Path | None) -> tuple[list[str], dict]:
    fig, axes = plt.subplots(2, 5, figsize=(5.5, 2.85), sharex="col", sharey=True)
    numbers: dict = {}
    for j, cohort in enumerate(COHORTS):
        t_max = 0.0
        n_pooled = None
        for i, (method, run) in enumerate(KM_METHODS):
            df = pooled_predictions(RESULTS / run / cohort)
            ax = axes[i, j]
            st = km_panel(ax, df)
            numbers[f"{method}|{COHORT_LABEL[cohort]}"] = st
            t_max = max(t_max, float(df["time"].max()))
            n_pooled = st["n_pooled"]
            ax.text(0.05, 0.06, fmt_p(st["logrank_p"]), transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8, color=INK)
        axes[0, j].set_title(f"{COHORT_LABEL[cohort]}\n$n$ = {n_pooled}", fontsize=8.5, pad=3, linespacing=1.15)
        axes[1, j].set_xlim(0, t_max * 1.03)
        step = 50 if t_max <= 200 else 100  # 3-4 labelled ticks per 0.9 in panel
        axes[1, j].set_xticks(np.arange(0, t_max + 1, step))
    for i, (method, _) in enumerate(KM_METHODS):
        axes[i, 0].set_ylabel(method, fontsize=8.5, color=INK, labelpad=3)
    for ax in axes.flat:
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1.0])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=2, pad=1.5)
    fig.supylabel("Disease-specific survival", fontsize=8.5, x=0.012, color=INK)
    fig.supxlabel("Months", fontsize=8.5, y=0.115, color=INK)
    handles = [(Patch(facecolor=c, alpha=0.18, lw=0), Line2D([0], [0], color=c, lw=1.2)) for c in (C_LOW, C_HIGH)]
    fig.legend(handles, ["Low-risk half", "High-risk half"], handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
               loc="lower center", ncol=2, bbox_to_anchor=(0.56, -0.005), handlelength=1.8, columnspacing=1.6,
               handletextpad=0.5, borderaxespad=0.0)
    fig.subplots_adjust(left=0.118, right=0.99, top=0.885, bottom=0.235, wspace=0.14, hspace=0.22)
    return save(fig, out_pdf, preview_dir), numbers


# --------------------------------------------------------------------------- pathway importance
ACRONYMS = {"KRAS", "DN", "UP", "IL2", "IL6", "JAK", "STAT3", "STAT5", "MTORC1", "TNFA", "NFKB", "PI3K", "AKT",
            "MTOR", "TGF", "UV", "DNA", "G2M", "E2F", "MYC", "V1", "V2", "WNT", "IFN", "ROS", "UPR", "EMT", "TNF"}


def pretty_name(raw: str, max_len: int = 45) -> str:
    """Full pathway name; Hallmark sets get an (H) prefix; shortened with an ellipsis only beyond max_len."""
    if raw.startswith("HALLMARK_"):
        words = raw[len("HALLMARK_"):].split("_")
        out = []
        for i, w in enumerate(words):
            if w in ACRONYMS:
                out.append(w)
            elif w == "P53":
                out.append("p53")
            elif i == 0:
                out.append(w.capitalize())
            else:
                out.append(w.lower())
        name = "(H) " + " ".join(out)
    else:
        name = re.sub(r"_+", " ", raw).strip()
    if len(name) > max_len:
        cut = name[: max_len - 1]
        if " " in cut[max_len // 2:]:
            cut = cut[: cut.rfind(" ")]
        name = cut.rstrip() + "…"
    return name


def _per_fold_samples(csv: Path, df: pd.DataFrame) -> pd.DataFrame | None:
    """Per-fold attention values if stored (wide columns attention_fold<k>, or a long-format sidecar)."""
    fold_cols = [c for c in df.columns if re.fullmatch(r"attention_fold\d+", c)]
    if fold_cols:
        long = df.melt(id_vars="pathway", value_vars=fold_cols, var_name="fold", value_name="attention")
        long["fold"] = long["fold"].str.extract(r"(\d+)").astype(int)
        return long[["pathway", "fold", "attention"]]
    sidecar = csv.with_name("pathway_importance_folds.csv")
    if sidecar.exists():
        long = pd.read_csv(sidecar)
        if {"pathway", "fold", "attention"} <= set(long.columns):
            return long[["pathway", "fold", "attention"]]
    return None


def load_pathway_table(cohort: str, run: str = PATHWAY_RUN) -> dict:
    seeds_found, means, sds, samples = [], [], [], []
    seeds_missing = []
    for seed, suf in SEED_SUFFIX.items():
        csv = RESULTS / f"{run}{suf}" / cohort / "analysis" / "pathway_importance.csv"
        if not csv.exists():
            seeds_missing.append(seed)
            continue
        df = pd.read_csv(csv)
        seeds_found.append(seed)
        means.append(df.set_index("pathway")["attention_mean"].rename(seed))
        sds.append(df.set_index("pathway")["attention_std"].rename(seed))
        s = _per_fold_samples(csv, df)
        if s is not None:
            samples.append(s.assign(seed=seed))
        else:
            samples.append(None)
    if not seeds_found:
        raise FileNotFoundError(f"no pathway_importance.csv for {run}/{cohort} in any seed")
    mean_tab = pd.concat(means, axis=1)
    sd_tab = pd.concat(sds, axis=1)
    n_pathways = int(len(mean_tab))
    point = mean_tab.mean(axis=1) * n_pathways
    if all(s is not None for s in samples):
        long = pd.concat(samples, ignore_index=True)
        long["x_uniform"] = long["attention"] * n_pathways
        q = long.groupby("pathway")["x_uniform"].quantile([0.25, 0.75]).unstack()
        lo = q[0.25].reindex(point.index).clip(lower=0.0)
        hi = q[0.75].reindex(point.index).clip(lower=0.0)
        kind = (f"25-75% quantiles over {long.groupby('seed')['fold'].nunique().sum()} (fold, seed) values, "
                f"clipped at 0; seeds {seeds_found}")
        n_samples = int(long.groupby("pathway").size().iloc[0])
    else:
        # only mean and std over folds are stored: mean +- std with the lower end clipped at 0
        sd = np.sqrt((sd_tab ** 2).mean(axis=1)) * n_pathways
        lo = (point - sd).clip(lower=0.0)
        hi = point + sd
        kind = (f"mean +- std over the 5 folds (seeds {seeds_found}; per-fold values are not stored, "
                "so 25-75% quantiles cannot be computed), lower end clipped at 0")
        n_samples = 5 * len(seeds_found)
    tab = pd.DataFrame({"pathway": point.index, "point": point.to_numpy(),
                        "lo": lo.to_numpy(), "hi": hi.to_numpy()})
    tab["err_lo"] = (tab["point"] - tab["lo"]).clip(lower=0.0)
    tab["err_hi"] = (tab["hi"] - tab["point"]).clip(lower=0.0)
    tab = tab.sort_values("point", ascending=False).reset_index(drop=True)
    return {"table": tab, "n_pathways": n_pathways, "seeds_found": seeds_found, "seeds_missing": seeds_missing,
            "error_bar": kind, "n_samples_per_pathway": n_samples}


def figure_pathways(out_pdf: Path, preview_dir: Path | None, top_k: int) -> tuple[list[str], dict]:
    data = {c: load_pathway_table(c) for c in COHORTS}
    n_pathways = {d["n_pathways"] for d in data.values()}
    n_pw = n_pathways.pop() if len(n_pathways) == 1 else None
    fig = plt.figure(figsize=(5.5, 0.25 + 1.42 * len(COHORTS)))
    gs = fig.add_gridspec(len(COHORTS), 1, hspace=0.58, left=0.47, right=0.985, top=0.95, bottom=0.06)
    axes = []
    numbers: dict = {}
    for i, cohort in enumerate(COHORTS):
        d = data[cohort]
        tab = d["table"].head(top_k).iloc[::-1]
        ax = fig.add_subplot(gs[i])
        axes.append(ax)
        y = np.arange(len(tab))
        colors = [C_HALLMARK if p.startswith("HALLMARK_") else C_REACTOME for p in tab["pathway"]]
        ax.barh(y, tab["point"], height=0.62, color=colors, lw=0, zorder=2)
        ax.errorbar(tab["point"], y, xerr=[tab["err_lo"], tab["err_hi"]], fmt="none", ecolor=INK2,
                    elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
        ax.axvline(1.0, color=MUTED, ls=(0, (3, 2)), lw=0.6, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([pretty_name(p) for p in tab["pathway"]])
        ax.set_ylim(-0.6, len(tab) - 0.4)
        ax.set_xlim(0, float((tab["point"] + tab["err_hi"]).max()) * 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_title(COHORT_LABEL[cohort], loc="left", fontsize=9, pad=3)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", length=2, pad=1.5)
        numbers[COHORT_LABEL[cohort]] = {
            "seeds_found": d["seeds_found"], "seeds_missing": d["seeds_missing"], "n_pathways": d["n_pathways"],
            "error_bar": d["error_bar"],
            "top": [{"pathway": r.pathway, "label": pretty_name(r.pathway), "x_uniform": round(float(r.point), 3),
                     "bar_lo": round(float(r.lo), 3), "bar_hi": round(float(r.hi), 3)}
                    for r in d["table"].head(top_k).itertuples()],
        }
    unif = f"1/{n_pw}" if n_pw else "1/#pathways"
    fig.supxlabel(f"Mean genomic cross-attention per pathway token (× uniform = {unif})", fontsize=8.5,
                  color=INK, y=0.012)
    handles = [Patch(facecolor=C_REACTOME, lw=0), Patch(facecolor=C_HALLMARK, lw=0)]
    fig.legend(handles, ["Reactome pathway", "MSigDB Hallmark gene set (H)"], loc="upper right",
               bbox_to_anchor=(0.985, 0.998), ncol=2, handlelength=1.2, columnspacing=1.4, handletextpad=0.5,
               borderaxespad=0.0)
    # fit the left margin to the widest rendered pathway label so nothing is clipped
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widest = max(lab.get_window_extent(renderer).width for ax in axes for lab in ax.get_yticklabels())
    left = (widest / fig.dpi + 0.1) / fig.get_figwidth()
    gs.update(left=min(max(left, 0.3), 0.6))
    return save(fig, out_pdf, preview_dir), numbers


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True, help="directory for the PDF figures")
    ap.add_argument("--preview_dir", default=None, help="directory for PNG previews and the numbers JSON")
    ap.add_argument("--top_k", type=int, default=8, help="pathways per cohort panel")
    ap.add_argument("--only", choices=["km", "pathways"], default=None)
    ap.add_argument("--force", action="store_true", help="overwrite this script's own earlier outputs")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    preview = Path(args.preview_dir) if args.preview_dir else None
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
    set_style()

    written: list[str] = []
    numbers: dict = {}
    if args.only in (None, "km"):
        pdf = out / "km_grid.pdf"
        check_out(pdf, args.force)
        w, numbers["km_grid"] = figure_km(pdf, preview)
        written += w
    if args.only in (None, "pathways"):
        pdf = out / "pathway_importance_v2.pdf"
        check_out(pdf, args.force)
        w, numbers["pathway_importance_v2"] = figure_pathways(pdf, preview, args.top_k)
        written += w

    if preview is not None:
        js = preview / "paper_figures_km_numbers.json"
        with open(js, "w") as f:
            json.dump(numbers, f, indent=1)
        written.append(str(js))
    print(json.dumps(numbers, indent=1))
    print("written:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
