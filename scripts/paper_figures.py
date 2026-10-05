"""Vector figures for the TMLR revision, read straight from the stored fold results.

    python scripts/paper_figures.py --root pod_results/outputs_e20 --out <figure dir>

Produces, in --out (existing files are never overwritten unless --overwrite is given):

  audit_deltas.pdf          change in C-index of the SAME checkpoint relative to its complete-input
                            evaluation under four test-time conditions (RNA removed, RNA permuted
                            across patients, slide removed, slide permuted), for SurvPath (mean
                            imputation), PathQ-Former w/o aux heads and PathQ-Former (null codes);
                            (a) pooled over cohorts with 95 % bootstrap CIs, (b) per cohort.
  missing_rate_seeds.pdf    per cohort, C-index when 0/10/20/30/50 % of validation patients lack one
                            modality at random, PathQ-Former and PathQ-Former w/o aux heads, mean over
                            three seeds with a +-sd band; SurvPath's complete-input score as reference.
  epoch_curves_pooled.pdf   validation C-index per epoch pooled over cohorts and folds, mean +-sd
                            over seeds, for PathQ-Former, PathQ-Former w/o aux heads and SurvPath.
  epoch_curves_by_cohort.pdf  the same per cohort (2 x 3 panels).

Condition mapping (test-time, same checkpoint):
  RNA removed     -> `wsi_only`       (PathQ-Former, null code)  / `wsi_only_impute`     (SurvPath, training mean)
  RNA permuted    -> `rna_permuted`   (gene vectors permuted across validation patients)
  Slide removed   -> `genomic_only`   (PathQ-Former, null code)  / `genomic_only_impute` (SurvPath, training mean)
  Slide permuted  -> `wsi_permuted`   (patch embeddings permuted across validation patients)
PathQ-Former w/o aux heads (pathq_fast_e20) has no permuted conditions; those cells are marked n/a.

Seed convention: runs are <root>/<run>[_seed<N>]; fold k is the same validation split in every run.
Audit deltas: per (cohort, fold) the delta is averaged over the available seeds, then summarised over the
25 (cohort, fold) pairs; the CI resamples those 25 folds (numpy default_rng(0), 10 000 draws, percentile).
Missing-rate and epoch curves: per seed the pooled (or per-cohort) fold mean, then mean +-sd over seeds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ----------------------------------------------------------------------------- data conventions
COHORTS = ["blca", "brca", "coadread", "hnsc", "stad"]
COHORT_LABEL = {c: c.upper() for c in COHORTS}
RATES = [0.0, 0.1, 0.2, 0.3, 0.5]  # 0.0 is the complete-input (`both`) evaluation
EPOCHS = 20

# ordered: the order of bars/legend entries in every figure
MODELS = {
    "pathq": dict(run="pathq_fast_e20_aux", label="PathQ-Former"),
    "pathq_noaux": dict(run="pathq_fast_e20", label="PathQ-Former w/o aux heads"),
    "survpath": dict(run="survpath_e20", label="SurvPath"),
}

CONDITIONS = [
    ("RNA removed", dict(pathq="wsi_only", pathq_noaux="wsi_only", survpath="wsi_only_impute")),
    ("RNA permuted", dict(pathq="rna_permuted", pathq_noaux=None, survpath="rna_permuted")),
    ("Slide removed", dict(pathq="genomic_only", pathq_noaux="genomic_only", survpath="genomic_only_impute")),
    ("Slide permuted", dict(pathq="wsi_permuted", pathq_noaux=None, survpath="wsi_permuted")),
]

# ----------------------------------------------------------------------------- style (one system for all figures)
# Categorical slots 1-3 of the validated reference palette (colour-blind safe as a set; aqua's
# light-surface contrast is relieved by the legend plus a distinct marker / line style per model).
COLOR = {"pathq": "#2a78d6", "pathq_noaux": "#eb6834", "survpath": "#1baf7a"}
MARKER = {"pathq": "o", "pathq_noaux": "s", "survpath": "^"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "text.color": INK,
            "axes.labelcolor": INK,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


class Saver:
    """Writes <out>/<name>.pdf and, when a preview dir is given, <preview>/<name>.png at 150 dpi."""

    def __init__(self, out: Path, preview: Path | None):
        self.out, self.preview, self.written = out, preview, []

    def __call__(self, fig, name: str) -> None:
        fig.savefig(self.out / f"{name}.pdf")
        self.written.append(self.out / f"{name}.pdf")
        if self.preview is not None:
            self.preview.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.preview / f"{name}.png", dpi=150)
        plt.close(fig)


def panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=8.5, pad=3, color=INK)


def legend_handles(keys, kind="line") -> list[Line2D]:
    hs = []
    for k in keys:
        if kind == "line":
            hs.append(Line2D([], [], color=COLOR[k], marker=MARKER[k], lw=1.1, label=MODELS[k]["label"]))
        else:
            hs.append(Line2D([], [], color=COLOR[k], marker=MARKER[k], lw=0, label=MODELS[k]["label"]))
    return hs


# ----------------------------------------------------------------------------- loading
def discover_runs(root: Path, run: str) -> dict[int, Path]:
    """{seed index: run dir} for <run> and <run>_seed<N> (exact prefix match only)."""
    pat = re.compile(rf"^{re.escape(run)}(?:_seed(\d+))?$")
    out = {}
    for d in sorted(root.iterdir()):
        m = pat.match(d.name)
        if m and d.is_dir():
            out[int(m.group(1) or 0)] = d
    return out


def load(root: Path) -> dict:
    """data[model][seed][cohort][fold] = {'metrics': {cond: c}, 'partial': {rate: c}, 'history': [c per epoch]}"""
    data: dict = {}
    for key, spec in MODELS.items():
        runs = discover_runs(root, spec["run"])
        if not runs:
            sys.exit(f"no run directories for {spec['run']} under {root}")
        data[key] = {}
        for seed, rd in runs.items():
            data[key][seed] = {}
            for cohort in COHORTS:
                folds = {}
                for fr in sorted((rd / cohort).glob("fold_*/fold_results.json")):
                    k = int(fr.parent.name.split("_")[1])
                    with open(fr) as f:
                        d = json.load(f)
                    metrics = {
                        c: m["c_index"]
                        for c, m in (d.get("metrics") or {}).items()
                        if isinstance(m, dict) and m.get("c_index") is not None
                    }
                    partial = {float(p["rate"]): float(p["c_index_mean"]) for p in (d.get("partial_missing") or [])}
                    if "both" in metrics:
                        partial[0.0] = metrics["both"]
                    history = [h["val_cindex"] for h in (d.get("history") or [])]
                    folds[k] = dict(metrics=metrics, partial=partial, history=history)
                data[key][seed][cohort] = folds
        n = {c: sorted({len(f) for f in (data[key][s][c] for s in runs)}) for c in COHORTS}
        print(f"loaded {spec['run']:20s} seeds={sorted(runs)} folds per cohort={n}")
    return data


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def seed_averaged_deltas(data: dict, model: str, cond: str) -> dict[tuple[str, int], float]:
    """{(cohort, fold): mean over seeds of c[cond] - c['both'] (same run, same fold)}"""
    per_fold = defaultdict(list)
    for seed, by_cohort in data[model].items():
        for cohort, folds in by_cohort.items():
            for k, f in folds.items():
                m = f["metrics"]
                if cond in m and "both" in m:
                    per_fold[(cohort, k)].append(m[cond] - m["both"])
    return {k: float(np.mean(v)) for k, v in per_fold.items()}


def seed_level(data: dict, model: str, getter, cohorts=COHORTS) -> np.ndarray:
    """One number per seed: the mean over the folds of `cohorts` of getter(fold_record) (None -> skipped)."""
    out = []
    for seed in sorted(data[model]):
        vals = [
            v
            for cohort in cohorts
            for f in data[model][seed].get(cohort, {}).values()
            if (v := getter(f)) is not None
        ]
        if vals:
            out.append(float(np.mean(vals)))
    return np.asarray(out)


def seed_level_history(data: dict, model: str, cohorts=COHORTS) -> np.ndarray:
    """(n_seeds, EPOCHS): per seed, the mean over the folds of `cohorts` of val C-index at every epoch."""
    rows = []
    for seed in sorted(data[model]):
        hs = [f["history"][:EPOCHS] for c in cohorts for f in data[model][seed].get(c, {}).values() if f["history"]]
        if hs:
            rows.append(np.mean(np.asarray(hs, dtype=float), axis=0))
    return np.asarray(rows)


# ----------------------------------------------------------------------------- figure 1: audit deltas
def fig_audit_deltas(data: dict, save: Saver, numbers: dict) -> None:
    keys = list(MODELS)
    offsets = {k: o for k, o in zip(keys, (-0.24, 0.0, 0.24))}
    pooled, per_cohort = {}, {}
    for cname, cmap in CONDITIONS:
        for k in keys:
            cond = cmap[k]
            if cond is None:
                continue
            d = seed_averaged_deltas(data, k, cond)
            if not d:
                continue
            vals = np.asarray([d[key] for key in sorted(d)])
            lo, hi = bootstrap_ci(vals)
            pooled[(cname, k)] = dict(mean=float(vals.mean()), ci=(lo, hi), n=len(vals), condition_key=cond)
            per_cohort[(cname, k)] = {
                c: float(np.mean([v for (cc, _), v in d.items() if cc == c])) for c in COHORTS if any(cc == c for cc, _ in d)
            }

    fig = plt.figure(figsize=(5.5, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1.0], hspace=0.5, wspace=0.2)
    ax = fig.add_subplot(gs[0, :])
    xs = np.arange(len(CONDITIONS))
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    for i, (cname, cmap) in enumerate(CONDITIONS):
        for k in keys:
            x = xs[i] + offsets[k]
            p = pooled.get((cname, k))
            if p is None:
                ax.text(x, 0.006, "n/a", ha="center", va="bottom", fontsize=6.5, color=MUTED)
                continue
            lo, hi = p["ci"]
            ax.errorbar(
                [x], [p["mean"]], yerr=[[p["mean"] - lo], [hi - p["mean"]]],
                fmt=MARKER[k], color=COLOR[k], ms=4.2, mec=SURFACE, mew=0.6, elinewidth=1.0, capsize=2.2, capthick=0.8, zorder=3,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([c for c, _ in CONDITIONS])
    ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
    ax.set_ylabel(r"$\Delta$ C-index vs. complete input")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", length=0)
    for x in xs[:-1]:
        ax.axvline(x + 0.5, color=GRID, lw=0.5, zorder=0)
    panel_label(ax, "(a) Pooled over cohorts (25 folds; 95% bootstrap CI)")
    ax.legend(handles=legend_handles(keys, "point"), ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.12), borderaxespad=0)

    axes_b = []
    cx = np.arange(len(COHORTS))
    for i, (cname, cmap) in enumerate(CONDITIONS):
        axb = fig.add_subplot(gs[1, i], sharey=axes_b[0] if axes_b else None)
        axes_b.append(axb)
        axb.axhline(0, color=AXIS, lw=0.6, zorder=1)
        for k in keys:
            vals = per_cohort.get((cname, k))
            if vals is None:
                continue
            axb.scatter(
                cx + offsets[k] * 0.9, [vals[c] for c in COHORTS], s=15, marker=MARKER[k], color=COLOR[k],
                edgecolors=SURFACE, linewidths=0.5, zorder=3,
            )
        if any(cmap[k] is None for k in keys):
            axb.text(0.5, 0.03, "w/o aux heads: n/a", transform=axb.transAxes, ha="center", va="bottom", fontsize=6.5, color=MUTED)
        axb.set_xticks(cx)
        axb.set_xticklabels([COHORT_LABEL[c] for c in COHORTS], rotation=55, ha="right", rotation_mode="anchor", fontsize=7)
        axb.tick_params(axis="x", length=0, pad=1.5)
        axb.set_xlim(-0.6, len(COHORTS) - 0.4)
        axb.set_title(f"({'bcde'[i]}) {cname}", loc="left", fontsize=7.5, pad=3, color=INK)
        axb.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        axb.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        if i:
            plt.setp(axb.get_yticklabels(), visible=False)
            axb.tick_params(axis="y", length=0)
        else:
            axb.set_ylabel(r"$\Delta$ C-index, per cohort")
    axes_b[0].set_ylim(*_padded_limits([list(v.values()) for v in per_cohort.values()], step=0.05, pad=0.012))
    save(fig, "audit_deltas")

    numbers["audit_deltas"] = {
        "pooled": {f"{c} | {MODELS[k]['label']}": v for (c, k), v in pooled.items()},
        "per_cohort": {f"{c} | {MODELS[k]['label']}": {COHORT_LABEL[cc]: vv for cc, vv in v.items()} for (c, k), v in per_cohort.items()},
    }
    print("\n== audit_deltas: delta C-index vs complete input (seed-averaged per fold; mean [95% bootstrap CI] over 25 folds)")
    for cname, _ in CONDITIONS:
        for k in keys:
            p = pooled.get((cname, k))
            if p is None:
                print(f"  {cname:15s} {MODELS[k]['label']:28s}  n/a")
                continue
            pc = "  ".join(f"{COHORT_LABEL[c]}={per_cohort[(cname, k)][c]:+.3f}" for c in COHORTS)
            print(f"  {cname:15s} {MODELS[k]['label']:28s} {p['mean']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  n={p['n']}  ({p['condition_key']})   {pc}")


# ----------------------------------------------------------------------------- figure 2: missing rate with seeds
def fig_missing_rate(data: dict, save: Saver, numbers: dict) -> None:
    keys = ["pathq", "pathq_noaux"]
    stats = {}
    for cohort in COHORTS:
        for k in keys:
            curve = np.array([seed_level(data, k, lambda f, r=r: f["partial"].get(r), [cohort]) for r in RATES])  # (rates, seeds)
            stats[(cohort, k)] = dict(mean=curve.mean(axis=1), sd=curve.std(axis=1, ddof=1) if curve.shape[1] > 1 else np.zeros(len(RATES)), n=curve.shape[1])
        ref = seed_level(data, "survpath", lambda f: f["metrics"].get("both"), [cohort])
        stats[(cohort, "survpath")] = dict(mean=float(ref.mean()), sd=float(ref.std(ddof=1)) if len(ref) > 1 else 0.0, n=len(ref))

    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.12)
    x = [int(r * 100) for r in RATES]
    for ax, cohort in zip(axes.flat, COHORTS):
        s = stats[(cohort, "survpath")]
        ax.axhline(s["mean"], color=COLOR["survpath"], lw=1.0, ls=(0, (4, 2)), zorder=2)
        for k in keys:
            m, sd = stats[(cohort, k)]["mean"], stats[(cohort, k)]["sd"]
            ax.fill_between(x, m - sd, m + sd, color=COLOR[k], alpha=0.16, lw=0, zorder=2)
            ax.plot(x, m, color=COLOR[k], marker=MARKER[k], mec=SURFACE, mew=0.5, zorder=3)
        panel_label(ax, COHORT_LABEL[cohort])
        ax.set_xticks(x)
        ax.tick_params(axis="x", pad=2)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    for ax in axes[1]:
        ax.set_xlabel("Patients missing\none modality (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("C-index (DSS)")
    axes[1, 2].axis("off")
    handles = legend_handles(keys, "line") + [
        Line2D([], [], color=COLOR["survpath"], lw=1.0, ls=(0, (4, 2)), label="SurvPath, complete input"),
    ]
    axes[1, 2].legend(handles=handles, loc="center left", bbox_to_anchor=(-0.05, 0.5), labelspacing=0.9)
    axes[0, 0].set_ylim(*_padded_limits([stats[(c, k)]["mean"] + s * stats[(c, k)]["sd"] for c in COHORTS for k in keys for s in (-1, 1)] + [stats[(c, "survpath")]["mean"] for c in COHORTS], step=0.05))
    save(fig, "missing_rate_seeds")

    numbers["missing_rate_seeds"] = {
        COHORT_LABEL[c]: {
            **{MODELS[k]["label"]: {f"{int(r * 100)}%": f"{stats[(c, k)]['mean'][i]:.3f} +- {stats[(c, k)]['sd'][i]:.3f}" for i, r in enumerate(RATES)} for k in keys},
            "SurvPath complete input": f"{stats[(c, 'survpath')]['mean']:.3f} +- {stats[(c, 'survpath')]['sd']:.3f}",
        }
        for c in COHORTS
    }
    print("\n== missing_rate_seeds: C-index, mean +- sd over seeds of the 5-fold mean (rate 0 = complete input)")
    print("  cohort     model                        " + "  ".join(f"{int(r*100):>3d}%          " for r in RATES))
    for c in COHORTS:
        for k in keys:
            s = stats[(c, k)]
            print(f"  {COHORT_LABEL[c]:10s} {MODELS[k]['label']:28s} " + "  ".join(f"{m:.3f} +- {sd:.3f}" for m, sd in zip(s["mean"], s["sd"])) + f"   (n={s['n']} seeds)")
        s = stats[(c, "survpath")]
        print(f"  {COHORT_LABEL[c]:10s} {'SurvPath complete input':28s} {s['mean']:.3f} +- {s['sd']:.3f}   (n={s['n']} seeds)")


def _padded_limits(values, step: float, pad: float = 0.01) -> tuple[float, float]:
    """Axis limits rounded outwards to a multiple of `step` around every scalar/array in `values`."""
    flat = np.concatenate([np.ravel(v) for v in values])
    lo, hi = flat.min() - pad, flat.max() + pad
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


# ----------------------------------------------------------------------------- figure 3: epoch curves
def _draw_curves(ax, data: dict, cohorts, numbers_key: str | None, numbers: dict) -> None:
    ep = np.arange(1, EPOCHS + 1)
    for k in MODELS:
        h = seed_level_history(data, k, cohorts)
        m, sd = h.mean(axis=0), (h.std(axis=0, ddof=1) if h.shape[0] > 1 else np.zeros(EPOCHS))
        ax.fill_between(ep, m - sd, m + sd, color=COLOR[k], alpha=0.16, lw=0, zorder=2)
        ax.plot(ep, m, color=COLOR[k], marker=MARKER[k], ms=2.6, mec=SURFACE, mew=0.4, zorder=3, label=MODELS[k]["label"])
        if numbers_key is not None:
            numbers.setdefault(numbers_key, {})[MODELS[k]["label"]] = dict(
                n_seeds=int(h.shape[0]), mean=[round(float(v), 4) for v in m], sd=[round(float(v), 4) for v in sd]
            )
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, EPOCHS + 0.5)


def fig_epoch_curves(data: dict, save: Saver, numbers: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    _draw_curves(ax, data, COHORTS, "epoch_curves_pooled", numbers)
    ax.set_xlabel("Training epoch (fixed budget, final checkpoint)")
    ax.set_ylabel("Validation C-index, pooled")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0)
    save(fig, "epoch_curves_pooled")

    fig, axes = plt.subplots(2, 3, figsize=(5.3, 3.5), sharex=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.34)
    for ax, cohort in zip(axes.flat, COHORTS):
        _draw_curves(ax, data, [cohort], f"epoch_curves_{COHORT_LABEL[cohort]}", numbers)
        panel_label(ax, COHORT_LABEL[cohort])
        # same y-span in every panel (offsets differ) so slopes and band widths are comparable across cohorts
        lo, hi = ax.get_ylim()
        mid, span = (lo + hi) / 2, 0.26
        ax.set_ylim(mid - span / 2, mid + span / 2)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    for ax in axes[1]:
        ax.set_xlabel("Training epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation C-index")
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles=legend_handles(list(MODELS), "line"), loc="center left", bbox_to_anchor=(-0.05, 0.5), labelspacing=0.9)
    save(fig, "epoch_curves_by_cohort")

    print("\n== epoch_curves_pooled: validation C-index per epoch, pooled over cohorts and folds; mean +- sd over seeds")
    for label, v in numbers["epoch_curves_pooled"].items():
        print(f"  {label:28s} n_seeds={v['n_seeds']}")
        print("    mean " + " ".join(f"{x:.3f}" for x in v["mean"]))
        print("    sd   " + " ".join(f"{x:.3f}" for x in v["sd"]))
    print("\n== epoch_curves_by_cohort: final-epoch (20) value, mean +- sd over seeds")
    for c in COHORTS:
        row = "  ".join(f"{label}={v['mean'][-1]:.3f}+-{v['sd'][-1]:.3f}" for label, v in numbers[f"epoch_curves_{COHORT_LABEL[c]}"].items())
        print(f"  {COHORT_LABEL[c]:10s} {row}")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="pod_results/outputs_e20")
    ap.add_argument("--out", required=True, help="directory for the PDF figures")
    ap.add_argument("--preview-dir", default=None, help="also save 150-dpi PNG previews here")
    ap.add_argument("--dump-numbers", default=None, help="write the plotted numbers to this JSON file")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing files in --out")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = ["audit_deltas.pdf", "missing_rate_seeds.pdf", "epoch_curves_pooled.pdf", "epoch_curves_by_cohort.pdf"]
    clashes = [n for n in names if (out / n).exists()]
    if clashes and not args.overwrite:
        sys.exit(f"refusing to overwrite existing files in {out}: {clashes} (pass --overwrite)")

    set_style()
    data = load(root)
    numbers: dict = {}
    save = Saver(out, Path(args.preview_dir) if args.preview_dir else None)
    fig_audit_deltas(data, save, numbers)
    fig_missing_rate(data, save, numbers)
    fig_epoch_curves(data, save, numbers)
    print("\nwritten:", *[str(p) for p in save.written], sep="\n  ")

    if args.dump_numbers:
        Path(args.dump_numbers).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_numbers, "w") as f:
            json.dump(numbers, f, indent=1)
        print(f"numbers -> {args.dump_numbers}")


if __name__ == "__main__":
    main()
