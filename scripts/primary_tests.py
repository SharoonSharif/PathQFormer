"""Primary statistics for the 20-epoch campaign: seed-averaged paired comparisons with bootstrap CIs and
Holm-corrected per-cohort tests, plus the secondary metrics (Uno's IPCW C-index, IBS). No training; reads the
archived results.json files only.

    python scripts/primary_tests.py                                   # -> results/final/table_primary_tests.txt
                                                                      #    results/final/table_secondary_metrics.txt
    python scripts/primary_tests.py --e20 pod_results/outputs_e20 --os pod_results/outputs_os --out-dir results/final
    python scripts/primary_tests.py --comparisons pathq_fast_e20_aux:survpath_e20 --n-boot 2000

Unit of analysis. In the 5-fold CV the validation fold IS the reported held-out fold (per fold c_index ==
last_epoch_val_cindex, selection 'last'), and the seeds of one fold share the validation patients, so the 75
(fold, seed) pairs of Table 1 (scripts/seed_table.py) are not independent units. Here every (cohort, fold) is first
averaged over the seeds available for BOTH methods of a comparison (3 for the trained e20 methods, 2 against late
fusion, 1 for the OS endpoint); the paired analysis then runs over the 25 (cohort, fold) units pooled and over the
5 folds per cohort. The pooled delta gets a 95 % percentile bootstrap CI (resampling folds, 10,000 draws, numpy
seed 0) and the per-cohort p values are Holm-adjusted over the 5 cohorts. Everything is deterministic.

Run directories are <root>/<method>[_seed<N>]/<cohort>/results.json; seed 0 has no suffix. Multimodal runs store
their metrics under metrics['both']; the single-modality baselines store their only condition (ABMIL: wsi_only,
SNN / MLP: genomic_only), which is used in its place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

try:  # Windows consoles default to cp1252; never let a symbol in a table crash the run
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

COHORTS = ["blca", "brca", "coadread", "hnsc", "stad"]
METRICS = ("c_index", "c_index_ipcw", "ibs")
DEFAULT_COMPARISONS = [
    "pathq_fast_e20_aux:survpath_e20",
    "pathq_fast_e20:survpath_e20",
    "pathq_fast_e20_aux:mlp_omics_e20",
    "pathq_fast_e20:mlp_omics_e20",
    "pathq_fast_e20_aux:late_fusion_e20",
    "pathq_fast_e20:late_fusion_e20",
    "abmil_e20:survpath_e20",
    "snn_e20:survpath_e20",
    "mlp_omics_e20:survpath_e20",
]
DEFAULT_OS_COMPARISONS = ["pathq_fast_e20_aux:survpath_e20"]
DEFAULT_E10_ROOTS = ["pod_results/outputs_v2", "pod_results/outputs_ablate"]   # seed 0 and seeds 1-2 of the 10-epoch runs
DEFAULT_E10_COMPARISONS = ["pathq_fast_e10:survpath_e10"]
DEFAULT_SECONDARY = ["pathq_fast_e20_aux", "pathq_fast_e20", "abmil_e20", "mlp_omics_e20", "survpath_e20"]

Fold = dict[str, float]                                   # metric -> value
Runs = dict[str, dict[str, dict[int, dict[int, Fold]]]]   # method -> cohort -> seed -> fold -> metrics


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------
def primary_condition(metrics: dict) -> tuple[str, dict] | None:
    """`both` for multimodal runs; a single-modality baseline stores its only condition."""
    if isinstance(metrics.get("both"), dict):
        return "both", metrics["both"]
    conds = [k for k, v in metrics.items() if isinstance(v, dict) and v.get("c_index") is not None]
    if len(conds) == 1:
        return conds[0], metrics[conds[0]]
    return None


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


def collect(root: Path, methods: set[str]) -> tuple[Runs, dict[str, str], list[str]]:
    """-> ({method: {cohort: {seed: {fold: {metric: value}}}}}, {method: condition used}, consistency notes)."""
    data: Runs = defaultdict(lambda: defaultdict(dict))
    conditions: dict[str, str] = {}
    notes: list[str] = []
    pat = re.compile(r"^(?P<m>.+?)(?:_seed(?P<s>\d+))?$")
    for rj in sorted(root.glob("*/*/results.json")):
        run, cohort = rj.parts[-3], rj.parts[-2]
        m = pat.match(run)
        method, seed = m.group("m"), int(m.group("s") or 0)   # the run's seed; per-fold seeds are seed + fold
        if method not in methods:
            continue
        with open(rj) as f:
            res = json.load(f)
        folds: dict[int, Fold] = {}
        for d in res["folds"]:
            pc = primary_condition(d.get("metrics", {}))
            if pc is None:
                continue
            cond, met = pc
            conditions.setdefault(method, cond)
            last = d.get("last_epoch_val_cindex")
            if _finite(last) and abs(last - d["c_index"]) > 1e-9:   # trained runs only; late fusion stores NaN
                notes.append(f"{run}/{cohort} fold {d['fold']}: c_index {d['c_index']:.4f} != last_epoch_val_cindex {last:.4f}")
            if abs(met["c_index"] - d["c_index"]) > 1e-9:
                notes.append(f"{run}/{cohort} fold {d['fold']}: metrics[{cond}].c_index != c_index")
            folds[int(d["fold"])] = {k: float(met.get(k, float("nan"))) for k in METRICS}
        if folds:
            data[method][cohort][seed] = folds
    return data, conditions, notes


def collect_many(roots: list[Path], methods: set[str]) -> tuple[Runs, dict[str, str], list[str]]:
    """collect() over several roots whose runs complement each other (e.g. seed 0 in one campaign root, seeds 1-2 in another)."""
    data: Runs = defaultdict(lambda: defaultdict(dict))
    conditions: dict[str, str] = {}
    notes: list[str] = []
    for root in roots:
        d, c, n = collect(root, methods)
        for method, cohorts in d.items():
            for cohort, seeds in cohorts.items():
                data[method][cohort].update(seeds)
        for k, v in c.items():
            conditions.setdefault(k, v)
        notes += n
    return data, conditions, notes


# ---------------------------------------------------------------------------------------
# Seed averaging and paired statistics
# ---------------------------------------------------------------------------------------
def seed_average(seed_folds: dict[int, dict[int, Fold]], metric: str, seeds: list[int]) -> dict[int, float]:
    """One value per fold: mean over the given seeds (non-finite values dropped; all-NaN folds stay NaN)."""
    by_fold: dict[int, list[float]] = defaultdict(list)
    for s in seeds:
        for k, v in seed_folds.get(s, {}).items():
            by_fold[k].append(v[metric])
    out = {}
    for k, vals in by_fold.items():
        arr = np.asarray(vals, float)
        arr = arr[np.isfinite(arr)]
        out[k] = float(arr.mean()) if len(arr) else float("nan")
    return out


def common_seeds(data: Runs, a: str, b: str, cohort: str) -> list[int]:
    return sorted(set(data[a].get(cohort, {})) & set(data[b].get(cohort, {})))


def paired_units(data: Runs, a: str, b: str, cohort: str, metric: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Seed-averaged values of a and b on the folds where both are finite, plus the seeds averaged over."""
    seeds = common_seeds(data, a, b, cohort)
    fa = seed_average(data[a].get(cohort, {}), metric, seeds)
    fb = seed_average(data[b].get(cohort, {}), metric, seeds)
    keys = [k for k in sorted(set(fa) & set(fb)) if np.isfinite(fa[k]) and np.isfinite(fb[k])]
    return np.array([fa[k] for k in keys]), np.array([fb[k] for k in keys]), seeds


def paired_stats(x: np.ndarray, y: np.ndarray, n_boot: int, boot_seed: int) -> dict:
    """delta = mean(x - y); percentile bootstrap CI over the paired units; paired t and Wilcoxon p; wins."""
    d = x - y
    n = int(len(d))
    out = {"n": n, "delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
           "t_p": float("nan"), "w_p": float("nan"), "wins": int(np.sum(d > 0))}
    if n == 0:
        return out
    out["delta"] = float(d.mean())
    if n < 2:
        return out
    rng = np.random.default_rng(boot_seed)                # re-seeded per call: results do not depend on call order
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    out["lo"], out["hi"] = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    out["t_p"] = float(stats.ttest_rel(x, y).pvalue)
    try:
        out["w_p"] = float(stats.wilcoxon(x, y).pvalue)   # two-sided; exact for n <= 50 without ties
    except ValueError:                                    # all differences zero
        pass
    return out


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment; NaN entries are left out of the family and stay NaN."""
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return adj
    order = np.argsort(p[ok])
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[ok][order])
    vals = np.empty(m)
    vals[order] = np.minimum(stepped, 1.0)
    adj[ok] = vals
    return adj


def fmt_p(p: float, nd: int = 3) -> str:
    """Fixed decimals, switching to scientific notation instead of printing a p value as 0.000."""
    if not np.isfinite(p):
        return "nan".rjust(nd + 2)
    return f"{p:.1e}" if 0 < p < 10 ** -nd else f"{p:.{nd}f}"


def seeds_available(data: Runs) -> str:
    return "; ".join(f"{m} {','.join(str(s) for s in sorted({s for coh in data[m].values() for s in coh}))}" for m in sorted(data))


def undefined_folds(data: Runs, methods: list[str], metric: str) -> dict[tuple[str, int], set[str]]:
    """(cohort, fold) -> methods with a non-finite value of `metric` in any seed."""
    out: dict[tuple[str, int], set[str]] = defaultdict(set)
    for m in methods:
        for c, seeds in data.get(m, {}).items():
            for folds in seeds.values():
                for k, v in folds.items():
                    if not np.isfinite(v[metric]):
                        out[(c, k)].add(m)
    return out


def seeds_label(per_cohort: dict[str, list[int]]) -> str:
    sets = {tuple(s) for s in per_cohort.values()}
    if len(sets) == 1:
        return ",".join(str(s) for s in next(iter(sets)))
    return "; ".join(f"{c}:{','.join(str(s) for s in v)}" for c, v in per_cohort.items())


# ---------------------------------------------------------------------------------------
# Primary table
# ---------------------------------------------------------------------------------------
def comparison_block(data: Runs, a: str, b: str, label: str, metric: str, n_boot: int, boot_seed: int) -> tuple[list[str], dict]:
    """Per-cohort rows (Holm over the cohorts) and the pooled row for one comparison; also returns the pooled stats."""
    rows, per, seeds_used = [], {}, {}
    xs, ys = [], []
    for c in COHORTS:
        if c not in data[a] or c not in data[b]:
            continue
        x, y, seeds = paired_units(data, a, b, c, metric)
        per[c] = paired_stats(x, y, n_boot, boot_seed)
        seeds_used[c] = seeds
        xs.append(x)
        ys.append(y)
    if not per:
        return [f"== {a} vs {b} ==  ({label}): no common cohorts"], {}
    holm_t = dict(zip(per, holm([per[c]["t_p"] for c in per])))
    holm_w = dict(zip(per, holm([per[c]["w_p"] for c in per])))
    pooled = paired_stats(np.concatenate(xs), np.concatenate(ys), n_boot, boot_seed)
    pooled["seeds"] = seeds_label(seeds_used)
    rows.append(f"== {a} vs {b} ==  ({label}; seeds averaged per (cohort, fold): {pooled['seeds']}; "
                f"Holm over {len(per)} cohorts)")
    for c, s in per.items():
        rows.append(f"  {c:10s} delta {s['delta']:+.3f}  t p={fmt_p(s['t_p'])}  W p={fmt_p(s['w_p'])}  "
                    f"Holm t p={fmt_p(holm_t[c])}  Holm W p={fmt_p(holm_w[c])}  wins {s['wins']}/{s['n']}  n={s['n']} folds")
    rows.append(f"  {'ALL':10s} delta {pooled['delta']:+.3f}  95% CI [{pooled['lo']:+.3f}, {pooled['hi']:+.3f}]  "
                f"t p={fmt_p(pooled['t_p'], 4)}  W p={fmt_p(pooled['w_p'], 4)}  wins {pooled['wins']}/{pooled['n']}  n={pooled['n']} folds")
    return rows, pooled


def primary_table(e20: Runs, os_: Runs, comparisons: list[tuple[str, str]], os_comparisons: list[tuple[str, str]],
                  notes: list[str], n_boot: int, boot_seed: int, e20_root: str, os_root: str,
                  e10: Runs | None = None, e10_comparisons: list[tuple[str, str]] | None = None, e10_roots: str = "") -> str:
    lines = [
        "Primary comparisons: Harrell's C-index on the held-out fold, 20-epoch budget, selection 'last' (last epoch)",
        f"Campaigns: DSS = {e20_root}, OS = {os_root}. Seeds available (run directories <method>[_seedN]):",
        f"  DSS: {seeds_available(e20)}",
        f"  OS:  {seeds_available(os_)}",
    ] + ([f"  DSS, 10-epoch budget ({e10_roots}): {seeds_available(e10)}"] if e10 else []) + [
        "Unit of analysis: one value per (cohort, fold) = mean over the seeds available for BOTH methods (listed per block);",
        "  paired over the 25 (cohort, fold) units pooled (n=25) and over the 5 folds per cohort (n=5). The 75 (fold, seed)",
        "  pairs of Table 1 are not independent units (the seeds of one fold share the validation patients), so n here is",
        "  25 / 5, not 75 / 15. Against late fusion both sides are restricted to seeds 0-1; the OS block has a single seed",
        "  (no averaging). Per-fold seeds are run seed + fold.",
        "Held-out fold: in the 5-fold CV the validation fold IS the reported fold (c_index == last_epoch_val_cindex for every",
        "  trained fold, verified while loading); no checkpoint selection is applied, and any post-hoc selection on val_loss /",
        "  val_cindex would be evaluated on the same fold it selects on.",
        "Statistics: delta = first method - second method (positive favours the first); 95% CI = percentile bootstrap of the",
        f"  mean delta over folds ({n_boot:,} resamples with replacement of the paired units, numpy default_rng({boot_seed}));",
        "  t p = paired t-test (scipy.stats.ttest_rel); W p = Wilcoxon signed-rank (two-sided, exact for n <= 50 without",
        "  ties); Holm = Holm-Bonferroni step-down over the 5 per-cohort tests of one comparison (t and W adjusted",
        "  separately; the pooled test is not part of that family); wins = folds with delta > 0.",
        "",
    ]
    if notes:
        lines += ["Consistency notes (folds where c_index != last_epoch_val_cindex or metrics.c_index):"] + [f"  {n}" for n in notes] + [""]
    else:
        lines += ["Consistency check: every trained fold has c_index == last_epoch_val_cindex and == metrics[<condition>].c_index.", ""]

    blocks, summary = [], []
    campaigns = [(e20, comparisons, "DSS"), (os_, os_comparisons, "OS")]
    if e10 and e10_comparisons:
        campaigns.append((e10, e10_comparisons, "DSS-10ep"))
    for data, pairs, label in campaigns:
        for a, b in pairs:
            if a not in data or b not in data:
                missing = [m for m in (a, b) if m not in data]
                blocks.append(f"== {a} vs {b} ==  ({label}): missing run(s) {', '.join(missing)}")
                blocks.append("")
                continue
            rows, pooled = comparison_block(data, a, b, label, "c_index", n_boot, boot_seed)
            blocks += rows + [""]
            if pooled:
                summary.append(f"  {a + ' vs ' + b:44s} {label:8s} {pooled['seeds']:>6s}  {pooled['delta']:+.3f}  "
                               f"[{pooled['lo']:+.3f}, {pooled['hi']:+.3f}]  {fmt_p(pooled['t_p'], 4):>7s}  {fmt_p(pooled['w_p'], 4):>7s}  "
                               f"{pooled['wins']:2d}/{pooled['n']}")
    lines += ["Pooled over the seed-averaged (cohort, fold) units (n = 25 folds unless stated):",
              f"  {'comparison':44s} {'end.':8s} {'seeds':>6s}  {'delta':>6s}  {'95% bootstrap CI':16s}  {'t p':>7s}  {'W p':>7s}  wins"]
    lines += summary + ["", "Per cohort (n = 5 folds each) and pooled, per comparison:", ""] + blocks
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------------------
# Secondary metrics table
# ---------------------------------------------------------------------------------------
def seed_mean_std(seed_folds: dict[int, dict[int, Fold]], metric: str) -> tuple[float, float, int, int]:
    """Mean over seeds of the per-seed fold mean (non-finite folds dropped), sample std over seeds, n seeds, folds used."""
    means, n_folds = [], set()
    for folds in seed_folds.values():
        vals = np.asarray([v[metric] for v in folds.values()], float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            means.append(float(vals.mean()))
            n_folds.add(len(vals))
    if not means:
        return float("nan"), float("nan"), 0, 0
    sd = float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
    return float(np.mean(means)), sd, len(means), max(n_folds)


def secondary_table(data: Runs, methods: list[str], ref: str, conditions: dict[str, str], n_boot: int, boot_seed: int,
                    root: str) -> str:
    methods = [m for m in methods if m in data]
    lines = [
        "Secondary metrics: Uno's IPCW C-index (c_index_ipcw, truncated at the 75th percentile of training event times) and the",
        f"integrated Brier score (ibs, interior quartile grid; lower is better), 20-epoch DSS campaign ({root}), selection 'last'.",
        "Cells: mean over seeds of the per-seed 5-fold mean +- sample std over seeds (n = seeds). Held-out fold = the CV validation",
        "  fold (c_index == last_epoch_val_cindex; no checkpoint selection; any post-hoc selection on val_loss / val_cindex would",
        "  be evaluated on the same fold it selects on).",
        f"Paired tests vs {ref}: one value per (cohort, fold) = mean over the seeds available for both methods, then paired over",
        "  the pooled (cohort, fold) units (n = folds with a finite value on both sides); delta = method - reference; 95% CI =",
        f"  percentile bootstrap over the paired units ({n_boot:,} resamples, numpy default_rng({boot_seed})); t p = paired t-test;",
        "  W p = Wilcoxon signed-rank (two-sided, exact); wins = folds with delta > 0 (for IBS a negative delta is the improvement).",
        "Condition used per method: " + ", ".join(f"{m}: metrics[{conditions.get(m, '?')}]" for m in methods),
        "Seeds available: " + seeds_available({m: data[m] for m in methods}),
    ]
    for metric in ("c_index_ipcw", "ibs"):
        for (c, k), ms in sorted(undefined_folds(data, methods, metric).items()):
            who = "every method" if set(ms) == set(methods) else ", ".join(sorted(ms))
            lines.append(f"{metric} is undefined (NaN) on {c} fold {k} for {who} (sksurv could not evaluate it on that fold's grid); the fold is")
            lines.append("  dropped from the affected cell means (marked *, folds used stated) and from the paired units (n stated).")
    for metric, title in (("c_index_ipcw", "Uno's IPCW C-index (higher is better)"), ("ibs", "Integrated Brier score (lower is better)")):
        lines += ["", f"== {title} ==", f"{'cohort':10s} " + " ".join(f"{m:>29s}" for m in methods)]
        short = set()
        for c in COHORTS:
            cells = []
            for m in methods:
                mu, sd, n, nf = seed_mean_std(data[m].get(c, {}), metric)
                if n == 0:
                    cells.append(f"{'-':>29s}")
                    continue
                if nf < 5:
                    short.add(nf)
                cells.append(f"{mu:.3f} +- {sd:.3f} (n={n}){'*' if nf < 5 else ''}".rjust(29))
            lines.append(f"{c:10s} " + " ".join(cells))
        if short:
            lines.append(f"  * mean over {', '.join(str(x) for x in sorted(short))} folds (fold with an undefined value dropped)")
        if ref in data:
            lines.append(f"paired vs {ref}, pooled over seed-averaged (cohort, fold) units: delta, 95% bootstrap CI, t p, W p, wins, n, seeds")
            for m in methods:
                if m == ref:
                    continue
                xs, ys, seeds_used = [], [], {}
                for c in COHORTS:
                    if c in data[m] and c in data[ref]:
                        x, y, seeds = paired_units(data, m, ref, c, metric)
                        xs.append(x)
                        ys.append(y)
                        seeds_used[c] = seeds
                s = paired_stats(np.concatenate(xs), np.concatenate(ys), n_boot, boot_seed) if xs else paired_stats(np.array([]), np.array([]), n_boot, boot_seed)
                lines.append(f"  {m:24s} delta {s['delta']:+.4f}  95% CI [{s['lo']:+.4f}, {s['hi']:+.4f}]  t p={fmt_p(s['t_p'], 4)}  "
                             f"W p={fmt_p(s['w_p'], 4)}  wins {s['wins']}/{s['n']}  n={s['n']} folds  seeds {seeds_label(seeds_used)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
def parse_pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for it in items:
        if it.count(":") != 1:
            raise SystemExit(f"comparison must be <method>:<reference>, got {it!r}")
        a, b = it.split(":")
        pairs.append((a, b))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e20", default="pod_results/outputs_e20", help="20-epoch DSS campaign root")
    ap.add_argument("--os", dest="os_root", default="pod_results/outputs_os", help="OS-endpoint campaign root")
    ap.add_argument("--comparisons", nargs="+", default=DEFAULT_COMPARISONS, help="<method>:<reference> pairs in --e20")
    ap.add_argument("--os-comparisons", nargs="+", default=DEFAULT_OS_COMPARISONS, help="<method>:<reference> pairs in --os")
    ap.add_argument("--e10", nargs="+", default=DEFAULT_E10_ROOTS, help="10-epoch DSS campaign roots (merged)")
    ap.add_argument("--e10-comparisons", nargs="+", default=DEFAULT_E10_COMPARISONS, help="<method>:<reference> pairs in --e10")
    ap.add_argument("--secondary-methods", nargs="+", default=DEFAULT_SECONDARY)
    ap.add_argument("--secondary-ref", default="survpath_e20")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/final", help="where table_primary_tests.txt and table_secondary_metrics.txt go")
    args = ap.parse_args()

    comparisons, os_comparisons = parse_pairs(args.comparisons), parse_pairs(args.os_comparisons)
    e20_methods = {m for p in comparisons for m in p} | set(args.secondary_methods) | {args.secondary_ref}
    e20, conditions, notes = collect(Path(args.e20), e20_methods)
    os_, _os_conditions, os_notes = collect(Path(args.os_root), {m for p in os_comparisons for m in p})
    e10_comparisons = parse_pairs(args.e10_comparisons)
    e10, _e10_conditions, e10_notes = collect_many([Path(r) for r in args.e10], {m for p in e10_comparisons for m in p})

    primary = primary_table(e20, os_, comparisons, os_comparisons, notes + os_notes + e10_notes, args.n_boot, args.boot_seed, args.e20, args.os_root,
                            e10=e10, e10_comparisons=e10_comparisons, e10_roots=", ".join(args.e10))
    secondary = secondary_table(e20, args.secondary_methods, args.secondary_ref, conditions, args.n_boot, args.boot_seed, args.e20)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (("table_primary_tests.txt", primary), ("table_secondary_metrics.txt", secondary)):
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(text)
        print(f"written: {out / name}\n")


if __name__ == "__main__":
    main()
