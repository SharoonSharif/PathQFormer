"""Missing-modality table over seeds: same checkpoint evaluated with both / WSI-only / RNA-only inputs.

    python scripts/missing_modality_table.py pod_results/outputs_e20 --methods pathq_fast_e20_aux pathq_fast_e20 survpath_e20

For every method it collects <method>, <method>_seed1, <method>_seed2, ... and reports, per cohort, the mean over
seeds of the 5-fold mean C-index (+- std over seeds) for each test-time condition stored in results.json
(`both`, `wsi_only`, `genomic_only`, and the mean-imputation variants `*_impute` when scripts/eval_missing_impute.py
has been run), plus the random-missing curve (10/20/30/50 % of patients missing one modality). Paired tests compare
each condition with `both` over (fold, seed) pairs and, seed-averaged, over folds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

COHORTS = ["blca", "brca", "coadread", "hnsc", "stad"]


def collect(root: Path, method: str):
    """-> {cohort: {condition: {(fold, seed): c}}}, {cohort: {rate: {(fold, seed): c}}}"""
    cond = defaultdict(lambda: defaultdict(dict))
    partial = defaultdict(lambda: defaultdict(dict))
    for rd in sorted(root.glob(f"{method}*")):
        tail = rd.name[len(method):]
        if tail and not tail.startswith("_seed"):
            continue
        run_seed = int(tail[len("_seed"):]) if tail else 0   # the run's seed; per-fold seeds are seed + fold
        for res in rd.glob("*/results.json"):
            cohort = res.parent.name
            d = json.load(open(res))
            for f in d["folds"]:
                key = (f["fold"], run_seed)
                for c, m in f.get("metrics", {}).items():
                    if isinstance(m, dict) and m.get("c_index") is not None:
                        cond[cohort][c][key] = m["c_index"]
                for pm in f.get("partial_missing", []) or []:
                    partial[cohort][pm["rate"]][key] = pm["c_index_mean"]
    return cond, partial


def seed_mean_std(vals: dict) -> tuple[float, float, int]:
    by_seed = defaultdict(list)
    for (fold, seed), c in vals.items():
        by_seed[seed].append(c)
    means = [np.mean(v) for v in by_seed.values()]
    return float(np.mean(means)), float(np.std(means, ddof=1)) if len(means) > 1 else float("nan"), len(means)   # sample sd over seeds, as seed_table.py


def paired(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys])
    y = np.array([b[k] for k in keys])
    if len(keys) < 3:
        return float("nan"), float("nan"), float("nan"), len(keys)
    t = stats.ttest_rel(x, y).pvalue
    try:
        w = stats.wilcoxon(x, y).pvalue
    except ValueError:
        w = float("nan")
    return float(np.mean(x - y)), float(t), float(w), len(keys)


def seed_average(vals: dict) -> dict:
    by_fold = defaultdict(list)
    for (fold, seed), c in vals.items():
        by_fold[fold].append(c)
    return {f: float(np.mean(v)) for f, v in by_fold.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--methods", nargs="+", default=["pathq_fast_e20_aux", "pathq_fast_e20", "survpath_e20"])
    args = ap.parse_args()

    for method in args.methods:
        cond_all, part_all = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))
        for root in args.roots:
            c, p = collect(Path(root), method)
            for cohort in c:
                for k, v in c[cohort].items():
                    cond_all[cohort][k].update(v)
                for k, v in p[cohort].items():
                    part_all[cohort][k].update(v)
        conditions = sorted({k for coh in cond_all.values() for k in coh}, key=lambda k: (k != "both", k))
        print(f"\n== {method} ==  (mean over seeds of the 5-fold mean C-index +- std over seeds; n = seeds)")
        print("cohort     " + " ".join(f"{c:>26s}" for c in conditions))
        pooled = defaultdict(dict)
        for cohort in COHORTS:
            if cohort not in cond_all:
                continue
            row = f"{cohort:10s} "
            for c in conditions:
                vals = cond_all[cohort].get(c, {})
                if vals:
                    m, s, n = seed_mean_std(vals)
                    row += f" {m:.3f} +- {s:.3f} (n={n})".rjust(27)
                    for k, v in vals.items():
                        pooled[c][(cohort, *k)] = v
                else:
                    row += f"{'-':>27s}"
            print(row)
        if part_all:
            rates = sorted({r for coh in part_all.values() for r in coh})
            print("random missing (one modality):  " + "  ".join(f"{int(r * 100):>3d}%" for r in rates))
            for cohort in COHORTS:
                if cohort in part_all:
                    print(f"  {cohort:10s}" + "  ".join(f"{seed_mean_std(part_all[cohort][r])[0]:.3f}" for r in rates) if all(r in part_all[cohort] for r in rates) else "")
        if "both" in pooled:
            print("paired vs both, pooled over cohorts:  delta | t p | Wilcoxon p | n   [(fold,seed) pairs]   ||   seed-averaged folds")
            for c in conditions:
                if c == "both" or c not in pooled:
                    continue
                d, t, w, n = paired(pooled[c], pooled["both"])
                sa_c = {}
                sa_b = {}
                for cohort in COHORTS:
                    for f, v in seed_average(cond_all[cohort].get(c, {})).items():
                        sa_c[(cohort, f)] = v
                    for f, v in seed_average(cond_all[cohort].get("both", {})).items():
                        sa_b[(cohort, f)] = v
                d2, t2, w2, n2 = paired(sa_c, sa_b)
                print(f"  {c:22s} {d:+.3f} | {t:.3g} | {w:.3g} | {n:3d}   ||   {d2:+.3f} | {t2:.3g} | {w2:.3g} | {n2:3d}")


if __name__ == "__main__":
    main()
