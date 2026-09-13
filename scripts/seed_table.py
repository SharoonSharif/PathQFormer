"""Table 1 with seeds: per (method, cohort) the mean C-index over runs (one run = 5-fold CV with one seed),
its spread over seeds, and a paired comparison against a reference method over all (fold, seed) pairs.

    python scripts/seed_table.py pod_results/outputs_v2 pod_results/outputs_ablate \
        --methods pathq_fast_e10 survpath_e10 --ref survpath_e10
    python scripts/seed_table.py pod_results/outputs_e20 --methods pathq_fast_e20 pathq_fast_e20_aux survpath_e20 --ref survpath_e20

Run directories are <root>/<method>[_seed<N>]/<cohort>/results.json; seed 0 has no suffix.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def collect(roots, methods):
    """{method: {cohort: {seed: {fold: c_index}}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    pat = re.compile(r"^(?P<m>.+?)(?:_seed(?P<s>\d+))?$")
    for root in roots:
        for rj in Path(root).glob("*/*/results.json"):
            run, cohort = rj.parts[-3], rj.parts[-2]
            m = pat.match(run)
            method, seed = m.group("m"), int(m.group("s") or 0)
            if method not in methods:
                continue
            with open(rj) as f:
                res = json.load(f)
            folds = {d["fold"]: d["c_index"] for d in res["folds"]}
            if len(folds) == 5:
                data[method][cohort][seed] = folds
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()
    data = collect(args.roots, args.methods)
    cohorts = sorted({c for m in data.values() for c in m})

    print(f"{'cohort':10s} " + " ".join(f"{m:>28s}" for m in args.methods))
    for c in cohorts:
        cells = []
        for m in args.methods:
            seeds = data[m].get(c, {})
            if not seeds:
                cells.append(f"{'-':>28s}")
                continue
            run_means = [np.mean(list(f.values())) for f in seeds.values()]
            mu, sd = np.mean(run_means), (np.std(run_means, ddof=1) if len(run_means) > 1 else float("nan"))
            cells.append(f"{mu:.3f} +- {sd:.3f} (n={len(run_means)})".rjust(28))
        print(f"{c:10s} " + " ".join(cells))

    if args.ref and args.ref in data:
        print(f"\nPaired over (fold, seed) pairs vs {args.ref}: delta, paired-t p, Wilcoxon p, n pairs")
        for m in args.methods:
            if m == args.ref:
                continue
            for c in cohorts:
                a, b = [], []
                for seed, folds in data[m].get(c, {}).items():
                    ref = data[args.ref].get(c, {}).get(seed)
                    if not ref:
                        continue
                    for k in folds:
                        if k in ref:
                            a.append(folds[k]); b.append(ref[k])
                if len(a) >= 3:
                    a, b = np.array(a), np.array(b)
                    tp = stats.ttest_rel(a, b).pvalue
                    try:
                        wp = stats.wilcoxon(a, b).pvalue
                    except ValueError:
                        wp = float("nan")
                    print(f"  {m:24s} {c:10s} delta {np.mean(a - b):+.3f}  t p={tp:.3f}  W p={wp:.3f}  n={len(a)}")
        # all cohorts pooled
        for m in args.methods:
            if m == args.ref:
                continue
            a, b = [], []
            for c in cohorts:
                for seed, folds in data[m].get(c, {}).items():
                    ref = data[args.ref].get(c, {}).get(seed)
                    if ref:
                        for k in folds:
                            if k in ref:
                                a.append(folds[k]); b.append(ref[k])
            if len(a) >= 3:
                a, b = np.array(a), np.array(b)
                print(f"  {m:24s} {'ALL':10s} delta {np.mean(a - b):+.3f}  t p={stats.ttest_rel(a, b).pvalue:.4f}  W p={stats.wilcoxon(a, b).pvalue:.4f}  n={len(a)}")


if __name__ == "__main__":
    main()
