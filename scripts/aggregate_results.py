"""Collect every ``results.json`` under one or more output roots into comparison tables.

Usage:
    python scripts/aggregate_results.py outputs_v2 --out results/summary_all.md
    python scripts/aggregate_results.py outputs_v2 outputs_ablation --compare hybrid

Each run directory is ``<root>/<run_name>/<cancer>/results.json``; the run name is the
sub-folder of the root (e.g. ``outputs_v2/hybrid/blca`` -> run ``hybrid``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.evaluate import paired_test  # noqa: E402


def fmt(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def load_runs(roots: list[str]) -> list[dict]:
    runs = []
    for root in roots:
        for path in sorted(Path(root).glob("**/results.json")):
            with open(path) as f:
                res = json.load(f)
            rel = path.relative_to(root).parts
            run_name = "/".join(rel[:-2]) if len(rel) > 2 else str(root)
            res["_run"], res["_path"] = run_name, str(path)
            runs.append(res)
    return runs


def main_table(runs: list[dict]) -> list[str]:
    lines = [
        "| Run | Cancer | Folds | Train modalities | Selection | C-index (both) | 95% CI | IPCW | IBS | td-AUC | WSI-only | Genomics-only | log-rank p (median) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        s, c = r["summary"], r["config"]
        conds = s.get("conditions", {})
        primary = "both" if "both" in conds else next(iter(conds), None)
        if primary is None:
            continue
        b = conds[primary]
        wsi = conds.get("wsi_only", {}).get("c_index", {}).get("mean")
        gen = conds.get("genomic_only", {}).get("c_index", {}).get("mean")
        p_vals = [d["metrics"][primary].get("logrank_p") for d in r["folds"]]
        lines.append(
            f"| {r['_run']} | {r['cancer_type'].upper()} | {s['n_folds']} | {c.get('train_modalities', 'both')} | {c.get('selection_metric')} "
            f"| **{fmt(b['c_index']['mean'])} ± {fmt(b['c_index']['std'])}** | [{fmt(b['c_index']['ci95'][0], 3)}, {fmt(b['c_index']['ci95'][1], 3)}] "
            f"| {fmt(b['c_index_ipcw']['mean'])} | {fmt(b['ibs']['mean'])} | {fmt(b['iauc']['mean'])} "
            f"| {fmt(wsi)} | {fmt(gen)} | {fmt(float(np.nanmedian([p for p in p_vals if p is not None])), 3) if p_vals else 'nan'} |"
        )
    return lines


def missing_table(runs: list[dict]) -> list[str]:
    rows = [r for r in runs if r["summary"].get("partial_missing")]
    if not rows:
        return []
    rates = sorted({pm["rate"] for r in rows for pm in r["summary"]["partial_missing"]})
    lines = ["", "### Random missing modality at test time (C-index, mean over folds)", "",
             "| Run | Cancer | 0% | " + " | ".join(f"{int(x * 100)}%" for x in rates) + " | WSI-only (100%) | Genomics-only (100%) |",
             "|---|---|---|" + "---|" * len(rates) + "---|---|"]
    for r in rows:
        s = r["summary"]
        by_rate = {pm["rate"]: pm["mean"] for pm in s["partial_missing"]}
        conds = s["conditions"]
        lines.append(
            f"| {r['_run']} | {r['cancer_type'].upper()} | {fmt(conds['both']['c_index']['mean'])} | "
            + " | ".join(fmt(by_rate.get(x)) for x in rates)
            + f" | {fmt(conds.get('wsi_only', {}).get('c_index', {}).get('mean'))} | {fmt(conds.get('genomic_only', {}).get('c_index', {}).get('mean'))} |"
        )
    return lines


def per_fold_table(runs: list[dict]) -> list[str]:
    lines = ["", "### Per-fold C-index (selected checkpoint)", "", "| Run | Cancer | " + " | ".join(f"fold {i}" for i in range(5)) + " | mean |", "|---|---|" + "---|" * 6]
    for r in runs:
        per = {d["fold"]: d["c_index"] for d in r["folds"]}
        vals = [per.get(i) for i in range(5)]
        lines.append(f"| {r['_run']} | {r['cancer_type'].upper()} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(r['summary']['c_index']['mean'])} |")
    return lines


def comparisons(runs: list[dict], reference: str) -> list[str]:
    ref = {r["cancer_type"]: r for r in runs if r["_run"] == reference}
    if not ref:
        return []
    lines = ["", f"### Paired tests vs `{reference}` (same folds)", "", "| Run | Cancer | Δ C-index | paired t p | Wilcoxon p | n folds |", "|---|---|---|---|---|---|"]
    for r in runs:
        if r["_run"] == reference or r["cancer_type"] not in ref:
            continue
        a = {d["fold"]: d["c_index"] for d in r["folds"]}
        b = {d["fold"]: d["c_index"] for d in ref[r["cancer_type"]]["folds"]}
        common = sorted(set(a) & set(b))
        if len(common) < 2:
            continue
        av, bv = [a[k] for k in common], [b[k] for k in common]
        t = paired_test(av, bv)
        lines.append(f"| {r['_run']} | {r['cancer_type'].upper()} | {fmt(np.mean(av) - np.mean(bv), 4)} | {fmt(t['t_p'], 3)} | {fmt(t['wilcoxon_p'], 3)} | {t['n']} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="results/summary_all.md")
    ap.add_argument("--compare", default=None, help="run name to use as the paired-test reference")
    args = ap.parse_args()

    runs = load_runs(args.roots)
    if not runs:
        raise SystemExit("no results.json found")
    lines = ["# PathQ-Former results", "", f"{len(runs)} run(s) under {', '.join(args.roots)}", "", *main_table(runs)]
    lines += missing_table(runs) + per_fold_table(runs)
    if args.compare:
        lines += comparisons(runs, args.compare)
    md = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
