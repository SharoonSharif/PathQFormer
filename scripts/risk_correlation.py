"""Correlations between per-patient risk scores, from the predictions_*.csv files each fold writes.

    python scripts/risk_correlation.py pod_results/outputs_e20 --run pathq_fast_e20_aux --others survpath_e20 mlp_omics_e20 abmil_e20

Two questions, both answered with Spearman rank correlations over the validation patients of a fold, then averaged
over folds (and seeds when <run>_seed1, _seed2 exist):

1. Within the multimodal model ("routing"): how much does the fused risk agree with the same checkpoint's WSI-only
   and RNA-only risks? A model that ignores one modality has correlation ~1 with the other branch.
2. Across models: does the multimodal risk rank patients like SurvPath / the RNA MLP / ABMIL do? Low correlation
   with an equally accurate model means the two capture different signal (ensemble headroom).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

COHORTS = ["blca", "brca", "coadread", "hnsc", "stad"]


def run_dirs(root: Path, run: str):
    for rd in sorted(root.glob(f"{run}*")):
        tail = rd.name[len(run):]
        if not tail or tail.startswith("_seed"):
            yield rd


def load(fold_dir: Path, cond: str) -> pd.Series | None:
    """Risk per patient for a test-time condition; for single-modality baselines `both` falls back to the
    only predictions file they write (predictions_genomic_only.csv for SNN/MLP, predictions_wsi_only.csv for ABMIL)."""
    p = fold_dir / f"predictions_{cond}.csv"
    if not p.exists() and cond == "both":
        cands = sorted(fold_dir.glob("predictions_*.csv")) if fold_dir.is_dir() else []
        if len(cands) != 1:
            return None
        p = cands[0]
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.set_index("case_id")["risk"]


def spearman(a: pd.Series, b: pd.Series) -> float:
    idx = a.index.intersection(b.index)
    if len(idx) < 5:
        return float("nan")
    return float(stats.spearmanr(a.loc[idx], b.loc[idx]).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--run", default="pathq_fast_e20_aux")
    ap.add_argument("--others", nargs="+", default=["survpath_e20", "mlp_omics_e20", "abmil_e20", "pathq_fast_e20"])
    args = ap.parse_args()
    root = Path(args.root)

    within = defaultdict(lambda: defaultdict(list))   # cohort -> pair -> [rho per fold/seed]
    across = defaultdict(lambda: defaultdict(list))
    for rd in run_dirs(root, args.run):
        seed_tag = rd.name[len(args.run):] or "_seed0"
        for cohort in COHORTS:
            for fd in sorted((rd / cohort).glob("fold_*")):
                both = load(fd, "both")
                if both is None:
                    continue
                for cond in ("wsi_only", "genomic_only"):
                    s = load(fd, cond)
                    if s is not None:
                        within[cohort][f"both~{cond}"].append(spearman(both, s))
                w, g = load(fd, "wsi_only"), load(fd, "genomic_only")
                if w is not None and g is not None:
                    within[cohort]["wsi_only~genomic_only"].append(spearman(w, g))
                for other in args.others:
                    od = root / f"{other}{seed_tag if seed_tag != '_seed0' else ''}" / cohort / fd.name
                    s = load(od, "both")
                    if s is not None:
                        across[cohort][other].append(spearman(both, s))

    def table(title, data):
        cols = sorted({k for coh in data.values() for k in coh})
        print(f"\n== {title} (Spearman rho, mean +- std over folds x seeds; n) ==")
        print(f"{'cohort':10s} " + " ".join(f"{c:>28s}" for c in cols))
        for cohort in COHORTS:
            if cohort not in data:
                continue
            row = f"{cohort:10s} "
            for c in cols:
                v = np.array(data[cohort].get(c, []), dtype=float)
                v = v[~np.isnan(v)]
                row += f" {v.mean():.2f} +- {v.std():.2f} (n={len(v)})".rjust(29) if len(v) else f"{'-':>29s}"
            print(row)
        pooled = {c: np.nanmean([x for coh in data.values() for x in coh.get(c, [])]) for c in cols}
        print(f"{'pooled':10s} " + " ".join(f"{pooled[c]:>28.2f}" for c in cols))

    table(f"{args.run}: fused risk vs the same checkpoint's single-modality risks", within)
    table(f"{args.run} fused risk vs other models' risk (same fold, same seed)", across)


if __name__ == "__main__":
    main()
