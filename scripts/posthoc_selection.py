"""Post-hoc checkpoint-selection analysis from the stored per-epoch histories of the archived runs (no training).

    python scripts/posthoc_selection.py                                      # -> results/final/table_selection_rules.txt
    python scripts/posthoc_selection.py --patience 5 --min-epochs 6 --out -  # the outputs_v2 setting, to stdout only

Every fold of a fixed-budget run (selection 'last') stores history[{epoch, train_loss, val_loss, val_cindex, ...}],
so the C-index that a different checkpoint-selection rule would have reported can be read off without retraining:

    (i)   last    C-index at the final epoch: the fixed budget the paper reports (c_index == last_epoch_val_cindex)
    (ii)  es      validation-loss early stopping with patience P and min_epochs M, replayed exactly as
                  src/training/train.py does it: track the best val_loss (strict improvement, finite values only);
                  stop at the first epoch >= M after P consecutive non-improving epochs; the selected checkpoint is
                  the best-val_loss epoch seen before stopping (if patience never fires: argmin val_loss over all epochs)
    (iii) argmin  argmin val_loss over all epochs (no patience)
    (iv)  oracle  max val_cindex over all epochs (= best_val_cindex_any_epoch)

CAVEAT (repeated in every table header): in the 5-fold CV the validation fold IS the reported held-out fold, so every
rule selects on the same patients it is evaluated on. (ii)-(iv) are therefore optimistically biased relative to a
protocol with a separate selection set, and (iv) is an upper bound no protocol can reach. Rule (i) selects nothing.

Table A  per method x cohort: mean over (fold, seed) of the C-index under (i)-(iv), selected / run epochs under (ii)
Table B  paired tests of (ii) - (i) and (iii) - (i): pooled over (fold, seed) pairs and seed-averaged over folds
Table C  patients and events per validation fold per cohort (identical for every method and seed)
Table D  the actual outputs_v2 val_loss campaign (patience 5, min_epochs 6): reported C-index, selected epoch, oracle;
         the same replay is run with each run's own config and checked against its stored selected_epoch / epochs_run

Run directories are <root>/<method>[_seed<N>]/<cohort>/fold_<k>/fold_results.json; seed 0 has no suffix. Runs
without histories (late fusion) are skipped. When one (method, seed, cohort) exists under several roots the copy in
the first root listed is used (seed 0 of the 10-epoch runs is in outputs_v2, seeds 1-2 in outputs_ablate).
Everything is a deterministic function of the archived JSON files (sorted traversal, no randomness, no timestamps).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

try:  # Windows consoles default to cp1252; the tables are ASCII but never let the console crash the run
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

COHORTS = ["blca", "brca", "coadread", "hnsc", "stad"]
RUN_PAT = re.compile(r"^(?P<m>.+?)(?:_seed(?P<s>\d+))?$")
E20_METHODS = ["pathq_fast_e20", "pathq_fast_e20_aux", "survpath_e20", "abmil_e20", "snn_e20", "mlp_omics_e20",
               "pathq_e20_wsi_only", "pathq_e20_genomic_only"]
E10_METHODS = ["pathq_fast_e10", "survpath_e10"]
V2_METHODS = ["baseline_survpath", "baseline_abmil", "baseline_snn", "baseline_mlp_omics", "hybrid"]
CLAIM_COHORTS = ["stad", "hnsc", "brca"]   # the cohorts the manuscript names
CAVEAT = ("CAVEAT: the validation fold is the reported held-out fold, so every selection rule picks the checkpoint on "
          "the same patients it is scored on; (ii)-(iv) are optimistically biased and (iv) is an unattainable bound.")


# ----------------------------------------------------------------------------------------------- replay / records
def emulate_early_stopping(val_loss: np.ndarray, patience: int, min_epochs: int) -> tuple[int, int | None]:
    """Replay train.py's val_loss early stopping on a stored history.

    Returns (selected_epoch, stop_epoch), both 1-based; stop_epoch is None when the budget ran out before patience
    fired, in which case selected_epoch is the argmin of val_loss over all epochs (first occurrence, as in training).
    """
    best, best_epoch, bad = None, -1, 0
    for i, v in enumerate(val_loss):
        epoch = i + 1
        improved = best is None or (np.isfinite(v) and v < best)
        if improved:
            best, best_epoch, bad = v, epoch, 0
        else:
            bad += 1
        if bad >= patience and epoch >= min_epochs:
            return best_epoch, epoch
    return best_epoch, None


def fold_record(d: dict, patience: int, min_epochs: int) -> dict | None:
    """One fold's C-index under rules (i)-(iv), or None when the fold stored no history (late fusion)."""
    hist = d.get("history") or []
    if not hist:
        return None
    vl = np.array([h["val_loss"] for h in hist], dtype=float)
    vc = np.array([h["val_cindex"] for h in hist], dtype=float)
    n = len(hist)
    ep_es, stop = emulate_early_stopping(vl, patience, min_epochs)
    ep_argmin, _ = emulate_early_stopping(vl, n + 1, 0)   # same tie-breaking as (ii), patience never fires
    ep_oracle = int(np.nanargmax(vc)) + 1
    return {
        "fold": int(d["fold"]), "n_val": int(d["n_val"]), "events_val": int(d["events_val"]), "epochs": n,
        "c_last": float(vc[-1]), "c_reported": float(d["c_index"]),
        "c_es": float(vc[ep_es - 1]), "ep_es": ep_es, "run_es": stop if stop is not None else n, "stopped": stop is not None,
        "c_argmin": float(vc[ep_argmin - 1]), "ep_argmin": ep_argmin,
        "c_oracle": float(vc[ep_oracle - 1]), "ep_oracle": ep_oracle,
        "selected_epoch": d.get("selected_epoch"), "best_any": d.get("best_val_cindex_any_epoch"),
    }


def collect(roots: list[str], methods: list[str], patience: int, min_epochs: int):
    """-> {method: {cohort: {(fold, seed): record}}}, notes (skipped duplicates, incomplete cells)."""
    data: dict = defaultdict(lambda: defaultdict(dict))
    seen: dict = {}
    notes: list[str] = []
    for root in roots:
        for rd in sorted(Path(root).iterdir()):
            if not rd.is_dir():
                continue
            m = RUN_PAT.match(rd.name)
            method, seed = m.group("m"), int(m.group("s") or 0)
            if method not in methods:
                continue
            for cd in sorted(rd.iterdir()):
                if not cd.is_dir() or cd.name not in COHORTS:
                    continue
                key = (method, seed, cd.name)
                if key in seen:
                    notes.append(f"{root}/{rd.name}/{cd.name}: duplicate of {seen[key]}/{rd.name}/{cd.name}, skipped")
                    continue
                recs = {}
                for fr in sorted(cd.glob("fold_*/fold_results.json")):
                    with open(fr) as f:
                        r = fold_record(json.load(f), patience, min_epochs)
                    if r is not None:
                        recs[(r["fold"], seed)] = r
                if recs:
                    seen[key] = root
                    data[method][cd.name].update(recs)
                    if len(recs) < 5:
                        notes.append(f"{root}/{rd.name}/{cd.name}: only {len(recs)} folds with a history")
                else:
                    notes.append(f"{root}/{rd.name}/{cd.name}: no fold histories, skipped")
    return data, notes


# ----------------------------------------------------------------------------------------------- helpers
def mean(vals) -> float:
    vals = list(vals)
    return float(np.mean(vals)) if vals else float("nan")


def paired(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int, int]:
    """delta mean(x - y), paired-t p, Wilcoxon p, n, wins (x > y)."""
    if len(x) < 3:
        return float(np.mean(x - y)) if len(x) else float("nan"), float("nan"), float("nan"), len(x), int(np.sum(x > y))
    t = stats.ttest_rel(x, y).pvalue
    try:
        w = stats.wilcoxon(x, y).pvalue
    except ValueError:  # all differences zero
        w = float("nan")
    return float(np.mean(x - y)), float(t), float(w), len(x), int(np.sum(x > y))


def pairs(cells: dict, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(cells)
    return np.array([cells[k][a] for k in keys]), np.array([cells[k][b] for k in keys])


def seed_average(cells: dict, field: str) -> dict:
    """{fold: mean over seeds}; seeds of one fold share the validation patients, so this is the conservative unit."""
    by_fold = defaultdict(list)
    for (fold, seed), r in cells.items():
        by_fold[fold].append(r[field])
    return {f: float(np.mean(v)) for f, v in by_fold.items()}


def pooled(data: dict, method: str) -> dict:
    """{(cohort, fold, seed): record} over all cohorts of one method."""
    return {(c, *k): r for c in COHORTS for k, r in data[method].get(c, {}).items()}


def fmt_p(p: float) -> str:
    return "  nan" if not np.isfinite(p) else f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def min_epochs_note(patience: int, min_epochs: int) -> str:
    """With patience P the first epoch always 'improves', so patience cannot fire before epoch P + 1."""
    if min_epochs <= patience + 1:
        return (f"min_epochs {min_epochs} never binds (patience {patience} cannot fire before epoch {patience + 1}), so the "
                f"replay is identical for min_epochs 0-{patience + 1}, which covers the outputs_v2 setting (min_epochs 6).")
    return f"min_epochs {min_epochs} delays the stop beyond the first possible epoch {patience + 1}."


# ----------------------------------------------------------------------------------------------- tables
def table_a(data: dict, methods: list[str], patience: int, min_epochs: int) -> list[str]:
    lines = [
        "=" * 118,
        "Table A. C-index under four post-hoc checkpoint-selection rules, mean over (fold, seed) runs (n), read off the "
        "stored histories",
        f"         (i) last = final epoch, the fixed budget the paper reports;  (ii) es = val_loss early stopping, patience "
        f"{patience}, min_epochs {min_epochs};",
        "         (iii) argmin = argmin val_loss over all epochs;  (iv) oracle = max val_cindex over all epochs.",
        "         ep(ii) = epoch the early-stopping rule selects (mean [min-max]); run(ii) = epochs it would have trained "
        "(mean, of budget B); stopped = folds where patience fired.",
        f"         {min_epochs_note(patience, min_epochs)}",
        CAVEAT,
        "=" * 118,
        f"{'method':24s} {'cohort':9s} {'n':>3s} {'B':>3s} {'(i) last':>9s} {'(ii) es':>9s} {'(iii) argmin':>12s} "
        f"{'(iv) oracle':>11s} | {'ep(ii)':>14s} {'run(ii)':>7s} {'stopped':>7s} | {'ep(iii)':>7s} {'ep(iv)':>6s}",
    ]
    for method in methods:
        if method not in data:
            continue
        for cohort in COHORTS + ["ALL"]:
            cells = pooled(data, method) if cohort == "ALL" else data[method].get(cohort, {})
            if not cells:
                continue
            r = list(cells.values())
            ep = [x["ep_es"] for x in r]
            lines.append(
                f"{method:24s} {cohort:9s} {len(r):3d} {max(x['epochs'] for x in r):3d} "
                f"{mean(x['c_last'] for x in r):9.3f} {mean(x['c_es'] for x in r):9.3f} "
                f"{mean(x['c_argmin'] for x in r):12.3f} {mean(x['c_oracle'] for x in r):11.3f} | "
                f"{mean(ep):6.1f} [{min(ep):2d}-{max(ep):2d}] {mean(x['run_es'] for x in r):7.1f} "
                f"{sum(x['stopped'] for x in r):3d}/{len(r):<3d} | "
                f"{mean(x['ep_argmin'] for x in r):7.1f} {mean(x['ep_oracle'] for x in r):6.1f}"
            )
        lines.append("")
    return lines


def table_b(data: dict, methods: list[str], patience: int, min_epochs: int) -> list[str]:
    lines = [
        "=" * 118,
        "Table B. Paired differences between selection rules on the same folds: delta = mean(rule - (i) last), paired-t p, "
        "Wilcoxon p, n, wins (rule > last)",
        "         left block: every (fold, seed) run is one pair (seeds of a fold share patients, so pairs are not "
        "independent);",
        "         right block: seed-averaged, one value per (cohort, fold) = mean over seeds, as scripts/seed_table.py does "
        "(25 folds pooled).",
        f"         (ii) es = val_loss early stopping, patience {patience}, min_epochs {min_epochs}; (iii) argmin val_loss; "
        "(iv) oracle is reported as a mean delta only (it is >= 0 by construction).",
        CAVEAT,
        "=" * 118,
        f"{'method':24s} {'cohort':9s} {'rule':12s} {'delta':>7s} {'t p':>7s} {'W p':>7s} {'n':>3s} {'wins':>7s}   ||   "
        f"{'delta':>7s} {'t p':>7s} {'W p':>7s} {'n':>3s} {'wins':>7s}",
    ]
    for method in methods:
        if method not in data:
            continue
        for rule, field, label in (("(ii) es", "c_es", "(ii)-(i)"), ("(iii) argmin", "c_argmin", "(iii)-(i)")):
            for cohort in ["ALL"] + COHORTS:
                cells = pooled(data, method) if cohort == "ALL" else data[method].get(cohort, {})
                if not cells:
                    continue
                x, y = pairs(cells, field, "c_last")
                d, t, w, n, wins = paired(x, y)
                if cohort == "ALL":
                    sa_x, sa_y = {}, {}
                    for c in COHORTS:
                        sa_x.update({(c, f): v for f, v in seed_average(data[method].get(c, {}), field).items()})
                        sa_y.update({(c, f): v for f, v in seed_average(data[method].get(c, {}), "c_last").items()})
                else:
                    sa_x, sa_y = seed_average(cells, field), seed_average(cells, "c_last")
                keys = sorted(sa_x)
                d2, t2, w2, n2, wins2 = paired(np.array([sa_x[k] for k in keys]), np.array([sa_y[k] for k in keys]))
                lines.append(
                    f"{method:24s} {cohort:9s} {label:12s} {d:+7.3f} {fmt_p(t):>7s} {fmt_p(w):>7s} {n:3d} {wins:3d}/{n:<3d}   ||   "
                    f"{d2:+7.3f} {fmt_p(t2):>7s} {fmt_p(w2):>7s} {n2:3d} {wins2:3d}/{n2:<3d}"
                )
            if rule == "(iii) argmin":
                cells = pooled(data, method)
                x, y = pairs(cells, "c_oracle", "c_last")
                lines.append(f"{method:24s} {'ALL':9s} {'(iv)-(i)':12s} {float(np.mean(x - y)):+7.3f} {'-':>7s} {'-':>7s} {len(x):3d} "
                             f"{int(np.sum(x > y)):3d}/{len(x):<3d}   ||   (mean only; oracle >= last by construction)")
        lines.append("")
    return lines


def table_c(root: str) -> tuple[list[str], dict]:
    """Patients and events per validation fold, per cohort, from every fold_results.json under one campaign root.

    Returns the lines and {cohort | 'all': (min, median, max) of events_val over folds}.
    """
    seen: dict = defaultdict(set)
    for fr in sorted(Path(root).glob("*/*/fold_*/fold_results.json")):
        cohort = fr.parts[-3]
        if cohort not in COHORTS:
            continue
        with open(fr) as f:
            d = json.load(f)
        seen[(cohort, int(d["fold"]))].add((int(d["n_val"]), int(d["events_val"])))
    lines = [
        "=" * 118,
        f"Table C. Patients (n_val) and disease-specific events (events_val) per validation fold, from every run under {root}",
        "         The 5-fold split is fixed per cohort (same patients for every method and seed); min/median/max are "
        "over the 5 folds.",
        "=" * 118,
        f"{'cohort':9s} {'n_val per fold (0..4)':>24s} {'events_val per fold (0..4)':>28s} {'min':>4s} {'median':>7s} "
        f"{'max':>4s} {'sum n':>6s} {'sum ev':>6s} {'ev/n':>5s} {'min n_val':>9s}",
    ]
    ev_stats: dict = {}
    all_events: list[int] = []
    for cohort in COHORTS:
        folds = sorted(k[1] for k in seen if k[0] == cohort)
        if not folds:
            continue
        n_val, events = [], []
        for k in folds:
            vals = sorted(seen[(cohort, k)])
            if len(vals) > 1:
                lines.append(f"  NOTE {cohort} fold {k}: inconsistent (n_val, events_val) across runs: {vals}; using the first")
            n_val.append(vals[0][0])
            events.append(vals[0][1])
        all_events += events
        ev_stats[cohort] = (min(events), float(np.median(events)), max(events))
        lines.append(
            f"{cohort:9s} {' '.join(f'{v:4d}' for v in n_val):>24s} {' '.join(f'{v:4d}' for v in events):>28s} "
            f"{min(events):4d} {float(np.median(events)):7.1f} {max(events):4d} {sum(n_val):6d} {sum(events):6d} "
            f"{sum(events) / sum(n_val):5.2f} {min(n_val):9d}"
        )
    if all_events:
        ev_stats["all"] = (min(all_events), float(np.median(all_events)), max(all_events))
        lines.append(f"{'all':9s} {'':>24s} {'':>28s} {min(all_events):4d} {float(np.median(all_events)):7.1f} "
                     f"{max(all_events):4d}   (over {len(all_events)} cohort-folds)")
    lines.append("")
    return lines, ev_stats


def collect_v2(root: str, methods: list[str], patience: int, min_epochs: int):
    """The actual val_loss-selected campaign: {method: {cohort: {(fold, seed): record}}}, replayed with the run's own
    (patience, min_epochs) from results.json so the replay can be checked against the stored selection."""
    data: dict = defaultdict(lambda: defaultdict(dict))
    cfgs: dict = defaultdict(int)
    for rd in sorted(Path(root).iterdir()):
        if not rd.is_dir():
            continue
        m = RUN_PAT.match(rd.name)
        method, seed = m.group("m"), int(m.group("s") or 0)
        if method not in methods:
            continue
        for cd in sorted(rd.iterdir()):
            if not cd.is_dir() or cd.name not in COHORTS:
                continue
            cfg = {}
            if (cd / "results.json").exists():
                with open(cd / "results.json") as f:
                    cfg = json.load(f).get("config", {})
            p, me = int(cfg.get("patience", patience)), int(cfg.get("min_epochs", min_epochs))
            cfgs[(str(cfg.get("selection_metric")), p, me, int(cfg.get("epochs", 0)))] += 1
            for fr in sorted(cd.glob("fold_*/fold_results.json")):
                with open(fr) as f:
                    d = json.load(f)
                r = fold_record(d, p, me)
                if r is None:
                    continue
                r["seed"] = seed
                r["epochs_run"] = int(d.get("epochs_run", r["epochs"]))
                r["last_any"] = float(d.get("last_epoch_val_cindex", r["c_last"]))
                r["sel_actual"] = int(d.get("selected_epoch", -1))
                r["sel_ok"] = r["ep_es"] == r["sel_actual"] and abs(r["c_reported"] - r["c_es"]) < 1e-9
                r["run_ok"] = r["run_es"] == r["epochs_run"]
                data[method][cd.name][(r["fold"], seed)] = r
    return data, cfgs


def table_d(root: str, methods: list[str], patience: int, min_epochs: int) -> tuple[list[str], dict]:
    data, cfgs = collect_v2(root, methods, patience, min_epochs)
    cfg_txt = "; ".join(f"selection={k[0]}, patience={k[1]}, min_epochs={k[2]}, epochs={k[3]} ({v} cohort runs)"
                        for k, v in sorted(cfgs.items()))
    recs = [(m, c, k, r) for m in methods if m in data for c in COHORTS for k, r in sorted(data[m].get(c, {}).items())]
    sel_ok = sum(r["sel_ok"] for *_, r in recs)
    run_bad = [f"{m}/{c}/fold_{k[0]}" + (f"/seed{k[1]}" if k[1] else "") + f" (trained {r['epochs_run']}, replay stops at {r['run_es']})"
               for m, c, k, r in recs if not r["run_ok"]]
    lines = [
        "=" * 118,
        f"Table D. The ACTUAL val_loss-selected campaign under {root} (the first multi-cohort campaign), per method x "
        "cohort, mean over (fold, seed) runs (n)",
        f"         run configs: {cfg_txt}",
        "         c_index = reported (checkpoint selected by val_loss, scored on the same fold); sel.ep = stored "
        "selected_epoch (mean [min-max]); run = epochs_run (mean);",
        "         last = val C-index at the last epoch trained; oracle = best_val_cindex_any_epoch; seeds = run seeds "
        "present (BLCA has only the _seed1/_seed2 runs in this archive).",
        f"         replay check: the early-stopping replay with each run's own (patience, min_epochs) reproduces the stored "
        f"selected_epoch and reported c_index in {sel_ok}/{len(recs)} folds and epochs_run in "
        f"{len(recs) - len(run_bad)}/{len(recs)}" + (f"; exceptions: {', '.join(run_bad)}" if run_bad else "") + ".",
        CAVEAT,
        "=" * 118,
        f"{'method':24s} {'cohort':9s} {'n':>3s} {'seeds':>5s} {'c_index':>8s} {'sel.ep':>14s} {'run':>5s} {'last':>7s} "
        f"{'oracle':>7s} {'oracle-c':>8s}",
    ]
    for method in methods:
        if method not in data:
            continue
        for cohort in COHORTS + ["ALL"]:
            cells = pooled(data, method) if cohort == "ALL" else data[method].get(cohort, {})
            if not cells:
                continue
            r = list(cells.values())
            ep = [x["sel_actual"] for x in r]
            seeds = ",".join(str(s) for s in sorted({x["seed"] for x in r}))
            lines.append(
                f"{method:24s} {cohort:9s} {len(r):3d} {seeds:>5s} {mean(x['c_reported'] for x in r):8.3f} "
                f"{mean(ep):6.1f} [{min(ep):2d}-{max(ep):2d}] {mean(x['epochs_run'] for x in r):5.1f} "
                f"{mean(x['last_any'] for x in r):7.3f} {mean(x['best_any'] for x in r):7.3f} "
                f"{mean(x['best_any'] - x['c_reported'] for x in r):+8.3f}"
            )
        lines.append("")
    return lines, data


# ----------------------------------------------------------------------------------------------- summary
def summary(data: dict, methods: list[str], v2: dict, ev_stats: dict, patience: int, min_epochs: int,
            band: tuple[float, float]) -> list[str]:
    lo, hi = band
    budgets = {m: max(r["epochs"] for c in data[m].values() for r in c.values()) for m in methods if m in data}
    b20 = [m for m in methods if budgets.get(m) == 20]
    b10 = [m for m in methods if m in budgets and budgets[m] != 20]
    recs20 = [r for m in b20 for c in data[m].values() for r in c.values()]
    ep20 = np.array([r["ep_es"] for r in recs20])
    run20 = np.array([r["run_es"] for r in recs20])

    def cell(m: str, c: str, field: str) -> float:
        return mean(r[field] for r in data[m].get(c, {}).values())

    def pooled_delta(m: str) -> tuple[float, float, int]:
        x, y = pairs(pooled(data, m), "c_es", "c_last")
        d, _, w, n, _ = paired(x, y)
        return d, w, n

    def tag(m: str) -> str:
        d, w, n = pooled_delta(m)
        return f"{m} {d:+.3f} (W p={fmt_p(w).strip()}, n={n})"

    by_delta = sorted(b20, key=lambda m: pooled_delta(m)[0])
    sig = [m for m in by_delta if pooled_delta(m)[0] < 0 and pooled_delta(m)[1] < 0.05]
    nil = [m for m in by_delta if m not in sig]
    n_cells = sum(len(data[m]) for m in b20)
    in_band = [(m, c) for m in b20 for c in COHORTS if c in data[m] and lo <= cell(m, c, "c_es") <= hi]
    in_band_last = [(m, c) for m in b20 for c in COHORTS if c in data[m] and lo <= cell(m, c, "c_last") <= hi]
    below = [(m, c) for m in b20 for c in COHORTS if c in data[m] and cell(m, c, "c_es") < lo]
    per_cohort = {c: (sum(1 for m in b20 if c in data[m] and cell(m, c, "c_es") <= hi), sum(1 for m in b20 if c in data[m]))
                  for c in COHORTS}
    exceptions = [(m, c, cell(m, c, "c_es")) for c in CLAIM_COHORTS for m in b20 if c in data[m] and cell(m, c, "c_es") > hi]
    peak = lambda ms: mean(r["ep_oracle"] for m in ms for c in data[m].values() for r in c.values())  # noqa: E731

    s = (
        f"Replaying validation-loss early stopping (patience {patience}, min_epochs {min_epochs}) on the {len(recs20)} stored "
        f"fold histories of the {len(b20)} fixed-budget 20-epoch methods ({', '.join(b20)}) selects epoch "
        f"{float(np.median(ep20)):.0f} on median (mean {ep20.mean():.1f}; {100 * np.mean(ep20 <= 3):.0f}% of folds select "
        f"epoch 1-3) and would have stopped training after {run20.mean():.1f} of 20 epochs on average "
        f"({100 * np.mean([r['stopped'] for r in recs20]):.0f}% of folds hit the patience limit). "
    )
    s += (
        f"Pooled over the five cohorts, the early-stopped checkpoint scores significantly below the final-epoch checkpoint "
        f"the paper reports (Wilcoxon p < 0.05 over (fold, seed) pairs) for {len(sig)} of {len(b20)} methods: "
        + "; ".join(tag(m) for m in sig) + ". "
    )
    if nil:
        s += (
            f"The deficit is absent for {', '.join(tag(m) for m in nil)}; these reach their best validation C-index at epoch "
            f"{peak(nil):.1f} on average versus {peak(sig):.1f} for the others (ep(iv) in Table A), so an epoch-1-2 checkpoint "
            "costs them little. "
        )
    s += (
        f"Counting a cell as near chance when its mean C-index under rule (ii) lies within [{lo:.2f}, {hi:.2f}], "
        f"{len(in_band)} of {n_cells} method x cohort cells qualify"
        + (f" and {len(below)} fall below {lo:.2f}" if below else "")
        + f", against {len(in_band_last)} under the reported rule (i); at or below {hi:.2f} per cohort: "
        + ", ".join(f"{c.upper()} {k}/{n}" for c, (k, n) in per_cohort.items()) + ". "
    )
    s += (
        "So on the three cohorts the manuscript names, the replay "
        + ("supports 'every method' near chance. " if not exceptions else
           "does NOT support 'every method' near chance literally: the exceptions are "
           + ", ".join(f"{m}/{c} ({v:.3f})" for m, c, v in exceptions) + ". ")
    )
    if b10:
        recs10 = [r for m in b10 for c in data[m].values() for r in c.values()]
        s += (
            f"The 10-epoch runs ({', '.join(b10)}; {len(recs10)} folds) behave the same way: median selected epoch "
            f"{float(np.median([r['ep_es'] for r in recs10])):.0f}, pooled delta " + ", ".join(tag(m) for m in b10) + ". "
        )
    if v2:
        vr = [r for m in v2.values() for c in m.values() for r in c.values()]
        vep = np.array([r["sel_actual"] for r in vr])
        vcells = {(m, c): mean(r["c_reported"] for r in v2[m][c].values()) for m in V2_METHODS if m in v2 for c in COHORTS if c in v2[m]}
        v_claim = {c: (sum(1 for (m, cc), v in vcells.items() if cc == c and v <= hi), sum(1 for (m, cc) in vcells if cc == c))
                   for c in CLAIM_COHORTS}
        v_exc = [(m, c, v) for c in CLAIM_COHORTS for (m, cc), v in vcells.items() if cc == c and v > hi]
        s += (
            f"The actual val_loss campaign (Table D; patience 5, min_epochs 6; {len(vr)} folds) selected epoch "
            f"{float(np.median(vep)):.0f} on median (mean {vep.mean():.1f}; {100 * np.mean(vep <= 3):.0f}% of folds epoch 1-3), "
            f"trained {mean(r['epochs_run'] for r in vr):.1f} epochs on average, and reported {sum(lo <= v <= hi for v in vcells.values())} "
            f"of {len(vcells)} method x cohort cells within [{lo:.2f}, {hi:.2f}] (mean reported C-index "
            f"{mean(vcells.values()):.3f} vs oracle {mean(r['best_any'] for r in vr):.3f}); at or below {hi:.2f}: "
            + ", ".join(f"{c.upper()} {k}/{n}" for c, (k, n) in v_claim.items())
            + (", the exceptions being " + ", ".join(f"{m}/{c} ({v:.3f})" for m, c, v in v_exc) if v_exc else "") + ". "
        )
    if ev_stats:
        s += (
            "Events per validation fold (Table C): "
            + ", ".join(f"{c.upper()} {ev_stats[c][0]}-{ev_stats[c][2]}" for c in CLAIM_COHORTS if c in ev_stats)
            + (f"; all cohorts {ev_stats['all'][0]}-{ev_stats['all'][2]}, median {ev_stats['all'][1]:.0f}" if "all" in ev_stats else "")
            + ". "
        )
    s += (
        "All of this selects and scores on the same fold, so rules (ii)-(iv) are biased in favour of selection, which makes "
        "the deficit of rule (ii) a conservative estimate of what early stopping costs at these event counts."
    )
    return ["=" * 118, "Summary (plain language; every number is a deterministic function of the archived histories)", "=" * 118,
            *textwrap.wrap(s, width=118), ""]


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["pod_results/outputs_e20", "pod_results/outputs_v2", "pod_results/outputs_ablate"],
                    help="fixed-budget campaign roots, in order of precedence for duplicate (method, seed, cohort)")
    ap.add_argument("--methods", nargs="+", default=E20_METHODS + E10_METHODS)
    ap.add_argument("--events-root", default=None, help="root for Table C (default: the first of --roots)")
    ap.add_argument("--v2-root", default="pod_results/outputs_v2", help="the actual val_loss-selected campaign (Table D)")
    ap.add_argument("--v2-methods", nargs="+", default=V2_METHODS)
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience replayed in rule (ii)")
    ap.add_argument("--min-epochs", type=int, default=0, help="min_epochs replayed in rule (ii); outputs_v2 used 6")
    ap.add_argument("--chance-band", type=float, nargs=2, default=(0.45, 0.55), metavar=("LO", "HI"))
    ap.add_argument("--out", default="results/final/table_selection_rules.txt", help="'-' = stdout only")
    args = ap.parse_args()

    data, notes = collect(args.roots, args.methods, args.patience, args.min_epochs)
    if not data:
        raise SystemExit("no fold histories found under " + ", ".join(args.roots))
    n_folds = sum(len(c) for m in data.values() for c in m.values())
    lines = [
        "Post-hoc checkpoint-selection analysis (scripts/posthoc_selection.py) from the stored per-epoch histories; no training.",
        f"roots: {', '.join(args.roots)}   methods: {', '.join(m for m in args.methods if m in data)}   "
        f"fold histories: {n_folds}   rule (ii): patience {args.patience}, min_epochs {args.min_epochs}",
        "In the 5-fold CV the validation fold is the reported held-out fold (c_index == last_epoch_val_cindex), so any "
        "post-hoc selection on val_loss or val_cindex is evaluated on the fold it selects on.",
    ]
    if notes:
        lines += ["notes:"] + [f"  {n}" for n in notes]
    lines.append("")
    lines += table_a(data, args.methods, args.patience, args.min_epochs)
    lines += table_b(data, args.methods, args.patience, args.min_epochs)
    c_lines, ev_stats = table_c(args.events_root or args.roots[0])
    lines += c_lines
    d_lines, v2 = table_d(args.v2_root, args.v2_methods, args.patience, args.min_epochs) if Path(args.v2_root).is_dir() else ([], {})
    lines += d_lines
    lines += summary(data, args.methods, v2, ev_stats, args.patience, args.min_epochs, tuple(args.chance_band))

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out != "-":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"written: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
