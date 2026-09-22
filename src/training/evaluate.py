"""Survival evaluation: concordance (Harrell / IPCW), Brier score & IBS, time-dependent AUC,
bootstrap confidence intervals and Kaplan-Meier log-rank stratification.

The protocol mirrors SurvPath's ``_calculate_metrics`` so numbers are comparable to the
literature; every optional metric degrades to NaN with a *visible* warning instead of a silent 0.
"""

from __future__ import annotations

import warnings

import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index as _lifelines_cindex
from scipy import stats

try:  # scikit-survival has no wheels for some very new Python versions
    from sksurv.metrics import (
        brier_score,
        concordance_index_censored,
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )
    from sksurv.util import Surv

    HAS_SKSURV = True
except ImportError:  # pragma: no cover
    HAS_SKSURV = False


# ---------------------------------------------------------------------------------------
# Concordance
# ---------------------------------------------------------------------------------------
def concordance(event, time, risk) -> float:
    """Harrell's C-index on *continuous* time. ``risk``: higher = worse prognosis."""
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if event.sum() == 0 or len(np.unique(risk)) == 1:
        return float("nan")
    try:
        if HAS_SKSURV:
            return float(concordance_index_censored(event, time, risk, tied_tol=1e-8)[0])
        return float(_lifelines_cindex(time, -risk, event))
    except Exception:  # noqa: BLE001  e.g. sksurv NoComparablePairException in tiny / event-poor (re)samples
        return float("nan")


def bootstrap_cindex_ci(event, time, risk, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the C-index over patients. Returns (lo, hi, se)."""
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(event)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = concordance(event[idx], time[idx], risk[idx])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")


# ---------------------------------------------------------------------------------------
# Time-dependent metrics (need scikit-survival)
# ---------------------------------------------------------------------------------------
def _bin_of(times, edges) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges, float), np.asarray(times, float), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def interior_edges(edges) -> np.ndarray:
    return np.array([e for e in np.asarray(edges, float)[1:-1] if np.isfinite(e)], dtype=float)


def eval_times(train_time, val_time, edges) -> np.ndarray:
    """Grid for Brier / IBS / td-AUC: the interior bin edges, i.e. the quartiles of the uncensored
    training event times, restricted to where both the test follow-up and the training censoring
    distribution are defined.

    SurvPath additionally evaluates at val_min and val_max. At val_max typically one patient is still
    at risk (BLCA fold 0: AUC 0.13 there vs 0.62-0.76 at the quartiles), which made the mean td-AUC
    and the IBS degenerate, so those two points are deliberately not used.
    """
    train_time = np.asarray(train_time, float)
    val_time = np.asarray(val_time, float)
    lo = max(val_time.min(), train_time.min()) + 1e-4
    hi = min(val_time.max(), train_time.max()) - 1e-4
    grid = interior_edges(edges)
    return np.unique(grid[(grid >= lo) & (grid <= hi)])


def ipcw_horizon(train_time, edges) -> float:
    """Truncation time for Uno's IPCW C-index: the last interior edge (75th percentile of training
    event times). With ~70 % censoring the censoring survival G(t) falls to ~0.03 by 120 months and the
    1/G^2 weights explode; truncating keeps them bounded (G ~ 0.5 at the 75th percentile)."""
    inner = interior_edges(edges)
    tau = float(inner[-1]) if len(inner) else float(np.asarray(train_time, float).max())
    return min(tau, float(np.asarray(train_time, float).max()))


def survival_metrics(train_event, train_time, event, time, risk, survival_by_bin, edges) -> dict:
    """C-index, IPCW C-index, Brier scores, IBS and mean time-dependent AUC for one split.

    Args:
        train_event/train_time: training split (for the IPCW censoring estimate)
        event/time/risk: validation split; ``risk`` higher = worse
        survival_by_bin: (N, T) predicted S_t per bin
        edges: T+1 bin edges (outer edges may be +-inf)
    """
    out = {"c_index": concordance(event, time, risk)}
    if not HAS_SKSURV:
        warnings.warn("scikit-survival not installed: IPCW C-index, Brier/IBS and td-AUC skipped")
        return out

    train_event = np.asarray(train_event, bool)
    train_time = np.asarray(train_time, float)
    event = np.asarray(event, bool)
    time = np.asarray(time, float)
    risk = np.asarray(risk, float)
    S = np.asarray(survival_by_bin, float)

    surv_train = Surv.from_arrays(event=train_event, time=train_time)
    surv_test = Surv.from_arrays(event=event, time=time)

    tau = ipcw_horizon(train_time, edges)
    out["ipcw_tau"] = tau
    try:
        out["c_index_ipcw"] = float(concordance_index_ipcw(surv_train, surv_test, risk, tau=tau)[0])
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"IPCW C-index failed: {exc}")
        out["c_index_ipcw"] = float("nan")

    times = eval_times(train_time, time, edges)
    out["eval_times"] = [float(t) for t in times]
    out.update(brier=[], auc=[], ibs=float("nan"), iauc=float("nan"))
    if len(times) == 0:
        return out
    S_at = S[:, _bin_of(times, edges)]  # step-function evaluation of the discrete model

    try:
        _, bs = brier_score(surv_train, surv_test, S_at, times)
        out["brier"] = [float(b) for b in bs]
        if len(times) >= 2:
            out["ibs"] = float(integrated_brier_score(surv_train, surv_test, S_at, times))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Brier/IBS failed: {exc}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            aucs, iauc = cumulative_dynamic_auc(surv_train, surv_test, 1.0 - S_at, times)
        out["auc"] = [float(a) for a in np.atleast_1d(aucs)]
        out["iauc"] = float(iauc)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"time-dependent AUC failed: {exc}")
    return out


# ---------------------------------------------------------------------------------------
# Stratification and aggregation
# ---------------------------------------------------------------------------------------
def km_logrank_split(event, time, risk) -> dict:
    """Median-risk split into high/low groups; log-rank p-value between the two KM curves."""
    event = np.asarray(event, bool)
    time = np.asarray(time, float)
    risk = np.asarray(risk, float)
    high = risk > np.median(risk)
    if high.sum() == 0 or (~high).sum() == 0:
        return {"logrank_p": float("nan"), "n_high": int(high.sum()), "n_low": int((~high).sum())}
    res = logrank_test(time[high], time[~high], event_observed_A=event[high], event_observed_B=event[~high])
    return {"logrank_p": float(res.p_value), "n_high": int(high.sum()), "n_low": int((~high).sum())}


def mean_ci95(values) -> dict:
    """Mean, sample std and 95% t-interval across folds (NaNs ignored)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    if len(v) > 1:
        half = stats.t.ppf(0.975, len(v) - 1) * std / np.sqrt(len(v))
        ci = [mean - half, mean + half]
    else:
        ci = [mean, mean]
    return {"mean": mean, "std": std, "ci95": [float(ci[0]), float(ci[1])], "n": int(len(v))}


def paired_test(a, b) -> dict:
    """Paired t-test and Wilcoxon signed-rank over folds (a vs b)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return {"t_p": float("nan"), "wilcoxon_p": float("nan"), "n": int(len(a))}
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = float("nan")
    return {"t_p": t_p, "wilcoxon_p": w_p, "n": int(len(a))}


# Backwards-compatible name used by the old training loop.
def compute_cindex(risk_scores, survival_times, censorships) -> float:
    return concordance(1 - np.asarray(censorships), survival_times, risk_scores)
