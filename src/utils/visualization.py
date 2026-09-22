"""Visualization utilities: attention heatmaps, Kaplan-Meier curves, pathway importance."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def plot_kaplan_meier(
    risk_scores: np.ndarray,
    survival_times: np.ndarray,
    events: np.ndarray,
    n_groups: int = 2,
    save_path: str | Path | None = None,
    title: str = "Kaplan-Meier Survival Curves",
) -> plt.Figure:
    """
    Plot KM curves stratified by predicted risk groups.

    Args:
        risk_scores: (N,) predicted risk
        survival_times: (N,) observed survival times
        events: (N,) 1 if event occurred, 0 if censored
        n_groups: number of risk groups to stratify into
        save_path: optional path to save figure
        title: plot title
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    quantiles = np.linspace(0, 1, n_groups + 1)
    thresholds = np.quantile(risk_scores, quantiles)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    labels = []

    for i in range(n_groups):
        mask = (risk_scores >= thresholds[i]) & (risk_scores < thresholds[i + 1])
        if i == n_groups - 1:
            mask = risk_scores >= thresholds[i]

        if mask.sum() == 0:
            continue

        kmf = KaplanMeierFitter()
        label = f"Group {i+1} (n={mask.sum()})"
        kmf.fit(survival_times[mask], event_observed=events[mask], label=label)
        kmf.plot_survival_function(ax=ax)
        labels.append(label)

    if n_groups == 2:
        low_mask = risk_scores < np.median(risk_scores)
        high_mask = ~low_mask
        result = logrank_test(
            survival_times[low_mask], survival_times[high_mask],
            events[low_mask], events[high_mask],
        )
        ax.set_title(f"{title}\nLog-rank p={result.p_value:.4f}")
    else:
        ax.set_title(title)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pathway_importance(
    pathway_names: list[str],
    importance_scores: np.ndarray,
    top_k: int = 20,
    save_path: str | Path | None = None,
    title: str = "Top Pathway Importance",
) -> plt.Figure:
    """Bar plot of top-K most important pathways by attention weight."""
    sorted_idx = np.argsort(importance_scores)[::-1][:top_k]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    names = [pathway_names[i] for i in sorted_idx]
    scores = importance_scores[sorted_idx]

    ax.barh(range(len(names)), scores[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Mean Attention Weight")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
