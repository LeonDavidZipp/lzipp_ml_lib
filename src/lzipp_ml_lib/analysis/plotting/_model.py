from typing import Literal

import numpy as np
import xgboost as xgb
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._style import (
    ACCENT,
    BASELINE,
    CATEGORICAL,
    INK_MUTED,
    INK_SECONDARY,
    figure_and_axes,
    panel_grid,
    styled,
)


@styled
def plot_feature_importance(
    model: xgb.XGBModel | xgb.Booster,
    *,
    importance_type: Literal[
        "gain", "total_gain", "weight", "cover", "total_cover"
    ] = "gain",
    top_k: int = 20,
    ax: Axes | None = None,
) -> Figure:
    """Plots each feature's share of the model's total importance, largest on top.
    Features the model never splits on are counted in the title.

    Args:
        model (xgb.XGBModel | xgb.Booster): The fitted model.
        importance_type (Literal["gain", "total_gain", "weight", "cover",
            "total_cover"]): How importance is measured. "gain" is the average loss
            reduction of a feature's splits: how useful it is when used. "weight" counts
            how often it's split on, which favours features with many distinct values
            whether or not they help. Defaults to "gain".
        top_k (int): The number of features shown; the rest are folded into one "other"
            bar. Defaults to 20.
        ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its look
            is adapted to the style. If None, a new figure is created. Defaults to None.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If the model has no splits.
    """
    booster = model.get_booster() if isinstance(model, xgb.XGBModel) else model
    # multi-output models score a feature per output; their sum ranks it overall
    scores = {
        name: float(np.sum(value))
        for name, value in booster.get_score(importance_type=importance_type).items()
    }
    if not scores:
        raise ValueError("the model has no splits, so no feature importance")
    n_features = len(booster.feature_names or []) or len(scores)
    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > top_k:
        rest = ranked[top_k:]
        ranked = ranked[:top_k] + [(f"other ({len(rest)})", sum(v for _, v in rest))]

    # reversed, so the most important bar ends up on top
    names = [name for name, _ in ranked][::-1]
    shares = [value / total for _, value in ranked][::-1]
    colors = [INK_MUTED if n.startswith("other (") else ACCENT for n in names]
    fig, ax = figure_and_axes(ax, (8, 0.32 * len(names) + 1.2))
    bars = ax.barh(names, shares, height=0.6, color=colors)  # type: ignore
    ax.bar_label(  # type: ignore
        bars,
        labels=[f"{s:.1%}" for s in shares],
        padding=4,
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_xlim(0, max(shares) * 1.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")  # type: ignore
    ax.grid(axis="y", visible=False)  # type: ignore
    ax.grid(axis="x", visible=True)  # type: ignore
    ax.margins(y=0.02)  # type: ignore
    unused = n_features - len(scores)
    title = f"Feature importance ({importance_type})"
    if unused > 0:
        title += f"  ·  {unused} of {n_features} features unused"
    ax.set_title(title)  # type: ignore
    return fig


@styled
def plot_learning_curves(model: xgb.XGBModel) -> Figure:
    """Plots each evaluation metric per boosting round, one panel per metric and one
    line per eval set, with the best round marked if early stopping was on. A training
    curve that keeps falling while the validation one turns up is overfitting.

    Args:
        model (xgb.XGBModel): A model fit with an `eval_set`, e.g. `[(x_train, y_train),
            (x_val, y_val)]`. The `fit_xgb_*` functions' final models are fit without
            one; fit one yourself with their best parameters.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If the model has no evaluation results.
    """
    try:
        results = model.evals_result()
    except xgb.core.XGBoostError:
        results = {}
    if not results:
        raise ValueError(
            "the model has no evaluation results: fit it with an eval_set "
            "(the fit_xgb_* final models are fit without one)"
        )
    sets = list(results)
    metrics = list(results[sets[0]])
    best = getattr(model, "best_iteration", None)

    fig, axes = panel_grid(len(metrics), n_cols=2, panel_size=(6, 3.4))
    for ax, metric in zip(axes, metrics, strict=True):
        for i, name in enumerate(sets):
            ax.plot(  # type: ignore
                results[name][metric],
                color=CATEGORICAL[i % len(CATEGORICAL)],
                linewidth=1.5,
                label=f"eval_set[{i}]" if name.startswith("validation_") else name,
            )
        title = metric
        if best is not None:
            # the round goes in the title: inside the plot, some curve could cover it
            ax.axvline(best, color=BASELINE, linewidth=1, zorder=1)  # type: ignore
            title += f"  ·  best round {best}"
        ax.set_title(title)  # type: ignore
        ax.set_xlabel("Boosting round")  # type: ignore
        ax.margins(x=0)  # type: ignore
        if len(sets) > 1:
            ax.legend(loc="upper right")  # type: ignore
    return fig
