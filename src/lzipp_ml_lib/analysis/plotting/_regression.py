import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._style import (
    ACCENT,
    BASELINE,
    CATEGORICAL,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL,
    SURFACE,
    date_axis,
    panel_grid,
    styled,
)
from ._utils import Values, as_1d, same_length

# past this many points a scatter is one blob; hexbin shows where they pile up
_HEXBIN_FROM = 20_000


@styled
def plot_regression_diagnostics(y_true: Values, y_pred: Values) -> Figure:
    """Plots three views of a regression model's errors, side by side. Predicted vs
    actual (with R², MAE and RMSE): points off the diagonal are errors. Residuals
    (actual - predicted) vs predicted, with their binned mean: a curve or funnel means
    the errors depend on the prediction's size, so the model misses a shape or the
    target wants a transform (e.g. log). The residual distribution: a shifted centre is
    bias, heavy tails are outliers. Above 20,000 points, scatters become hexbins.

    Args:
        y_true (Values): The true labels or values, as a list, array, Series or
            single-column DataFrame (like the `y` frames the fit functions take).
        y_pred (Values): The predicted values, in the same forms as `y_true`.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If the inputs differ in length.
    """
    actual = as_1d(y_true, "y_true").astype(float)
    predicted = as_1d(y_pred, "y_pred").astype(float)
    same_length(y_true=actual, y_pred=predicted)
    residual = actual - predicted

    fig, (left, middle, right) = panel_grid(3, n_cols=3, panel_size=(4.6, 4.2))
    _draw_predicted_vs_actual(left, actual, predicted)
    _draw_residuals_vs_predicted(middle, predicted, residual)
    _draw_residual_distribution(right, residual)
    return fig


def _draw_points(ax: Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) > _HEXBIN_FROM:
        ax.hexbin(x, y, gridsize=60, cmap=SEQUENTIAL, mincnt=1, linewidths=0)  # type: ignore
    else:
        alpha = float(np.clip(2_000 / max(len(x), 1), 0.15, 0.8))
        ax.scatter(x, y, s=10, color=ACCENT, alpha=alpha, linewidths=0)  # type: ignore


def _draw_predicted_vs_actual(
    ax: Axes, actual: np.ndarray, predicted: np.ndarray
) -> None:
    _draw_points(ax, actual, predicted)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    pad = (high - low) * 0.03
    ax.plot([low, high], [low, high], color=INK_MUTED, linewidth=1, zorder=1)  # type: ignore
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal")
    ax.grid(axis="x", visible=True)  # type: ignore
    residual = actual - predicted
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - (residual**2).sum() / ss_tot if ss_tot else float("nan")
    ax.text(  # type: ignore
        0.03,
        0.97,
        f"R² {r2:.3f}\nMAE {np.abs(residual).mean():.4g}\n"
        f"RMSE {np.sqrt((residual**2).mean()):.4g}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=INK_SECONDARY,
    )
    ax.set_title("Predicted vs actual")  # type: ignore
    ax.set_xlabel("Actual")  # type: ignore
    ax.set_ylabel("Predicted")  # type: ignore


def _draw_residuals_vs_predicted(
    ax: Axes, predicted: np.ndarray, residual: np.ndarray
) -> None:
    _draw_points(ax, predicted, residual)
    ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=1)  # type: ignore
    binned = (
        pl.DataFrame({"pred": predicted, "res": residual})
        .with_columns(
            bin=pl.col("pred").bin_quantiles(
                int(np.clip(len(predicted) // 25, 1, 20)),
                labels=False,
                right_closed=True,
            )
        )
        .group_by("bin")
        .agg(x=pl.col("pred").median(), y=pl.col("res").mean())
        .sort("x")
    )
    ax.plot(  # type: ignore
        binned["x"],
        binned["y"],
        color=CATEGORICAL[1],
        linewidth=2,
        label="binned mean",
    )
    ax.legend(loc="upper right")  # type: ignore
    ax.grid(axis="x", visible=True)  # type: ignore
    ax.set_title("Residuals vs predicted")  # type: ignore
    ax.set_xlabel("Predicted")  # type: ignore
    ax.set_ylabel("Actual − predicted")  # type: ignore


def _draw_residual_distribution(ax: Axes, residual: np.ndarray) -> None:
    ax.hist(residual, bins="auto", color=ACCENT, edgecolor=SURFACE, linewidth=0.8)  # type: ignore
    ax.axvline(0, color=INK_MUTED, linewidth=1)  # type: ignore
    ax.set_title(  # type: ignore
        f"Residuals  ·  mean {residual.mean():.3g}  ·  std {residual.std():.3g}"
    )
    ax.set_xlabel("Actual − predicted")  # type: ignore
    ax.set_ylabel("Count")  # type: ignore


def _break_at_gaps(x: np.ndarray, *series: np.ndarray) -> tuple[np.ndarray, ...]:
    """Insert a NaN point into every gap of more than twice the usual step, so
    lines break there instead of bridging missing data with a straight line."""
    if len(x) < 3 or not np.issubdtype(x.dtype, np.datetime64):
        return (x, *series)
    steps = np.diff(x)
    gaps = np.flatnonzero(steps > 2 * np.median(steps)) + 1
    if not len(gaps):
        return (x, *series)
    x = np.insert(x, gaps, x[gaps - 1] + steps[gaps - 1] // 2)
    return (x, *(np.insert(s.astype(float), gaps, np.nan) for s in series))


@styled
def plot_forecast(
    y_true: Values,
    y_pred: Values,
    ts: Values | None = None,
) -> Figure:
    """Plots actual and predicted values over time, with the residual below. Long
    stretches where the residual stays on one side of 0 are something the model
    systematically misses, like a level shift or a holiday. Lines break where timestamps
    are missing.

    Args:
        y_true (Values): The true labels or values, as a list, array, Series or
            single-column DataFrame (like the `y` frames the fit functions take).
        y_pred (Values): The predicted values, in the same forms as `y_true`.
        ts (Values | None): The timestamps of the values (e.g. the test set's `ts`
            column); the values are sorted by them. If None, the x axis is the row
            number. Defaults to None.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If the inputs differ in length.
    """
    actual = as_1d(y_true, "y_true").astype(float)
    predicted = as_1d(y_pred, "y_pred").astype(float)
    same_length(y_true=actual, y_pred=predicted)
    if ts is not None:
        x = as_1d(ts, "ts")
        same_length(y_true=actual, ts=x)
        order = np.argsort(x, kind="stable")
        x, actual, predicted = x[order], actual[order], predicted[order]
        x, actual, predicted = _break_at_gaps(x, actual, predicted)
    else:
        x = np.arange(len(actual))

    fig, (top, bottom) = panel_grid(2, n_cols=1, panel_size=(12, 2.8))
    bottom.sharex(top)
    top.plot(x, actual, color=INK_MUTED, linewidth=1.25, label="actual")  # type: ignore
    top.plot(x, predicted, color=ACCENT, linewidth=1.5, label="predicted")  # type: ignore
    top.legend(loc="upper left", ncols=2)  # type: ignore
    top.set_title("Actual vs predicted")  # type: ignore
    residual = actual - predicted
    bottom.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)  # type: ignore
    bottom.fill_between(x, residual, 0, color=ACCENT, alpha=0.25, linewidth=0)  # type: ignore
    bottom.plot(x, residual, color=ACCENT, linewidth=0.8)  # type: ignore
    bottom.set_title("Residual (actual − predicted)")  # type: ignore
    for ax in (top, bottom):
        ax.margins(x=0)  # type: ignore
    top.tick_params(axis="x", labelbottom=False)  # type: ignore
    if ts is not None and np.issubdtype(x.dtype, np.datetime64):
        date_axis(bottom)
    else:
        bottom.set_xlabel("Row" if ts is None else "")  # type: ignore
    return fig
