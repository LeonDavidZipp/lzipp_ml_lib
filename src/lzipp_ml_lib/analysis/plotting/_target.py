from collections.abc import Sequence

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    NullLocator,
    SymmetricalLogLocator,
)

from ._style import ACCENT, ACCENT_WASH, INK_MUTED, SURFACE, panel_grid, styled
from ._utils import (
    PolarsFrame,
    categorical_columns,
    maybe_sample,
    numeric_columns,
    prepare_and_collect,
)


@styled
def plot_feature_target(
    data: PolarsFrame,
    target: str,
    columns: Sequence[str] | None = None,
    *,
    n_bins: int = 10,
    top_k: int = 10,
    sample: int | None = None,
    log_x: bool | Sequence[str] = False,
) -> Figure:
    """Plots the mean of `target` across the values of every other column, one panel
    each. Numeric features are cut into quantile bins, with each bin's mean target
    plotted at the bin's median feature value and a 95% confidence band; categorical
    features show the mean target of their most frequent values. A thin line marks the
    overall mean, so features whose curve stays on it carry little signal on their own.
    For a 0/1 target the mean is the positive rate.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        target (str): The target column; numeric or boolean.
        columns (Sequence[str] | None): Columns to restrict the plot to, in this order.
            If None, all columns but the target are used. Defaults to None.
        n_bins (int): The number of quantile bins (equally many rows each) per numeric
            feature. Defaults to 10.
        top_k (int): The number of most frequent values shown per categorical feature.
            Defaults to 10.
        sample (int | None): Plot at most this many random rows (with a fixed seed), for
            speed on large data. Defaults to None.
        log_x (bool | Sequence[str]): Features whose x axis goes on a log scale, so
            skewed features' bins don't bunch up at one end: True for every numeric
            feature, or a list of feature names. Only the axis changes, not the binning.
            A feature with a bin median <= 0 gets a symmetric log scale instead (linear
            around 0), so no bins are dropped. Defaults to False.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If `target` isn't numeric or boolean, or if `log_x` names a feature
            that isn't a numeric one of the plot.
    """
    df = maybe_sample(prepare_and_collect(data, columns, keep=(target,)), sample)
    if df[target].dtype == pl.Boolean:
        df = df.with_columns(pl.col(target).cast(pl.Float64))
    if not df[target].dtype.is_numeric():
        raise ValueError(
            f"target {target!r} must be numeric or boolean, not {df[target].dtype}"
        )
    df = df.filter(pl.col(target).is_not_null())
    numeric = numeric_columns(df, exclude=(target,))
    categorical = categorical_columns(df, exclude=(target,))
    cols = numeric + categorical
    log_cols = set(numeric if log_x is True else (log_x or ()))
    if unknown := log_cols - set(numeric):
        raise ValueError(f"log_x names non-numeric or missing features: {unknown}")
    overall = df[target].mean()

    fig, axes = panel_grid(len(cols), n_cols=3, panel_size=(4.6, 3.2))
    for ax, col in zip(axes, cols, strict=True):
        if col in numeric:
            _draw_binned(ax, df, col, target, n_bins, log=col in log_cols)
        else:
            _draw_categories(ax, df, col, target, top_k)
        if overall is not None:
            ax.axhline(overall, color=INK_MUTED, linewidth=0.8, zorder=1)  # type: ignore
        ax.set_title(col)  # type: ignore
        ax.set_xlabel("")  # type: ignore
    for ax in axes[:: min(3, len(axes))]:
        ax.set_ylabel(f"Mean {target}")  # type: ignore
    return fig


def _draw_binned(
    ax: Axes, df: pl.DataFrame, col: str, target: str, n_bins: int, log: bool
) -> None:
    stats = (
        df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
        .with_columns(
            bin=pl.col(col).bin_quantiles(n_bins, labels=False, right_closed=True)
        )
        .group_by("bin")
        .agg(
            median=pl.col(col).median(),
            mean=pl.col(target).mean(),
            sem=pl.col(target).std() / pl.len().sqrt(),
        )
        .sort("median")
        .with_columns(pl.col("sem").fill_null(0))
    )
    median, mean, sem = stats["median"], stats["mean"], stats["sem"]
    ax.fill_between(  # type: ignore
        median,
        mean - 1.96 * sem,  # type: ignore
        mean + 1.96 * sem,  # type: ignore
        color=ACCENT_WASH,
        linewidth=0,
    )
    ax.plot(  # type: ignore
        median,
        mean,
        color=ACCENT,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    if log:
        _set_log_x(ax, median.to_numpy())


def _set_log_x(ax: Axes, x: np.ndarray) -> None:
    """Log x axis; symmetric log when some bin sits at or below 0, linear up to the
    smallest positive bin, so a pile of zeros keeps its point."""
    if len(x) == 0 or x.min() > 0:
        ax.set_xscale("log")  # type: ignore
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    else:
        positive = np.abs(x[x != 0])
        linthresh = positive.min() if len(positive) else 1.0
        ax.set_xscale("symlog", linthresh=linthresh)  # type: ignore
        ax.xaxis.set_major_locator(
            SymmetricalLogLocator(base=10, linthresh=linthresh, subs=[1, 2, 5])
        )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # type: ignore
    ax.xaxis.set_minor_locator(NullLocator())


def _draw_categories(
    ax: Axes, df: pl.DataFrame, col: str, target: str, top_k: int
) -> None:
    stats = (
        df.group_by(pl.col(col).cast(pl.String).fill_null("(null)"))
        .agg(mean=pl.col(target).mean(), n=pl.len())
        .sort("n", descending=True)
        .head(top_k)
    )
    ax.bar(stats[col].to_list(), stats["mean"].to_list(), width=0.6, color=ACCENT)  # type: ignore
    ax.tick_params(axis="x", labelrotation=30)  # type: ignore
    for label in ax.get_xticklabels():  # type: ignore[operator]
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
