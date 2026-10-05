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
    ensure_collected,
    maybe_sample,
    numeric_columns,
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
    """Plots the mean of `target` across the values of every other column.

    Numeric features are cut into `n_bins` quantile bins (equally many rows each);
    each bin's mean target is plotted at the bin's median feature value, with a
    95% confidence band. Categorical features show the mean target of their
    `top_k` most frequent values. A thin line marks the overall mean, so features
    whose curve stays on it carry little signal on their own.

    For a 0/1 target the mean is the positive rate. `target` must be numeric (or
    boolean). `columns` restricts it to these features; `sample` plots at most
    this many random rows.

    `log_x` puts the x axis of skewed features on a log scale, so their bins don't
    bunch up at one end: True for every numeric feature, or a list of feature
    names. Binning is unchanged, only the axis is. If a bin's median is <= 0 (e.g.
    a pile of zeros), that feature gets a symmetric log scale instead (linear
    around 0), so no bins are dropped.
    """
    df = maybe_sample(ensure_collected(data, columns, keep=(target,)), sample)
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
        .with_columns(bin=pl.col(col).qcut(n_bins, allow_duplicates=True).to_physical())
        .group_by("bin")
        .agg(
            x=pl.col(col).median(),
            mean=pl.col(target).mean(),
            sem=pl.col(target).std() / pl.len().sqrt(),
        )
        .sort("x")
        .with_columns(pl.col("sem").fill_null(0))
    )
    x, mean, sem = stats["x"], stats["mean"], stats["sem"]
    ax.fill_between(  # type: ignore
        x,
        mean - 1.96 * sem,  # type: ignore
        mean + 1.96 * sem,  # type: ignore
        color=ACCENT_WASH,
        linewidth=0,
    )
    ax.plot(  # type: ignore
        x,
        mean,
        color=ACCENT,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    if log:
        _set_log_x(ax, x.to_numpy())


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
    # plain numbers (0.05, 2, 500) instead of powers of ten
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
    for label in ax.get_xticklabels():  # type: ignore[operator]  # broken matplotlib-stubs
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
