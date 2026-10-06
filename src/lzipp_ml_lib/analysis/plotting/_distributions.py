from collections.abc import Callable, Sequence
from typing import Any

import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ._style import (
    ACCENT,
    ACCENT_WASH,
    INK_MUTED,
    SURFACE,
    add_group_legend,
    group_palette,
    panel_grid,
    styled,
)
from ._utils import (
    PolarsFrame,
    ensure_collected,
    maybe_sample,
    numeric_columns,
    panel_values,
)


@styled
def plot_kde(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    *,
    by: str | None = None,
    sample: int | None = None,
    log: bool = False,
    clip: tuple[float, float] | None = None,
) -> Figure:
    """Plots a density curve for every numeric column, one panel each. With `by`, each
    group gets its own curve, normalized on its own, so shapes compare even when group
    sizes differ. Curves stop at the data's range.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        columns (Sequence[str] | None): Columns to restrict the plot to, in this order.
            If None, all numeric columns are used. Defaults to None.
        by (str | None): A column whose groups are compared within each panel, e.g. the
            class or the train/test split; at most 8 groups. Defaults to None.
        sample (int | None): Plot at most this many random rows (with a fixed seed), for
            speed on large data. Defaults to None.
        log (bool): Whether to put the values on a log scale; non-positive values are
            dropped. Defaults to False.
        clip (tuple[float, float] | None): Keep only values inside this quantile range
            per column, e.g. (0.01, 0.99), so a few extremes don't flatten the plot.
            Defaults to None.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If none of the selected columns is numeric, or if `by` has more than
            8 groups.
    """

    def draw(ax: Axes, sub: pl.DataFrame, col: str, palette: dict[Any, str]) -> None:
        # cut=0: no curve past the data, e.g. no density below 0 for a count
        kws: dict[str, Any] = {
            "data": sub,
            "x": col,
            "ax": ax,
            "log_scale": log,
            "cut": 0,
        }
        if by is None:
            sns.kdeplot(**kws, fill=True, color=ACCENT, alpha=0.12, lw=0)
            sns.kdeplot(**kws, color=ACCENT, linewidth=2)
        else:
            group: dict[str, Any] = {
                "hue": by,
                "palette": palette,
                "common_norm": False,
            }
            sns.kdeplot(**kws, **group, fill=True, alpha=0.08, lw=0, legend=False)
            sns.kdeplot(**kws, **group, linewidth=2, legend=False)
        ax.set_ylabel("Density")  # type: ignore
        ax.set_ylim(bottom=0)

    return _panels(data, columns, by, sample, log, clip, draw, n_cols=2)


@styled
def plot_histograms(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    *,
    bins: int | str = "auto",
    by: str | None = None,
    sample: int | None = None,
    log: bool = False,
    clip: tuple[float, float] | None = None,
) -> Figure:
    """Plots a histogram for every numeric column, one panel each. Unlike a density
    curve, it shows hard bounds, gaps and spikes (e.g. a pile of zeros) as they are.
    With `by`, each group is drawn as an outline, normalized on its own.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        columns (Sequence[str] | None): Columns to restrict the plot to, in this order.
            If None, all numeric columns are used. Defaults to None.
        bins (int | str): The number of bins, or a numpy rule like "auto" or "sturges".
            Integer columns with at most 50 distinct values always get one bar per
            value. Defaults to "auto".
        by (str | None): A column whose groups are compared within each panel, e.g. the
            class or the train/test split; at most 8 groups. Defaults to None.
        sample (int | None): Plot at most this many random rows (with a fixed seed), for
            speed on large data. Defaults to None.
        log (bool): Whether to put the values on a log scale; non-positive values are
            dropped. Defaults to False.
        clip (tuple[float, float] | None): Keep only values inside this quantile range
            per column, e.g. (0.01, 0.99), so a few extremes don't flatten the plot.
            Defaults to None.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If none of the selected columns is numeric, or if `by` has more than
            8 groups.
    """

    def draw(ax: Axes, sub: pl.DataFrame, col: str, palette: dict[Any, str]) -> None:
        discrete = sub[col].dtype.is_integer() and sub[col].n_unique() <= 50
        kws: dict[str, Any] = {
            "data": sub,
            "x": col,
            "ax": ax,
            "log_scale": log,
            "discrete": discrete,
            "bins": "auto" if discrete else bins,
        }
        if by is None:
            sns.histplot(
                **kws,
                color=ACCENT,
                alpha=1,
                edgecolor=SURFACE,
                linewidth=0.8,
                shrink=0.8 if discrete else 1,
            )
            ax.set_ylabel("Count")  # type: ignore
        else:
            sns.histplot(
                **kws,
                hue=by,
                palette=palette,
                stat="density",
                common_norm=False,
                element="step",
                alpha=0.08,
                linewidth=1.5,
                legend=False,
            )
            ax.set_ylabel("Density")  # type: ignore
        if discrete:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return _panels(data, columns, by, sample, log, clip, draw, n_cols=2)


@styled
def plot_boxplots(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    *,
    by: str | None = None,
    sample: int | None = None,
    log: bool = False,
    clip: tuple[float, float] | None = None,
) -> Figure:
    """Plots a boxplot for every numeric column, one panel each, to show spread and
    outliers. With `by`, each panel gets one box per group.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        columns (Sequence[str] | None): Columns to restrict the plot to, in this order.
            If None, all numeric columns are used. Defaults to None.
        by (str | None): A column whose groups are compared within each panel, e.g. the
            class or the train/test split; at most 8 groups. Defaults to None.
        sample (int | None): Plot at most this many random rows (with a fixed seed), for
            speed on large data. Defaults to None.
        log (bool): Whether to put the values on a log scale; non-positive values are
            dropped. Defaults to False.
        clip (tuple[float, float] | None): Keep only values inside this quantile range
            per column, e.g. (0.01, 0.99), so a few extremes don't flatten the plot.
            Defaults to None.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If none of the selected columns is numeric, or if `by` has more than
            8 groups.
    """

    def draw(ax: Axes, sub: pl.DataFrame, col: str, palette: dict[Any, str]) -> None:
        sns.boxplot(
            data=sub,
            x=col,
            y=by,
            order=list(palette) or None,
            orient="h",
            ax=ax,
            log_scale=log,
            width=0.4 if by is None else 0.6,
            linewidth=1.25,
            color=ACCENT,
            linecolor=ACCENT,
            boxprops={"facecolor": ACCENT_WASH},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": ACCENT_WASH,
                "markeredgecolor": ACCENT,
                "markeredgewidth": 0.8,
            },
        )
        _style_horizontal(ax, has_groups=by is not None)

    return _panels(data, columns, by, sample, log, clip, draw, n_cols=3, legend=False)


@styled
def plot_violinplots(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    *,
    by: str | None = None,
    sample: int | None = None,
    log: bool = False,
    clip: tuple[float, float] | None = None,
) -> Figure:
    """Plots a violin plot for every numeric column, one panel each, to show the shape
    of its distribution with its quartiles. With `by`, each panel gets one violin per
    group.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        columns (Sequence[str] | None): Columns to restrict the plot to, in this order.
            If None, all numeric columns are used. Defaults to None.
        by (str | None): A column whose groups are compared within each panel, e.g. the
            class or the train/test split; at most 8 groups. Defaults to None.
        sample (int | None): Plot at most this many random rows (with a fixed seed), for
            speed on large data. Defaults to None.
        log (bool): Whether to put the values on a log scale; non-positive values are
            dropped. Defaults to False.
        clip (tuple[float, float] | None): Keep only values inside this quantile range
            per column, e.g. (0.01, 0.99), so a few extremes don't flatten the plot.
            Defaults to None.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If none of the selected columns is numeric, or if `by` has more than
            8 groups.
    """

    def draw(ax: Axes, sub: pl.DataFrame, col: str, palette: dict[Any, str]) -> None:
        sns.violinplot(
            data=sub,
            x=col,
            y=by,
            order=list(palette) or None,
            orient="h",
            ax=ax,
            log_scale=log,
            color=ACCENT_WASH,
            linecolor=ACCENT,
            linewidth=1.25,
            inner="quart",
            cut=0,
            inner_kws={"color": ACCENT, "linewidth": 1},
            saturation=1,
        )
        _style_horizontal(ax, has_groups=by is not None)

    return _panels(data, columns, by, sample, log, clip, draw, n_cols=3, legend=False)


def _panels(
    data: PolarsFrame,
    columns: Sequence[str] | None,
    by: str | None,
    sample: int | None,
    log: bool,
    clip: tuple[float, float] | None,
    draw: Callable[[Axes, pl.DataFrame, str, dict[Any, str]], None],
    n_cols: int,
    legend: bool = True,
) -> Figure:
    """One panel per numeric column, each drawn by `draw` from that column's
    plottable rows. Box and violin plots name their groups on the y axis, so they
    skip the colour legend."""
    keep = (by,) if by is not None else ()
    df = maybe_sample(ensure_collected(data, columns, keep=keep), sample)
    cols = numeric_columns(df, exclude=keep)
    palette = group_palette(df, by) if by is not None else {}
    if by is not None:
        df = df.filter(pl.col(by).is_not_null())
    panel_size = (6, 3.2) if n_cols == 2 else (5, 2.2 + 0.35 * len(palette))
    fig, axes = panel_grid(len(cols), n_cols=n_cols, panel_size=panel_size)
    for ax, col in zip(axes, cols, strict=True):
        draw(ax, panel_values(df, col, log, clip), col, palette)
        ax.set_title(col)  # type: ignore
        ax.set_xlabel("")  # type: ignore
    if by is not None and legend:
        add_group_legend(fig, palette, title=by)
    return fig


def _style_horizontal(ax: Axes, has_groups: bool) -> None:
    """Horizontal distributions: vertical grid only; the y axis only names groups."""
    if has_groups:
        ax.set_ylabel("")  # type: ignore
    else:
        ax.set_yticks([])  # type: ignore[operator]
    ax.grid(axis="y", visible=False)  # type: ignore
    ax.grid(axis="x", visible=True)  # type: ignore
    ax.tick_params(axis="x", colors=INK_MUTED, labelcolor=INK_MUTED)  # type: ignore
