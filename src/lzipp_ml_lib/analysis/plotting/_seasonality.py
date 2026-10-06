from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.tsa.seasonal as _seasonal  # type: ignore
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._style import (
    ACCENT,
    ACCENT_WASH,
    BASELINE,
    INK_MUTED,
    SEQUENTIAL,
    SURFACE,
    date_axis,
    figure_and_axes,
    panel_grid,
    styled,
)
from ._utils import PolarsFrame, default_period, infer_interval, series_frame

# STL is compiled (Cython) and ships without type information, so the checker sees
# it as Unknown; typing both as Any says that's deliberate
_STL: Any = cast(Any, _seasonal).STL
_MSTL: Any = cast(Any, _seasonal).MSTL

CalendarUnit = Literal[
    "minute", "hour", "weekday", "day", "week", "month", "quarter", "year"
]

_UNIT_EXPR: Mapping[str, Callable[[str], pl.Expr]] = {
    "minute": lambda t: pl.col(t).dt.minute(),
    "hour": lambda t: pl.col(t).dt.hour(),
    "weekday": lambda t: pl.col(t).dt.weekday(),  # 1 = Monday
    "day": lambda t: pl.col(t).dt.day(),
    "week": lambda t: pl.col(t).dt.week(),
    "month": lambda t: pl.col(t).dt.month(),
    "quarter": lambda t: pl.col(t).dt.quarter(),
    "year": lambda t: pl.col(t).dt.year(),
}
_UNIT_NAMES = {
    "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "month": [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
}


@styled
def plot_seasonal_profile(
    data: PolarsFrame,
    time_col: str = "ts",
    target_col: str = "val",
    *,
    rows: CalendarUnit = "weekday",
    cols: CalendarUnit | None = "hour",
    agg: Literal["mean", "median"] = "mean",
    ax: Axes | None = None,
) -> Figure:
    """Plots a series' typical value per calendar position: a heatmap of rows x cols,
    e.g. weekday x hour, or a line over `rows` alone, with the interquartile range as a
    band. Both show whether calendar features are worth adding: a flat profile means
    they carry little.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        time_col (str): The timestamp column. Defaults to "ts".
        target_col (str): The column of the series. Defaults to "val".
        rows (CalendarUnit): The calendar unit of the rows (or the x axis of the line):
            "minute", "hour", "weekday", "day", "week", "month", "quarter" or "year".
            Defaults to "weekday".
        cols (CalendarUnit | None): The calendar unit of the columns. If None, a line
            over `rows` is drawn instead of a heatmap. Defaults to "hour".
        agg (Literal["mean", "median"]): How the values per calendar position are
            summarized. Defaults to "mean".
        ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its look
            is adapted to the style. If None, a new figure is created. Defaults to None.

    Returns:
        Figure: The figure.
    """
    df = series_frame(data, time_col, target_col)
    units = [rows] if cols is None else [rows, cols]
    df = df.with_columns(**{u: _UNIT_EXPR[u](time_col) for u in units})
    value = pl.col(target_col).mean() if agg == "mean" else pl.col(target_col).median()

    if cols is None:
        fig, ax = figure_and_axes(ax, (9, 3.6))
        _draw_profile_line(ax, df, rows, target_col, value)
        ax.set_title(f"{target_col} by {rows}")  # type: ignore
        return fig

    grid = (
        df.group_by(rows, cols)
        .agg(value=value)
        .pivot(on=cols, index=rows, values="value", sort_columns=True)
        .sort(rows)
    )
    row_keys = grid[rows].to_list()
    col_keys = [c for c in grid.columns if c != rows]
    width = min(14, 3 + 0.42 * len(col_keys))
    fig, ax = figure_and_axes(ax, (width, 1.2 + 0.42 * len(row_keys)))
    sns.heatmap(  # type: ignore
        grid.drop(rows).to_numpy(),
        cmap=SEQUENTIAL,
        linewidths=1,
        linecolor=SURFACE,
        xticklabels=_unit_labels(cols, [int(c) for c in col_keys]),
        yticklabels=_unit_labels(rows, row_keys),
        cbar_kws={"shrink": 0.8, "label": f"{agg} {target_col}"},
        ax=ax,
    )
    ax.grid(visible=False)  # type: ignore
    ax.tick_params(axis="y", labelrotation=0)  # type: ignore
    ax.tick_params(axis="x", labelrotation=0)  # type: ignore
    ax.set_xlabel(cols)  # type: ignore
    ax.set_ylabel(rows)  # type: ignore
    cbar = ax.collections[0].colorbar  # type: ignore
    if cbar is not None:
        cbar.outline.set_visible(False)  # type: ignore
        cbar.ax.tick_params(length=0)  # type: ignore
    ax.set_title(f"{agg.capitalize()} {target_col} by {rows} × {cols}")  # type: ignore
    return fig


def _draw_profile_line(
    ax: Axes, df: pl.DataFrame, unit: str, target_col: str, value: pl.Expr
) -> None:
    stats = (
        df.group_by(unit)
        .agg(
            value=value,
            p25=pl.col(target_col).quantile(0.25),
            p75=pl.col(target_col).quantile(0.75),
        )
        .sort(unit)
    )
    x = stats[unit].to_numpy()
    ax.fill_between(x, stats["p25"], stats["p75"], color=ACCENT_WASH, linewidth=0)  # type: ignore
    ax.plot(  # type: ignore
        x,
        stats["value"],
        color=ACCENT,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    ax.set_xticks(x, _unit_labels(unit, stats[unit].to_list()))  # type: ignore
    ax.set_xlabel(unit)  # type: ignore
    ax.set_ylabel(target_col)  # type: ignore


def _unit_labels(unit: str, keys: Sequence[int]) -> list[str]:
    names = _UNIT_NAMES.get(unit)
    return [names[k - 1] if names else str(k) for k in keys]


@styled
def plot_decomposition(
    data: PolarsFrame,
    time_col: str = "ts",
    target_col: str = "val",
    *,
    periods: int | Sequence[int] | None = None,
    robust: bool = True,
) -> Figure:
    """Splits a series into trend, seasonality and residual (STL), one panel each.

    STL needs evenly spaced data without gaps: duplicate timestamps are averaged,
    missing steps linearly interpolated, and how many were filled is noted at the top.
    On long series it gets slow; slicing to the range of interest first helps.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        time_col (str): The timestamp column. Defaults to "ts".
        target_col (str): The column of the series. Defaults to "val".
        periods (int | Sequence[int] | None): Seasonal cycle lengths in steps, e.g. (24,
            168) for a daily and a weekly cycle in hourly data; several periods use
            MSTL, with one seasonal panel each. If None, one natural period for the
            data's step: a day of sub-daily data, a week of daily data, a year of weekly
            or monthly data. Defaults to None.
        robust (bool): Whether to keep outliers out of the trend and seasonality, so
            they end up in the residual, where they can be seen. Defaults to True.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If the series spans fewer than two of the longest period, or has
            fewer than two timestamps.
    """
    df = series_frame(data, time_col, target_col)
    interval = infer_interval(df[time_col])
    regular = (
        df.group_by(time_col)
        .agg(pl.col(target_col).mean())
        .sort(time_col)
        .upsample(time_col, every=interval)
    )
    n_filled = regular[target_col].null_count()
    regular = regular.with_columns(pl.col(target_col).interpolate())

    if periods is None:
        periods = [default_period(interval)]
    elif isinstance(periods, int):
        periods = [periods]
    periods = sorted(periods)
    if regular.height < 2 * periods[-1]:
        raise ValueError(
            f"a decomposition with period {periods[-1]} needs at least "
            f"{2 * periods[-1]} steps, but the series has {regular.height}"
        )

    y = regular[target_col].to_numpy()
    if len(periods) == 1:
        result = _STL(y, period=periods[0], robust=robust).fit()
        seasonals = {periods[0]: np.asarray(result.seasonal)}
    else:
        result = _MSTL(y, periods=periods, stl_kwargs={"robust": robust}).fit()
        seasonals = {
            p: np.asarray(result.seasonal[:, i]) for i, p in enumerate(periods)
        }

    ts = regular[time_col].to_numpy()
    panels = [
        (target_col, y),
        ("trend", np.asarray(result.trend)),
        *((f"seasonal · period {p}", s) for p, s in seasonals.items()),
        ("residual", np.asarray(result.resid)),
    ]
    fig, axes = panel_grid(len(panels), n_cols=1, panel_size=(12, 2.1))
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for ax, (title, values) in zip(axes, panels, strict=True):
        linewidth = 0.6 if title in ("residual", target_col) else 1.25
        ax.plot(ts, values, color=ACCENT, linewidth=linewidth)  # type: ignore
        if title == "residual" or title.startswith("seasonal"):
            ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)  # type: ignore
        ax.set_title(title)  # type: ignore
        ax.margins(x=0)  # type: ignore
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)  # type: ignore
    date_axis(axes[-1])
    if n_filled:
        fig.text(  # type: ignore
            0.995,
            0.995,
            f"{n_filled} missing step{'s' * (n_filled != 1)} interpolated",
            ha="right",
            va="top",
            fontsize=8,
            color=INK_MUTED,
        )
    return fig
