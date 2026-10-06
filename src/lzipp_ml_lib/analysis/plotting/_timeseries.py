from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import polars as pl
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore

from ._style import (
    ACCENT,
    ACCENT_WASH,
    BASELINE,
    GRID,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    date_axis,
    figure_and_axes,
    panel_grid,
    styled,
)
from ._utils import (
    PolarsFrame,
    default_period,
    ensure_collected,
    infer_interval,
    numeric_columns,
    series_frame,
)


@styled
def plot_timeseries_grid(
    data: PolarsFrame, time_col: str = "ts", columns: Sequence[str] | None = None
) -> Figure:
    """Plots a line chart over time for every numeric column, one panel each.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        time_col (str): The timestamp column. Defaults to "ts".
        columns (Sequence[str] | None): Columns to restrict the plot to, in this order.
            If None, all numeric columns but `time_col` are used. Defaults to None.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If none of the selected columns is numeric.
    """
    df = ensure_collected(data, columns, keep=(time_col,)).sort(time_col)
    cols = numeric_columns(df, exclude=(time_col,))
    fig, axes = panel_grid(len(cols), n_cols=1, panel_size=(12, 2.4))
    ts = df[time_col].to_numpy()
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for ax, col in zip(axes, cols, strict=True):
        ax.plot(ts, df[col].to_numpy(), color=ACCENT, linewidth=1.25)  # type: ignore
        ax.set_title(col)  # type: ignore
        ax.margins(x=0)  # type: ignore
    if df[time_col].dtype.is_temporal():
        date_axis(axes[-1])
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)  # type: ignore
    return fig


@styled
def plot_missing_values(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plots the percentage of missing values per column, most missing on top.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        columns (Sequence[str] | None): Columns to restrict the plot to, in this order.
            If None, all columns are used. Defaults to None.
        ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its look
            is adapted to the style. If None, a new figure is created. Defaults to None.

    Returns:
        Figure: The figure.
    """
    df = ensure_collected(data, columns)
    missing = (
        df.null_count()
        .transpose(include_header=True, header_name="column", column_names=["nulls"])
        .with_columns(pct=pl.col("nulls") / max(df.height, 1) * 100)
        .sort("pct")
    )
    pct = missing["pct"].to_list()

    fig, ax = figure_and_axes(ax, (8, max(2.5, 0.32 * len(pct) + 1)))
    bars = ax.barh(missing["column"].to_list(), pct, height=0.6, color=ACCENT)  # type: ignore
    ax.bar_label(  # type: ignore
        bars,
        labels=[f"{p:.1f}%" if p > 0 else "" for p in pct],
        padding=4,
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_xlim(0, max(max(pct, default=0), 1) * 1.15)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=None))  # type: ignore
    ax.grid(axis="y", visible=False)  # type: ignore
    ax.grid(axis="x", visible=True)  # type: ignore
    ax.margins(y=0.02)  # type: ignore
    ax.set_title("Missing values")  # type: ignore
    return fig


@styled
def plot_autocorrelation(
    data: PolarsFrame,
    target_col: str = "val",
    lags: int = 50,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plots the autocorrelation function (ACF) of a series, with its 95% confidence
    band. The ACF at lag k mixes the direct effect of k steps back with everything
    passed on through the steps in between; see `plot_partial_autocorrelation` for the
    direct effect alone.

    The values are taken in the data's row order, so the data should be sorted by time.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        target_col (str): The column of the series. Defaults to "val".
        lags (int): The number of lags to show. Defaults to 50.
        ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its look
            is adapted to the style. If None, a new figure is created. Defaults to None.

    Returns:
        Figure: The figure.
    """
    series = ensure_collected(data).get_column(target_col).drop_nulls().to_numpy()
    fig, ax = figure_and_axes(ax, (12, 3.6))
    plot_acf(series, lags=lags, ax=ax, alpha=0.05, **_correlogram_kws())
    _style_correlogram(ax, f"Autocorrelation of {target_col}")
    return fig


@styled
def plot_partial_autocorrelation(
    data: PolarsFrame,
    target_col: str = "val",
    lags: int = 50,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plots the partial autocorrelation function (PACF) of a series, with its 95%
    confidence band. The PACF at lag k is the correlation with k steps back after
    removing what the steps in between already explain, so lags sticking out of the band
    are candidates for lag features (e.g. 1, 24 and 168 for hourly data with daily and
    weekly cycles).

    The values are taken in the data's row order, so the data should be sorted by time.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        target_col (str): The column of the series. Defaults to "val".
        lags (int): The number of lags to show; must be below half the series length.
            Defaults to 50.
        ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its look
            is adapted to the style. If None, a new figure is created. Defaults to None.

    Returns:
        Figure: The figure.
    """
    series = ensure_collected(data).get_column(target_col).drop_nulls().to_numpy()
    fig, ax = figure_and_axes(ax, (12, 3.6))
    plot_pacf(series, lags=lags, ax=ax, alpha=0.05, method="ywm", **_correlogram_kws())
    _style_correlogram(ax, f"Partial autocorrelation of {target_col}")
    return fig


def _correlogram_kws() -> dict[str, Any]:
    return {
        "title": "",
        "color": ACCENT,
        "markersize": 4,
        "vlines_kwargs": {"colors": ACCENT, "linewidth": 1.25},
    }


def _style_correlogram(ax: Axes, title: str) -> None:
    for coll in ax.collections:  # type: ignore
        if isinstance(coll, PolyCollection):  # the confidence band
            coll.set_alpha(None)  # statsmodels' own alpha would override the wash
            coll.set_facecolor(ACCENT_WASH)
            coll.set_edgecolor("none")
    for line in ax.lines:  # type: ignore
        if line.get_marker() in ("None", None, ""):  # type: ignore
            line.set_color(BASELINE)  # type: ignore
            line.set_linewidth(0.8)  # type: ignore
    ax.set_title(title)  # type: ignore
    ax.set_xlabel("Lag")  # type: ignore
    ax.set_ylabel("Correlation")  # type: ignore
    ax.set_ylim(-1.05, 1.05)  # type: ignore
    ax.margins(x=0.01)  # type: ignore


@styled
def plot_rolling_stats(
    data: PolarsFrame,
    time_col: str = "ts",
    target_col: str = "val",
    *,
    window: str | int | None = None,
) -> Figure:
    """Plots a series' rolling mean (with a ±1 std band) over the raw series, and the
    rolling std below it, to check stationarity at a glance: a drifting mean or a
    changing std means the level or the volatility isn't constant.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        time_col (str): The timestamp column. Defaults to "ts".
        target_col (str): The column of the series. Defaults to "val".
        window (str | int | None): The window, as a polars duration like "30d" (by time)
            or a number of rows; either way, nothing is drawn until the first full
            window. If None, one natural seasonal period (a day of hourly data, a week
            of daily data), so the seasonal swing averages out of the mean. Defaults to
            None.

    Returns:
        Figure: The figure.
    """
    df = series_frame(data, time_col, target_col)
    if window is None:
        interval = infer_interval(df[time_col])
        window = f"{round(default_period(interval) * interval.total_seconds())}s"
        label = _duration_label(default_period(interval) * interval)
    else:
        label = str(window) if isinstance(window, str) else f"{window} rows"
    value = pl.col(target_col)
    if isinstance(window, str):
        mean = value.rolling_mean_by(time_col, window_size=window)
        std = value.rolling_std_by(time_col, window_size=window)
    else:
        mean = value.rolling_mean(window)
        std = value.rolling_std(window)
    df = df.with_columns(mean=mean, std=std)
    if isinstance(window, str):
        # like a row window, show nothing until one full window of time has passed
        warm = pl.col(time_col) - pl.col(time_col).min() >= _as_timedelta(window)
        df = df.with_columns(pl.when(warm).then(pl.col("mean", "std")))

    # std is undefined until the window has 2 values; NaN leaves that stretch empty
    ts, mean_v = df[time_col], df["mean"]
    std_v = df["std"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    fig, (top, bottom) = panel_grid(2, n_cols=1, panel_size=(12, 2.6))
    bottom.sharex(top)
    # the raw series as quiet context: on long series it's a dense block
    top.plot(ts, df[target_col], color=GRID, linewidth=0.5, zorder=1)  # type: ignore
    top.fill_between(  # type: ignore
        ts,
        mean_v - std_v,  # type: ignore
        mean_v + std_v,  # type: ignore
        color=ACCENT_WASH,
        linewidth=0,
        zorder=2,
    )
    top.plot(ts, mean_v, color=ACCENT, linewidth=1.5, zorder=3)  # type: ignore
    top.set_title(f"{target_col} · rolling mean ± std ({label})")  # type: ignore
    bottom.plot(ts, std_v, color=ACCENT, linewidth=1.5)  # type: ignore
    bottom.set_title(f"Rolling std ({label})")  # type: ignore
    bottom.set_ylim(bottom=0)
    for ax in (top, bottom):
        ax.margins(x=0)  # type: ignore
    top.tick_params(axis="x", labelbottom=False)  # type: ignore
    date_axis(bottom)
    return fig


@styled
def plot_gaps(
    data: PolarsFrame,
    time_col: str = "ts",
    *,
    interval: str | timedelta | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plots where timestamps are missing: one stem per gap, at its start, as tall as
    the number of missing steps. The title sums up the gaps and duplicate timestamps.
    Lag features look a timestamp up by time, so gaps are where they come out null.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        time_col (str): The timestamp column. Defaults to "ts".
        interval (str | timedelta | None): The expected step, as a polars duration like
            "1h" or a timedelta. If None, the most common step. Defaults to None.
        ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its look
            is adapted to the style. If None, a new figure is created. Defaults to None.

    Returns:
        Figure: The figure.
    """
    ts = ensure_collected(data, [time_col])[time_col].drop_nulls().sort()
    n_duplicates = ts.len() - ts.n_unique()
    ts = ts.unique(maintain_order=True)
    step = _as_timedelta(interval) if interval is not None else infer_interval(ts)
    diffs = ts.diff()
    is_gap = diffs > step
    starts = ts.shift(1).filter(is_gap)
    missing = (diffs.filter(is_gap) / step).cast(pl.Float64).round() - 1
    n_steps = round((ts.max() - ts.min()) / step) + 1  # type: ignore

    fig, ax = figure_and_axes(ax, (12, 3))
    if starts.len():
        ax.vlines(starts, 0, missing, color=ACCENT, linewidth=1.25)  # type: ignore
        ax.plot(  # type: ignore
            starts,
            missing,
            "o",
            color=ACCENT,
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        if missing.max() > 20 * max(missing.min(), 1):  # type: ignore
            ax.set_yscale("log")  # type: ignore
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0.8 if ax.get_yscale() == "log" else 0)  # type: ignore[operator]
    else:
        ax.text(  # type: ignore
            0.5,
            0.5,
            "No gaps",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=INK_MUTED,
        )
    ax.set_xlim(ts.min(), ts.max())  # type: ignore
    date_axis(ax)
    ax.set_ylabel("Missing steps")  # type: ignore
    summary = [
        f"{starts.len()} gaps",
        f"{int(missing.sum())} missing steps ({missing.sum() / n_steps:.2%})",
        f"expected every {_duration_label(step)}",
    ]
    if n_duplicates:
        summary.append(f"{n_duplicates} duplicate timestamps")
    ax.set_title(f"Gaps in {time_col}  ·  " + "  ·  ".join(summary))  # type: ignore
    return fig


def _as_timedelta(duration: str | timedelta) -> timedelta:
    if isinstance(duration, timedelta):
        return duration
    origin = datetime(2000, 1, 1)
    shifted = pl.select(pl.lit(origin).dt.offset_by(duration)).item()
    return shifted - origin


def _duration_label(duration: timedelta) -> str:
    """A compact label like "1h", "15min", "7d"."""
    seconds = round(duration.total_seconds())
    for unit, size in (("d", 86_400), ("h", 3_600), ("min", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"
