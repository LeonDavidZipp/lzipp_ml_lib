import matplotlib.dates as mdates
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from statsmodels.graphics.tsaplots import plot_acf  # type: ignore

from ._style import ACCENT, ACCENT_WASH, BASELINE, INK_SECONDARY, panel_grid, styled
from ._utils import PolarsFrame, ensure_collected, numeric_columns


@styled
def plot_timeseries_grid(data: PolarsFrame, time_col: str = "ts") -> Figure:
    """Plots a line chart over time for every numeric column in the dataset."""
    df = ensure_collected(data).sort(time_col)
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
        locator = mdates.AutoDateLocator()
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))  # type: ignore
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)  # type: ignore
    return fig


@styled
def plot_missing_values(data: PolarsFrame) -> Figure:
    """Plots the percentage of missing values per column, most missing on top."""
    df = ensure_collected(data)
    missing = (
        df.null_count()
        .transpose(include_header=True, header_name="column", column_names=["nulls"])
        .with_columns(pct=pl.col("nulls") / max(df.height, 1) * 100)
        .sort("pct")
    )
    pct = missing["pct"].to_list()

    fig = plt.figure(figsize=(8, max(2.5, 0.32 * len(pct) + 1)), layout="constrained")
    ax = fig.add_subplot()
    bars = ax.barh(missing["column"].to_list(), pct, height=0.6, color=ACCENT)
    ax.bar_label(
        bars,
        labels=[f"{p:.1f}%" if p > 0 else "" for p in pct],
        padding=4,
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_xlim(0, max(max(pct, default=0), 1) * 1.15)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(y=0.02)
    ax.set_title("Missing values")
    return fig


@styled
def plot_autocorrelation(
    data: PolarsFrame, target_col: str = "y", lags: int = 50
) -> Figure:
    """Plots the autocorrelation function (ACF) for the target variable, with its
    95% confidence band."""
    df = ensure_collected(data)
    series = df.get_column(target_col).drop_nulls().to_numpy()

    fig = plt.figure(figsize=(12, 3.6), layout="constrained")
    ax = fig.add_subplot()
    plot_acf(
        series,
        lags=lags,
        ax=ax,
        alpha=0.05,
        title=f"Autocorrelation of {target_col}",
        color=ACCENT,
        markersize=4,
        vlines_kwargs={"colors": ACCENT, "linewidth": 1.25},
    )
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):  # the confidence band
            coll.set_alpha(None)  # statsmodels' own alpha would override the wash
            coll.set_facecolor(ACCENT_WASH)
            coll.set_edgecolor("none")
    for line in ax.lines:
        if line.get_marker() in ("None", None, ""):  # the zero line
            line.set_color(BASELINE)
            line.set_linewidth(0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.margins(x=0.01)
    return fig
