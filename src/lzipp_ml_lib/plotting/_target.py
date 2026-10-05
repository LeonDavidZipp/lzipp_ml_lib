from collections.abc import Sequence

import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    overall = df[target].mean()

    fig, axes = panel_grid(len(cols), n_cols=3, panel_size=(4.6, 3.2))
    for ax, col in zip(axes, cols, strict=True):
        if col in numeric:
            _draw_binned(ax, df, col, target, n_bins)
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
    ax: Axes, df: pl.DataFrame, col: str, target: str, n_bins: int
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
