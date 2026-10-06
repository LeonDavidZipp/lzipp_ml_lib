from collections.abc import Sequence

import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._style import ACCENT, INK_MUTED, INK_SECONDARY, panel_grid, styled
from ._utils import PolarsFrame, categorical_columns, prepare_and_collect


@styled
def plot_category_counts(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    as_categorical: Sequence[str] | None = None,
    *,
    top_k: int = 10,
) -> Figure:
    """Plots how often each value occurs in every string, categorical and boolean
    column, one panel each, most frequent on top. Nulls get their own bar.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        columns (Sequence[str] | None): Columns to restrict the plot to, in this order.
            If None, all categorical columns are used. Defaults to None.
        as_categorical (Sequence[str] | None): Numeric columns to count as categories,
            e.g. codes like a region id. Defaults to None.
        top_k (int): The number of most frequent values shown per column; the rest are
            folded into one "other" bar. Defaults to 10.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If none of the selected columns is categorical.
    """
    df = prepare_and_collect(data, columns)
    if as_categorical:
        df = df.with_columns(pl.col(as_categorical).cast(pl.String))
    cols = categorical_columns(df)
    fig, axes = panel_grid(
        len(cols), n_cols=2, panel_size=(6, 0.32 * (min(top_k, 12) + 2) + 1.2)
    )
    for ax, col in zip(axes, cols, strict=True):
        _draw_counts(ax, df[col], top_k)
    return fig


def _draw_counts(ax: Axes, s: pl.Series, top_k: int) -> None:
    counts = (
        s.cast(pl.String)
        .fill_null("(null)")
        .value_counts(sort=True, name="n", parallel=True)
        .rename({s.name: "value"})
    )
    n_values = counts.height
    if n_values > top_k:
        rest = counts.slice(top_k)
        counts = pl.concat(
            [
                counts.head(top_k),
                pl.DataFrame(
                    {
                        "value": [f"other ({rest.height})"],
                        "n": [rest.get_column("n").sum()],
                    },
                    schema=counts.schema,
                ),
            ]
        )
    # reversed for matplotlib
    labels = counts.get_column("value").to_list()[::-1]
    values = counts.get_column("n").to_list()[::-1]
    colors = [
        INK_MUTED if label.startswith("other (") or label == "(null)" else ACCENT
        for label in labels
    ]
    bars = ax.barh(labels, values, height=0.6, color=colors)  # type: ignore
    total = s.len()
    ax.bar_label(  # type: ignore
        bars,
        labels=[f"{v / total:.0%}" if total else "" for v in values],
        padding=4,
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_xlim(0, max(values, default=1) * 1.15)
    # one slot per possible bar, so bars always keep same thickness
    slots = min(top_k, 12) + 1
    ax.set_ylim(len(labels) - slots - 0.5, len(labels) - 0.5)
    ax.grid(axis="y", visible=False)  # type: ignore
    ax.grid(axis="x", visible=True)  # type: ignore
    ax.set_title(f"{s.name}  ·  {n_values} distinct")  # type: ignore
