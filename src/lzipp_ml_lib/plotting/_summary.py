from collections.abc import Sequence
from typing import Any

import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._style import ACCENT, INK_MUTED, INK_SECONDARY, panel_grid, styled
from ._utils import PolarsFrame, categorical_columns, ensure_collected

_SUMMARY_SCHEMA: dict[str, Any] = {
    "column": pl.String,
    "dtype": pl.String,
    "count": pl.UInt32,
    "null_pct": pl.Float64,
    "n_unique": pl.UInt32,
    "mean": pl.Float64,
    "std": pl.Float64,
    "min": pl.Float64,
    "p25": pl.Float64,
    "median": pl.Float64,
    "p75": pl.Float64,
    "max": pl.Float64,
    "skew": pl.Float64,
    "zeros_pct": pl.Float64,
}


def summarize(data: PolarsFrame, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """One row of summary statistics per column.

    Every column gets its dtype, non-null `count`, `null_pct` and `n_unique`.
    Numeric ones also get mean, std, min, quartiles, max, `skew` and `zeros_pct`
    (all over their non-null, non-NaN values); for other columns these are null.
    Percentages are 0-100. `columns` restricts it to these columns.

    The less common columns are the useful ones: a high `skew` asks for a log
    transform, a low `n_unique` for treating the column as categorical, and a high
    `zeros_pct` for a zero-inflated model or an "is zero" flag.
    """
    df = ensure_collected(data, columns)
    rows = [_summary_row(df[col], df.height) for col in df.columns]
    return pl.DataFrame(rows, schema=_SUMMARY_SCHEMA, orient="row")


def _summary_row(s: pl.Series, height: int) -> tuple[Any, ...]:
    common = (
        s.name,
        str(s.dtype),
        s.len() - s.null_count(),
        s.null_count() / height * 100 if height else 0.0,
        s.n_unique(),
    )
    if not s.dtype.is_numeric():
        return common + (None,) * 9
    v = s.drop_nulls().cast(pl.Float64).drop_nans()
    if v.len() == 0:
        return common + (None,) * 9
    return common + (
        v.mean(),
        v.std(),
        v.min(),
        v.quantile(0.25, interpolation="linear"),
        v.median(),
        v.quantile(0.75, interpolation="linear"),
        v.max(),
        v.skew(),
        (v == 0).sum() / v.len() * 100,
    )


@styled
def plot_category_counts(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    *,
    top_k: int = 10,
) -> Figure:
    """Plots how often each value occurs in every string, categorical and boolean
    column, most frequent on top.

    Past the `top_k` most frequent values, the rest are folded into one "other"
    bar. Nulls get their own bar. `columns` restricts it to these columns.
    """
    df = ensure_collected(data, columns)
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
        .value_counts(sort=True, name="n")
        .rename({s.name: "value"})
    )
    n_values = counts.height
    if n_values > top_k:
        rest = counts.slice(top_k)
        counts = pl.concat(
            [
                counts.head(top_k),
                pl.DataFrame(
                    {"value": [f"other ({rest.height})"], "n": [rest["n"].sum()]},
                    schema=counts.schema,
                ),
            ]
        )
    # reversed, so the most frequent bar ends up on top
    labels = counts["value"].to_list()[::-1]
    values = counts["n"].to_list()[::-1]
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
    # one slot per possible bar, so bars keep their thickness however few there are
    slots = min(top_k, 12) + 1
    ax.set_ylim(len(labels) - slots - 0.5, len(labels) - 0.5)
    ax.grid(axis="y", visible=False)  # type: ignore
    ax.grid(axis="x", visible=True)  # type: ignore
    ax.set_title(f"{s.name}  ·  {n_values} distinct")  # type: ignore
