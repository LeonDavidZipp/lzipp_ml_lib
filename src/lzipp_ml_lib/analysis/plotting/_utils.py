from collections.abc import Sequence

import polars as pl

PolarsFrame = pl.DataFrame | pl.LazyFrame


def ensure_collected(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    keep: Sequence[str] = (),
) -> pl.DataFrame:
    """Evaluates LazyFrames to DataFrames, passes DataFrames through.

    With `columns`, only those columns (plus any in `keep` that aren't among them)
    are selected, before collecting, so a LazyFrame only computes what is plotted.
    """
    if columns is not None:
        data = data.select(*[c for c in keep if c not in columns], *columns)
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def numeric_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    return [c for c in df.columns if c not in exclude and df[c].dtype.is_numeric()]


def categorical_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """String, categorical, enum and boolean columns."""
    kinds = (pl.String, pl.Categorical, pl.Enum, pl.Boolean)
    return [
        c for c in df.columns if c not in exclude and isinstance(df[c].dtype, kinds)
    ]


def maybe_sample(df: pl.DataFrame, sample: int | None) -> pl.DataFrame:
    """At most `sample` random rows (fixed seed, so plots are reproducible)."""
    if sample is not None and df.height > sample:
        return df.sample(sample, seed=0)
    return df


def panel_values(
    df: pl.DataFrame,
    col: str,
    log: bool = False,
    clip: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """`df`'s rows with a plottable value in `col`: non-null, positive on a log
    scale, and inside the `clip` quantile range (e.g. (0.01, 0.99))."""
    sub = df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
    if log:
        sub = sub.filter(pl.col(col) > 0)
    if clip is not None and sub.height > 0:
        low = sub[col].quantile(clip[0], interpolation="linear")
        high = sub[col].quantile(clip[1], interpolation="linear")
        sub = sub.filter(pl.col(col).is_between(low, high))
    return sub
