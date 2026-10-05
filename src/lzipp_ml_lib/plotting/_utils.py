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
