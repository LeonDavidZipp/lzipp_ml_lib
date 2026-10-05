from collections.abc import Sequence
from typing import Any

import polars as pl

from .plotting._utils import PolarsFrame, ensure_collected

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
