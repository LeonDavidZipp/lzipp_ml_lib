from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import ArrayLike

PolarsFrame = pl.DataFrame | pl.LazyFrame


def ensure_collected(
    data: PolarsFrame,
    columns: Sequence[str] | None = None,
    keep: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Evaluates LazyFrames to DataFrames, passes DataFrames through.

    With `columns`, only those columns (plus any in `keep` that aren't among them)
    are selected, before collecting, so a LazyFrame only computes what is plotted.
    """
    if columns is not None:
        data = data.select(*[c for c in (keep or []) if c not in columns], *columns)
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def numeric_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    cols = df.select(cs.numeric()).columns
    return [col for col in cols if col not in exclude]


def categorical_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """String, categorical, enum and boolean columns."""
    kinds = (pl.String, pl.Categorical, pl.Enum, pl.Boolean)
    return [
        col
        for col, dtype in df.schema.items()
        if isinstance(dtype, kinds) and col not in exclude
    ]


def maybe_sample(df: pl.DataFrame, sample: int | None, seed: int = 0) -> pl.DataFrame:
    """At most `sample` random rows (fixed seed, so plots are reproducible)."""
    if sample is not None and df.height > sample:
        return df.sample(sample, seed=seed)
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


def series_frame(data: PolarsFrame, time_col: str, target_col: str) -> pl.DataFrame:
    """`time_col` and `target_col` only, sorted by time, without null targets."""
    return (
        ensure_collected(data, [target_col], keep=(time_col,))
        .filter(pl.col(target_col).is_not_null())
        .sort(time_col)
    )


def infer_interval(ts: pl.Series) -> timedelta:
    """The most common step between consecutive timestamps."""
    steps = ts.sort().diff().drop_nulls()
    steps = steps.filter(steps > timedelta(0))
    if steps.len() == 0:
        raise ValueError(f"can't infer a time step from {ts.name!r}: < 2 timestamps")
    return steps.mode().min()  # type: ignore


def default_period(interval: timedelta) -> int:
    """The natural seasonal period, in steps, for data at this interval: a day of
    sub-daily data, a week of daily data, a year of weekly or monthly data."""
    day = timedelta(days=1)
    if interval < day:
        return max(2, round(day / interval))
    if interval < timedelta(days=2):
        return 7
    if interval < timedelta(days=14):
        return 52
    if interval < timedelta(days=60):
        return 12
    return 4


# what the evaluation plots take for labels, predictions and probabilities
Values = ArrayLike | pl.Series | pl.DataFrame


def as_1d(values: Values, name: str) -> np.ndarray:
    """A flat array from a list, array, Series or single-column DataFrame (like the
    `y` frames the fit functions take)."""
    if isinstance(values, pl.DataFrame):
        if values.width != 1:
            raise ValueError(f"{name} needs exactly one column, got {values.width}")
        values = values.to_series()
    array = values.to_numpy() if isinstance(values, pl.Series) else np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def as_proba(values: Values) -> np.ndarray:
    """Class probabilities as an (n, n_classes) array. A flat input is taken as the
    positive class's probability of a binary problem."""
    array = values.to_numpy() if isinstance(values, pl.DataFrame) else None
    if array is None:
        array = (
            values.to_numpy() if isinstance(values, pl.Series) else np.asarray(values)
        )
    if array.ndim == 1:
        return np.column_stack([1 - array, array])
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"y_proba must be (n,) or (n, n_classes), got {array.shape}")
    return array


def same_length(**arrays: np.ndarray) -> None:
    lengths = {name: len(a) for name, a in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"lengths differ: {lengths}")
