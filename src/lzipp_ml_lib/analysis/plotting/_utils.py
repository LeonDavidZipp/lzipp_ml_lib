from collections.abc import Sequence
from datetime import timedelta
from typing import Any

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
    """Collects LazyFrames into DataFrames and passes DataFrames through. With
    `columns`, only those are selected, before collecting, so a LazyFrame only computes
    what is plotted.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data.
        columns (Sequence[str] | None): Columns to select, in this order. If None, all
            are kept. Defaults to None.
        keep (Sequence[str] | None): Columns that are not in the columns being plotted,
            but needed for the plot nonetheless (i.e. the x-axis labels).

    Returns:
        pl.DataFrame: The collected data.
    """
    if columns is not None:
        data = data.select(*[c for c in (keep or []) if c not in columns], *columns)
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def numeric_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Lists the numeric columns of `df`.

    Args:
        df (pl.DataFrame): The data.
        exclude (tuple[str, ...]): Columns to leave out. Defaults to ().

    Returns:
        list[str]: The numeric columns, in `df`'s order.
    """
    cols = df.select(cs.numeric()).columns
    return [col for col in cols if col not in exclude]


def categorical_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Lists the string, categorical, enum and boolean columns of `df`.

    Args:
        df (pl.DataFrame): The data.
        exclude (tuple[str, ...]): Columns to leave out. Defaults to ().

    Returns:
        list[str]: The categorical columns, in `df`'s order.
    """
    kinds = (pl.String, pl.Categorical, pl.Enum, pl.Boolean)
    return [
        col
        for col, dtype in df.schema.items()
        if isinstance(dtype, kinds) and col not in exclude
    ]


def maybe_sample(df: pl.DataFrame, sample: int | None, seed: int = 0) -> pl.DataFrame:
    """Samples at most `sample` random rows, with a fixed seed so plots are
    reproducible.

    Args:
        df (pl.DataFrame): The data.
        sample (int | None): The maximum number of rows. If None, all rows.
        seed (int): The random seed. Defaults to 0.

    Returns:
        pl.DataFrame: `df`, or a random sample of its rows.
    """
    if sample is not None and df.height > sample:
        return df.sample(sample, seed=seed)
    return df


def panel_values(
    df: pl.DataFrame,
    col: str,
    log: bool = False,
    clip: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Keeps the rows of `df` with a plottable value in `col`: non-null and non-NaN,
    positive on a log scale, and inside the `clip` quantile range.

    Args:
        df (pl.DataFrame): The data.
        col (str): The column to check.
        log (bool): Whether the values go on a log scale. Defaults to False.
        clip (tuple[float, float] | None): The quantile range to keep, e.g. (0.01,
            0.99). Defaults to None.

    Returns:
        pl.DataFrame: The plottable rows.
    """
    sub = df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
    if log:
        sub = sub.filter(pl.col(col) > 0)
    if clip is not None and sub.height > 0:
        low = sub[col].quantile(clip[0], interpolation="linear")
        high = sub[col].quantile(clip[1], interpolation="linear")
        sub = sub.filter(pl.col(col).is_between(low, high))
    return sub


def series_frame(data: PolarsFrame, time_col: str, target_col: str) -> pl.DataFrame:
    """Selects the time and target columns, sorted by time, without null targets.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data.
        time_col (str): The timestamp column.
        target_col (str): The column of the series.

    Returns:
        pl.DataFrame: The two columns, sorted by time.
    """
    return (
        ensure_collected(data, [target_col], keep=(time_col,))
        .filter(pl.col(target_col).is_not_null())
        .sort(time_col)
    )


def infer_interval(ts: pl.Series) -> timedelta:
    """Infers the step of a time series as the most common gap between timestamps.

    Args:
        ts (pl.Series): The timestamps, in any order.

    Returns:
        timedelta: The most common step (the smallest, on a tie).

    Raises:
        ValueError: If there are fewer than two distinct timestamps.
    """
    steps = ts.sort().diff().drop_nulls()
    steps = steps.filter(steps > timedelta(0))
    if steps.len() == 0:
        raise ValueError(f"can't infer a time step from {ts.name!r}: < 2 timestamps")
    return steps.mode().min()  # type: ignore


def default_period(interval: timedelta) -> int:
    """Picks the natural seasonal period, in steps, for data at this interval: a day of
    sub-daily data, a week of daily data, a year of weekly or monthly data, and quarters
    otherwise.

    Args:
        interval (timedelta): The step of the data.

    Returns:
        int: The period, in steps.
    """
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


# what the evaluation plots take for labels, predictions, probabilities and
# timestamps (numpy's ArrayLike alone doesn't cover lists of datetimes)
Values = ArrayLike | Sequence[Any] | pl.Series | pl.DataFrame


def as_1d(values: Values, name: str) -> np.ndarray:
    """Converts labels, predictions or timestamps into a flat array.

    Args:
        values (Values): A list, array, Series or single-column DataFrame (like the `y`
            frames the fit functions take).
        name (str): The argument's name, for error messages.

    Returns:
        np.ndarray: The values as a flat array.

    Raises:
        ValueError: If `values` has more than one column or dimension.
    """
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
    """Converts class probabilities into an (n, n_classes) array.

    Args:
        values (Values): The probabilities, as from `predict_proba`: (n, n_classes), or
            (n,) for the positive class of a binary problem.

    Returns:
        np.ndarray: The probabilities, one column per class.

    Raises:
        ValueError: If `values` isn't (n,) or (n, n_classes) with 2+ classes.
    """
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
    """Checks that arrays have the same length.

    Args:
        **arrays (np.ndarray): The arrays, by the name used in the error message.

    Raises:
        ValueError: If the lengths differ.
    """
    lengths = {name: len(a) for name, a in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"lengths differ: {lengths}")
