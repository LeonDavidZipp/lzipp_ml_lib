from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import polars as pl


class ClassificationFeatures:
    """Feature encoders for tabular classification data.

    `fit` learns the categories of the categorical columns from the training data;
    the encoding methods then apply exactly those categories, so train and test get
    the same encoding and the same columns:

        fe = ClassificationFeatures()
        fe.fit(train, categorical_columns=["city", "device"])
        train_enc = fe.categorical_to_enum(train)
        test_enc = fe.categorical_to_enum(test)
    """

    def __init__(self):
        self._categorical_values: Mapping[str, list[str]] | None = None
        self._bins: Mapping[str, tuple[float, float]] | None = None

    def fit(
        self,
        lf: pl.LazyFrame,
        categorical_columns: Sequence[str] | pl.Expr | None = None,
        n_bins: Mapping[str, int | list[float]] | None = None,
    ) -> Self:
        """Learns the categories of the categorical columns from the training data:
        each column's distinct values, as sorted strings, without nulls and NaNs.

        Collects `lf`.

        Args:
            lf (pl.LazyFrame): Training data.
            categorical_columns (Sequence[str] | pl.Expr | None): The columns to learn the
                categories of, as names or a selector. To choose all categorical and string columns,
                choose `cs.string(include_categorical=True)`, which selects `pl.Categorical`
                and `pl.String` columns.
        Returns:
            Self: Itself.
        """

        cols = (
            list(categorical_columns)
            if isinstance(categorical_columns, Sequence)
            else lf.collect_schema().names()
        )
        cols += list(n_bins.keys()) if n_bins else []
        df = lf.select(cols).collect()
        if categorical_columns is not None:
            self._categorical_values = _get_categorical_values(df, categorical_columns)
        if n_bins:
            self._bins = _get_bins(df, n_bins)
        return self

    def categorical_to_enum(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Casts the fitted columns to enums of their training categories, so a
        category gets the same code in every frame. The column order is kept.

        Args:
            lf (pl.LazyFrame): The data to encode; must contain the fitted columns.

        Returns:
            pl.LazyFrame: `lf` with the fitted columns as `pl.Enum`.

        Raises:
            RuntimeError: If `fit` wasn't called, or found no categorical columns.
            polars.exceptions.InvalidOperationError: When collected, if a column
                holds a category that wasn't in the training data.
        """
        if not self._categorical_values:
            raise RuntimeError(
                "call fit() with `categorical_columns=...` before categorical_to_enum()"
            )
        frame_cols = lf.collect_schema().names()
        return lf.with_columns(
            (
                pl.col(col).cast(pl.Enum(val))
                for col, val in self._categorical_values.items()
            )
        ).select(frame_cols)

    def categorical_to_onehot(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Replaces the fitted columns by one 0/1 column per training category, named
        `{column}_{category}`. The columns come from the training categories, not from
        `lf`, so every frame gets the same ones; categories that weren't in the
        training data, and nulls, become all zeros.

        Args:
            lf (pl.LazyFrame): The data to encode; must contain the fitted columns.

        Returns:
            pl.LazyFrame: `lf` without the fitted columns, with their one-hot
                columns (`pl.UInt8`) appended.

        Raises:
            RuntimeError: If `fit` wasn't called.
        """
        if not self._categorical_values:
            raise RuntimeError(
                "call fit() with `categorical_columns=...` before categorical_to_onehot()"
            )
        return lf.with_columns(
            (pl.col(col).cast(pl.String) == value)
            .fill_null(False)
            .cast(pl.UInt8)
            .alias(f"{col}_{value}")
            for col, values in self._categorical_values.items()
            for value in values
        ).drop(list(self._categorical_values))

    def numeric_to_bin(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if not self._bins:
            raise RuntimeError("call fit() with `n_bins=...` before numeric_to_bin()")
        frame_cols = lf.collect_schema().names()
        return lf.with_columns(
            pl.col(col).cut(edges) for col, edges in self._bins.items()
        ).select(frame_cols)


def _get_bins(
    df: pl.DataFrame, n_bins: Mapping[str, int | Sequence[float]]
) -> dict[str, tuple[float, float]]:
    """Learns the bin edges of each column from the training data.

    Args:
        df (pl.DataFrame): Training data.
        n_bins (Mapping[str, int | Sequence[float]]): Per column, the number of
            equally filled bins, or the quantiles to split at (like `qcut`).

    Returns:
        dict[str, list[float]]: Each column's inner bin edges, sorted, with
            duplicates merged (e.g. when a quarter of the values are 0).
    """
    quantiles = {
        col: np.linspace(0, 1, spec + 1)[1:-1].tolist()
        if isinstance(spec, int)
        else list(spec)
        for col, spec in n_bins.items()
    }
    return df.select(
        pl.concat_list(pl.col(col).quantile(q, interpolation="linear") for q in qs)
        .list.unique()
        .list.sort()
        .alias(col)
        for col, qs in quantiles.items()
    ).row(0, named=True)


def _get_categorical_values(
    df: pl.DataFrame,
    categorical_columns: Sequence[str] | pl.Expr,
) -> dict[str, list[Any]]:
    """Collects the distinct values of the selected columns.

    Args:
        lf (pl.LazyFrame): The data.
        categorical_columns (Sequence[str] | pl.Expr): The columns, as names or a
            selector. Defaults to `cs.string(include_categorical=True)`.

    Returns:
        dict[str, list[Any]]: Each column's distinct values, as sorted strings,
            without nulls and NaNs.
    """
    cols = df.select(categorical_columns).collect_schema().names()
    series = df.select(
        pl.col(col).drop_nulls().drop_nans().unique().cast(pl.String).sort()
        for col in cols
    ).get_columns()
    return {s.name: s.to_list() for s in series}


def to_xgboost(df: pl.DataFrame) -> pl.DataFrame: ...


def numeric_to_bin(): ...
