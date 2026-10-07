from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import polars as pl


class ClassificationFeatures:
    """Feature encoders for tabular classification data.

    `fit` learns everything the encoders need from the training data (categories, bin
    edges, class rates, shares, which columns have missing values); the encoding
    methods then apply exactly that, so train and test get the same encoding and the
    same columns:

        fe = ClassificationFeatures().fit(
            train,
            categorical_columns=["device"],
            target="y",
            target_encode=["city"],
            frequency_encode=["city"],
            missing_indicators=True,
        )
        train_enc = (
            fe.target_encode(train, folds=5)  # out-of-fold on the training data
            .pipe(fe.frequency_encode)
            .pipe(fe.missing_indicators)
            .pipe(fe.categorical_to_enum)
        )
        test_enc = (
            fe.target_encode(test)  # the learned rates on new data
            .pipe(fe.frequency_encode)
            .pipe(fe.missing_indicators)
            .pipe(fe.categorical_to_enum)
        )
    """

    def __init__(self):
        self._categorical_values: Mapping[str, list[str]] | None = None
        self._bins: Mapping[str, list[float]] | None = None
        self._target: str | None = None
        self._target_classes: list[Any] = []
        self._target_encodings: dict[str, _TargetEncoding] | None = None
        self._smoothing = 10.0
        self._frequencies: dict[str, _Frequencies] | None = None
        self._missing_columns: list[str] | None = None

    def fit(
        self,
        lf: pl.LazyFrame,
        categorical_columns: Sequence[str] | pl.Expr | None = None,
        n_bins: Mapping[str, int | list[float]] | None = None,
        *,
        target: str | None = None,
        target_encode: Sequence[str] | None = None,
        smoothing: float = 10.0,
        frequency_encode: Sequence[str] | None = None,
        missing_indicators: Sequence[str] | bool = False,
    ) -> Self:
        """Learns the categories of the categorical columns from the training data:
        each column's distinct values, as sorted strings, without nulls and NaNs.

        Collects `lf`.

        Args:
            lf (pl.LazyFrame): Training data.
            categorical_columns (Sequence[str] | pl.Expr | None): The columns to learn
                the categories of, as names or a selector. To choose all categorical and
                string columns, choose `cs.string(include_categorical=True)`, which
                selects `pl.Categorical` and `pl.String` columns.
            n_bins (Mapping[str, int | list[float]] | None): Per column, the number of
                equally filled bins, or the quantiles to split at, for `numeric_to_bin`.
                Defaults to None.
            target (str | None): The class column, needed for `target_encode`.
                Defaults to None.
            target_encode (Sequence[str] | None): The columns to learn the per-category
                class rates of, for `target_encode`. Defaults to None.
            smoothing (float): How many rows' worth of weight the overall class rate
                gets in every category's rate, so rare categories are pulled towards
                it instead of taking the rate of their few rows. Defaults to 10.0.
            frequency_encode (Sequence[str] | None): The columns to learn the
                per-category shares of, for `frequency_encode`. Defaults to None.
            missing_indicators (Sequence[str] | bool): The columns to flag missing
                values of, for `missing_indicators`; True for every column with nulls
                (or NaNs) in the training data. Defaults to False.

        Returns:
            Self: Itself.

        Raises:
            ValueError: If `target_encode` is given without `target`, or the target has
                fewer than two classes.
        """
        if target_encode and target is None:
            raise ValueError("target_encode needs a target")
        df = lf.select(
            _needed_columns(
                lf,
                categorical_columns,
                n_bins,
                target if target_encode else None,
                target_encode,
                frequency_encode,
                missing_indicators,
            )
        ).collect()
        if categorical_columns is not None:
            self._categorical_values = _get_categorical_values(df, categorical_columns)
        if n_bins:
            self._bins = _get_bins(df, n_bins)
        if target_encode and target is not None:
            self._target = target
            self._target_classes = _target_classes(df.get_column(target))
            self._smoothing = smoothing
            labelled = df.filter(pl.col(target).is_not_null())
            self._target_encodings = {
                col: _learn_target_encoding(
                    labelled, col, target, self._target_classes, smoothing
                )
                for col in target_encode
            }
        if frequency_encode:
            self._frequencies = {
                col: _learn_frequencies(df.get_column(col)) for col in frequency_encode
            }
        if missing_indicators is True:
            self._missing_columns = [
                col for col in df.columns if _missing_series(df.get_column(col)).any()
            ]
        elif missing_indicators:
            self._missing_columns = list(missing_indicators)
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
                "call fit() with `categorical_columns=...` before "
                + "categorical_to_onehot()"
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

    def target_encode(
        self, lf: pl.LazyFrame, *, folds: int | None = None, seed: int = 0
    ) -> pl.LazyFrame:
        """Adds each fitted column's smoothed class rate per category: the share of
        training rows of that category in a class, pulled towards the overall rate by
        `smoothing` rows' worth of weight. A binary target gets one column,
        `{column}_te`, the rate of the positive (second) class; a multiclass target one
        column per class, `{column}_te_{class}`. The original columns are kept.

        On new data (`folds=None`), the rates learned by `fit` are used; categories that
        weren't in the training data, and nulls, get the overall rate. On the training
        data itself, pass `folds`: each row is then encoded from the other folds only,
        so it never sees its own label. Without that, the encoding leaks the label and
        the model overrates the column.

        Args:
            lf (pl.LazyFrame): The data to encode. With `folds`, it must contain the
                target column.
            folds (int | None): The number of folds for out-of-fold encoding of the
                training data. If None, the rates learned by `fit` are used. Defaults
                to None.
            seed (int): The seed of the random fold assignment. Defaults to 0.

        Returns:
            pl.LazyFrame: `lf` with the encoding columns (`pl.Float64`) appended.

        Raises:
            RuntimeError: If `fit` wasn't called with `target_encode=...`.
            ValueError: If `folds` is below 2.
        """
        if not self._target_encodings or self._target is None:
            raise RuntimeError(
                "call fit() with `target=...` and `target_encode=...` before "
                "target_encode()"
            )
        if folds is None:
            return lf.with_columns(
                expr
                for col, encoding in self._target_encodings.items()
                for expr in encoding.apply(col)
            )
        if folds < 2:
            raise ValueError(f"folds must be at least 2, got {folds}")
        return _out_of_fold_encode(
            lf,
            list(self._target_encodings),
            self._target,
            self._target_classes,
            self._smoothing,
            folds,
            seed,
        )

    def frequency_encode(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Adds each fitted column's category share in the training data, as
        `{column}_freq`. Categories that weren't in the training data get 0; nulls get
        the training data's share of nulls. The original columns are kept.

        Args:
            lf (pl.LazyFrame): The data to encode; must contain the fitted columns.

        Returns:
            pl.LazyFrame: `lf` with the share columns (`pl.Float64`) appended.

        Raises:
            RuntimeError: If `fit` wasn't called with `frequency_encode=...`.
        """
        if not self._frequencies:
            raise RuntimeError(
                "call fit() with `frequency_encode=...` before frequency_encode()"
            )
        return lf.with_columns(
            frequencies.apply(col) for col, frequencies in self._frequencies.items()
        )

    def missing_indicators(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Adds a 0/1 column `{column}_missing` per fitted column, 1 where the value is
        null (or NaN). The columns come from `fit`, so every frame gets the same ones,
        even where it has no missing values. The original columns are kept.

        Args:
            lf (pl.LazyFrame): The data; must contain the fitted columns.

        Returns:
            pl.LazyFrame: `lf` with the indicator columns (`pl.UInt8`) appended.

        Raises:
            RuntimeError: If `fit` wasn't called with `missing_indicators=...`.
        """
        if self._missing_columns is None:
            raise RuntimeError(
                "call fit() with `missing_indicators=...` before missing_indicators()"
            )
        schema = lf.collect_schema()
        return lf.with_columns(
            _missing_expr(col, schema[col]).cast(pl.UInt8).alias(f"{col}_missing")
            for col in self._missing_columns
        )


@dataclass(frozen=True)
class _TargetEncoding:
    """The smoothed class rates of one column's categories."""

    output_columns: list[str]
    category_rate_mappings: list[dict[str, float]]  # category -> rate, per class
    global_means: list[float]  # overall rate, per class (synonym: prior)

    def apply(self, col: str) -> Iterator[pl.Expr]:
        key = pl.col(col).cast(pl.String)
        for col, rates, global_mean in zip(
            self.output_columns,
            self.category_rate_mappings,
            self.global_means,
            strict=True,
        ):
            yield key.replace_strict(
                rates, default=global_mean, return_dtype=pl.Float64
            ).alias(col)


@dataclass(frozen=True)
class _Frequencies:
    """The share of each of one column's categories in the training data."""

    shares: dict[str, float]
    null_share: float

    def apply(self, col: str) -> pl.Expr:
        key = pl.col(col).cast(pl.String)
        return (
            pl.when(key.is_null())
            .then(self.null_share)
            .otherwise(
                key.replace_strict(self.shares, default=0.0, return_dtype=pl.Float64)
            )
            .alias(f"{col}_freq")
        )


def _needed_columns(
    lf: pl.LazyFrame,
    categorical_columns: Sequence[str] | pl.Expr | None,
    n_bins: Mapping[str, Any] | None,
    target: str | None,
    target_encode: Sequence[str] | None,
    frequency_encode: Sequence[str] | None,
    missing_indicators: Sequence[str] | bool,
) -> list[str]:
    """The columns `fit` needs, so only those are collected; all of them when a
    selector or `missing_indicators=True` has to see every column."""
    if isinstance(categorical_columns, pl.Expr) or missing_indicators is True:
        return lf.collect_schema().names()
    needed = [
        *(categorical_columns or []),
        *(n_bins or {}),
        *([target] if target else []),
        *(target_encode or []),
        *(frequency_encode or []),
        *(missing_indicators or []),
    ]
    return list(dict.fromkeys(needed))


def _target_classes(target: pl.Series) -> list[Any]:
    """The classes to encode: only the positive (second) one of a binary target."""
    classes = target.drop_nulls().unique().sort().to_list()
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError(f"the target needs at least two classes, got {classes}")
    return classes[1:] if n_classes == 2 else classes


def _encoding_names(col: str, classes: list[Any], binary: bool) -> list[str]:
    return [f"{col}_te"] if binary else [f"{col}_te_{c}" for c in classes]


def _learn_target_encoding(
    df: pl.DataFrame, col: str, target: str, classes: list[Any], smoothing: float
) -> _TargetEncoding:
    is_class = [pl.col(target).eq(c).cast(pl.Float64) for c in classes]
    global_means = df.select(
        e.mean().alias(f"p{i}") for i, e in enumerate(is_class)
    ).row(0)
    stats = (
        df.filter(pl.col(col).is_not_null())
        .group_by(key=pl.col(col).cast(pl.String))
        .agg(
            pl.len().alias("n"),
            *(e.sum().alias(f"s{i}") for i, e in enumerate(is_class)),
        )
    )
    rates = [
        dict(
            zip(
                stats["key"],
                (stats[f"s{i}"] + smoothing * global_mean) / (stats["n"] + smoothing),
                strict=True,
            )
        )
        for i, global_mean in enumerate(global_means)
    ]
    names = _encoding_names(col, classes, binary=len(classes) == 1)
    return _TargetEncoding(names, rates, list(global_means))


def _out_of_fold_encode(
    lf: pl.LazyFrame,
    columns: list[str],
    target: str,
    classes: list[Any],
    smoothing: float,
    folds: int,
    seed: int,
) -> pl.LazyFrame:
    """Encodes every row from the rows of the other folds: per category, the totals
    minus the row's own fold."""
    fold = "__fold"
    lf = lf.with_columns(
        (pl.int_range(pl.len()).shuffle(seed=seed) % folds).alias(fold)
    )
    labelled = pl.col(target).is_not_null()
    exprs: list[pl.Expr] = []
    for col in columns:
        key = pl.col(col).cast(pl.String)
        names = _encoding_names(col, classes, binary=len(classes) == 1)
        for name, c in zip(names, classes, strict=True):
            hit = (pl.col(target).eq(c) & labelled).cast(pl.Float64)
            seen = labelled.cast(pl.Float64)
            global_mean = (hit.sum() - hit.sum().over(fold)) / (
                seen.sum() - seen.sum().over(fold)
            )
            hits = hit.sum().over(key) - hit.sum().over(key, fold)
            rows = seen.sum().over(key) - seen.sum().over(key, fold)
            rate = (hits + smoothing * global_mean) / (rows + smoothing)
            exprs.append(
                pl.when(key.is_null()).then(global_mean).otherwise(rate).alias(name)
            )
    return lf.with_columns(exprs).drop(fold)


def _learn_frequencies(s: pl.Series) -> _Frequencies:
    n = max(s.len(), 1)
    counts = (
        s.drop_nulls()
        .cast(pl.String)
        .value_counts(name="n")
        .with_columns(share=pl.col("n") / n)
    ).to_dict()
    shares = dict(zip(counts[s.name], counts["share"], strict=True))
    return _Frequencies(shares, s.null_count() / n)


def _missing_series(s: pl.Series) -> pl.Series:
    return s.is_null() | s.is_nan() if s.dtype.is_float() else s.is_null()


def _missing_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    missing = pl.col(col).is_null()
    return missing | pl.col(col).is_nan() if dtype.is_float() else missing


def _get_bins(
    df: pl.DataFrame, n_bins: Mapping[str, int | Sequence[float]]
) -> dict[str, list[float]]:
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
