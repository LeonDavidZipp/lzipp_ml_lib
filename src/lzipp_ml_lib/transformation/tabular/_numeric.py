from collections.abc import Mapping, Sequence
from typing import Literal, Self, TypeVar

import numpy as np
import polars as pl
from scipy.stats import yeojohnson_normmax  # type: ignore

from ..._shared import ColumnOrExpr

ScaleMethod = Literal["standard", "minmax", "robust"]
ImputeStrategy = Literal["median", "mean", "zero"]
LogKind = Literal["log1p", "signed"]
_T = TypeVar("_T")


class NumericFeatures:
    """Transforms for numeric columns whose parameters are learned on the training data.

    `fit` learns everything from the training data (fill values, clip bounds, which
    columns to log, Yeo-Johnson lambdas, scaling parameters); the methods then apply
    exactly that, so train and test are transformed the same way. Computing these per
    frame instead, e.g. scaling the test data by its own mean, is a classic leak.

    The steps work on the same columns one after another, in a fixed order: impute,
    clip, auto_log / power_transform, scale. `fit` learns each step on the output of
    the steps before it (e.g. the scaling parameters after clipping and logging), so
    the steps must be applied in that order too; `transform` does exactly that:

        nf = NumericFeatures().fit(
            train,
            impute=["income", "age"],
            clip=["income"],
            auto_log=True,
            scale=["income", "age"],
        )
        train_num = nf.transform(train)
        test_num = nf.transform(test)

    Each step replaces its columns, or adds new ones with a `suffix`. NaNs count as
    missing everywhere: they're ignored when learning and filled by `impute`.
    """

    def __init__(self) -> None:
        self._impute: dict[str, float] | None = None
        self._clip: dict[str, tuple[float, float]] | None = None
        self._log: dict[str, LogKind] | None = None
        self._power: dict[str, float] | None = None
        self._scale: dict[str, tuple[float, float]] | None = None

    def fit(
        self,
        lf: pl.LazyFrame,
        *,
        impute: Sequence[str] | None = None,
        impute_strategy: ImputeStrategy = "median",
        clip: Sequence[str] | None = None,
        clip_quantiles: tuple[float, float] = (0.01, 0.99),
        auto_log: Sequence[str] | bool = False,
        skew_threshold: float = 1.0,
        power_transform: Sequence[str] | None = None,
        scale: Sequence[str] | None = None,
        scale_method: ScaleMethod = "standard",
    ) -> Self:
        """Learns the parameters of every requested step from the training data, in
        the order the steps are applied: each step is learned on the output of the
        ones before it.

        Collects `lf`.

        Args:
            lf (pl.LazyFrame): Training data.
            impute (Sequence[str] | None): The columns to fill nulls and NaNs of, for
                `impute`. Defaults to None.
            impute_strategy (ImputeStrategy): The fill value: the training
                `"median"` or `"mean"`, or `"zero"`. Defaults to `"median"`.
            clip (Sequence[str] | None): The columns to clip (winsorize), for `clip`.
                Defaults to None.
            clip_quantiles (tuple[float, float]): The training quantiles to clip at.
                Defaults to (0.01, 0.99).
            auto_log (Sequence[str] | bool): The columns `auto_log` may log; True for
                every numeric column. Of these, only the ones with a skew above
                `skew_threshold` are logged. Defaults to False.
            skew_threshold (float): The skew above which `auto_log` logs a column.
                Defaults to 1.0.
            power_transform (Sequence[str] | None): The columns to Yeo-Johnson
                transform, for `power_transform`. Defaults to None.
            scale (Sequence[str] | None): The columns to scale, for `scale`. Defaults
                to None.
            scale_method (ScaleMethod): `"standard"` (minus the mean, divided by the
                std), `"minmax"` (onto 0 to 1) or `"robust"` (minus the median,
                divided by the interquartile range; little affected by outliers).
                Defaults to `"standard"`.

        Returns:
            Self: Itself.

        Raises:
            ValueError: If a column is in both `auto_log` and `power_transform`, if a
                requested column isn't numeric, or if `clip_quantiles` aren't two
                increasing values in [0, 1].
        """
        low, high = clip_quantiles
        if not 0 <= low < high <= 1:
            raise ValueError(
                f"clip_quantiles must increase within [0, 1], got {clip_quantiles}"
            )
        schema = lf.collect_schema()
        numeric = [col for col, dtype in schema.items() if dtype.is_numeric()]
        log_candidates = numeric if auto_log is True else list(auto_log or [])
        if both := set(log_candidates) & set(power_transform or []):
            if auto_log is not True:
                raise ValueError(
                    f"columns in both auto_log and power_transform: {sorted(both)}"
                )
            # with auto_log=True, power_transform's columns take precedence
            log_candidates = [col for col in log_candidates if col not in both]
        requested = [
            *(impute or []),
            *(clip or []),
            *log_candidates,
            *(power_transform or []),
            *(scale or []),
        ]
        if not_numeric := sorted({col for col in requested if col not in numeric}):
            raise ValueError(f"not numeric columns: {not_numeric}")
        df = (
            lf.select(list(dict.fromkeys(requested)))
            .collect()
            .with_columns(pl.all().cast(pl.Float64).fill_nan(None))
        )

        if impute:
            self._impute = _learn_impute(df, impute, impute_strategy)
            df = df.with_columns(_impute_exprs(self._impute, suffix=None))
        if clip:
            self._clip = _learn_clip(df, clip, low, high)
            df = df.with_columns(_clip_exprs(self._clip, suffix=None))
        if auto_log is not False:
            self._log = _learn_log(df, log_candidates, skew_threshold)
            df = df.with_columns(_log_exprs(self._log, suffix=None))
        if power_transform:
            self._power = _learn_power(df, power_transform)
            df = df.with_columns(_power_exprs(self._power, suffix=None))
        if scale:
            self._scale = _learn_scale(df, scale, scale_method)
        return self

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Applies every fitted step, in the order `fit` learned them: impute, clip,
        auto_log, power_transform, scale.

        Args:
            lf (pl.LazyFrame): The data; must contain the fitted columns.

        Returns:
            pl.LazyFrame: `lf` with the fitted columns transformed in place.

        Raises:
            RuntimeError: If `fit` learned no step.
        """
        steps = [
            (self._impute, self.impute),
            (self._clip, self.clip),
            (self._log, self.auto_log),
            (self._power, self.power_transform),
            (self._scale, self.scale),
        ]
        if all(fitted is None for fitted, _ in steps):
            raise RuntimeError("call fit() with at least one step before transform()")
        for fitted, step in steps:
            if fitted is not None:
                lf = step(lf)
        return lf

    def impute(self, lf: pl.LazyFrame, *, suffix: str | None = None) -> pl.LazyFrame:
        """Fills nulls and NaNs with the value learned from the training data.

        Args:
            lf (pl.LazyFrame): The data; must contain the fitted columns.
            suffix (str | None): If given, the filled columns are added as
                `{column}{suffix}` and the originals kept. Defaults to None.

        Returns:
            pl.LazyFrame: `lf` with the fitted columns filled (`pl.Float64`).

        Raises:
            RuntimeError: If `fit` wasn't called with `impute=...`.
        """
        return lf.with_columns(_impute_exprs(_fitted(self._impute, "impute"), suffix))

    def clip(self, lf: pl.LazyFrame, *, suffix: str | None = None) -> pl.LazyFrame:
        """Clips (winsorizes) values to the bounds learned from the training data, so
        extreme values, including ones beyond anything in the training data, don't
        dominate. Nulls stay null.

        Args:
            lf (pl.LazyFrame): The data; must contain the fitted columns.
            suffix (str | None): If given, the clipped columns are added as
                `{column}{suffix}` and the originals kept. Defaults to None.

        Returns:
            pl.LazyFrame: `lf` with the fitted columns clipped (`pl.Float64`).

        Raises:
            RuntimeError: If `fit` wasn't called with `clip=...`.
        """
        return lf.with_columns(_clip_exprs(_fitted(self._clip, "clip"), suffix))

    def auto_log(self, lf: pl.LazyFrame, *, suffix: str | None = None) -> pl.LazyFrame:
        """Logs the columns `fit` found right-skewed: `log1p` for columns without
        negative training values, and the signed `sign(x) * log1p(|x|)` for columns
        with some, which keeps the sign and is defined everywhere. Columns that
        weren't skewed are left as they are.

        Args:
            lf (pl.LazyFrame): The data; must contain the fitted columns.
            suffix (str | None): If given, the logged columns are added as
                `{column}{suffix}` and the originals kept. Defaults to None.

        Returns:
            pl.LazyFrame: `lf` with the skewed columns logged (`pl.Float64`).

        Raises:
            RuntimeError: If `fit` wasn't called with `auto_log=...`.
        """
        return lf.with_columns(_log_exprs(_fitted(self._log, "auto_log"), suffix))

    def power_transform(
        self, lf: pl.LazyFrame, *, suffix: str | None = None
    ) -> pl.LazyFrame:
        """Applies the Yeo-Johnson transform with each column's lambda learned from
        the training data, which makes a column as close to normally distributed as a
        power transform can. Unlike a log, it also handles zeros and negatives, and
        it picks the strength of the transform per column. It keeps the order of the
        values.

        Args:
            lf (pl.LazyFrame): The data; must contain the fitted columns.
            suffix (str | None): If given, the transformed columns are added as
                `{column}{suffix}` and the originals kept. Defaults to None.

        Returns:
            pl.LazyFrame: `lf` with the fitted columns transformed (`pl.Float64`).

        Raises:
            RuntimeError: If `fit` wasn't called with `power_transform=...`.
        """
        return lf.with_columns(
            _power_exprs(_fitted(self._power, "power_transform"), suffix)
        )

    def scale(self, lf: pl.LazyFrame, *, suffix: str | None = None) -> pl.LazyFrame:
        """Scales the columns with the parameters learned from the training data
        (see `scale_method` in `fit`). A column that was constant in the training data
        is only shifted, not divided by its zero spread.

        Args:
            lf (pl.LazyFrame): The data; must contain the fitted columns.
            suffix (str | None): If given, the scaled columns are added as
                `{column}{suffix}` and the originals kept. Defaults to None.

        Returns:
            pl.LazyFrame: `lf` with the fitted columns scaled (`pl.Float64`).

        Raises:
            RuntimeError: If `fit` wasn't called with `scale=...`.
        """
        return lf.with_columns(_scale_exprs(_fitted(self._scale, "scale"), suffix))


def signed_log1p_expr(column: ColumnOrExpr) -> pl.Expr:
    """Builds `sign(x) * log1p(|x|)`: like `log1p` for positive values, mirrored for
    negative ones, so it's defined everywhere and keeps the sign and order of the
    values.

    Args:
        column (str | pl.Expr): The values, as a column name or an expression.

    Returns:
        pl.Expr: The signed log, as `pl.Float64`; null where the value is null.
    """
    x = pl.col(column) if isinstance(column, str) else column
    return x.sign() * x.abs().log1p()


def _fitted(state: _T | None, step: str) -> _T:
    if state is None:
        raise RuntimeError(f"call fit() with `{step}=...` before {step}()")
    return state


def _named(col: str, expr: pl.Expr, suffix: str | None) -> pl.Expr:
    return expr.alias(col if suffix is None else f"{col}{suffix}")


def _as_float(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64).fill_nan(None)


# ---- impute --------------------------------------------------------------------------
def _learn_impute(
    df: pl.DataFrame, columns: Sequence[str], strategy: ImputeStrategy
) -> dict[str, float]:
    if strategy == "zero":
        return dict.fromkeys(columns, 0.0)
    stat = {"median": pl.median, "mean": pl.mean}[strategy]
    values = df.select(stat(col) for col in columns).row(0, named=True)
    return {
        col: 0.0 if value is None else float(value) for col, value in values.items()
    }


def _impute_exprs(fills: Mapping[str, float], suffix: str | None) -> list[pl.Expr]:
    return [
        _named(col, _as_float(col).fill_null(fill), suffix)
        for col, fill in fills.items()
    ]


# ---- clip ----------------------------------------------------------------------------
def _learn_clip(
    df: pl.DataFrame, columns: Sequence[str], low: float, high: float
) -> dict[str, tuple[float, float]]:
    bounds = df.select(
        pl.concat_list(
            pl.col(col).quantile(low, interpolation="linear"),
            pl.col(col).quantile(high, interpolation="linear"),
        ).alias(col)
        for col in columns
    ).row(0, named=True)
    return {col: (lo, hi) for col, (lo, hi) in bounds.items()}


def _clip_exprs(
    bounds: Mapping[str, tuple[float, float]], suffix: str | None
) -> list[pl.Expr]:
    return [
        _named(col, _as_float(col).clip(lo, hi), suffix)
        for col, (lo, hi) in bounds.items()
    ]


# ---- auto_log ------------------------------------------------------------------------
def _learn_log(
    df: pl.DataFrame, columns: Sequence[str], skew_threshold: float
) -> dict[str, LogKind]:
    if not columns:
        return {}
    stats = df.select(
        pl.struct(skew=pl.col(col).skew(), min=pl.col(col).min()).alias(col)
        for col in columns
    ).row(0, named=True)
    return {
        col: "signed" if stat["min"] < 0 else "log1p"
        for col, stat in stats.items()
        if stat["skew"] is not None and stat["skew"] > skew_threshold
    }


def _log_exprs(kinds: Mapping[str, LogKind], suffix: str | None) -> list[pl.Expr]:
    return [
        _named(
            col,
            signed_log1p_expr(_as_float(col))
            if kind == "signed"
            else _as_float(col).log1p(),
            suffix,
        )
        for col, kind in kinds.items()
    ]


# ---- power_transform -----------------------------------------------------------------
def _learn_power(df: pl.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    lambdas: dict[str, float] = {}
    for col in columns:
        values = df.get_column(col).drop_nulls().to_numpy()
        magnitude = max(1.0, float(np.abs(values).max())) if len(values) else 1.0
        varies = len(values) > 1 and float(np.ptp(values)) > 1e-9 * magnitude
        lambdas[col] = float(yeojohnson_normmax(values)) if varies else 1.0
    return lambdas


def _yeo_johnson(x: pl.Expr, lmbda: float) -> pl.Expr:
    """The Yeo-Johnson transform: a Box-Cox-like power transform that also handles
    zeros and negatives, with separate formulas on either side of 0."""
    if abs(lmbda) < 1e-12:
        positive = x.log1p()
    else:
        positive = ((x + 1).pow(lmbda) - 1) / lmbda
    if abs(lmbda - 2) < 1e-12:
        negative = -(-x).log1p()
    else:
        negative = -((1 - x).pow(2 - lmbda) - 1) / (2 - lmbda)
    return pl.when(x >= 0).then(positive).otherwise(negative)


def _power_exprs(lambdas: Mapping[str, float], suffix: str | None) -> list[pl.Expr]:
    return [
        _named(col, _yeo_johnson(_as_float(col), lmbda), suffix)
        for col, lmbda in lambdas.items()
    ]


# ---- scale ---------------------------------------------------------------------------
def _learn_scale(
    df: pl.DataFrame, columns: Sequence[str], method: ScaleMethod
) -> dict[str, tuple[float, float]]:
    """Per column, `(center, spread)`, so scaling is `(x - center) / spread`."""
    if method == "standard":
        centers = [pl.col(col).mean() for col in columns]
        spreads = [pl.col(col).std() for col in columns]
    elif method == "minmax":
        centers = [pl.col(col).min() for col in columns]
        spreads = [pl.col(col).max() - pl.col(col).min() for col in columns]
    else:
        centers = [pl.col(col).median() for col in columns]
        spreads = [
            pl.col(col).quantile(0.75, interpolation="linear")
            - pl.col(col).quantile(0.25, interpolation="linear")
            for col in columns
        ]
    row = df.select(
        pl.concat_list(center, spread).alias(col)
        for col, center, spread in zip(columns, centers, spreads, strict=True)
    ).row(0, named=True)
    return {
        col: (center or 0.0, spread if spread is not None and spread > 0 else 1.0)
        for col, (center, spread) in row.items()
    }


def _scale_exprs(
    params: Mapping[str, tuple[float, float]], suffix: str | None
) -> list[pl.Expr]:
    return [
        _named(col, (_as_float(col) - center) / spread, suffix)
        for col, (center, spread) in params.items()
    ]
