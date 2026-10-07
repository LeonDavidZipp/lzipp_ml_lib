from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

_ColumnOrExpr = str | pl.Expr


@dataclass(frozen=True)
class DifferenceSpec:
    minuend: _ColumnOrExpr
    subtrahend: _ColumnOrExpr


@dataclass(frozen=True)
class ProductSpec:
    factors: Sequence[_ColumnOrExpr]
    scale: int | float = 1


@dataclass(frozen=True)
class RatioSpec:
    dividend: _ColumnOrExpr
    divisor: _ColumnOrExpr


class CrossFeatures:
    def sum(
        self, lf: pl.LazyFrame, name: str, addends: Sequence[_ColumnOrExpr]
    ) -> pl.LazyFrame:
        return lf.with_columns(sum_expr(addends).alias(name))

    def difference(
        self,
        lf: pl.LazyFrame,
        name: str,
        minuend: _ColumnOrExpr,
        subtrahend: _ColumnOrExpr,
    ) -> pl.LazyFrame:
        return lf.with_columns(difference_expr(minuend, subtrahend).alias(name))

    def product(
        self,
        lf: pl.LazyFrame,
        name: str,
        factors: Sequence[_ColumnOrExpr],
        scale: int | float = 1,
    ) -> pl.LazyFrame:
        return lf.with_columns(product_expr(factors, scale).alias(name))

    def ratio(
        self,
        lf: pl.LazyFrame,
        name: str,
        dividend: _ColumnOrExpr,
        divisor: _ColumnOrExpr,
    ) -> pl.LazyFrame:
        return lf.with_columns(ratio_expr(dividend, divisor).alias(name))

    def custom_expression(
        self, lf: pl.LazyFrame, name: str, expr: pl.Expr
    ) -> pl.LazyFrame:
        return lf.with_columns(expr.alias(name))

    def sum_multi(
        self,
        lf: pl.LazyFrame,
        spec: Mapping[str, Sequence[_ColumnOrExpr]],
    ) -> pl.LazyFrame:
        return lf.with_columns(
            sum_expr(addends).alias(name) for name, addends in spec.items()
        )

    def difference_multi(
        self,
        lf: pl.LazyFrame,
        spec: Mapping[str, tuple[_ColumnOrExpr, _ColumnOrExpr]],
    ) -> pl.LazyFrame:
        return lf.with_columns(
            difference_expr(minuend, subtrahend).alias(name)
            for name, (minuend, subtrahend) in spec.items()
        )

    def product_multi(
        self,
        lf: pl.LazyFrame,
        spec: Mapping[str, ProductSpec],
    ) -> pl.LazyFrame:
        exprs = [
            product_expr(item.factors, item.scale).alias(name)
            for name, item in spec.items()
        ]
        return lf.with_columns(exprs)

    def ratio_multi(
        self,
        lf: pl.LazyFrame,
        spec: Mapping[str, tuple[_ColumnOrExpr, _ColumnOrExpr]],
    ) -> pl.LazyFrame:
        return lf.with_columns(
            ratio_expr(dividend, divisor).alias(name)
            for name, (dividend, divisor) in spec.items()
        )


def _into_expr(val: _ColumnOrExpr) -> pl.Expr:
    """Converts a column name or Polars expression into a pl.Expr."""
    return pl.col(val) if isinstance(val, str) else val


def sum_expr(addends: Sequence[_ColumnOrExpr]) -> pl.Expr:
    exprs = [_into_expr(col) for col in addends]
    return pl.fold(pl.lit(0.0), lambda x, y: x + y, exprs=exprs)


def difference_expr(
    minuend: _ColumnOrExpr,
    subtrahend: _ColumnOrExpr,
) -> pl.Expr:
    return _into_expr(minuend) - _into_expr(subtrahend)


def product_expr(
    factors: Sequence[_ColumnOrExpr],
    scale: int | float = 1,
) -> pl.Expr:
    exprs = [_into_expr(col) for col in factors]
    return pl.fold(pl.lit(scale), lambda x, y: x * y, exprs=exprs)


def ratio_expr(
    dividend: _ColumnOrExpr,
    divisor: _ColumnOrExpr,
) -> pl.Expr:
    return _into_expr(dividend) - _into_expr(divisor)
