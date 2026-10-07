import polars as pl

from .._shared import ColumnOrExpr, into_expr


class NumericFeatures:
    def __init__(self): ...

    @staticmethod
    def power(
        lf: pl.LazyFrame,
        name: str,
        column: ColumnOrExpr,
        exponent: int | float,
    ) -> pl.LazyFrame:
        return lf.with_columns(into_expr(column).pow(exponent).alias(name))

    @staticmethod
    def root(
        lf: pl.LazyFrame,
        name: str,
        column: ColumnOrExpr,
        degree: int | float = 2,
    ) -> pl.LazyFrame:
        return lf.with_columns(into_expr(column).pow(1.0 / degree).alias(name))

    @staticmethod
    def log(
        lf: pl.LazyFrame,
        name: str,
        column: ColumnOrExpr,
        plus_one: bool = True,
        base: int | float | None = None,
    ) -> pl.LazyFrame:
        expr = into_expr(column)
        expr = expr.log1p() if plus_one else expr.log()
        if base is not None:
            expr = expr / pl.lit(base).log()
        return lf.with_columns(expr.alias(name))

    @staticmethod
    def absolute(
        lf: pl.LazyFrame,
        name: str,
        column: ColumnOrExpr,
    ) -> pl.LazyFrame:
        return lf.with_columns(into_expr(column).abs().alias(name))

    @staticmethod
    def clip(
        lf: pl.LazyFrame,
        name: str,
        column: ColumnOrExpr,
        lower_bound: int | float | None = None,
        upper_bound: int | float | None = None,
    ) -> pl.LazyFrame:
        return lf.with_columns(
            into_expr(column)
            .clip(lower_bound=lower_bound, upper_bound=upper_bound)
            .alias(name)
        )
