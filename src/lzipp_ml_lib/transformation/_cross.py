from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from .._shared import ColumnOrExpr, into_expr


@dataclass(frozen=True)
class DifferenceSpec:
    """A difference to compute: `minuend - subtrahend`.

    Args:
        minuend (ColumnOrExpr): The value subtracted from, as a column name or an
            expression.
        subtrahend (ColumnOrExpr): The value subtracted, as a column name or an
            expression.
    """

    minuend: ColumnOrExpr
    subtrahend: ColumnOrExpr


@dataclass(frozen=True)
class ProductSpec:
    """A product to compute: `scale * factors[0] * factors[1] * ...`.

    Args:
        factors (Sequence[ColumnOrExpr]): The values to multiply, as column names or
            expressions.
        scale (int | float): A constant the product is multiplied by, e.g. 100 for a
            percentage. Defaults to 1.
    """

    factors: Sequence[ColumnOrExpr]
    scale: int | float = 1


@dataclass(frozen=True)
class RatioSpec:
    """A ratio to compute: `dividend / divisor`.

    Args:
        dividend (ColumnOrExpr): The value divided, as a column name or an
            expression.
        divisor (ColumnOrExpr): The value divided by, as a column name or an
            expression.
    """

    dividend: ColumnOrExpr
    divisor: ColumnOrExpr


class CrossFeatures:
    """Features combining several columns arithmetically: sums, differences,
    products and ratios, e.g. debt / income or price - average price. Trees
    approximate such combinations only with many splits, so spelling them out often
    helps.

    Each method adds one column per feature and keeps the others, so they compose
    with `pipe`. The `*_multi` methods add several features of one kind at once.
    Inputs are column names or polars expressions:

        cf = CrossFeatures()
        features = (
            lf.pipe(cf.ratio, "debt_to_income", "debt", "income")
            .pipe(cf.difference, "price_gap", "price", pl.col("price").mean())
            .pipe(cf.sum_multi, {"total_spend": ["food", "rent", "travel"]})
        )

    Nothing is learned from the data, so train and test need no `fit`; but an
    expression aggregating over the frame (like the mean above) is computed per
    frame, so on test data it uses the test data's mean.
    """

    def sum(
        self, lf: pl.LazyFrame, name: str, addends: Sequence[ColumnOrExpr]
    ) -> pl.LazyFrame:
        """Adds the sum of several values as a new column.

        Args:
            lf (pl.LazyFrame): The data.
            name (str): The name of the new column.
            addends (Sequence[ColumnOrExpr]): The values to add up, as column names or
                expressions.

        Returns:
            pl.LazyFrame: `lf` with the sum (`pl.Float64`) added; null where any addend
                is null.
        """
        return lf.with_columns(sum_expr(addends).alias(name))

    def difference(
        self,
        lf: pl.LazyFrame,
        name: str,
        minuend: ColumnOrExpr,
        subtrahend: ColumnOrExpr,
    ) -> pl.LazyFrame:
        """Adds `minuend - subtrahend` as a new column.

        Args:
            lf (pl.LazyFrame): The data.
            name (str): The name of the new column.
            minuend (ColumnOrExpr): The value subtracted from, as a column name or an
                expression.
            subtrahend (ColumnOrExpr): The value subtracted, as a column name or an
                expression.

        Returns:
            pl.LazyFrame: `lf` with the difference added; null where either value is
                null.
        """
        return lf.with_columns(difference_expr(minuend, subtrahend).alias(name))

    def product(
        self,
        lf: pl.LazyFrame,
        name: str,
        factors: Sequence[ColumnOrExpr],
        scale: int | float = 1,
    ) -> pl.LazyFrame:
        """Adds the product of several values, times `scale`, as a new column.

        Args:
            lf (pl.LazyFrame): The data.
            name (str): The name of the new column.
            factors (Sequence[ColumnOrExpr]): The values to multiply, as column names
                or expressions.
            scale (int | float): A constant the product is multiplied by, e.g. 100 for
                a percentage. Defaults to 1.

        Returns:
            pl.LazyFrame: `lf` with the product added; null where any factor is null.

        Raises:
            ValueError: If a product has no factors.
        """
        return lf.with_columns(product_expr(factors, scale).alias(name))

    def ratio(
        self,
        lf: pl.LazyFrame,
        name: str,
        dividend: ColumnOrExpr,
        divisor: ColumnOrExpr,
    ) -> pl.LazyFrame:
        """Adds `dividend / divisor` as a new column.

        Args:
            lf (pl.LazyFrame): The data.
            name (str): The name of the new column.
            dividend (ColumnOrExpr): The value divided, as a column name or an
                expression.
            divisor (ColumnOrExpr): The value divided by, as a column name or an
                expression.

        Returns:
            pl.LazyFrame: `lf` with the ratio (`pl.Float64`) added; `inf` or `-inf`
                where the divisor is 0 (`NaN` for 0 / 0), null where either value is
                null.
        """
        return lf.with_columns(ratio_expr(dividend, divisor).alias(name))

    def custom_expression(
        self, lf: pl.LazyFrame, name: str, expr: pl.Expr
    ) -> pl.LazyFrame:
        """Adds any expression as a new column, for combinations the other methods
        don't cover, e.g. `(pl.col("a") - pl.col("b")) / pl.col("c")`.

        Args:
            lf (pl.LazyFrame): The data.
            name (str): The name of the new column.
            expr (pl.Expr): The expression to compute.

        Returns:
            pl.LazyFrame: `lf` with the expression's result added.
        """
        return lf.with_columns(expr.alias(name))

    def sum_multi(
        self,
        lf: pl.LazyFrame,
        spec: Mapping[str, Sequence[ColumnOrExpr]],
    ) -> pl.LazyFrame:
        """Adds several sums at once; see `sum`.

        Args:
            lf (pl.LazyFrame): The data.
            spec (Mapping[str, Sequence[ColumnOrExpr]]): Per new column name, the
                values to add up.

        Returns:
            pl.LazyFrame: `lf` with one sum column per entry of `spec` added.
        """
        return lf.with_columns(
            sum_expr(addends).alias(name) for name, addends in spec.items()
        )

    def difference_multi(
        self,
        lf: pl.LazyFrame,
        spec: Mapping[str, tuple[ColumnOrExpr, ColumnOrExpr]],
    ) -> pl.LazyFrame:
        """Adds several differences at once; see `difference`.

        Args:
            lf (pl.LazyFrame): The data.
            spec (Mapping[str, tuple[ColumnOrExpr, ColumnOrExpr]]): Per new column
                name, the `(minuend, subtrahend)` pair.

        Returns:
            pl.LazyFrame: `lf` with one difference column per entry of `spec` added.
        """
        return lf.with_columns(
            difference_expr(minuend, subtrahend).alias(name)
            for name, (minuend, subtrahend) in spec.items()
        )

    def product_multi(
        self,
        lf: pl.LazyFrame,
        spec: Mapping[str, ProductSpec],
    ) -> pl.LazyFrame:
        """Adds several products at once; see `product`.

        Args:
            lf (pl.LazyFrame): The data.
            spec (Mapping[str, ProductSpec]): Per new column name, the factors and
                scale.

        Returns:
            pl.LazyFrame: `lf` with one product column per entry of `spec` added.

        Raises:
            ValueError: If a product has no factors.
        """
        exprs = [
            product_expr(item.factors, item.scale).alias(name)
            for name, item in spec.items()
        ]
        return lf.with_columns(exprs)

    def ratio_multi(
        self,
        lf: pl.LazyFrame,
        spec: Mapping[str, tuple[ColumnOrExpr, ColumnOrExpr]],
    ) -> pl.LazyFrame:
        """Adds several ratios at once; see `ratio`.

        Args:
            lf (pl.LazyFrame): The data.
            spec (Mapping[str, tuple[ColumnOrExpr, ColumnOrExpr]]): Per new column
                name, the `(dividend, divisor)` pair.

        Returns:
            pl.LazyFrame: `lf` with one ratio column per entry of `spec` added.
        """
        return lf.with_columns(
            ratio_expr(dividend, divisor).alias(name)
            for name, (dividend, divisor) in spec.items()
        )


def sum_expr(addends: Sequence[ColumnOrExpr]) -> pl.Expr:
    """Builds the sum of several values, for use in your own `with_columns`.

    Args:
        addends (Sequence[ColumnOrExpr]): The values to add up, as column names or
            expressions.

    Returns:
        pl.Expr: The sum, as `pl.Float64`; null where any addend is null.
    """
    exprs = [into_expr(col) for col in addends]
    return pl.fold(pl.lit(0.0, dtype=pl.Int64), lambda x, y: x + y, exprs=exprs)


def difference_expr(
    minuend: ColumnOrExpr,
    subtrahend: ColumnOrExpr,
) -> pl.Expr:
    """Builds `minuend - subtrahend`, for use in your own `with_columns`.

    Args:
        minuend (ColumnOrExpr): The value subtracted from, as a column name or an
            expression.
        subtrahend (ColumnOrExpr): The value subtracted, as a column name or an
            expression.

    Returns:
        pl.Expr: The difference; null where either value is null.
    """
    return into_expr(minuend) - into_expr(subtrahend)


def product_expr(
    factors: Sequence[ColumnOrExpr],
    scale: int | float = 1,
) -> pl.Expr:
    """Builds the product of several values, times `scale`, for use in your own
    `with_columns`.

    Args:
        factors (Sequence[ColumnOrExpr]): The values to multiply, as column names or
            expressions.
        scale (int | float): A constant the product is multiplied by. Defaults to 1.

    Returns:
        pl.Expr: The product, in the factors' common dtype; null where any factor is
            null.

    Raises:
        ValueError: If `factors` is empty.
    """
    exprs = [into_expr(col) for col in factors]
    return pl.fold(pl.lit(scale, dtype=pl.Int64), lambda x, y: x * y, exprs=exprs)


def ratio_expr(
    dividend: ColumnOrExpr,
    divisor: ColumnOrExpr,
) -> pl.Expr:
    """Builds `dividend / divisor`, for use in your own `with_columns`.

    Args:
        dividend (ColumnOrExpr): The value divided, as a column name or an
            expression.
        divisor (ColumnOrExpr): The value divided by, as a column name or an
            expression.

    Returns:
        pl.Expr: The ratio, as `pl.Float64`; `inf` or `-inf` where the divisor is 0
            (`NaN` for 0 / 0), null where either value is null.
    """
    return into_expr(dividend) / into_expr(divisor)
