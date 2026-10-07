import polars as pl

ColumnOrExpr = str | pl.Expr


def into_expr(val: ColumnOrExpr) -> pl.Expr:
    """Converts a column name or Polars expression into a pl.Expr."""
    return pl.col(val) if isinstance(val, str) else val
