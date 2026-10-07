# from lzipp_ml_lib.transformation import CrossFeatures, difference_expr, product_expr, ratio_expr, sum_expr
import polars as pl
import polars.selectors as cs

# from hypothesis import (given, strategies as st)
# from hypothesis.strategies import composite
import pytest

from lzipp_ml_lib.transformation._cross import _into_expr  # type: ignore


@pytest.mark.parametrize("input", ("some_name", "another", pl.col("x"), cs.all()))
def test_into_expr(input: str | pl.Expr | cs.Selector):
    assert isinstance(_into_expr(input), pl.Expr)
