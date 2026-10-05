import math

import numpy as np
import polars as pl
import pytest
from hypothesis import given

from lzipp_ml_lib.analysis.summary import summarize
from tests.composites import BASIC_SETTINGS

from .composites import frames

_NUMERIC_STATS = [
    "mean",
    "std",
    "min",
    "p25",
    "median",
    "p75",
    "max",
    "skew",
    "zeros_pct",
]


@BASIC_SETTINGS
@given(frames())
def test_summarize_has_one_row_per_column_in_order(df: pl.DataFrame):
    summary = summarize(df)
    assert summary["column"].to_list() == df.columns
    assert summary["dtype"].to_list() == [str(dt) for dt in df.dtypes]


@BASIC_SETTINGS
@given(frames())
def test_summarize_counts_add_up(df: pl.DataFrame):
    for row in summarize(df).iter_rows(named=True):
        s = df[row["column"]]
        assert row["count"] + s.null_count() == df.height
        assert row["n_unique"] == s.n_unique()
        expected_pct = s.null_count() / df.height * 100 if df.height else 0.0
        assert row["null_pct"] == pytest.approx(expected_pct)


@BASIC_SETTINGS
@given(frames())
def test_summarize_numeric_stats_match_numpy(df: pl.DataFrame):
    for row in summarize(df).iter_rows(named=True):
        s = df[row["column"]]
        if not s.dtype.is_numeric():
            assert all(row[stat] is None for stat in _NUMERIC_STATS)
            continue
        values = s.drop_nulls().cast(pl.Float64).to_numpy()
        values = values[~np.isnan(values)]
        if len(values) == 0:
            assert all(row[stat] is None for stat in _NUMERIC_STATS)
            continue
        assert row["min"] <= row["p25"] <= row["median"] <= row["p75"] <= row["max"]
        assert row["mean"] == pytest.approx(values.mean(), rel=1e-9, abs=1e-6)
        assert row["median"] == pytest.approx(np.median(values), rel=1e-9, abs=1e-6)
        assert row["p25"] == pytest.approx(
            np.quantile(values, 0.25), rel=1e-9, abs=1e-6
        )
        assert row["zeros_pct"] == pytest.approx((values == 0).mean() * 100)
        if len(values) > 1:
            assert row["std"] == pytest.approx(values.std(ddof=1), rel=1e-6, abs=1e-6)


@BASIC_SETTINGS
@given(frames(min_rows=1))
def test_summarize_respects_columns(df: pl.DataFrame):
    columns = df.columns[::-1][:2]
    assert summarize(df, columns)["column"].to_list() == columns
    assert summarize(df.lazy(), columns)["column"].to_list() == columns


def test_summarize_flags_skew_and_zeros():
    df = pl.DataFrame({"x": [0.0] * 8 + [1.0, 100.0]})
    row = summarize(df).row(0, named=True)
    assert row["zeros_pct"] == 80
    assert row["skew"] > 2
    assert not math.isnan(row["std"])
