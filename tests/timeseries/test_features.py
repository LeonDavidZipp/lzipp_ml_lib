from collections.abc import Sequence
from datetime import datetime

import polars as pl
import polars.testing as plt
import pytest

from lzipp_ml_lib import TimeseriesFeatures, TimeseriesSchema
from lzipp_ml_lib.timeseries._features import _approx_seconds  # type: ignore

DAY = 86_400
YEAR = 365.25 * DAY


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        ("1s", 1),
        ("1m", 60),
        ("1h", 3_600),
        ("1d", DAY),
        ("1w", 7 * DAY),
        ("1mo", YEAR / 12),
        ("1q", YEAR / 4),
        ("1y", YEAR),
        ("0h", 0),
        ("24h", DAY),
        ("5mo", 5 * YEAR / 12),  # "mo" is months, not minutes + "o"
        ("5m", 5 * 60),
        ("1d12h", 1.5 * DAY),
        ("2w3d", 17 * DAY),
        ("10m30s", 630),
        ("1y1mo", YEAR + YEAR / 12),
    ],
)
def test_approx_seconds(duration: str, expected: float) -> None:
    assert _approx_seconds(duration) == pytest.approx(expected)


@pytest.mark.parametrize(
    "duration",
    [
        pytest.param("", id="empty"),
        pytest.param("1x", id="unknown-unit"),
        pytest.param("1D", id="uppercase-unit"),
        pytest.param("h", id="missing-number"),
        pytest.param("1", id="missing-unit"),
        pytest.param("1.5h", id="fractional"),
        pytest.param("-1d", id="negative"),
        pytest.param("1 d", id="inner-space"),
        pytest.param("1d 2h", id="space-between-parts"),
        pytest.param("1ms", id="milliseconds"),
        pytest.param("d1", id="unit-before-number"),
    ],
)
def test_approx_seconds_rejects_invalid(duration: str) -> None:
    with pytest.raises(ValueError, match="invalid duration"):
        _approx_seconds(duration)


def _ts_lf(days: list[int], vals: Sequence[float | int | None]) -> pl.LazyFrame:
    """A ts/val frame with one row per day of January 2024."""
    return pl.LazyFrame(
        {"ts": [datetime(2024, 1, d) for d in days], "val": vals},
        schema={"ts": pl.Datetime("us"), "val": pl.Float64},
    )


@pytest.mark.parametrize(
    ("unique", "sort", "expected_days", "expected_failures"),
    [
        pytest.param(True, True, [1, 2, 3], {}, id="dedup-and-sort"),
        # Both rows of the duplicate timestamp fail the unique rule.
        pytest.param(False, True, [2, 3], {"ts|unique": 2}, id="keep-duplicates"),
        # is_sorted is checked on the whole frame, so every row fails it.
        pytest.param(True, False, [], {"is_sorted": 3}, id="keep-order"),
        pytest.param(False, False, [], {"is_sorted": 4, "ts|unique": 2}, id="neither"),
    ],
)
def test_prepare_unique_and_sort(
    unique: bool,
    sort: bool,
    expected_days: list[int],
    expected_failures: dict[str, int],
) -> None:
    # Unsorted, with day 1 twice: the first occurrence (val 1.0) should be kept.
    lf = _ts_lf([3, 1, 2, 1], [3.0, 1.0, 2.0, 9.0])

    result, failure = TimeseriesFeatures.prepare(lf, unique=unique, sort=sort)

    plt.assert_frame_equal(
        result.collect(), _ts_lf(expected_days, expected_days).collect()
    )
    assert failure.counts() == expected_failures


@pytest.mark.parametrize(
    ("bad_val", "rule"),
    [
        pytest.param(None, "val|nullability", id="null"),
        pytest.param(float("nan"), "val|nan", id="nan"),
        pytest.param(float("inf"), "val|inf", id="inf"),
        pytest.param(float("-inf"), "val|inf", id="-inf"),
    ],
)
def test_prepare_drops_invalid_values(bad_val: float | None, rule: str) -> None:
    lf = _ts_lf([1, 2, 3], [1.0, bad_val, 3.0])

    result, failure = TimeseriesFeatures.prepare(lf)

    plt.assert_frame_equal(result.collect(), _ts_lf([1, 3], [1.0, 3.0]).collect())
    assert failure.counts() == {rule: 1}


@pytest.mark.parametrize(
    ("lf", "expected_rows", "expected_failures"),
    [
        pytest.param(
            pl.LazyFrame({"ts": [datetime(2024, 1, 1)], "val": [1]}),
            1,
            {},
            id="int-val-cast-to-float",
        ),
        pytest.param(
            pl.LazyFrame({"ts": ["2024-01-01 00:00:00"], "val": [1.0]}),
            0,
            {"ts|dtype": 1},
            id="string-ts-not-parsed",
        ),
    ],
)
def test_prepare_casting(
    lf: pl.LazyFrame, expected_rows: int, expected_failures: dict[str, int]
) -> None:
    result, failure = TimeseriesFeatures.prepare(lf)

    assert result.collect().height == expected_rows
    assert failure.counts() == expected_failures


def test_prepare_real_data(base_timeseries_lf: pl.LazyFrame) -> None:
    result, failure = TimeseriesFeatures.prepare(base_timeseries_lf)

    df = result.collect()
    assert failure.counts() == {}
    assert (
        df.height == base_timeseries_lf.select(pl.col("ts").n_unique()).collect().item()
    )
    assert TimeseriesSchema.is_valid(df)
