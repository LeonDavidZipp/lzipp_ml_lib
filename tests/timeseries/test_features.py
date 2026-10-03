import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import dataframely as dy
import polars as pl
import polars.testing as plt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from lzipp_ml_lib import TimeseriesFeatures, TimeseriesSchema

from ._composites import (
    UNITS,
    daily_with_calendar_lag,
    grid_with_gaps,
    messy_rows,
    shift_back,
)

# ------------------------------------------------------------------------------------ #
#                              TimeseriesFeatures.prepare                              #
# ------------------------------------------------------------------------------------ #


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
            pl.LazyFrame({"ts": ["2024-01-01T00:00:00"], "val": [1.0]}),
            1,
            {},
            id="string-ts-parsed",
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


@given(rows=messy_rows())
def test_prepare_keeps_first_occurrence_of_valid_rows_sorted(
    rows: list[tuple[datetime | None, float | None]],
) -> None:
    lf = pl.LazyFrame(
        rows, schema={"ts": pl.Datetime("us"), "val": pl.Float64}, orient="row"
    )

    result, failure = TimeseriesFeatures.prepare(lf)
    df = result.collect()

    # Reference: keep the first row per timestamp (nulls form one group), then
    # drop rows with a null timestamp or a non-finite value, then sort.
    first: dict[datetime | None, float | None] = {}
    for t, v in rows:
        first.setdefault(t, v)
    expected = sorted(
        (t, v) for t, v in first.items() if t is not None and _is_finite(v)
    )

    assert TimeseriesSchema.is_valid(df)
    assert df.rows() == expected
    # Every deduplicated row is either kept or reported as a failure.
    assert df.height + failure.invalid().height == len(first)


# ------------------------------------------------------------------------------------ #
#                                TimeseriesFeatures.fit                                #
# ------------------------------------------------------------------------------------ #


def test_fit_learns_min_year_and_origin() -> None:
    fe = TimeseriesFeatures().fit(_fit_lf())

    assert fe._min_year == 2023  # type: ignore
    assert fe._origin == datetime(2023, 12, 31, 22)  # type: ignore


def test_fit_ignores_extra_columns() -> None:
    plain = TimeseriesFeatures().fit(_fit_lf())
    extra = TimeseriesFeatures().fit(_fit_lf(other=[9.0, 9.0, 9.0, 9.0]))

    assert (extra._min_year, extra._origin) == (plain._min_year, plain._origin)  # type: ignore


def test_refit_replaces_previous_state() -> None:
    later = TimeseriesSchema.validate(
        pl.LazyFrame(
            {"ts": [datetime(2030, 6, 1, 3)], "val": [42.0]},
            schema={"ts": pl.Datetime("us"), "val": pl.Float64},
        )
    )
    fe = TimeseriesFeatures().fit(_fit_lf()).fit(later)

    assert fe._min_year == 2030  # type: ignore
    assert fe._origin == datetime(2030, 6, 1, 3)  # type: ignore


def test_trend_needs_fit() -> None:
    lf = _fit_lf()

    with pytest.raises(RuntimeError, match="fit"):
        TimeseriesFeatures().trend(lf)

    fitted = TimeseriesFeatures().fit(lf)
    assert fitted.trend(lf).collect().height == len(_FIT_TS)


# ------------------------------------------------------------------------------------ #
#                                       lags                                           #
# ------------------------------------------------------------------------------------ #

_START = datetime(2024, 1, 1)


@pytest.mark.parametrize(
    ("unit_kwarg", "suffix", "timestamps"),
    [
        pytest.param(
            "yearly", "y", [datetime(2020 + i, 1, 1) for i in range(6)], id="yearly"
        ),
        pytest.param(
            "monthly", "mo", [datetime(2024, 1 + i, 1) for i in range(6)], id="monthly"
        ),
        pytest.param(
            "weekly", "w", [_START + timedelta(weeks=i) for i in range(6)], id="weekly"
        ),
        pytest.param(
            "daily", "d", [_START + timedelta(days=i) for i in range(6)], id="daily"
        ),
        pytest.param(
            "hourly", "h", [_START + timedelta(hours=i) for i in range(6)], id="hourly"
        ),
        pytest.param(
            "minutely",
            "m",
            [_START + timedelta(minutes=i) for i in range(6)],
            id="minutely",
        ),
        pytest.param(
            "secondly",
            "s",
            [_START + timedelta(seconds=i) for i in range(6)],
            id="secondly",
        ),
    ],
)
@pytest.mark.parametrize("n", [1, 2])
def test_lag_each_unit(
    unit_kwarg: str, suffix: str, timestamps: list[datetime], n: int
) -> None:
    # One row per unit step, so lag n is simply the value n rows earlier.
    vals = [float(i) for i in range(len(timestamps))]
    lf = _series_lf(timestamps, vals)

    lags: dict[str, Any] = {unit_kwarg: (n,)}
    out = TimeseriesFeatures().lag(lf, **lags).collect()

    assert out.columns == ["ts", "val", f"lag_{n}{suffix}"]
    assert out[f"lag_{n}{suffix}"].to_list() == [None] * n + vals[:-n]


def test_lag_multiple_lags_and_units() -> None:
    ts = [_START + timedelta(hours=i) for i in range(48)]
    lf = _series_lf(ts, [float(i) for i in range(48)])

    out = TimeseriesFeatures().lag(lf, daily=(1,), hourly=(1, 2)).collect()

    # Lag columns follow the unit order y, mo, w, d, h, m, s, then the given order.
    assert out.columns == ["ts", "val", "lag_1d", "lag_1h", "lag_2h"]
    last = out.row(-1, named=True)
    assert (last["lag_1d"], last["lag_1h"], last["lag_2h"]) == (23.0, 46.0, 45.0)


def test_lag_uses_timestamps_not_row_positions() -> None:
    # 02:00 is missing: the 03:00 row has no value one hour earlier, so it gets
    # null instead of the 01:00 value a row-based shift would give.
    ts = [_START + timedelta(hours=h) for h in (0, 1, 3, 4)]
    lf = _series_lf(ts, [0.0, 1.0, 3.0, 4.0])

    out = TimeseriesFeatures().lag(lf, hourly=(1,)).collect()

    assert out["lag_1h"].to_list() == [None, 0.0, None, 3.0]


def test_lag_monthly_is_calendar_aware() -> None:
    # Daily data from 2024-02-28 to 2024-03-31; 2024 is a leap year.
    ts = [datetime(2024, 2, 28) + timedelta(days=i) for i in range(33)]
    lf = _series_lf(ts, [float(i) for i in range(33)])

    out = TimeseriesFeatures().lag(lf, monthly=(1,)).collect()
    by_day = {row["ts"]: row["lag_1mo"] for row in out.iter_rows(named=True)}

    feb_29 = 1.0  # val of 2024-02-29
    assert by_day[datetime(2024, 3, 29)] == feb_29
    # Mar 30 and Mar 31 have no Feb 30/31, so both clamp to Feb 29.
    assert by_day[datetime(2024, 3, 30)] == feb_29
    assert by_day[datetime(2024, 3, 31)] == feb_29
    assert out.height == len(ts)


def test_lag_keeps_rows_and_order() -> None:
    ts = [_START + timedelta(hours=i) for i in range(100)]
    lf = _series_lf(ts, [float(i) for i in range(100)])

    out = TimeseriesFeatures().lag(lf, hourly=(1, 5), daily=(1,)).collect()

    assert out["ts"].to_list() == ts
    assert out["val"].to_list() == [float(i) for i in range(100)]


def test_lag_without_lags_returns_input() -> None:
    lf = _series_lf([_START + timedelta(hours=i) for i in range(5)], [1.0] * 5)

    plt.assert_frame_equal(TimeseriesFeatures().lag(lf).collect(), lf.collect())


@pytest.mark.parametrize(
    ("drop_ts", "drop_nulls", "expected_columns", "expected_height"),
    [
        pytest.param(False, False, ["ts", "val", "lag_2h"], 5, id="keep-all"),
        pytest.param(True, False, ["val", "lag_2h"], 5, id="drop-ts"),
        pytest.param(False, True, ["ts", "val", "lag_2h"], 3, id="drop-nulls"),
        pytest.param(True, True, ["val", "lag_2h"], 3, id="drop-both"),
    ],
)
def test_lag_cleanup_flags(
    drop_ts: bool, drop_nulls: bool, expected_columns: list[str], expected_height: int
) -> None:
    lf = _series_lf([_START + timedelta(hours=i) for i in range(5)], [1.0] * 5)

    out = (
        TimeseriesFeatures()
        .lag(lf, hourly=(2,), drop_ts=drop_ts, drop_nulls=drop_nulls)
        .collect()
    )

    assert out.columns == expected_columns
    assert out.height == expected_height


@pytest.mark.parametrize(
    "lags",
    [
        pytest.param({"hourly": (0,)}, id="zero"),
        pytest.param({"daily": (-1,)}, id="negative"),
        pytest.param({"weekly": (1, 0)}, id="one-of-many"),
    ],
)
def test_lag_rejects_non_positive(lags: dict[str, Any]) -> None:
    lf = _series_lf([_START], [1.0])

    with pytest.raises(ValueError, match="positive integers"):
        TimeseriesFeatures().lag(lf, **lags)


@pytest.mark.parametrize(
    ("lags", "allowed"),
    [
        pytest.param({"hourly": (1,)}, False, id="1h-too-short"),
        pytest.param({"hourly": (23,)}, False, id="23h-too-short"),
        pytest.param({"hourly": (24,)}, True, id="24h-ok"),
        pytest.param({"daily": (1,)}, True, id="1d-ok"),
        pytest.param({"daily": (1,), "hourly": (1,)}, False, id="any-too-short"),
        pytest.param({"weekly": (1,)}, True, id="1w-ok"),
    ],
)
def test_lag_respects_horizon(lags: dict[str, Any], allowed: bool) -> None:
    lf = _series_lf([_START + timedelta(hours=i) for i in range(30)], [1.0] * 30)
    fe = TimeseriesFeatures(horizon="1d")

    if allowed:
        assert fe.lag(lf, **lags).collect().height == 30
    else:
        with pytest.raises(ValueError, match="shorter than the horizon"):
            fe.lag(lf, **lags)


@given(
    case=st.one_of(grid_with_gaps(), daily_with_calendar_lag()),
    data=st.data(),
)
def test_lag_equals_value_exactly_n_units_earlier(
    case: tuple[list[datetime], str, int], data: st.DataObject
) -> None:
    ts, unit, n = case
    finite = st.floats(allow_nan=False, allow_infinity=False)
    vals = data.draw(st.lists(finite, min_size=len(ts), max_size=len(ts)))
    kwarg = next(k for k, u in UNITS.items() if u == unit)

    lags: dict[str, Any] = {kwarg: (n,)}
    out = TimeseriesFeatures().lag(_series_lf(ts, vals), **lags).collect()

    by_ts = dict(zip(ts, vals))
    expected = [by_ts.get(shift_back(t, n, unit)) for t in ts]
    assert out[f"lag_{n}{unit}"].to_list() == expected
    # Rows and their order are untouched.
    assert out["ts"].to_list() == ts
    assert out["val"].to_list() == vals


# ------------------------------------------------------------------------------------ #
#                                lag_diff / lag_ratios                                 #
# ------------------------------------------------------------------------------------ #


def _counting_lf(hours: int) -> dy.LazyFrame[TimeseriesSchema]:
    """Hourly data with val = 0, 1, 2, ..., so lag n of row i is i - n."""
    ts = [_START + timedelta(hours=i) for i in range(hours)]
    return _series_lf(ts, [float(i) for i in range(hours)])


def test_lag_diff() -> None:
    out = TimeseriesFeatures().lag_diff(_counting_lf(5), hourly=[(1, 2), (2, 1)])
    df = out.collect()

    # Only the requested columns are added; the looked-up lags are temporary.
    assert df.columns == ["ts", "val", "lag_1h_minus_lag_2h", "lag_2h_minus_lag_1h"]
    assert df["lag_1h_minus_lag_2h"].to_list() == [None, None, 1.0, 1.0, 1.0]
    assert df["lag_2h_minus_lag_1h"].to_list() == [None, None, -1.0, -1.0, -1.0]


def test_lag_ratios() -> None:
    out = TimeseriesFeatures().lag_ratios(_counting_lf(5), hourly=[(1, 2)])
    df = out.collect()

    assert df.columns == ["ts", "val", "lag_1h_over_lag_2h"]
    # Row 2 divides by lag_2h == 0, which gives null instead of inf.
    assert df["lag_1h_over_lag_2h"].to_list() == [None, None, None, 2.0, 1.5]


def test_lag_diff_matches_lag_columns() -> None:
    lf = _counting_lf(24 * 10)
    fe = TimeseriesFeatures()

    lags = fe.lag(lf, daily=(1, 7)).collect()
    diff = fe.lag_diff(lf, daily=[(1, 7)]).collect()

    plt.assert_series_equal(
        diff["lag_1d_minus_lag_7d"],
        (lags["lag_1d"] - lags["lag_7d"]).alias("lag_1d_minus_lag_7d"),
    )


def test_lag_pairs_multiple_units() -> None:
    out = TimeseriesFeatures().lag_diff(
        _counting_lf(24 * 3), daily=[(1, 2)], hourly=[(1, 3)]
    )

    assert out.collect_schema().names()[2:] == [
        "lag_1d_minus_lag_2d",
        "lag_1h_minus_lag_3h",
    ]
    last = out.collect().row(-1, named=True)
    assert (last["lag_1d_minus_lag_2d"], last["lag_1h_minus_lag_3h"]) == (24.0, 2.0)


def test_lag_pairs_keep_existing_lag_columns() -> None:
    # A frame that already has lag_1h from lag() keeps it untouched.
    fe = TimeseriesFeatures()
    lf = fe.lag(_counting_lf(5), hourly=(1,))

    df = fe.lag_diff(lf, hourly=[(1, 2)]).collect()  # type: ignore[arg-type]

    assert df.columns == ["ts", "val", "lag_1h", "lag_1h_minus_lag_2h"]
    assert df["lag_1h"].to_list() == [None, 0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("method", ["lag_diff", "lag_ratios"])
def test_lag_pairs_without_pairs_returns_input(method: str) -> None:
    lf = _counting_lf(5)

    out = getattr(TimeseriesFeatures(), method)(lf)

    plt.assert_frame_equal(out.collect(), lf.collect())


@pytest.mark.parametrize(
    ("method", "drop_nulls", "expected_height"),
    [
        pytest.param("lag_diff", False, 5, id="diff-keep-nulls"),
        # Rows 0-1 have no value 2 hours back.
        pytest.param("lag_diff", True, 3, id="diff-drop-nulls"),
        pytest.param("lag_ratios", False, 5, id="ratios-keep-nulls"),
        # Rows 0-1 have no value 2 hours back, and row 2 divides by 0.
        pytest.param("lag_ratios", True, 2, id="ratios-drop-nulls"),
    ],
)
@pytest.mark.parametrize("drop_ts", [False, True])
def test_lag_pairs_cleanup_flags(
    method: str, drop_nulls: bool, expected_height: int, drop_ts: bool
) -> None:
    out = getattr(TimeseriesFeatures(), method)(
        _counting_lf(5), hourly=[(1, 2)], drop_ts=drop_ts, drop_nulls=drop_nulls
    ).collect()

    assert ("ts" in out.columns) is not drop_ts
    assert out.height == expected_height


@pytest.mark.parametrize("method", ["lag_diff", "lag_ratios"])
@pytest.mark.parametrize(
    ("pairs", "horizon", "match"),
    [
        pytest.param({"hourly": [(0, 1)]}, None, "positive integers", id="zero"),
        pytest.param({"daily": [(1, -1)]}, None, "positive integers", id="negative"),
        pytest.param(
            {"hourly": [(1, 24)]}, "1d", "shorter than the horizon", id="horizon"
        ),
    ],
)
def test_lag_pairs_validate_lags(
    method: str, pairs: dict[str, Any], horizon: str | None, match: str
) -> None:
    fe = TimeseriesFeatures(horizon=horizon)

    with pytest.raises(ValueError, match=match):
        getattr(fe, method)(_counting_lf(5), **pairs)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        # lag_1h is overwritten with 10.0, so only the existing column can produce
        # these values; a fresh lookup would give i - 1 - (i - 2) = 1.
        pytest.param("lag_diff", [None, None, 10.0, 9.0, 8.0], id="diff"),
        pytest.param("lag_ratios", [None, None, None, 10.0, 5.0], id="ratios"),
    ],
)
def test_lag_pairs_lags_exist_uses_existing_columns(
    method: str, expected: list[float | None]
) -> None:
    fe = TimeseriesFeatures()
    lf = fe.lag(_counting_lf(5), hourly=(2,)).with_columns(lag_1h=pl.lit(10.0))

    df = getattr(fe, method)(lf, hourly=[(1, 2)], lags_exist=True).collect()

    op = "minus" if method == "lag_diff" else "over"
    assert df.columns == ["ts", "val", "lag_2h", "lag_1h", f"lag_1h_{op}_lag_2h"]
    assert df[f"lag_1h_{op}_lag_2h"].to_list() == expected


@pytest.mark.parametrize("method", ["lag_diff", "lag_ratios"])
def test_lag_pairs_lags_exist_requires_columns(method: str) -> None:
    lf = TimeseriesFeatures().lag(_counting_lf(5), hourly=(1,))  # no lag_2h

    with pytest.raises(ValueError, match=r"no columns \['lag_2h'\]"):
        getattr(TimeseriesFeatures(), method)(lf, hourly=[(1, 2)], lags_exist=True)


@pytest.mark.parametrize("method", ["lag_diff", "lag_ratios"])
def test_lag_pairs_keep_lags(method: str) -> None:
    fe = TimeseriesFeatures()
    lf = _counting_lf(24 * 10)

    df = getattr(fe, method)(lf, daily=[(1, 7)], keep_lags=True).collect()
    lags = fe.lag(lf, daily=(1, 7)).collect()

    op = "minus" if method == "lag_diff" else "over"
    assert df.columns == ["ts", "val", "lag_1d", "lag_7d", f"lag_1d_{op}_lag_7d"]
    plt.assert_frame_equal(
        df.select("lag_1d", "lag_7d"), lags.select("lag_1d", "lag_7d")
    )


@pytest.mark.parametrize("method", ["lag_diff", "lag_ratios"])
def test_lag_pairs_keep_lags_refuses_to_overwrite(method: str) -> None:
    lf = TimeseriesFeatures().lag(_counting_lf(5), hourly=(1,))

    with pytest.raises(ValueError, match="pass lags_exist=True"):
        getattr(TimeseriesFeatures(), method)(lf, hourly=[(1, 2)], keep_lags=True)


@pytest.mark.parametrize("method", ["lag_diff", "lag_ratios"])
def test_lag_pairs_lags_exist_still_validates(method: str) -> None:
    fe = TimeseriesFeatures(horizon="1d")
    lf = TimeseriesFeatures().lag(_counting_lf(5), hourly=(1, 2))

    with pytest.raises(ValueError, match="shorter than the horizon"):
        getattr(fe, method)(lf, hourly=[(1, 2)], lags_exist=True)


# ------------------------------------------------------------------------------------ #
#                                       helpers                                        #
# ------------------------------------------------------------------------------------ #


def _is_finite(val: float | None) -> bool:
    return val is not None and math.isfinite(val)


_FIT_TS = [
    datetime(2023, 12, 31, 22),
    datetime(2024, 1, 1, 10),
    datetime(2024, 1, 1, 22),
    datetime(2024, 1, 8, 10),
]
_FIT_VALS = [1.0, 3.0, 5.0, 7.0]


def _fit_lf(**extra_columns: list[object]) -> dy.LazyFrame[TimeseriesSchema]:
    return TimeseriesSchema.validate(
        pl.LazyFrame(
            {"ts": _FIT_TS, "val": _FIT_VALS, **extra_columns},
            schema_overrides={"ts": pl.Datetime("us")},
        )
    )


def _series_lf(
    ts: list[datetime], vals: Sequence[float]
) -> dy.LazyFrame[TimeseriesSchema]:
    return TimeseriesSchema.validate(
        pl.LazyFrame(
            {"ts": ts, "val": vals},
            schema={"ts": pl.Datetime("us"), "val": pl.Float64},
        )
    )


def _ts_lf(days: list[int], vals: Sequence[float | int | None]) -> pl.LazyFrame:
    """A ts/val frame with one row per day of January 2024."""
    return pl.LazyFrame(
        {"ts": [datetime(2024, 1, d) for d in days], "val": vals},
        schema={"ts": pl.Datetime("us"), "val": pl.Float64},
    )
