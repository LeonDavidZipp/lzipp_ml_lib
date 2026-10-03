from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import dataframely as dy
import polars as pl
import polars.testing as plt
import pytest

from lzipp_ml_lib import TimeseriesFeatures, TimeseriesSchema

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


# ------------------------------------------------------------------------------------ #
#                                       helpers                                        #
# ------------------------------------------------------------------------------------ #

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
