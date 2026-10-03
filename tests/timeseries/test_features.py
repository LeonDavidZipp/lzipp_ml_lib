from collections.abc import Sequence
from datetime import datetime

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


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        pytest.param("hour", {10: 5.0, 22: 3.0}, id="hour"),
        pytest.param("weekday", {7: 1.0, 1: 5.0}, id="weekday"),
        pytest.param("month", {12: 1.0, 1: 5.0}, id="month"),
        # (weekday - 1) * 24 + hour: Sun 22:00 -> 166, Mon 10:00 -> 10, Mon 22:00 -> 22
        pytest.param("hour_of_week", {166: 1.0, 10: 5.0, 22: 5.0}, id="hour_of_week"),
        pytest.param("day_of_year", {365: 1.0, 1: 4.0, 8: 7.0}, id="day_of_year"),
    ],
)
def test_fit_learns_profile_means(key: str, expected: dict[int, float]) -> None:
    fe = TimeseriesFeatures().fit(_fit_lf())

    profile = fe._profiles[key]  # type: ignore
    assert profile.columns == ["_key", f"profile_{key}"]
    assert dict(profile.iter_rows()) == pytest.approx(expected)


def test_fit_ignores_extra_columns() -> None:
    plain = TimeseriesFeatures().fit(_fit_lf())
    extra = TimeseriesFeatures().fit(_fit_lf(other=[9.0, 9.0, 9.0, 9.0]))

    assert (extra._min_year, extra._origin) == (plain._min_year, plain._origin)  # type: ignore
    for key, profile in plain._profiles.items():  # type: ignore
        plt.assert_frame_equal(extra._profiles[key], profile, check_row_order=False)  # type: ignore


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
    assert dict(fe._profiles["hour"].iter_rows()) == {3: 42.0}  # type: ignore


@pytest.mark.parametrize("method", ["trend", "profile"])
def test_methods_need_fit(method: str) -> None:
    lf = _fit_lf()

    with pytest.raises(RuntimeError, match="fit"):
        getattr(TimeseriesFeatures(), method)(lf)

    fitted = TimeseriesFeatures().fit(lf)
    assert getattr(fitted, method)(lf).collect().height == len(_FIT_TS)


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


def _ts_lf(days: list[int], vals: Sequence[float | int | None]) -> pl.LazyFrame:
    """A ts/val frame with one row per day of January 2024."""
    return pl.LazyFrame(
        {"ts": [datetime(2024, 1, d) for d in days], "val": vals},
        schema={"ts": pl.Datetime("us"), "val": pl.Float64},
    )
