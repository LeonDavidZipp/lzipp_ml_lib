from datetime import datetime

import polars as pl
import polars.testing as plt
import pytest

from lzipp_ml_lib.timeseries._features import (
    _approx_seconds,  # type: ignore
    _cleanup,  # type: ignore
    _maybe_drop_nulls,  # type: ignore
    _maybe_drop_ts,  # type: ignore
)

# ------------------------------------------------------------------------------------ #
#                                   _approx_seconds                                    #
# ------------------------------------------------------------------------------------ #

_DAY = 86_400
_YEAR = 365.25 * _DAY


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        ("1s", 1),
        ("1m", 60),
        ("1h", 3_600),
        ("1d", _DAY),
        ("1w", 7 * _DAY),
        ("1mo", _YEAR / 12),
        ("1q", _YEAR / 4),
        ("1y", _YEAR),
        ("0h", 0),
        ("24h", _DAY),
        ("5mo", 5 * _YEAR / 12),  # "mo" is months, not minutes + "o"
        ("5m", 5 * 60),
        ("1d12h", 1.5 * _DAY),
        ("2w3d", 17 * _DAY),
        ("10m30s", 630),
        ("1y1mo", _YEAR + _YEAR / 12),
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


# ------------------------------------------------------------------------------------ #
#                    _maybe_drop_ts / _maybe_drop_nulls / _cleanup                     #
# ------------------------------------------------------------------------------------ #

_D1, _D2 = datetime(2024, 1, 1), datetime(2024, 1, 2)


def _frame() -> pl.LazyFrame:
    """One complete row, one with a null val and one with a null ts."""
    return pl.LazyFrame(
        {"ts": [_D1, _D2, None], "val": [1.0, None, 3.0]},
        schema={"ts": pl.Datetime("us"), "val": pl.Float64},
    )


@pytest.mark.parametrize(
    ("drop_ts", "expected_columns"),
    [
        pytest.param(False, ["ts", "val"], id="keep"),
        pytest.param(True, ["val"], id="drop"),
    ],
)
def test_maybe_drop_ts(drop_ts: bool, expected_columns: list[str]) -> None:
    out = _maybe_drop_ts(_frame(), drop_ts)

    plt.assert_frame_equal(out.collect(), _frame().select(expected_columns).collect())


@pytest.mark.parametrize(
    ("drop_nulls", "expected_rows"),
    [
        pytest.param(False, [(_D1, 1.0), (_D2, None), (None, 3.0)], id="keep"),
        pytest.param(True, [(_D1, 1.0)], id="drop"),
    ],
)
def test_maybe_drop_nulls(
    drop_nulls: bool, expected_rows: list[tuple[datetime | None, float | None]]
) -> None:
    out = _maybe_drop_nulls(_frame(), drop_nulls)

    assert out.collect().rows() == expected_rows


@pytest.mark.parametrize(
    ("drop_ts", "drop_nulls", "expected_columns", "expected_rows"),
    [
        pytest.param(
            False,
            False,
            ["ts", "val"],
            [(_D1, 1.0), (_D2, None), (None, 3.0)],
            id="nothing",
        ),
        pytest.param(
            True, False, ["val"], [(1.0,), (None,), (3.0,)], id="drop-ts-only"
        ),
        pytest.param(False, True, ["ts", "val"], [(_D1, 1.0)], id="drop-nulls-only"),
        # ts is dropped first, so a null in ts alone no longer drops its row.
        pytest.param(True, True, ["val"], [(1.0,), (3.0,)], id="both"),
    ],
)
def test_cleanup(
    drop_ts: bool,
    drop_nulls: bool,
    expected_columns: list[str],
    expected_rows: list[tuple[object, ...]],
) -> None:
    out = _cleanup(_frame(), drop_ts=drop_ts, drop_nulls=drop_nulls).collect()

    assert out.columns == expected_columns
    assert out.rows() == expected_rows


def test_helpers_stay_lazy() -> None:
    lf = _frame()

    assert isinstance(_maybe_drop_ts(lf, True), pl.LazyFrame)
    assert isinstance(_maybe_drop_nulls(lf, True), pl.LazyFrame)
    assert isinstance(_cleanup(lf, drop_ts=True, drop_nulls=True), pl.LazyFrame)
