import pytest

from lzipp_ml_lib.timeseries._features import _approx_seconds # type: ignore
from lzipp_ml_lib import TimeseriesFeatures, TimeseriesSchema
import polars as pl
import polars.testing as plt
import pytest

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

