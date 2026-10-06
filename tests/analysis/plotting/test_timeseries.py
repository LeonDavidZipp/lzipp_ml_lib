from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from lzipp_ml_lib.analysis import plotting
from lzipp_ml_lib.analysis.plotting._timeseries import (
    _as_timedelta,  # type: ignore
    _duration_label,  # type: ignore
)
from lzipp_ml_lib.analysis.plotting._utils import default_period, infer_interval
from tests.composites import SAMPLE_SETTINGS

_START = datetime(2024, 1, 1)


def _hourly(n: int = 24 * 28, seed: int = 0) -> pl.DataFrame:
    """Four weeks of hourly data: a level, a daily sine and a little noise."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n)
    return pl.DataFrame(
        {
            "ts": [_START + timedelta(hours=int(h)) for h in hours],
            "val": 100 + 10 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1, n),
        }
    )


def _titles(fig: Figure) -> list[str]:
    return [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]


# ------------------------------------------------------------------------------------ #
#                                      helpers                                         #
# ------------------------------------------------------------------------------------ #


def test_infer_interval_takes_the_most_common_step():
    ts = pl.Series("ts", [_START + timedelta(hours=h) for h in [0, 1, 2, 4, 5, 5, 6]])
    assert infer_interval(ts) == timedelta(hours=1)


def test_infer_interval_needs_two_timestamps():
    with pytest.raises(ValueError, match="can't infer"):
        infer_interval(pl.Series("ts", [_START, _START]))


@pytest.mark.parametrize(
    ("interval", "period"),
    [
        (timedelta(minutes=15), 96),
        (timedelta(hours=1), 24),
        (timedelta(days=1), 7),
        (timedelta(weeks=1), 52),
        (timedelta(days=30), 12),
        (timedelta(days=91), 4),
    ],
)
def test_default_period_is_the_natural_cycle(interval: timedelta, period: int):
    assert default_period(interval) == period


@pytest.mark.parametrize(
    ("text", "expected", "label"),
    [
        ("1h", timedelta(hours=1), "1h"),
        ("15m", timedelta(minutes=15), "15min"),
        ("7d", timedelta(days=7), "7d"),
        ("90s", timedelta(seconds=90), "90s"),
    ],
)
def test_durations_parse_and_print(text: str, expected: timedelta, label: str):
    assert _as_timedelta(text) == expected
    assert _duration_label(expected) == label


# ------------------------------------------------------------------------------------ #
#                                  autocorrelation                                     #
# ------------------------------------------------------------------------------------ #


def test_pacf_of_an_ar1_series_cuts_off_after_lag_1():
    rng = np.random.default_rng(0)
    y = np.zeros(2000)
    for t in range(1, len(y)):
        y[t] = 0.8 * y[t - 1] + rng.normal()
    ax = plotting.plot_partial_autocorrelation(
        pl.DataFrame({"y": y}), "y", lags=5
    ).axes[0]
    (markers,) = [line for line in ax.lines if line.get_marker() == "o"]
    pacf = markers.get_ydata()
    assert pacf[1] == pytest.approx(0.8, abs=0.05)  # type: ignore
    assert np.all(np.abs(pacf[2:]) < 0.1)  # type: ignore


@pytest.mark.parametrize(
    "plot", [plotting.plot_autocorrelation, plotting.plot_partial_autocorrelation]
)
def test_correlograms_draw_side_by_side(plot: Callable[..., Figure]):
    fig, (left, right) = plt.subplots(1, 2)
    assert plot(_hourly(), "val", lags=30, ax=right) is fig
    assert "utocorrelation of val" in right.get_title(loc="left")
    assert not left.has_data()


# ------------------------------------------------------------------------------------ #
#                                  seasonal profile                                    #
# ------------------------------------------------------------------------------------ #


def test_seasonal_profile_heatmap_holds_the_mean_per_cell():
    df = _hourly()
    ax = plotting.plot_seasonal_profile(df).axes[0]
    grid = ax.collections[0].get_array().reshape(7, 24)  # type: ignore
    expected = (
        df.group_by(weekday=pl.col("ts").dt.weekday(), hour=pl.col("ts").dt.hour())
        .agg(pl.col("val").mean())
        .filter(weekday=1, hour=6)["val"]
        .item()
    )
    assert grid[0, 6] == pytest.approx(expected)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def test_seasonal_profile_line_recovers_the_daily_cycle():
    ax = plotting.plot_seasonal_profile(_hourly(), cols=None, rows="hour").axes[0]
    means = ax.lines[0].get_ydata()
    assert len(means) == 24  # type: ignore
    expected = 100 + 10 * np.sin(2 * np.pi * np.arange(24) / 24)
    np.testing.assert_allclose(means, expected, atol=1)  # type: ignore


def test_seasonal_profile_median_differs_from_mean_on_outliers():
    df = _hourly().with_columns(
        val=pl.when(pl.int_range(pl.len()) == 6).then(1e6).otherwise("val")
    )
    mean = plotting.plot_seasonal_profile(df, rows="hour", cols=None).axes[0]
    median = plotting.plot_seasonal_profile(
        df, rows="hour", cols=None, agg="median"
    ).axes[0]
    assert mean.lines[0].get_ydata()[6] > 1e4  # type: ignore
    assert median.lines[0].get_ydata()[6] < 200  # type: ignore


# ------------------------------------------------------------------------------------ #
#                                    decomposition                                     #
# ------------------------------------------------------------------------------------ #


def test_decomposition_components_add_up_to_the_series():
    fig = plotting.plot_decomposition(_hourly())
    assert _titles(fig) == ["val", "trend", "seasonal · period 24", "residual"]
    observed, trend, seasonal, resid = (ax.lines[0].get_ydata() for ax in fig.axes)
    np.testing.assert_allclose(trend + seasonal + resid, observed)  # type: ignore
    assert np.ptp(seasonal) == pytest.approx(20, rel=0.15)  # type: ignore # the 2 x 10 amplitude


def test_decomposition_with_several_periods_gets_one_panel_each():
    fig = plotting.plot_decomposition(_hourly(), periods=(168, 24))
    assert _titles(fig)[2:4] == ["seasonal · period 24", "seasonal · period 168"]


def test_decomposition_interpolates_gaps_and_says_so():
    df = _hourly().filter(~pl.int_range(pl.len()).is_in([100, 101, 300]))
    fig = plotting.plot_decomposition(df)
    assert [t.get_text() for t in fig.texts] == ["3 missing steps interpolated"]
    assert len(fig.axes[0].lines[0].get_ydata()) == _hourly().height  # type: ignore


def test_decomposition_needs_two_full_periods():
    with pytest.raises(ValueError, match="needs at least 336 steps"):
        plotting.plot_decomposition(_hourly(n=300), periods=168)


# ------------------------------------------------------------------------------------ #
#                                    rolling stats                                     #
# ------------------------------------------------------------------------------------ #


def test_rolling_stats_default_window_is_one_period():
    fig = plotting.plot_rolling_stats(_hourly())
    assert _titles(fig) == ["val · rolling mean ± std (1d)", "Rolling std (1d)"]
    mean = fig.axes[0].lines[1].get_ydata()
    # a full day averages the daily sine out, leaving the level
    assert np.nanmax(np.abs(mean - 100)) < 1  # type: ignore


@pytest.mark.parametrize("window", ["1d", 24])
def test_rolling_stats_draw_nothing_until_the_first_full_window(window: str | int):
    fig = plotting.plot_rolling_stats(_hourly(), window=window)
    std = np.asarray(fig.axes[1].lines[0].get_ydata(), dtype=float)
    assert np.isnan(std[:23]).all()
    assert not np.isnan(std[24:]).any()


# ------------------------------------------------------------------------------------ #
#                                        gaps                                          #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(
    removed=st.sets(st.integers(1, 98), max_size=30),
    duplicates=st.integers(0, 3),
)
def test_gaps_finds_every_run_of_missing_steps(removed: set[int], duplicates: int):
    kept = [h for h in range(100) if h not in removed]
    ts = [_START + timedelta(hours=h) for h in kept + kept[:duplicates]]
    ax = plotting.plot_gaps(pl.DataFrame({"ts": ts}), interval="1h").axes[0]

    runs: list[int] = []  # lengths of consecutive removed hours
    for h in sorted(removed):
        if runs and h - 1 in removed:
            runs[-1] += 1
        else:
            runs.append(1)
    stems: list[float] = list(np.asarray(ax.lines[0].get_ydata())) if runs else []
    assert sorted(stems) == sorted(runs)
    title = ax.get_title(loc="left")
    assert f"{len(runs)} gaps" in title
    assert f"{len(removed)} missing steps" in title
    assert (f"{duplicates} duplicate" in title) == (duplicates > 0)
    plt.close("all")


def test_gaps_without_gaps_says_so():
    ax = plotting.plot_gaps(_hourly()).axes[0]
    assert [t.get_text() for t in ax.texts] == ["No gaps"]
    assert "expected every 1h" in ax.get_title(loc="left")
