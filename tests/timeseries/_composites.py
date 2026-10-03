"""Hypothesis strategies for the timeseries tests."""

import calendar
from datetime import datetime, timedelta

from hypothesis import strategies as st

# lag() keyword -> unit suffix
UNITS = {
    "yearly": "y",
    "monthly": "mo",
    "weekly": "w",
    "daily": "d",
    "hourly": "h",
    "minutely": "m",
    "secondly": "s",
}
_FIXED = {
    "w": timedelta(weeks=1),
    "d": timedelta(days=1),
    "h": timedelta(hours=1),
    "m": timedelta(minutes=1),
    "s": timedelta(seconds=1),
}
# A small pool of timestamps, so duplicates are common.
_TS_POOL = [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(12)]


def shift_back(ts: datetime, n: int, unit: str) -> datetime:
    """`ts` minus `n` units in plain Python (negative `n` shifts forward).

    Independent of polars on purpose, so tests can use it as a reference: months
    and years clamp to the last day of shorter months, e.g. 2024-03-31 minus 1
    month is 2024-02-29.
    """
    if unit in ("mo", "y"):
        months = ts.year * 12 + ts.month - 1 - n * (12 if unit == "y" else 1)
        year, month = divmod(months, 12)
        last_day = calendar.monthrange(year, month + 1)[1]
        return ts.replace(year=year, month=month + 1, day=min(ts.day, last_day))
    return ts - n * _FIXED[unit]


def _steps_with_gaps(min_size: int, max_size: int) -> st.SearchStrategy[list[int]]:
    """Increasing step indices 0, ... that mostly advance by 1 but sometimes skip
    1-2 steps, so lag targets usually exist and gaps still occur."""
    gaps = st.lists(
        st.sampled_from([1, 1, 1, 2, 3]), min_size=min_size, max_size=max_size
    )
    return gaps.map(lambda g: [sum(g[:i]) for i in range(len(g))])


@st.composite
def grid_with_gaps(draw: st.DrawFn) -> tuple[list[datetime], str, int]:
    """Timestamps on a grid of one unit with some steps missing, plus a lag
    `(unit, n)` in that unit."""
    unit = draw(st.sampled_from(list(UNITS.values())))
    start = draw(st.datetimes(datetime(2000, 1, 1), datetime(2030, 1, 1)))
    ts = [shift_back(start, -k, unit) for k in draw(_steps_with_gaps(2, 25))]
    return ts, unit, draw(st.integers(1, 3))


@st.composite
def daily_with_calendar_lag(draw: st.DrawFn) -> tuple[list[datetime], str, int]:
    """Daily timestamps over up to ~2 years with some days missing, plus a monthly
    or yearly lag, so month ends, leap days and year boundaries all occur."""
    start = draw(st.datetimes(datetime(2019, 1, 1), datetime(2026, 1, 1)))
    ts = [start + timedelta(days=d) for d in draw(_steps_with_gaps(40, 400))]
    return ts, draw(st.sampled_from(["mo", "y"])), draw(st.integers(1, 2))


@st.composite
def messy_rows(draw: st.DrawFn) -> list[tuple[datetime | None, float | None]]:
    """Unsorted `(ts, val)` rows with duplicate and null timestamps and null / NaN /
    inf values."""
    ts = st.one_of(st.none(), st.sampled_from(_TS_POOL))
    val = st.one_of(st.none(), st.floats())  # st.floats() includes NaN and +-inf
    return draw(st.lists(st.tuples(ts, val), max_size=40))


def lag_pairs(max_lag: int = 4) -> st.SearchStrategy[list[tuple[int, int]]]:
    """1-3 distinct `(a, b)` lag pairs, sorted for a stable column order."""
    pair = st.tuples(st.integers(1, max_lag), st.integers(1, max_lag))
    return st.sets(pair, min_size=1, max_size=3).map(sorted)


# Seconds per duration unit, written out independently of the library's table.
DURATION_SECONDS = {
    "y": 365.25 * 86_400,
    "q": 365.25 / 4 * 86_400,
    "mo": 365.25 / 12 * 86_400,
    "w": 7 * 86_400,
    "d": 86_400,
    "h": 3_600,
    "m": 60,
    "s": 1,
}


def duration_parts() -> st.SearchStrategy[list[tuple[int, str]]]:
    """1-4 `(count, unit)` parts of a valid duration string like `1d12h`."""
    part = st.tuples(st.integers(0, 1_000), st.sampled_from(list(DURATION_SECONDS)))
    return st.lists(part, min_size=1, max_size=4)


def _join(parts: list[tuple[int, str]]) -> str:
    return "".join(f"{n}{unit}" for n, unit in parts)


@st.composite
def _with_inserted_char(draw: st.DrawFn) -> str:
    """A valid duration with one character inserted, e.g. `1d 2h` or `1d1x2h`."""
    text = _join(draw(duration_parts()))
    pos = draw(st.integers(0, len(text)))
    return text[:pos] + draw(st.sampled_from(list(" .-xDmo0"))) + text[pos:]


# Valid durations, almost-valid ones and random text over the same characters.
duration_like_text = st.one_of(
    duration_parts().map(_join),
    _with_inserted_char(),
    st.text(alphabet="0123456789moyqwdhsDx .-", max_size=12),
)
