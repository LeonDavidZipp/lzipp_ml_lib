import re
from collections.abc import Callable, Sequence
from datetime import date, datetime
from functools import lru_cache
from typing import Literal, Self, TypeVar

import dataframely as dy
import holidays
import numpy as np
import polars as pl

_T = TypeVar("_T")


class TimeseriesSchema(dy.Schema):
    ts = dy.Datetime(nullable=False, unique=True)
    val = dy.Float(nullable=False, allow_inf=False, allow_nan=False)

    @dy.rule()
    def is_sorted(cls) -> pl.Expr:
        return pl.col("ts").is_sorted(descending=False)


CalendarFeature = Literal[
    "quarter",
    "month",
    "week",
    "day",
    "day_of_year",
    "weekday",
    "hour",
    "minute",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "days_in_month",
]
CyclicalFeature = Literal[
    "month", "weekday", "hour", "hour_of_week", "day_of_year", "day_of_month"
]
RollingStat = Literal["mean", "std", "min", "max", "median"]
TrendUnit = Literal["s", "m", "h", "d"]


class TimeseriesFeatures:
    """Feature builders for a `ts` / `val` time series.

    Each method adds columns and keeps `ts` unless `drop_ts=True`, so they compose
    with `pipe`:

        fe = TimeseriesFeatures(horizon="1d").fit(train)
        features = (
            train.pipe(fe.lag, daily=(1, 7))
            .pipe(fe.rolling, windows=("7d",))
            .pipe(fe.cyclical)
            .pipe(fe.calendar)
            .pipe(fe.holiday, country="DE")
            .pipe(fe.trend, drop_ts=True)
            .drop_nulls()
        )

    Features built from past values (`lag`, `rolling`, `ewm`, `gap`, and
    `exogenous` with `known_in_advance=False`) only use data available
    `horizon` before each row, so they can be computed at prediction time.
    `trend` uses values learned by `fit`, so train and test get consistent
    features.
    """

    def __init__(self, horizon: str | None = None):
        """
        Args:
            horizon (str | None): How far ahead you forecast, as a polars duration
                string, e.g. `"1d"` or `"6h"`. Past-value features only use data
                at least this old. Defaults to None (one step ahead: everything
                before the row's own timestamp is available).

        Raises:
            ValueError: If `horizon` is not a valid duration.
        """
        if horizon is not None:
            _approx_seconds(horizon)
        self.horizon = horizon
        self._min_year: int | None = None
        self._origin: datetime | None = None

    @staticmethod
    def prepare(
        lf: pl.LazyFrame, unique: bool = True, sort: bool = True
    ) -> tuple[dy.LazyFrame[TimeseriesSchema], dy.FailureInfo]:
        """Prepares the LazyFrame to ensure it is passable to the classes' other
        methods. Returns both the prepared LazyFrame as well as potential failures
        encountered in the preparation.

        Args:
            lf (pl.LazyFrame): The LazyFrame to be prepared. Should contain both a 'ts'
                and a 'val' column
            unique (bool): Whether to drop duplicate timestamps. Keeps the first
                occurence of duplicate values. Defaults to True.
            sort (bool): Whether to sort the data. Keeps the first
                occurence of duplicate values. Defaults to True.

        Returns:
            tuple[dy.LazyFrame[TimeSeriesSchema], dy.FailureInfo]: A tuple containing
                the prepared LazyFrame as well as failure info.
        """
        out = lf
        if unique:
            out = out.unique("ts", keep="first", maintain_order=not sort)
        if sort:
            out = out.sort(by="ts")
        info = TimeseriesSchema.filter(out, cast=True)
        return info.result, info.failure

    def fit(self, lf: dy.LazyFrame[TimeseriesSchema]) -> Self:
        """Learn the training-set values used by `trend`.

        Collects `lf`.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Training data.

        Returns:
            Self: The fitted feature engineer, for chaining.
        """
        df = lf.select("ts", "val").collect()
        self._min_year = df.select(_ts.dt.year().min()).item()
        self._origin = df.select(_ts.min()).item()
        return self

    def lag(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        yearly: Sequence[int] | None = None,
        monthly: Sequence[int] | None = None,
        weekly: Sequence[int] | None = None,
        daily: Sequence[int] | None = None,
        hourly: Sequence[int] | None = None,
        minutely: Sequence[int] | None = None,
        secondly: Sequence[int] | None = None,
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add lagged copies of `val`.

        Lags are looked up by timestamp (calendar-aware), so gaps or irregular
        sampling don't shift values onto the wrong rows. Rows with no value `n`
        units back get a null.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with a unique `ts` datetime
                column and a `val` column.
            yearly (Sequence[int] | None): Yearly lags, named `lag_{n}y`, e.g. `(1,)`
                adds `lag_1y`. Defaults to None.
            monthly (Sequence[int] | None): Monthly lags, named `lag_{n}mo`.
                Defaults to None.
            weekly (Sequence[int] | None): Weekly lags, named `lag_{n}w`.
                Defaults to None.
            daily (Sequence[int] | None): Daily lags, named `lag_{n}d`, e.g. `(1, 7)`
                adds `lag_1d` and `lag_7d`. Defaults to None.
            hourly (Sequence[int] | None): Hourly lags, named `lag_{n}h`.
                Defaults to None.
            minutely (Sequence[int] | None): Minutely lags, named `lag_{n}m`.
                Defaults to None.
            secondly (Sequence[int] | None): Secondly lags, named `lag_{n}s`.
                Defaults to None.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with the requested lag columns appended (without
                `ts` if `drop_ts`).

        Raises:
            ValueError: If any lag is smaller than 1, or shorter than the horizon.
        """
        lags = _by_unit(yearly, monthly, weekly, daily, hourly, minutely, secondly)
        self._check_lags({unit: list(ns or ()) for unit, ns in lags.items()})
        out = _join_lags(
            lf,
            [
                (n, unit, f"lag_{n}{unit}")
                for unit, ns in lags.items()
                for n in ns or ()
            ],
        )
        return _cleanup(out, drop_ts, drop_nulls)

    def lag_diff(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        yearly: Sequence[tuple[int, int]] | None = None,
        monthly: Sequence[tuple[int, int]] | None = None,
        weekly: Sequence[tuple[int, int]] | None = None,
        daily: Sequence[tuple[int, int]] | None = None,
        hourly: Sequence[tuple[int, int]] | None = None,
        minutely: Sequence[tuple[int, int]] | None = None,
        secondly: Sequence[tuple[int, int]] | None = None,
        lags_exist: bool = False,
        keep_lags: bool = False,
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add differences between two lags of `val` in the same unit.

        Each pair `(a, b)` adds the value `a` units back minus the value `b` units
        back. The lags are looked up like in `lag`, so `lf` needs no lag columns.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with `ts` and `val` columns.
            yearly (Sequence[tuple[int, int]] | None): Yearly lag pairs, e.g.
                `[(1, 2)]` adds `lag_1y_minus_lag_2y`. Defaults to None.
            monthly (Sequence[tuple[int, int]] | None): Monthly lag pairs, named
                `lag_{a}mo_minus_lag_{b}mo`. Defaults to None.
            weekly (Sequence[tuple[int, int]] | None): Weekly lag pairs, named
                `lag_{a}w_minus_lag_{b}w`. Defaults to None.
            daily (Sequence[tuple[int, int]] | None): Daily lag pairs, e.g. `[(1, 7)]`
                adds `lag_1d_minus_lag_7d`. Defaults to None.
            hourly (Sequence[tuple[int, int]] | None): Hourly lag pairs, named
                `lag_{a}h_minus_lag_{b}h`. Defaults to None.
            minutely (Sequence[tuple[int, int]] | None): Minutely lag pairs, named
                `lag_{a}m_minus_lag_{b}m`. Defaults to None.
            secondly (Sequence[tuple[int, int]] | None): Secondly lag pairs, named
                `lag_{a}s_minus_lag_{b}s`. Defaults to None.
            lags_exist (bool): Use the `lag_{n}{unit}` columns already in `lf`, e.g.
                from `lag`, instead of looking the lags up. Defaults to False.
            keep_lags (bool): Keep the looked-up lags as `lag_{n}{unit}` columns
                instead of dropping them. Has no effect with `lags_exist`, whose
                columns are always kept. Defaults to False.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with the requested difference columns appended (without
                `ts` if `drop_ts`).

        Raises:
            ValueError: If any lag is smaller than 1, or shorter than the horizon;
                if `lags_exist` and a needed lag column is missing; or if
                `keep_lags` would overwrite an existing lag column.
        """
        pairs = _by_unit(yearly, monthly, weekly, daily, hourly, minutely, secondly)
        out = self._combine_lags(
            lf, pairs, "minus", lambda a, b: a - b, lags_exist, keep_lags
        )
        return _cleanup(out, drop_ts, drop_nulls)

    def lag_ratios(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        yearly: Sequence[tuple[int, int]] | None = None,
        monthly: Sequence[tuple[int, int]] | None = None,
        weekly: Sequence[tuple[int, int]] | None = None,
        daily: Sequence[tuple[int, int]] | None = None,
        hourly: Sequence[tuple[int, int]] | None = None,
        minutely: Sequence[tuple[int, int]] | None = None,
        secondly: Sequence[tuple[int, int]] | None = None,
        lags_exist: bool = False,
        keep_lags: bool = False,
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add ratios between two lags of `val` in the same unit.

        Each pair `(a, b)` adds the value `a` units back divided by the value `b`
        units back, null where the latter is 0. The lags are looked up like in
        `lag`, so `lf` needs no lag columns.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with `ts` and `val` columns.
            yearly (Sequence[tuple[int, int]] | None): Yearly lag pairs, e.g.
                `[(1, 2)]` adds `lag_1y_over_lag_2y`. Defaults to None.
            monthly (Sequence[tuple[int, int]] | None): Monthly lag pairs, named
                `lag_{a}mo_over_lag_{b}mo`. Defaults to None.
            weekly (Sequence[tuple[int, int]] | None): Weekly lag pairs, named
                `lag_{a}w_over_lag_{b}w`. Defaults to None.
            daily (Sequence[tuple[int, int]] | None): Daily lag pairs, e.g. `[(1, 7)]`
                adds `lag_1d_over_lag_7d`. Defaults to None.
            hourly (Sequence[tuple[int, int]] | None): Hourly lag pairs, named
                `lag_{a}h_over_lag_{b}h`. Defaults to None.
            minutely (Sequence[tuple[int, int]] | None): Minutely lag pairs, named
                `lag_{a}m_over_lag_{b}m`. Defaults to None.
            secondly (Sequence[tuple[int, int]] | None): Secondly lag pairs, named
                `lag_{a}s_over_lag_{b}s`. Defaults to None.
            lags_exist (bool): Use the `lag_{n}{unit}` columns already in `lf`, e.g.
                from `lag`, instead of looking the lags up. Defaults to False.
            keep_lags (bool): Keep the looked-up lags as `lag_{n}{unit}` columns
                instead of dropping them. Has no effect with `lags_exist`, whose
                columns are always kept. Defaults to False.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with the requested ratio columns appended (without `ts`
                if `drop_ts`).

        Raises:
            ValueError: If any lag is smaller than 1, or shorter than the horizon;
                if `lags_exist` and a needed lag column is missing; or if
                `keep_lags` would overwrite an existing lag column.
        """
        pairs = _by_unit(yearly, monthly, weekly, daily, hourly, minutely, secondly)
        out = self._combine_lags(
            lf,
            pairs,
            "over",
            lambda a, b: pl.when(b != 0).then(a / b),
            lags_exist,
            keep_lags,
        )
        return _cleanup(out, drop_ts, drop_nulls)

    def rolling(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        windows: Sequence[str],
        stats: Sequence[RollingStat] = ("mean", "std"),
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add rolling statistics of past `val`s.

        Each row gets the statistic over the `window` ending at the latest
        observation available at prediction time (see `horizon`), never
        including the row's own value.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with `ts` and `val` columns.
            windows (Sequence[str]): Window lengths as polars durations, e.g.
                `("24h", "7d")`.
            stats (Sequence[RollingStat]): Statistics to compute. Defaults to
                `("mean", "std")`.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with `roll_{stat}_{window}` columns appended
                (without `ts` if `drop_ts`).
        """
        values = lf.select(
            "ts",
            *(
                _ROLLING[stat](w).alias(f"roll_{stat}_{w}")
                for w in windows
                for stat in stats
            ),
        )
        return _cleanup(self._latest_available(lf, values), drop_ts, drop_nulls)

    def ewm(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        half_lives: Sequence[str],
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add exponentially weighted means of past `val`s.

        Like `rolling`, each row only sees observations available at prediction
        time.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with `ts` and `val` columns.
            half_lives (Sequence[str]): Half-lives as polars durations, e.g.
                `("1d", "7d")`.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with `ewm_{half_life}` columns appended (without `ts`
                if `drop_ts`).
        """
        values = lf.select(
            "ts",
            *(
                pl.col("val").ewm_mean_by("ts", half_life=hl).alias(f"ewm_{hl}")
                for hl in half_lives
            ),
        )
        return _cleanup(self._latest_available(lf, values), drop_ts, drop_nulls)

    def gap(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add the time since the latest observation available at prediction time.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with a `ts` column.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with `secs_since_last_obs` appended (without `ts` if
                `drop_ts`).
        """
        values = lf.select("ts", _last_obs_ts=pl.col("ts"))
        out = (
            self._latest_available(lf, values)
            .with_columns(
                secs_since_last_obs=(_ts - pl.col("_last_obs_ts")).dt.total_seconds()
            )
            .drop("_last_obs_ts")
        )
        return _cleanup(out, drop_ts, drop_nulls)

    def cyclical(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        features: Sequence[CyclicalFeature] = ("month", "weekday", "hour"),
        harmonics: int = 1,
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add sine/cosine (Fourier) encodings of cyclical calendar features.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with a `ts` datetime column.
            features (Sequence[CyclicalFeature]): Cycles to encode. Defaults to
                `("month", "weekday", "hour")`.
            harmonics (int): Number of sine/cosine pairs per cycle. Harmonic `k`
                repeats `k` times per cycle and lets the model fit sharper
                seasonal shapes. Defaults to 1.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with `{feature}_sin` / `{feature}_cos` appended, plus
                `{feature}_sin{k}` / `{feature}_cos{k}` for harmonics `k > 1`
                (without `ts` if `drop_ts`).

        Raises:
            ValueError: If `harmonics` is smaller than 1.
        """
        if harmonics < 1:
            raise ValueError(f"harmonics must be at least 1, got {harmonics}")
        exprs: list[pl.Expr] = []
        for name in features:
            expr, period = _CYCLICAL[name]
            for k in range(1, harmonics + 1):
                angle = expr * (2 * np.pi * k / period)
                suffix = "" if k == 1 else str(k)
                exprs.append(angle.sin().alias(f"{name}_sin{suffix}"))
                exprs.append(angle.cos().alias(f"{name}_cos{suffix}"))
        return _cleanup(lf.with_columns(exprs), drop_ts, drop_nulls)

    def calendar(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        features: Sequence[CalendarFeature] = ("quarter", "month", "day", "hour"),
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add raw calendar features.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with a `ts` datetime column.
            features (Sequence[CalendarFeature]): Features to add, each named as in
                the literal. Defaults to `("quarter", "month", "day", "hour")`.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with the requested calendar columns appended (without
                `ts` if `drop_ts`).
        """
        out = lf.with_columns(_CALENDAR[name].alias(name) for name in features)
        return _cleanup(out, drop_ts, drop_nulls)

    def holiday(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        country: str,
        subdiv: str | None = None,
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add public holiday features from the `holidays` package.

        Collects the min and max year of `ts` to build the holiday calendar.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with a `ts` datetime column.
            country (str): ISO country code, e.g. `"DE"`.
            subdiv (str | None): Subdivision code, e.g. `"BY"` for Bavaria.
                Defaults to None (national holidays only).
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with `is_holiday`, `days_to_holiday`,
                `days_since_holiday` and `is_bridge_day` (a workday between two days
                off) appended (without `ts` if `drop_ts`).
        """
        lo, hi = (
            lf.select(_ts.dt.year().min(), _ts.dt.year().max().alias("hi"))
            .collect()
            .row(0)
        )
        calendar = holidays.country_holidays(
            country, subdiv=subdiv, years=range(lo - 1, hi + 2)
        )
        holidays_df = pl.DataFrame(
            {"_hol": sorted(calendar.keys())}, schema={"_hol": pl.Date}
        )
        holidays_lf = holidays_df.lazy()
        day_off = pl.col("is_holiday") | (pl.col("_date").dt.weekday() >= 6)
        days = (
            pl.LazyFrame(
                {
                    "_date": pl.date_range(
                        date(lo - 1, 1, 1), date(hi + 1, 12, 31), eager=True
                    )
                }
            )
            .with_columns(
                is_holiday=pl.col("_date").is_in(holidays_df.get_column("_hol"))
            )
            .join_asof(
                holidays_lf, left_on="_date", right_on="_hol", strategy="forward"
            )
            .with_columns(
                days_to_holiday=(pl.col("_hol") - pl.col("_date")).dt.total_days()
            )
            .drop("_hol")
            .join_asof(
                holidays_lf,
                left_on="_date",
                right_on="_hol",
                strategy="backward",
            )
            .with_columns(
                days_since_holiday=(pl.col("_date") - pl.col("_hol")).dt.total_days()
            )
            .drop("_hol")
            .with_columns(is_bridge_day=~day_off & day_off.shift(1) & day_off.shift(-1))
        )
        out = (
            lf.with_columns(_date=_ts.dt.date())
            .join(days, on="_date", how="left", maintain_order="left")
            .drop("_date")
        )
        return _cleanup(out, drop_ts, drop_nulls)

    def trend(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        unit: TrendUnit = "h",
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add trend features relative to the training data seen by `fit`.

        Tree models like XGBoost can't extrapolate a trend beyond the training
        range; consider detrending the target instead.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with a `ts` datetime column.
            unit (TrendUnit): Unit of the time index. Defaults to `"h"`.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with `diff_from_min_year` (years since the first
                training year) and `t_{unit}` (time since the first training
                timestamp) appended (without `ts` if `drop_ts`).

        Raises:
            RuntimeError: If `fit` hasn't been called.
        """
        if self._min_year is None or self._origin is None:
            raise RuntimeError("call fit() before trend()")
        out = lf.with_columns(
            diff_from_min_year=_ts.dt.year() - self._min_year,
            **{
                f"t_{unit}": (_ts - pl.lit(self._origin)).dt.total_seconds()
                / _UNIT_SECONDS[unit]
            },
        )
        return _cleanup(out, drop_ts, drop_nulls)

    def exogenous(
        self,
        lf: dy.LazyFrame[TimeseriesSchema],
        exog: pl.LazyFrame,
        known_in_advance: bool = True,
        tolerance: str | None = None,
        drop_ts: bool = False,
        drop_nulls: bool = False,
    ) -> pl.LazyFrame:
        """Add external series, e.g. weather or prices, matched by timestamp.

        Each row gets the latest `exog` row at or before its timestamp.

        Args:
            lf (dy.LazyFrame[TimeseriesSchema]): Frame with a `ts` datetime column.
            exog (pl.LazyFrame): Frame with a `ts` column (sorted, same dtype as in
                `lf`) and the feature columns.
            known_in_advance (bool): Whether the values are known at prediction
                time, e.g. a weather forecast. If False (e.g. measured weather),
                only values available `horizon` before each row are used.
                Defaults to True.
            tolerance (str | None): Max distance to the matched `exog` row as a
                polars duration, e.g. `"1h"`; farther matches become null.
                Defaults to None (no limit).
            drop_ts (bool): Drop `ts` from the result. Defaults to False.
            drop_nulls (bool): Drops all rows containing nulls. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with the `exog` columns appended (without `ts` if
                `drop_ts`).
        """
        exog_sorted = exog.sort(by="ts", descending=False)
        if known_in_advance:
            out = lf.join_asof(
                exog_sorted, on="ts", strategy="backward", tolerance=tolerance
            )
        else:
            out = self._latest_available(lf, exog_sorted, tolerance)
        return _cleanup(out, drop_ts, drop_nulls)

    def _latest_available(
        self,
        lf: pl.LazyFrame,
        values: pl.LazyFrame,
        tolerance: str | None = None,
    ) -> pl.LazyFrame:
        """Join `values` (`ts` + feature columns, computed including each row's own
        observation) so every row gets the latest values known at prediction time:
        at or before `ts - horizon`, or strictly before `ts` without a horizon."""
        cutoff = (
            _ts if self.horizon is None else _ts.dt.offset_by(f"-{self.horizon}")
        ).set_sorted()
        return (
            lf.with_columns(_cutoff=cutoff)
            .join_asof(
                values.rename({"ts": "_cutoff"}),
                on="_cutoff",
                strategy="backward",
                allow_exact_matches=self.horizon is not None,
                tolerance=tolerance,
            )
            .drop("_cutoff")
        )

    def _check_lags(self, lags: dict[str, list[int]]) -> None:
        """Raise if a lag isn't positive or is shorter than the horizon."""
        for unit, ns in lags.items():
            if any(n < 1 for n in ns):
                raise ValueError(
                    f"lags must be positive integers, got {ns} for unit '{unit}'"
                )
        if self.horizon is not None:
            horizon = _approx_seconds(self.horizon)
            too_short = [
                f"{n}{unit}"
                for unit, ns in lags.items()
                for n in ns
                if n * _UNIT_SECONDS[unit] < horizon
            ]
            if too_short:
                raise ValueError(
                    f"lags {too_short} are shorter than the horizon '{self.horizon}' "
                    "and won't be known at prediction time"
                )

    def _combine_lags(
        self,
        lf: pl.LazyFrame,
        pairs: dict[str, Sequence[tuple[int, int]] | None],
        op: str,
        combine: Callable[[pl.Expr, pl.Expr], pl.Expr],
        lags_exist: bool,
        keep_lags: bool,
    ) -> pl.LazyFrame:
        """Add `lag_{a}{unit}_{op}_lag_{b}{unit}` = `combine(lag a, lag b)` for each
        pair. The lags come from existing columns (`lags_exist`), or are looked up
        into columns that are kept (`keep_lags`) or dropped afterwards."""
        needed = {
            unit: sorted({n for pair in unit_pairs or () for n in pair})
            for unit, unit_pairs in pairs.items()
        }
        self._check_lags(needed)
        lag_names = [f"lag_{n}{unit}" for unit, ns in needed.items() for n in ns]
        existing = set(lf.collect_schema().names())

        if lags_exist:
            if missing := [name for name in lag_names if name not in existing]:
                raise ValueError(
                    f"lags_exist=True, but lf has no columns {missing}; "
                    "add them with lag() first"
                )
            prefix, to_drop = "lag_", []
        else:
            if keep_lags and (clashes := [n for n in lag_names if n in existing]):
                raise ValueError(
                    f"keep_lags=True would overwrite the existing columns {clashes}; "
                    "pass lags_exist=True to use them instead"
                )
            prefix = "lag_" if keep_lags else "_lag_"
            lookups = [
                (n, unit, f"{prefix}{n}{unit}")
                for unit, ns in needed.items()
                for n in ns
            ]
            lf = _join_lags(lf, lookups)
            to_drop = [] if keep_lags else [name for _, _, name in lookups]

        return lf.with_columns(
            combine(pl.col(f"{prefix}{a}{unit}"), pl.col(f"{prefix}{b}{unit}")).alias(
                f"lag_{a}{unit}_{op}_lag_{b}{unit}"
            )
            for unit, unit_pairs in pairs.items()
            for a, b in unit_pairs or ()
        ).drop(to_drop)


_ts = pl.col("ts")
# weekday() and hour() are Int8, which overflows past 127 (Saturday 08:00).
_hour_of_week = (_ts.dt.weekday().cast(pl.Int16) - 1) * 24 + _ts.dt.hour()

_CALENDAR: dict[str, pl.Expr] = {
    "quarter": _ts.dt.quarter(),
    "month": _ts.dt.month(),
    "week": _ts.dt.week(),
    "day": _ts.dt.day(),
    "day_of_year": _ts.dt.ordinal_day(),
    "weekday": _ts.dt.weekday(),
    "hour": _ts.dt.hour(),
    "minute": _ts.dt.minute(),
    "is_weekend": _ts.dt.weekday() >= 6,
    "is_month_start": _ts.dt.day() == 1,
    "is_month_end": _ts.dt.day() == _ts.dt.days_in_month(),
    "days_in_month": _ts.dt.days_in_month(),
}

# name -> (position in the cycle, cycle length)
_CYCLICAL: dict[str, tuple[pl.Expr, float]] = {
    "month": (_ts.dt.month(), 12),
    "weekday": (_ts.dt.weekday(), 7),
    "hour": (_ts.dt.hour(), 24),
    "hour_of_week": (_hour_of_week, 168),
    "day_of_year": (_ts.dt.ordinal_day() - 1, 365.25),
    "day_of_month": ((_ts.dt.day() - 1) / _ts.dt.days_in_month(), 1),
}

_ROLLING: dict[str, Callable[[str], pl.Expr]] = {
    "mean": lambda w: pl.col("val").rolling_mean_by("ts", w),
    "std": lambda w: pl.col("val").rolling_std_by("ts", w),
    "min": lambda w: pl.col("val").rolling_min_by("ts", w),
    "max": lambda w: pl.col("val").rolling_max_by("ts", w),
    "median": lambda w: pl.col("val").rolling_median_by("ts", w),
}

# Approximate unit lengths, only used to compare durations against the horizon.
_DAY_SEC = 86400
_YEAR_DAYS = 365.25
_UNIT_SECONDS = {
    "y": _YEAR_DAYS * _DAY_SEC,
    "q": _YEAR_DAYS / 4 * _DAY_SEC,
    "mo": _YEAR_DAYS / 12 * _DAY_SEC,
    "w": 7 * _DAY_SEC,
    "d": _DAY_SEC,
    "h": 3600,
    "m": 60,
    "s": 1,
}
_DURATION = re.compile(r"(\d+)(mo|y|q|w|d|h|m|s)")


@lru_cache()
def _approx_seconds(duration: str) -> float:
    parts = _DURATION.findall(duration)
    if not parts or "".join(n + u for n, u in parts) != duration:
        raise ValueError(
            f"invalid duration '{duration}', expected e.g. '1d', '6h' or '1d12h' "
            f"using units {list(_UNIT_SECONDS)}"
        )
    return sum(int(n) * _UNIT_SECONDS[u] for n, u in parts)


_LAG_UNITS = ("y", "mo", "w", "d", "h", "m", "s")


def _by_unit(*per_unit: _T) -> dict[str, _T]:
    """Map the per-unit arguments (yearly ... secondly) to their unit suffixes."""
    return dict(zip(_LAG_UNITS, per_unit, strict=True))


def _join_lags(lf: pl.LazyFrame, lags: Sequence[tuple[int, str, str]]) -> pl.LazyFrame:
    """Add, for each `(n, unit, name)`, a column `name` with the value of `val` at
    `ts - n units`, looked up by timestamp (null if there is none)."""
    out = lf
    for n, unit, name in lags:
        past = lf.select(pl.col("ts").alias("_lag_ts"), pl.col("val").alias(name))
        out = (
            out.with_columns(_lag_ts=pl.col("ts").dt.offset_by(f"-{n}{unit}"))
            .join(past, on="_lag_ts", how="left", maintain_order="left")
            .drop("_lag_ts")
        )
    return out


def _maybe_drop_ts(lf: pl.LazyFrame, drop_ts: bool) -> pl.LazyFrame:
    return lf.drop("ts") if drop_ts else lf


def _maybe_drop_nulls(lf: pl.LazyFrame, drop_nulls: bool) -> pl.LazyFrame:
    return lf.drop_nulls() if drop_nulls else lf


def _cleanup(lf: pl.LazyFrame, drop_ts: bool, drop_nulls: bool) -> pl.LazyFrame:
    out = _maybe_drop_ts(lf, drop_ts)
    out = _maybe_drop_nulls(out, drop_nulls)
    return out
