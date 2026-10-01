from collections.abc import Sequence

import dataframely as dy
import numpy as np
import polars as pl


class TSSchema(dy.Schema):
    ts = dy.Datetime(nullable=False, unique=True)
    val = dy.Float(nullable=False, allow_inf=False, allow_nan=False)

    @dy.rule()
    def is_sorted(cls) -> pl.Expr:
        return pl.col("ts").is_sorted(descending=False)


class TSFeatureEngineer:
    """Feature builders for a `ts` / `val` time series.

    Each method adds columns and keeps `ts` unless `drop_ts=True`, so they compose
    with `pipe`:

        features = (
            lf.pipe(fe.lag, daily=(1, 7))
            .pipe(fe.cyclical)
            .pipe(fe.calendar, drop_ts=True)
            .drop_nulls()
        )
    """

    def __init__(self): ...

    def lag(
        self,
        lf: dy.LazyFrame[TSSchema],
        yearly: Sequence[int] | None = None,
        monthly: Sequence[int] | None = None,
        weekly: Sequence[int] | None = None,
        daily: Sequence[int] | None = None,
        hourly: Sequence[int] | None = None,
        minutely: Sequence[int] | None = None,
        secondly: Sequence[int] | None = None,
        drop_ts: bool = False,
    ) -> pl.LazyFrame:
        """Add lagged copies of `val`.

        Lags are looked up by timestamp (calendar-aware), so gaps or irregular
        sampling don't shift values onto the wrong rows. Rows with no value `n`
        units back get a null.

        Args:
            lf (pl.LazyFrame): Frame with a unique `ts` datetime column and a `val`
                column, e.g. a `dy.LazyFrame[TSSchema]`.
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

        Returns:
            pl.LazyFrame: `lf` with the requested lag columns appended (without
                `ts` if `drop_ts`).

        Raises:
            ValueError: If any lag is smaller than 1.
        """
        lags = {
            "y": yearly,
            "mo": monthly,
            "w": weekly,
            "d": daily,
            "h": hourly,
            "m": minutely,
            "s": secondly,
        }
        for unit, ns in lags.items():
            if ns is not None and any(n < 1 for n in ns):
                raise ValueError(
                    f"lags must be positive integers, got {ns} for unit '{unit}'"
                )

        out = lf
        for unit, ns in lags.items():
            for n in ns or ():
                past = lf.select(
                    pl.col("ts").alias("_lag_ts"), pl.col("val").alias(f"lag_{n}{unit}")
                )
                out = (
                    out.with_columns(_lag_ts=pl.col("ts").dt.offset_by(f"-{n}{unit}"))
                    .join(past, on="_lag_ts", how="left")
                    .drop("_lag_ts")
                )
        return out.drop("ts") if drop_ts else out

    def cyclical(
        self, lf: dy.LazyFrame[TSSchema], drop_ts: bool = False
    ) -> pl.LazyFrame:
        """Add sine/cosine encodings of month, weekday and hour.

        Args:
            lf (pl.LazyFrame): Frame with a `ts` datetime column.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with `month_sin`, `month_cos`, `weekday_sin`,
                `weekday_cos`, `hour_sin` and `hour_cos` appended (without `ts` if
                `drop_ts`).
        """
        cyclical = {
            "month": (pl.col("ts").dt.month(), 12),
            "weekday": (pl.col("ts").dt.weekday(), 7),
            "hour": (pl.col("ts").dt.hour(), 24),
        }
        features: list[pl.Expr] = []
        for name, (expr, period) in cyclical.items():
            angle = expr * (2 * np.pi / period)
            features.append(angle.sin().alias(f"{name}_sin"))
            features.append(angle.cos().alias(f"{name}_cos"))
        out = lf.with_columns(features)
        return out.drop("ts") if drop_ts else out

    def calendar(
        self, lf: dy.LazyFrame[TSSchema], drop_ts: bool = False
    ) -> pl.LazyFrame:
        """Add raw calendar features.

        Args:
            lf (pl.LazyFrame): Frame with a `ts` datetime column.
            drop_ts (bool): Drop `ts` from the result. Defaults to False.

        Returns:
            pl.LazyFrame: `lf` with `diff_from_min_year`, `quarter`, `month`, `day`
                and `hour` appended (without `ts` if `drop_ts`).
        """
        out = lf.with_columns(
            diff_from_min_year=pl.col("ts").dt.year() - pl.col("ts").dt.year().min(),
            quarter=pl.col("ts").dt.quarter(),
            month=pl.col("ts").dt.month(),
            day=pl.col("ts").dt.day(),
            hour=pl.col("ts").dt.hour(),
        )
        return out.drop("ts") if drop_ts else out
