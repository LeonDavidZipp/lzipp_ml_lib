from collections.abc import Sequence

import dataframely as dy
import numpy as np
import polars as pl


class XGBSchema(dy.Schema):
    ts = dy.Datetime(nullable=False, unique=True)
    val = dy.Float(nullable=False, allow_inf=False, allow_nan=False)
    
    @dy.rule()
    def is_sorted(cls) -> pl.Expr:
        return pl.col("ts").is_sorted(descending=False)


class TSFeatureEngineer:
    def __init__(self): ...

    def xgboost(
        self,
        lf: dy.LazyFrame[XGBSchema],
        lag_yearly: Sequence[int] | None = None,
        lag_monthly: Sequence[int] | None = None,
        lag_weekly: Sequence[int] | None = None,
        lag_daily: Sequence[int] | None = None,
        lag_hourly: Sequence[int] | None = None,
        lag_minutely: Sequence[int] | None = None,
        lag_secondly: Sequence[int] | None = None,
        cyclicality: bool = True,
        drop_nulls: bool = True,
    ) -> pl.LazyFrame:
        """Build datetime and lag features for XGBoost.

        Lags are looked up by timestamp (calendar-aware), so gaps or irregular
        sampling don't shift values onto the wrong rows.

        Args:
            lf (dy.LazyFrame[XGBSchema]): Validated frame with a unique `ts` datetime
                column and a `val` float column.
            lag_yearly (Sequence[int] | None): Yearly lags to add, e.g. `(1,)` adds
                `lag_1y`. Defaults to None (no yearly lags).
            lag_monthly (Sequence[int] | None): Monthly lags, named `lag_{n}mo`.
                Defaults to None.
            lag_weekly (Sequence[int] | None): Weekly lags, named `lag_{n}w`.
                Defaults to None.
            lag_daily (Sequence[int] | None): Daily lags, named `lag_{n}d`, e.g.
                `(1, 7)` adds `lag_1d` and `lag_7d`. Defaults to None.
            lag_hourly (Sequence[int] | None): Hourly lags, named `lag_{n}h`.
                Defaults to None.
            lag_minutely (Sequence[int] | None): Minutely lags, named `lag_{n}m`.
                Defaults to None.
            lag_secondly (Sequence[int] | None): Secondly lags, named `lag_{n}s`.
                Defaults to None.
            cyclicality (bool): Add sine/cosine encodings of month, weekday and hour
                (`month_sin`, `month_cos`, ...). Defaults to True.
            drop_nulls (bool): Drop rows with a null in any column, e.g. rows with
                no value `n` units back for a requested lag. Defaults to True.

        Returns:
            pl.LazyFrame: `val`, the requested lag columns, the cyclical features
                (if enabled), and `diff_from_min_year`, `quarter`, `month`, `day`,
                `hour`. `ts` is dropped.

        Raises:
            ValueError: If any lag is smaller than 1.
        """
        out = self._add_lags(
            lf,
            lag_yearly,
            lag_monthly,
            lag_weekly,
            lag_daily,
            lag_hourly,
            lag_minutely,
            lag_secondly,
        )
        cyclical_features = self._generate_cyclical_features() if cyclicality else ()
        diff_from_min_year = pl.col("ts").dt.year() - pl.col("ts").dt.year().min()
        out = out.with_columns(
            *cyclical_features,
            diff_from_min_year=diff_from_min_year,
            quarter=pl.col("ts").dt.quarter(),
            month=pl.col("ts").dt.month(),
            day=pl.col("ts").dt.day(),
            hour=pl.col("ts").dt.hour(),
        ).drop("ts")
        if drop_nulls:
            out = out.drop_nulls()
        return out

    def _add_lags(
        self,
        lf: pl.LazyFrame,
        lag_yearly: Sequence[int] | None,
        lag_monthly: Sequence[int] | None,
        lag_weekly: Sequence[int] | None,
        lag_daily: Sequence[int] | None,
        lag_hourly: Sequence[int] | None,
        lag_minutely: Sequence[int] | None,
        lag_secondly: Sequence[int] | None,
    ) -> pl.LazyFrame:
        lags = {
            "y": lag_yearly,
            "mo": lag_monthly,
            "w": lag_weekly,
            "d": lag_daily,
            "h": lag_hourly,
            "m": lag_minutely,
            "s": lag_secondly,
        }
        for unit, ns in lags.items():
            if ns is not None and any(n < 1 for n in ns):
                raise ValueError(
                    f"lags must be positive integers, got {ns} for unit '{unit}'"
                )

        for unit, ns in lags.items():
            for n in ns or ():
                past = lf.select(
                    pl.col("ts").alias("_lag_ts"), pl.col("val").alias(f"lag_{n}{unit}")
                )
                lf = (
                    lf.with_columns(_lag_ts=pl.col("ts").dt.offset_by(f"-{n}{unit}"))
                    .join(past, on="_lag_ts", how="left")
                    .drop("_lag_ts")
                )
        return lf

    def _generate_cyclical_features(self):
        cyclical = {
            "month": (pl.col("ts").dt.month(), 12),
            "weekday": (pl.col("ts").dt.weekday(), 7),
            "hour": (pl.col("ts").dt.hour(), 24),
        }
        cyclical_features: list[pl.Expr] = []
        for name, (expr, period) in cyclical.items():
            angle = expr * (2 * np.pi / period)
            cyclical_features.append(angle.sin().alias(f"{name}_sin"))
            cyclical_features.append(angle.cos().alias(f"{name}_cos"))
        return cyclical_features
