from collections.abc import Sequence

import dataframely as dy
import numpy as np
import polars as pl


class XGBSchema(dy.Schema):
    ts = dy.Datetime(nullable=False, unique=True)
    val = dy.Float(nullable=False, allow_inf=False, allow_nan=False)


class TSFeatureEngineer:
    def __init__(self): ...

    def xgboost_datetime(
        self,
        lf: dy.LazyFrame[XGBSchema],
        lag_yearly: Sequence[int] | None = None,
        lag_monthly: Sequence[int] | None = None,
        lag_weekly: Sequence[int] | None = None,
        lag_daily: Sequence[int] | None = None,
        lag_hourly: Sequence[int] | None = None,
        lag_minutely: Sequence[int] | None = None,
        lag_secondly: Sequence[int] | None = None,
    ) -> pl.LazyFrame:
        """Build datetime and lag features for XGBoost.

        Each `lag_*` param lists how many units back to look, e.g. `lag_daily=(1, 7)`
        adds `lag_1d` and `lag_7d`. Lags are looked up by timestamp (calendar-aware),
        so gaps or irregular sampling don't shift values onto the wrong rows. Rows
        without a value for every lag are dropped.
        """
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

        diff_from_min_year = pl.col("ts").dt.year() - pl.col("ts").dt.year().min()
        return (
            out.with_columns(
                diff_from_min_year=diff_from_min_year,
                quarter=pl.col("ts").dt.quarter(),
                month=pl.col("ts").dt.month(),
                month_sin=(pl.col("ts").dt.month() * (2 * np.pi / 12)).sin(),
                month_cos=(pl.col("ts").dt.month() * (2 * np.pi / 12)).cos(),
                weekday_sin=(pl.col("ts").dt.weekday() * (2 * np.pi / 7)).sin(),
                weekday_cos=(pl.col("ts").dt.weekday() * (2 * np.pi / 7)).cos(),
                day=pl.col("ts").dt.day(),
                hour_sin=(pl.col("ts").dt.hour() * (2 * np.pi / 24)).sin(),
                hour_cos=(pl.col("ts").dt.hour() * (2 * np.pi / 24)).cos(),
            )
            .drop("ts")
            .drop_nulls()
        )
