from typing import Literal

import dataframely as dy
import polars as pl


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
ProfileKey = Literal["hour", "weekday", "month", "hour_of_week", "day_of_year"]
RollingStat = Literal["mean", "std", "min", "max", "median"]
TrendUnit = Literal["s", "m", "h", "d"]
