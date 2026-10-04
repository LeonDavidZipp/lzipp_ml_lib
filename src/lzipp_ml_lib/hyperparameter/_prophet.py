
import dataframely as dy
import pandas as pd
import polars as pl
import rustuna
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from collections.abc import Sequence

from ._space import HyperparameterSpace
from ._types import DIRECTIONS, FinalFitData, RegressionEvalMetric, RegressionFitResult, RegressionMetrics


class ProphetSchema(dy.Schema):
    ds = dy.Datetime(nullable=False, unique=True)
    y = dy.Float(nullable=False, allow_inf=False, allow_nan=False)

    @dy.rule()
    def min_data_points(cls) -> pl.Expr:
        return pl.col("ds").unique().len() >= 100

    @dy.rule()
    def is_sorted(cls) -> pl.Expr:
        return pl.col("ds").is_sorted(descending=False)


def fit_prophet(
    df: dy.DataFrame[ProphetSchema],
    search_space: HyperparameterSpace | None = None,
    cross_validation_threshold: int = 1000,
    metric: RegressionEvalMetric = "mape",
):
    """
    Fits a prophet model.

    Args:
        df (dy.DataFrame[ProphetSchema]): A prophet-typical dataframe containing
            'ds' and 'y' columns.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
        cross_validation_threshold (int): Number of datapoints needed to perform cross
            validation instead of a train-test-split. Defaults to 1000.

    Returns:
        HyperparameterFitResult[Prophet]: The fitted model and associated metrics.

    Default Hyperparameter Space:
        If `search_space` is None, the following search space is used:

        ```python
        {
            "changepoint_prior_scale": FloatDimension(
                "changepoint_prior_scale", low=0.001, high=0.5, log=True
            ),
            "seasonality_prior_scale": FloatDimension(
                "seasonality_prior_scale", low=0.01, high=10.0, log=True
            ),
            "holidays_prior_scale": FloatDimension(
                "holidays_prior_scale", low=0.01, high=10.0, log=True
            ),
            "seasonality_mode": CategoricalDimension(
                "seasonality_mode", choices=["additive", "multiplicative"]
            ),
            "changepoint_range": FloatDimension(
                "changepoint_range", low=0.8, high=0.95
            ),
        }
        ```
    """
    space = search_space or HyperparameterSpace.default_prophet()

    def objective(trial: rustuna.Trial) -> float:
        params = {key: val.suggest(trial) for key, val in space.items()}
        model = Prophet(**params)  # type: ignore
        model.fit(df.to_pandas())

        # TODO: implement
        return 0.0

    rustuna.create_study().optimize(objective, n_trials=50)
    return RegressionFitResult(model=Prophet(), metrics=Metrics(mape=0.0))
    ...


def generate_fitted_model(
    y: dy.DataFrame[ProphetSchema],
    y_test: pl.DataFrame,
    regressors: Sequence[str] | None = None,
    interval_width: float = 0.95,
    search_space: HyperparameterSpace | None = None,
    n_trials: int = 100,
    final_fit_data: FinalFitData = "train",
    metric: RegressionEvalMetric = "mape",
) -> Prophet:
    """
    Fit a Prophet model with the given df and search space.

    Args:
        df: Lazy frame containing the time series df
        search_space: Configuration for parameter search

    Returns:
        Fitted Prophet model
    """

    search_space = search_space or HyperparameterSpace.default_prophet()
    y_pd = y.to_pandas()
    regressors: list[str] = [col for col in y.columns if col not in ["ds", "y"]]

    def objective(trial: rustuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object for parameter suggestions

        Returns:
            Score for the current parameter combination
        """

        params = {key: val.suggest(trial) for key, val in search_space.items()}
        params["interval_width"] = interval_width
        model = Prophet(**params)  # type: ignore
        for reg in regressors:
            model = model.add_regressor(reg)  # type: ignore
        model.fit(y_pd)  # type: ignore

        results: pd.DataFrame = cross_validation(  # type: ignore
            model,
            horizon="180 days",
            initial="365 days",
            period="180 days",
            parallel="processes",
        )

        m: pd.DataFrame | None = performance_metrics(results)
        if m is None:
            raise ValueError("Performance metrics could not be computed.")
        metrics_df = pl.from_pandas(m)
        weights = pl.linear_space(1, 2, pl.len(), closed="none")
        val: float = metrics_df.select(pl.col(metric) * weights / weights.sum()).item()
        return val

    study = rustuna.create_study(direction=DIRECTIONS[metric])
    study.optimize(objective, n_trials=n_trials)

    best_model = Prophet(**study.best_trial.params).fit(y_pd)  # type: ignore
    inp = best_model.make_future_dataframe(y_test.height)
    y_pred = best_model.predict(inp)["yhat"]
    RegressionFitResult(best_model, RegressionMetrics.calculate(y_test, y_pred))
