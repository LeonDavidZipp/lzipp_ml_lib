from collections.abc import Sequence
from typing import Any, cast

import dataframely as dy
import pandas as pd
import polars as pl
import rustuna
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from rustuna.trial import TrialState

from ._space import CategoricalDimension, HyperparameterSpace
from ._types import (
    DIRECTIONS,
    ProphetEvalMetric,
    ProphetFinalFitData,
    RegressionFitResult,
    RegressionMetrics,
)


class ProphetSchema(dy.Schema):
    ds = dy.Datetime(nullable=False, unique=True)
    y = dy.Float(nullable=False, allow_inf=False, allow_nan=False)

    @dy.rule()
    def min_data_points(cls) -> pl.Expr:
        return pl.col("ds").unique().len() >= 100

    @dy.rule()
    def is_sorted(cls) -> pl.Expr:
        return pl.col("ds").is_sorted(descending=False)


def generate_fitted_model(
    y: dy.DataFrame[ProphetSchema],
    y_test: dy.DataFrame[ProphetSchema],
    regressors: Sequence[str] | None = None,
    interval_width: float = 0.8,
    search_space: HyperparameterSpace | None = None,
    n_trials: int = 100,
    final_fit_data: ProphetFinalFitData = "train",
    metric: ProphetEvalMetric = "mape",
    horizon: str | pd.Timedelta = "180 days",
) -> RegressionFitResult[Prophet]:
    """
    Tunes and fits a Prophet model.

    Runs `n_trials` hyperparameter trials, each scored with Prophet's time series
    cross validation on `y` (see below), then fits the final model with the best
    hyperparameters on the data selected by `final_fit_data` and scores it on
    `y_test`.

    Args:
        y (dy.DataFrame[ProphetSchema]): Training data with 'ds', 'y' and any
            regressor columns.
        y_test (dy.DataFrame[ProphetSchema]): Test data following `y` in time, with
            'ds', 'y' and the same regressor columns. Never used during tuning.
        regressors (Sequence[str] | None): Columns of `y` added as extra
            regressors. If None, every column except 'ds' and 'y' is used.
            Defaults to None.
        interval_width (float): Width of the uncertainty intervals. Overrides any
            `interval_width` in `search_space`. Defaults to 0.95.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
        n_trials (int): Number of hyperparameter trials. Defaults to 100.
        final_fit_data (ProphetFinalFitData): Data the final model is fit on with
            the best hyperparameters: `"train"` (`y` only) or `"train_test"` (`y`
            followed by `y_test`; the returned test metrics are then in-sample).
            Since Prophet validates via cross validation within `y` and takes no
            separate validation set, `"train_val"` behaves exactly like `"train"`
            and `"train_val_test"` exactly like `"train_test"`. Defaults to
            `"train"`.
        metric (ProphetEvalMetric): Metric the trials are minimized on, one of
            `"mape"`, `"mae"`, `"rmse"` or `"mse"`. Defaults to `"mape"`.
        horizon (str | pd.Timedelta): How far ahead each cross validation fold
            forecasts; should match the horizon the model is used for. Defaults to
            `"180 days"`.

    Returns:
        RegressionFitResult[Prophet]: The final model and its metrics on `y_test`.

    Raises:
        ValueError: If `y` is too short for cross validation with the given
            `horizon`, if no trial produced performance metrics, or if
            `final_fit_data` includes `y_test` and `y_test` doesn't start after
            `y` ends.

    Cross Validation:
        Each fold trains on all data before a cutoff and forecasts the following
        `horizon`. The first cutoff leaves at least `max(longest seasonality,
        3 * horizon)` of training data, and cutoffs are spaced to give at least 3
        folds where the data allows, and never further apart than `horizon`. A
        trial's score is `metric` averaged over the forecast horizon, with later
        horizon steps weighted up to twice as much as early ones.

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

        These untuned parameters are pinned to their library defaults, so every
        trial records them too:

        ```python
        {
            "growth": "linear",
            "n_changepoints": 25,
            "yearly_seasonality": "auto",
            "weekly_seasonality": "auto",
            "daily_seasonality": "auto",
            "mcmc_samples": 0,
            "uncertainty_samples": 1000,
            "scaling": "absmax",
            "interval_width": 0.8,
        }
        ```
    """
    search_space = HyperparameterSpace(
        search_space or HyperparameterSpace.default_prophet()
    )
    # the argument wins over the space, so trials record the width actually used
    search_space["interval_width"] = CategoricalDimension(
        "interval_width", [interval_width]
    )
    horizon = pd.Timedelta(horizon)
    regressors = _get_regressors(y, regressors)
    y_pd = y.to_pandas()

    def objective(trial: rustuna.Trial) -> float:
        params = {key: val.suggest(trial) for key, val in search_space.items()}
        model = _build_model(params, regressors)
        model.fit(y_pd)  # type: ignore

        initial, period = _cv_windows(model, horizon)
        results: pd.DataFrame = cross_validation(  # type: ignore
            model,
            horizon=horizon,
            initial=initial,
            period=period,
            parallel="processes",
        )

        m: pd.DataFrame | None = performance_metrics(results)
        if m is None:
            raise rustuna.TrialPruned("Performance metrics could not be computed.")
        metrics_df = pl.from_pandas(m)
        weights = pl.linear_space(1, 2, pl.len(), closed="none")
        val: float = metrics_df.select(
            (pl.col(metric) * weights).sum() / weights.sum()
        ).item()
        return val

    study = rustuna.create_study(direction=DIRECTIONS[metric])
    study.optimize(objective, n_trials=n_trials)
    if not any(t.state == TrialState.COMPLETE for t in study.trials):
        raise ValueError(
            f"All {n_trials} trials were pruned; performance metrics could not be "
            "computed for any hyperparameter combination."
        )

    final_df = _join_final_fit_data(final_fit_data, y, y_test, regressors)
    best_model = _build_model(study.best_trial.params, regressors).fit(
        final_df.to_pandas()
    )
    y_pred = best_model.predict(y_test.select("ds", *regressors).to_pandas())["yhat"]
    return RegressionFitResult(
        best_model, RegressionMetrics.calculate(y_test["y"], y_pred)
    )


def _get_regressors(y: pl.DataFrame, regressors: Sequence[str] | None) -> list[str]:
    internal_regressors = set([col for col in y.columns if col not in ("ds", "y")])
    if regressors and list(internal_regressors & set(regressors)) != list(regressors):
        ValueError(
            f"Not all provided regressors {regressors} could be found in the available "
            + f"regressor columns {internal_regressors}"
        )
    return (
        list(regressors)
        if regressors is not None
        else [col for col in y.columns if col not in ["ds", "y"]]
    )


def _join_final_fit_data(
    final_fit_data: ProphetFinalFitData,
    y: pl.DataFrame,
    y_test: pl.DataFrame,
    regressors: Sequence[str],
) -> pl.DataFrame:
    """Stack the data the final model is fit on, in train -> test order.

    Raises:
        ValueError: If `y_test` doesn't start after `y` ends.
    """
    cols = ["ds", "y", *regressors]
    if final_fit_data in ("train", "train_val"):
        return y.select(cols)
    if y_test["ds"].min() <= y["ds"].max():  # type: ignore
        raise ValueError(
            f"final_fit_data='{final_fit_data}' needs y_test to start after y ends"
        )
    return pl.concat([y.select(cols), y_test.select(cols)])


def _build_model(params: dict[str, Any], regressors: Sequence[str]) -> Prophet:
    model = Prophet(**params)  # type: ignore
    for reg in regressors:
        model = model.add_regressor(reg)  # type: ignore
    return model


def _cv_windows(
    model: Prophet, horizon: pd.Timedelta, min_folds: int = 3
) -> tuple[pd.Timedelta, pd.Timedelta]:
    """
    Derive `initial` and `period` for Prophet's `cross_validation` from the horizon.

    `initial` covers the longest active seasonality and at least 3 horizons (Prophet's
    default). `period` spreads `min_folds` cutoffs over the remaining data, capped at
    `horizon` so no data goes unscored, and snapped to a multiple of the data frequency.

    Args:
        model: A fitted Prophet model, used for its history and seasonalities.
        horizon: How far ahead each fold forecasts.
        min_folds: Number of folds to aim for if enough data is available.

    Returns:
        Tuple of (initial, period).
    """
    ds = cast(pd.DataFrame, model.history)["ds"]
    span = pd.Timedelta(ds.max() - ds.min())
    freq = pd.Timedelta(ds.diff().median())

    longest_season = max(
        (pd.Timedelta(days=s["period"]) for s in model.seasonalities.values()),
        default=pd.Timedelta(0),
    )
    initial = max(longest_season, 3 * horizon)

    usable = span - initial - horizon
    if usable < pd.Timedelta(0):
        raise ValueError(
            f"Cross validation needs at least {initial + horizon} of data, got {span}."
        )

    period = min(horizon, usable / max(min_folds - 1, 1))
    period = max(freq, (period // freq) * freq)
    return initial, period
