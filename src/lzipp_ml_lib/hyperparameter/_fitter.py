from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Literal, Self, TypeVar

import dataframely as dy
import polars as pl
import rustuna
import xgboost as xgb
from numpy.typing import ArrayLike
from prophet import Prophet
from rustuna.study import StudyDirection
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from ._space import HyperparameterSpace

M = TypeVar("M", bound=xgb.XGBModel | Prophet)
R = TypeVar("R", bound=xgb.XGBRegressor)
RegressionEvalMetric = Literal["mape", "mae", "rmse", "mse", "r2"]
FinalFitData = Literal["train", "train_val", "train_val_test"]
ClassificationEvalMetric = Literal[
    "accuracy", "precision", "recall", "f1_score", "roc_auc", "log_loss"
]

_METRICS = {
    "mape": mean_absolute_percentage_error,
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
    "mse": mean_squared_error,
    "r2": r2_score,
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score,
    "roc_auc": roc_auc_score,
    "log_loss": log_loss,
}
_DIRECTIONS = {
    "mape": StudyDirection.MINIMIZE,
    "mae": StudyDirection.MINIMIZE,
    "rmse": StudyDirection.MINIMIZE,
    "mse": StudyDirection.MINIMIZE,
    "r2": StudyDirection.MAXIMIZE,
    "accuracy": StudyDirection.MAXIMIZE,
    "precision": StudyDirection.MAXIMIZE,
    "recall": StudyDirection.MAXIMIZE,
    "f1_score": StudyDirection.MAXIMIZE,
    "roc_auc": StudyDirection.MAXIMIZE,
    "log_loss": StudyDirection.MINIMIZE,
}


@dataclass
class RegressionMetrics:
    mape: float
    mae: float
    rmse: float
    mse: float
    r2: float

    @classmethod
    def calculate(cls, y_true: ArrayLike, y_pred: ArrayLike) -> Self:
        return cls(
            float(mean_absolute_percentage_error(y_true, y_pred)),
            float(mean_absolute_error(y_true, y_pred)),
            float(root_mean_squared_error(y_true, y_pred)),
            float(mean_squared_error(y_true, y_pred)),
            float(r2_score(y_true, y_pred)),
        )


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    log_loss: float


@dataclass
class RegressionFitResult(Generic[M]):
    model: M
    metrics: RegressionMetrics


@dataclass
class ClassificationFitResult(Generic[M]):
    model: M
    metrics: ClassificationMetrics


def _join_final_fit_data(
    final_fit_data: FinalFitData,
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    x_test: pl.DataFrame | None = None,
    y_test: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stack the data the final model is fit on, in train -> val -> test order.

    Raises:
        ValueError: If `final_fit_data` includes data that wasn't passed.
    """
    parts = [(x_train, y_train)]
    if final_fit_data in ("train_val", "train_val_test"):
        if not eval_set:
            raise ValueError(f"final_fit_data='{final_fit_data}' needs an eval_set")
        parts.extend(eval_set)
    if final_fit_data == "train_val_test":
        if x_test is None or y_test is None:
            raise ValueError("final_fit_data='train_val_test' needs x_test and y_test")
        parts.append((x_test, y_test))
    return pl.concat(x for x, _ in parts), pl.concat(y for _, y in parts)


def _fit_any_xgb_regressor(
    model_type: type[R],
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None,
    search_space: HyperparameterSpace,
    early_stopping_rounds: int,
    n_trials: int,
    final_fit_data: FinalFitData,
    metric: RegressionEvalMetric,
) -> RegressionFitResult[R]:
    def objective(trial: rustuna.Trial) -> float:
        params = search_space.suggest(trial)
        if "early_stopping_rounds" not in search_space:
            params["early_stopping_rounds"] = early_stopping_rounds
        model = model_type(**params)
        model.fit(x_train, y_train, eval_set=eval_set)
        y_pred = model.predict(x_test)
        return float(_METRICS[metric](y_test, y_pred))

    study = rustuna.create_study(direction=_DIRECTIONS[metric])
    study.optimize(objective, n_trials=n_trials)
    x_final, y_final = _join_final_fit_data(
        final_fit_data, x_train, y_train, eval_set, x_test, y_test
    )
    best_model = model_type(**study.best_trial.params)
    best_model.fit(x_final, y_final)
    y_pred_final = best_model.predict(x_test)
    return RegressionFitResult(
        best_model, RegressionMetrics.calculate(y_test, y_pred_final)
    )


def fit_xgb_regressor(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    search_space: HyperparameterSpace | None = None,
    early_stopping_rounds: int = 50,
    n_trials: int = 100,
    final_fit_data: FinalFitData = "train",
    metric: RegressionEvalMetric = "mape",
) -> RegressionFitResult[xgb.XGBRegressor]:
    """
    Tunes and fits an XGBoost regressor.

    Runs `n_trials` hyperparameter trials, each fit on the training data and scored
    on the test data with `metric`, then fits the final model with the best
    hyperparameters on the data selected by `final_fit_data`.

    Args:
        x_train (pl.DataFrame): Training features.
        y_train (pl.DataFrame): Training target.
        x_test (pl.DataFrame): Test features. Each trial is scored on them, and so
            are the returned metrics.
        y_test (pl.DataFrame): Test target.
        eval_set (Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None): Validation
            `(x, y)` pairs XGBoost monitors during each trial's fit; the last one
            is used for early stopping. Defaults to None.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
        early_stopping_rounds (int): Stop a trial's boosting after this many rounds
            without improvement on the last `eval_set` pair. Ignored if
            `search_space` tunes `early_stopping_rounds` itself. Defaults to 50.
        n_trials (int): Number of hyperparameter trials. Defaults to 100.
        final_fit_data (FinalFitData): Data the final model is fit on with the
            best hyperparameters: `"train"` (training data only), `"train_val"`
            (plus all `eval_set` data) or `"train_val_test"` (plus the test data;
            the returned test metrics are then in-sample). Defaults to `"train"`.
        metric (RegressionEvalMetric): Metric the trials are optimized for, one of
            `"mape"`, `"mae"`, `"rmse"`, `"mse"` (minimized) or `"r2"`
            (maximized). Defaults to `"mape"`.

    Returns:
        RegressionFitResult[xgb.XGBRegressor]: The final model and its metrics on
            the test data.

    Raises:
        ValueError: If `final_fit_data` includes `eval_set` data but none was
            passed.

    Default Hyperparameter Space:
        If `search_space` is None, the following search space is used:

        ```python
        {
            "n_estimators": IntegerDimension(
                "n_estimators", low=100, high=1000, step=50
            ),
            "max_depth": IntegerDimension("max_depth", low=3, high=10),
            "learning_rate": FloatDimension(
                "learning_rate", low=1e-3, high=0.3, log=True
            ),
            "subsample": FloatDimension("subsample", low=0.5, high=1.0),
            "colsample_bytree": FloatDimension("colsample_bytree", low=0.5, high=1.0),
            "min_child_weight": IntegerDimension("min_child_weight", low=1, high=10),
            "gamma": FloatDimension("gamma", low=0.0, high=5.0),
            "reg_alpha": FloatDimension("reg_alpha", low=1e-8, high=100.0, log=True),
            "reg_lambda": FloatDimension("reg_lambda", low=1e-8, high=100.0, log=True),
        }
        ```
    """
    search_space = search_space or HyperparameterSpace.default_xgb_regressor()
    return _fit_any_xgb_regressor(
        model_type=xgb.XGBRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        eval_set=eval_set,
        search_space=search_space,
        early_stopping_rounds=early_stopping_rounds,
        n_trials=n_trials,
        final_fit_data=final_fit_data,
        metric=metric,
    )


def fit_xgb_rf_regressor(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    search_space: HyperparameterSpace | None = None,
    early_stopping_rounds: int = 50,
    n_trials: int = 100,
    final_fit_data: FinalFitData = "train",
    metric: RegressionEvalMetric = "mape",
) -> RegressionFitResult[xgb.XGBRFRegressor]:
    """
    Tunes and fits an XGBoost random forest regressor.

    Runs `n_trials` hyperparameter trials, each fit on the training data and scored
    on the test data with `metric`, then fits the final model with the best
    hyperparameters on the data selected by `final_fit_data`.

    Args:
        x_train (pl.DataFrame): Training features.
        y_train (pl.DataFrame): Training target.
        x_test (pl.DataFrame): Test features. Each trial is scored on them, and so
            are the returned metrics.
        y_test (pl.DataFrame): Test target.
        eval_set (Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None): Validation
            `(x, y)` pairs XGBoost monitors during each trial's fit; the last one
            is used for early stopping. Defaults to None.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
        early_stopping_rounds (int): Stop a trial's boosting after this many rounds
            without improvement on the last `eval_set` pair. A random forest is
            built in a single boosting round, so this has practically no effect.
            Ignored if `search_space` tunes `early_stopping_rounds` itself.
            Defaults to 50.
        n_trials (int): Number of hyperparameter trials. Defaults to 100.
        final_fit_data (FinalFitData): Data the final model is fit on with the
            best hyperparameters: `"train"` (training data only), `"train_val"`
            (plus all `eval_set` data) or `"train_val_test"` (plus the test data;
            the returned test metrics are then in-sample). Defaults to `"train"`.
        metric (RegressionEvalMetric): Metric the trials are optimized for, one of
            `"mape"`, `"mae"`, `"rmse"`, `"mse"` (minimized) or `"r2"`
            (maximized). Defaults to `"mape"`.

    Returns:
        RegressionFitResult[xgb.XGBRFRegressor]: The final model and its metrics on
            the test data.

    Raises:
        ValueError: If `final_fit_data` includes `eval_set` data but none was
            passed.

    Default Hyperparameter Space:
        If `search_space` is None, the following search space is used:

        ```python
        {
            "n_estimators": IntegerDimension(
                "n_estimators", low=100, high=1000, step=50
            ),
            "max_depth": IntegerDimension("max_depth", low=5, high=20),
            "subsample": FloatDimension("subsample", low=0.5, high=0.95),
            "colsample_bynode": FloatDimension("colsample_bynode", low=0.4, high=0.9),
            "min_child_weight": IntegerDimension("min_child_weight", low=1, high=10),
        }
        ```
    """
    search_space = search_space or HyperparameterSpace.default_xgb_rf_regressor()
    return _fit_any_xgb_regressor(
        model_type=xgb.XGBRFRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        eval_set=eval_set,
        search_space=search_space,
        early_stopping_rounds=early_stopping_rounds,
        n_trials=n_trials,
        final_fit_data=final_fit_data,
        metric=metric,
    )


def fit_xgb_classifier(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    search_space: HyperparameterSpace | None = None,
    final_fit_data: FinalFitData = "train",
) -> ClassificationFitResult[xgb.XGBClassifier]:
    """
    Fits an XGBoost classifier.

    Args:
        x_train (pl.DataFrame): Training features.
        y_train (pl.DataFrame): Training target.
        eval_set (Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None): Validation
            `(x, y)` pairs used to score each hyperparameter trial. Defaults to None.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
        final_fit_data (FinalFitData): Data the final model is fit on with the
            best hyperparameters: `"train"` (training data only), `"train_val"`
            (plus all `eval_set` data) or `"train_val_test"` (plus the test data;
            the returned test metrics are then in-sample). Defaults to `"train"`.

    Returns:
        HyperparameterFitResult[xgb.XGBClassifier]: The fitted model and associated
            metrics.

    Default Hyperparameter Space:
        If `search_space` is None, the following search space is used:

        ```python
        {
            "n_estimators": IntegerDimension(
                "n_estimators", low=100, high=1000, step=50
            ),
            "max_depth": IntegerDimension("max_depth", low=3, high=10),
            "learning_rate": FloatDimension(
                "learning_rate", low=1e-3, high=0.3, log=True
            ),
            "subsample": FloatDimension("subsample", low=0.5, high=1.0),
            "colsample_bytree": FloatDimension("colsample_bytree", low=0.5, high=1.0),
            "min_child_weight": IntegerDimension("min_child_weight", low=1, high=10),
            "gamma": FloatDimension("gamma", low=0.0, high=5.0),
            "reg_alpha": FloatDimension("reg_alpha", low=1e-8, high=100.0, log=True),
            "reg_lambda": FloatDimension("reg_lambda", low=1e-8, high=100.0, log=True),
        }
        ```
    """
    # space = search_space or HyperparameterSpace.default_space_from_model(
    #     self._model_type
    # )

    # def objective(trial: rustuna.Trial) -> float:
    #     params = {key: val.suggest(trial) for key, val in space.items()}
    #     model = self._model_type(**params)  # type: ignore
    #     model.fit(x_train, y_train, eval_set=eval_set, verbose=False)

    #     # Extract and return the validation metric
    #     return calculate_rmse(model, validation_sets[0][0], validation_sets[0][1])

    # rustuna.create_study().optimize(objective, n_trials=50)
    # # Finalization logic here...
    # return HyperparameterFitResult(...)
    ...


def fit_xgb_rf_classifier(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    search_space: HyperparameterSpace | None = None,
    final_fit_data: FinalFitData = "train",
) -> ClassificationFitResult[xgb.XGBRFClassifier]:
    """
    Fits an XGBoost random forest classifier.

    Args:
        x_train (pl.DataFrame): Training features.
        y_train (pl.DataFrame): Training target.
        eval_set (Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None): Validation
            `(x, y)` pairs used to score each hyperparameter trial. Defaults to None.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
        final_fit_data (FinalFitData): Data the final model is fit on with the
            best hyperparameters: `"train"` (training data only), `"train_val"`
            (plus all `eval_set` data) or `"train_val_test"` (plus the test data;
            the returned test metrics are then in-sample). Defaults to `"train"`.

    Returns:
        HyperparameterFitResult[xgb.XGBRFClassifier]: The fitted model and associated
            metrics.

    Default Hyperparameter Space:
        If `search_space` is None, the following search space is used:

        ```python
        {
            "n_estimators": IntegerDimension(
                "n_estimators", low=100, high=1000, step=50
            ),
            "max_depth": IntegerDimension("max_depth", low=5, high=20),
            "subsample": FloatDimension("subsample", low=0.5, high=0.95),
            "colsample_bynode": FloatDimension("colsample_bynode", low=0.4, high=0.9),
            "min_child_weight": IntegerDimension("min_child_weight", low=1, high=10),
        }
        ```
    """
    ...


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
    # space = search_space or HyperparameterSpace.default_prophet()

    # def objective(trial: rustuna.Trial) -> float:
    #     params = {key: val.suggest(trial) for key, val in space.items()}
    #     model = Prophet(**params)  # type: ignore
    #     model.fit(df.to_pandas())

    #     # TODO: implement
    #     return 0.0

    # rustuna.create_study().optimize(objective, n_trials=50)
    # return HyperparameterFitResult(model=Prophet(), metrics=Metrics(mape=0.0))
    ...
