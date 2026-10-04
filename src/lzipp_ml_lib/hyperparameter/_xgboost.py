import warnings
from collections.abc import Callable, Sequence
from typing import TypeVar

import polars as pl
import rustuna
import xgboost as xgb
from rustuna.study import StudyDirection

from ._space import HyperparameterSpace
from ._types import (
    DIRECTIONS,
    METRICS,
    ClassificationEvalMetric,
    ClassificationFitResult,
    ClassificationMetrics,
    FinalFitData,
    RegressionEvalMetric,
    RegressionFitResult,
    RegressionMetrics,
)

T = TypeVar("T", bound=xgb.XGBModel)
R = TypeVar("R", bound=xgb.XGBRegressor)
C = TypeVar("C", bound=xgb.XGBClassifier)


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
            without improvement on the last `eval_set` pair; the final model is
            then trained for as many rounds as the best trial used. Ignored
            without an `eval_set` or if `search_space` tunes
            `early_stopping_rounds` itself. Defaults to 50.
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
        early_stopping_rounds=early_stopping_rounds if eval_set else None,
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
            `(x, y)` pairs XGBoost monitors during each trial's fit. Defaults to
            None.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
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
        early_stopping_rounds=None,  # unsupported by XGBRFRegressor
        n_trials=n_trials,
        final_fit_data=final_fit_data,
        metric=metric,
    )


def fit_xgb_classifier(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    search_space: HyperparameterSpace | None = None,
    early_stopping_rounds: int = 50,
    n_trials: int = 100,
    final_fit_data: FinalFitData = "train",
    metric: ClassificationEvalMetric = "log_loss",
) -> ClassificationFitResult[xgb.XGBClassifier]:
    """
    Tunes and fits an XGBoost classifier.

    Runs `n_trials` hyperparameter trials, each fit on the training data and scored
    on the test data with `metric`, then fits the final model with the best
    hyperparameters on the data selected by `final_fit_data`.

    Args:
        x_train (pl.DataFrame): Training features.
        y_train (pl.DataFrame): Training class labels `0..k-1`.
        x_test (pl.DataFrame): Test features. Each trial is scored on them, and so
            are the returned metrics.
        y_test (pl.DataFrame): Test class labels.
        eval_set (Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None): Validation
            `(x, y)` pairs XGBoost monitors during each trial's fit; the last one
            is used for early stopping. Defaults to None.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
        early_stopping_rounds (int): Stop a trial's boosting after this many rounds
            without improvement on the last `eval_set` pair; the final model is
            then trained for as many rounds as the best trial used. Ignored
            without an `eval_set` or if `search_space` tunes
            `early_stopping_rounds` itself. Defaults to 50.
        n_trials (int): Number of hyperparameter trials. Defaults to 100.
        final_fit_data (FinalFitData): Data the final model is fit on with the
            best hyperparameters: `"train"` (training data only), `"train_val"`
            (plus all `eval_set` data) or `"train_val_test"` (plus the test data;
            the returned test metrics are then in-sample). Defaults to `"train"`.
        metric (ClassificationEvalMetric): Metric the trials are optimized for, one
            of `"accuracy"`, `"precision"`, `"recall"`, `"f1_score"`, `"roc_auc"`
            (maximized) or `"log_loss"` (minimized). See
            `ClassificationMetrics.calculate` for the multiclass averaging.
            Defaults to `"log_loss"`.

    Returns:
        ClassificationFitResult[xgb.XGBClassifier]: The final model and its metrics on
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
    search_space = search_space or HyperparameterSpace.default_xgb_classifier()
    return _fit_any_xgb_classifier(
        model_type=xgb.XGBClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        eval_set=eval_set,
        search_space=search_space,
        early_stopping_rounds=early_stopping_rounds if eval_set else None,
        n_trials=n_trials,
        final_fit_data=final_fit_data,
        metric=metric,
    )


def fit_xgb_rf_classifier(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    search_space: HyperparameterSpace | None = None,
    n_trials: int = 100,
    final_fit_data: FinalFitData = "train",
    metric: ClassificationEvalMetric = "log_loss",
) -> ClassificationFitResult[xgb.XGBRFClassifier]:
    """
    Tunes and fits an XGBoost random forest classifier.

    Runs `n_trials` hyperparameter trials, each fit on the training data and scored
    on the test data with `metric`, then fits the final model with the best
    hyperparameters on the data selected by `final_fit_data`.

    Args:
        x_train (pl.DataFrame): Training features.
        y_train (pl.DataFrame): Training class labels `0..k-1`.
        x_test (pl.DataFrame): Test features. Each trial is scored on them, and so
            are the returned metrics.
        y_test (pl.DataFrame): Test class labels.
        eval_set (Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None): Validation
            `(x, y)` pairs XGBoost monitors during each trial's fit. Defaults to
            None.
        search_space (HyperparameterSpace | None): The hyperparameter space used
            for optimization. If None, defaults to a basic hyperparameter space (see
            below).
        n_trials (int): Number of hyperparameter trials. Defaults to 100.
        final_fit_data (FinalFitData): Data the final model is fit on with the
            best hyperparameters: `"train"` (training data only), `"train_val"`
            (plus all `eval_set` data) or `"train_val_test"` (plus the test data;
            the returned test metrics are then in-sample). Defaults to `"train"`.
        metric (ClassificationEvalMetric): Metric the trials are optimized for, one
            of `"accuracy"`, `"precision"`, `"recall"`, `"f1_score"`, `"roc_auc"`
            (maximized) or `"log_loss"` (minimized). See
            `ClassificationMetrics.calculate` for the multiclass averaging.
            Defaults to `"log_loss"`.

    Returns:
        ClassificationFitResult[xgb.XGBRFClassifier]: The final model and its metrics on
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
    search_space = search_space or HyperparameterSpace.default_xgb_rf_classifier()
    return _fit_any_xgb_classifier(
        model_type=xgb.XGBRFClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        eval_set=eval_set,
        search_space=search_space,
        early_stopping_rounds=None,  # unsupported by XGBRFClassifier
        n_trials=n_trials,
        final_fit_data=final_fit_data,
        metric=metric,
    )


def _fit_any_xgb_classifier(
    model_type: type[C],
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None,
    search_space: HyperparameterSpace,
    early_stopping_rounds: int | None,
    n_trials: int,
    final_fit_data: FinalFitData,
    metric: ClassificationEvalMetric,
) -> ClassificationFitResult[C]:
    def score(model: C) -> float:
        return getattr(_classification_metrics(model, x_test, y_test), metric)

    best_model = _tune_and_fit_xgb(
        model_type=model_type,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        eval_set=eval_set,
        search_space=search_space,
        early_stopping_rounds=early_stopping_rounds,
        n_trials=n_trials,
        final_fit_data=final_fit_data,
        score=score,
        direction=DIRECTIONS[metric],
    )
    return ClassificationFitResult(
        best_model, _classification_metrics(best_model, x_test, y_test)
    )


def _tune_and_fit_xgb(
    model_type: type[T],
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None,
    search_space: HyperparameterSpace,
    early_stopping_rounds: int | None,
    n_trials: int,
    final_fit_data: FinalFitData,
    score: Callable[[T], float],
    direction: StudyDirection,
) -> T:
    """Tune `model_type` with `score` (a fitted model -> its test score), then fit
    the final model with the best hyperparameters on `final_fit_data`."""
    if early_stopping_rounds is not None and not eval_set:
        warnings.warn(
            f"early_stopping_rounds={early_stopping_rounds} is ignored because no "
            "eval_set was passed; trials train all n_estimators rounds",
            stacklevel=4,
        )

    def objective(trial: rustuna.Trial) -> float:
        params = search_space.suggest(trial)
        if (
            early_stopping_rounds is not None
            and eval_set
            and "early_stopping_rounds" not in search_space
        ):
            params["early_stopping_rounds"] = early_stopping_rounds
        model = model_type(**params)
        model.fit(x_train, y_train, eval_set=eval_set)
        if params.get("early_stopping_rounds") is not None:
            trial.set_user_attr("n_estimators", str(model.best_iteration + 1))
        return score(model)

    study = rustuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    x_final, y_final = _join_final_fit_data(
        final_fit_data, x_train, y_train, eval_set, x_test, y_test
    )
    # The final fit has no eval_set to stop on (it may be part of the final data),
    # so train exactly as many rounds as the best trial used instead.
    best_params = dict(study.best_trial.params)
    best_params.pop("early_stopping_rounds", None)
    if "n_estimators" in study.best_trial.user_attrs:
        best_params["n_estimators"] = int(study.best_trial.user_attrs["n_estimators"])
    best_model = model_type(**best_params)
    best_model.fit(x_final, y_final)
    return best_model


def _fit_any_xgb_regressor(
    model_type: type[R],
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None,
    search_space: HyperparameterSpace,
    early_stopping_rounds: int | None,
    n_trials: int,
    final_fit_data: FinalFitData,
    metric: RegressionEvalMetric,
) -> RegressionFitResult[R]:
    def score(model: R) -> float:
        return float(METRICS[metric](y_test, model.predict(x_test)))

    best_model = _tune_and_fit_xgb(
        model_type=model_type,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        eval_set=eval_set,
        search_space=search_space,
        early_stopping_rounds=early_stopping_rounds,
        n_trials=n_trials,
        final_fit_data=final_fit_data,
        score=score,
        direction=DIRECTIONS[metric],
    )
    return RegressionFitResult(
        best_model, RegressionMetrics.calculate(y_test, best_model.predict(x_test))
    )


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


def _classification_metrics(
    model: xgb.XGBClassifier, x: pl.DataFrame, y: pl.DataFrame
) -> ClassificationMetrics:
    return ClassificationMetrics.calculate(y, model.predict(x), model.predict_proba(x))
