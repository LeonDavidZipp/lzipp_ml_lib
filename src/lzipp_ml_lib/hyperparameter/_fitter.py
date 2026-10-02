from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import dataframely as dy
import polars as pl
import rustuna
import xgboost as xgb
from prophet import Prophet

from ._space import HyperparameterSpace, M


@dataclass
class Metrics:
    mape: float | None = None
    mae: float | None = None
    rmse: float | None = None
    mse: float | None = None
    r2: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    roc_auc: float | None = None
    log_loss: float | None = None


@dataclass
class HyperparameterFitResult(Generic[M]):
    model: M
    metrics: Metrics


def fit_xgb_regressor(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    hyperparameter_space: HyperparameterSpace | None = None,
    refit_with_all: bool = False,
) -> HyperparameterFitResult[xgb.XGBRegressor]: ...


def fit_xgb_rf_regressor(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    hyperparameter_space: HyperparameterSpace | None = None,
    refit_with_all: bool = False,
) -> HyperparameterFitResult[xgb.XGBRFRegressor]: ...


def fit_xgb_classifier(
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    eval_set: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
    hyperparameter_space: HyperparameterSpace | None = None,
    refit_with_all: bool = False,
) -> HyperparameterFitResult[xgb.XGBClassifier]:
    # space = hyperparameter_space or HyperparameterSpace.default_space_from_model(
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
    hyperparameter_space: HyperparameterSpace | None = None,
    refit_with_all: bool = False,
) -> HyperparameterFitResult[xgb.XGBRFClassifier]: ...


class ProphetSchema(dy.Schema):
    ts = dy.Datetime(nullable=False, unique=True)
    y = dy.Float(nullable=False, allow_inf=False, allow_nan=False)

    @dy.rule()
    def min_data_points(cls) -> pl.Expr:
        return pl.len() >= 100


def fit_prophet(
    df_train: dy.DataFrame[ProphetSchema],
    hyperparameter_space: HyperparameterSpace | None = None,
) -> HyperparameterFitResult[Prophet]:
    space = hyperparameter_space or HyperparameterSpace.default_prophet()

    def objective(trial: rustuna.Trial) -> float:
        params = {key: val.suggest(trial) for key, val in space.items()}
        model = Prophet(**params)  # type: ignore
        model.fit(df_train.to_pandas())

        # TODO: implement
        return 0.0

    rustuna.create_study().optimize(objective, n_trials=50)
    return HyperparameterFitResult(model=Prophet(), metrics=Metrics(mape=0.0))
