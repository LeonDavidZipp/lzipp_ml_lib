from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import polars as pl
import rustuna

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


class HyperparameterFitter(Generic[M]):
    def __init__(self, model_type: type[M]):
        self._model_type: type[M] = model_type
        self._model: M | None = None

    def fit(
        self,
        x: pl.DataFrame,
        y: pl.DataFrame,
        hyperparameter_space: HyperparameterSpace | None = None,
        *,
        validation_sets: Sequence[tuple[pl.DataFrame, pl.DataFrame]] | None = None,
        final_fit_on_everything: bool = False,
    ) -> HyperparameterFitResult[M]:
        # if hyperparameter_space is None:
        #     hyperparameter_space = HyperparameterSpace.default_space_from_model(
        #         self._model_type
        #     )
        # _ = hyperparameter_space

        # def objective(trial: rustuna.Trial):
        #     params = {
        #         key: val.suggest(trial) for key, val in hyperparameter_space.items()
        #     }
        #     model = self._model_type(**params)  # type: ignore
        #     model.fit(x, y)

        # trial = rustuna.create_study()
        ...
