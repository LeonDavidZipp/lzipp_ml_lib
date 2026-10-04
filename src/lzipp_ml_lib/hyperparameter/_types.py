from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar

import numpy as np
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

M = TypeVar("M", bound=xgb.XGBModel | Prophet)
RegressionEvalMetric = Literal["mape", "mae", "rmse", "mse", "r2"]
ClassificationEvalMetric = Literal[
    "accuracy", "precision", "recall", "f1_score", "roc_auc", "log_loss"
]
EvalMetric = RegressionEvalMetric | ClassificationEvalMetric
FinalFitData = Literal["train", "train_val", "train_val_test"]


METRICS: dict[EvalMetric, Any] = {
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

DIRECTIONS: dict[EvalMetric, Any] = {
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

    @classmethod
    def calculate(
        cls, y_true: ArrayLike, y_pred: ArrayLike, y_proba: ArrayLike
    ) -> Self:
        y_true = np.ravel(np.asarray(y_true))
        y_proba = np.asarray(y_proba)
        labels = np.arange(int(y_proba.shape[1]))
        binary = len(labels) == 2
        prf_kwargs: dict[str, Any] = {
            "labels": labels,
            "average": "binary" if binary else "macro",
            # scores classes that are never predicted as 0 instead of warning
            "zero_division": 0,
        }
        roc_auc = (
            roc_auc_score(y_true, y_proba[:, 1])
            if binary
            else roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels)
        )
        return cls(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, **prf_kwargs)),
            recall=float(recall_score(y_true, y_pred, **prf_kwargs)),
            f1_score=float(f1_score(y_true, y_pred, **prf_kwargs)),
            roc_auc=float(roc_auc),
            log_loss=float(log_loss(y_true, y_proba, labels=labels)),
        )


@dataclass
class RegressionFitResult(Generic[M]):
    model: M
    metrics: RegressionMetrics


@dataclass
class ClassificationFitResult(Generic[M]):
    model: M
    metrics: ClassificationMetrics
