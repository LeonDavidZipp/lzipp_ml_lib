from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Generic, Self

import numpy as np
import xgboost as xgb
from prophet import Prophet
from rustuna import Trial

from ._types import M

CategoricalChoiceType = float | int | str | bool | None


class HyperparameterDimension(ABC):
    @abstractmethod
    def suggest(self, trial: Trial) -> CategoricalChoiceType: ...

    @abstractmethod
    def values(self) -> Sequence[CategoricalChoiceType]: ...


class CategoricalDimension(HyperparameterDimension):
    def __init__(self, name: str, choices: list[CategoricalChoiceType]):
        self.name = name
        self.choices = choices

    def suggest(self, trial: Trial) -> CategoricalChoiceType:
        return trial.suggest_categorical(self.name, self.choices)

    def values(self) -> Sequence[CategoricalChoiceType]:
        return self.choices


class IntegerDimension(HyperparameterDimension):
    def __init__(
        self, name: str, low: int, high: int, step: int = 1, log: bool = False
    ):
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def suggest(self, trial: Trial) -> int:
        return trial.suggest_int(
            self.name, self.low, self.high, step=self.step, log=self.log
        )

    def values(self) -> Sequence[int]:
        return list(range(self.low, self.high + 1, self.step))


class FloatDimension(HyperparameterDimension):
    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
        log: bool = False,
    ):
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def suggest(self, trial: Trial) -> float:
        return trial.suggest_float(
            self.name, self.low, self.high, step=self.step, log=self.log
        )

    def values(self) -> Sequence[float]:
        step = self.step if self.step is not None else 1.0
        n = round((self.high - self.low) / step) + 1
        return np.log(np.linspace(self.low, self.high, n)).tolist()


def _fixed(**defaults: CategoricalChoiceType) -> dict[str, HyperparameterDimension]:
    """Pin parameters to one value each, so trials still record them in their params."""
    return {
        name: CategoricalDimension(name, [value]) for name, value in defaults.items()
    }


# Library defaults (xgboost 3.x, prophet 1.x) for every scalar hyperparameter, used
# to pin whatever a search space doesn't tune. Left out: xgboost's `objective` (the
# classifiers pick it from the number of classes), `scale_pos_weight` (binary only),
# `early_stopping_rounds` (handled by the fitting functions) and prophet's
# `changepoints` / `holidays` (not scalar choices).
_XGB_TREE_DEFAULTS: dict[str, CategoricalChoiceType] = {
    "booster": "gbtree",
    "tree_method": "hist",
    "grow_policy": "depthwise",
    "max_depth": 6,
    "max_leaves": 0,
    "max_bin": 256,
    "max_delta_step": 0.0,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "sampling_method": "uniform",
    "colsample_bytree": 1.0,
    "colsample_bylevel": 1.0,
    "max_cat_to_onehot": 4,
    "max_cat_threshold": 64,
    "n_estimators": 100,
}
_XGB_BOOST_DEFAULTS: dict[str, CategoricalChoiceType] = {
    **_XGB_TREE_DEFAULTS,
    "learning_rate": 0.3,
    "reg_lambda": 1.0,
    "subsample": 1.0,
    "colsample_bynode": 1.0,
    "num_parallel_tree": 1,
}
# XGBRF* set num_parallel_tree from n_estimators themselves
_XGB_RF_DEFAULTS: dict[str, CategoricalChoiceType] = {
    **_XGB_TREE_DEFAULTS,
    "learning_rate": 1.0,
    "reg_lambda": 1e-5,
    "subsample": 0.8,
    "colsample_bynode": 0.8,
}
_PROPHET_DEFAULTS: dict[str, CategoricalChoiceType] = {
    "growth": "linear",
    "n_changepoints": 25,
    "changepoint_range": 0.8,
    "yearly_seasonality": "auto",
    "weekly_seasonality": "auto",
    "daily_seasonality": "auto",
    "seasonality_mode": "additive",
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "changepoint_prior_scale": 0.05,
    "mcmc_samples": 0,
    "interval_width": 0.8,
    "uncertainty_samples": 1000,
    "scaling": "absmax",
}


def _model_defaults(
    model_type: type[xgb.XGBModel] | type[Prophet],
) -> dict[str, CategoricalChoiceType]:
    if issubclass(model_type, (xgb.XGBRFRegressor, xgb.XGBRFClassifier)):
        return _XGB_RF_DEFAULTS
    if issubclass(model_type, xgb.XGBModel):
        return _XGB_BOOST_DEFAULTS
    return _PROPHET_DEFAULTS


class HyperparameterSpace(dict[str, HyperparameterDimension], Generic[M]):
    """Hyperparameter space for a model type: parameter names mapped to dimensions."""

    def __init__(
        self,
        model_type: type[M],
        dimensions: Mapping[str, HyperparameterDimension] | None = None,
    ):
        super().__init__(dimensions or {})
        self.model_type = model_type

    def suggest(self, trial: Trial) -> dict[str, int | float | CategoricalChoiceType]:
        return {key: val.suggest(trial) for key, val in self.items()}

    def with_defaults(self) -> Self:
        """
        Return a copy with every hyperparameter of the model type this space doesn't
        cover pinned to its library default.

        Pinned parameters go through `suggest_categorical` like any other, so each
        trial records the model's full configuration, not just the tuned part.
        Dimensions already in the space are kept as they are.
        """
        missing = {
            name: value
            for name, value in _model_defaults(self.model_type).items()
            if name not in self
        }
        return type(self)(self.model_type, {**self, **_fixed(**missing)})

    @classmethod
    def default_space_from_model(cls, model_type: type[M]) -> HyperparameterSpace[M]:
        """
        Dynamically return the correct default search space based on the model class.
        """
        dims: Mapping[str, HyperparameterDimension]
        if issubclass(model_type, xgb.XGBRFRegressor):
            dims = cls.default_xgb_rf_regressor()
        elif issubclass(model_type, xgb.XGBRegressor):
            dims = cls.default_xgb_regressor()
        elif issubclass(model_type, xgb.XGBRFClassifier):
            dims = cls.default_xgb_rf_classifier()
        elif issubclass(model_type, xgb.XGBClassifier):
            dims = cls.default_xgb_classifier()
        elif issubclass(model_type, xgb.XGBRanker):
            dims = cls.default_xgb_ranker()
        elif issubclass(model_type, Prophet):
            dims = cls.default_prophet()
        else:
            raise TypeError(
                "No default hyperparameter space defined for model type: "
                + f"{model_type.__name__}"
            )
        return HyperparameterSpace(model_type, dims)

    @classmethod
    def default_xgb_regressor(cls) -> HyperparameterSpace[xgb.XGBRegressor]:
        """Standard search space for XGBoost regression tasks."""
        return HyperparameterSpace(
            xgb.XGBRegressor,
            {
                "n_estimators": IntegerDimension(
                    "n_estimators", low=100, high=1000, step=50
                ),
                "max_depth": IntegerDimension("max_depth", low=3, high=10),
                "learning_rate": FloatDimension(
                    "learning_rate", low=1e-3, high=0.3, log=True
                ),
                "subsample": FloatDimension("subsample", low=0.5, high=1.0),
                "colsample_bytree": FloatDimension(
                    "colsample_bytree", low=0.5, high=1.0
                ),
                "min_child_weight": IntegerDimension(
                    "min_child_weight", low=1, high=10
                ),
                "gamma": FloatDimension("gamma", low=0.0, high=5.0),
                "reg_alpha": FloatDimension(
                    "reg_alpha", low=1e-8, high=100.0, log=True
                ),
                "reg_lambda": FloatDimension(
                    "reg_lambda", low=1e-8, high=100.0, log=True
                ),
            },
        ).with_defaults()

    @classmethod
    def default_xgb_rf_regressor(cls) -> HyperparameterSpace[xgb.XGBRFRegressor]:
        """Standard search space for XGBoost Random Forest regression tasks."""
        return HyperparameterSpace(
            xgb.XGBRFRegressor,
            {
                "n_estimators": IntegerDimension(
                    "n_estimators", low=100, high=1000, step=50
                ),
                "max_depth": IntegerDimension("max_depth", low=5, high=20),
                "subsample": FloatDimension("subsample", low=0.5, high=0.95),
                "colsample_bynode": FloatDimension(
                    "colsample_bynode", low=0.4, high=0.9
                ),
                "min_child_weight": IntegerDimension(
                    "min_child_weight", low=1, high=10
                ),
            },
        ).with_defaults()

    @classmethod
    def default_xgb_classifier(cls) -> HyperparameterSpace[xgb.XGBClassifier]:
        """Standard search space for XGBoost classification tasks."""
        return HyperparameterSpace(xgb.XGBClassifier, cls.default_xgb_regressor())

    @classmethod
    def default_xgb_rf_classifier(cls) -> HyperparameterSpace[xgb.XGBRFClassifier]:
        """Standard search space for XGBoost Random Forest classification tasks."""
        return HyperparameterSpace(xgb.XGBRFClassifier, cls.default_xgb_rf_regressor())

    @classmethod
    def default_xgb_ranker(cls) -> HyperparameterSpace[xgb.XGBRanker]:
        """Standard search space for XGBoost learning-to-rank tasks."""
        space = HyperparameterSpace(xgb.XGBRanker, cls.default_xgb_regressor())
        space["max_depth"] = IntegerDimension("max_depth", low=2, high=8)
        return space

    @classmethod
    def default_prophet(cls) -> HyperparameterSpace[Prophet]:
        """Standard search space for Meta Prophet."""
        return HyperparameterSpace(
            Prophet,
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
            },
        ).with_defaults()
