from abc import ABC, abstractmethod
from typing import Self

import xgboost as xgb
from prophet import Prophet
from rustuna import Trial

CategoricalChoiceType = float | int | str | bool | None


class HyperparameterDimension(ABC):
    @abstractmethod
    def suggest(self, trial: Trial) -> CategoricalChoiceType: ...


class CategoricalDimension(HyperparameterDimension):
    def __init__(self, name: str, choices: list[CategoricalChoiceType]):
        self.name = name
        self.choices = choices

    def suggest(self, trial: Trial) -> CategoricalChoiceType:
        return trial.suggest_categorical(self.name, self.choices)


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


class HyperparameterSpace(dict[str, HyperparameterDimension]):
    """Hyperparameter space containing the parameter names and the dimensions"""

    @classmethod
    def default_space_from_model(
        cls, model_type: type[xgb.XGBModel] | type[Prophet]
    ) -> Self:
        """
        Dynamically return the correct default search space based on the model class.
        """
        # The RF models subclass their boosting counterparts, so check them first.
        if issubclass(model_type, xgb.XGBRFRegressor):
            return cls.default_xgb_rf_regressor()
        elif issubclass(model_type, xgb.XGBRegressor):
            return cls.default_xgb_regressor()
        elif issubclass(model_type, xgb.XGBRFClassifier):
            return cls.default_xgb_rf_classifier()
        elif issubclass(model_type, xgb.XGBClassifier):
            return cls.default_xgb_classifier()
        elif issubclass(model_type, xgb.XGBRanker):
            return cls.default_xgb_ranker()
        elif issubclass(model_type, Prophet):
            return cls.default_prophet()
        else:
            raise TypeError(
                "No default hyperparameter space defined for model type: "
                + f"{model_type.__name__}"
            )

    @classmethod
    def default_xgb_regressor(cls) -> Self:
        """Standard search space for XGBoost regression tasks."""
        return cls(
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
            }
        )

    @classmethod
    def default_xgb_rf_regressor(cls) -> Self:
        """Standard search space for XGBoost Random Forest regression tasks."""
        return cls(
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
            }
        )

    @classmethod
    def default_xgb_classifier(cls) -> Self:
        """Standard search space for XGBoost classification tasks."""
        space = cls.default_xgb_regressor()
        return space

    @classmethod
    def default_xgb_rf_classifier(cls) -> Self:
        """Standard search space for XGBoost Random Forest classification tasks."""
        space = cls.default_xgb_rf_regressor()
        return space

    @classmethod
    def default_xgb_ranker(cls) -> Self:
        """Standard search space for XGBoost learning-to-rank tasks."""
        space = cls.default_xgb_regressor()
        space["max_depth"] = IntegerDimension("max_depth", low=2, high=8)
        return space

    @classmethod
    def default_prophet(cls) -> Self:
        """Standard search space for Meta Prophet."""
        return cls(
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
        )
