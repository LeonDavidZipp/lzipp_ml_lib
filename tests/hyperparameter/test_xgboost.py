from functools import cache

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from hypothesis import given
from sklearn.datasets import make_regression
from sklearn.utils.validation import check_is_fitted  # type: ignore

from lzipp_ml_lib.hyperparameter import (
    CategoricalDimension,
    HyperparameterSpace,
    IntegerDimension,
    fit_xgb_regressor,
)
from lzipp_ml_lib.hyperparameter._types import (
    FinalFitData,
    RegressionEvalMetric,
)
from tests.hyperparameter.composites import (
    FIT_SETTINGS,
    regression_train_test_dfs,
    regression_train_val_test_dfs,
)


@cache
def _regression_split() -> tuple[pl.DataFrame, ...]:
    """Small fixed dataset for deterministic behaviour tests.

    Ordering: (x_train, x_val, x_test, y_train, y_val, y_test)
    """
    x, y = make_regression(  # type: ignore
        n_samples=400, n_features=5, n_informative=5, noise=5.0, random_state=0
    )
    y = y - y.min() + y.std()  # positive, so MAPE stays meaningful
    xs = pl.DataFrame(x, schema=[f"feat{i}" for i in range(x.shape[1])])
    ys = pl.DataFrame({"y": y})
    return xs[:240], xs[240:320], xs[320:], ys[:240], ys[240:320], ys[320:]


def _fixed_space(
    **values: int | float | str | None,
) -> HyperparameterSpace[xgb.XGBRegressor]:
    """A space with every given parameter pinned to one value."""
    return HyperparameterSpace(
        xgb.XGBRegressor,
        {name: CategoricalDimension(name, [value]) for name, value in values.items()},
    )


FAST_SPACE = _fixed_space(
    n_estimators=30, max_depth=3, learning_rate=0.3, early_stopping_rounds=None
)

# ------------------------------------------------------------------------------------ #
#                              fit_xgb_regressor                                       #
# ------------------------------------------------------------------------------------ #


@FIT_SETTINGS
@given(
    x_y_dfs=regression_train_test_dfs(),
)
def test_fit_xgb_regressor_train_test(
    x_y_dfs: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
):
    x_train, x_test, y_train, y_test = x_y_dfs
    result = fit_xgb_regressor(x_train, y_train, x_test, y_test)
    model = result.model
    check_is_fitted(model)
    expected_space = HyperparameterSpace.default_xgb_regressor().value_spaces()
    for key, val in expected_space.items():
        if key == "n_estimators":
            assert 1 <= model.n_estimators <= max(val)  # type: ignore
            continue
        assert getattr(model, key) in val


@FIT_SETTINGS
@given(
    x_y_dfs=regression_train_val_test_dfs(),
)
def test_fit_xgb_regressor_train_val_test(
    x_y_dfs: tuple[
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
    ],
):
    x_train, x_val, x_test, y_train, y_val, y_test = x_y_dfs
    result = fit_xgb_regressor(
        x_train, y_train, x_test, y_test, eval_set=((x_val, y_val),)
    )
    model = result.model
    check_is_fitted(model)
    expected_space = HyperparameterSpace.default_xgb_regressor().value_spaces()
    for key, val in expected_space.items():
        if key == "n_estimators":
            assert 1 <= model.n_estimators <= max(val)  # type: ignore
            continue
        assert getattr(model, key) in val


@FIT_SETTINGS
@given(
    x_y_dfs=regression_train_test_dfs(),
)
def test_fit_xgb_regressor_train_test_custom_parameter_space(
    x_y_dfs: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
):
    x_train, x_test, y_train, y_test = x_y_dfs
    expected_space = HyperparameterSpace.default_xgb_regressor()
    result = fit_xgb_regressor(
        x_train, y_train, x_test, y_test, search_space=expected_space
    )
    model = result.model
    check_is_fitted(model)
    expected_values = expected_space.value_spaces()
    for key, val in expected_values.items():
        if key == "n_estimators":
            assert 1 <= model.n_estimators <= max(val)  # type: ignore
            continue
        assert getattr(model, key) in val


@FIT_SETTINGS
@given(
    x_y_dfs=regression_train_val_test_dfs(),
)
def test_fit_xgb_regressor_train_val_test_custom_parameter_space(
    x_y_dfs: tuple[
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
    ],
):
    x_train, x_val, x_test, y_train, y_val, y_test = x_y_dfs
    expected_space = HyperparameterSpace.default_xgb_regressor()
    result = fit_xgb_regressor(
        x_train,
        y_train,
        x_test,
        y_test,
        eval_set=((x_val, y_val),),
        search_space=expected_space,
    )
    model = result.model
    check_is_fitted(model)
    expected_values = expected_space.value_spaces()
    for key, val in expected_values.items():
        if key == "n_estimators":
            assert 1 <= model.n_estimators <= max(val)  # type: ignore
            continue
        assert getattr(model, key) in val


@FIT_SETTINGS
@given(x_y_dfs=regression_train_test_dfs())
def test_fit_xgb_regressor_pins_untuned_parameters_to_defaults(
    x_y_dfs: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
):
    x_train, x_test, y_train, y_test = x_y_dfs
    space = HyperparameterSpace(
        xgb.XGBRegressor, {"max_depth": IntegerDimension("max_depth", low=2, high=4)}
    )
    result = fit_xgb_regressor(
        x_train, y_train, x_test, y_test, search_space=space, n_trials=2
    )
    params = result.model.get_params()
    assert 2 <= params["max_depth"] <= 4
    for name, dim in space.with_defaults().items():
        if name != "max_depth":
            assert params[name] == dim.values()[0], name
    # the caller's space is left as it was
    assert list(space) == ["max_depth"]


@pytest.mark.parametrize("final_fit_data", ["train", "train_val", "train_val_test"])
def test_fit_xgb_regressor_final_model_fit_on_final_fit_data(
    final_fit_data: FinalFitData,
):
    x_train, x_val, x_test, y_train, y_val, y_test = _regression_split()
    result = fit_xgb_regressor(
        x_train,
        y_train,
        x_test,
        y_test,
        eval_set=[(x_val, y_val)],
        search_space=FAST_SPACE,
        n_trials=1,
        final_fit_data=final_fit_data,
    )
    parts = {
        "train": [(x_train, y_train)],
        "train_val": [(x_train, y_train), (x_val, y_val)],
        "train_val_test": [(x_train, y_train), (x_val, y_val), (x_test, y_test)],
    }[final_fit_data]
    params = {name: dim.values()[0] for name, dim in FAST_SPACE.with_defaults().items()}
    expected = xgb.XGBRegressor(**params).fit(
        pl.concat(x for x, _ in parts), pl.concat(y for _, y in parts)
    )
    np.testing.assert_allclose(
        result.model.predict(x_test), expected.predict(x_test), rtol=1e-6
    )


@pytest.mark.parametrize("final_fit_data", ["train_val", "train_val_test"])
def test_fit_xgb_regressor_final_fit_data_without_eval_set_raises(
    final_fit_data: FinalFitData,
):
    x_train, _, x_test, y_train, _, y_test = _regression_split()
    with pytest.raises(ValueError, match="needs an eval_set"):
        fit_xgb_regressor(
            x_train,
            y_train,
            x_test,
            y_test,
            search_space=FAST_SPACE,
            n_trials=1,
            final_fit_data=final_fit_data,
        )


def test_fit_xgb_regressor_early_stopping_shortens_final_model():
    x_train, x_val, x_test, y_train, y_val, y_test = _regression_split()
    result = fit_xgb_regressor(
        x_train,
        y_train,
        x_test,
        y_test,
        eval_set=[(x_val, y_val)],
        search_space=_fixed_space(n_estimators=1000, max_depth=3, learning_rate=0.5),
        early_stopping_rounds=3,
        n_trials=1,
    )
    # trained for the rounds the best trial used, without stopping on its own
    n_estimators = result.model.n_estimators
    assert n_estimators is not None and n_estimators < 1000
    assert result.model.early_stopping_rounds is None


@pytest.mark.parametrize("metric", ["mape", "mae", "rmse", "mse", "r2"])
def test_fit_xgb_regressor_optimizes_metric_in_right_direction(
    metric: RegressionEvalMetric,
):
    """A single tree is clearly worse than 200; a flipped direction would pick it."""
    x_train, _, x_test, y_train, _, y_test = _regression_split()
    space = HyperparameterSpace(
        xgb.XGBRegressor,
        {
            "n_estimators": CategoricalDimension("n_estimators", [1, 200]),
            "learning_rate": CategoricalDimension("learning_rate", [0.3]),
        },
    )
    # enough trials that both choices are sampled with near certainty
    result = fit_xgb_regressor(
        x_train,
        y_train,
        x_test,
        y_test,
        search_space=space,
        n_trials=20,
        metric=metric,
    )
    assert result.model.n_estimators == 200
    assert result.metrics.r2 > 0.5
