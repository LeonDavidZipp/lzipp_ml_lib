import polars as pl
from hypothesis import given
from sklearn.utils.validation import check_is_fitted  # type: ignore

from lzipp_ml_lib.hyperparameter import HyperparameterSpace, fit_xgb_regressor
from tests.hyperparameter.composites import (
    FIT_SETTINGS,
    regression_train_test_dfs,
    regression_train_val_test_dfs,
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
        assert getattr(model, key) in val
