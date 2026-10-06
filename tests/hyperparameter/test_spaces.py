import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import rustuna
import xgboost as xgb
from hypothesis import given
from hypothesis import strategies as st
from prophet import Prophet

from lzipp_ml_lib.hyperparameter import (
    CategoricalDimension,
    FloatDimension,
    HyperparameterSpace,
    IntegerDimension,
)
from lzipp_ml_lib.hyperparameter._space import HyperparameterDimension
from tests.hyperparameter.composites import (
    categorical_dimensions,
    integer_dimensions,
    linear_float_dimensions,
    log_float_dimensions,
    partial_default_spaces,
    stepped_float_dimensions,
)


def _recorded_params(
    space: HyperparameterSpace[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Suggest from `space` in a fresh trial; return (suggested, recorded) params."""
    study = rustuna.create_study()
    trial = study.ask()
    suggested = space.suggest(trial)
    study.tell(trial.number, 0.0)
    return suggested, dict(study.trials[0].params)


def _pinned(space: HyperparameterSpace[Any]) -> dict[str, Any]:
    """The single-choice categorical dimensions of `space`, mapped to their value."""
    return {
        name: dim.choices[0]
        for name, dim in space.items()
        if isinstance(dim, CategoricalDimension) and len(dim.choices) == 1
    }


DEFAULT_SPACES: list[tuple[Callable[[], HyperparameterSpace[Any]], type[Any]]] = [
    (HyperparameterSpace.default_xgb_regressor, xgb.XGBRegressor),
    (HyperparameterSpace.default_xgb_classifier, xgb.XGBClassifier),
    (HyperparameterSpace.default_xgb_ranker, xgb.XGBRanker),
    (HyperparameterSpace.default_prophet, Prophet),
]
DEFAULT_SPACE_IDS = [model_type.__name__ for _, model_type in DEFAULT_SPACES]


def _constructor_params(model_type: type[Any]) -> set[str]:
    if issubclass(model_type, xgb.XGBModel):
        # the sklearn wrappers take most parameters via **kwargs
        return set(model_type().get_params())
    return set(inspect.signature(model_type.__init__).parameters) - {"self"}


# ------------------------------------------------------------------------------------ #
#                                    dimensions                                        #
# ------------------------------------------------------------------------------------ #


@given(categorical_dimensions())
def test_categorical_dimension_suggests_and_records_one_of_its_choices(
    dim: CategoricalDimension,
):
    suggested, recorded = _recorded_params(
        HyperparameterSpace(xgb.XGBRegressor, {"c": dim})
    )
    assert suggested["c"] in dim.choices
    assert recorded == suggested
    assert dim.values() == dim.choices


@given(integer_dimensions())
def test_integer_dimension_suggests_on_grid(dim: IntegerDimension):
    suggested, _ = _recorded_params(HyperparameterSpace(xgb.XGBRegressor, {"i": dim}))
    assert suggested["i"] in dim.values()


@given(integer_dimensions())
def test_integer_dimension_values_step_from_low_to_last_grid_point(
    dim: IntegerDimension,
):
    values = dim.values()
    assert values[0] == dim.low
    assert values[-1] == dim.high - (dim.high - dim.low) % dim.step
    assert all(b - a == dim.step for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize(("low", "high", "step"), [(1, 10, 1), (100, 1000, 50)])
def test_integer_dimension_values_include_both_bounds(low: int, high: int, step: int):
    values = IntegerDimension("i", low=low, high=high, step=step).values()
    assert values[0] == low
    assert values[-1] == high


@given(log_float_dimensions())
def test_float_dimension_suggests_within_bounds(dim: FloatDimension):
    suggested, _ = _recorded_params(HyperparameterSpace(xgb.XGBRegressor, {"f": dim}))
    assert dim.low <= suggested["f"] <= dim.high


@given(stepped_float_dimensions())
def test_float_dimension_suggests_on_grid(dim: FloatDimension):
    suggested, _ = _recorded_params(HyperparameterSpace(xgb.XGBRegressor, {"f": dim}))
    assert suggested["f"] == pytest.approx(
        min(dim.values(), key=lambda v: abs(v - suggested["f"]))
    )


@given(stepped_float_dimensions())
def test_float_dimension_values_step_from_low_to_high(dim: FloatDimension):
    assert dim.step is not None
    values = dim.values()
    assert values[0] == pytest.approx(dim.low)
    assert values[-1] == pytest.approx(dim.high)
    assert np.diff(values) == pytest.approx(dim.step)


@given(stepped_float_dimensions().filter(lambda d: d.low > 0))
def test_float_dimension_log_values_are_log_of_linear_values(dim: FloatDimension):
    linear = dim.values()
    dim.log = True
    assert dim.values() == pytest.approx(np.log(linear).tolist())


@given(linear_float_dimensions())
def test_float_dimension_without_step_spreads_n_points_evenly(dim: FloatDimension):
    values = dim.values()
    assert len(values) == dim.n_points
    assert values[0] == pytest.approx(dim.low)
    assert values[-1] == pytest.approx(dim.high)
    assert np.diff(values) == pytest.approx((dim.high - dim.low) / (dim.n_points - 1))


@given(log_float_dimensions())
def test_log_float_dimension_without_step_spreads_n_points_geometrically(
    dim: FloatDimension,
):
    values = dim.values()  # natural logs of the points
    assert len(values) == dim.n_points
    assert values[0] == pytest.approx(np.log(dim.low))
    assert values[-1] == pytest.approx(np.log(dim.high))
    assert np.diff(values) == pytest.approx(
        np.log(dim.high / dim.low) / (dim.n_points - 1), abs=1e-9
    )


@given(
    st.one_of(
        categorical_dimensions(),
        integer_dimensions(),
        stepped_float_dimensions(),
        linear_float_dimensions(),
        log_float_dimensions(),
    )
)
def test_dimension_contains_its_suggestions(dim: HyperparameterDimension):
    suggested, _ = _recorded_params(HyperparameterSpace(xgb.XGBRegressor, {"d": dim}))
    assert dim.contains(suggested["d"])


@given(st.one_of(categorical_dimensions(), integer_dimensions()))
def test_discrete_dimension_contains_exactly_its_values(
    dim: CategoricalDimension | IntegerDimension,
):
    assert all(dim.contains(v) for v in dim.values())
    if isinstance(dim, IntegerDimension):
        assert not dim.contains(dim.low - 1)
        assert not dim.contains(dim.high + 1)
        if dim.step > 1:
            assert not dim.contains(dim.low + 1)


@given(stepped_float_dimensions())
def test_stepped_float_dimension_contains_grid_points_only(dim: FloatDimension):
    assert dim.step is not None
    assert all(dim.contains(v) for v in dim.values())
    if dim.high > dim.low:
        assert not dim.contains(dim.low + dim.step / 2)
    assert not dim.contains(dim.high + dim.step)


@given(st.one_of(integer_dimensions(), linear_float_dimensions()))
def test_numeric_dimension_rejects_non_numbers(
    dim: IntegerDimension | FloatDimension,
):
    assert not dim.contains(True)
    assert not dim.contains(None)
    assert not dim.contains(str(dim.low))


def test_float_dimension_values_include_both_bounds():
    values = FloatDimension("f", low=0.1, high=0.5, step=0.1).values()
    assert values == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


def test_float_dimension_log_values_include_both_bounds():
    values = FloatDimension("f", low=0.1, high=0.5, step=0.1, log=True).values()
    expected_values = np.log(np.array([0.1, 0.2, 0.3, 0.4, 0.5])).tolist()
    assert values == pytest.approx(expected_values)


# ------------------------------------------------------------------------------------ #
#                                 HyperparameterSpace                                  #
# ------------------------------------------------------------------------------------ #


@given(partial_default_spaces())
def test_space_suggests_and_records_every_dimension(space: HyperparameterSpace[Any]):
    suggested, recorded = _recorded_params(space)
    assert suggested.keys() == space.keys()
    assert recorded == suggested


@given(partial_default_spaces())
def test_value_spaces_cover_every_dimension(space: HyperparameterSpace[Any]):
    value_spaces = space.value_spaces()
    assert value_spaces.keys() == space.keys()
    for name, dim in space.items():
        assert value_spaces[name] == dim.values()


@given(partial_default_spaces())
def test_with_defaults_fills_missing_and_keeps_given_dimensions(
    space: HyperparameterSpace[Any],
):
    given_dims = dict(space)
    full = space.with_defaults()

    assert full is not space
    assert dict(space) == given_dims  # original untouched
    assert full.model_type is space.model_type
    assert set(full) == set(space) | set(
        HyperparameterSpace(space.model_type).with_defaults()
    )
    for name, dim in given_dims.items():
        assert full[name] is dim


def test_with_defaults_pins_library_defaults():
    max_depth = IntegerDimension("max_depth", low=2, high=4)
    full = HyperparameterSpace(
        xgb.XGBRegressor, {"max_depth": max_depth}
    ).with_defaults()
    pinned = _pinned(full)
    assert "max_depth" not in pinned
    assert pinned["learning_rate"] == 0.3
    assert pinned["n_estimators"] == 100


@given(partial_default_spaces())
def test_with_defaults_is_idempotent(space: HyperparameterSpace[Any]):
    once = space.with_defaults()
    twice = once.with_defaults()
    assert twice.keys() == once.keys()
    assert _pinned(twice) == _pinned(once)


def test_with_defaults_leaves_out_early_stopping_rounds():
    full = HyperparameterSpace(xgb.XGBRegressor).with_defaults()
    assert "early_stopping_rounds" not in full


# ------------------------------------------------------------------------------------ #
#                                   default spaces                                     #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("make_space", "model_type"), DEFAULT_SPACES, ids=DEFAULT_SPACE_IDS
)
def test_default_space_is_bound_to_its_model_type(
    make_space: Callable[[], HyperparameterSpace[Any]], model_type: type[Any]
):
    assert make_space().model_type is model_type


@pytest.mark.parametrize(
    ("make_space", "model_type"), DEFAULT_SPACES, ids=DEFAULT_SPACE_IDS
)
def test_default_space_is_complete(
    make_space: Callable[[], HyperparameterSpace[Any]], model_type: type[Any]
):
    space = make_space()
    assert space.with_defaults().keys() == space.keys()


@pytest.mark.parametrize(
    ("make_space", "model_type"), DEFAULT_SPACES, ids=DEFAULT_SPACE_IDS
)
def test_default_space_only_has_constructor_parameters(
    make_space: Callable[[], HyperparameterSpace[Any]], model_type: type[Any]
):
    assert set(make_space()) <= _constructor_params(model_type)


@pytest.mark.parametrize(
    ("make_space", "model_type"), DEFAULT_SPACES, ids=DEFAULT_SPACE_IDS
)
def test_default_space_suggestions_construct_the_model(
    make_space: Callable[[], HyperparameterSpace[Any]], model_type: type[Any]
):
    suggested, _ = _recorded_params(make_space())
    model = model_type(**suggested)
    if isinstance(model, xgb.XGBModel):
        params = model.get_params()
        for name, value in suggested.items():
            assert params[name] == value, name


@pytest.mark.parametrize(
    ("make_space", "model_type"), DEFAULT_SPACES, ids=DEFAULT_SPACE_IDS
)
def test_pinned_values_match_explicit_library_defaults(
    make_space: Callable[[], HyperparameterSpace[Any]], model_type: type[Any]
):
    """Where the library states a default itself, the pinned value must match it.

    The xgboost wrappers leave most parameters at None (resolved natively), so only
    the ones they set explicitly, like the RF learning rate, can be checked here.
    """
    if issubclass(model_type, xgb.XGBModel):
        explicit = {k: v for k, v in model_type().get_params().items() if v is not None}
    else:
        explicit = {
            name: param.default
            for name, param in inspect.signature(model_type.__init__).parameters.items()
            if param.default is not inspect.Parameter.empty
        }
    for name, value in _pinned(make_space()).items():
        if name in explicit:
            assert value == explicit[name], name


def test_default_ranker_space_narrows_max_depth():
    max_depth = HyperparameterSpace.default_xgb_ranker()["max_depth"]
    assert isinstance(max_depth, IntegerDimension)
    assert (max_depth.low, max_depth.high) == (2, 8)


def test_default_classifier_spaces_match_regressor_spaces():
    assert (
        HyperparameterSpace.default_xgb_classifier().keys()
        == HyperparameterSpace.default_xgb_regressor().keys()
    )


# ------------------------------------------------------------------------------------ #
#                               default_space_from_model                               #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("make_space", "model_type"), DEFAULT_SPACES, ids=DEFAULT_SPACE_IDS
)
def test_default_space_from_model_matches_dedicated_constructor(
    make_space: Callable[[], HyperparameterSpace[Any]], model_type: type[Any]
):
    space = HyperparameterSpace.default_space_from_model(model_type)
    assert space.model_type is model_type
    assert space.keys() == make_space().keys()


def test_default_space_from_model_keeps_subclass():
    class MyRegressor(xgb.XGBRegressor):
        pass

    space: HyperparameterSpace[MyRegressor] = (
        HyperparameterSpace.default_space_from_model(MyRegressor)
    )
    assert space.model_type is MyRegressor
    assert space.keys() == HyperparameterSpace.default_xgb_regressor().keys()


def test_default_space_from_model_rejects_unknown_type():
    with pytest.raises(TypeError, match="No default hyperparameter space"):
        HyperparameterSpace.default_space_from_model(int)  # type: ignore
