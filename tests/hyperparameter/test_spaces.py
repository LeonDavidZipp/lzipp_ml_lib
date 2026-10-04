import inspect
from collections.abc import Callable
from typing import Any

import pytest
import rustuna
import xgboost as xgb
from prophet import Prophet

from lzipp_ml_lib.hyperparameter import (
    CategoricalDimension,
    FloatDimension,
    HyperparameterSpace,
    IntegerDimension,
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
    (HyperparameterSpace.default_xgb_rf_regressor, xgb.XGBRFRegressor),
    (HyperparameterSpace.default_xgb_classifier, xgb.XGBClassifier),
    (HyperparameterSpace.default_xgb_rf_classifier, xgb.XGBRFClassifier),
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


def test_categorical_dimension_suggests_one_of_its_choices():
    dim = CategoricalDimension("c", ["a", "b", None])
    suggested, recorded = _recorded_params(
        HyperparameterSpace(xgb.XGBRegressor, {"c": dim})
    )
    assert suggested["c"] in dim.choices
    assert recorded == suggested
    assert dim.values() == ["a", "b", None]


@pytest.mark.parametrize(("low", "high", "step"), [(1, 10, 1), (100, 1000, 50)])
def test_integer_dimension_suggests_on_grid(low: int, high: int, step: int):
    dim = IntegerDimension("i", low=low, high=high, step=step)
    for _ in range(20):
        suggested, _ = _recorded_params(
            HyperparameterSpace(xgb.XGBRegressor, {"i": dim})
        )
        assert suggested["i"] in dim.values()


@pytest.mark.parametrize(("low", "high", "step"), [(1, 10, 1), (100, 1000, 50)])
def test_integer_dimension_values_include_both_bounds(low: int, high: int, step: int):
    values = IntegerDimension("i", low=low, high=high, step=step).values()
    assert values[0] == low
    assert values[-1] == high
    assert all(b - a == step for a, b in zip(values, values[1:], strict=False))


def test_float_dimension_suggests_within_bounds():
    dim = FloatDimension("f", low=1e-3, high=0.3, log=True)
    for _ in range(20):
        suggested, _ = _recorded_params(
            HyperparameterSpace(xgb.XGBRegressor, {"f": dim})
        )
        assert 1e-3 <= suggested["f"] <= 0.3


@pytest.mark.xfail(
    strict=True, reason="FloatDimension.values() returns np.log of the grid points"
)
def test_float_dimension_values_include_both_bounds():
    values = FloatDimension("f", low=0.1, high=0.5, step=0.1).values()
    assert values == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


# ------------------------------------------------------------------------------------ #
#                                 HyperparameterSpace                                  #
# ------------------------------------------------------------------------------------ #


def test_space_suggests_and_records_every_dimension():
    space = HyperparameterSpace.default_xgb_regressor()
    suggested, recorded = _recorded_params(space)
    assert suggested.keys() == space.keys()
    assert recorded == suggested


def test_value_spaces_cover_every_dimension():
    space = HyperparameterSpace.default_xgb_regressor()
    value_spaces = space.value_spaces()
    assert value_spaces.keys() == space.keys()
    for name, dim in space.items():
        assert value_spaces[name] == dim.values()


def test_with_defaults_fills_missing_and_keeps_given_dimensions():
    max_depth = IntegerDimension("max_depth", low=2, high=4)
    space = HyperparameterSpace(xgb.XGBRegressor, {"max_depth": max_depth})
    full = space.with_defaults()

    assert full is not space
    assert list(space) == ["max_depth"]  # original untouched
    assert full.model_type is xgb.XGBRegressor
    assert full["max_depth"] is max_depth
    pinned = _pinned(full)
    assert "max_depth" not in pinned
    assert pinned["learning_rate"] == 0.3
    assert pinned["n_estimators"] == 100


def test_with_defaults_is_idempotent():
    once = HyperparameterSpace(xgb.XGBRegressor).with_defaults()
    twice = once.with_defaults()
    assert twice.keys() == once.keys()
    assert _pinned(twice) == _pinned(once)


def test_with_defaults_uses_rf_defaults_for_rf_models():
    boost = _pinned(HyperparameterSpace(xgb.XGBRegressor).with_defaults())
    rf = _pinned(HyperparameterSpace(xgb.XGBRFRegressor).with_defaults())
    assert boost["learning_rate"] == 0.3
    assert rf["learning_rate"] == 1.0
    assert rf["reg_lambda"] == 1e-5
    # XGBRF* derive num_parallel_tree from n_estimators themselves
    assert "num_parallel_tree" in boost
    assert "num_parallel_tree" not in rf


def test_with_defaults_leaves_out_early_stopping_rounds():
    # the fitting functions treat early_stopping_rounds in the space specially
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
    assert (
        HyperparameterSpace.default_xgb_rf_classifier().keys()
        == HyperparameterSpace.default_xgb_rf_regressor().keys()
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
