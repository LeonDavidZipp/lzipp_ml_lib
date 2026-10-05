from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from prophet import Prophet
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split  # type: ignore

from lzipp_ml_lib.hyperparameter import (
    CategoricalDimension,
    FloatDimension,
    HyperparameterSpace,
    IntegerDimension,
)

FIT_SETTINGS = settings(
    max_examples=1, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@st.composite
def regression_train_test(
    draw: st.DrawFn,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Ordering: (x_train, x_test, y_train, y_test)"""
    x, y = draw(_regression_data())
    seed = draw(st.integers(0, 2**32 - 1))
    x_train, x_test, y_train, y_test = train_test_split(  # type: ignore
        x, y, test_size=0.2, random_state=seed
    )
    return x_train, x_test, y_train, y_test  # type: ignore


@st.composite
def regression_train_test_dfs(
    draw: st.DrawFn,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Ordering: (x_train, x_test, y_train, y_test)"""
    x_train, x_test, y_train, y_test = draw(regression_train_test())
    return _x_lf(x_train), _x_lf(x_test), _y_lf(y_train), _y_lf(y_test)


@st.composite
def regression_train_val_test_dfs(
    draw: st.DrawFn,
) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame
]:
    """Ordering: (x_train, x_val, x_test, y_train, y_val, y_test)"""
    x_train, x_test, y_train, y_test = draw(regression_train_test())
    seed = draw(st.integers(0, 2**32 - 1))
    x_train, x_val, y_train, y_val = train_test_split(  # type: ignore
        x_train, y_train, test_size=0.2, random_state=seed
    )
    return (
        _x_lf(x_train),  # type: ignore
        _x_lf(x_val),  # type: ignore
        _x_lf(x_test),
        _y_lf(y_train),  # type: ignore
        _y_lf(y_val),  # type: ignore
        _y_lf(y_test),
    )


@st.composite
def _regression_data(
    draw: st.DrawFn,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_informative = draw(st.integers(1, 15))
    n_noise = draw(st.integers(0, 5))
    seed = draw(st.integers(0, 2**32 - 1))
    x, y = make_regression(  # type: ignore
        n_samples=draw(st.integers(100, 5000)),
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        random_state=seed,
    )
    # Noise relative to the signal's spread, so the linear trend always dominates
    # (R^2 of the true model stays above ~0.9).
    noise_ratio = draw(st.floats(0, 0.3))
    y += np.random.default_rng(seed).normal(0, noise_ratio * y.std(), len(y))
    # Keep y positive and away from 0, so MAPE stays meaningful.
    y = y - y.min() + y.std()
    return x, y


@st.composite
def classification_train_test(
    draw: st.DrawFn, min_classes: int = 2, max_classes: int = 30
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]
]:
    """Ordering: (x_train, x_test, y_train, y_test)"""
    x, y = draw(_classification_data(min_classes, max_classes))
    seed = draw(st.integers(0, 2**32 - 1))
    x_train, x_test, y_train, y_test = train_test_split(  # type: ignore
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    return x_train, x_test, y_train, y_test  # type: ignore


@st.composite
def classification_train_test_dfs(
    draw: st.DrawFn, min_classes: int = 2, max_classes: int = 30
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Ordering: (x_train, x_test, y_train, y_test)"""
    x_train, x_test, y_train, y_test = draw(
        classification_train_test(min_classes, max_classes)
    )
    return _x_lf(x_train), _x_lf(x_test), _y_lf(y_train), _y_lf(y_test)


@st.composite
def classification_train_val_test_dfs(
    draw: st.DrawFn, min_classes: int = 2, max_classes: int = 30
) -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame
]:
    """Ordering: (x_train, x_val, x_test, y_train, y_val, y_test)"""
    x_train, x_test, y_train, y_test = draw(
        classification_train_test(min_classes, max_classes)
    )
    seed = draw(st.integers(0, 2**32 - 1))
    x_train, x_val, y_train, y_val = train_test_split(  # type: ignore
        x_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    return (
        _x_lf(x_train),  # type: ignore
        _x_lf(x_val),  # type: ignore
        _x_lf(x_test),
        _y_lf(y_train),  # type: ignore
        _y_lf(y_val),  # type: ignore
        _y_lf(y_test),
    )


@st.composite
def _classification_data(
    draw: st.DrawFn, min_classes: int, max_classes: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    n_classes = draw(st.integers(min_classes, max_classes))
    # sklearn needs n_classes <= 2**n_informative
    n_informative = draw(st.integers(max(2, (n_classes - 1).bit_length()), 15))
    n_noise = draw(st.integers(0, 5))
    x, y = make_classification(  # type: ignore
        n_samples=draw(st.integers(40, 50)) * n_classes,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=draw(st.floats(0, 0.1)),
        random_state=draw(st.integers(0, 2**32 - 1)),
    )
    return x, y  # type: ignore


def _x_lf(x: NDArray[np.float64]) -> pl.DataFrame:
    return pl.DataFrame(x, schema=[f"feat{i}" for i in range(x.shape[1])])


def _y_lf(y: NDArray[np.float64] | NDArray[np.int64]) -> pl.DataFrame:
    return pl.DataFrame({"y": y})


_MODEL_TYPES: list[
    type[xgb.XGBRegressor]
    | type[xgb.XGBClassifier]
    | type[xgb.XGBRanker]
    | type[Prophet]
] = [
    xgb.XGBRegressor,
    xgb.XGBClassifier,
    xgb.XGBRanker,
    Prophet,
]
_CHOICES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(2**63), 2**63 - 1),  # rustuna stores ints as i64
    st.floats(allow_nan=False),
    st.text(),
)


@st.composite
def categorical_dimensions(draw: st.DrawFn) -> CategoricalDimension:
    return CategoricalDimension("c", draw(st.lists(_CHOICES, min_size=1, max_size=10)))


@st.composite
def integer_dimensions(draw: st.DrawFn) -> IntegerDimension:
    """`high` is not necessarily on the `step` grid from `low`."""
    low = draw(st.integers(-1000, 1000))
    step = draw(st.integers(1, 100))
    high = draw(st.integers(low, low + 50 * step))
    return IntegerDimension("i", low=low, high=high, step=step)


@st.composite
def stepped_float_dimensions(draw: st.DrawFn) -> FloatDimension:
    """`high` lies on the `step` grid from `low`."""
    low = draw(st.floats(-1e3, 1e3))
    step = draw(st.floats(1e-3, 10))
    high = low + draw(st.integers(0, 50)) * step
    return FloatDimension("f", low=low, high=high, step=step)


@st.composite
def log_float_dimensions(draw: st.DrawFn) -> FloatDimension:
    low = draw(st.floats(1e-8, 1e3))
    # rustuna panics ("cannot sample empty range") when `high` is the float right
    # after `low`: in log space the rounded bounds can cross
    high = draw(st.one_of(st.just(low), st.floats(low * (1 + 1e-9), 1e4)))
    return FloatDimension("f", low=low, high=high, log=True)


@st.composite
def linear_float_dimensions(draw: st.DrawFn) -> FloatDimension:
    low = draw(st.floats(-1e3, 1e3))
    high = draw(st.floats(low, low + 1e3))
    n_points = draw(st.integers(2, 50))
    return FloatDimension("f", low=low, high=high, n_points=n_points)


@st.composite
def partial_default_spaces(draw: st.DrawFn) -> HyperparameterSpace[Any]:
    """A model type's default space with only some of its tuned dimensions kept."""
    model_type = draw(st.sampled_from(_MODEL_TYPES))
    default = HyperparameterSpace.default_space_from_model(model_type)
    tuned = sorted(
        name
        for name, dim in default.items()
        if not (isinstance(dim, CategoricalDimension) and len(dim.choices) == 1)
    )
    kept = draw(st.sets(st.sampled_from(tuned))) if tuned else set[str]()
    return HyperparameterSpace(
        model_type, {name: default[name] for name in sorted(kept)}
    )
