import numpy as np
import polars as pl
from hypothesis import strategies as st
from numpy.typing import NDArray
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split  # type: ignore


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
def regression_train_test_lfs(
    draw: st.DrawFn,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Ordering: (x_train, x_test, y_train, y_test)"""
    x_train, x_test, y_train, y_test = draw(regression_train_test())
    return _x_lf(x_train), _x_lf(x_test), _y_lf(y_train), _y_lf(y_test)


@st.composite
def regression_train_val_test_lfs(
    draw: st.DrawFn,
) -> tuple[
    pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame
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


def _x_lf(x: NDArray[np.float64]) -> pl.LazyFrame:
    return pl.LazyFrame(x, schema=[f"feat{i}" for i in range(x.shape[1])])


def _y_lf(y: NDArray[np.float64]) -> pl.LazyFrame:
    return pl.LazyFrame({"y": y})


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
def multiclass_lfs(
    draw: st.DrawFn, min_classes: int = 3
) -> st.SearchStrategy[pl.LazyFrame]:
    n_classes = draw(st.integers(min_classes, 30))
    return _classification_lfs(n_classes)


def binary_lfs() -> st.SearchStrategy[pl.LazyFrame]:
    return _classification_lfs(2)


@st.composite
def _classification_lfs(draw: st.DrawFn, n_classes: int) -> pl.LazyFrame:
    # sklearn needs n_classes <= 2**n_informative
    n_informative = draw(st.integers(max(2, (n_classes - 1).bit_length()), 15))
    n_noise = draw(st.integers(0, 5))
    x, y = make_classification(
        n_samples=draw(st.integers(40, 50)) * n_classes,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=draw(st.floats(0, 0.1)),
        random_state=draw(st.integers(0, 2**32 - 1)),
    )
    return pl.LazyFrame(x, schema=[f"feat{i}" for i in range(x.shape[1])]).with_columns(
        y=pl.Series(y)
    )
