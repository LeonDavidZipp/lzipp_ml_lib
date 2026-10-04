import numpy as np
import polars as pl
from hypothesis import strategies as st
from sklearn.datasets import make_classification, make_regression


@st.composite
def regression_lfs(draw: st.DrawFn) -> tuple[pl.LazyFrame, pl.LazyFrame]:
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
    x_lf = pl.LazyFrame(x, schema=[f"feat{i}" for i in range(x.shape[1])])
    y_lf = pl.LazyFrame({"y": y})
    return x_lf, y_lf


@st.composite
def multiclass_lfs(draw: st.DrawFn, min_classes: int = 3) -> pl.LazyFrame:
    n_classes = draw(st.integers(min_classes, 30))
    return _classification_lfs(draw, n_classes)


@st.composite
def binary_lfs(draw: st.DrawFn) -> pl.LazyFrame:
    return _classification_lfs(draw, 2)


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
