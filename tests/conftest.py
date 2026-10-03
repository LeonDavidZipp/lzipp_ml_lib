import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import strategies as st
from numpy.random import Generator
from sklearn.datasets import make_classification

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"
SEED = int(os.getenv("SEED", "42"))
_rng = np.random.default_rng(seed=SEED)


@pytest.fixture
def rng() -> Generator:
    return _rng


@pytest.fixture
def btc_lf() -> pl.LazyFrame:
    return pl.scan_csv(DATA_DIR / "btc.csv", infer_schema_length=None)


@pytest.fixture
def housing_lf() -> pl.LazyFrame:
    return pl.scan_csv(DATA_DIR / "housing.csv", infer_schema_length=None)


@pytest.fixture
def energy_prices_lf() -> pl.LazyFrame:
    return pl.scan_csv(DATA_DIR / "energy_prices.csv", infer_schema_length=None)


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
