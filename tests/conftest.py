import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.random import Generator

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
def base_timeseries_lf() -> pl.LazyFrame:
    return pl.scan_csv(
        DATA_DIR / "energy_prices.csv", infer_schema_length=None, try_parse_dates=True
    )
