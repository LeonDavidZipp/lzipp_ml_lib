from pathlib import Path

import polars as pl
import pytest

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"


@pytest.fixture
def btc_lf() -> pl.LazyFrame:
    return pl.scan_csv(DATA_DIR / "btc.csv", infer_schema_length=None)


@pytest.fixture
def housing_lf() -> pl.LazyFrame:
    return pl.scan_csv(DATA_DIR / "housing.csv", infer_schema_length=None)


@pytest.fixture
def energy_prices_lf() -> pl.LazyFrame:
    return pl.scan_csv(DATA_DIR / "energy_prices.csv", infer_schema_length=None)
