import math
from collections.abc import Sequence
from typing import get_args

import numpy as np
import polars as pl
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy.stats import yeojohnson  # type: ignore

from lzipp_ml_lib.transformation._numeric import (
    ImputeStrategy,
    NumericFeatures,
    ScaleMethod,
    signed_log1p_expr,
)
from tests.composites import SAMPLE_SETTINGS

_FINITE = st.floats(-1e4, 1e4, allow_nan=False)


@st.composite
def columns(
    draw: st.DrawFn, min_size: int = 3, nulls: bool = True
) -> Sequence[float | None]:
    """A numeric column with at least two distinct values, optionally with nulls."""
    values = draw(st.lists(_FINITE, min_size=min_size, max_size=50))
    if len(set(values)) < 2:
        values = [*values, max(values) + 1]
    if nulls:
        holes = draw(
            st.sets(st.integers(0, len(values) - 1), max_size=len(values) // 3)
        )
        return [None if i in holes else v for i, v in enumerate(values)]
    return values


def _col_as_np(lf: pl.LazyFrame, name: str = "x") -> np.ndarray:
    return lf.collect().get_column(name).to_numpy()


# ------------------------------------------------------------------------------------ #
#                                        impute                                        #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(train=columns(), test=columns())
def test_impute_fills_with_the_training_median(
    train: list[float | None], test: list[float | None]
):
    nf = NumericFeatures().fit(pl.LazyFrame({"x": train}), impute=["x"])
    filled = _col_as_np(nf.impute(pl.LazyFrame({"x": test})))
    median = float(np.median([v for v in train if v is not None]))
    for before, after in zip(test, filled, strict=True):
        assert after == pytest.approx(median if before is None else before)


@pytest.mark.parametrize(
    ("strategy", "fill"), [("median", 2.0), ("mean", 3.0), ("zero", 0.0)]
)
def test_impute_treats_nan_as_missing(strategy: ImputeStrategy, fill: float):
    train = pl.LazyFrame({"x": [1.0, 2.0, 6.0, math.nan, None]})
    nf = NumericFeatures().fit(train, impute=["x"], impute_strategy=strategy)
    test = pl.LazyFrame({"x": [math.nan, None]})
    assert _col_as_np(nf.impute(test)).tolist() == [fill, fill]


# ------------------------------------------------------------------------------------ #
#                                         clip                                         #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(train=columns(nulls=False), test=columns(nulls=False))
def test_clip_uses_the_training_quantiles(train: list[float], test: list[float]):
    nf = NumericFeatures().fit(
        pl.LazyFrame({"x": train}), clip=["x"], clip_quantiles=(0.1, 0.9)
    )
    low, high = np.quantile(train, [0.1, 0.9])
    clipped = _col_as_np(nf.clip(pl.LazyFrame({"x": test})))
    np.testing.assert_allclose(clipped, np.clip(test, low, high))


def test_clip_rejects_bad_quantiles():
    with pytest.raises(ValueError, match="clip_quantiles"):
        NumericFeatures().fit(
            pl.LazyFrame({"x": [1.0]}), clip=["x"], clip_quantiles=(0.9, 0.1)
        )


# ------------------------------------------------------------------------------------ #
#                                       auto_log                                       #
# ------------------------------------------------------------------------------------ #


def test_auto_log_only_logs_right_skewed_col_as_npumns():
    rng = np.random.default_rng(0)
    train = pl.LazyFrame(
        {
            "skewed": rng.lognormal(size=2000),
            "symmetric": rng.normal(size=2000),
            "left_skewed": -rng.lognormal(size=2000),
        }
    )
    nf = NumericFeatures().fit(train, auto_log=True)
    out = nf.auto_log(train).collect()
    np.testing.assert_allclose(out["skewed"], np.log1p(train.collect()["skewed"]))
    assert out["symmetric"].equals(train.collect()["symmetric"])
    assert out["left_skewed"].equals(train.collect()["left_skewed"])


def test_auto_log_uses_the_signed_log_for_col_as_npumns_with_negatives():
    train = pl.LazyFrame({"x": [-5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 500.0]})
    nf = NumericFeatures().fit(train, auto_log=["x"])
    logged = _col_as_np(nf.auto_log(pl.LazyFrame({"x": [-5.0, 0.0, 500.0]})))
    np.testing.assert_allclose(logged, [-math.log1p(5), 0.0, math.log1p(500)])


@SAMPLE_SETTINGS
@given(values=st.lists(_FINITE, min_size=2, max_size=50))
def test_signed_log_keeps_the_order_of_the_values(values: list[float]):
    logged = pl.select(signed_log1p_expr(pl.lit(pl.Series(values)))).to_series()
    assert np.array_equal(
        np.argsort(values, kind="stable"), np.argsort(logged, kind="stable")
    )


# ------------------------------------------------------------------------------------ #
#                                    power_transform                                   #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(train=columns(min_size=5, nulls=False), test=columns(nulls=False))
def test_power_transform_matches_scipy(train: list[float], test: list[float]):
    nf = NumericFeatures().fit(pl.LazyFrame({"x": train}), power_transform=["x"])
    lmbda = nf._power["x"]  # type: ignore
    ours = _col_as_np(nf.power_transform(pl.LazyFrame({"x": test})))
    np.testing.assert_allclose(
        ours,
        yeojohnson(np.array(test), lmbda),  # type: ignore
        rtol=1e-6,
        atol=1e-6,
    )


@SAMPLE_SETTINGS
@given(train=columns(min_size=5, nulls=False), test=columns(nulls=False))
# a range of only rounding noise used to break scipy's lambda search
@example(train=[0.0, 0.0, 0.0, 0.0, 2.2250738585072014e-308], test=[0.0, 1.0])
def test_power_transform_keeps_the_order_of_the_values(
    train: list[float], test: list[float]
):
    nf = NumericFeatures().fit(pl.LazyFrame({"x": train}), power_transform=["x"])
    out = _col_as_np(nf.power_transform(pl.LazyFrame({"x": test})))
    order = np.argsort(test, kind="stable")
    assert np.all(np.diff(out[order]) >= -1e-9)


def test_power_transform_removes_skew():
    train = pl.LazyFrame({"x": np.random.default_rng(0).lognormal(size=5000)})
    nf = NumericFeatures().fit(train, power_transform=["x"])
    before = train.collect()["x"].skew()
    after = nf.power_transform(train).collect()["x"].skew()
    # Yeo-Johnson maximizes normal likelihood rather than zeroing the skew, so it gets
    # close to 0, not exactly there
    assert abs(after) < 0.05 * abs(before)  # type: ignore


def test_auto_log_and_power_transform_cannot_share_a_col_as_npumn():
    with pytest.raises(ValueError, match="both auto_log and power_transform"):
        NumericFeatures().fit(
            pl.LazyFrame({"x": [1.0, 2.0]}), auto_log=["x"], power_transform=["x"]
        )


# ------------------------------------------------------------------------------------ #
#                                        scale                                         #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(
    train=columns(nulls=False),
    test=columns(nulls=False),
    method=st.sampled_from(get_args(ScaleMethod)),
)
def test_scale_uses_the_training_parameters(
    train: list[float], test: list[float], method: ScaleMethod
):
    nf = NumericFeatures().fit(
        pl.LazyFrame({"x": train}), scale=["x"], scale_method=method
    )
    t = np.array(train)
    center, spread = {
        "standard": (t.mean(), t.std(ddof=1)),
        "minmax": (t.min(), t.max() - t.min()),
        "robust": (np.median(t), np.subtract(*np.quantile(t, [0.75, 0.25]))),
    }[method]
    spread = spread if spread > 0 else 1.0
    scaled = _col_as_np(nf.scale(pl.LazyFrame({"x": test})))
    np.testing.assert_allclose(
        scaled, (np.array(test) - center) / spread, rtol=1e-9, atol=1e-9
    )


def test_scale_only_shifts_a_constant_col_as_npumn():
    nf = NumericFeatures().fit(pl.LazyFrame({"x": [3.0, 3.0, 3.0]}), scale=["x"])
    assert _col_as_np(nf.scale(pl.LazyFrame({"x": [3.0, 5.0]}))).tolist() == [0.0, 2.0]


# ------------------------------------------------------------------------------------ #
#                                  the steps together                                  #
# ------------------------------------------------------------------------------------ #


def test_fit_learns_each_step_on_the_output_of_the_ones_before():
    # scaling parameters learned after imputing, clipping and logging, so the
    # transformed training data is exactly standardized
    rng = np.random.default_rng(0)
    income = rng.lognormal(10, 1, 3000)
    income[rng.random(3000) < 0.1] = np.nan
    train = pl.LazyFrame({"income": income})
    nf = NumericFeatures().fit(
        train, impute=["income"], clip=["income"], auto_log=True, scale=["income"]
    )
    out = nf.transform(train).collect()["income"]
    assert out.mean() == pytest.approx(0, abs=1e-9)
    assert out.std() == pytest.approx(1)


def test_transform_applies_the_steps_in_order():
    train = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, None]})
    nf = NumericFeatures().fit(train, impute=["x"], scale=["x"], scale_method="minmax")
    by_hand = nf.scale(nf.impute(train))
    assert nf.transform(train).collect().equals(by_hand.collect())


def test_suffix_adds_col_as_npumns_and_keeps_the_originals():
    train = pl.LazyFrame({"x": [1.0, 2.0, 3.0]})
    nf = NumericFeatures().fit(train, scale=["x"])
    assert nf.scale(train, suffix="_scaled").collect_schema().names() == [
        "x",
        "x_scaled",
    ]


@pytest.mark.parametrize(
    "method", ["impute", "clip", "auto_log", "power_transform", "scale", "transform"]
)
def test_methods_need_their_fit_arguments(method: str):
    nf = NumericFeatures().fit(pl.LazyFrame({"x": [1.0, 2.0]}))
    with pytest.raises(RuntimeError, match="call fit"):
        getattr(nf, method)(pl.LazyFrame({"x": [1.0]}))


def test_fit_rejects_non_numeric_col_as_npumns():
    with pytest.raises(ValueError, match="not numeric"):
        NumericFeatures().fit(pl.LazyFrame({"s": ["a"]}), scale=["s"])
