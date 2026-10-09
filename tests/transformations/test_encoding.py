import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from lzipp_ml_lib.transformation.tabular._encoding import EncodingFeatures
from tests.composites import SAMPLE_SETTINGS

_CITIES = st.sampled_from(["a", "b", "c", None])


@st.composite
def labelled_frames(draw: st.DrawFn, min_rows: int = 4) -> pl.DataFrame:
    """Small frames with a nullable `city` and a 0/1 target with both classes."""
    n = draw(st.integers(min_rows, 40))
    cities = draw(st.lists(_CITIES, min_size=n, max_size=n))
    labels = draw(
        st.lists(st.integers(0, 1), min_size=n, max_size=n).filter(
            lambda ys: 0 < sum(ys) < len(ys)
        )
    )
    return pl.DataFrame(
        {"city": cities, "y": labels}, schema={"city": pl.String, "y": pl.Int64}
    )


def _fit(df: pl.DataFrame, smoothing: float = 10.0) -> EncodingFeatures:
    return EncodingFeatures().fit(
        df.lazy(), target="y", target_encode=["city"], smoothing=smoothing
    )


# ------------------------------------------------------------------------------------ #
#                                   target encoding                                    #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(df=labelled_frames(), smoothing=st.floats(0.5, 50))
def test_target_encode_matches_a_plain_python_reference(
    df: pl.DataFrame, smoothing: float
):
    encoded = _fit(df, smoothing).target_encode(df.lazy()).collect()
    labels = df["y"].to_list()
    prior = sum(labels) / len(labels)
    for city, value in zip(df["city"], encoded["city_te"], strict=True):
        if city is None:
            assert value == pytest.approx(prior)
            continue
        rows = [y for c, y in zip(df["city"], labels, strict=True) if c == city]
        expected = (sum(rows) + smoothing * prior) / (len(rows) + smoothing)
        assert value == pytest.approx(expected)


def test_target_encode_gives_unseen_categories_the_overall_rate():
    df = pl.DataFrame({"city": ["a", "a", "b", "b"], "y": [1, 1, 0, 1]})
    new = pl.LazyFrame({"city": ["z", None]})
    encoded = _fit(df).target_encode(new).collect()["city_te"].to_list()
    assert encoded == pytest.approx([0.75, 0.75])


@SAMPLE_SETTINGS
@given(df=labelled_frames(min_rows=6), data=st.data())
def test_out_of_fold_encoding_never_sees_the_rows_own_label(
    df: pl.DataFrame, data: st.DataObject
):
    row = data.draw(st.integers(0, df.height - 1))
    flipped = df.with_columns(
        y=pl.when(pl.int_range(pl.len()) == row).then(1 - pl.col("y")).otherwise("y")
    )
    fe = _fit(df)
    before = fe.target_encode(df.lazy(), folds=3).collect()["city_te"][row]
    after = fe.target_encode(flipped.lazy(), folds=3).collect()["city_te"][row]
    assert before == pytest.approx(after)


def test_out_of_fold_encoding_differs_from_the_in_sample_one():
    # a category with one row: in-sample, its rate includes its own label
    df = pl.DataFrame({"city": ["solo"] + ["x"] * 9, "y": [1] + [0] * 9})
    fe = _fit(df, smoothing=1.0)
    in_sample = fe.target_encode(df.lazy()).collect()["city_te"][0]
    out_of_fold = fe.target_encode(df.lazy(), folds=2).collect()["city_te"][0]
    assert in_sample > out_of_fold


def test_target_encode_multiclass_gets_one_column_per_class_summing_to_one():
    df = pl.DataFrame(
        {"city": ["a", "a", "b", "c", "c"], "y": ["x", "y", "z", "x", "x"]}
    )
    encoded = _fit(df).target_encode(df.lazy()).collect()
    columns = ["city_te_x", "city_te_y", "city_te_z"]
    assert encoded.columns == ["city", "y", *columns]
    np.testing.assert_allclose(
        encoded.select(pl.sum_horizontal(columns)).to_series(), 1
    )


def test_target_encode_keeps_the_original_columns():
    df = pl.DataFrame({"city": ["a", "b"], "y": [0, 1]})
    assert _fit(df).target_encode(df.lazy()).collect_schema().names() == [
        "city",
        "y",
        "city_te",
    ]


def test_target_encode_needs_a_target_and_two_classes():
    lf = pl.LazyFrame({"city": ["a", "b"], "y": [1, 1]})
    with pytest.raises(ValueError, match="needs a target"):
        EncodingFeatures().fit(lf, target_encode=["city"])
    with pytest.raises(ValueError, match="at least two classes"):
        EncodingFeatures().fit(lf, target="y", target_encode=["city"])


def test_target_encode_rejects_a_single_fold():
    df = pl.DataFrame({"city": ["a", "b"], "y": [0, 1]})
    with pytest.raises(ValueError, match="at least 2"):
        _fit(df).target_encode(df.lazy(), folds=1)


# ------------------------------------------------------------------------------------ #
#                                 frequency encoding                                   #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(df=labelled_frames())
def test_frequency_shares_of_the_training_categories_sum_to_one(df: pl.DataFrame):
    fe = EncodingFeatures().fit(df.lazy(), frequency_encode=["city"])
    encoded = fe.frequency_encode(df.lazy()).collect()
    per_category = encoded.unique("city").get_column("city_freq")
    assert per_category.sum() == pytest.approx(1)
    for city, share in encoded.select("city", "city_freq").unique().iter_rows():
        is_city = df["city"].is_null() if city is None else df["city"] == city
        expected = is_city.fill_null(False).mean()
        assert share == pytest.approx(expected)


def test_frequency_encode_gives_unseen_categories_zero():
    fe = EncodingFeatures().fit(
        pl.LazyFrame({"city": ["a", "b"]}), frequency_encode=["city"]
    )
    encoded = fe.frequency_encode(pl.LazyFrame({"city": ["z"]})).collect()
    assert encoded["city_freq"].to_list() == [0.0]


# ------------------------------------------------------------------------------------ #
#                                         fit                                          #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("method", ["target_encode", "frequency_encode"])
def test_methods_need_their_fit_arguments(method: str):
    fe = EncodingFeatures().fit(pl.LazyFrame({"a": [1]}))
    with pytest.raises(RuntimeError, match="call fit"):
        getattr(fe, method)(pl.LazyFrame({"a": [1]}))


def test_fit_only_collects_the_columns_it_needs():
    # computing `broken` fails when collected ("a" isn't an integer), so fit must
    # not touch it
    lf = pl.LazyFrame({"city": ["a", "b"], "y": [0, 1]}).with_columns(
        broken=pl.col("city").cast(pl.Int64)
    )
    fe = EncodingFeatures().fit(
        lf, target="y", target_encode=["city"], frequency_encode=["city"]
    )
    assert fe.frequency_encode(lf.drop("broken")).collect().height == 2
