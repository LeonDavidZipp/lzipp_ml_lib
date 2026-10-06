import re
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from hypothesis import given
from hypothesis import strategies as st
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

from lzipp_ml_lib.analysis import plotting
from lzipp_ml_lib.analysis.plotting import _regression  # type: ignore
from lzipp_ml_lib.analysis.plotting._utils import as_1d, as_proba
from tests.composites import SAMPLE_SETTINGS

# scores are shown with 3 decimals, so they're off by up to half a unit of the last
# digit (an AUC of 0.3125 shows as "0.312"), plus float noise on top
_SHOWN = 5e-4 + 1e-9


def _number(text: str, label: str) -> float:
    match = re.search(rf"{label} (-?[\d.]+)", text)
    assert match, f"{label!r} not in {text!r}"
    return float(match.group(1))


def _regression_data(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    actual = rng.normal(50, 10, n)
    return actual, actual + rng.normal(0, 3, n)


def _binary_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn with exactly the predicted probability: perfectly calibrated."""
    rng = np.random.default_rng(seed)
    proba = rng.uniform(0, 1, n)
    return (rng.uniform(0, 1, n) < proba).astype(int), proba


# ------------------------------------------------------------------------------------ #
#                                       inputs                                         #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0],
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0], [2.0], [3.0]]),
        pl.Series("y", [1.0, 2.0, 3.0]),
        pl.DataFrame({"y": [1.0, 2.0, 3.0]}),
    ],
    ids=["list", "array", "column array", "series", "frame"],
)
def test_as_1d_accepts_every_kind_of_input(values: object):
    np.testing.assert_array_equal(as_1d(values, "y"), [1.0, 2.0, 3.0])  # type: ignore


def test_as_1d_rejects_several_columns():
    with pytest.raises(ValueError, match="exactly one column"):
        as_1d(pl.DataFrame({"a": [1], "b": [2]}), "y")


def test_as_proba_takes_a_flat_input_as_the_positive_class():
    np.testing.assert_allclose(as_proba([0.2, 0.9]), [[0.8, 0.2], [0.1, 0.9]])


# ------------------------------------------------------------------------------------ #
#                                     regression                                       #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(30, 300))
def test_regression_diagnostics_report_sklearns_r2(seed: int, n: int):
    actual, predicted = _regression_data(n, seed)
    fig = plotting.plot_regression_diagnostics(actual, predicted)
    text = fig.axes[0].texts[0].get_text()  # type: ignore
    assert _number(text, "R²") == pytest.approx(r2_score(actual, predicted), abs=_SHOWN)  # type: ignore
    plt.close(fig)


def test_regression_diagnostics_histogram_holds_every_residual():
    actual, predicted = _regression_data()
    fig = plotting.plot_regression_diagnostics(
        pl.DataFrame({"y": actual}), pl.Series(predicted)
    )
    assert sum(p.get_height() for p in fig.axes[2].patches) == len(actual)  # type: ignore
    title = fig.axes[2].get_title(loc="left")
    assert _number(title, "mean") == pytest.approx(
        (actual - predicted).mean(), abs=5e-3
    )


def test_regression_diagnostics_switch_to_hexbin_for_many_points(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(_regression, "_HEXBIN_FROM", 100)
    fig = plotting.plot_regression_diagnostics(*_regression_data(n=300))
    assert any(isinstance(c, PolyCollection) for c in fig.axes[0].collections)  # type: ignore


def test_regression_diagnostics_reject_different_lengths():
    with pytest.raises(ValueError, match="lengths differ"):
        plotting.plot_regression_diagnostics([1, 2, 3], [1, 2])


def test_forecast_sorts_by_time():
    ts = [datetime(2024, 1, 1) + timedelta(hours=h) for h in [2, 0, 1]]
    fig = plotting.plot_forecast([20.0, 0.0, 10.0], [21.0, 1.0, 11.0], ts)  # type: ignore
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [0, 10, 20])  # type: ignore


def test_forecast_breaks_lines_at_gaps():
    hours = [0, 1, 2, 3, 10, 11, 12]
    ts = pl.Series([datetime(2024, 1, 1) + timedelta(hours=h) for h in hours])
    fig = plotting.plot_forecast(np.arange(7.0), np.arange(7.0), ts)
    actual = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)  # type: ignore
    assert np.isnan(actual).sum() == 1  # type: ignore
    assert np.isnan(actual[4])  # between hour 3 and hour 10


def test_forecast_without_timestamps_uses_the_row_number():
    fig = plotting.plot_forecast([1.0, 2.0, 3.0], [1.0, 2.0, 2.0])
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [0, 1, 2])  # type: ignore
    assert fig.axes[1].get_xlabel() == "Row"


# ------------------------------------------------------------------------------------ #
#                                      the model                                       #
# ------------------------------------------------------------------------------------ #


def _model_data(seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = pl.DataFrame(
        {
            "signal": rng.normal(size=500),
            "noise": rng.normal(size=500),
            "constant": np.ones(500),
        }
    )
    return x, x["signal"].to_numpy() * 10 + rng.normal(0, 0.1, 500)


def test_feature_importance_ranks_the_signal_first_and_sums_to_one():
    x, y = _model_data()
    model = xgb.XGBRegressor(n_estimators=20).fit(x, y)
    ax = plotting.plot_feature_importance(model).axes[0]
    names = [t.get_text() for t in ax.get_yticklabels()]
    shares = [p.get_width() for p in ax.patches]  # type: ignore
    assert names[-1] == "signal"  # top bar
    assert sum(shares) == pytest.approx(1)  # type: ignore
    assert "1 of 3 features unused" in ax.get_title(loc="left")  # "constant"


def test_feature_importance_folds_the_rest_into_other():
    x, y = _model_data()
    model = xgb.XGBRegressor(n_estimators=20).fit(x, y)
    ax = plotting.plot_feature_importance(model.get_booster(), top_k=1).axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["other (1)", "signal"]


def test_learning_curves_need_an_eval_set():
    x, y = _model_data()
    with pytest.raises(ValueError, match="fit it with an eval_set"):
        plotting.plot_learning_curves(xgb.XGBRegressor(n_estimators=5).fit(x, y))


def test_learning_curves_draw_every_eval_set_and_metric():
    x, y = _model_data()
    model = xgb.XGBRegressor(
        n_estimators=200, early_stopping_rounds=5, eval_metric=["rmse", "mae"]
    ).fit(x[:400], y[:400], eval_set=[(x[:400], y[:400]), (x[400:], y[400:])])
    fig = plotting.plot_learning_curves(model)
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert titles == [
        f"rmse  ·  best round {model.best_iteration}",
        f"mae  ·  best round {model.best_iteration}",
    ]
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["eval_set[0]", "eval_set[1]"]  # type: ignore


# ------------------------------------------------------------------------------------ #
#                                   classification                                     #
# ------------------------------------------------------------------------------------ #


def test_confusion_matrix_colours_are_row_shares():
    actual = ["a", "a", "a", "b", "b", "c"]
    predicted = ["a", "a", "b", "b", "c", "c"]
    ax = plotting.plot_confusion_matrix(actual, predicted).axes[0]
    shares = ax.collections[0].get_array().reshape(3, 3)  # type: ignore
    np.testing.assert_allclose(shares[0], [2 / 3, 1 / 3, 0])  # type: ignore
    assert "accuracy 66.7%" in ax.get_title(loc="left")


def test_confusion_matrix_follows_the_given_class_order():
    ax = plotting.plot_confusion_matrix(
        [0, 1, 1], [0, 1, 0], classes=[1, 0], normalize=False
    ).axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["1", "0"]
    np.testing.assert_array_equal(
        ax.collections[0].get_array().reshape(2, 2),  # type: ignore
        [[1, 1], [0, 1]],  # type: ignore
    )


@SAMPLE_SETTINGS
@given(
    labels=st.lists(st.integers(0, 1), min_size=10, max_size=60).filter(
        lambda ls: 0 < sum(ls) < len(ls)
    ),
    data=st.data(),
)
def test_roc_and_precision_recall_report_sklearns_scores(
    labels: list[int], data: st.DataObject
):
    proba = data.draw(
        st.lists(st.floats(0, 1), min_size=len(labels), max_size=len(labels))
    )
    roc = plotting.plot_roc_curves(labels, proba).axes[0].get_title(loc="left")
    pr = plotting.plot_precision_recall(labels, proba).axes[0].get_title(loc="left")  # type: ignore
    assert _number(roc, "AUC") == pytest.approx(
        roc_auc_score(labels, proba), abs=_SHOWN
    )
    expected_ap = average_precision_score(labels, proba)
    assert _number(pr, "AP") == pytest.approx(expected_ap, abs=_SHOWN)
    plt.close("all")


def test_multiclass_roc_has_one_curve_per_class():
    rng = np.random.default_rng(0)
    actual = rng.choice(["x", "y", "z"], 300)
    proba = rng.dirichlet([1, 1, 1], 300)
    ax = plotting.plot_roc_curves(actual, proba).axes[0]
    entries = [t.get_text() for t in ax.get_legend().get_texts()]  # type: ignore
    assert [e.split()[0] for e in entries] == ["x", "y", "z"]
    assert "macro AUC" in ax.get_title(loc="left")


def test_calibration_of_a_calibrated_model_follows_the_diagonal():
    actual, proba = _binary_data(n=20_000)
    ax = plotting.plot_calibration(actual, proba).axes[0]
    curve = next(line for line in ax.lines if line.get_marker() == "o")  # type: ignore
    np.testing.assert_allclose(curve.get_ydata(), curve.get_xdata(), atol=0.03)  # type: ignore


def test_classification_diagnostics_has_four_panels():
    actual, proba = _binary_data()
    fig = plotting.plot_classification_diagnostics(pl.DataFrame({"y": actual}), proba)
    titles = [ax.get_title(loc="left").split("  ·")[0] for ax in fig.axes]
    assert titles == ["Confusion matrix", "ROC", "Precision-recall", "Calibration"]


def test_probability_plots_check_classes_against_proba_columns():
    with pytest.raises(ValueError, match="3 columns but there are 2 classes"):
        plotting.plot_roc_curves([0, 1, 1], np.full((3, 3), 1 / 3))


def test_probability_plots_cap_the_number_of_curves():
    actual = np.arange(9).repeat(2)
    with pytest.raises(ValueError, match="at most 8 curves"):
        plotting.plot_roc_curves(actual, np.full((18, 9), 1 / 9))
