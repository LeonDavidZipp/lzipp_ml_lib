import inspect
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from lzipp_ml_lib.analysis import plotting
from lzipp_ml_lib.analysis.plotting import (
    ClassifierPlotter,
    EDAPlotter,
    ModelPlotter,
    RegressionPlotter,
    TimeSeriesPlotter,
)

# method -> (wrapped function, the function's parameters the constructor supplies)
WRAPPED: dict[Callable[..., Figure], tuple[Callable[..., Figure], set[str]]] = {
    EDAPlotter.histograms: (plotting.plot_histograms, {"data", "columns", "sample"}),
    EDAPlotter.kde: (plotting.plot_kde, {"data", "columns", "sample"}),
    EDAPlotter.boxplots: (plotting.plot_boxplots, {"data", "columns", "sample"}),
    EDAPlotter.violinplots: (plotting.plot_violinplots, {"data", "columns", "sample"}),
    EDAPlotter.category_counts: (
        plotting.plot_category_counts,
        {"data", "columns", "as_categorical"},
    ),
    EDAPlotter.corr_heatmap: (plotting.plot_corr_heatmap, {"data", "columns"}),
    EDAPlotter.missing_values: (plotting.plot_missing_values, {"data", "columns"}),
    EDAPlotter.feature_target: (
        plotting.plot_feature_target,
        {"data", "columns", "sample"},
    ),
    TimeSeriesPlotter.grid: (
        plotting.plot_timeseries_grid,
        {"data", "time_col", "columns"},
    ),
    TimeSeriesPlotter.gaps: (plotting.plot_gaps, {"data", "time_col"}),
    TimeSeriesPlotter.seasonal_profile: (
        plotting.plot_seasonal_profile,
        {"data", "time_col", "target_col"},
    ),
    TimeSeriesPlotter.autocorrelation: (
        plotting.plot_autocorrelation,
        {"data", "target_col"},
    ),
    TimeSeriesPlotter.partial_autocorrelation: (
        plotting.plot_partial_autocorrelation,
        {"data", "target_col"},
    ),
    TimeSeriesPlotter.decomposition: (
        plotting.plot_decomposition,
        {"data", "time_col", "target_col"},
    ),
    TimeSeriesPlotter.rolling_stats: (
        plotting.plot_rolling_stats,
        {"data", "time_col", "target_col"},
    ),
    ClassifierPlotter.diagnostics: (
        plotting.plot_classification_diagnostics,
        {"y_true", "y_proba", "classes"},
    ),
    ClassifierPlotter.confusion_matrix: (
        plotting.plot_confusion_matrix,
        {"y_true", "y_pred", "classes"},
    ),
    ClassifierPlotter.roc_curves: (
        plotting.plot_roc_curves,
        {"y_true", "y_proba", "classes"},
    ),
    ClassifierPlotter.precision_recall: (
        plotting.plot_precision_recall,
        {"y_true", "y_proba", "classes"},
    ),
    ClassifierPlotter.calibration: (
        plotting.plot_calibration,
        {"y_true", "y_proba", "classes"},
    ),
    RegressionPlotter.diagnostics: (
        plotting.plot_regression_diagnostics,
        {"y_true", "y_pred"},
    ),
    RegressionPlotter.forecast: (plotting.plot_forecast, {"y_true", "y_pred", "ts"}),
    ModelPlotter.feature_importance: (plotting.plot_feature_importance, {"model"}),
    ModelPlotter.learning_curves: (plotting.plot_learning_curves, {"model"}),
}


def _params(f: Callable[..., Any]) -> dict[str, inspect.Parameter]:
    return {n: p for n, p in inspect.signature(f).parameters.items() if n != "self"}


def test_every_plot_function_has_a_plotter_method():
    wrapped = {function for function, _ in WRAPPED.values()}
    functions = {
        getattr(plotting, name) for name in plotting.__all__ if name.startswith("plot_")
    }
    assert functions == wrapped


@pytest.mark.parametrize("method", list(WRAPPED), ids=lambda m: m.__qualname__)
def test_plotter_method_forwards_every_parameter(method: Callable[..., Figure]):
    function, from_constructor = WRAPPED[method]
    method_params = _params(method)
    function_params = _params(function)
    assert set(method_params) | from_constructor == set(function_params)
    for name, param in method_params.items():
        assert param.default == function_params[name].default, name


# ------------------------------------------------------------------------------------ #
#                                      behaviour                                       #
# ------------------------------------------------------------------------------------ #


def _titles(fig: Figure) -> list[str]:
    return [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]


def _table() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
            "code": rng.integers(1, 4, 200),
            "y": rng.normal(size=200),
        }
    )


def test_eda_plotter_applies_its_columns_and_sample_to_every_plot():
    plotter = EDAPlotter(_table().lazy(), ["b", "a"], sample=50)
    assert _titles(plotter.kde()) == ["b", "a"]
    histogram = plotter.histograms(bins=5).axes[0]
    bars = [p for p in histogram.patches if isinstance(p, Rectangle)]
    assert sum(bar.get_height() for bar in bars) == 50
    assert _titles(plotter.feature_target("y")) == ["b", "a"]


def test_eda_plotter_counts_as_categorical_columns():
    fig = EDAPlotter(_table(), as_categorical=["code"]).category_counts()
    assert _titles(fig) == ["code  ·  3 distinct"]


def test_timeseries_plotter_sorts_by_time_first():
    rng = np.random.default_rng(0)
    y = np.zeros(500)
    for t in range(1, 500):
        y[t] = 0.9 * y[t - 1] + rng.normal()
    ts = [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(500)]
    shuffled = pl.DataFrame({"ts": ts, "val": y}).sample(
        fraction=1, shuffle=True, seed=0
    )
    fig = TimeSeriesPlotter(shuffled).autocorrelation(lags=3)
    (markers,) = [line for line in fig.axes[0].lines if line.get_marker() == "o"]
    # in time order the lag-1 autocorrelation is ~0.9; shuffled it would be ~0
    assert np.asarray(markers.get_ydata())[1] > 0.8


def test_classifier_plotter_needs_predictions():
    with pytest.raises(ValueError, match="pass y_pred, y_proba or both"):
        ClassifierPlotter([0, 1])


def test_classifier_plotter_needs_probabilities_for_curves():
    with pytest.raises(ValueError, match="roc_curves needs y_proba"):
        ClassifierPlotter([0, 1, 1], y_pred=[0, 1, 0]).roc_curves()


def test_classifier_plotter_derives_labels_from_probabilities():
    actual = ["x", "y", "y"]
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    from_proba = ClassifierPlotter(actual, y_proba=proba).confusion_matrix(
        normalize=False
    )
    from_labels = plotting.plot_confusion_matrix(
        actual, ["x", "y", "x"], normalize=False
    )
    np.testing.assert_array_equal(
        from_proba.axes[0].collections[0].get_array(),
        from_labels.axes[0].collections[0].get_array(),
    )


def test_regression_plotter_forecast_uses_the_timestamps():
    ts = [datetime(2024, 1, 1) + timedelta(hours=h) for h in [2, 0, 1]]
    fig = RegressionPlotter([20.0, 0.0, 10.0], [20.0, 0.0, 10.0], ts=ts).forecast()
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [0, 10, 20])


def test_model_plotter_rejects_a_booster_for_learning_curves():
    x = np.random.default_rng(0).normal(size=(50, 2))
    booster = xgb.XGBRegressor(n_estimators=3).fit(x, x[:, 0]).get_booster()
    assert isinstance(ModelPlotter(booster).feature_importance(), Figure)
    with pytest.raises(ValueError, match="not a Booster"):
        ModelPlotter(booster).learning_curves()
    plt.close("all")
