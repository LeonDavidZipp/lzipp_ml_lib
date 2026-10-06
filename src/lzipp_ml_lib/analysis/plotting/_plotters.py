from collections.abc import Sequence
from typing import Literal

from ._category_counts import plot_category_counts
from ._classification import (
    plot_calibration,
    plot_classification_diagnostics,
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc_curves,
)
from ._correlations import plot_corr_heatmap
from ._distributions import (
    plot_boxplots,
    plot_histograms,
    plot_kde,
    plot_violinplots,
)
from ._model import plot_feature_importance, plot_learning_curves
from ._regression import plot_forecast, plot_regression_diagnostics
from ._seasonality import plot_decomposition, plot_seasonal_profile
from ._style import styled
from ._target import plot_feature_target
from ._timeseries import (
    plot_autocorrelation,
    plot_gaps,
    plot_missing_values,
    plot_partial_autocorrelation,
    plot_rolling_stats,
    plot_timeseries_grid,
)
from ._utils import PolarsFrame


class AnalysisPlotter:
    def __init__(
        self,
        data: PolarsFrame,
        columns: Sequence[str] | None = None,
        as_categorical: Sequence[str] | None = None,
        sample: int | None = None,
    ):
        self._data = data
        self._columns = columns
        self._as_categorical = as_categorical
        self._sample = sample


class EDAPlotter:
    """Handles distributions, categories, and correlations."""

    def __init__(
        self,
        data: PolarsFrame,
        columns: Sequence[str] | None = None,
    ):
        self._data = data
        self._columns = columns

    def category_counts(
        self, top_k: int = 10, as_categorical: Sequence[str] | None = None
    ):
        return plot_category_counts(
            self._data, self._columns, as_categorical, top_k=top_k
        )

    def corr_heatmap(
        self,
        figsize: tuple[float, float] = (10, 8),
        method: Literal["pearson", "spearman"] = "pearson",
    ):
        return plot_corr_heatmap(
            self._data, figsize, columns=self._columns, method=method
        )

    def histograms(
        self,
    ):
        """Plots histograms for the numeric columns."""
        return plot_histograms(
            self._data,
            columns=self._columns,
        )

    def boxplots(
        self,
    ):
        """Plots boxplots for the numeric columns."""
        return plot_boxplots(
            self._data,
            columns=self._columns,
        )

    def violinplots(
        self,
    ):
        """Plots violin plots for the numeric columns."""
        return plot_violinplots(
            self._data,
            columns=self._columns,
        )

    def kde(
        self,
    ):
        """Plots Kernel Density Estimates (KDE) for the numeric columns."""
        return plot_kde(
            self._data,
            columns=self._columns,
        )

    def feature_target(
        self,
        target: str,
    ):
        return plot_feature_target(
            self._data,
            target,
            columns=self._columns,
        )


class TimeSeriesPlotter:
    """Handles everything requiring a time index."""

    def __init__(
        self, data: PolarsFrame, time_col: str, value_cols: Sequence[str] | None = None
    ):
        self._data = data
        self.time_col = time_col
        self.value_cols = value_cols

    def rolling_stats(
        self,
        window: int = 7,
    ):
        return plot_rolling_stats(
            self._data,
            self.time_col,
            self.value_cols,
            window=window,
        )

    def seasonality(
        self,
        period: int,
    ):
        return plot_seasonal_profile(
            self._data,
            self.time_col,
            self.value_cols,
            period=period,
        )

    def gaps_and_missing(
        self,
    ):
        # You could even combine these into a single figure in the wrapper
        pass


class ClassifierPlotter:
    """Handles classification model evaluations."""

    def __init__(
        self, y_true, y_pred, y_prob=None, classes: Sequence[str] | None = None
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.classes = classes

    def diagnostics(
        self,
    ):
        return plot_classification_diagnostics(
            self.y_true,
            self.y_pred,
            self.classes,
        )

    def roc_curves(
        self,
    ):
        if self.y_prob is None:
            raise ValueError("ROC curves require y_prob")
        return plot_roc_curves(
            self.y_true,
            self.y_prob,
            self.classes,
        )


# --- 4. Regression Diagnostics ---
class RegressionPlotter:
    """Handles regression model evaluations."""

    def __init__(self, y_true, y_pred, time_col=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.time_col = time_col

    def diagnostics(
        self,
    ):
        return plot_regression_diagnostics(
            self.y_true,
            self.y_pred,
        )

    def forecast(
        self,
    ):
        if self.time_col is None:
            raise ValueError("Forecast plots require a time_col")
        return plot_forecast(
            self.time_col,
            self.y_true,
            self.y_pred,
        )
