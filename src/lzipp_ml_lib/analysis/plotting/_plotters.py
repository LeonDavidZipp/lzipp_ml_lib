"""Classes bundling the plot functions around one thing being analysed.

The constructor takes what stays the same across plots: the data and its columns,
the labels and predictions, the model. The methods take what varies from plot to
plot: bins, scales, grouping, the axes to draw on, and so on. Every method
returns its figure, like the functions it calls; see those for the details.
"""

from collections.abc import Sequence
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import xgboost as xgb
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._category_counts import plot_category_counts
from ._classification import (
    _prepare,  # type: ignore
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
from ._seasonality import CalendarUnit, plot_decomposition, plot_seasonal_profile
from ._target import plot_feature_target
from ._timeseries import (
    plot_autocorrelation,
    plot_gaps,
    plot_missing_values,
    plot_partial_autocorrelation,
    plot_rolling_stats,
    plot_timeseries_grid,
)
from ._utils import PolarsFrame, Values


class EDAPlotter:
    """Distributions, categories, correlations, missing values and the relation to a
    target, for one table.

    `columns` restricts every plot to these columns. `sample` caps the rows of the
    plots that can sample (the distributions and the feature-target plot), for
    speed on large data. `as_categorical` names numeric columns to count as
    categories in `category_counts`, e.g. codes like a region id.
    """

    def __init__(
        self,
        data: PolarsFrame,
        columns: Sequence[str] | None = None,
        *,
        sample: int | None = None,
        as_categorical: Sequence[str] | None = None,
    ):
        self._data = data
        self._columns = columns
        self._sample = sample
        self._as_categorical = as_categorical

    def histograms(
        self,
        *,
        bins: int | str = "auto",
        by: str | None = None,
        log: bool = False,
        clip: tuple[float, float] | None = None,
    ) -> Figure:
        """A histogram per numeric column; see `plot_histograms`."""
        return plot_histograms(
            self._data,
            self._columns,
            bins=bins,
            by=by,
            sample=self._sample,
            log=log,
            clip=clip,
        )

    def kde(
        self,
        *,
        by: str | None = None,
        log: bool = False,
        clip: tuple[float, float] | None = None,
    ) -> Figure:
        """A density curve per numeric column; see `plot_kde`."""
        return plot_kde(
            self._data, self._columns, by=by, sample=self._sample, log=log, clip=clip
        )

    def boxplots(
        self,
        *,
        by: str | None = None,
        log: bool = False,
        clip: tuple[float, float] | None = None,
    ) -> Figure:
        """A boxplot per numeric column; see `plot_boxplots`."""
        return plot_boxplots(
            self._data, self._columns, by=by, sample=self._sample, log=log, clip=clip
        )

    def violinplots(
        self,
        *,
        by: str | None = None,
        log: bool = False,
        clip: tuple[float, float] | None = None,
    ) -> Figure:
        """A violin plot per numeric column; see `plot_violinplots`."""
        return plot_violinplots(
            self._data, self._columns, by=by, sample=self._sample, log=log, clip=clip
        )

    def category_counts(self, *, top_k: int = 10) -> Figure:
        """Value counts per categorical column; see `plot_category_counts`."""
        return plot_category_counts(
            self._data, self._columns, self._as_categorical, top_k=top_k
        )

    def corr_heatmap(
        self,
        *,
        method: Literal["pearson", "spearman"] = "pearson",
        figsize: tuple[float, float] = (10, 8),
        ax: Axes | None = None,
    ) -> Figure:
        """Pairwise correlations of the numeric columns; see `plot_corr_heatmap`."""
        return plot_corr_heatmap(
            self._data, figsize, self._columns, method=method, ax=ax
        )

    def missing_values(self, *, ax: Axes | None = None) -> Figure:
        """The share of missing values per column; see `plot_missing_values`."""
        return plot_missing_values(self._data, self._columns, ax=ax)

    def feature_target(
        self,
        target: str,
        *,
        n_bins: int = 10,
        top_k: int = 10,
        log_x: bool | Sequence[str] = False,
    ) -> Figure:
        """The mean of `target` across every other column; see
        `plot_feature_target`. `target` is always included, even if `columns`
        leaves it out."""
        return plot_feature_target(
            self._data,
            target,
            self._columns,
            n_bins=n_bins,
            top_k=top_k,
            sample=self._sample,
            log_x=log_x,
        )


class TimeSeriesPlotter:
    """Gaps, calendar profiles, autocorrelation, decomposition and rolling
    statistics of one series, plus line charts of several.

    `target_col` is the series most plots analyse; `columns` are the ones `grid`
    draws (by default all numeric columns). The data is sorted by `time_col` before
    every plot, so the (partial) autocorrelation, which only sees the order of the
    values, is computed in time order too.
    """

    def __init__(
        self,
        data: PolarsFrame,
        time_col: str = "ts",
        target_col: str = "val",
        *,
        columns: Sequence[str] | None = None,
    ):
        self._data = data.sort(time_col)
        self._time_col = time_col
        self._target_col = target_col
        self._columns = columns

    def grid(self) -> Figure:
        """A line chart per column over time; see `plot_timeseries_grid`."""
        return plot_timeseries_grid(self._data, self._time_col, self._columns)

    def gaps(
        self,
        *,
        interval: str | timedelta | None = None,
        ax: Axes | None = None,
    ) -> Figure:
        """Where timestamps are missing; see `plot_gaps`."""
        return plot_gaps(self._data, self._time_col, interval=interval, ax=ax)

    def seasonal_profile(
        self,
        *,
        rows: CalendarUnit = "weekday",
        cols: CalendarUnit | None = "hour",
        agg: Literal["mean", "median"] = "mean",
        ax: Axes | None = None,
    ) -> Figure:
        """The target's typical value per calendar position; see
        `plot_seasonal_profile`."""
        return plot_seasonal_profile(
            self._data,
            self._time_col,
            self._target_col,
            rows=rows,
            cols=cols,
            agg=agg,
            ax=ax,
        )

    def autocorrelation(self, *, lags: int = 50, ax: Axes | None = None) -> Figure:
        """The target's ACF; see `plot_autocorrelation`."""
        return plot_autocorrelation(self._data, self._target_col, lags, ax=ax)

    def partial_autocorrelation(
        self, *, lags: int = 50, ax: Axes | None = None
    ) -> Figure:
        """The target's PACF; see `plot_partial_autocorrelation`."""
        return plot_partial_autocorrelation(self._data, self._target_col, lags, ax=ax)

    def decomposition(
        self,
        *,
        periods: int | Sequence[int] | None = None,
        robust: bool = True,
    ) -> Figure:
        """Trend, seasonality and residual; see `plot_decomposition`."""
        return plot_decomposition(
            self._data,
            self._time_col,
            self._target_col,
            periods=periods,
            robust=robust,
        )

    def rolling_stats(self, *, window: str | int | None = None) -> Figure:
        """Rolling mean and std of the target; see `plot_rolling_stats`."""
        return plot_rolling_stats(
            self._data, self._time_col, self._target_col, window=window
        )


class ClassifierPlotter:
    """Confusion matrix, ROC, precision-recall and calibration for one set of
    classification results.

    `y_pred` holds predicted labels, `y_proba` predicted probabilities (as from
    `predict_proba`); pass either or both. Without `y_pred`, the confusion matrix
    uses the most probable class. The ROC, precision-recall and calibration plots
    need `y_proba`. `classes` are the labels of `y_proba`'s columns, in order; by
    default the sorted distinct labels of `y_true`.
    """

    def __init__(
        self,
        y_true: Values,
        y_pred: Values | None = None,
        y_proba: Values | None = None,
        *,
        classes: Sequence[Any] | None = None,
    ):
        if y_pred is None and y_proba is None:
            raise ValueError("pass y_pred, y_proba or both")
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_proba = y_proba
        self._classes = classes

    def diagnostics(self) -> Figure:
        """All four plots at once; see `plot_classification_diagnostics`."""
        return plot_classification_diagnostics(
            self._y_true, self._proba("diagnostics"), classes=self._classes
        )

    def confusion_matrix(
        self, *, normalize: bool = True, ax: Axes | None = None
    ) -> Figure:
        """Actual vs predicted classes; see `plot_confusion_matrix`."""
        y_pred = self._y_pred
        if y_pred is None:
            _, proba, labels = _prepare(
                self._y_true, self._proba("confusion_matrix"), self._classes
            )
            y_pred = np.asarray(labels)[proba.argmax(axis=1)]
        return plot_confusion_matrix(
            self._y_true, y_pred, classes=self._classes, normalize=normalize, ax=ax
        )

    def roc_curves(self, *, ax: Axes | None = None) -> Figure:
        """ROC curves with their AUC; see `plot_roc_curves`."""
        return plot_roc_curves(
            self._y_true, self._proba("roc_curves"), classes=self._classes, ax=ax
        )

    def precision_recall(self, *, ax: Axes | None = None) -> Figure:
        """Precision-recall curves with their AP; see `plot_precision_recall`."""
        return plot_precision_recall(
            self._y_true, self._proba("precision_recall"), classes=self._classes, ax=ax
        )

    def calibration(self, *, n_bins: int = 10, ax: Axes | None = None) -> Figure:
        """Predicted probability vs observed frequency; see `plot_calibration`."""
        return plot_calibration(
            self._y_true,
            self._proba("calibration"),
            classes=self._classes,
            n_bins=n_bins,
            ax=ax,
        )

    def _proba(self, plot: str) -> Values:
        if self._y_proba is None:
            raise ValueError(f"{plot} needs y_proba (predicted probabilities)")
        return self._y_proba


class RegressionPlotter:
    """Error diagnostics and the forecast view for one set of regression results.

    `ts` holds the timestamps of the predictions (e.g. the test set's `ts`
    column), for `forecast`; without it, its x axis is the row number.
    """

    def __init__(
        self,
        y_true: Values,
        y_pred: Values,
        *,
        ts: Values | None = None,
    ):
        self._y_true = y_true
        self._y_pred = y_pred
        self._ts = ts

    def diagnostics(self) -> Figure:
        """Predicted vs actual, residuals and their distribution; see
        `plot_regression_diagnostics`."""
        return plot_regression_diagnostics(self._y_true, self._y_pred)

    def forecast(self) -> Figure:
        """Actual and predicted over time, with the residual; see
        `plot_forecast`."""
        return plot_forecast(self._y_true, self._y_pred, self._ts)


class ModelPlotter:
    """Feature importance and learning curves of one fitted XGBoost model."""

    def __init__(self, model: xgb.XGBModel | xgb.Booster):
        self._model = model

    def feature_importance(
        self,
        *,
        importance_type: Literal[
            "gain", "total_gain", "weight", "cover", "total_cover"
        ] = "gain",
        top_k: int = 20,
        ax: Axes | None = None,
    ) -> Figure:
        """Each feature's share of the total importance; see
        `plot_feature_importance`."""
        return plot_feature_importance(
            self._model, importance_type=importance_type, top_k=top_k, ax=ax
        )

    def learning_curves(self) -> Figure:
        """The eval metrics per boosting round; see `plot_learning_curves`. Needs
        a scikit-learn style model fit with an `eval_set`."""
        if not isinstance(self._model, xgb.XGBModel):
            raise ValueError(
                "learning curves need a scikit-learn style model (e.g. "
                "XGBRegressor) fit with an eval_set, not a Booster"
            )
        return plot_learning_curves(self._model)
