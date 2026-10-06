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
    """Plots the distributions, categories, correlations, missing values and the
    relation to a target of one table.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        columns (Sequence[str] | None): Columns to restrict every plot to, in this
            order. If None, all columns are used. Defaults to None.
        sample (int | None): Plot at most this many random rows in the plots that can
            sample (the distributions and `feature_target`), for speed on large data.
            Defaults to None.
        as_categorical (Sequence[str] | None): Numeric columns to count as categories in
            `category_counts`, e.g. codes like a region id. Defaults to None.
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
        """Plots a histogram for every numeric column. See `plot_histograms`.

        Args:
            bins (int | str): The number of bins, or a numpy rule like "auto" or
                "sturges". Integer columns with at most 50 distinct values always get
                one bar per value. Defaults to "auto".
            by (str | None): A column whose groups are compared within each panel, e.g.
                the class or the train/test split; at most 8 groups. Defaults to None.
            log (bool): Whether to put the values on a log scale; non-positive values
                are dropped. Defaults to False.
            clip (tuple[float, float] | None): Keep only values inside this quantile
                range per column, e.g. (0.01, 0.99), so a few extremes don't flatten the
                plot. Defaults to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If none of the selected columns is numeric, or if `by` has more
                than 8 groups.
        """
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
        """Plots a density curve for every numeric column. See `plot_kde`.

        Args:
            by (str | None): A column whose groups are compared within each panel, e.g.
                the class or the train/test split; at most 8 groups. Defaults to None.
            log (bool): Whether to put the values on a log scale; non-positive values
                are dropped. Defaults to False.
            clip (tuple[float, float] | None): Keep only values inside this quantile
                range per column, e.g. (0.01, 0.99), so a few extremes don't flatten the
                plot. Defaults to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If none of the selected columns is numeric, or if `by` has more
                than 8 groups.
        """
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
        """Plots a boxplot for every numeric column. See `plot_boxplots`.

        Args:
            by (str | None): A column whose groups are compared within each panel, e.g.
                the class or the train/test split; at most 8 groups. Defaults to None.
            log (bool): Whether to put the values on a log scale; non-positive values
                are dropped. Defaults to False.
            clip (tuple[float, float] | None): Keep only values inside this quantile
                range per column, e.g. (0.01, 0.99), so a few extremes don't flatten the
                plot. Defaults to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If none of the selected columns is numeric, or if `by` has more
                than 8 groups.
        """
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
        """Plots a violin plot for every numeric column. See `plot_violinplots`.

        Args:
            by (str | None): A column whose groups are compared within each panel, e.g.
                the class or the train/test split; at most 8 groups. Defaults to None.
            log (bool): Whether to put the values on a log scale; non-positive values
                are dropped. Defaults to False.
            clip (tuple[float, float] | None): Keep only values inside this quantile
                range per column, e.g. (0.01, 0.99), so a few extremes don't flatten the
                plot. Defaults to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If none of the selected columns is numeric, or if `by` has more
                than 8 groups.
        """
        return plot_violinplots(
            self._data, self._columns, by=by, sample=self._sample, log=log, clip=clip
        )

    def category_counts(self, *, top_k: int = 10) -> Figure:
        """Plots how often each value occurs in every categorical column. See
        `plot_category_counts`.

        Args:
            top_k (int): The number of most frequent values shown per column; the rest
                are folded into one "other" bar. Defaults to 10.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If none of the columns is categorical.
        """
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
        """Plots the pairwise correlations of the numeric columns. See
        `plot_corr_heatmap`.

        Args:
            method (Literal["pearson", "spearman"]): "spearman" correlates ranks instead
                of values: it catches any monotonic relationship, not just linear ones,
                and isn't thrown off by skew or outliers. Defaults to "pearson".
            figsize (tuple[float, float]): The size of a new figure; ignored with `ax`.
                Defaults to (10, 8).
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If fewer than two of the columns are numeric.
        """
        return plot_corr_heatmap(
            self._data, figsize, self._columns, method=method, ax=ax
        )

    def missing_values(self, *, ax: Axes | None = None) -> Figure:
        """Plots the percentage of missing values per column. See `plot_missing_values`.

        Args:
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.
        """
        return plot_missing_values(self._data, self._columns, ax=ax)

    def feature_target(
        self,
        target: str,
        *,
        n_bins: int = 10,
        top_k: int = 10,
        log_x: bool | Sequence[str] = False,
    ) -> Figure:
        """Plots the mean of `target` across the values of every other column. `target`
        is always included, even if `columns` leaves it out. See `plot_feature_target`.

        Args:
            target (str): The target column; numeric or boolean.
            n_bins (int): The number of quantile bins (equally many rows each) per
                numeric feature. Defaults to 10.
            top_k (int): The number of most frequent values shown per categorical
                feature. Defaults to 10.
            log_x (bool | Sequence[str]): Features whose x axis goes on a log scale, so
                skewed features' bins don't bunch up at one end: True for every numeric
                feature, or a list of feature names. Only the axis changes, not the
                binning. A feature with a bin median <= 0 gets a symmetric log scale
                instead (linear around 0), so no bins are dropped. Defaults to False.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If `target` isn't numeric or boolean, or if `log_x` names a
                feature that isn't a numeric one of the plot.
        """
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
    """Plots the gaps, calendar profiles, autocorrelation, decomposition and rolling
    statistics of one series, plus line charts of several. The data is sorted by
    `time_col` up front, so the (partial) autocorrelation, which only sees the order of
    the values, is computed in time order too.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to plot.
        time_col (str): The timestamp column. Defaults to "ts".
        target_col (str): The series most plots analyse. Defaults to "val".
        columns (Sequence[str] | None): The columns `grid` draws. If None, all numeric
            columns but `time_col`. Defaults to None.
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
        """Plots a line chart over time per column. See `plot_timeseries_grid`.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If none of the columns is numeric.
        """
        return plot_timeseries_grid(self._data, self._time_col, self._columns)

    def gaps(
        self,
        *,
        interval: str | timedelta | None = None,
        ax: Axes | None = None,
    ) -> Figure:
        """Plots where timestamps are missing. See `plot_gaps`.

        Args:
            interval (str | timedelta | None): The expected step, as a polars duration
                like "1h" or a timedelta. If None, the most common step. Defaults to
                None.
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.
        """
        return plot_gaps(self._data, self._time_col, interval=interval, ax=ax)

    def seasonal_profile(
        self,
        *,
        rows: CalendarUnit = "weekday",
        cols: CalendarUnit | None = "hour",
        agg: Literal["mean", "median"] = "mean",
        ax: Axes | None = None,
    ) -> Figure:
        """Plots the series' typical value per calendar position. See
        `plot_seasonal_profile`.

        Args:
            rows (CalendarUnit): The calendar unit of the rows (or the x axis of the
                line): "minute", "hour", "weekday", "day", "week", "month", "quarter" or
                "year". Defaults to "weekday".
            cols (CalendarUnit | None): The calendar unit of the columns. If None, a
                line over `rows` is drawn instead of a heatmap. Defaults to "hour".
            agg (Literal["mean", "median"]): How the values per calendar position are
                summarized. Defaults to "mean".
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.
        """
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
        """Plots the series' autocorrelation function (ACF). See `plot_autocorrelation`.

        Args:
            lags (int): The number of lags to show. Defaults to 50.
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.
        """
        return plot_autocorrelation(self._data, self._target_col, lags, ax=ax)

    def partial_autocorrelation(
        self, *, lags: int = 50, ax: Axes | None = None
    ) -> Figure:
        """Plots the series' partial autocorrelation function (PACF). See
        `plot_partial_autocorrelation`.

        Args:
            lags (int): The number of lags to show; must be below half the series
                length. Defaults to 50.
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.
        """
        return plot_partial_autocorrelation(self._data, self._target_col, lags, ax=ax)

    def decomposition(
        self,
        *,
        periods: int | Sequence[int] | None = None,
        robust: bool = True,
    ) -> Figure:
        """Splits the series into trend, seasonality and residual. See
        `plot_decomposition`.

        Args:
            periods (int | Sequence[int] | None): Seasonal cycle lengths in steps, e.g.
                (24, 168) for a daily and a weekly cycle in hourly data; several periods
                use MSTL, with one seasonal panel each. If None, one natural period for
                the data's step: a day of sub-daily data, a week of daily data, a year
                of weekly or monthly data. Defaults to None.
            robust (bool): Whether to keep outliers out of the trend and seasonality, so
                they end up in the residual, where they can be seen. Defaults to True.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If the series spans fewer than two of the longest period.
        """
        return plot_decomposition(
            self._data,
            self._time_col,
            self._target_col,
            periods=periods,
            robust=robust,
        )

    def rolling_stats(self, *, window: str | int | None = None) -> Figure:
        """Plots the series' rolling mean and std. See `plot_rolling_stats`.

        Args:
            window (str | int | None): The window, as a polars duration like "30d" (by
                time) or a number of rows; either way, nothing is drawn until the first
                full window. If None, one natural seasonal period (a day of hourly data,
                a week of daily data), so the seasonal swing averages out of the mean.
                Defaults to None.

        Returns:
            Figure: The figure.
        """
        return plot_rolling_stats(
            self._data, self._time_col, self._target_col, window=window
        )


class ClassifierPlotter:
    """Plots the confusion matrix, ROC, precision-recall and calibration of one set of
    classification results. Pass `y_pred`, `y_proba` or both; the ROC, precision-recall
    and calibration plots need `y_proba`.

    Args:
        y_true (Values): The true labels or values, as a list, array, Series or
            single-column DataFrame (like the `y` frames the fit functions take).
        y_pred (Values | None): The predicted labels. If None, the confusion matrix uses
            the most probable class of `y_proba`. Defaults to None.
        y_proba (Values | None): The predicted probabilities, as from `predict_proba`:
            (n, n_classes), or (n,) for the positive class of a binary problem. Defaults
            to None.
        classes (Sequence[Any] | None): The labels of `y_proba`'s columns, in order. If
            None, the sorted distinct labels of `y_true`, which is also
            `predict_proba`'s order. Defaults to None.

    Raises:
        ValueError: If neither `y_pred` nor `y_proba` is passed.
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
        """Plots the confusion matrix, ROC, precision-recall and calibration together.
        See `plot_classification_diagnostics`.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If no `y_proba` was passed, or if the classes don't match its
                columns.
        """
        return plot_classification_diagnostics(
            self._y_true, self._proba("diagnostics"), classes=self._classes
        )

    def confusion_matrix(
        self, *, normalize: bool = True, ax: Axes | None = None
    ) -> Figure:
        """Plots how often each actual class was predicted as each class, from `y_pred`,
        or else the most probable class of `y_proba`. See `plot_confusion_matrix`.

        Args:
            normalize (bool): Whether colours and the big number show each row's share,
                so a diagonal cell is that class's recall and rare classes stay
                readable; the count is shown below it. If False, raw counts. Defaults to
                True.
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If no `y_pred` was passed and the classes don't match
                `y_proba`'s columns.
        """
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
        """Plots the ROC curves with their AUC. See `plot_roc_curves`.

        Args:
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If no `y_proba` was passed, or if the classes don't match its
                columns.
        """
        return plot_roc_curves(
            self._y_true, self._proba("roc_curves"), classes=self._classes, ax=ax
        )

    def precision_recall(self, *, ax: Axes | None = None) -> Figure:
        """Plots the precision-recall curves with their AP. See `plot_precision_recall`.

        Args:
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If no `y_proba` was passed, or if the classes don't match its
                columns.
        """
        return plot_precision_recall(
            self._y_true, self._proba("precision_recall"), classes=self._classes, ax=ax
        )

    def calibration(self, *, n_bins: int = 10, ax: Axes | None = None) -> Figure:
        """Plots the observed frequency against the predicted probability. See
        `plot_calibration`.

        Args:
            n_bins (int): The number of bins, each with equally many predictions.
                Defaults to 10.
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If no `y_proba` was passed, or if the classes don't match its
                columns.
        """
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
    """Plots the error diagnostics and the forecast view of one set of regression
    results.

    Args:
        y_true (Values): The true labels or values, as a list, array, Series or
            single-column DataFrame (like the `y` frames the fit functions take).
        y_pred (Values): The predicted values, in the same forms as `y_true`.
        ts (Values | None): The timestamps of the predictions (e.g. the test set's `ts`
            column), for `forecast`. If None, its x axis is the row number. Defaults to
            None.
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
        """Plots predicted vs actual, the residuals and their distribution. See
        `plot_regression_diagnostics`.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If the inputs differ in length.
        """
        return plot_regression_diagnostics(self._y_true, self._y_pred)

    def forecast(self) -> Figure:
        """Plots actual and predicted values over time, with the residual. See
        `plot_forecast`.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If the inputs differ in length.
        """
        return plot_forecast(self._y_true, self._y_pred, self._ts)


class ModelPlotter:
    """Plots the feature importance and learning curves of one fitted XGBoost model.

    Args:
        model (xgb.XGBModel | xgb.Booster): The fitted model.
    """

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
        """Plots each feature's share of the model's total importance. See
        `plot_feature_importance`.

        Args:
            importance_type (Literal["gain", "total_gain", "weight", "cover",
                "total_cover"]): How importance is measured. "gain" is the average loss
                reduction of a feature's splits: how useful it is when used. "weight"
                counts how often it's split on, which favours features with many
                distinct values whether or not they help. Defaults to "gain".
            top_k (int): The number of features shown; the rest are folded into one
                "other" bar. Defaults to 20.
            ax (Axes | None): Axes to draw on, e.g. to combine plots in one figure; its
                look is adapted to the style. If None, a new figure is created. Defaults
                to None.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If the model has no splits.
        """
        return plot_feature_importance(
            self._model, importance_type=importance_type, top_k=top_k, ax=ax
        )

    def learning_curves(self) -> Figure:
        """Plots the evaluation metrics per boosting round; needs a scikit-learn style
        model fit with an `eval_set`. See `plot_learning_curves`.

        Returns:
            Figure: The figure.

        Raises:
            ValueError: If the model is a Booster, or has no evaluation results.
        """
        if not isinstance(self._model, xgb.XGBModel):
            raise ValueError(
                "learning curves need a scikit-learn style model (e.g. "
                "XGBRegressor) fit with an eval_set, not a Booster"
            )
        return plot_learning_curves(self._model)
