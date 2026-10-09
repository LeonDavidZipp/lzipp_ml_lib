# lzipp-ml-lib

Small helpers for ML work with [polars](https://pola.rs): EDA and evaluation
plots, leak-safe feature engineering for tabular data and time series, image
preprocessing, and XGBoost hyperparameter tuning. Splitting, metrics and
statistics come from the standard libraries listed [below](#what-to-use-from-elsewhere).

## Setup

The project uses [pixi](https://pixi.sh):

```sh
pixi install      # create the environment
pixi run test     # pytest + hypothesis
pixi run lint     # ruff check
pixi run fmt      # ruff format + fix
pixi run build    # build the wheel
```

## What's included

| Module                                        | Contents                                                   |
| --------------------------------------------- | ---------------------------------------------------------- |
| `lzipp_ml_lib.analysis`                       | `summarize`, plot functions                                |
| `lzipp_ml_lib.analysis.plotting`              | The plot functions bundled into plotter classes, `styled`  |
| `lzipp_ml_lib.transformation`                 | `TimeseriesFeatures`, `NumericFeatures`, `EncodingFeatures` |
| `lzipp_ml_lib.transformation.images`          | Image reading/writing, preprocessing steps, summaries      |
| `lzipp_ml_lib.hyperparameter`                 | `fit_xgb_regressor`, `fit_xgb_classifier`, search spaces   |

All feature classes follow the same pattern: `fit` learns everything from the
training data, the other methods apply exactly that to any frame, so train and
test are transformed the same way and nothing leaks from the test data.

### Analysis

`summarize(data)` gives one row per column: dtype, count, `null_pct`,
`n_unique`, and for numeric columns mean, std, quartiles, `skew` and
`zeros_pct`.

Every plot function takes a polars frame or arrays and returns its matplotlib
figure, drawn in the library's style (`styled` applies it to your own plot
functions). The plotter classes take the data once and expose the plots as
methods:

| Class               | Constructed from                    | Methods                                                                                                                  |
| ------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `EDAPlotter`        | a frame (`sample=` for large data)  | `histograms`, `kde`, `boxplots`, `violinplots`, `category_counts`, `corr_heatmap`, `missing_values`, `feature_target`     |
| `TimeSeriesPlotter` | a frame with `ts` / `val` columns   | `grid`, `gaps`, `seasonal_profile`, `autocorrelation`, `partial_autocorrelation`, `decomposition`, `rolling_stats`        |
| `ClassifierPlotter` | `y_true`, `y_pred` and/or `y_proba` | `diagnostics`, `confusion_matrix`, `roc_curves`, `precision_recall`, `calibration`                                       |
| `RegressionPlotter` | `y_true`, `y_pred`, optional `ts`   | `diagnostics`, `forecast`                                                                                                |
| `ModelPlotter`      | a fitted XGBoost model              | `feature_importance`, `learning_curves`                                                                                  |

The distribution plots accept `by=` (compare groups, e.g. classes or
train/test), `log=` and `clip=` (quantile range).

```python
from lzipp_ml_lib.analysis import summarize
from lzipp_ml_lib.analysis.plotting import EDAPlotter

summarize(df)
eda = EDAPlotter(df, sample=50_000)
eda.histograms(by="split")
eda.feature_target("price")
```

### Time series features

`TimeseriesFeatures` builds features for a series with a `ts` datetime and a
`val` float column. `TimeseriesFeatures.prepare(lf)` deduplicates and sorts the
frame and validates it against `TimeseriesSchema` (a
[dataframely](https://github.com/Quantco/dataframely) schema); it returns the
valid rows and the failure info.

Each method takes a LazyFrame, adds columns and keeps `ts` (unless
`drop_ts=True`), so the methods chain with `.pipe`:

| Method       | Adds                                                                                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `lag`        | Values `n` years / months / weeks / days / hours / minutes / seconds back, looked up by timestamp                      |
| `lag_diff`   | Difference between two lags of the same unit, e.g. 1 vs. 7 days back                                                   |
| `lag_ratios` | Ratio between two lags of the same unit (null when dividing by 0)                                                      |
| `rolling`    | Rolling mean / std / min / max / median over past windows                                                              |
| `ewm`        | Exponentially weighted means                                                                                           |
| `gap`        | Seconds since the latest available observation                                                                         |
| `cyclical`   | Sine/cosine encodings of month, weekday, hour, hour of week, … with optional extra harmonics                           |
| `calendar`   | Quarter, month, week, day, weekday, hour, weekend and month-start/end flags, …                                         |
| `holiday`    | Public holidays, days to/since the nearest holiday, bridge days (via [`holidays`](https://pypi.org/project/holidays/)) |
| `trend`      | Time since the start of the training data                                                                              |
| `exogenous`  | External series (weather, prices, …) matched by timestamp                                                              |

Two settings keep the features from leaking future data:

- **`horizon`** (`TimeseriesFeatures(horizon="1d")`): how far ahead you forecast.
  `lag`, `rolling`, `ewm`, `gap` and `exogenous(known_in_advance=False)` only use
  data at least that old, and `lag` rejects lags shorter than the horizon.
- **`fit(train)`**: `trend` uses values learned from the training data. Fit on
  the training rows of each fold only.

### Numeric features

`NumericFeatures().fit(train, ...)` learns the parameters of each step you ask
for; `transform(lf)` applies all of them in this order, or call a step on its
own (`suffix=` keeps the original column):

| Step              | Learned from the training data                                      |
| ----------------- | ------------------------------------------------------------------- |
| `impute`          | Fill value for nulls/NaNs (`"median"`, `"mean"` or `"zero"`)        |
| `clip`            | Winsorizing bounds (quantiles, default 1 % / 99 %)                  |
| `auto_log`        | Which columns are right-skewed (`log1p`, or signed log if negative) |
| `power_transform` | Yeo-Johnson lambda per column                                       |
| `scale`           | Scaling parameters (`scale_method`, default `"standard"`)           |

```python
num = NumericFeatures().fit(train, impute=cols, clip=cols, auto_log=True, scale=cols)
train, test = num.transform(train), num.transform(test)
```

`signed_log1p_expr(col)` is the signed log as a polars expression.

### Categorical encoding

`EncodingFeatures().fit(train, ...)` learns categories, bin edges, class rates
and shares:

| Method                  | Does                                                                                    |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `categorical_to_enum`   | Casts to `pl.Enum` of the training categories (same codes in every frame)               |
| `categorical_to_onehot` | One 0/1 column per training category; unseen categories become all zeros               |
| `numeric_to_bin`        | Bins numeric columns by the edges given in `n_bins`                                     |
| `target_encode`         | Smoothed class rate per category; pass `folds=` on the training data for out-of-fold    |
| `frequency_encode`      | Category share in the training data                                                     |

```python
enc = EncodingFeatures().fit(
    train, categorical_columns=["device"], target="y", target_encode=["city"]
)
train_enc = enc.target_encode(train, folds=5).pipe(enc.categorical_to_enum)
test_enc = enc.target_encode(test).pipe(enc.categorical_to_enum)
```

### Images

`lzipp_ml_lib.transformation.images` wraps OpenCV so images are always RGB
numpy arrays (`height x width x 3`) or grayscale (`height x width`):

- **I/O:** `read_image`, `read_images`, `write_image`, async `write_images`.
- **Steps:** `Resize` (letterbox / crop / stretch), `Scale`, `CenterCrop`,
  `Rotate`, `Flip`, `Grayscale`, `Smooth`, `Sharpen`, `Equalize` (CLAHE),
  `ToFloat`, `Normalize` (ImageNet stats by default), `ChannelsFirst` (for
  PyTorch). Subclass `Step`, or use any `image -> image` function.
- **`ImagePipeline(steps)`:** applies steps in order; `load_many(paths)` reads
  and processes files in threads, yielding them in order with bounded memory.
- **Inspection:** `summarize_images(paths)` (sizes, aspect ratios, brightness,
  unreadable files) and `plot_image_summary`, `show_images`, `show_pipeline`
  (each step's output side by side).

```python
from lzipp_ml_lib.transformation.images import (
    ChannelsFirst, ImagePipeline, Normalize, Resize, ToFloat, read_image, show_pipeline,
)

prep = ImagePipeline([Resize(224), ToFloat(), Normalize(), ChannelsFirst()])
show_pipeline(prep, read_image("cat.jpg"))
batch = list(prep.load_many(paths))
```

### Hyperparameter tuning

`fit_xgb_regressor` and `fit_xgb_classifier` run `n_trials` trials (via
[rustuna](https://pypi.org/project/rustuna/)), each fit on the training data and
scored on the test data with `metric`, then fit the final model with the best
parameters on `final_fit_data` (`"train"`, `"train_val"` or
`"train_val_test"`). They return the model and its test metrics.

Search spaces are `HyperparameterSpace`s of `IntegerDimension`,
`FloatDimension` and `CategoricalDimension`. Defaults exist for XGBoost
regressor / classifier / ranker and Prophet
(`HyperparameterSpace.default_xgb_regressor()`, …).

```python
result = fit_xgb_regressor(x_train, y_train, x_test, y_test, n_trials=50, metric="mae")
result.model, result.metrics.mae
```

## What to use from elsewhere

This library doesn't wrap these; use them directly:

| Task                                   | Use                                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| Train/test splits and cross-validation | `sklearn.model_selection.TimeSeriesSplit` (works on a polars DataFrame, not a LazyFrame) |
| Models                                 | `xgboost`, `sklearn`, `prophet`                                                          |
| Metrics                                | `sklearn.metrics`                                                                        |
| Statistical models and tests           | `statsmodels`, `scipy.stats`                                                             |
| Validating other data                  | `dataframely` schemas                                                                    |

## Example: cross-validated day-ahead forecast

Predicts hourly load in `data/energy_prices.csv` one day ahead, with five
expanding-window folds of 30 days each.

```python
import polars as pl
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from lzipp_ml_lib.transformation import TimeseriesFeatures

raw = pl.scan_csv("data/energy_prices.csv", try_parse_dates=True)
valid, failures = TimeseriesFeatures.prepare(raw)  # drops e.g. duplicate DST hours
print(failures.counts())
df = valid.collect()  # TimeSeriesSplit needs a DataFrame, not a LazyFrame

fe = TimeseriesFeatures(horizon="1d")


def build_features(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.pipe(fe.lag, daily=(1, 7), yearly=(1,))
        .pipe(fe.rolling, windows=("7d",))
        .pipe(fe.cyclical)
        .pipe(fe.calendar)
        .pipe(fe.holiday, country="US")
        .pipe(fe.trend, drop_ts=True)
        .drop_nulls()
        .collect()
    )


tscv = TimeSeriesSplit(n_splits=5, test_size=24 * 30, gap=24)
for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
    train = df[train_idx].lazy()
    fe.fit(train)  # trend learns from the training rows only

    # Test rows need history for lags/rolling windows, so build features on
    # everything up to the end of the test period, then keep only the test rows.
    train_feats = build_features(train)
    test_feats = build_features(df[: test_idx[-1] + 1].lazy()).tail(len(test_idx))

    model = XGBRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(train_feats.drop("val"), train_feats["val"])
    mae = mean_absolute_error(test_feats["val"], model.predict(test_feats.drop("val")))
    print(f"fold {fold}: MAE {mae:.0f} MW")
```

Splitting notes:

- `TimeSeriesSplit` splits by row position, so rows must be sorted by `ts`
  (`prepare` sorts them, `TimeseriesSchema` checks it). `test_size` and `gap`
  count rows, not time.
- Set `gap` to at least the horizon in rows (24 for `"1d"` on hourly data), so no
  training target falls inside the forecast window of a test row.
