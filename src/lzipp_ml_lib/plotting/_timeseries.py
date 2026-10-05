import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf  # type: ignore

from ._utils import PolarsFrame, ensure_collected


def plot_timeseries_grid(data: PolarsFrame, time_col: str = "ts") -> Figure:
    """Plots a line chart over time for every numeric column in the dataset."""
    df = ensure_collected(data)

    numeric_cols = [
        col for col in df.columns if col != time_col and df[col].dtype.is_numeric()
    ]

    n_cols = 1
    n_rows = len(numeric_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for i, col in enumerate(numeric_cols):
        sns.lineplot(data=df, x=time_col, y=col, ax=axes[i], linewidth=1)
        axes[i].set_title(f"Time Series: {col}")
        axes[i].set_ylabel("Value")

    plt.tight_layout()
    plt.show()
    return fig


def plot_missing_values(data: PolarsFrame) -> Figure:
    """Plots a bar chart of the percentage of missing values per column."""
    df = ensure_collected(data)

    null_counts = df.null_count().transpose(include_header=True)
    null_counts.columns = ["column", "null_count"]

    null_counts = null_counts.with_columns(
        missing_pct=(pl.col("null_count") / df.height) * 100
    ).sort("missing_pct", descending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df.columns) * 0.3)))

    sns.barplot(data=null_counts, x="missing_pct", y="column", ax=ax, palette="Reds_r")

    ax.set_title("Missing Values Percentage by Feature")  # type: ignore
    ax.set_xlabel("% Missing")  # type: ignore
    ax.set_ylabel("")  # type: ignore

    plt.tight_layout()
    plt.show()
    return fig


def plot_autocorrelation(
    data: PolarsFrame, target_col: str = "y", lags: int = 50
) -> Figure:
    """Plots the autocorrelation function (ACF) for the target variable."""
    df = ensure_collected(data)
    fig, ax = plt.subplots(figsize=(12, 4))
    series = df.get_column(target_col).drop_nulls().to_numpy()

    plot_acf(series, lags=lags, ax=ax, alpha=0.05)
    ax.set_title(f"Autocorrelation of {target_col} (up to {lags} lags)")  # type: ignore
    ax.set_xlabel("Lag")  # type: ignore
    ax.set_ylabel("Correlation Coefficient")  # type: ignore

    plt.tight_layout()
    plt.show()
    return fig
