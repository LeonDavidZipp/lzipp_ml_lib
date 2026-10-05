import math

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from ._utils import PolarsFrame, ensure_collected


def plot_kde(data: PolarsFrame) -> Figure:
    df = ensure_collected(data)
    n_cols = 2
    n_rows = math.ceil(len(df.columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    i = 0
    for i, col in enumerate(df.columns):
        sns.kdeplot(data=df, x=col, ax=axes[i], fill=True)
        axes[i].set_title(f"Density Curve of {col}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


def plot_boxplots(data: PolarsFrame) -> Figure:
    """Plots boxplots for all numeric columns to identify outliers and spread."""
    df = data.collect() if isinstance(data, pl.LazyFrame) else data
    numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]

    n_cols = 3
    n_rows = math.ceil(len(numeric_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x=col, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribution & Outliers: {col}")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


def plot_violinplots(data: PolarsFrame) -> Figure:
    """Plots boxplots for all numeric columns to identify outliers and spread."""
    df = data.collect() if isinstance(data, pl.LazyFrame) else data
    numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]

    n_cols = 3
    n_rows = math.ceil(len(numeric_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.violinplot(data=df, x=col, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribution & Outliers: {col}")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig
