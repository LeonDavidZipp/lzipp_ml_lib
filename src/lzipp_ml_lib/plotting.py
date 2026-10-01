import math

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure, SubFigure


def plot_corr_heatmap(data: pl.DataFrame | pl.LazyFrame) -> Figure | SubFigure:
    hm = data.collect().corr() if isinstance(data, pl.LazyFrame) else data.corr()
    cols = hm.columns

    ax = sns.heatmap(  # type: ignore
        hm,
        annot=True,
        annot_kws={"size": 6},
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=cols,
        yticklabels=cols,
    )

    plt.xticks(rotation=45, ha="right")  # type: ignore
    plt.show()  # type: ignore
    return ax.figure


def plot_kde(data: pl.DataFrame | pl.LazyFrame) -> Figure:
    df = data.collect() if isinstance(data, pl.LazyFrame) else data
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
    plt.show()  # type: ignore
    return fig
