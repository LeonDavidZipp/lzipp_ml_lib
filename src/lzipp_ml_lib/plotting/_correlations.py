import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure, SubFigure

from lzipp_ml_lib.plotting._utils import ensure_collected


def plot_corr_heatmap(
    data: pl.DataFrame | pl.LazyFrame, figsize: tuple[int, int] = (10, 8)
) -> Figure | SubFigure:
    hm = ensure_collected(data).corr()
    cols = hm.columns

    fig, _ = plt.subplots(figsize=figsize)
    sns.heatmap(  # type: ignore
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

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig
