import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ._style import DIVERGING, INK_SECONDARY, SURFACE, styled
from ._utils import PolarsFrame, ensure_collected, numeric_columns


@styled
def plot_corr_heatmap(
    data: PolarsFrame, figsize: tuple[float, float] = (10, 8)
) -> Figure:
    """Plots the pairwise correlations of all numeric columns.

    Only the lower triangle is drawn: the upper one mirrors it and the diagonal is
    always 1.
    """
    df = ensure_collected(data)
    cols = numeric_columns(df)
    if len(cols) < 2:
        raise ValueError("a correlation heatmap needs at least two numeric columns")
    # + 0.0 turns -0.0 into 0.0, so no "-0.00" annotations
    corr = df.select(cols).corr().to_numpy()[1:, :-1].round(2) + 0.0
    n = len(cols) - 1

    fig = plt.figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot()  # type: ignore
    sns.heatmap(  # type: ignore
        corr,
        mask=np.triu(np.ones((n, n), dtype=bool), k=1),
        annot=True,
        fmt=".2f",
        annot_kws={"size": max(6, min(10, 120 / n))},
        cmap=DIVERGING,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=2,
        linecolor=SURFACE,
        xticklabels=cols[:-1],
        yticklabels=cols[1:],
        cbar_kws={"shrink": 0.6, "ticks": [-1, -0.5, 0, 0.5, 1]},
        ax=ax,
    )
    ax.grid(visible=False)  # type: ignore
    ax.tick_params(axis="x", labelrotation=45)  # type: ignore
    ax.tick_params(axis="y", labelrotation=0)  # type: ignore
    for label in ax.get_yticklabels():  # type: ignore[operator]  # broken matplotlib-stubs
        label.set_horizontalalignment("right")
    for label in ax.get_xticklabels():  # type: ignore[operator]  # broken matplotlib-stubs
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    cbar = ax.collections[0].colorbar  # type: ignore
    if cbar is not None:
        cbar.outline.set_visible(False)  # type: ignore
        cbar.ax.tick_params(length=0, labelcolor=INK_SECONDARY)  # type: ignore
    ax.set_title("Correlations")  # type: ignore
    return fig
