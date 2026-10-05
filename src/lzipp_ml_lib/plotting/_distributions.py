import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._style import ACCENT, ACCENT_WASH, INK_MUTED, panel_grid, styled
from ._utils import PolarsFrame, ensure_collected, numeric_columns


@styled
def plot_kde(data: PolarsFrame) -> Figure:
    """Plots a density curve for every numeric column."""
    df = ensure_collected(data)
    cols = numeric_columns(df)
    fig, axes = panel_grid(len(cols), n_cols=2, panel_size=(6, 3.2))
    for ax, col in zip(axes, cols, strict=True):
        sns.kdeplot(data=df, x=col, ax=ax, fill=True, color=ACCENT, alpha=0.12, lw=0)
        sns.kdeplot(data=df, x=col, ax=ax, color=ACCENT, linewidth=2)
        ax.set_title(col)  # type: ignore
        ax.set_xlabel("")  # type: ignore
        ax.set_ylabel("Density")  # type: ignore
        ax.set_ylim(bottom=0)
    return fig


@styled
def plot_boxplots(data: PolarsFrame) -> Figure:
    """Plots a boxplot for every numeric column, to show spread and outliers."""
    df = ensure_collected(data)
    cols = numeric_columns(df)
    fig, axes = panel_grid(len(cols), n_cols=3, panel_size=(5, 2.2))
    for ax, col in zip(axes, cols, strict=True):
        sns.boxplot(
            data=df,
            x=col,
            ax=ax,
            width=0.4,
            linewidth=1.25,
            linecolor=ACCENT,
            boxprops={"facecolor": ACCENT_WASH},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": ACCENT_WASH,
                "markeredgecolor": ACCENT,
                "markeredgewidth": 0.8,
            },
        )
        _style_single_distribution(ax, col)
    return fig


@styled
def plot_violinplots(data: PolarsFrame) -> Figure:
    """Plots a violin plot for every numeric column, to show the shape of its
    distribution."""
    df = ensure_collected(data)
    cols = numeric_columns(df)
    fig, axes = panel_grid(len(cols), n_cols=3, panel_size=(5, 2.2))
    for ax, col in zip(axes, cols, strict=True):
        sns.violinplot(
            data=df,
            x=col,
            ax=ax,
            color=ACCENT_WASH,
            linecolor=ACCENT,
            linewidth=1.25,
            inner="quart",
            inner_kws={"color": ACCENT, "linewidth": 1},
            saturation=1,
        )
        _style_single_distribution(ax, col)
    return fig


def _style_single_distribution(ax: Axes, col: str) -> None:
    """One horizontal distribution per panel: vertical grid only, no y axis."""
    ax.set_title(col)  # type: ignore
    ax.set_xlabel("")  # type: ignore
    ax.set_yticks([])  # type: ignore[operator]  # broken matplotlib-stubs
    ax.grid(axis="y", visible=False)  # type: ignore
    ax.grid(axis="x", visible=True)  # type: ignore
    ax.tick_params(axis="x", colors=INK_MUTED, labelcolor=INK_MUTED)  # type: ignore
