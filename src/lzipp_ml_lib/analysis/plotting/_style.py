import functools
import math
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import matplotlib.dates as mdates
import numpy as np
import polars as pl
from cycler import cycler
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"
ACCENT_WASH = "#2a78d61f"  # the accent at ~12% opacity, for fills behind a line

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
DIVERGING = LinearSegmentedColormap.from_list(  # type: ignore
    "lzipp_diverging", ["#a3302f", "#e34948", "#f0efec", "#2a78d6", "#104281"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(  # type: ignore
    "lzipp_sequential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

_INSTALLED = set(font_manager.get_font_names())  # type: ignore[operator]
FONTS = [
    family
    for family in ("Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
    if family in _INSTALLED
] or ["DejaVu Sans"]

RC: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 110,
    "font.family": FONTS,
    "font.size": 10,
    "text.color": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.titlecolor": INK,
    "axes.labelsize": 9,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=CATEGORICAL),
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 9,
}

_P = ParamSpec("_P")
_R = TypeVar("_R")


def styled(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorates a plotting function to draw in the library's style. The style only
    applies while the function runs, so the global matplotlib settings stay as they are.

    Args:
        func (Callable[_P, _R]): The plotting function.

    Returns:
        Callable[_P, _R]: The function, drawing in the library's style.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with plt.rc_context(RC):  # type: ignore
            return func(*args, **kwargs)

    return wrapper


def panel_grid(
    n_panels: int, n_cols: int, panel_size: tuple[float, float]
) -> tuple[Figure, list[Axes]]:
    """Creates a grid of axes, one panel each, `n_cols` wide; leftover cells are hidden.

    Args:
        n_panels (int): The number of panels.
        n_cols (int): The number of columns, capped at `n_panels`.
        panel_size (tuple[float, float]): The size of one panel, in inches.

    Returns:
        tuple[Figure, list[Axes]]: The figure and its `n_panels` visible axes.

    Raises:
        ValueError: If `n_panels` is 0.
    """
    if n_panels == 0:
        raise ValueError("nothing to plot: no matching columns")
    n_cols = min(n_cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(  # type: ignore[call-overload]
        n_rows,
        n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        squeeze=False,
        layout="constrained",  # type: ignore
    )
    flat: list[Axes] = list(np.ravel(axes))  # type: ignore
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat[:n_panels]  # type: ignore


def figure_and_axes(
    ax: Axes | None, figsize: tuple[float, float]
) -> tuple[Figure, Axes]:
    """Returns `ax` and the figure it lives on, or a new single-axes figure. A passed
    `ax` was created outside the style, so its axes-level look (surface, spines, grid,
    ticks) is applied here; everything drawn on it afterwards picks up the style anyway.

    Args:
        ax (Axes | None): Axes to draw on. If None, a new figure is created.
        figsize (tuple[float, float]): The size of a new figure.

    Returns:
        tuple[Figure, Axes]: The figure and the axes to draw on.
    """
    if ax is not None:
        _restyle_axes(ax)
        return ax.get_figure(root=True), ax  # type: ignore
    fig = plt.figure(figsize=figsize, layout="constrained")  # type: ignore
    return fig, fig.add_subplot()  # type: ignore


def _restyle_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)  # type: ignore
    ax.grid(axis="both", color=GRID, linewidth=0.8, linestyle="-")  # type: ignore
    ax.xaxis.grid(False)  # type: ignore
    ax.set_axisbelow(True)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set(fontfamily=FONTS, fontsize=9, color=INK_SECONDARY)  # type: ignore
    ax.tick_params(  # type: ignore
        length=0,
        pad=6,
        colors=BASELINE,
        labelcolor=INK_SECONDARY,
        labelsize=9,
        labelfontfamily=FONTS[0],
    )


def group_palette(df: pl.DataFrame, by: str) -> dict[Any, str]:
    """Assigns a fixed colour to every group of `by`, in sorted group order, so every
    panel (and every call on the same groups) colours a group the same way.

    Args:
        df (pl.DataFrame): The data.
        by (str): The column holding the groups.

    Returns:
        dict[Any, str]: The colour of each group, keyed by group.

    Raises:
        ValueError: If there are more groups than categorical colours (8).
    """
    groups = df[by].drop_nulls().unique().sort().to_list()
    if len(groups) > len(CATEGORICAL):
        raise ValueError(
            f"by={by!r} has {len(groups)} groups, but at most {len(CATEGORICAL)} "
            "can be told apart by colour; bin or filter it first"
        )
    return dict(zip(groups, CATEGORICAL, strict=False))


def add_group_legend(fig: Figure, palette: dict[Any, str], title: str) -> None:
    """Adds one legend for the whole figure, outside the panels.

    Args:
        fig (Figure): The figure.
        palette (dict[Any, str]): The colour of each group, keyed by group.
        title (str): The legend's title.
    """
    handles = [Line2D([], [], color=c, linewidth=2) for c in palette.values()]
    fig.legend(  # type: ignore
        handles,
        [str(g) for g in palette],
        title=title,
        loc="outside upper right",
        alignment="left",
    )


def date_axis(ax: Axes) -> None:
    """Gives `ax` compact date ticks (e.g. "2024", "Feb", "Mar") instead of full
    timestamps.

    Args:
        ax (Axes): The axes.
    """
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))  # type: ignore
