import functools
import math
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

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
# fixed categorical order; it's what keeps neighbouring series colourblind-safe
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
# red <-> neutral gray <-> blue, equal steps per arm
DIVERGING = LinearSegmentedColormap.from_list(  # type: ignore
    "lzipp_diverging", ["#a3302f", "#e34948", "#f0efec", "#2a78d6", "#104281"]
)

# Concrete families rather than the generic "sans-serif": text resolves a generic
# family when it's drawn, which happens after the style's context has exited. Only
# installed ones, since matplotlib warns about every missing family on every text.
_INSTALLED = set(font_manager.get_font_names())  # type: ignore[operator]  # broken matplotlib-stubs
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
    """Apply RC while `func` runs, leaving the global rcParams alone."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with plt.rc_context(RC):  # type: ignore
            return func(*args, **kwargs)

    return wrapper


def panel_grid(
    n_panels: int, n_cols: int, panel_size: tuple[float, float]
) -> tuple[Figure, list[Axes]]:
    """A grid of `n_panels` axes, `n_cols` wide; leftover cells are hidden."""
    if n_panels == 0:
        raise ValueError("nothing to plot: no matching columns")
    n_cols = min(n_cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(  # type: ignore[call-overload]  # broken matplotlib-stubs
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
    """`ax` and the figure it lives on, or a new single-axes figure if `ax` is None.

    A passed `ax` was created outside the style, so its axes-level look (surface,
    spines, grid, ticks) is applied here; everything drawn on it afterwards picks
    up the style anyway.
    """
    if ax is not None:
        _restyle_axes(ax)
        return ax.get_figure(root=True), ax  # type: ignore
    fig = plt.figure(figsize=figsize, layout="constrained")
    return fig, fig.add_subplot()  # type: ignore


def _restyle_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)  # type: ignore
    # set the grid's look on both axes, so whichever one a plot turns on matches
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
    """A fixed colour per group of `by`, in sorted group order, so every panel (and
    every call on the same groups) colours a group the same way."""
    groups = df[by].drop_nulls().unique().sort().to_list()
    if len(groups) > len(CATEGORICAL):
        raise ValueError(
            f"by={by!r} has {len(groups)} groups, but at most {len(CATEGORICAL)} "
            "can be told apart by colour; bin or filter it first"
        )
    return dict(zip(groups, CATEGORICAL, strict=False))


def add_group_legend(fig: Figure, palette: dict[Any, str], title: str) -> None:
    """One legend for the whole figure, outside the panels."""
    handles = [Line2D([], [], color=c, linewidth=2) for c in palette.values()]
    fig.legend(  # type: ignore
        handles,
        [str(g) for g in palette],
        title=title,
        loc="outside upper right",
        alignment="left",
    )
