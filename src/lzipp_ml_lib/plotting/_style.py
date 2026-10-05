import functools
import math
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import numpy as np
from cycler import cycler
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

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
