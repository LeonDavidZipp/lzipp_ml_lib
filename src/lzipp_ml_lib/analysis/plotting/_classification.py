from collections.abc import Sequence
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from ._style import (
    ACCENT,
    BASELINE,
    CATEGORICAL,
    GRID,
    SEQUENTIAL,
    SURFACE,
    figure_and_axes,
    panel_grid,
    styled,
)
from ._utils import Values, as_1d, as_proba, same_length

_CONFUSION = LinearSegmentedColormap.from_list(  # type: ignore
    "lzipp_confusion", ["#f4f8fd", *SEQUENTIAL(np.linspace(0.15, 1, 6))]
)

# Shared by the probability plots:
#   y_proba  the predicted probabilities, as from `predict_proba`: (n, n_classes),
#            or (n,) for the positive class of a binary problem.
#   classes  the class each probability column belongs to; by default the sorted
#            distinct values of y_true (which is also predict_proba's order).
# Binary problems get one curve, for the positive (second) class. Multiclass
# problems get one curve per class, each class against the rest (at most 8).


@styled
def plot_confusion_matrix(
    y_true: Values,
    y_pred: Values,
    *,
    classes: Sequence[Any] | None = None,
    normalize: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plots how often each actual class (rows) was predicted as each class
    (columns), with the accuracy in the title.

    With `normalize` (the default), colours and the big number are each row's
    share, so a diagonal cell is that class's recall, and rare classes stay
    readable; the count is shown below it. `classes` fixes which classes and in
    what order. With `ax`, it's drawn there.
    """
    actual = as_1d(y_true, "y_true")
    predicted = as_1d(y_pred, "y_pred")
    same_length(y_true=actual, y_pred=predicted)
    fig, ax = figure_and_axes(ax, (6, 5))
    _draw_confusion(ax, actual, predicted, classes, normalize)
    return fig


def _draw_confusion(
    ax: Axes,
    actual: np.ndarray,
    predicted: np.ndarray,
    classes: Sequence[Any] | None,
    normalize: bool,
) -> None:
    labels = list(classes) if classes is not None else _classes(actual, predicted)
    counts = confusion_matrix(actual, predicted, labels=labels)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    if normalize:
        annot = [
            [f"{s:.0%}\n{c}" for s, c in zip(rs, rc)] for rs, rc in zip(shares, counts)
        ]
    else:
        annot = [[str(c) for c in row] for row in counts]
    n = len(labels)
    sns.heatmap(  # type: ignore
        shares if normalize else counts,
        annot=np.array(annot),
        fmt="",
        annot_kws={"size": max(6, min(10, 80 / n))},
        cmap=_CONFUSION,
        vmin=0,
        vmax=1 if normalize else None,
        square=True,
        linewidths=2,
        linecolor=SURFACE,
        xticklabels=[str(c) for c in labels],
        yticklabels=[str(c) for c in labels],
        cbar=False,
        ax=ax,
    )
    ax.grid(visible=False)  # type: ignore
    ax.tick_params(axis="y", labelrotation=0)  # type: ignore
    ax.set_xlabel("Predicted")  # type: ignore
    ax.set_ylabel("Actual")  # type: ignore
    accuracy = np.trace(counts) / max(counts.sum(), 1)
    ax.set_title(f"Confusion matrix  ·  accuracy {accuracy:.1%}")  # type: ignore


@styled
def plot_roc_curves(
    y_true: Values,
    y_proba: Values,
    *,
    classes: Sequence[Any] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plots the ROC curve: the true positive rate against the false positive rate
    as the decision threshold moves, with the area under it (AUC) in the legend or
    title. The diagonal is guessing. See the module comment for the inputs. With
    `ax`, it's drawn there.
    """
    actual, proba, labels = _prepare(y_true, y_proba, classes)
    fig, ax = figure_and_axes(ax, (5.2, 5))
    _draw_roc(ax, actual, proba, labels)
    return fig


def _draw_roc(
    ax: Axes, actual: np.ndarray, proba: np.ndarray, labels: list[Any]
) -> None:
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1, zorder=1)  # type: ignore
    areas = []
    for i, label, color in _curves(labels):
        fpr, tpr, _ = roc_curve(actual == label, proba[:, i])
        area = auc(fpr, tpr)
        areas.append(area)  # type: ignore
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} ({area:.3f})")  # type: ignore
    _square_unit_axes(ax, "False positive rate", "True positive rate")
    if len(areas) == 1:  # type: ignore
        ax.set_title(f"ROC  ·  AUC {areas[0]:.3f}")  # type: ignore
    else:
        ax.set_title(f"ROC (one vs rest)  ·  macro AUC {np.mean(areas):.3f}")  # type: ignore
        ax.legend(loc="lower right", title="class (AUC)")  # type: ignore


@styled
def plot_precision_recall(
    y_true: Values,
    y_proba: Values,
    *,
    classes: Sequence[Any] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plots precision against recall as the decision threshold moves, with the
    average precision (AP) in the legend or title.

    More telling than ROC for rare positives: a classifier that guesses sits at
    the positives' share (the gray line, binary only), not at 0.5. See the module
    comment for the inputs. With `ax`, it's drawn there.
    """
    actual, proba, labels = _prepare(y_true, y_proba, classes)
    fig, ax = figure_and_axes(ax, (5.2, 5))
    _draw_precision_recall(ax, actual, proba, labels)
    return fig


def _draw_precision_recall(
    ax: Axes, actual: np.ndarray, proba: np.ndarray, labels: list[Any]
) -> None:
    scores = []
    curves = _curves(labels)
    for i, label, color in curves:
        positive = actual == label
        precision, recall, _ = precision_recall_curve(positive, proba[:, i])
        ap = average_precision_score(positive, proba[:, i])
        scores.append(ap)  # type: ignore
        ax.plot(  # type: ignore
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{label} ({ap:.3f})",
            drawstyle="steps-post",
        )
        if len(curves) == 1:
            ax.axhline(positive.mean(), color=BASELINE, linewidth=1, zorder=1)  # type: ignore
    _square_unit_axes(ax, "Recall", "Precision")
    if len(scores) == 1:  # type: ignore
        ax.set_title(f"Precision-recall  ·  AP {scores[0]:.3f}")  # type: ignore
    else:
        ax.set_title(f"Precision-recall  ·  mean AP {np.mean(scores):.3f}")  # type: ignore
        ax.legend(loc="lower left", title="class (AP)")  # type: ignore


@styled
def plot_calibration(
    y_true: Values,
    y_proba: Values,
    *,
    classes: Sequence[Any] | None = None,
    n_bins: int = 10,
    ax: Axes | None = None,
) -> Figure:
    """Plots how often the class actually occurs against its predicted probability,
    in `n_bins` bins of equally many predictions.

    On the diagonal, a predicted 70% means 70% of the time; below it the model is
    overconfident, above it underconfident. For binary problems, the gray bars at
    the bottom show where the predictions fall. See the module comment for the
    inputs. With `ax`, it's drawn there.
    """
    actual, proba, labels = _prepare(y_true, y_proba, classes)
    fig, ax = figure_and_axes(ax, (5.2, 5))
    _draw_calibration(ax, actual, proba, labels, n_bins)
    return fig


def _draw_calibration(
    ax: Axes, actual: np.ndarray, proba: np.ndarray, labels: list[Any], n_bins: int
) -> None:
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1, zorder=1)  # type: ignore
    curves = _curves(labels)
    for i, label, color in curves:
        observed, predicted = calibration_curve(
            actual == label, proba[:, i], n_bins=n_bins, strategy="quantile"
        )
        ax.plot(  # type: ignore
            predicted,
            observed,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=str(label),
        )
    if len(curves) == 1:
        # where the predictions fall, as a strip along the bottom fifth
        counts, edges = np.histogram(proba[:, curves[0][0]], bins=20, range=(0, 1))
        ax.bar(  # type: ignore
            edges[:-1],
            counts / counts.max() * 0.2,
            width=np.diff(edges),
            align="edge",
            color=GRID,
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=0,
        )
    else:
        ax.legend(loc="upper left", title="class")  # type: ignore
    _square_unit_axes(ax, "Predicted probability", "Observed frequency")
    ax.set_title("Calibration")  # type: ignore


@styled
def plot_classification_diagnostics(
    y_true: Values,
    y_proba: Values,
    *,
    classes: Sequence[Any] | None = None,
) -> Figure:
    """Plots the confusion matrix (of the most probable class), ROC,
    precision-recall and calibration together. See the module comment for the
    inputs and the single plots for how to read them.
    """
    actual, proba, labels = _prepare(y_true, y_proba, classes)
    predicted = np.asarray(labels)[proba.argmax(axis=1)]
    fig, axes = panel_grid(4, n_cols=2, panel_size=(5.4, 5))
    _draw_confusion(axes[0], actual, predicted, labels, normalize=True)
    _draw_roc(axes[1], actual, proba, labels)
    _draw_precision_recall(axes[2], actual, proba, labels)
    _draw_calibration(axes[3], actual, proba, labels, n_bins=10)
    return fig


def _prepare(
    y_true: Values, y_proba: Values, classes: Sequence[Any] | None
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    actual = as_1d(y_true, "y_true")
    proba = as_proba(y_proba)
    same_length(y_true=actual, y_proba=proba)
    labels = list(classes) if classes is not None else _classes(actual)
    if len(labels) != proba.shape[1]:
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but there are {len(labels)} "
            f"classes ({labels}); pass `classes` in predict_proba's column order"
        )
    if len(labels) > max(2, len(CATEGORICAL)):
        raise ValueError(
            f"{len(labels)} classes, but at most {len(CATEGORICAL)} curves can be "
            "told apart by colour"
        )
    return actual, proba, labels


def _classes(*arrays: np.ndarray) -> list[Any]:
    return sorted(set().union(*(a.tolist() for a in arrays)))  # type: ignore


def _curves(labels: list[Any]) -> list[tuple[int, Any, str]]:
    """(probability column, class, colour) per curve: only the positive class for
    a binary problem, every class otherwise."""
    if len(labels) == 2:
        return [(1, labels[1], ACCENT)]
    return [(i, label, CATEGORICAL[i]) for i, label in enumerate(labels)]


def _square_unit_axes(ax: Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.grid(axis="x", visible=True)  # type: ignore
    ax.set_xlabel(xlabel)  # type: ignore
    ax.set_ylabel(ylabel)  # type: ignore
