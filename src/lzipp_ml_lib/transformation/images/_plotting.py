import dataclasses
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...analysis.plotting._style import (
    ACCENT,
    INK_MUTED,
    SURFACE,
    panel_grid,
    styled,
)
from ._preprocessing import (
    _IMAGENET_MEAN,  # type: ignore
    _IMAGENET_STD,  # type: ignore
    Image,
    ImagePipeline,
    StepLike,
)


@styled
def show_images(
    images: Iterable[Image],
    titles: Sequence[str] | None = None,
    *,
    ncols: int = 4,
    size: float = 3.0,
    mean: float | Sequence[float] | None = None,
    std: float | Sequence[float] | None = None,
) -> Figure:
    """Shows images in a grid, each displayed as it should look, whatever its form:

    - RGB order (the library's), height x width x channels or channels first (as
      after `ChannelsFirst`);
    - uint8, uint16 or float images, scaled by their dtype's range;
    - normalized float images (values outside [0, 1], as after `Normalize`): `mean`
      and `std` are undone, by default the ImageNet statistics; anything still out of
      range is rescaled to fill it;
    - grayscale in gray, on a fixed 0-1 scale, so a dark image looks dark instead of
      being contrast-stretched as `plt.imshow` does by default.

    Args:
        images (Iterable[Image]): The images; a generator works too, e.g. from
            `load_many`, but all of them are shown, so keep it short.
        titles (Sequence[str] | None): A title per image. Defaults to None.
        ncols (int): The number of images per row. Defaults to 4.
        size (float): The size of each panel, in inches. Defaults to 3.0.
        mean (float | Sequence[float] | None): The mean `Normalize` used, to undo it.
            If None, the ImageNet mean. Defaults to None.
        std (float | Sequence[float] | None): The std `Normalize` used, to undo it.
            If None, the ImageNet std. Defaults to None.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If there are no images, the number of titles doesn't match, or
            an image isn't a 2D or 3D array with 1, 3 or 4 channels.
    """
    images = list(images)
    if not images:
        raise ValueError("no images to show")
    if titles is not None and len(titles) != len(images):
        raise ValueError(f"{len(titles)} titles for {len(images)} images")
    shown = [_displayable(image, mean, std) for image in images]
    # panels shaped like a typical image, so wide images don't leave empty bands
    aspect = float(
        np.clip(np.median([img.shape[0] / img.shape[1] for img in shown]), 0.3, 3)
    )
    fig, axes = panel_grid(len(shown), n_cols=ncols, panel_size=(size, size * aspect))
    for i, (ax, image) in enumerate(zip(axes, shown, strict=True)):
        _draw(ax, image, mean, std)
        if titles is not None:
            ax.set_title(titles[i], fontsize=9)  # type: ignore
    return fig


@styled
def show_pipeline(
    pipeline: ImagePipeline | Sequence[StepLike],
    image: Image,
    *,
    ncols: int = 4,
    size: float = 3.0,
) -> Figure:
    """Shows an image before and after each step of a pipeline, to see what every step
    does: each panel is titled with the step and the shape, dtype and value range of
    its output. Displays images as `show_images` does.

    Args:
        pipeline (ImagePipeline | Sequence[StepLike]): The pipeline, or its steps.
        image (Image): The input image, in RGB order or grayscale.
        ncols (int): The number of panels per row. Defaults to 4.
        size (float): The size of each panel, in inches. Defaults to 3.0.

    Returns:
        Figure: The figure.
    """
    steps = pipeline.steps if isinstance(pipeline, ImagePipeline) else list(pipeline)
    stages = [("original", image)]
    for step in steps:
        image = step(image)
        stages.append((_describe(step), image))
    fig, axes = panel_grid(len(stages), n_cols=ncols, panel_size=(size, size))
    for ax, (name, stage) in zip(axes, stages, strict=True):
        _draw(ax, stage, None, None)
        ax.set_title(f"{name}\n{_shape_label(stage)}", fontsize=9)  # type: ignore
    return fig


@styled
def plot_image_summary(summary: pl.DataFrame) -> Figure:
    """Plots a `summarize_images` frame: the image sizes, aspect ratios and
    brightness, with the counts of unreadable, grayscale and alpha images in the
    title.

    Args:
        summary (pl.DataFrame): The output of `summarize_images`.

    Returns:
        Figure: The figure.

    Raises:
        ValueError: If no image in `summary` is readable.
    """
    readable = summary.filter(pl.col("readable"))
    if readable.is_empty():
        raise ValueError("no readable images to plot")
    fig, (sizes, aspects, brightness) = panel_grid(3, n_cols=3, panel_size=(4.4, 3.6))

    counts = readable.group_by("width", "height").len()
    sizes.scatter(  # type: ignore
        counts["width"],
        counts["height"],
        s=20 + 180 * counts["len"] / counts["len"].max(),
        color=ACCENT,
        alpha=0.6,
        linewidths=0,
    )
    sizes.set_xlabel("Width (px)")  # type: ignore
    sizes.set_ylabel("Height (px)")  # type: ignore
    sizes.grid(axis="x", visible=True)  # type: ignore
    n_sizes = counts.height
    sizes.set_title(f"Sizes  ·  {n_sizes} distinct")  # type: ignore

    for ax, column, title in [
        (aspects, "aspect", "Aspect ratio (width / height)"),
        (brightness, "brightness", "Brightness (mean gray, 0-255)"),
    ]:
        ax.hist(  # type: ignore
            readable[column].to_numpy(),
            bins="auto",
            color=ACCENT,
            edgecolor=SURFACE,
            linewidth=0.8,
        )
        ax.set_title(title)  # type: ignore
        ax.set_ylabel("Images")  # type: ignore

    notes = [f"{readable.height} images"]
    if unreadable := summary.height - readable.height:
        notes.append(f"{unreadable} unreadable")
    for channels, name in [(1, "grayscale"), (4, "with alpha")]:
        if n := readable.filter(pl.col("channels") == channels).height:
            notes.append(f"{n} {name}")
    fig.suptitle("  ·  ".join(notes), x=0.01, ha="left", fontsize=10, color=INK_MUTED)  # type: ignore
    return fig


# ---- display -------------------------------------------------------------------------
def _draw(
    ax: Axes,
    image: Image,
    mean: float | Sequence[float] | None,
    std: float | Sequence[float] | None,
) -> None:
    shown = _displayable(image, mean, std)
    if shown.ndim == 2:
        # a fixed range: imshow would otherwise stretch each image's contrast
        ax.imshow(shown, cmap="gray", vmin=0, vmax=1)  # type: ignore
    else:
        ax.imshow(shown)  # type: ignore
    ax.set_axis_off()


def _displayable(
    image: Image,
    mean: float | Sequence[float] | None = None,
    std: float | Sequence[float] | None = None,
) -> np.ndarray:
    """The image as height x width (x 3 or 4) floats in [0, 1], the way it should
    look."""
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"an image has 2 or 3 dimensions, got shape {image.shape}")
    if (
        image.ndim == 3
        and image.shape[0] in (1, 3, 4)
        and image.shape[2] not in (1, 3, 4)
    ):
        image = image.transpose(1, 2, 0)  # channels first, as after ChannelsFirst
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError(f"an image has 1, 3 or 4 channels, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float64) / np.iinfo(image.dtype).max
    values = image.astype(np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < -1e-3 or finite.max() > 1 + 1e-3):
        values = _undo_normalize(values, mean, std)
        finite = values[np.isfinite(values)]
        if finite.size and (finite.min() < -1e-3 or finite.max() > 1 + 1e-3):
            low, high = finite.min(), finite.max()
            values = (values - low) / (high - low) if high > low else values * 0
    return np.clip(np.nan_to_num(values), 0, 1)


def _undo_normalize(
    values: np.ndarray,
    mean: float | Sequence[float] | None,
    std: float | Sequence[float] | None,
) -> np.ndarray:
    channels = values.shape[2] if values.ndim == 3 else 1
    if mean is None and std is None and channels != 3:
        return values  # the ImageNet statistics are for RGB
    mean_ = np.asarray(_IMAGENET_MEAN if mean is None else mean, dtype=np.float64)
    std_ = np.asarray(_IMAGENET_STD if std is None else std, dtype=np.float64)
    if values.ndim == 2:
        mean_, std_ = mean_.ravel()[0], std_.ravel()[0]
    return values * std_ + mean_


# ---- labels --------------------------------------------------------------------------
def _describe(step: StepLike) -> str:
    """A short name for a step: its class with the arguments that differ from the
    defaults, e.g. `Resize(224)` or `Smooth(sigma=2.0)`."""
    if isinstance(step, ImagePipeline):
        return f"ImagePipeline({len(step.steps)} steps)"
    if dataclasses.is_dataclass(step) and not isinstance(step, type):
        args: list[str] = []
        for field in dataclasses.fields(step):
            value = getattr(step, field.name)
            if field.default is dataclasses.MISSING:  # required: shown positionally
                args.append(repr(value))
            elif value != field.default:
                args.append(f"{field.name}={value!r}")
        return f"{type(step).__name__}({', '.join(args)})"
    name = getattr(step, "__name__", type(step).__name__)
    return "lambda" if name == "<lambda>" else name


def _shape_label(image: Image) -> str:
    shape = "×".join(str(side) for side in image.shape)
    low, high = float(np.min(image)), float(np.max(image))
    return f"{shape} {image.dtype} [{low:.3g}, {high:.3g}]"
