"""Deterministic image preprocessing on top of OpenCV.

Images are numpy arrays in RGB order (height x width x 3), or grayscale (height x
width), of dtype uint8, uint16 or float32. Every step takes an image and returns a
new one; `ImagePipeline` applies several in order, to one image or many. The steps
take care of OpenCV's quiet traps: BGR order, interpolation, aspect ratio, rotation
cropping and value overflow.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from dataclasses import dataclass
from itertools import islice
from os import PathLike
from pathlib import Path
from typing import Literal

import aiofiles
import cv2
import numpy as np

Image = np.ndarray
Interpolation = Literal["auto", "nearest", "linear", "cubic", "area", "lanczos"]

_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


# ---- reading and writing -------------------------------------------------------------
def read_image(path: str | PathLike[str], grayscale: bool = False) -> Image:
    """Reads an image file as RGB (or grayscale), unlike `cv2.imread`'s BGR.

    Works with non-ASCII paths, and raises instead of returning None when the file
    can't be read. Alpha channels are dropped; 16-bit images are read as 8-bit.

    Args:
        path (str | PathLike[str]): The image file.
        grayscale (bool): Whether to read the image as grayscale (height x width).
            Defaults to False.

    Returns:
        Image: The image, as a uint8 array: height x width x 3 in RGB order, or
            height x width if `grayscale`.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file isn't an image OpenCV can decode.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no image at {path}")
    data = np.fromfile(path, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flag)
    if image is None:
        raise ValueError(f"can't decode {path} as an image")
    return image if grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_images(
    paths: Sequence[str | PathLike[str]], grayscale: bool = False
) -> Generator[Image]:
    for path in paths:
        yield read_image(path, grayscale)


def write_image(path: str | PathLike[str], image: Image) -> None:
    """Writes an RGB (or grayscale) image to a file; the format follows the suffix.

    Args:
        path (str | PathLike[str]): The file to write, e.g. `"out.png"`.
        image (Image): The image, in RGB order or grayscale.

    Raises:
        ValueError: If OpenCV can't encode the image in the suffix's format.
    """
    path = Path(path)
    path.write_bytes(_encode(path.suffix, image))


async def write_images(
    paths: Sequence[str | PathLike[str]],
    images: Sequence[Image],
    max_concurrency: int = 8,
) -> None:
    """Writes many RGB (or grayscale) images to files concurrently; each file's
    format follows its suffix, as in `write_image`.

    Encoding (e.g. PNG compression) is the slow part, not the disk; it runs in
    threads, which OpenCV lets work in parallel, and the encoded bytes are written
    with `aiofiles`, so neither blocks the event loop. At most `max_concurrency`
    images are encoded or written at once, which also bounds the memory the encoded
    bytes take up. If one image fails, the others are cancelled and the error is
    raised.

    It's a coroutine: `await write_images(...)` in async code and in notebooks, or
    `asyncio.run(write_images(...))` in plain scripts.

    Args:
        paths (Sequence[str | PathLike[str]]): The files to write, one per image.
        images (Sequence[Image]): The images, in RGB order or grayscale.
        max_concurrency (int): The most images encoded or written at once. Defaults
            to 8.

    Raises:
        ValueError: If `paths` and `images` differ in length, if `max_concurrency`
            is below 1, or if OpenCV can't encode an image in its suffix's format.
        FileNotFoundError: If a file's directory doesn't exist.
    """
    if len(paths) != len(images):
        raise ValueError(f"{len(paths)} paths but {len(images)} images")
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be at least 1, got {max_concurrency}")
    slots = asyncio.Semaphore(max_concurrency)

    async def write(path: Path, image: Image) -> None:
        async with slots:
            data = await asyncio.to_thread(_encode, path.suffix, image)
            async with aiofiles.open(path, "wb") as file:
                await file.write(data)

    try:
        async with asyncio.TaskGroup() as tasks:
            for path, image in zip(paths, images, strict=True):
                tasks.create_task(write(Path(path), image))
    except ExceptionGroup as group:
        raise group.exceptions[0] from group


def _encode(suffix: str, image: Image) -> bytes:
    """The image encoded in the format of `suffix` (e.g. ".png"), as OpenCV
    expects it: BGR."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if _is_color(image) else image
    try:
        ok, encoded = cv2.imencode(suffix, bgr)
    except cv2.error as error:
        raise ValueError(f"can't encode the image as {suffix!r}") from error
    if not ok:
        raise ValueError(f"can't encode the image as {suffix!r}")
    return encoded.tobytes()


# ---- steps ---------------------------------------------------------------------------
class Step(ABC):
    """Base class of the preprocessing steps: an image goes in, a new one comes out.

    Subclasses implement `transform`; calling a step calls it, so `step(image)` and
    `step.transform(image)` are the same. Plain functions from image to image work in
    an `ImagePipeline` too, without subclassing.
    """

    @abstractmethod
    def transform(self, image: Image) -> Image:
        """Applies the step to an image.

        Args:
            image (Image): The image, in RGB order or grayscale.

        Returns:
            Image: The transformed image, as a new array.
        """

    def __call__(self, image: Image) -> Image:
        return self.transform(image)


StepLike = Step | Callable[[Image], Image]


@dataclass(frozen=True)
class Resize(Step):
    """Resizes to a fixed size, by default keeping the aspect ratio.

    Args:
        size (int | tuple[int, int]): The output size, as `(width, height)`, or one
            int for a square.
        keep_aspect (Literal["letterbox", "crop", "stretch"]): How to fit an image
            of another aspect ratio. `"letterbox"` scales it to fit inside and pads
            the rest with `pad_value`; `"crop"` scales it to cover the size and cuts
            off the overflow, centred; `"stretch"` distorts it to the size. Defaults
            to `"letterbox"`.
        interpolation (Interpolation): How pixels are resampled. `"auto"` uses
            `"area"` when shrinking (no aliasing artefacts) and `"cubic"` when
            enlarging. Defaults to `"auto"`.
        pad_value (int | float): The value letterbox padding is filled with.
            Defaults to 0 (black).
    """

    size: int | tuple[int, int]
    keep_aspect: Literal["letterbox", "crop", "stretch"] = "letterbox"
    interpolation: Interpolation = "auto"
    pad_value: int | float = 0

    def transform(self, image: Image) -> Image:
        width, height = (
            (self.size, self.size) if isinstance(self.size, int) else self.size
        )
        if width <= 0 or height <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        h, w = image.shape[:2]
        if self.keep_aspect == "stretch":
            return _resize(image, width, height, self.interpolation)
        fit = min if self.keep_aspect == "letterbox" else max
        scale = fit(width / w, height / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = _resize(image, new_w, new_h, self.interpolation)
        if self.keep_aspect == "crop":
            return _center_crop(resized, width, height)
        return _pad_center(resized, width, height, self.pad_value)


@dataclass(frozen=True)
class Scale(Step):
    """Scales by a factor, keeping the aspect ratio.

    Args:
        factor (float): The scale factor, e.g. 0.5 for half the size.
        interpolation (Interpolation): How pixels are resampled; `"auto"` uses
            `"area"` when shrinking and `"cubic"` when enlarging. Defaults to
            `"auto"`.
    """

    factor: float
    interpolation: Interpolation = "auto"

    def transform(self, image: Image) -> Image:
        if self.factor <= 0:
            raise ValueError(f"factor must be positive, got {self.factor}")
        h, w = image.shape[:2]
        new_w, new_h = max(1, round(w * self.factor)), max(1, round(h * self.factor))
        return _resize(image, new_w, new_h, self.interpolation)


@dataclass(frozen=True)
class Grayscale(Step):
    """Converts RGB to grayscale with the standard luminance weights
    (0.299 R + 0.587 G + 0.114 B). Grayscale images pass through.

    Args:
        keep_channels (bool): Whether to return 3 identical channels instead of one,
            for steps or models that expect color input. Defaults to False.
    """

    keep_channels: bool = False

    def transform(self, image: Image) -> Image:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if _is_color(image) else image
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if self.keep_channels else gray


@dataclass(frozen=True)
class Smooth(Step):
    """Smooths (blurs) the image, e.g. to remove sensor noise before extracting
    features.

    Args:
        sigma (float): The strength: the Gaussian's standard deviation in pixels,
            or for `"median"`, the radius of the window (rounded). Defaults to 1.0.
        method (Literal["gaussian", "median"]): `"gaussian"` blurs evenly;
            `"median"` removes salt-and-pepper noise while keeping edges sharp.
            Defaults to `"gaussian"`.
    """

    sigma: float = 1.0
    method: Literal["gaussian", "median"] = "gaussian"

    def transform(self, image: Image) -> Image:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.method == "median":
            size = 2 * max(1, round(self.sigma)) + 1
            return cv2.medianBlur(image, size)
        return cv2.GaussianBlur(image, (0, 0), self.sigma)


@dataclass(frozen=True)
class Sharpen(Step):
    """Sharpens with an unsharp mask: adds back `amount` times the difference to a
    blurred copy. Computed in floating point and clipped to the dtype's range, so
    bright pixels don't wrap around to dark ones.

    Args:
        amount (float): How strongly edges are enhanced; 0 is no change. Defaults to
            1.0.
        sigma (float): The blur's standard deviation in pixels; larger values
            sharpen coarser structures. Defaults to 1.0.
    """

    amount: float = 1.0
    sigma: float = 1.0

    def transform(self, image: Image) -> Image:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        source = image.astype(np.float32)
        blurred = cv2.GaussianBlur(source, (0, 0), self.sigma)
        sharpened = source + self.amount * (source - blurred)
        return _clip_to_dtype(sharpened, image.dtype)


@dataclass(frozen=True)
class Rotate(Step):
    """Rotates by an angle, counter-clockwise for positive angles.

    Multiples of 90° are exact (no resampling). Other angles enlarge the canvas so
    no corner is cut off (unless `expand=False`), and fill the new area with
    `fill_value`.

    Args:
        angle (float): The angle in degrees; positive is counter-clockwise.
        expand (bool): Whether to enlarge the canvas to fit the whole rotated image.
            If False, the size is kept and the corners are cut off. Defaults to True.
        fill_value (int | float): The value the area outside the image is filled
            with. Defaults to 0 (black).
        interpolation (Interpolation): How pixels are resampled; `"auto"` means
            `"linear"`. Defaults to `"auto"`.
    """

    angle: float
    expand: bool = True
    fill_value: int | float = 0
    interpolation: Interpolation = "auto"

    def transform(self, image: Image) -> Image:
        quarter_turns = self.angle / 90
        if quarter_turns == round(quarter_turns):
            return np.ascontiguousarray(np.rot90(image, k=round(quarter_turns) % 4))
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), self.angle, 1.0)
        out_w, out_h = w, h
        if self.expand:
            cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
            out_w, out_h = round(h * sin + w * cos), round(h * cos + w * sin)
            matrix[0, 2] += out_w / 2 - w / 2
            matrix[1, 2] += out_h / 2 - h / 2
        flags = (
            cv2.INTER_LINEAR
            if self.interpolation == "auto"
            else _INTERPOLATIONS[self.interpolation]
        )
        return cv2.warpAffine(
            image,
            matrix,
            (out_w, out_h),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_fill(image, self.fill_value),
        )


@dataclass(frozen=True)
class Flip(Step):
    """Mirrors the image.

    Args:
        horizontal (bool): Whether to mirror left-right. Defaults to True.
        vertical (bool): Whether to mirror top-bottom. Defaults to False.
    """

    horizontal: bool = True
    vertical: bool = False

    def transform(self, image: Image) -> Image:
        if self.horizontal:
            image = image[:, ::-1]
        if self.vertical:
            image = image[::-1]
        return np.ascontiguousarray(image)


@dataclass(frozen=True)
class CenterCrop(Step):
    """Cuts a centred region out of the image.

    Args:
        size (int | tuple[int, int]): The region's size, as `(width, height)`, or one
            int for a square; at most the image's size.
    """

    size: int | tuple[int, int]

    def transform(self, image: Image) -> Image:
        width, height = (
            (self.size, self.size) if isinstance(self.size, int) else self.size
        )
        h, w = image.shape[:2]
        if width > w or height > h:
            raise ValueError(f"crop size {width}x{height} exceeds the image's {w}x{h}")
        return _center_crop(image, width, height)


@dataclass(frozen=True)
class Equalize(Step):
    """Spreads the brightness over the full range, to even out lighting and
    contrast between images. Color images are equalized on their lightness only,
    so colors don't shift. Needs a uint8 image.

    Args:
        adaptive (bool): Whether to equalize locally (CLAHE), which evens out uneven
            lighting within an image and doesn't over-amplify noise. If False,
            the whole image is equalized at once. Defaults to True.
        clip_limit (float): For `adaptive`, how strongly contrast may be amplified;
            higher is stronger. Defaults to 2.0.
        tile_grid (int): For `adaptive`, the number of tiles per side the image is
            equalized in. Defaults to 8.
    """

    adaptive: bool = True
    clip_limit: float = 2.0
    tile_grid: int = 8

    def transform(self, image: Image) -> Image:
        if image.dtype != np.uint8:
            raise ValueError(f"Equalize needs a uint8 image, got {image.dtype}")
        if self.adaptive:
            clahe = cv2.createCLAHE(self.clip_limit, (self.tile_grid, self.tile_grid))
            equalize: Callable[[Image], Image] = clahe.apply
        else:
            equalize = cv2.equalizeHist
        if not _is_color(image):
            return equalize(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[..., 0] = equalize(np.ascontiguousarray(lab[..., 0]))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


@dataclass(frozen=True)
class ToFloat(Step):
    """Converts integer images to float32 in [0, 1], e.g. before feature extraction
    or a model; float images pass through."""

    def transform(self, image: Image) -> Image:
        if np.issubdtype(image.dtype, np.integer):
            return image.astype(np.float32) / np.iinfo(image.dtype).max
        return image.astype(np.float32)


# ---- pipeline ------------------------------------------------------------------------
class ImagePipeline(Step):
    """Applies a fixed sequence of preprocessing steps, the same way to every image:

        prep = ImagePipeline([Resize(224), Grayscale(), Smooth(1.0), Sharpen(0.5)])
        for image in prep.load_many(paths):  # one image in memory at a time
            ...
        image = prep(some_array)

    A step is a `Step`, or any function from image to image, so custom ones fit in
    too. A pipeline is a `Step` itself, so pipelines can be nested.

    Args:
        steps (Sequence[StepLike]): The steps, in the order they're applied.
    """

    def __init__(self, steps: Sequence[StepLike]):
        self.steps = list(steps)

    def transform(self, image: Image) -> Image:
        """Applies the steps to an image.

        Args:
            image (Image): The image, in RGB order or grayscale.

        Returns:
            Image: The processed image.
        """
        for step in self.steps:
            image = step(image)
        return image

    def load(self, path: str | PathLike[str], grayscale: bool = False) -> Image:
        """Reads an image file and applies the steps.

        Args:
            path (str | PathLike[str]): The image file.
            grayscale (bool): Whether to read the image as grayscale. Defaults to
                False.

        Returns:
            Image: The processed image.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file isn't an image OpenCV can decode.
        """
        return self(read_image(path, grayscale=grayscale))

    def load_many(
        self,
        paths: Iterable[str | PathLike[str]],
        grayscale: bool = False,
        workers: int | None = None,
        chunk_size: int = 4,
    ) -> Generator[Image]:
        """Like `load_many_sequential`, but reads and processes several images at
        once in threads, while still yielding them in the order of `paths`.

        Images are processed in chunks of `chunk_size` per task, and at most about
        `2 * workers` chunks are in flight at once, so memory stays bounded however
        many paths there are. Stopping early, or an error, cancels the remaining work.

        Threads suit the built-in steps: OpenCV releases Python's GIL while decoding
        and filtering, so threads run in parallel, start instantly and share memory.
        Custom steps that hold the GIL (pure-Python or numpy-loop code) gain little.

        Args:
            paths (Iterable[str | PathLike[str]]): The image files; may itself be
                lazy, e.g. `Path("images").glob("*.png")`.
            grayscale (bool): Whether to read the images as grayscale. Defaults to
                False.
            workers (int | None): The number of threads. If None, the number of CPUs.
                Defaults to None.
            chunk_size (int): The number of images per task; larger chunks lower
                the per-task overhead, smaller ones spread the work more evenly.
                Defaults to 4.

        Yields:
            Image: The processed images, in the order of `paths`.

        Raises:
            ValueError: If `workers` or `chunk_size` is below 1, or when iterating
                reaches a file that isn't an image OpenCV can decode.
            FileNotFoundError: When iterating reaches a file that doesn't exist.
        """
        workers = workers if workers is not None else os.cpu_count() or 1
        if workers < 1 or chunk_size < 1:
            raise ValueError(
                f"workers and chunk_size must be at least 1, got {workers} and "
                f"{chunk_size}"
            )
        executor = ThreadPoolExecutor(workers)
        pending: deque[Future[list[Image]]] = deque()
        try:
            for chunk in _chunks(paths, chunk_size):
                pending.append(
                    executor.submit(_load_chunk, self.steps, chunk, grayscale)
                )
                if len(pending) >= 2 * workers:
                    yield from pending.popleft().result()
            while pending:
                yield from pending.popleft().result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def load_many_sequential(
        self, paths: Iterable[str | PathLike[str]], grayscale: bool = False
    ) -> Generator[Image]:
        """Reads image files and applies the steps to each, one at a time: only one
        image is in memory at once, however many paths there are. Wrap it in `list()`
        to get them all at once.

        Args:
            paths (Iterable[str | PathLike[str]]): The image files; may itself be
                lazy, e.g. `Path("images").glob("*.png")`.
            grayscale (bool): Whether to read the images as grayscale. Defaults to
                False.

        Yields:
            Image: The processed images, in the order of `paths`.

        Raises:
            FileNotFoundError: When iterating reaches a file that doesn't exist.
            ValueError: When iterating reaches a file that isn't an image OpenCV can
                decode.
        """
        for path in paths:
            yield self.load(path, grayscale=grayscale)


# ---- helpers -------------------------------------------------------------------------
def _load_chunk(
    steps: Sequence[StepLike], paths: Sequence[str | PathLike[str]], grayscale: bool
) -> list[Image]:
    """Loads and processes a chunk of images: the unit of work of one thread."""
    pipeline = ImagePipeline(steps)
    return [pipeline.load(path, grayscale=grayscale) for path in paths]


def _chunks(
    items: Iterable[str | PathLike[str]], size: int
) -> Iterator[list[str | PathLike[str]]]:
    """`items` in lists of `size` (the last one shorter), consumed lazily."""
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def _is_color(image: Image) -> bool:
    return image.ndim == 3 and image.shape[2] == 3


def _resize(
    image: Image, width: int, height: int, interpolation: Interpolation
) -> Image:
    if interpolation == "auto":
        shrinking = width * height < image.shape[0] * image.shape[1]
        flag = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    else:
        flag = _INTERPOLATIONS[interpolation]
    resized = cv2.resize(image, (width, height), interpolation=flag)
    # cubic overshoots at edges; keep integer images within their range
    return _clip_to_dtype(resized, image.dtype) if flag == cv2.INTER_CUBIC else resized


def _center_crop(image: Image, width: int, height: int) -> Image:
    h, w = image.shape[:2]
    top, left = (h - height) // 2, (w - width) // 2
    return np.ascontiguousarray(image[top : top + height, left : left + width])


def _pad_center(image: Image, width: int, height: int, value: int | float) -> Image:
    h, w = image.shape[:2]
    top, left = (height - h) // 2, (width - w) // 2
    return cv2.copyMakeBorder(
        image,
        top,
        height - h - top,
        left,
        width - w - left,
        cv2.BORDER_CONSTANT,
        value=_fill(image, value),
    )


def _fill(image: Image, value: int | float) -> tuple[float, ...]:
    """A border value OpenCV applies to every channel."""
    channels = image.shape[2] if image.ndim == 3 else 1
    return (float(value),) * max(channels, 1)


def _clip_to_dtype(values: np.ndarray, dtype: np.dtype) -> Image:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)
