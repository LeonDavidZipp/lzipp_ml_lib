from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

_SCHEMA: dict[str, Any] = {
    "path": pl.String,
    "readable": pl.Boolean,
    "error": pl.String,
    "width": pl.UInt32,
    "height": pl.UInt32,
    "aspect": pl.Float64,
    "channels": pl.UInt8,
    "dtype": pl.String,
    "file_kb": pl.Float64,
    "brightness": pl.Float64,
    "contrast": pl.Float64,
}


def summarize_images(
    paths: Iterable[str | PathLike[str]], *, workers: int | None = None
) -> pl.DataFrame:
    """Summarizes image files in one row each, to choose the preprocessing: are the
    images the same size (and if not, should `Resize` letterbox or crop), are some
    unreadable, grayscale or with alpha, much darker or flatter than the rest?

    Files are read in parallel threads. Unreadable files don't raise; they get a row
    with `readable` False, the `error`, and nulls elsewhere.

    Args:
        paths (Iterable[str | PathLike[str]]): The image files.
        workers (int | None): The number of threads reading files. If None, the
            number of CPUs. Defaults to None.

    Returns:
        pl.DataFrame: One row per file, in the order of `paths`, with the columns
            path, readable, error, width, height, aspect (width / height), channels
            (1 grayscale, 3 color, 4 with alpha), dtype (e.g. "uint8", "uint16"),
            file_kb, brightness (the mean of the grayscale image, 0-255) and
            contrast (its standard deviation).
    """
    with ThreadPoolExecutor(workers) as executor:
        rows = list(executor.map(_summarize_file, paths))
    return pl.DataFrame(rows, schema=_SCHEMA, orient="row")


def _summarize_file(path: str | PathLike[str]) -> tuple[Any, ...]:
    path = Path(path)
    failed = (str(path), False)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        return (*failed, f"{type(error).__name__}: {error}", *(None,) * 8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return (*failed, "can't decode the file as an image", *(None,) * 8)
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    gray = _to_gray_8bit(image, channels)
    return (
        str(path),
        True,
        None,
        width,
        height,
        width / height,
        channels,
        str(image.dtype),
        len(data) / 1024,
        float(gray.mean()),
        float(gray.std()),
    )


def _to_gray_8bit(image: np.ndarray, channels: int) -> np.ndarray:
    """Grayscale on a 0-255 scale, whatever the channels and bit depth."""
    if channels == 4:
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY))
    elif channels == 3:
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float64) * 255 / np.iinfo(image.dtype).max
    return np.clip(image.astype(np.float64), 0, 1) * 255
