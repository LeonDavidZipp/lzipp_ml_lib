import asyncio
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from lzipp_ml_lib.transformation.images import (
    CenterCrop,
    ChannelsFirst,
    Equalize,
    Flip,
    Grayscale,
    ImagePipeline,
    Normalize,
    Resize,
    Rotate,
    Scale,
    Sharpen,
    Smooth,
    Step,
    ToFloat,
    read_image,
    write_image,
    write_images,
)
from lzipp_ml_lib.transformation.images._preprocessing import (
    _IMAGENET_MEAN,  # type: ignore
)
from tests.composites import SAMPLE_SETTINGS

_SIDES = st.integers(1, 64)


@st.composite
def images(draw: st.DrawFn, color: bool | None = None) -> np.ndarray:
    """Random uint8 images, color (RGB) or grayscale, of random size."""
    h, w = draw(_SIDES), draw(_SIDES)
    is_color = draw(st.booleans()) if color is None else color
    shape = (h, w, 3) if is_color else (h, w)
    return draw(arrays(np.uint8, shape))


def _rgb(red: int, green: int, blue: int, size: int = 4) -> np.ndarray:
    return np.full((size, size, 3), (red, green, blue), dtype=np.uint8)


# ------------------------------------------------------------------------------------ #
#                                  reading and writing                                 #
# ------------------------------------------------------------------------------------ #


def test_read_and_write_keep_rgb_order(tmp_path: Path):
    path = tmp_path / "red.png"
    write_image(path, _rgb(255, 0, 0))
    assert read_image(path)[0, 0].tolist() == [255, 0, 0]
    assert cv2.imread(str(path))[0, 0].tolist() == [0, 0, 255]  # type: ignore


def test_read_handles_non_ascii_paths(tmp_path: Path):
    path = tmp_path / "bild_größe_日本.png"
    write_image(path, _rgb(10, 20, 30))
    assert read_image(path)[0, 0].tolist() == [10, 20, 30]


def test_read_grayscale(tmp_path: Path):
    path = tmp_path / "gray.png"
    write_image(path, _rgb(255, 255, 255))
    assert read_image(path, grayscale=True).shape == (4, 4)


def test_read_raises_for_missing_and_unreadable_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no image"):
        read_image(tmp_path / "missing.png")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="can't decode"):
        read_image(broken)


# ------------------------------------------------------------------------------------ #
#                                        resizing                                      #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(
    image=images(),
    width=_SIDES,
    height=_SIDES,
    mode=st.sampled_from(["letterbox", "crop", "stretch"]),
)
def test_resize_always_gives_the_requested_size(
    image: np.ndarray, width: int, height: int, mode: str
):
    out = Resize((width, height), keep_aspect=mode)(image)  # type: ignore[arg-type]
    assert out.shape[:2] == (height, width)
    assert out.ndim == image.ndim
    assert out.dtype == image.dtype


def test_letterbox_keeps_the_aspect_ratio_and_pads():
    wide = np.full((10, 40, 3), 200, dtype=np.uint8)  # 4:1
    out = Resize(20, pad_value=7)(wide)
    content_rows = np.flatnonzero((out != 7).any(axis=(1, 2)))
    assert len(content_rows) == 5  # 20 wide -> 5 high, 4:1 kept
    assert (out[0] == 7).all() and (out[-1] == 7).all()


def test_crop_fills_the_size_without_padding():
    wide = np.full((10, 40), 200, dtype=np.uint8)
    out = Resize(20, keep_aspect="crop")(wide)
    assert (out == 200).all()


def test_shrinking_averages_instead_of_aliasing():
    # a 1-pixel checkerboard has no structure at a lower resolution, so a correct
    # shrink is uniform gray; sampling pixels instead leaves a false pattern (aliasing)
    board = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)
    auto = Resize(23, keep_aspect="stretch")(board)
    linear = Resize(23, keep_aspect="stretch", interpolation="linear")(board)
    assert auto.std() < 2
    assert linear.std() > 20


def test_scale_keeps_the_aspect_ratio():
    assert Scale(0.5)(np.zeros((40, 100), dtype=np.uint8)).shape == (20, 50)
    assert Scale(2)(np.zeros((3, 5, 3), dtype=np.uint8)).shape == (6, 10, 3)


@pytest.mark.parametrize("step", [Resize(0), Scale(0), Smooth(0), Sharpen(sigma=0)])
def test_steps_reject_non_positive_parameters(step: object):
    with pytest.raises(ValueError, match="positive"):
        step(np.zeros((4, 4), dtype=np.uint8))  # type: ignore[operator]


# ------------------------------------------------------------------------------------ #
#                                    color and tone                                    #
# ------------------------------------------------------------------------------------ #


def test_grayscale_uses_rgb_luminance_weights():
    # 0.299 * 255 = 76; a BGR mix-up would weigh red as blue (0.114 * 255 = 29)
    assert Grayscale()(_rgb(255, 0, 0))[0, 0] == 76
    assert Grayscale()(_rgb(0, 0, 255))[0, 0] == 29


def test_grayscale_can_keep_three_channels_and_passes_gray_through():
    assert Grayscale(keep_channels=True)(_rgb(255, 0, 0)).shape == (4, 4, 3)
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert np.array_equal(Grayscale()(gray), gray)


def test_equalize_stretches_low_contrast():
    rng = np.random.default_rng(0)
    dull = rng.integers(100, 130, (64, 64)).astype(np.uint8)
    for step in (Equalize(), Equalize(adaptive=False)):
        out = step(dull)
        assert int(out.max()) - int(out.min()) > 2 * (int(dull.max()) - int(dull.min()))


def test_equalize_keeps_the_hue_of_color_images():
    rng = np.random.default_rng(0)
    reddish = np.stack(
        [
            rng.integers(120, 150, (32, 32)),
            rng.integers(40, 60, (32, 32)),
            rng.integers(40, 60, (32, 32)),
        ],
        axis=-1,
    ).astype(np.uint8)
    out = Equalize(adaptive=False)(reddish)
    assert (out[..., 0] >= out[..., 1]).mean() > 0.95  # still red-dominant


def test_equalize_needs_uint8():
    with pytest.raises(ValueError, match="uint8"):
        Equalize()(np.zeros((4, 4), dtype=np.float32))


def test_to_float_maps_onto_zero_to_one():
    out = ToFloat()(np.array([[0, 255]], dtype=np.uint8))
    assert out.dtype == np.float32 and out.tolist() == [[0.0, 1.0]]
    assert ToFloat()(np.array([[65535]], dtype=np.uint16)).item() == 1.0


# ------------------------------------------------------------------------------------ #
#                                  smoothing, sharpening                               #
# ------------------------------------------------------------------------------------ #


def test_smooth_reduces_noise_and_median_keeps_edges():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, (64, 64)).astype(np.uint8)
    assert Smooth(2.0)(noisy).std() < noisy.std() / 2
    edge = np.zeros((32, 32), dtype=np.uint8)
    edge[:, 16:] = 255
    assert np.array_equal(Smooth(2.0, method="median")(edge), edge)


@SAMPLE_SETTINGS
@given(image=images(), amount=st.floats(0, 5), sigma=st.floats(0.3, 3))
def test_sharpen_never_wraps_around(image: np.ndarray, amount: float, sigma: float):
    out = Sharpen(amount, sigma)(image)
    assert out.dtype == image.dtype and out.shape == image.shape
    # where the image is at its brightest, sharpening can't make it darker than
    # the original would wrap to; clipping keeps max-value pixels at the max
    assert (out[image == 255] >= 128).all()


def test_sharpen_keeps_a_bright_spot_bright():
    image = np.full((9, 9), 100, dtype=np.uint8)
    image[4, 4] = 250
    out = Sharpen(amount=3.0)(image)
    assert out[4, 4] == 255  # clipped, not wrapped to a dark value


def test_sharpen_with_zero_amount_changes_nothing():
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3)).astype(np.uint8)
    assert np.array_equal(Sharpen(amount=0.0)(image), image)


# ------------------------------------------------------------------------------------ #
#                                       geometry                                       #
# ------------------------------------------------------------------------------------ #


@SAMPLE_SETTINGS
@given(image=images(), quarter_turns=st.integers(-8, 8))
def test_rotating_by_multiples_of_90_is_exact(image: np.ndarray, quarter_turns: int):
    out = Rotate(90 * quarter_turns)(image)
    assert np.array_equal(out, np.rot90(image, quarter_turns))


def test_rotate_is_counter_clockwise():
    image = np.zeros((2, 2), dtype=np.uint8)
    image[0, 1] = 1  # top right
    assert Rotate(90)(image)[0, 0] == 1  # moves to top left


def test_rotate_expands_the_canvas_so_nothing_is_cut_off():
    image = np.full((20, 40), 255, dtype=np.uint8)
    expanded = Rotate(45)(image)
    cropped = Rotate(45, expand=False)(image)
    assert expanded.shape[0] > 20 and expanded.shape[1] > 40
    assert cropped.shape == (20, 40)
    # the whole rectangle survives the expanded rotation (area 800, up to resampling)
    assert expanded.astype(float).sum() / 255 == pytest.approx(800, rel=0.05)
    assert cropped.astype(float).sum() / 255 < 700


def test_rotate_fills_the_new_area():
    out = Rotate(30, fill_value=9)(np.full((10, 10, 3), 200, dtype=np.uint8))
    assert out[0, 0].tolist() == [9, 9, 9]


def test_flip_and_center_crop():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(Flip()(image), image[:, ::-1])
    assert np.array_equal(Flip(horizontal=False, vertical=True)(image), image[::-1])
    assert np.array_equal(CenterCrop((2, 1))(image), image[1:2, 1:3])
    with pytest.raises(ValueError, match="exceeds"):
        CenterCrop(5)(image)


# ------------------------------------------------------------------------------------ #
#                                       pipeline                                       #
# ------------------------------------------------------------------------------------ #


def test_pipeline_applies_the_steps_in_order(tmp_path: Path):
    image = np.random.default_rng(0).integers(0, 256, (30, 60, 3)).astype(np.uint8)
    steps = [Resize(32), Grayscale(), Smooth(1.0), Sharpen(0.5), Rotate(90)]
    by_hand = image
    for step in steps:
        by_hand = step(by_hand)
    assert np.array_equal(ImagePipeline(steps)(image), by_hand)

    paths = [tmp_path / f"{i}.png" for i in range(3)]
    for path in paths:
        write_image(path, image)
    loaded = list(ImagePipeline(steps).load_many_sequential(paths))
    assert len(loaded) == 3 and all(np.array_equal(out, by_hand) for out in loaded)


@SAMPLE_SETTINGS
@given(image=images(color=True))
def test_pipeline_is_deterministic(image: np.ndarray):
    pipeline = ImagePipeline(
        [Resize(16), Equalize(), Smooth(1.0), Sharpen(1.0), Rotate(17), ToFloat()]
    )
    assert np.array_equal(pipeline(image), pipeline(image.copy()))


def test_pipeline_takes_custom_steps():
    invert = ImagePipeline([lambda image: 255 - image])
    assert invert(np.zeros((2, 2), dtype=np.uint8)).tolist() == [[255, 255], [255, 255]]


# ------------------------------------------------------------------------------------ #
#                                  concurrent writing                                  #
# ------------------------------------------------------------------------------------ #


def test_write_images_writes_the_same_files_as_write_image(tmp_path: Path):
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (20, 30, 3), dtype=np.uint8) for _ in range(12)]
    images.append(rng.integers(0, 256, (20, 30), dtype=np.uint8))  # grayscale too
    paths = [tmp_path / f"{i}.png" for i in range(len(images))]
    asyncio.run(write_images(paths, images, max_concurrency=3))
    for path, image in zip(paths, images, strict=True):
        single = tmp_path / f"single_{path.name}"
        write_image(single, image)
        assert path.read_bytes() == single.read_bytes()
        assert np.array_equal(read_image(path, grayscale=image.ndim == 2), image)


def test_write_images_keeps_rgb_order(tmp_path: Path):
    path = tmp_path / "red.png"
    asyncio.run(write_images([path], [_rgb(255, 0, 0)]))
    assert read_image(path)[0, 0].tolist() == [255, 0, 0]


def test_write_images_never_exceeds_max_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import lzipp_ml_lib.transformation.images._preprocessing as preprocessing

    active, peak = 0, 0
    encode = preprocessing._encode  # type: ignore

    def counting_encode(suffix: str, image: np.ndarray) -> bytes:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            time.sleep(0.01)  # long enough for the other tasks to pile up
            return encode(suffix, image)
        finally:
            active -= 1

    monkeypatch.setattr(preprocessing, "_encode", counting_encode)
    images = [_rgb(1, 2, 3)] * 20
    paths = [tmp_path / f"{i}.png" for i in range(20)]
    asyncio.run(write_images(paths, images, max_concurrency=4))
    assert 1 < peak <= 4
    assert all(path.exists() for path in paths)


def test_write_images_raises_the_first_error(tmp_path: Path):
    paths = [tmp_path / "ok.png", tmp_path / "bad.unknown-format"]
    # the error itself, not wrapped in an ExceptionGroup
    with pytest.raises(ValueError, match="can't encode"):
        asyncio.run(write_images(paths, [_rgb(1, 2, 3)] * 2))


def test_write_image_raises_value_error_for_unknown_formats(tmp_path: Path):
    with pytest.raises(ValueError, match="can't encode"):
        write_image(tmp_path / "image.unknown-format", _rgb(1, 2, 3))


def test_write_images_checks_its_arguments(tmp_path: Path):
    with pytest.raises(ValueError, match="2 paths but 1 images"):
        asyncio.run(
            write_images([tmp_path / "a.png", tmp_path / "b.png"], [_rgb(1, 2, 3)])
        )
    with pytest.raises(ValueError, match="max_concurrency"):
        asyncio.run(write_images([], [], max_concurrency=0))


def test_load_many_is_lazy(tmp_path: Path):
    paths = [tmp_path / "ok.png", tmp_path / "missing.png"]
    write_image(paths[0], _rgb(1, 2, 3))
    images = ImagePipeline([]).load_many_sequential(
        paths
    )  # nothing read yet, so no error yet
    assert next(images)[0, 0].tolist() == [1, 2, 3]
    with pytest.raises(FileNotFoundError):
        next(images)


def test_load_many_takes_lazy_paths(tmp_path: Path):
    for i in range(3):
        write_image(tmp_path / f"{i}.png", _rgb(i, i, i))
    images = ImagePipeline([Grayscale()]).load_many_sequential(
        sorted(tmp_path.glob("*.png"))
    )
    assert [image[0, 0] for image in images] == [0, 1, 2]


# ------------------------------------------------------------------------------------ #
#                                   parallel loading                                   #
# ------------------------------------------------------------------------------------ #


def _write_numbered(tmp_path: Path, n: int) -> list[Path]:
    """`n` small images whose pixel value is their index, to check the order."""
    paths = [tmp_path / f"{i:03}.png" for i in range(n)]
    for i, path in enumerate(paths):
        write_image(path, _rgb(i, i, i, size=8))
    return paths


@SAMPLE_SETTINGS
@given(workers=st.integers(1, 6), chunk_size=st.integers(1, 7), n=st.integers(0, 25))
def test_parallel_loading_matches_load_many_in_order(
    tmp_path_factory: pytest.TempPathFactory, workers: int, chunk_size: int, n: int
):
    paths = _write_numbered(tmp_path_factory.mktemp("images"), n)
    pipeline = ImagePipeline([Grayscale(), Sharpen(0.5)])
    parallel = pipeline.load_many(paths, workers=workers, chunk_size=chunk_size)
    sequential = pipeline.load_many_sequential(paths)
    assert all(np.array_equal(a, b) for a, b in zip(parallel, sequential, strict=True))


def test_parallel_loading_takes_any_callable_as_a_step(tmp_path: Path):
    paths = _write_numbered(tmp_path, 3)
    images = ImagePipeline([lambda image: 255 - image]).load_many(paths)
    assert [int(image[0, 0, 0]) for image in images] == [255, 254, 253]


def test_parallel_loading_takes_lazy_paths(tmp_path: Path):
    _write_numbered(tmp_path, 5)
    images = ImagePipeline([]).load_many(sorted(tmp_path.glob("*.png")))
    assert [int(image[0, 0, 0]) for image in images] == list(range(5))


def test_parallel_loading_raises_errors_in_order(tmp_path: Path):
    paths = [*_write_numbered(tmp_path, 3), tmp_path / "missing.png"]
    images = ImagePipeline([]).load_many(paths, chunk_size=1)
    assert [int(next(images)[0, 0, 0]) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(FileNotFoundError):
        next(images)


def test_parallel_loading_keeps_only_a_few_chunks_in_flight(tmp_path: Path):
    paths = _write_numbered(tmp_path, 40)
    processed = 0

    def count(image: np.ndarray) -> np.ndarray:
        nonlocal processed
        processed += 1
        return image

    images = ImagePipeline([count]).load_many(paths, workers=2, chunk_size=2)
    next(images)
    time.sleep(0.2)  # give the workers time to run ahead, if they could
    images.close()  # stops early: the remaining work is cancelled
    # at most 2 * workers chunks of 2 were ever submitted, not all 40 images
    assert processed <= 2 * 2 * 2


def test_parallel_loading_checks_its_arguments(tmp_path: Path):
    paths = _write_numbered(tmp_path, 1)
    for kwargs in ({"workers": 0}, {"chunk_size": 0}):
        # a generator: the check runs when iteration starts
        with pytest.raises(ValueError, match="at least 1"):
            next(ImagePipeline([]).load_many(paths, **kwargs))  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------ #
#                                    the Step base                                     #
# ------------------------------------------------------------------------------------ #


def test_a_step_needs_a_transform_method():
    class Incomplete(Step):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_calling_a_step_calls_its_transform():
    class Invert(Step):
        def transform(self, image: np.ndarray) -> np.ndarray:
            return 255 - image

    image = np.zeros((2, 2), dtype=np.uint8)
    assert np.array_equal(Invert()(image), Invert().transform(image))
    assert ImagePipeline([Invert()])(image).tolist() == [[255, 255], [255, 255]]


@pytest.mark.parametrize(
    "step",
    [
        Resize(8),
        Scale(0.5),
        Grayscale(),
        Smooth(),
        Sharpen(),
        Rotate(10),
        Flip(),
        CenterCrop(2),
        Equalize(),
        ToFloat(),
        ChannelsFirst(),
        ImagePipeline([]),
    ],
    ids=lambda step: type(step).__name__,
)
def test_every_built_in_step_is_a_step(step: Step):
    assert isinstance(step, Step)
    image = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(step(image), step.transform(image))


def test_pipelines_nest():
    inner = ImagePipeline([Grayscale(), Smooth(1.0)])
    outer = ImagePipeline([Resize(8), inner, Sharpen(0.5)])
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    by_hand = Sharpen(0.5)(Smooth(1.0)(Grayscale()(Resize(8)(image))))
    assert np.array_equal(outer(image), by_hand)


# ------------------------------------------------------------------------------------ #
#                                 model-ready output                                   #
# ------------------------------------------------------------------------------------ #


def test_normalize_standardizes_each_channel():
    image = np.full((2, 2, 3), (0.5, 0.25, 1.0), dtype=np.float32)
    out = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.25, 0.5))(image)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [0.0, -1.0, 1.0])


def test_normalize_defaults_to_imagenet_statistics():
    image = np.broadcast_to(np.array(_IMAGENET_MEAN, dtype=np.float32), (2, 2, 3))
    np.testing.assert_allclose(Normalize()(image), 0, atol=1e-6)


def test_normalize_broadcasts_one_value_and_handles_grayscale():
    gray = np.full((2, 2), 0.75, dtype=np.float32)
    assert Normalize(mean=0.5, std=0.25)(gray).tolist() == [[1.0, 1.0], [1.0, 1.0]]
    color = np.full((2, 2, 3), 0.75, dtype=np.float32)
    np.testing.assert_allclose(Normalize(mean=0.5, std=0.25)(color), 1.0)


@pytest.mark.parametrize(
    ("step", "image", "message"),
    [
        (Normalize(), np.zeros((2, 2, 3), dtype=np.uint8), "apply ToFloat first"),
        (Normalize(std=0.0), np.zeros((2, 2, 3), dtype=np.float32), "positive"),
        (
            Normalize(mean=(0.1, 0.2)),
            np.zeros((2, 2, 3), dtype=np.float32),
            "3 channels",
        ),
    ],
    ids=["integer image", "zero std", "wrong channel count"],
)
def test_normalize_rejects_bad_input(step: Normalize, image: np.ndarray, message: str):
    with pytest.raises(ValueError, match=message):
        step(image)


@SAMPLE_SETTINGS
@given(image=images())
def test_channels_first_moves_the_channels_to_the_front(image: np.ndarray):
    out = ChannelsFirst()(image)
    assert out.flags.c_contiguous
    if image.ndim == 2:
        assert out.shape == (1, *image.shape)
        assert np.array_equal(out[0], image)
    else:
        assert out.shape == (3, *image.shape[:2])
        for channel in range(3):
            assert np.array_equal(out[channel], image[..., channel])


def test_model_ready_pipeline():
    image = np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    prep = ImagePipeline([Resize(32), ToFloat(), Normalize(), ChannelsFirst()])
    out = prep(image)
    assert out.shape == (3, 32, 32) and out.dtype == np.float32
    assert out.flags.c_contiguous


def test_model_ready_output_goes_straight_into_torch():
    torch = pytest.importorskip("torch")
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = ImagePipeline([ToFloat(), Normalize(), ChannelsFirst()])(image)
    tensor = torch.from_numpy(out)
    assert tuple(tensor.shape) == (3, 16, 16)
    assert tensor.dtype == torch.float32


def test_normalize_is_a_step_too():
    image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert isinstance(Normalize(), Step)
    assert np.array_equal(Normalize()(image), Normalize().transform(image))
