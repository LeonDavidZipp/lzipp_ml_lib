from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from lzipp_ml_lib.transformation.images import (
    ChannelsFirst,
    Grayscale,
    ImagePipeline,
    Normalize,
    Resize,
    Smooth,
    StepLike,
    ToFloat,
    plot_image_summary,
    show_images,
    show_pipeline,
    summarize_images,
    write_image,
)
from lzipp_ml_lib.transformation.images._plotting import (
    _describe,  # type: ignore
    _displayable,  # type: ignore
)
from lzipp_ml_lib.transformation.images._preprocessing import Image  # type: ignore
from tests.composites import SAMPLE_SETTINGS


def _image(height: int = 12, width: int = 20) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _shown(fig: Figure) -> list[AxesImage]:  # type: ignore[no-untyped-def]
    return [im for ax in fig.axes for im in ax.get_images()]


# ------------------------------------------------------------------------------------ #
#                                 how images are shown                                 #
# ------------------------------------------------------------------------------------ #


def test_displayable_undoes_channels_first_and_normalize():
    image = _image()
    model_ready = ImagePipeline([ToFloat(), Normalize(), ChannelsFirst()])(image)
    np.testing.assert_allclose(_displayable(model_ready), image / 255, atol=1e-5)


def test_displayable_undoes_custom_normalize_statistics():
    image = _image()
    normalized = ImagePipeline([ToFloat(), Normalize(mean=0.5, std=0.25)])(image)
    np.testing.assert_allclose(
        _displayable(normalized, mean=0.5, std=0.25), image / 255, atol=1e-5
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_displayable_scales_integers_by_their_range(dtype: type):
    full = np.full((2, 2, 3), np.iinfo(dtype).max, dtype=dtype)  # type: ignore
    np.testing.assert_allclose(_displayable(full), 1.0)  # type: ignore


@SAMPLE_SETTINGS
@given(
    image=st.one_of(
        arrays(np.uint8, st.tuples(st.integers(1, 9), st.integers(1, 9))),
        arrays(np.uint8, st.tuples(st.integers(1, 9), st.integers(1, 9), st.just(3))),
        arrays(
            np.float32,
            st.tuples(st.just(3), st.integers(5, 9), st.integers(5, 9)),
            elements=st.floats(-5, 5, width=32),
        ),
    )
)
def test_displayable_always_gives_something_imshow_can_show(image: np.ndarray):
    shown = _displayable(image)
    assert shown.ndim == 2 or shown.shape[2] in (3, 4)
    assert np.isfinite(shown).all()
    assert shown.min() >= 0 and shown.max() <= 1


def test_grayscale_keeps_its_brightness_instead_of_being_stretched():
    dark = np.full((8, 8), 40, dtype=np.uint8)
    dark[0, 0] = 60
    (shown,) = _shown(show_images([dark]))
    assert shown.get_cmap().name == "gray"
    assert shown.get_clim() == (0, 1)  # a fixed scale: 40 stays dark gray


def test_show_images_lays_out_one_panel_per_image_with_titles():
    fig = show_images([_image()] * 5, ["a", "b", "c", "d", "e"], ncols=2)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title(loc="left") for ax in visible] == ["a", "b", "c", "d", "e"]
    assert len(_shown(fig)) == 5


def test_show_images_takes_a_generator(tmp_path: Path):
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    for path in paths:
        write_image(path, _image())
    fig = show_images(ImagePipeline([Grayscale()]).load_many_sequential(paths))
    assert len(_shown(fig)) == 3


@pytest.mark.parametrize(
    ("images", "titles", "message"),
    [
        ([], None, "no images"),
        ([np.zeros((2, 2))], ["a", "b"], "2 titles for 1 images"),
        ([np.zeros((2, 2, 2, 2))], None, "2 or 3 dimensions"),
        ([np.zeros((5, 5, 2))], None, "1, 3 or 4 channels"),
    ],
)
def test_show_images_rejects_bad_input(
    images: list[Image], titles: list[str] | None, message: str
):  # type: ignore[type-arg]
    with pytest.raises(ValueError, match=message):
        show_images(images, titles)


# ------------------------------------------------------------------------------------ #
#                                    show_pipeline                                     #
# ------------------------------------------------------------------------------------ #


def test_show_pipeline_shows_every_stage_with_its_output():
    fig = show_pipeline([Resize(16), Grayscale(), Smooth(sigma=2.0)], _image())
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]
    assert [t.split("\n")[0] for t in titles] == [
        "original",
        "Resize(16)",
        "Grayscale()",
        "Smooth(sigma=2.0)",
    ]
    assert titles[1].split("\n")[1].startswith("16×16×3 uint8")
    assert titles[2].split("\n")[1].startswith("16×16 uint8")


def test_show_pipeline_takes_a_pipeline():
    fig = show_pipeline(ImagePipeline([Grayscale(), ToFloat()]), _image())
    assert len(_shown(fig)) == 3


_DESCRIBED_STEPS: list[tuple[StepLike, str]] = [
    (Resize(224), "Resize(224)"),
    (Resize(224, keep_aspect="crop"), "Resize(224, keep_aspect='crop')"),
    (Normalize(mean=0.5, std=0.25), "Normalize(mean=0.5, std=0.25)"),
    (ImagePipeline([Grayscale(), ToFloat()]), "ImagePipeline(2 steps)"),
    (lambda image: image, "lambda"),
    (np.fliplr, "fliplr"),
]


@pytest.mark.parametrize(("step", "name"), _DESCRIBED_STEPS)
def test_steps_are_described_by_what_differs_from_the_defaults(
    step: StepLike, name: str
):
    assert _describe(step) == name


# ------------------------------------------------------------------------------------ #
#                                    image summaries                                   #
# ------------------------------------------------------------------------------------ #


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    write_image(tmp_path / "wide.png", np.full((10, 40, 3), 200, dtype=np.uint8))
    write_image(tmp_path / "square.jpg", np.zeros((30, 30, 3), dtype=np.uint8))
    write_image(tmp_path / "gray.png", np.full((8, 8), 100, dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "alpha.png"), np.zeros((5, 5, 4), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "deep.png"), np.full((4, 4), 65535, dtype=np.uint16))
    (tmp_path / "broken.png").write_bytes(b"not an image")
    return tmp_path


def test_summarize_images_describes_every_file(folder: Path):
    names = [
        "wide.png",
        "square.jpg",
        "gray.png",
        "alpha.png",
        "deep.png",
        "broken.png",
    ]
    summary = summarize_images([folder / name for name in names])
    rows = {Path(row["path"]).name: row for row in summary.iter_rows(named=True)}
    assert [Path(p).name for p in summary["path"]] == names  # in the order given
    assert rows["wide.png"] | {"path": None} == rows["wide.png"] | {
        "path": None,
        "readable": True,
        "error": None,
        "width": 40,
        "height": 10,
        "aspect": 4.0,
        "channels": 3,
        "dtype": "uint8",
    }
    assert rows["wide.png"]["brightness"] == pytest.approx(200)
    assert rows["gray.png"]["channels"] == 1
    assert rows["alpha.png"]["channels"] == 4
    assert rows["deep.png"]["dtype"] == "uint16"
    assert rows["deep.png"]["brightness"] == pytest.approx(255)  # on the 0-255 scale
    assert rows["broken.png"]["readable"] is False
    assert "can't decode" in rows["broken.png"]["error"]
    assert rows["broken.png"]["width"] is None


def test_summarize_images_reports_missing_files(tmp_path: Path):
    summary = summarize_images([tmp_path / "missing.png"])
    assert summary.row(0, named=True)["readable"] is False
    assert "FileNotFoundError" in summary.row(0, named=True)["error"]


def test_plot_image_summary_shows_sizes_aspects_and_brightness(folder: Path):
    fig = plot_image_summary(summarize_images(sorted(folder.iterdir())))
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert titles[0].startswith("Sizes")
    assert "1 unreadable" in fig.get_suptitle()
    # gray.png and the single-channel 16-bit deep.png
    assert "2 grayscale" in fig.get_suptitle() and "1 with alpha" in fig.get_suptitle()


def test_plot_image_summary_needs_a_readable_image(tmp_path: Path):
    summary = summarize_images([tmp_path / "missing.png"])
    with pytest.raises(ValueError, match="no readable images"):
        plot_image_summary(summary)
