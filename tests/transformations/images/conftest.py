from collections.abc import Iterator

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")
