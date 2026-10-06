from collections.abc import Callable
from typing import Any

import matplotlib as mpl
import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lzipp_ml_lib.analysis import plotting
from tests.composites import SAMPLE_SETTINGS


def _frame(n: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pl.DataFrame(
        {
            "a": a,
            "b": a * 2 + rng.normal(size=n),
            "c": np.exp(rng.normal(size=n)),
            "count": rng.poisson(3, size=n),
            "group": rng.choice(["x", "y", "z"], size=n),
            "flag": rng.random(n) > 0.7,
        }
    )


def _titles(fig: Figure) -> list[str]:
    return [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]


PANEL_PLOTS: list[Callable[..., Figure]] = [
    plotting.plot_kde,
    plotting.plot_histograms,
    plotting.plot_boxplots,
    plotting.plot_violinplots,
]
PANEL_IDS = [f.__name__ for f in PANEL_PLOTS]
NUMERIC = ["a", "b", "c", "count"]

# ------------------------------------------------------------------------------------ #
#                                  panel plots                                         #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("plot", PANEL_PLOTS, ids=PANEL_IDS)
def test_panel_plot_has_one_panel_per_numeric_column(plot: Callable[..., Figure]):
    assert _titles(plot(_frame())) == NUMERIC


@SAMPLE_SETTINGS
@given(
    plot=st.sampled_from(PANEL_PLOTS),
    columns=st.lists(st.sampled_from(NUMERIC), min_size=1, unique=True),
    lazy=st.booleans(),
)
def test_panel_plot_shows_exactly_the_chosen_columns_in_order(
    plot: Callable[..., Figure], columns: list[str], lazy: bool
):
    df = _frame()
    fig = plot(df.lazy() if lazy else df, columns)
    assert _titles(fig) == columns
    plt.close(fig)


@pytest.mark.parametrize("plot", PANEL_PLOTS, ids=PANEL_IDS)
def test_panel_plot_without_numeric_columns_raises(plot: Callable[..., Figure]):
    with pytest.raises(ValueError, match="nothing to plot"):
        plot(_frame(), ["group"])


@pytest.mark.parametrize("plot", PANEL_PLOTS, ids=PANEL_IDS)
def test_panel_plot_by_keeps_the_group_column_out_of_the_panels(
    plot: Callable[..., Figure],
):
    fig = plot(_frame(), ["a", "c"], by="group")
    assert _titles(fig) == ["a", "c"]


@pytest.mark.parametrize("plot", [plotting.plot_kde, plotting.plot_histograms])
def test_overlaid_plots_by_get_one_legend_entry_per_group(
    plot: Callable[..., Figure],
):
    fig = plot(_frame(), ["a"], by="group")
    (legend,) = fig.legends  # type: ignore
    assert [t.get_text() for t in legend.get_texts()] == ["x", "y", "z"]  # type: ignore
    assert legend.get_title().get_text() == "group"  # type: ignore


@pytest.mark.parametrize("plot", [plotting.plot_boxplots, plotting.plot_violinplots])
def test_side_by_side_plots_by_name_groups_on_the_y_axis(
    plot: Callable[..., Figure],
):
    fig = plot(_frame(), ["a"], by="group")
    assert not fig.legends  # type: ignore
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["x", "y", "z"]


@pytest.mark.parametrize("plot", PANEL_PLOTS, ids=PANEL_IDS)
def test_panel_plot_by_with_too_many_groups_raises(plot: Callable[..., Figure]):
    df = _frame().with_columns(many=pl.int_range(pl.len()) % 9)
    with pytest.raises(ValueError, match="9 groups"):
        plot(df, ["a"], by="many")


@pytest.mark.parametrize("plot", PANEL_PLOTS, ids=PANEL_IDS)
def test_panel_plot_log_drops_non_positive_values(plot: Callable[..., Figure]):
    # "a" is half negative; on a log scale those rows are dropped, not an error
    fig = plot(_frame(), ["a", "c"], log=True)
    assert all(ax.get_xscale() == "log" for ax in fig.axes if ax.get_visible())


def test_histogram_clip_narrows_the_range():
    df = _frame().with_columns(pl.Series("c", [1e6] + [1.0] * 199))
    full = plotting.plot_histograms(df, ["c"]).axes[0].get_xlim()
    clipped = plotting.plot_histograms(df, ["c"], clip=(0.0, 0.99)).axes[0].get_xlim()
    assert clipped[1] < full[1] / 100


def test_histogram_of_few_integers_has_one_bar_per_value():
    df = pl.DataFrame({"n": [1, 1, 2, 3, 3, 3, 7]})
    ax = plotting.plot_histograms(df).axes[0]
    heights = sorted(p.get_height() for p in ax.patches if p.get_height() > 0)  # type: ignore
    assert heights == [1, 1, 2, 3]  # type: ignore


def test_panel_plot_sample_caps_the_rows():
    df = _frame(n=5000)
    ax = plotting.plot_histograms(df, ["count"], sample=100).axes[0]
    assert sum(p.get_height() for p in ax.patches) == 100  # type: ignore


# ------------------------------------------------------------------------------------ #
#                                 category counts                                      #
# ------------------------------------------------------------------------------------ #


def test_category_counts_plots_string_and_boolean_columns():
    titles = _titles(plotting.plot_category_counts(_frame()))
    assert [t.split()[0] for t in titles] == ["group", "flag"]


def test_category_counts_plots_numerical_columns_specified_as_categorical():
    rng = np.random.default_rng(0)
    df = _frame()
    df = df.with_columns(
        as_categorical=pl.Series(rng.binomial(n=1, p=0.5, size=df.height))
    )
    titles = _titles(
        plotting.plot_category_counts(df, as_categorical=["as_categorical"])
    )
    assert [t.split()[0] for t in titles] == ["group", "flag", "as_categorical"]


def test_category_counts_folds_the_rest_into_other_and_counts_nulls():
    df = pl.DataFrame({"v": ["a"] * 5 + ["b"] * 4 + ["c", "d", None]})
    ax = plotting.plot_category_counts(df, top_k=2).axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["other (3)", "b", "a"]  # bottom to top
    assert ax.get_title(loc="left") == "v  ·  5 distinct"


# ------------------------------------------------------------------------------------ #
#                                 feature vs target                                    #
# ------------------------------------------------------------------------------------ #


def test_feature_target_plots_every_feature_but_the_target():
    fig = plotting.plot_feature_target(_frame(), "b")
    assert _titles(fig) == ["a", "c", "count", "group", "flag"]


def test_feature_target_recovers_a_monotonic_relationship():
    ax = plotting.plot_feature_target(_frame(), "b", ["a"], n_bins=5).axes[0]
    means = ax.lines[0].get_ydata()  # type: ignore
    assert len(means) == 5  # type: ignore
    assert np.all(np.diff(means) > 0)  # type: ignore


def test_feature_target_accepts_a_boolean_target():
    fig = plotting.plot_feature_target(_frame(), "flag", ["a"])
    means = fig.axes[0].lines[0].get_ydata()  # type: ignore
    assert np.all((means >= 0) & (means <= 1))  # type: ignore


def test_feature_target_rejects_a_string_target():
    with pytest.raises(ValueError, match="must be numeric"):
        plotting.plot_feature_target(_frame(), "group")


def test_feature_target_log_x_true_logs_every_numeric_feature():
    fig = plotting.plot_feature_target(_frame(), "b", log_x=True)
    scales = {
        ax.get_title(loc="left"): ax.get_xscale() for ax in fig.axes if ax.get_visible()
    }
    assert scales == {
        "a": "symlog",  # half its bins sit below 0
        "c": "log",
        "count": "log",  # has zeros, but they share a bin whose median is 1
        "group": "linear",
        "flag": "linear",
    }


def test_feature_target_log_x_uses_symlog_for_a_bin_at_zero():
    df = pl.DataFrame({"x": [0.0] * 50 + list(range(1, 51)), "y": range(100)})
    ax = plotting.plot_feature_target(df, "y", log_x=True, n_bins=4).axes[0]
    assert ax.get_xscale() == "symlog"
    assert ax.lines[0].get_xdata()[0] == 0  # type: ignore


def test_feature_target_log_x_list_logs_only_those_features():
    fig = plotting.plot_feature_target(_frame(), "b", ["a", "c"], log_x=["c"])
    assert [ax.get_xscale() for ax in fig.axes] == ["linear", "log"]


def test_feature_target_log_x_only_changes_the_axis_not_the_bins():
    plain = plotting.plot_feature_target(_frame(), "b", ["c"]).axes[0]
    logged = plotting.plot_feature_target(_frame(), "b", ["c"], log_x=True).axes[0]
    np.testing.assert_array_equal(
        plain.lines[0].get_xydata(),  # type: ignore
        logged.lines[0].get_xydata(),  # type: ignore
    )


def test_feature_target_log_x_rejects_unknown_features():
    with pytest.raises(ValueError, match="log_x names"):
        plotting.plot_feature_target(_frame(), "b", ["a"], log_x=["group"])


# ------------------------------------------------------------------------------------ #
#                                single-axes plots                                     #
# ------------------------------------------------------------------------------------ #


def test_spearman_heatmap_is_pearson_on_ranks():
    df = _frame()
    ranked = df.select(pl.col("a", "c").rank())
    spearman = plotting.plot_corr_heatmap(df, columns=["a", "c"], method="spearman")
    pearson = plotting.plot_corr_heatmap(ranked, columns=["a", "c"])
    np.testing.assert_allclose(
        spearman.axes[0].collections[0].get_array(),  # type: ignore
        pearson.axes[0].collections[0].get_array(),  # type: ignore
    )


SINGLE_AXES_PLOTS: list[tuple[Callable[..., Figure], dict[str, Any]]] = [
    (plotting.plot_corr_heatmap, {}),
    (plotting.plot_missing_values, {}),
    (plotting.plot_autocorrelation, {"target_col": "a", "lags": 10}),
]


@pytest.mark.parametrize(
    ("plot", "kwargs"), SINGLE_AXES_PLOTS, ids=lambda p: getattr(p, "__name__", "")
)
def test_single_axes_plot_draws_into_a_given_ax(
    plot: Callable[..., Figure], kwargs: dict[str, Any]
):
    fig, (left, right) = plt.subplots(1, 2)
    result = plot(_frame(), ax=right, **kwargs)
    assert result is fig
    assert right.get_title(loc="left")
    assert not left.has_data()


# ------------------------------------------------------------------------------------ #
#                                       style                                          #
# ------------------------------------------------------------------------------------ #


@pytest.mark.parametrize("plot", PANEL_PLOTS, ids=PANEL_IDS)
def test_plots_leave_global_rcparams_alone(plot: Callable[..., Figure]):
    before = dict(mpl.rcParams)  # type: ignore
    plot(_frame())
    assert dict(mpl.rcParams) == before


def test_styled_applies_the_style_only_inside_the_function():
    @plotting.styled
    def facecolor() -> str:
        return mpl.rcParams["axes.facecolor"]  # type: ignore

    assert facecolor() != mpl.rcParams["axes.facecolor"]


def test_plots_return_figures_without_showing_them(monkeypatch: pytest.MonkeyPatch):
    def fail() -> None:
        raise AssertionError("plt.show() was called")

    monkeypatch.setattr(plt, "show", fail)
    for plot in PANEL_PLOTS:
        assert isinstance(plot(_frame()), Figure)
    ax: Axes = plt.figure().add_subplot()  # type: ignore
    assert isinstance(plotting.plot_missing_values(_frame(), ax=ax), Figure)
