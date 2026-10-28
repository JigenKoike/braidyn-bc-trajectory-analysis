"""Tests for the shared figure pieces: formatting, palette, tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from analysis.lmm import PREDICTORS  # noqa: E402
from analysis.pairs import PAIR_ORDER  # noqa: E402
from figures import palette, style  # noqa: E402
from figures.formatting import (  # noqa: E402
    MINUS,
    decimal_formatter,
    fmt_p,
    p_paren,
    p_stars,
    signed,
    with_star,
)
from figures.tables import (  # noqa: E402
    WINDOW_LMM_COLUMNS,
    compact_tables,
    window_lmm_table,
    window_lmm_table_full,
)


# ---- formatting ----

@pytest.mark.parametrize(
    "p, expected",
    [(0.5, "0.5000"), (0.0071, "0.0071"), (1e-4, "0.0001"), (9.9e-5, "<0.0001"), (0.0, "<0.0001")],
)
def test_fmt_p(p: float, expected: str) -> None:
    assert fmt_p(p) == expected


def test_p_paren_switches_to_an_inequality_below_the_cutoff() -> None:
    assert p_paren(0.0421) == "(p = 0.0421)"
    assert p_paren(1e-9) == "(p < 0.0001)"


@pytest.mark.parametrize(
    "p, expected",
    [(0.0009, "***"), (0.001, "**"), (0.009, "**"), (0.01, "*"), (0.049, "*"), (0.05, "n.s.")],
)
def test_p_stars_cutoffs_are_strict(p: float, expected: str) -> None:
    assert p_stars(p) == expected


def test_negative_numbers_get_a_true_minus_sign() -> None:
    assert signed(-0.015, 3) == f"{MINUS}0.015"
    assert signed(0.015, 3) == "0.015"
    assert signed(0.52, 2, plus=True) == "+0.52"
    assert signed(-0.52, 2, plus=True) == f"{MINUS}0.52"
    assert "-" not in signed(-12.5, 1)


def test_a_value_that_rounds_to_zero_carries_no_sign_it_does_not_have() -> None:
    assert signed(-0.0001, 3) == "0.000"
    assert signed(-0.0001, 2, plus=True) == "+0.00"


def test_the_tick_formatter_writes_a_true_minus_sign() -> None:
    fmt = decimal_formatter(3)
    assert fmt(-0.015, 0) == f"{MINUS}0.015"
    assert fmt(0.0, 0) == "0.000"


@pytest.mark.parametrize("q, expected", [(0.0476, "0.0476*"), (0.05, "0.0500"), (0.2, "0.2000"),
                                         (0.00001, "<0.0001*")])
def test_with_star_marks_only_below_the_cutoff(q: float, expected: str) -> None:
    assert with_star(q) == expected


# ---- palette ----

def test_every_pair_has_a_colour_and_both_label_forms() -> None:
    for pair in PAIR_ORDER:
        assert pair in palette.PAIR_COLOR
        assert "–" in palette.PAIR_LABEL[pair]  # en dash, not a hyphen
        assert "\n" in palette.PAIR_LABEL_2LINE[pair]


def test_outcome_colours_stay_clear_of_the_pair_colours() -> None:
    # They appear in different figures of the same paper; reusing a hue would
    # suggest the panels show the same thing.
    assert not set(palette.OUTCOME_COLOR.values()) & set(palette.PAIR_COLOR.values())


def test_every_predictor_has_a_display_name() -> None:
    assert set(palette.PRED_DISPLAY) == set(PREDICTORS)


# ---- style ----

def test_save_fig_writes_one_file_per_format(tmp_path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = style.save_fig(fig, "demo", tmp_path, quiet=True)
    plt.close(fig)
    assert [p.name for p in paths] == ["demo.png", "demo.pdf", "demo.svg"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_save_fig_takes_a_shorter_list_of_formats(tmp_path) -> None:
    fig = plt.figure()
    paths = style.save_fig(fig, "demo", tmp_path, formats=("png",), quiet=True)
    plt.close(fig)
    assert [p.name for p in paths] == ["demo.png"]


def test_svg_keeps_its_text_as_text(tmp_path) -> None:
    # Outlined glyphs cannot be edited or searched; the point of the SVG is
    # that a label can still be moved or restyled in a vector editor.
    assert style.RC_PARAMS["svg.fonttype"] == "none"
    fig, ax = plt.subplots()
    ax.set_xlabel("Task day")
    style.use_paper_style(warn_on_font_fallback=False)
    style.save_fig(fig, "demo", tmp_path, formats=("svg",), quiet=True)
    plt.close(fig)
    svg = (tmp_path / "demo.svg").read_text()
    assert ">Task day<" in svg, "the label should be a text element, not an outline"


def test_save_fig_creates_the_output_directory(tmp_path) -> None:
    fig = plt.figure()
    style.save_fig(fig, "demo", tmp_path / "nested" / "deeper", quiet=True)
    plt.close(fig)
    assert (tmp_path / "nested" / "deeper" / "demo.png").exists()


def test_clean_axes_drops_the_top_and_right_spines() -> None:
    fig, ax = plt.subplots()
    style.clean_axes(ax)
    plt.close(fig)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_visible()


def test_panel_label_sits_at_the_top_left_of_its_region() -> None:
    fig = plt.figure(figsize=(4, 2))
    left, right = fig.subfigures(1, 2)
    label = style.panel_label(left, "a")
    plt.close(fig)
    assert label.get_text() == "a"
    assert label.get_fontweight() == "bold"
    assert label.get_position() == (0.0, 1.0)
    assert (label.get_ha(), label.get_va()) == ("left", "top")


def test_panel_labels_line_up_across_panels_whatever_the_tick_labels() -> None:
    # The point of anchoring to the region: one panel's wide tick labels must
    # not push its letter further right than its neighbour's.
    fig = plt.figure(figsize=(6, 2), layout="constrained")
    left, right = fig.subfigures(1, 2)
    for region, labels in ((left, ["0", "1"]), (right, ["a very wide label", "x"])):
        ax = region.subplots()
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels)
    a = style.panel_label(left, "a")
    b = style.panel_label(right, "b")
    fig.canvas.draw()
    gap = b.get_window_extent().x0 - a.get_window_extent().x0
    plt.close(fig)
    # Exactly half the figure apart, give or take the glyphs' own side
    # bearings. Anchoring to the axes instead would add the width of that
    # second panel's tick labels — tens of pixels, not a few.
    assert gap == pytest.approx(fig.bbox.width / 2, abs=5.0)


def test_the_type_scale_runs_in_order_down_to_an_eight_point_floor() -> None:
    assert style.PANEL_LABEL_SIZE > style.LABEL_SIZE > style.FONT_SIZE > style.TICK_SIZE
    assert style.TICK_SIZE == 8, "8 pt is the floor for anything a reader must read"
    assert style.DENSE_TICK_SIZE < style.TICK_SIZE, "the one sanctioned exception"


def test_rcparams_never_set_a_size_below_the_floor() -> None:
    sizes = {k: v for k, v in style.RC_PARAMS.items() if k.endswith("size")}
    assert sizes, "the rcParams should pin the type sizes"
    assert all(v >= style.TICK_SIZE for v in sizes.values()), sizes


def test_size_colorbar_puts_the_bar_on_the_shared_scale() -> None:
    fig, ax = plt.subplots()
    image = ax.imshow([[0, 1], [2, 3]])
    cbar = fig.colorbar(image, ax=ax)
    style.size_colorbar(cbar, "weight (a.u.)")
    plt.close(fig)
    assert cbar.ax.get_ylabel() == "weight (a.u.)"
    assert all(t.get_fontsize() == style.TICK_SIZE for t in cbar.ax.get_yticklabels())


def test_size_colorbar_can_leave_the_label_alone() -> None:
    fig, ax = plt.subplots()
    cbar = fig.colorbar(ax.imshow([[0, 1]]), ax=ax)
    style.size_colorbar(cbar)
    plt.close(fig)
    assert cbar.ax.get_ylabel() == ""


def test_the_font_stack_asks_for_arial_and_can_fall_back() -> None:
    assert style.FONT_STACK[0] == "Arial"
    assert "DejaVu Sans" in style.FONT_STACK  # matplotlib's own, always present


def test_text_is_embedded_as_truetype_not_type_3() -> None:
    # Type 3 is matplotlib's default and most journals reject it.
    assert style.RC_PARAMS["pdf.fonttype"] == 42
    assert style.RC_PARAMS["ps.fonttype"] == 42


def test_use_paper_style_warns_when_the_preferred_face_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(style, "FONT_STACK", ["NoSuchFaceAnywhere", "DejaVu Sans"])
    with pytest.warns(UserWarning, match="NoSuchFaceAnywhere"):
        style.use_paper_style()
    style.use_paper_style(warn_on_font_fallback=False)  # restore the real params


def test_staggering_offsets_every_other_label_and_leaves_the_rest() -> None:
    fig, ax = plt.subplots()
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"roi_{i}" for i in range(6)])
    fig.canvas.draw()
    before = [lab.get_window_extent().x1 for lab in ax.get_yticklabels()]

    style.staggered_tick_labels(ax, "y", offset_points=20)
    fig.canvas.draw()
    after = [lab.get_window_extent().x1 for lab in ax.get_yticklabels()]
    plt.close(fig)

    shift = [a - b for a, b in zip(after, before)]
    assert all(abs(s) < 1e-6 for s in shift[0::2]), "even labels should not move"
    assert all(s < -1.0 for s in shift[1::2]), "odd labels should move away from the axis"


def test_staggering_keeps_the_label_text() -> None:
    fig, ax = plt.subplots()
    ax.set_yticks(range(4))
    ax.set_yticklabels(list("abcd"))
    style.staggered_tick_labels(ax, "y")
    plt.close(fig)
    assert [lab.get_text() for lab in ax.get_yticklabels()] == list("abcd")


# ---- keeping panels the same size ----

def _row(fig, n: int = 3, ylabel: str | None = None, tick_labels: list[str] | None = None):
    """A row of panels. ``tick_labels`` is what actually costs a row width —
    a rotated axis label is only ever one line of type tall."""
    axes = fig.subplots(1, n, sharey=True)
    for ax in axes:
        ax.plot([0, 1], [0, 1])
    if ylabel:
        axes[0].set_ylabel(ylabel)
    if tick_labels:
        axes[0].set_yticks(np.linspace(0, 1, len(tick_labels)))
        axes[0].set_yticklabels(tick_labels)
    return list(axes)


def test_equal_panel_widths_names_the_drift_it_found() -> None:
    fig = plt.figure(figsize=(7.1, 2.0), layout="constrained")
    axes = _row(fig)
    fig.draw_without_rendering()
    assert style.equal_panel_widths(axes) > 0  # one gridspec: equal by construction

    axes[0].set_position((0.05, 0.1, 0.1, 0.8))  # make one of them narrow
    with pytest.raises(AssertionError, match="differ in width"):
        style.equal_panel_widths(axes)
    plt.close(fig)


def test_a_subfigure_per_panel_is_what_makes_them_unequal() -> None:
    # The reason uniform_columns exists: subfigures lay out independently, so
    # the panel carrying the axis label comes out markedly narrower.
    fig = plt.figure(figsize=(7.1, 2.0), layout="constrained")
    subfigs = fig.subfigures(1, 3)
    axes = []
    for i, sub in enumerate(subfigs):
        ax = sub.subplots()
        ax.plot([0, 1], [0, 1])
        if i == 0:
            ax.set_ylabel("a label long enough to cost real width")
        else:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    fig.draw_without_rendering()
    widths = [ax.get_position().width for ax in axes]
    plt.close(fig)
    assert widths[0] < 0.9 * widths[1], "the labelled panel should come out narrower"


def test_uniform_columns_gives_every_panel_one_width_and_one_grid() -> None:
    fig = plt.figure(figsize=(7.1, 4.0), layout="constrained")
    top, bottom = fig.subfigures(2, 1)
    row_a = _row(top, ylabel="AUC")
    row_b = _row(bottom, ylabel="β", tick_labels=["First-pull duration", "Baseline F (UV)"])
    fig.draw_without_rendering()
    assert row_a[0].get_position().x0 != pytest.approx(row_b[0].get_position().x0), \
        "the two rows should start out misaligned, or this test proves nothing"

    spans = style.uniform_columns(fig, row_a, row_b)
    assert len(spans) == 3
    for (x0, x1), ax_top, ax_bottom in zip(spans, row_a, row_b):
        assert ax_top.get_position().x0 == pytest.approx(x0)
        assert ax_bottom.get_position().x0 == pytest.approx(x0)
        assert ax_top.get_position().x1 == pytest.approx(x1)
        assert ax_bottom.get_position().x1 == pytest.approx(x1)
    assert style.equal_panel_widths(row_a + row_b, tol=1e-6) > 0
    plt.close(fig)


def test_uniform_columns_freezes_the_layout_so_saving_cannot_undo_it(tmp_path) -> None:
    fig = plt.figure(figsize=(7.1, 4.0), layout="constrained")
    row_a = _row(fig.subfigures(2, 1)[0], ylabel="AUC")
    fig.draw_without_rendering()
    style.uniform_columns(fig, row_a)
    placed = [ax.get_position().bounds for ax in row_a]
    style.save_fig(fig, "frozen", tmp_path, formats=("png",), quiet=True)
    assert [ax.get_position().bounds for ax in row_a] == placed
    plt.close(fig)


def test_uniform_columns_refuses_rows_of_different_lengths() -> None:
    fig = plt.figure(figsize=(7.1, 4.0), layout="constrained")
    top, bottom = fig.subfigures(2, 1)
    with pytest.raises(AssertionError, match="different numbers of panels"):
        style.uniform_columns(fig, _row(top, 3), _row(bottom, 2))
    plt.close(fig)


def test_panel_label_at_clears_the_tick_labels_of_its_own_panel() -> None:
    fig = plt.figure(figsize=(7.1, 2.0), layout="constrained")
    axes = _row(fig, ylabel="a label")
    fig.draw_without_rendering()
    first = style.panel_label_at(fig, axes[0], "a")
    second = style.panel_label_at(fig, axes[1], "b")
    assert first.get_position()[0] < axes[0].get_position().x0, "the letter should clear the labels"
    assert first.get_position()[0] < second.get_position()[0]
    assert first.get_fontweight() == "bold"
    assert first.get_fontsize() == style.PANEL_LABEL_SIZE
    plt.close(fig)


def test_a_section_heading_puts_its_letter_and_title_on_one_baseline_over_the_group() -> None:
    fig = plt.figure(figsize=(7.1, 2.4), layout="constrained")
    fig.suptitle(" ", fontsize=style.PANEL_LABEL_SIZE)   # the room the heading goes in
    axes = _row(fig, n=4, tick_labels=["a long tick label", "short"])
    for ax in axes:
        ax.set_title("Early window")
    fig.draw_without_rendering()
    left, left_title = style.section_heading(fig, axes[:2], "a", "PC weighting", level_with=axes)
    right, right_title = style.section_heading(fig, axes[2:], "b", "Trial count", level_with=axes)
    # One baseline for both letters and both headings.
    ys = {t.get_position()[1] for t in (left, left_title, right, right_title)}
    assert len(ys) == 1
    assert all(t.get_verticalalignment() == "baseline" for t in (left, left_title))
    # Above every title, and inside the figure.
    y = ys.pop()
    assert y > max(ax.get_tightbbox().transformed(fig.transFigure.inverted()).y1 for ax in axes)
    assert y < 1.0
    # The letter clears the group's tick labels; the heading is centred over its frames.
    assert left.get_position()[0] < axes[0].get_position().x0
    centre = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    assert left_title.get_position()[0] == pytest.approx(centre, abs=1e-6)
    assert left.get_fontsize() == style.PANEL_LABEL_SIZE and left_title.get_fontsize() == style.LABEL_SIZE
    plt.close(fig)


def test_a_group_xlabel_is_written_once_under_the_middle_of_its_plots() -> None:
    fig = plt.figure(figsize=(7.1, 2.4), layout="constrained")
    axes = _row(fig, n=4)
    for ax in axes:
        ax.set_xlabel(" ")    # the room the shared label goes in
    fig.draw_without_rendering()
    label = style.group_xlabel(fig, axes[:2], "Slope against task day (SD / day)")
    x, y = label.get_position()
    centre = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    assert x == pytest.approx(centre, abs=1e-6)
    # Below the plots, where the blank labels hold the room, and inside the figure.
    assert 0.0 < y < min(ax.get_position().y0 for ax in axes[:2])
    assert label.get_fontsize() == style.LABEL_SIZE
    plt.close(fig)


def test_an_outside_colour_bar_leaves_the_panels_their_full_width() -> None:
    def build(outside: bool):
        fig = plt.figure(figsize=(7.1, 2.0), layout="constrained")
        axes = _row(fig)
        mappable = axes[-1].imshow([[0, 1], [1, 0]])
        if outside:
            _, cax = style.outside_colorbar(fig, mappable, "ΔAIC")
            fig.draw_without_rendering()
            style.place_outside_colorbar(cax, axes[-1])
        else:
            fig.colorbar(mappable, ax=axes[-1])
            fig.draw_without_rendering()
        width = axes[0].get_position().width
        plt.close(fig)
        return width

    assert build(outside=True) > build(outside=False), \
        "a colour bar inside the layout takes its width out of the panels"


def test_assert_page_width_catches_a_figure_that_would_be_rescaled() -> None:
    fig = plt.figure(figsize=(7.1, 2.0), layout="constrained")
    _row(fig)
    fig.draw_without_rendering()
    assert style.assert_page_width(fig, width=7.1, tol=0.2) == pytest.approx(7.1, abs=0.2)
    with pytest.raises(AssertionError, match="not 3.5"):
        style.assert_page_width(fig, width=3.5)
    plt.close(fig)


# ---- manuscript-shaped tables ----

def _tidy_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_fit = pd.DataFrame([
        {"pair": pair, "n_sessions": 100 + i, "n_mice": 25, "AIC": -1234.5678 + i,
         "marginal_R2": 0.1234, "conditional_R2": 0.4321,
         "predictors_CI_excludes_0": "day", "predictors_wald_q_lt_0.05": ""}
        for i, pair in enumerate(PAIR_ORDER)
    ])
    loo = pd.DataFrame([
        {"pair": pair, "predictor": p, "delta_AIC": 1.5, "lrt_p": 0.0071, "lrt_q": 0.0495}
        for pair in PAIR_ORDER for p in PREDICTORS
    ])
    coef = pd.DataFrame([
        {"pair": pair, "predictor": p, "beta": 0.0025, "ci95_lo": 0.0011,
         "ci95_hi": 0.0039, "wald_p": 0.0071, "wald_q": 0.0495}
        for pair in PAIR_ORDER for p in PREDICTORS
    ])
    return model_fit, loo, coef


def test_compact_tables_have_the_manuscript_shapes() -> None:
    c2, c3, c4 = compact_tables(*_tidy_tables())
    assert len(c2) == len(PAIR_ORDER)                 # one row per pair
    assert list(c3.index) == list(PREDICTORS)         # one row per predictor
    assert len(c3.columns) == 3 * len(PAIR_ORDER)     # ΔAIC, LRT p, LRT q per pair
    assert len(c4) == 2 * len(PREDICTORS)             # beta row and CI row


def test_compact_tables_order_predictors_by_the_model_formula() -> None:
    _, c3, c4 = compact_tables(*_tidy_tables())
    assert list(c3.index) == list(PREDICTORS)
    assert list(c4["predictor"])[::2] == list(PREDICTORS)


def test_compact_tables_use_the_display_pair_labels() -> None:
    c2, c3, c4 = compact_tables(*_tidy_tables())
    labels = [palette.PAIR_LABEL[p] for p in PAIR_ORDER]
    assert list(c2.index) == labels
    assert all(any(lab in col for lab in labels) for col in c3.columns)
    assert [c for c in c4.columns if c not in ("predictor", "value")] == labels


def test_coefficient_table_pairs_each_estimate_with_its_interval() -> None:
    _, _, c4 = compact_tables(*_tidy_tables())
    first_pair = palette.PAIR_LABEL[PAIR_ORDER[0]]
    assert c4.loc[0, "value"] == "β (Wald p)"
    assert c4.loc[0, first_pair] == "+0.00250 (p = 0.0071, q = 0.0495)"
    assert c4.loc[1, "value"] == "95% CI"
    assert c4.loc[1, first_pair] == "[+0.00110, +0.00390]"


# ---- the one-sheet window table (new Tables 2 and 3) ----

def test_window_lmm_table_puts_a_heading_above_each_pair() -> None:
    t = window_lmm_table(*_tidy_tables())
    assert list(t.columns) == list(WINDOW_LMM_COLUMNS)
    assert len(t) == len(PAIR_ORDER) * (1 + len(PREDICTORS))
    headings = t.iloc[:: 1 + len(PREDICTORS)]
    for row, pair in zip(headings.itertuples(), PAIR_ORDER):
        assert palette.PAIR_LABEL[pair] in row.Predictor
        assert "n = " in row.Predictor and "mice" in row.Predictor
        assert "marginal R² = 0.123" in row.Predictor
        assert "conditional R² = 0.432" in row.Predictor
    # A heading introduces its pair and carries no numbers of its own.
    assert (headings[[c for c in WINDOW_LMM_COLUMNS if c != "Predictor"]] == "").all().all()


def test_window_lmm_table_reads_each_q_beside_the_p_it_corrects() -> None:
    assert list(WINDOW_LMM_COLUMNS).index("Wald q") == list(WINDOW_LMM_COLUMNS).index("Wald p") + 1
    assert list(WINDOW_LMM_COLUMNS).index("LRT q") == list(WINDOW_LMM_COLUMNS).index("LRT p") + 1
    t = window_lmm_table(*_tidy_tables())
    first = t.iloc[1]
    assert first["Predictor"] == palette.PRED_DISPLAY[PREDICTORS[0]]
    assert first["β"] == "+0.00250"
    assert first["95% CI"] == "[+0.00110, +0.00390]"
    assert (first["Wald p"], first["Wald q"]) == ("0.0071", "0.0495")
    assert (first["ΔAIC"], first["LRT p"], first["LRT q"]) == ("+1.50", "0.0071", "0.0495")


def test_window_lmm_table_full_is_one_row_per_pair_and_predictor() -> None:
    model_fit, loo, coef = _tidy_tables()
    full = window_lmm_table_full(model_fit, loo, coef)
    assert len(full) == len(PAIR_ORDER) * len(PREDICTORS)
    assert list(full["pair"]) == [p for p in PAIR_ORDER for _ in PREDICTORS]
    assert list(full["predictor"]) == list(PREDICTORS) * len(PAIR_ORDER)
    # The pair's fit travels with its rows, so the file stands on its own.
    for col in ("n_sessions", "n_mice", "marginal_R2", "conditional_R2", "beta", "delta_AIC"):
        assert col in full.columns
    assert (full.loc[full["pair"] == PAIR_ORDER[0], "n_sessions"] == 100).all()


def test_window_lmm_table_full_keeps_the_label_columns() -> None:
    model_fit, loo, coef = _tidy_tables()
    for frame in (model_fit, loo, coef):
        frame.insert(0, "window", "early")
    full = window_lmm_table_full(model_fit, loo, coef)
    assert len(full) == len(PAIR_ORDER) * len(PREDICTORS)
    assert (full["window"] == "early").all()


def test_leave_one_out_table_keeps_the_sign_of_delta_aic() -> None:
    model_fit, loo, coef = _tidy_tables()
    loo.loc[0, "delta_AIC"] = -1.1
    _, c3, _ = compact_tables(model_fit, loo, coef)
    assert c3.iloc[0, 0] == "-1.10"
    assert c3.iloc[1, 0] == "+1.50"


def test_display_path_is_written_from_the_repository_root(tmp_path) -> None:
    # An absolute path printed by a notebook is saved inside the notebook
    # file, carrying one machine's layout into the repository.
    repo = tmp_path / "repo"
    (repo / "output" / "paper_figures").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\n")
    assert style.display_path(repo / "output" / "paper_figures") == "output/paper_figures"
    assert style.display_path(repo) == "."


def test_display_path_leaves_a_path_outside_the_repository_alone(tmp_path) -> None:
    outside = tmp_path / "somewhere" / "else"
    outside.mkdir(parents=True)
    assert style.display_path(outside) == str(outside)


def test_save_fig_prints_a_relative_path(tmp_path, capsys) -> None:
    repo = tmp_path / "repo"
    out = repo / "output" / "paper_figures"
    out.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\n")
    fig = plt.figure()
    style.save_fig(fig, "demo", out, formats=("png",))
    plt.close(fig)
    printed = capsys.readouterr().out
    assert "output/paper_figures/demo" in printed
    assert str(tmp_path) not in printed, "no absolute path should reach the notebook"
