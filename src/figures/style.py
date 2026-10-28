"""Page geometry, type sizes and saving, for Frontiers' figure specification.

Call :func:`use_paper_style` once at the top of a figure notebook, lay the
figure out at :data:`WIDTH_2COL` or :data:`WIDTH_1COL` inches wide, and save
it with :func:`save_fig`.

Two layout habits that this project learned the hard way:

- Use ``constrained_layout=True`` and do **not** pass ``wspace`` / ``hspace``
  to a gridspec. Under constrained layout those are read as a fraction of the
  whole figure, which collapses the panels.
- Put a panel letter outside the top-left corner of its axes with
  :func:`panel_label`, not with a figure-level text call, so it travels with
  the panel when the layout changes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import SubFigure
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation

#: Frontiers' two-column figure width, in inches.
WIDTH_2COL = 7.1

#: Frontiers' single-column figure width, in inches.
WIDTH_1COL = 3.5

# Type sizes, in points. Four of them, so a reader meets one scale across the
# whole figure set rather than a different one per panel.
#
# TICK_SIZE is the floor: nothing in a figure goes below 8 pt, which is what
# stays legible at the printed width. DENSE_TICK_SIZE is the single sanctioned
# exception, for an axis with more categories than 8 pt can fit — Figure 1
# lists 44 ROIs. Reach for staggered_tick_labels before reaching for it.
PANEL_LABEL_SIZE = 14  # the bold a / b / c
LABEL_SIZE = 10  # axis labels and panel titles
FONT_SIZE = 9  # anything not named here
TICK_SIZE = 8  # tick labels, legends, colour bars — the floor
DENSE_TICK_SIZE = 7  # the exception; see above

#: Typeface, with fallbacks. Arial is what the journal asks for and what the
#: authors' machines have; Helvetica is metrically close, and DejaVu Sans is
#: matplotlib's own, so a machine with neither still renders the figures (at
#: slightly different text widths).
FONT_STACK: list[str] = ["Arial", "Helvetica", "DejaVu Sans"]

RC_PARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    # Embed text as TrueType rather than matplotlib's default Type 3, which
    # most journals reject and which no PDF reader can search or edit.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Leave SVG text as text, so a label can still be edited or restyled in a
    # vector editor. The cost is that a machine without Arial renders the SVG
    # in whatever it substitutes; "path" would draw the glyphs as outlines
    # instead, identical everywhere but no longer text.
    "svg.fonttype": "none",
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": TICK_SIZE,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}


def use_paper_style(warn_on_font_fallback: bool = True) -> None:
    """Apply the paper's rcParams to the current matplotlib session.

    Says so if the preferred typeface is missing, since matplotlib otherwise
    falls back silently and the figures come out in a different face than
    the ones that went to the journal.
    """
    plt.rcParams.update(RC_PARAMS)
    if warn_on_font_fallback:
        installed = {f.name for f in font_manager.fontManager.ttflist}
        if FONT_STACK[0] not in installed:
            present = next((f for f in FONT_STACK if f in installed), "a default face")
            warnings.warn(
                f"{FONT_STACK[0]} is not installed; falling back to {present}. "
                "Text metrics will differ from the submitted figures.",
                stacklevel=2,
            )


def staggered_tick_labels(ax: plt.Axes, axis: str = "y", offset_points: float = 14.0) -> None:
    """Push every other tick label further out, to fit twice as many legibly.

    A dense categorical axis — 44 ROIs down one side — otherwise forces a
    type size nobody can read. Two staggered columns give each label the room
    of two, at the cost of a wider margin.
    """
    target = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    pad = (-offset_points, 0) if axis == "y" else (0, -offset_points)
    for label in target[1::2]:
        label.set_transform(label.get_transform() + ScaledTranslation(
            pad[0] / 72, pad[1] / 72, ax.figure.dpi_scale_trans))


def panel_label(
    region: SubFigure | plt.Figure,
    letter: str,
    size: int = PANEL_LABEL_SIZE,
    xy: tuple[float, float] = (0.0, 1.0),
) -> Text:
    """Put a bold panel letter at the top-left corner of a panel's whole region.

    ``region`` is the subfigure the panel occupies, not its axes. Anchoring to
    the axes puts the letter beside the plot frame, which lands it to the
    right of however wide that panel's tick labels happen to be — a different
    place in every panel. Anchoring to the region lines the letters up down
    the page, which is what makes them scannable.

    Give each panel its own subfigure, so there is a region to anchor to.
    """
    return region.text(*xy, letter, fontsize=size, fontweight="bold",
                       ha="left", va="top")


def panel_label_at(
    region: SubFigure | plt.Figure,
    ax: plt.Axes,
    letter: str,
    size: int = PANEL_LABEL_SIZE,
    top: float | None = None,
) -> Text:
    """Panel letter at the top of ``region``, over the left edge of ``ax``'s panel.

    Use this instead of :func:`panel_label` when several panels share one
    gridspec, which is what keeps their widths equal: a subfigure per panel
    lays each one out on its own, so the panel carrying the axis label comes
    out narrower than its neighbours — by a quarter, in the figure that
    taught us this. One gridspec equalizes them, and the letters then need
    placing by hand, since there is only one region for all of them.

    "Left edge" means the left edge of everything the panel occupies, tick
    labels and axis label included, so the letter clears them rather than
    landing on top. ``top`` defaults to the top of that same extent, which
    lines the letters of one row up with each other; pass a number to put
    them all at one height in ``region``'s coordinates instead.

    **Call this after the layout has run** (``fig.draw_without_rendering()``);
    before that the positions are the ones the layout is about to replace.
    """
    bbox = ax.get_tightbbox().transformed(region.transSubfigure.inverted())
    return region.text(bbox.x0, bbox.y1 if top is None else top, letter,
                       fontsize=size, fontweight="bold", ha="left", va="top")


def section_heading(
    region: SubFigure | plt.Figure,
    axes: Sequence[plt.Axes],
    letter: str,
    title: str,
    *,
    level_with: Sequence[plt.Axes] | None = None,
    gap_points: float = 4.0,
    letter_size: int = PANEL_LABEL_SIZE,
    title_size: int = LABEL_SIZE,
) -> tuple[Text, Text]:
    """A panel letter and its heading on one baseline, above a panel made of several axes.

    For a panel that is a group of plots — one comparison drawn for each
    window, say — the letter belongs to the group and a heading names it.
    The letter goes at the left edge of everything the group occupies, as
    :func:`panel_label_at` puts it; the heading is centred over the group's
    plot frames; and both sit on the same baseline, just above the tallest
    of ``level_with`` (default: the group itself). Pass the whole row as
    ``level_with`` so that two groups side by side get their headings at one
    height.

    The layout has to leave room above the axes: give ``region`` a blank
    ``suptitle`` at ``letter_size`` before it runs. **Call this after the
    layout has run** (``fig.draw_without_rendering()``).
    """
    to_region = region.transSubfigure.inverted()
    boxes = [ax.get_tightbbox().transformed(to_region) for ax in axes]
    frames = [ax.get_window_extent().transformed(to_region) for ax in axes]
    # Not `level_with or axes`: subplots() hands back a numpy array, whose
    # truth value raises.
    level = axes if level_with is None else level_with
    tops = [ax.get_tightbbox().transformed(to_region).y1 for ax in level]
    y = max(tops) + gap_points * region.dpi / 72 / region.bbox.height
    centre = (min(f.x0 for f in frames) + max(f.x1 for f in frames)) / 2
    letter_text = region.text(min(b.x0 for b in boxes), y, letter, fontsize=letter_size,
                              fontweight="bold", ha="left", va="baseline")
    title_text = region.text(centre, y, title, fontsize=title_size, fontweight="bold",
                             ha="center", va="baseline")
    return letter_text, title_text


def group_xlabel(
    region: SubFigure | plt.Figure,
    axes: Sequence[plt.Axes],
    text: str,
    size: int = LABEL_SIZE,
) -> Text:
    """One x-axis label, centred under several plots that share it.

    A label per plot repeats itself, and under narrow plots runs into its
    neighbour's. Give each of ``axes`` a blank label (``ax.set_xlabel(" ")``)
    before the layout runs, so that the room is left; **after the layout**
    this writes the label once, where the blank ones sit, centred under the
    plots' frames.
    """
    to_region = region.transSubfigure.inverted()
    frames = [ax.get_window_extent().transformed(to_region) for ax in axes]
    blank = axes[0].xaxis.label.get_window_extent().transformed(to_region)
    centre = (min(f.x0 for f in frames) + max(f.x1 for f in frames)) / 2
    return region.text(centre, blank.y1, text, fontsize=size, ha="center", va="top")


def outside_colorbar(
    region: SubFigure | plt.Figure,
    mappable,
    label: str | None = None,
    size: float = TICK_SIZE,
):
    """A colour bar beyond ``region``'s right edge, invisible to the layout.

    A colour bar handed to the layout engine takes its width out of the
    panels, which leaves the panel it belongs to narrower than the ones
    beside it — and unequal panels read as though their contents were drawn
    at different sizes. This one is added by hand and excluded from the
    layout, so the panels keep their full width and the bar hangs past them.
    The saved bounding box is tight, so what hangs over still reaches the
    file; lay the figure out :data:`CBAR_OVERHANG` narrower than the page so
    that the two come back to the page width.

    Returns ``(cbar, cax)``. Call :func:`place_outside_colorbar` once the
    layout has run to put it against the panel it describes.
    """
    cax = region.add_axes((1.0, 0.0, 0.01, 1.0))  # placed properly later
    cax.set_in_layout(False)  # add_axes alone is not enough to free the width
    cbar = region.colorbar(mappable, cax=cax)
    size_colorbar(cbar, label, size)
    return cbar, cax


def place_outside_colorbar(
    cax: plt.Axes, ax: plt.Axes, gap: float = 0.012, width: float = 0.010
) -> None:
    """Put an :func:`outside_colorbar` beside ``ax``, matching its height.

    ``gap`` and ``width`` are fractions of the region the colour bar lives
    in. Call after the layout has run.
    """
    pos = ax.get_position()
    cax.set_position((pos.x1 + gap, pos.y0, width, pos.height))


#: How far outside the panels an :func:`outside_colorbar` and its label
#: reach, in inches. Lay a figure that uses one out this much narrower than
#: the page; :func:`assert_page_width` checks the sum.
CBAR_OVERHANG = 0.44


def assert_page_width(fig: plt.Figure, width: float = WIDTH_2COL, tol: float = 0.05) -> float:
    """Check that everything saved comes to ``width`` inches, and return it.

    A figure that reaches the journal wider than the column is scaled down
    to fit, and the type scale goes down with it — an 8 pt floor set at 7.74
    inches prints at 7.3 pt once squeezed to 7.1. Call after the layout has
    run.
    """
    found = fig.get_tightbbox().width
    assert abs(found - width) < tol, (
        f"the figure comes to {found:.3f} inch, not {width}; "
        f"adjust the width it is laid out at by {width - found:+.3f}"
    )
    return found


def uniform_columns(
    fig: plt.Figure, *rows: Sequence[plt.Axes], gap: float | None = None
) -> list[tuple[float, float]]:
    """Lay every row's panels on one grid of equal columns, and freeze the layout.

    The layout engine lines columns up inside one gridspec but not across
    two, and two rows rarely reserve the same margins — one row's left-hand
    panel may carry seven predictor names where the other's carries a number
    and a short label. Where the rows show the same things in the same order,
    one column per outcome pair say, a reader takes any offset between them,
    or any difference in size, to mean something.

    So: one left edge and one right edge for all the rows, the space between
    split into equal columns. Every panel ends up the same width, which
    costs the row with the narrower margins some whitespace on its left.
    ``gap`` defaults to the widest gap the rows already leave between
    neighbours.

    Call after the layout has run. Returns each column's ``(x0, x1)``.
    """
    assert len({len(r) for r in rows}) == 1, "the rows have different numbers of panels"
    extents = {(round(r[0].figure.bbox.x0, 6), round(r[0].figure.bbox.x1, 6)) for r in rows}
    assert len(extents) == 1, "the rows do not span the same width; their positions differ in meaning"

    n = len(rows[0])
    left = max(r[0].get_position().x0 for r in rows)
    right = min(r[-1].get_position().x1 for r in rows)
    if gap is None and n > 1:
        gap = max(r[i + 1].get_position().x0 - r[i].get_position().x1
                  for r in rows for i in range(n - 1))
    gap = gap or 0.0
    width = (right - left - gap * (n - 1)) / n
    assert width > 0, "the margins leave no room for the columns"

    spans = []
    for i in range(n):
        x0 = left + i * (width + gap)
        for row in rows:
            box = row[i].get_position()
            row[i].set_position((x0, box.y0, width, box.height))
        spans.append((x0, x0 + width))
    fig.set_layout_engine("none")
    return spans


def equal_panel_widths(axes: Iterable[plt.Axes], tol: float = 1e-9) -> float:
    """Check that ``axes`` came out the same width, and return that width.

    Panels showing the same kind of thing side by side have to be the same
    size or the eye reads the difference as data. Raises if they are not,
    naming the drift, since the usual cause — one panel carrying the axis
    label — is invisible until measured.
    """
    widths = [ax.get_position().width for ax in axes]
    drift = max(widths) - min(widths)
    assert drift < tol, (
        f"panels differ in width by {drift:.5f} ({[f'{w:.4f}' for w in widths]}); "
        "panels that share a row need to share one gridspec"
    )
    return widths[0]


def size_colorbar(cbar, label: str | None = None, size: float = TICK_SIZE) -> None:
    """Put a colour bar's label and ticks on the shared type scale.

    Colour bars are where stray type sizes creep in, because matplotlib sizes
    their text from rcParams that the rest of the figure does not use.
    """
    if label is not None:
        cbar.set_label(label, fontsize=size)
    cbar.ax.tick_params(labelsize=size)


def clean_axes(ax: plt.Axes) -> None:
    """Drop the top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


#: What every figure is written as, unless a caller asks for something else.
FORMATS: tuple[str, ...] = ("png", "pdf", "svg")


def display_path(path: Path | str) -> str:
    """A path written from the repository root, for printing.

    What a notebook prints is saved inside the notebook file, so an absolute
    path there is committed: it carries one machine's directory layout into
    the repository, says nothing to anyone else, and comes back as a diff
    every time someone on another machine runs the notebook. The repository
    root is found by walking up for ``pyproject.toml``; a path outside it is
    returned unchanged, since there is nothing to shorten it against.
    """
    path = Path(path)
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").exists():
            return str(path.relative_to(parent)) or "."
    return str(path)


def save_fig(
    fig: plt.Figure,
    name: str,
    out_dir: Path,
    formats: Iterable[str] = FORMATS,
    quiet: bool = False,
) -> list[Path]:
    """Save ``fig`` as ``<out_dir>/<name>.<ext>`` for each format, at 300 dpi.

    PNG for looking at, PDF for the submission, SVG for the times a figure
    has to be opened and altered — a journal asking for a label moved, or a
    panel lifted into a talk — without going back to the notebook for it.
    SVG keeps its text as text; see :data:`RC_PARAMS`. Returns the paths
    written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        paths.append(path)
    if not quiet:
        print(f"  saved: {display_path(out_dir)}/{name}.{{{','.join(formats)}}}")
    return paths
