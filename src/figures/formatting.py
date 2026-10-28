"""How numbers, p values and significance marks are written, identically everywhere."""

from __future__ import annotations

from matplotlib.ticker import FuncFormatter

#: The minus sign the figures use. The ASCII hyphen is a dash's length short
#: of one, and Python's formatting and matplotlib's FormatStrFormatter both
#: write the hyphen.
MINUS = "\u2212"

#: Where a corrected p or q earns its asterisk.
SIG_CUTOFF = 0.05


def fmt_p(p: float) -> str:
    """A p or q value for a table: four decimals, or ``<0.0001``."""
    return f"{p:.4f}" if p >= 1e-4 else "<0.0001"


def p_paren(p: float) -> str:
    """A p value in parentheses, as it reads inside a table cell."""
    return f"(p = {p:.4f})" if p >= 1e-4 else "(p < 0.0001)"


def p_stars(p: float) -> str:
    """Significance marks for a figure: ``***``, ``**``, ``*`` or ``n.s.``.

    Pass the *corrected* p or q value wherever a correction applies; the
    figure's note should say which, since the mark itself does not.
    """
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def signed(value: float, decimals: int, plus: bool = False) -> str:
    """A number for a figure: fixed decimals, a true minus sign, and no "−0.000".

    ``plus`` writes positive numbers with a ``+``, for values whose sign is
    the point (a slope, a ΔAIC).
    """
    text = f"{value:+.{decimals}f}" if plus else f"{value:.{decimals}f}"
    if float(text) == 0:          # rounding to zero leaves the sign of what was rounded
        text = text.replace("-", "+" if plus else "")
    return text.replace("-", MINUS)


def decimal_formatter(decimals: int) -> FuncFormatter:
    """Tick labels with a fixed number of decimals and a true minus sign.

    The replacement for ``FormatStrFormatter("%.3f")``, which writes negative
    ticks with a hyphen.
    """
    return FuncFormatter(lambda value, _pos: signed(value, decimals))


def with_star(p: float, cutoff: float = SIG_CUTOFF) -> str:
    """A corrected p or q as :func:`fmt_p` writes it, with ``*`` when below ``cutoff``."""
    return fmt_p(p) + ("*" if p < cutoff else "")
