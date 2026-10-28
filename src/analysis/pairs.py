"""The trial outcomes and the outcome pairs the analysis compares.

The ordering here is the one the manuscript uses throughout, so figures,
tables and CSVs list the pairs the same way without each restating it.
"""

from __future__ import annotations

OUTCOMES: tuple[str, str, str] = ("success", "failure", "miss")

#: Outcome pairs, as ``(a, b)``. ``success-miss`` first because it is the
#: comparison the original analysis was built around.
PAIRS: tuple[tuple[str, str], ...] = (
    ("success", "miss"),
    ("failure", "miss"),
    ("success", "failure"),
)

#: The same pairs as the ``"a-b"`` keys used in dataframes and file columns.
PAIR_ORDER: tuple[str, ...] = tuple(f"{a}-{b}" for a, b in PAIRS)
