"""Boolean-safe achievement-column parsing helpers.

`actual_M*_achieved` columns in `metrics.csv` are written as literal
string tokens `"True"` / `"False"`. `float()` / `int()` coercion raises
on `"False"`; if the coercion is wrapped in try/except → None, every
row silently drops and aggregates report `0.0 %`.

Canonical remediation per Phase D (boolean_parse_fix_plan.md). Use
`coerce_achievement_series` for bulk pandas aggregation and
`parse_achievement_bool` for row-scalar checks.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


_TRUE_TOKENS = frozenset({"true", "1", "1.0"})
_FALSE_TOKENS = frozenset({"false", "0", "0.0", ""})
_UNKNOWN_TOKENS = frozenset({"nan", "none"})


def parse_achievement_bool(v: Any) -> Optional[bool]:
    """Parse a single string-encoded boolean.

    Returns True/False for canonical tokens; None for unknown/missing
    (caller decides drop-vs-coerce semantics).
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v == 1:
            return True
        if v == 0:
            return False
        return None
    s = str(v).strip().lower()
    if s in _TRUE_TOKENS:
        return True
    if s in _FALSE_TOKENS:
        return False
    if s in _UNKNOWN_TOKENS:
        return None
    return None


def coerce_achievement_series(series: pd.Series) -> pd.Series:
    """Pandas-vectorized achievement-bool coercion.

    Returns a bool-dtype Series; NaN / unknown / empty tokens coerce
    to False (conservative choice for achievement aggregation — the
    caller's `.sum()` / `.mean()` intent is satisfied and the bug
    that burned H3-Beta cannot silently drop rows).

    Callers that need to distinguish unknown from False should use
    `parse_achievement_bool` row-wise and filter None manually.
    """
    mapping = {
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False, "": False,
        "nan": False, "none": False,
    }
    return (
        series.astype(str)
              .str.strip()
              .str.lower()
              .map(mapping)
              .fillna(False)
              .astype(bool)
    )
