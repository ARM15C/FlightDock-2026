#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: The collision ENVELOPE predicate as an importable streaming
#   counter (T-437). One instance per (leg, channel): feed each row's
#   behind/cross/below, read .fires afterwards. Semantics mirror the env's
#   runtime terminal (env_formation_sitl.py:7030-7063): in-zone means
#     cg < 3.0 AND below < -2.0 AND -2.0 <= behind <= 5.0,
#   cg = sqrt(behind^2 + cross^2 + below^2) computed from the SAME three
#   components (never a range column), fires at >= 3 CONSECUTIVE in-zone
#   rows. An unparseable/NaN row BREAKS the run (fail-closed: unparseable is
#   never in-zone) — the T-433 A1 rule (all rows, no dedupe) is the caller's
#   duty; this module never sees timestamps.
#   Constants MIRRORED from src/drl/config.py:486-490 deliberately — an
#   audit instrument must not import the module under audit (a constant
#   drift would silently follow the code). t433_audit.py:37-39 precedent.
# PUBLIC API: EnvelopeCounter (feed(behind, cross, below) -> None;
#   .ticks_in_zone; .max_consecutive; .fires); CG_MIN, ABOVE_BOOM, BEH_MIN,
#   BEH_MAX, SUSTAIN; CLI: --selfcheck.
# SIDE-EFFECTS: none (pure computation; CLI prints check lines).
# CAPS: file soft 90 / hard 120 code-LoC; function cap 30.
# COVERAGE: tests/unit/test_collision_envelope.py + in-module --selfcheck
#   (executed by the test battery; selfchecks live in-module because pytest
#   cannot run while a SITL leg is flying — the gse precedent).
# =============================================================================
from __future__ import annotations

import sys

# config.py:486-490 -- mirrored here deliberately (see header).
CG_MIN, ABOVE_BOOM, BEH_MIN, BEH_MAX, SUSTAIN = 3.0, -2.0, -2.0, 5.0, 3


class EnvelopeCounter:
    """Streaming run-length counter for the collision envelope."""

    def __init__(self):
        self.ticks_in_zone = 0
        self.max_consecutive = 0
        self._run = 0

    def feed(self, behind, cross, below):
        try:
            b, c, v = float(behind), float(cross), float(below)
            parsed = (b == b) and (c == c) and (v == v)   # NaN rejects
        except (TypeError, ValueError):
            parsed = False
        if not parsed:
            self._run = 0
            return
        cg = (b * b + c * c + v * v) ** 0.5
        if cg < CG_MIN and v < ABOVE_BOOM and BEH_MIN <= b <= BEH_MAX:
            self.ticks_in_zone += 1
            self._run += 1
            if self._run > self.max_consecutive:
                self.max_consecutive = self._run
        else:
            self._run = 0

    @property
    def fires(self):
        return self.max_consecutive >= SUSTAIN


def _fed(rows):
    ec = EnvelopeCounter()
    for r in rows:
        ec.feed(*r)
    return ec


def _selfcheck():
    IN = (0.5, 0.0, -2.05)          # cg~2.11 <3, below <-2, behind in range
    OUT = (0.5, 0.0, -1.0)
    checks = [
        ("i   3 consecutive in-zone FIRES", _fed([IN] * 3).fires),
        ("ii  2 consecutive does NOT fire", not _fed([IN] * 2).fires),
        ("iii gap resets the run (2+gap+2 = no fire, 4 ticks)",
         (lambda e: not e.fires and e.ticks_in_zone == 4
          and e.max_consecutive == 2)(_fed([IN, IN, OUT, IN, IN]))),
        ("iv  below exactly -2.0 NOT in zone (strict <)",
         _fed([(0.5, 0.0, -2.0)] * 3).ticks_in_zone == 0),
        ("v   behind -2.0 boundary IS in zone (inclusive); behind 5.0 is "
         "band-legal but cg-gated (cg 5.4 >= 3.0 rejects)",
         _fed([(-2.0, 0.0, -2.05)]).ticks_in_zone == 1
         and _fed([(5.0, 0.0, -2.05)]).ticks_in_zone == 0),
        ("vi  interior point with all three components nonzero fires",
         _fed([(1.0, 1.0, -2.05)] * 3).fires),
        ("vii cg >= 3.0 rejects even with below/behind in range",
         _fed([(2.0, 1.5, -2.05)] * 3).ticks_in_zone == 0),  # cg 3.24
        ("viii unparseable row BREAKS a sustained run",
         not _fed([IN, IN, ("x", 0.0, -2.05), IN, IN]).fires),
        ("ix  behind outside [-2,5] rejects",
         _fed([(-2.01, 0.0, -2.05)] * 3).ticks_in_zone == 0),
    ]
    for n, p in checks:
        print(("PASS " if p else "FAIL ") + n)
    print("selfcheck:", "PASS" if all(p for _, p in checks) else "FAIL")
    return 0 if all(p for _, p in checks) else 1


if __name__ == "__main__":
    sys.exit(_selfcheck() if "--selfcheck" in sys.argv else 0)
