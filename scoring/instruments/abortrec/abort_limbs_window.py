#!/usr/bin/env python3
# THE CHANNEL LIMBS -- a CAP-DRIVEN STRUCTURAL SPLIT out of abort_limbs_recon.py, moving no logic.

#   other than the one the table claims. Both limbs exist because of defects that were REAL, not hypothetical.
#   A10 -- the window columns. Three of them were borrowed from an instrument that computes its window at
#   3.0 m and published under a 10 m label. It hid because ever_within_window was
#   exactly `min_range_m <= 3.0` on 3,600/3,600 legs: an invariant that looks like it is passing. So A10 tests
#   the identity at the DECLARED window AND against an independently built instrument's denominator, because
#   either test alone can pass while the other fails.
#   A11 -- the peak columns. The banked readout's `truth_peak_up_m` is a WHOLE-LEG quantity (measured: 100.0 %
#   agreement whole-leg, 72.8 % against a 3 m windowed peak), so a table printing it under a 10 m heading is
#   unlabelled one level down, at the column rather than the page. Both peaks are now named for their channel
#   and the borrowed one is verified against this dispatch's own read rather than trusted.
#   A BORROWED COLUMN CARRIES THE LENDER'S WINDOW, NOT THE BORROWER'S LABEL. That is the rule both encode.
# PUBLIC API: a10(d, check); a11(d, check)
# SIDE-EFFECTS: none. Reads the dispatch's own artifacts read-only.
# SIZE: soft 45 / hard 60 code-LoC; function cap 25.
# COVERAGE: driven by abort_limbs.main on every run; each limb prints its denominator so a silent pass shows.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402


def a10(d, check):
    """THE WINDOW LIMB, added 2026-09-04 after the operator found three columns computed at 3 m and published
    under a 10 m label. Two tests, because either alone can pass while the other fails:
    (i) INTERNAL -- ever_within_window must equal (min_range_m <= WINDOW_M) on every leg. Before the fix this
        held against 3.0 m on 3,600/3,600 legs, which is precisely how the defect hid.
    (ii) CROSS-INSTRUMENT -- the per-corpus never-within count must equal the gustapproach APPROACH pull's
        own answer, banked in APPROACH_LEG_CENSUS.json as a measured artifact rather than read out of prose.
        Two instruments, built independently, one denominator."""
    ok = bool((d.ever_within_window == (d.min_range_m <= A.WINDOW_M)).all())
    check("A10 ever_within_window == (min_range_m <= %.1f m)" % A.WINDOW_M, ok,
          "%d/%d legs" % (int((d.ever_within_window == (d.min_range_m <= A.WINDOW_M)).sum()), len(d)))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "APPROACH_LEG_CENSUS.json")
    cen = json.load(open(path))
    mine = (~d.ever_within_window).groupby(d.tag).sum().to_dict()
    diff = {t: (int(mine.get(t, -1)), cen[t]["never_within_10m"]) for t in cen
            if int(mine.get(t, -1)) != cen[t]["never_within_10m"]}
    check("A10 never-within count == the gustapproach approach pull", not diff,
          "total %d vs %d; per corpus %s" % (int(sum(mine.values())),
                                             sum(v["never_within_10m"] for v in cen.values()),
                                             diff or "all six agree"))


def a11(d, check):
    """The peak columns name their own channel, and the borrowed one is verified rather than trusted.
    (i) this dispatch's own whole-leg peak must equal the banked readout's `truth_peak_up_m` -- which is a
        WHOLE-LEG quantity (measured: 100.0 % agreement whole-leg, only 72.8 % against a 3 m windowed peak),
        so a table printing it under a 10 m heading is unlabelled at the column level;
    (ii) the windowed peak can never exceed the whole-leg peak, and it is NULL exactly on the legs that never
        entered the window -- a null, never a 0.0, which is what makes the two denominators differ honestly."""
    ok = (d.peak_up_m_whole_leg - d.truth_peak_up_m_whole_leg_readout).abs() <= 0.005
    check("A11 own whole-leg peak == the readout's whole-leg peak", bool(ok.all()),
          "%d/%d legs within 5 mm" % (int(ok.sum()), len(d)))
    inw = d.peak_up_m_in_window
    le = (inw.isna()) | (inw <= d.peak_up_m_whole_leg + 1e-9)
    null_matches = bool((inw.isna() == ~d.ever_within_window).all())
    check("A11 windowed peak <= whole-leg peak, null exactly when never in window",
          bool(le.all()) and null_matches,
          "%d/%d ordered; %d nulls vs %d legs never within %.1f m"
          % (int(le.sum()), len(d), int(inw.isna().sum()), int((~d.ever_within_window).sum()), A.WINDOW_M))
