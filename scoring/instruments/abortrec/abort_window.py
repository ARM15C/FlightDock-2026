#!/usr/bin/env python3
# THE WINDOW-OF-RECORD TICK COUNTS.
# RESPONSIBILITY: count, per leg and on the TRUTH channel, the ticks inside the window of record -- and the
#   ticks over the 2.00 m bar inside it -- at <= 10.0 m of the boom tip.
#   WHY THIS FILE EXISTS: the first cut of this dispatch
#   copied `ticks_in_window`, `ticks_over_bar_in_window` and `ever_within_window` straight out of AMPSAFETY's
#   banked per-leg summary. AMPSAFETY computes those with `WINDOW = 3.0` (safety_channels.py:28), so three
#   columns were computed at 3 m and published under a 10 m label -- exactly the R5.1 defect class the board
#   pages had just been cleared of. Measured: `ever_within_window` was identical to `min_range_m <= 3.0` on
#   3,600/3,600 legs, and the never-within count read 275 when the 10 m answer is 234.
#   THE LESSON IN ONE LINE: a borrowed column carries the LENDER'S window, not the borrower's label. A window
#   is part of a column's identity, so it is recomputed here rather than inherited.
#   CROSS-INSTRUMENT ANCHOR: the 10 m tick set is exactly what the gustapproach APPROACH pull extracted,
#   so the per-corpus never-within counts must agree with that instrument leg for leg (limb A10).
#   T1: `up` = -raw_below_m is height above the BOOM TIP; crossing 2.00 m is a safety-EXCLUSION-ZONE breach.
# PUBLIC API: COLS; leg_window(args) -> dict; window_rows(tag) -> {leg: counts}
#   (the driver that fans this across a corpus lives in abort_window_run.py -- the same reduction-vs-I/O
#    split this dispatch uses for abort_ticks/abort_pass and abort_recovery/abort_recovery_run.)
# SIDE-EFFECTS: none -- reads telemetry and this dispatch's own WINDOW_TICKS files read-only.
# SIZE: soft 55 / hard 70 code-LoC; function cap 25.
# COVERAGE: limb A10 in abort_limbs.py -- ever_within_window equals (min_range_m <= WINDOW_M) on every leg,
#   and the per-corpus never-within count equals the gustapproach approach-pull census.
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402

COLS = ["sim_time_s", "raw_range_m", "raw_below_m"]


def leg_window(args):
    """One leg's truth-channel tick counts at the window of record. A leg that never enters the window is a
    counted denominator fact -- zeros and a finite min_range_m -- never a refusal and never an absent row."""
    tag, leg_dir, name = args
    paths = sorted(glob.glob(os.path.join(leg_dir, "env*", "flight_state_ep*.csv.gz")))
    if not paths:
        return {"tag": tag, "leg": name, "error": "no telemetry under %s" % leg_dir}
    d = pd.concat([pd.read_csv(p, usecols=COLS, low_memory=False) for p in paths], ignore_index=True)
    for c in COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    rng, up = d.raw_range_m.to_numpy(float), -d.raw_below_m.to_numpy(float)
    win = np.less_equal(rng, A.WINDOW_M, where=np.isfinite(rng), out=np.zeros(len(rng), bool))
    over = np.greater(up, A.BAR, where=np.isfinite(up), out=np.zeros(len(up), bool))
    return {"tag": tag, "leg": name, "n_ticks": int(len(d)),
            "ticks_in_window": int(win.sum()),
            "ticks_over_bar": int(over.sum()),
            "ticks_over_bar_in_window": int((over & win).sum()),
            "ticks_over_contact": int(np.greater(up, A.CONTACT, where=np.isfinite(up),
                                                 out=np.zeros(len(up), bool)).sum()),
            "min_range_m": (float(np.nanmin(rng)) if np.isfinite(rng).any() else None),
            # BOTH peaks, each naming its own channel. The banked readout carries only the WHOLE-LEG peak, so a
            # table that prints it under a 10 m heading is unlabelled at the column level -- the same defect
            # class as the tick counts above, one level down. A leg that never enters the window has NO
            # windowed peak: that is a null, never a 0.0, and it is why the two channels have different
            # denominators and cannot be reconciled by rounding.
            "peak_up_m_whole_leg": (float(np.nanmax(up)) if np.isfinite(up).any() else None),
            "peak_up_m_in_window": (float(np.nanmax(up[win])) if win.any() and np.isfinite(up[win]).any()
                                    else None),
            "ever_within_window": bool(win.any())}


def window_rows(tag):
    """This dispatch's OWN window-of-record tick counts, keyed by leg. NOT inherited from AMPSAFETY, whose
    ticks_win is computed at 3.0 m -- borrowing it published three columns at 3 m under a 10 m label
    (defect found on dev 2026-09-04). A missing or unreadable leg REFUSES: a silently absent denominator is
    how a rate flatters itself."""
    out = {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WINDOW_TICKS_%s.jsonl" % tag)
    for line in open(path):
        w = json.loads(line)
        if "error" in w:
            raise SystemExit("REFUSE rc=8: %s/%s has no window counts: %s" % (tag, w["leg"], w["error"]))
        out[w["leg"]] = {k: v for k, v in w.items() if k not in ("tag", "leg")}
    return out
