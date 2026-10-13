#!/usr/bin/env python3
# THE TWO DELIVERED TABLES.
# RESPONSIBILITY: join the banked event source (AMPSAFETY-20260903 per-leg summaries), the per-event truth
#   geometry from this dispatch's own tick pass, and the banked readout's per-leg labels into ABORT_EVENTS.csv
#   (one row per event) and ABORT_PER_LEG.csv (one row per leg, all 3,600).
#   THE WINDOW COLUMNS ARE THIS DISPATCH'S OWN, NOT BORROWED. ticks_in_window, ticks_over_bar_in_window and
#   ever_within_window come from abort_window.py at <= 10.0 m on the truth channel. The first cut copied them
#   from AMPSAFETY, which computes its ticks_win at 3.0 m -- three columns computed at 3 m and published under
#   a 10 m label, the same R5.1 defect class the board pages had just been cleared of. A borrowed column
#   carries the LENDER'S window, not the borrower's label.
#   EVERY LEG APPEARS, INCLUDING THE SILENT ONES: a leg with no fire and no truth crossing is a row of zeros
#   beside its denominators, never an absent row. A denominator that quietly shrinks is how a rate flatters
#   itself. Likewise a leg that never came within 10 m is a counted denominator fact, not a refusal.
#   CHANNEL (R5.1): every rate below is read on the TRUTH channel (raw_*) with the window of record at
#   <= 10.0 m of the boom tip. The guard's own comparison channel is the ESTIMATOR (env) and travels beside it.
#   TOLERANCE: four windows -- 0.25 / 0.5 / 1.0 (banked by AMPSAFETY) and 2.0 (added 2026-09-04). PRIMARY is
#   1.0 s. No column is dropped in favour of another; the reader sees whether a count is a window artefact.
#   T1: a fire is a guard fire; a crossing of 2.00 m is a safety-EXCLUSION-ZONE breach, never a "collision".
# PUBLIC API: leg_frames(tag); main(out_dir) (the CLASSIFICATION -- which truth events were eligible and
#   which fires were false -- lives in abort_eligible.py; the tally of its flags lives in abort_events.py)
# SIDE-EFFECTS: writes ABORT_EVENTS.csv and ABORT_PER_LEG.csv under out_dir. Reads banked files read-only.
# SIZE: soft 60 / hard 75 code-LoC; function cap 25. (The eligibility and tally logic was SPLIT into
#   abort_eligible.py when this file passed its HARD cap -- a hard cap is never raised, it is split.)
# COVERAGE: limbs A2 and A4 in abort_limbs.py -- 3,600 leg rows with no duplicate key, and per-leg breakaway
#   totals equal to the banked readout's guards.total on every leg.
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402
import abort_eligible as EL   # noqa: E402
import abort_events as E       # noqa: E402
import abort_window as WD      # noqa: E402

def _geo_join(events, geo, leg):
    """Attach the tick pass's per-event truth geometry and guard-arming coverage to each event, in place."""
    for k, e in enumerate(events):
        e.update({x: y for x, y in geo.get((leg, k), {}).items()
                  if x not in ("tag", "leg", "event_idx", "kind")})
    return events


def _leg_row(tag, ident, r, L, events, wrow):
    """One leg's row: the four abort classes beside the labels and denominators the banked readout carries.
    Split out of leg_frames at that function's declared cap rather than raising it."""
    by = L["guards"]["by_type"]
    return dict(ident, **wrow, n_breakaways=len(r.get("recovery", [])), **E.tally(events),
                m3b_success=bool(L["passed"]), prereg_success=bool(L["prereg_success"]),
                gradable=bool(L["gradable"]), clean=bool(L["clean"]),
                fault_class=L["fault_class"], terminal_reason=r.get("terminal_reason"),
                truth_peak_up_m_whole_leg_readout=L["truth_peak_up_m"], exceeds_2m=bool(L["exceeds_2m"]),
                readout_guards_total=L["guards"]["total"],
                readout_breakaways=L["guards"]["emergency_breakaways"],
                readout_guard_ceiling=by.get("guard_altitude_ceiling", 0),
                readout_guard_overtake=by.get("guard_hard_overtake", 0))


def leg_frames(tag):
    """One corpus -> (event rows, leg rows). Labels come from the banked readout, never re-derived."""
    ro, geo, win = A.readout_legs(tag), {}, WD.window_rows(tag)
    gp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EVENT_GEOM_%s.jsonl" % tag)
    for line in open(gp):
        g = json.loads(line)
        geo[(g["leg"], g.get("event_idx"))] = g
    ev_rows, leg_rows = [], []
    for r in A.summary_rows(tag):
        L = ro[r["leg"]]
        ident = {"tag": tag, "policy": A.POLICY[tag], "sigma": A.SIGMA[tag], "leg": r["leg"],
                 "wind_label": L["wind_label"], "wind_ms": L["wind_ms"], "seed": L["seed"],
                 "trajectory": L["trajectory"], "cell": L["cell"]}
        events = EL.flag(_geo_join(E.leg_events(r), geo, r["leg"]))
        for k, e in enumerate(events):
            ev_rows.append(dict(ident, event_idx=k, **e))
        if r["leg"] not in win:
            raise SystemExit("REFUSE rc=8: %s/%s has no window-of-record counts" % (tag, r["leg"]))
        leg_rows.append(_leg_row(tag, ident, r, L, events, win[r["leg"]]))
    return ev_rows, leg_rows


def main(out_dir):
    ev, lg = [], []
    for tag in A.TAGS:
        a, b = leg_frames(tag)
        ev.extend(a), lg.extend(b)
    de, dl = pd.DataFrame(ev), pd.DataFrame(lg)
    de.to_csv(os.path.join(out_dir, "ABORT_EVENTS.csv"), index=False)
    dl.to_csv(os.path.join(out_dir, "ABORT_PER_LEG.csv"), index=False)
    print("events=%d legs=%d event_kinds=%s" % (len(de), len(dl), de.kind.value_counts().to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
