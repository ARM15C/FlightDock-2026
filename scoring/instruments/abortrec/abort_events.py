#!/usr/bin/env python3
# THE FOUR ABORT CLASSES, made executable.
# RESPONSIBILITY: turn one leg's banked ampsafety summary into (a) one row per EVENT and (b) one row per LEG
#   carrying the four abort classes at each of the four tolerance windows.
#   THE FOUR CLASSES, each stated as the code reads it:
#     VERTICAL FALSE -- an altitude-ceiling guard fire with NO truth excursion above 2.00 m overlapping
#                         [t_fire - tol, t_fire + tol]. The guard compares the ESTIMATOR channel, so this is a
#                         perception-vs-truth event: the estimator saw a hazard the truth channel did not.
#     VERTICAL MISSED -- a truth excursion above 2.00 m that occurred INSIDE 10 m of the boom tip with NO
#                         ceiling fire in [t0 - tol, t1 + tol]. The 10 m restriction is the operator's, and it
#                         is the window of record; an excursion 40 m astern is not a docking hazard.
#     LONGITUDINAL FALSE -- a hard-overtake guard fire with NO truth no-pass run overlapping the tolerance.
#     LONGITUDINAL MISSED -- a truth no-pass run (raw_behind_m < 0) held at least GUARD_OVERTAKE_PERSIST_S, the
#                         guard's OWN persist gate read from formation_config, with no hard-overtake fire in
#                         the tolerance. Using the guard's own gate is what makes "missed" mean the guard had
#                         its stated trigger and did not fire, rather than "truth briefly crossed".
#   ZERO IS A ZERO: a leg with no fire and no truth event emits NO event rows and a per-leg row of zeros. The
#   two are distinguishable because the per-leg row also carries the denominators (n_ticks, ticks_in_window).
#   T1: a fire is a guard fire and a crossing is a safety-EXCLUSION-ZONE breach -- never a "collision".
# PUBLIC API: EVENT_KINDS, FIRE_KIND, TRUTH_KIND; classify(fires, truth, tol); leg_events(row);
#   tally(events) -- the four classes, counted off abort_eligible's flags. (leg_counts was REMOVED on
#   2026-09-04: every column it returned is a WINDOW quantity and is now recomputed by abort_window.py at the
#   window of record, rather than inherited from an instrument that computes its own at 3.0 m.)
# SIDE-EFFECTS: none. Pure computation over one banked summary row.
# SIZE: soft 70 / hard 90 code-LoC; function cap 25. (The three witness readers were SPLIT into
#   abort_witness.py at this file's declared soft cap rather than raising it -- caps are split, never raised.)
# COVERAGE: limb A1 in abort_limbs.py reproduces AMPSAFETY's own banked vertical counts at the three shared
#   tolerances, on all 3,600 legs -- if this classifier disagrees with the delivered instrument it refuses.
import abort_common as A
import abort_witness as W

FIRE_KIND = {"vertical": "ceiling_fire", "longitudinal": "overtake_fire"}
TRUTH_KIND = {"vertical": "truth_vertical_excursion", "longitudinal": "truth_overtake_run"}
EVENT_KINDS = ("ceiling_fire", "truth_vertical_excursion", "overtake_fire", "truth_overtake_run", "breakaway")


def _overlaps(a0, a1, b0, b1):
    return not (a1 < b0 or a0 > b1)


def classify(fire_t, truth_spans, tol):
    """FALSE fires and MISSED truth spans at one tolerance. Returns (false_idx, missed_idx) as index lists, so
    the caller can mark the EVENT rows themselves rather than only counting them."""
    false_i = [i for i, t in enumerate(fire_t)
               if not any(_overlaps(t - tol, t + tol, b0, b1) for b0, b1 in truth_spans)]
    missed_i = [j for j, (b0, b1) in enumerate(truth_spans)
                if not any(b0 - tol <= t <= b1 + tol for t in fire_t)]
    return false_i, missed_i


def leg_events(row):
    """Every event on one leg, as flat dicts. Times are on the SIM clock."""
    out = []
    for e in W._ceiling(row):
        out.append(dict(e, kind="ceiling_fire", axis="vertical", is_fire=True))
    for e in row["raw"]["excursions"]:
        out.append({"kind": "truth_vertical_excursion", "axis": "vertical", "is_fire": False,
                    "t0": e["t0"], "t1": e["t1"], "dur_s": e["dur_s"],
                    "peak_up_m": e["peak_up_m"], "min_range_m": e["min_range_m"]})
    for e in W._overtake(row):
        out.append(dict(e, kind="overtake_fire", axis="longitudinal", is_fire=True))
    for e in row["raw"]["ahead_runs"]:
        if e["dur_s"] >= A.OT_PERSIST_S:
            out.append({"kind": "truth_overtake_run", "axis": "longitudinal", "is_fire": False,
                        "t0": e["t0"], "t1": e["t1"], "dur_s": e["dur_s"], "max_ahead_m": e["max_ahead_m"]})
    out.extend(W.breakaway_rows(row))
    return out


def tally(events):
    """The four classes for one leg, counted off the SAME flagged events the event table publishes, so the two
    delivered tables can never disagree about how many aborts there were."""
    out = {}
    for axis in ("vertical", "longitudinal"):
        out["%s_fires" % axis] = sum(1 for e in events if e["kind"] == FIRE_KIND[axis])
        out["%s_truth_events" % axis] = sum(1 for e in events
                                            if e["kind"] == TRUTH_KIND[axis] and e.get("truth_eligible"))
        out["%s_truth_events_all" % axis] = sum(1 for e in events if e["kind"] == TRUTH_KIND[axis])
        for tol in A.TOLS:
            out["%s_false_tol%s" % (axis, tol)] = sum(1 for e in events if e["kind"] == FIRE_KIND[axis]
                                                      and e["is_false_tol%s" % tol])
            out["%s_missed_tol%s" % (axis, tol)] = sum(1 for e in events if e["kind"] == TRUTH_KIND[axis]
                                                       and e["is_missed_tol%s" % tol])
    return out
