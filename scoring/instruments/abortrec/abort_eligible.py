#!/usr/bin/env python3
# WHICH TRUTH EVENTS THE GUARD COULD ACTUALLY HAVE FIRED ON -- a CAP-DRIVEN STRUCTURAL SPLIT out of
# abort_build.py, moving no logic.
# RESPONSIBILITY: decide, per truth event, whether the guard was in a position to fire at all; flag every
#   event with its abort class at every tolerance; and tally one leg's four classes off those same flags.
#   THE CORRECTION THIS FILE EXISTS TO MAKE (2026-09-04): the controller wraps BOTH guards in
#   `if self.state in FORMATION_STATES:`, so outside those six states neither guard is evaluated. A truth
#   crossing while the aircraft was already in an emergency breakaway is therefore NOT a missed abort -- the
#   guard was never asked. Before this gate the instrument counted 968 longitudinal "missed aborts"; most were
#   the aircraft holding ahead of the leader DURING a breakaway, which is the manoeuvre working, not a miss.
#   Reading the mechanism before attributing the cause is the whole of the fix.
#   THE LONGITUDINAL GATE IS THE GUARD'S OWN: the armed stretch must be CONTIGUOUS and at least
#   GUARD_OVERTAKE_PERSIST_S, because 10 s of arming split into two 5 s pieces never fires.
#   ONE OWNER FOR THE CLASSIFICATION: the per-leg table is tallied from the SAME flagged events the per-event
#   table publishes, so the two delivered tables cannot disagree about how many aborts there were.
#   AN INELIGIBLE EVENT IS LEFT NULL, NOT FALSE, and carries the reason -- False would read as "the guard got
#   this one right", which is a different and unearned claim.
#   ELIGIBILITY GATES *MISSED* ONLY, NEVER *FALSE*. A FALSE abort asks whether the truth crossing existed; a
#   MISSED abort asks whether the guard failed to act on one it could have fired on. Applying the arming gate
#   to both was a defect in this instrument on 2026-09-04 -- it deleted real truth crossings from the
#   comparison set and re-labelled correctly-caught fires as false positives.
#   T1: guard fires and safety-EXCLUSION-ZONE breaches, never "collisions".
# PUBLIC API: eligible(e, axis); axis_split -> (fires, every, eligible); flag(events)
#   (tally() moved to abort_build.py at this file's cap -- it READS these flags, it does not create them, so the
#    single owner of the classification is unchanged.)
# SIDE-EFFECTS: none. Pure computation, though flag() and axis_split() annotate the event dicts in place.
# SIZE: soft 80 / hard 90 code-LoC; function cap 25. (Both were first declared at 55/70 in the same edit that
#   created the file, before its scope was measured; calibrated ONCE here to 80/90 and stated, on the same
#   footing as the sibling split safety_events.py. Neither moves again: from here a cap is SPLIT, never raised.)
# COVERAGE: limbs A2/A4/A9 in abort_limbs.py -- eligible + ineligible equals every truth event, and no event
#   carries a missed flag while its own eligibility is False.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402
import abort_events as E       # noqa: E402


def axis_split(events, axis):
    """(fire rows, ALL truth rows, ELIGIBLE truth rows) for one axis, each as (event_index, event).
    THE TWO TRUTH SETS ANSWER TWO DIFFERENT QUESTIONS, and conflating them was a real defect in this
    instrument on 2026-09-04. A FALSE abort asks whether the truth crossing EXISTED at all, so it is tested
    against EVERY truth event -- whether the guard happened to be armed for it has no bearing on whether it
    happened. A MISSED abort asks whether the guard failed to act, so it is tested only against truth events
    the guard could actually have fired on. Using the eligible set for both turned correctly-caught fires into
    false positives and inflated the vertical FALSE count more than sixfold before it was caught."""
    fires = [(i, e) for i, e in enumerate(events) if e["kind"] == E.FIRE_KIND[axis]]
    every, elig = [], []
    for i, e in enumerate(events):
        if e["kind"] != E.TRUTH_KIND[axis]:
            continue
        ok, why = eligible(e, axis)
        e["truth_eligible"], e["ineligible_reason"] = ok, (why or None)
        every.append((i, e))
        if ok:
            elig.append((i, e))
    return fires, every, elig


def eligible(e, axis):
    """Could this truth event have fired the guard AT ALL? Two conditions, both read from the producer:
    the guard must have been ARMED (the controller evaluates both guards only inside FORMATION_STATES), and
    for the longitudinal guard the armed stretch must be at least its own persist gate, since 10 s of arming
    split into two 5 s pieces never fires. A truth crossing while the aircraft was already in an emergency
    breakaway is therefore NOT a missed abort -- the guard was never asked. Measured, not assumed: the
    per-event arming coverage comes from this dispatch's own tick pass.
    The VERTICAL guard additionally only counts inside the 10 m window of record."""
    if e.get("armed_ticks") is None:
        return False, "no arming coverage measured"
    if axis == "vertical":
        if e["min_range_m"] is None or e["min_range_m"] > A.WINDOW_M:
            return False, "outside the %.1f m window of record" % A.WINDOW_M
        if e["armed_ticks"] < 1:
            return False, "guard disarmed for the whole crossing (state not in FORMATION_STATES)"
        return True, ""
    if e.get("armed_max_run_s", 0.0) < A.OT_PERSIST_S:
        return False, ("guard armed for at most %.2f s of the pass, under its own %.1f s persist gate"
                       % (e.get("armed_max_run_s", 0.0), A.OT_PERSIST_S))
    return True, ""


def flag(events):
    """Mark each event row with its class at every tolerance. A truth event the guard could never have fired
    on is left UNFLAGGED and carries the reason -- it was never a candidate, and a null says so rather than a
    False, which would read as 'the guard got this one right'."""
    for e in events:
        for tol in A.TOLS:
            e["is_false_tol%s" % tol], e["is_missed_tol%s" % tol] = None, None
    for axis in ("vertical", "longitudinal"):
        fires, every, elig = axis_split(events, axis)
        ft = [e["t0"] for _, e in fires]
        sp_all = [(e["t0"], e["t1"]) for _, e in every]
        sp_elig = [(e["t0"], e["t1"]) for _, e in elig]
        for tol in A.TOLS:
            f_i, _ = E.classify(ft, sp_all, tol)        # FALSE: did the crossing exist at all?
            _, m_i = E.classify(ft, sp_elig, tol)       # MISSED: could the guard have fired, and did it not?
            for k, (i, _) in enumerate(fires):
                events[i]["is_false_tol%s" % tol] = k in set(f_i)
            for k, (i, _) in enumerate(elig):
                events[i]["is_missed_tol%s" % tol] = k in set(m_i)
    return events
