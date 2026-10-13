#!/usr/bin/env python3
# THE TWO RECONCILIATION LIMBS -- a CAP-DRIVEN STRUCTURAL SPLIT out of abort_limbs.py, moving no logic.

#   it -- the banked READOUT of record and the delivered AMPSAFETY
#   abort classifier (A9). Both take the caller's `check` so every limb reports through one tally.
#   A4 IS THE PROOF STANDARD THE OPERATOR SET on 2026-09-04: per-leg guard totals equal the readout's
#   guards.by_type for all 3,600 legs.
#   A9 IS THE LIMB THAT WOULD HAVE CAUGHT THIS INSTRUMENT'S OWN DEFECT: the vertical FALSE count must equal
#   AMPSAFETY's banked false count on every leg, because both ask the same question. The MISSED counts are
#   deliberately NOT compared -- this dispatch adds the 10 m window and the guard-arming gate, so it answers a
#   narrower question by design, and a limb that compared them would cry wolf. A limb that cries wolf is
#   ignored, so it is scoped to the comparison that is genuinely like for like.
# PUBLIC API: a4(d, check); a9(d, ev, check)
# SIDE-EFFECTS: none. Reads the banked sources read-only.
# SIZE: soft 70 / hard 85 code-LoC; function cap 25. (Declared 45/60 before the window limb existed;
#   calibrated ONCE and stated. As promised there, the next growth was SPLIT, not raised: the two CHANNEL
#   limbs A10/A11 moved to abort_limbs_window.py.)
# COVERAGE: driven by abort_limbs.main on every run; each limb prints its own denominator so a silent pass is
#   visible rather than assumed.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402


def a4(d, check):
    """The s guards.by_type."""
    tot = (d.n_breakaways == d.readout_guards_total)
    brk = (d.n_breakaways == d.readout_breakaways)
    ceil_ = (d.vertical_fires == d.readout_guard_ceiling)
    over = (d.longitudinal_fires == d.readout_guard_overtake)
    check("A4 breakaways == readout guards.total", bool(tot.all()), "%d/%d legs" % (int(tot.sum()), len(d)))
    check("A4 breakaways == readout emergency_breakaways", bool(brk.all()), "%d/%d" % (int(brk.sum()), len(d)))
    check("A4 ceiling fires == by_type guard_altitude_ceiling", bool(ceil_.all()),
          "%d/%d legs (totals %d vs %d)" % (int(ceil_.sum()), len(d), int(d.vertical_fires.sum()),
                                            int(d.readout_guard_ceiling.sum())))
    check("A4 overtake fires == by_type guard_hard_overtake", bool(over.all()),
          "%d/%d legs (totals %d vs %d)" % (int(over.sum()), len(d), int(d.longitudinal_fires.sum()),
                                            int(d.readout_guard_overtake.sum())))


def a9(d, ev, check):
    """Two invariants that would have caught the 2026-09-04 defect in this instrument.
    (i) The vertical FALSE count must equal AMPSAFETY's own banked false count on every leg -- both ask the
        same question (a fire with no truth crossing), so any difference means a comparison set drifted.
        The MISSED counts are NOT compared: this dispatch adds the 10 m window and the guard-arming gate, so
        they answer a narrower question by design, and a limb that compared them would cry wolf.
    (ii) No event may carry a MISSED flag while its own eligibility is False -- that combination is exactly
        the conflation that inflated the vertical FALSE count more than sixfold before it was caught."""
    amp = {}
    for tag in A.TAGS:
        for r in A.summary_rows(tag):
            amp[(tag, r["leg"])] = r["ceiling_abort"][str(A.TOL_PRIMARY)]["false"]
    k = d.set_index(["tag", "leg"])["vertical_false_tol%s" % A.TOL_PRIMARY]
    same = sum(1 for key, v in k.items() if amp.get(key) == v)
    check("A9 vertical FALSE == AMPSAFETY banked false", same == len(k),
          "%d/%d legs, totals %d vs %d" % (same, len(k), int(k.sum()), sum(amp.values())))
    t = ev[ev.kind.isin(["truth_vertical_excursion", "truth_overtake_run"])]
    bad = t[(t.truth_eligible == False) & t["is_missed_tol%s" % A.TOL_PRIMARY].notna()]      # noqa: E712
    check("A9 no MISSED flag on an ineligible truth event", len(bad) == 0,
          "%d truth events, %d eligible, %d flagged while ineligible"
          % (len(t), int((t.truth_eligible == True).sum()), len(bad)))                       # noqa: E712
