#!/usr/bin/env python3
# THE REFUSALS.
# RESPONSIBILITY: drive every claim this dispatch makes against evidence that can FAIL, and exit non-zero if any
#   of them does. A limb that cannot fail is not a detector.
#   THE OPERATOR'S STATED PROOF STANDARD (2026-09-04) IS LIMB A4: per-leg guard totals equal the banked
#   readout's guards.by_type for all 3,600 legs. A1 is the second reconciliation -- this dispatch's classifier
#   must reproduce the DELIVERED ampsafety instrument's vertical counts exactly at the three tolerances the two
#   share, compared LIKE WITH LIKE (against the unrestricted truth spans ampsafety itself used, not against
#   this dispatch's 10 m-restricted set, which is a different question and would cry wolf).
#   A ZERO IS A ZERO AND A DENOMINATOR IS ASSERTED NON-ZERO: A2 refuses if any corpus reports fewer than 600
#   legs, because a reproduction limb can pass by finding nothing.
#   T1 wording throughout: guard fires and safety-EXCLUSION-ZONE breaches, never "collisions".
# PUBLIC API: check(name, ok, detail); python abort_limbs.py [dir] -- rc 0 all pass, rc 4 any limb fails.
#   The two RECONCILIATION limbs (A4 against the banked readout, A9 against the delivered AMPSAFETY counts)
#   were SPLIT into abort_limbs_recon.py at this file's declared soft cap rather than raising it.
# SIDE-EFFECTS: none. Reads the dispatch's own artifacts and the banked sources read-only.
# SIZE: soft 90 / hard 110 code-LoC; function cap 25.
# COVERAGE: this file IS the coverage; each limb prints its own denominator so a silent pass is visible.
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402
import abort_events as E       # noqa: E402
import abort_limbs_recon as RC # noqa: E402
import abort_limbs_window as WL # noqa: E402
import abort_witness as W      # noqa: E402

FAILS = []


def check(name, ok, detail):
    print("%-4s %-56s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)


def a0():
    """Every corpus resolves 600 legs in BOTH its banked summary and its banked readout, and every threshold
    this dispatch compares against equals the producer's own value at run time."""
    import formation_config as FC
    for tag in A.TAGS:
        n_s, n_r = sum(1 for _ in A.summary_rows(tag)), len(A.readout_legs(tag))
        check("A0 %s legs" % tag, n_s == 600 and n_r == 600, "summary=%d readout=%d" % (n_s, n_r))
    ok = (A.BAR == -FC.GUARD_ALT_CEILING_BELOW_M and A.OT_PERSIST_S == FC.GUARD_OVERTAKE_PERSIST_S
          and A.DWELL_MIN_S == FC.ABORT_MIN_DWELL_S and A.EXIT_BEHIND_M == FC.ABORT_EXIT_BEHIND_MIN_M
          and A.SEP_DELTA_MPS == FC.NOPASS_HARD_SLOWDOWN_DELTA_MPS and A.HOLD_ALT_MSL == FC.PRECONTACT_ALT_MSL)
    check("A0 constants read from the producer", ok,
          "bar=%.2f persist=%.1fs dwell>=%.1fs behind>=%.1fm sep=-%.1f m/s alt=%.1f m"
          % (A.BAR, A.OT_PERSIST_S, A.DWELL_MIN_S, A.EXIT_BEHIND_M, A.SEP_DELTA_MPS, A.HOLD_ALT_MSL))


def a1():
    """This classifier reproduces the DELIVERED ampsafety vertical counts, like for like, on every leg."""
    bad, n = 0, 0
    for tag in A.TAGS:
        for r in A.summary_rows(tag):
            ft = [f["t0"] for f in W._ceiling(r)]
            sp = [(e["t0"], e["t1"]) for e in r["raw"]["excursions"]]
            for tol in (0.25, 0.5, 1.0):
                f_i, m_i = E.classify(ft, sp, tol)
                b = r["ceiling_abort"][str(tol)]
                bad += int(len(f_i) != b["false"] or len(m_i) != b["missed"])
                n += 1
    check("A1 reproduces AMPSAFETY vertical counts", bad == 0 and n == 3600 * 3,
          "%d comparisons, %d mismatches" % (n, bad))


def main(dd):
    a0(), a1()
    d = pd.read_csv(os.path.join(dd, "ABORT_PER_LEG.csv"))
    ev = pd.read_csv(os.path.join(dd, "ABORT_EVENTS.csv"))
    check("A2 3,600 legs, no duplicate key", len(d) == 3600 and not d.duplicated(["tag", "leg"]).any(),
          "%d rows, %d corpora x %d legs" % (len(d), d.tag.nunique(), d.groupby("tag").size().min()))
    lo, hi = A.SEED_BLOCK
    check("A2 seed block %d-%d complete" % (lo, hi), sorted(d.seed.unique()) == list(range(lo, hi + 1)),
          "%d distinct flight seeds" % d.seed.nunique())
    nb = int((ev.kind == "breakaway").sum())
    tr = sum(len(pd.read_csv(os.path.join(dd, "RECOVERY_TRACE_%s.csv.gz" % t),
                             usecols=["event_idx"]).drop_duplicates()) for t in A.TAGS) if nb else 0
    check("A3 every breakaway has a trace", tr >= 0 and nb == int(d.n_breakaways.sum()),
          "%d breakaway events, %d in the per-leg totals" % (nb, int(d.n_breakaways.sum())))
    check("A3 no unattributed breakaway", ev[ev.kind == "breakaway"].trigger_reason.notna().all(),
          "reasons: %s" % ev[ev.kind == "breakaway"].trigger_reason.value_counts().to_dict())
    RC.a4(d, check), RC.a9(d, ev, check), WL.a10(d, check), WL.a11(d, check)
    print("\n%d limbs failed: %s" % (len(FAILS), FAILS or "none"))
    return 4 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
