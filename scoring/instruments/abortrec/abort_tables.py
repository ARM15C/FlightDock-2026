#!/usr/bin/env python3
# THE METRIC TABLE -- rates per arm, venue, wind rung and trajectory.
# RESPONSIBILITY: reduce ABORT_PER_LEG.csv to one tidy row per (arm, venue, pool, trajectory, metric), so a
#   reader can see the same quantity at the pooled level and rung by rung without re-deriving anything.
#   THE POOLS ARE THE GRADING POOLS OF RECORD, and each is named on its own row rather than assumed: all ten
#   rungs; the nine WIND rungs (S0 excluded); strong wind = 6, 7 and 8 m/s, which is the W12 + W14 + W16 set --
#   a rung LABEL is not its unit, the labels are 2x the meters per second. Every individual rung is also
#   emitted, because the pooled number can hide a rung where the whole effect lives.
#   S0 IS THE ZERO-DISTURBANCE CONTROL STAGE (zero wind AND zero turbulence) and is excluded from the wind and
#   strong-wind pools by construction; it appears only in its own rung row and in the all-ten row.
#   DENOMINATORS TRAVEL WITH RATES: every rate row carries n_legs, and the legs that never came within 10 m of
#   the boom tip are counted on their own row rather than dropped -- that is a finding about the approach, not
#   a defect, and removing them would inflate every in-window rate.
#   CHANNEL (R5.1): truth channel raw_*, window of record <= 10.0 m of the boom tip, tolerance stated per row.
#   T1: these are guard fires and safety-EXCLUSION-ZONE breaches, never "collisions".
# PUBLIC API: POOLS; metrics(d); main(in_dir)
# SIDE-EFFECTS: writes ABORT_METRICS.csv under in_dir. Reads ABORT_PER_LEG.csv read-only.
# SIZE: soft 75 / hard 95 code-LoC; function cap 25.
# COVERAGE: limb A6 in abort_limbs.py -- every pool's n_legs equals the count its own membership rule selects,
#   and the all-ten pool's leg count is 600 for every corpus.
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402

STRONG = ("W12", "W14", "W16")          # 6, 7 and 8 m/s -- the label is 2x the meters per second
POOLS = ("all_ten_rungs", "wind_rungs", "strong_wind")


def _pool(d, pool):
    if pool == "wind_rungs":
        return d[d.wind_label != "S0"]
    if pool == "strong_wind":
        return d[d.wind_label.isin(STRONG)]
    return d


def metrics(d):
    """Every rate for one cell, as {metric: (value, unit)}. An empty cell returns nothing rather than zeros."""
    n = len(d)
    if not n:
        return {}
    out = {"legs": (n, "legs"),
           "legs never within 10 m of the boom tip": (int((~d.ever_within_window).sum()), "legs"),
           "docked successfully (M3b dwell)": (100.0 * d.m3b_success.mean(), "% of legs"),
           "fired the altitude-ceiling guard": (100.0 * (d.vertical_fires > 0).mean(), "% of legs"),
           "fired the hard-overtake guard": (100.0 * (d.longitudinal_fires > 0).mean(), "% of legs"),
           "entered an emergency breakaway": (100.0 * (d.n_breakaways > 0).mean(), "% of legs"),
           "altitude-ceiling fires per leg": (d.vertical_fires.mean(), "fires"),
           "hard-overtake fires per leg": (d.longitudinal_fires.mean(), "fires"),
           "emergency breakaways per leg": (d.n_breakaways.mean(), "events"),
           "truth excursions above 2.00 m inside 10 m, per leg": (d.vertical_truth_events.mean(), "events"),
           "truth passes of the no-pass line past the guard's gate, per leg":
               (d.longitudinal_truth_events.mean(), "events")}
    for tol in A.TOLS:
        for axis, word in (("vertical", "vertical"), ("longitudinal", "longitudinal")):
            f, m = "%s_false_tol%s" % (axis, tol), "%s_missed_tol%s" % (axis, tol)
            out["FALSE %s aborts per leg (tol %.2f s)" % (word, tol)] = (d[f].mean(), "events")
            out["MISSED %s aborts per leg (tol %.2f s)" % (word, tol)] = (d[m].mean(), "events")
            out["legs with a FALSE %s abort (tol %.2f s)" % (word, tol)] = (100.0 * (d[f] > 0).mean(), "% of legs")
            out["legs with a MISSED %s abort (tol %.2f s)" % (word, tol)] = (100.0 * (d[m] > 0).mean(), "% of legs")
    return out


def _cells(df):
    """Every (arm, venue, pool, trajectory) cell: the three grading pools plus each rung on its own."""
    for tag, g in df.groupby("tag"):
        for pool in POOLS:
            yield tag, pool, _pool(g, pool)
        for lab, gr in g.groupby("wind_label"):
            yield tag, "rung_%s" % lab, gr


def main(in_dir):
    df = pd.read_csv(os.path.join(in_dir, "ABORT_PER_LEG.csv"))
    rows = []
    for tag, pool, g in _cells(df):
        for traj in ("combined", "straight", "racetrack"):
            d = g if traj == "combined" else g[g.trajectory == traj]
            for k, (v, unit) in metrics(d).items():
                rows.append({"tag": tag, "policy": A.POLICY[tag], "sigma": A.SIGMA[tag], "pool": pool,
                             "trajectory": traj, "metric": k, "value": round(float(v), 4), "unit": unit,
                             "n_legs": len(d), "channel": "truth (raw), window <= %.1f m" % A.WINDOW_M,
                             "seed_block": "%d-%d" % A.SEED_BLOCK})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(in_dir, "ABORT_METRICS.csv"), index=False)
    print("metric rows=%d cells=%d" % (len(out), out.groupby(["tag", "pool", "trajectory"]).ngroups))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
