#!/usr/bin/env python3
# =============================================================================

#   for every ALTITUDE-CEILING-triggered breakaway in this run's tick-level pass:
#     BRK.climbrate -- median follower climb rate over the second before the guard fired (f_vfr_climb_mps, trace);
#     BRK.arrest_s -- median time from the fire to the peak height above the tip inside the breakaway (trace);
#     BRK.cmd.<ARM> -- median APPLIED change to the altitude command (res_applied_2, per tick from the leg's
#                       flight_state record; positive = up, read from the producer) over the second before the fire.
#                       WHAT PRODUCES THAT CHANGE DIFFERS BY ARM AND EVERY PLACEHOLDER SAYS SO IN WORDS:
#                         A2 / A2O / S1 / S1O -- a LEARNED correction from the policy (A2O and S1O also carry the
#                                                fixed offset, so their figure is the two together);
#                         PIDO -- the FIXED 0.25 m down-offset ONLY. No policy acts: the checkpoint
#                                                is loaded and its output multiplied by zero. This is NOT a learned
#                                                correction and must never be tabulated or plotted as one;
#                         PID -- zero: no policy and no offset.
#   Reads the +-30 s recovery traces (RECOVERY_TRACE_<TAG>.csv.gz) and, for the applied change, each fire leg's
#   flight_state once. A2 and S1 are the policies; PIDO carries no policy but a measured 0.25 m offset; only the
#   plain PID-baseline's applied change is zero by construction and is reported as such, never read as a control
#   action. Turbulence 0.25/0.35 only (T5); T1 vocabulary.
# PUBLIC API: prefire(trace) -> DataFrame; applied_alt(legs_root, events) -> list; label(arm, values) -> str;
#   collect(run_dir); main(argv=None) -> int.
#   CLI: --run-dir DIR [--selftest]
# SIDE-EFFECTS: reads <run-dir>/ticklevel/RECOVERY_TRACE_*.csv.gz and the legs' flight_state files; writes
#   <run-dir>/verdicts/PREFIRE_PLACEHOLDERS.json and PREFIRE_EVENTS.csv. Spawns nothing.
# SIZE: soft 90 / hard 110 code-LoC; function cap 30.
# COVERAGE: --selftest: a synthetic trace with a known pre-fire climb rate and peak time reproduces both; the PIDO
#   placeholder text names the offset and denies a learned correction; the PID placeholder is the zero-by-construction line.
# =============================================================================
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

#: corpus-tag prefix -> shorthand. PIDO is read from its own ticks (2026-09-17): no policy acts, but the exam-time
#: 0.25 m down-offset makes its APPLIED altitude change a measured quantity, not zero. Only the plain PID-baseline
#: is asserted zero, and that is by construction (ACTION_SCALES zero and no offset).
ARM = {"A2P": "A2", "A2V": "A2O", "S1A": "S1", "CSV": "S1O", "PIDO": "PIDO"}
#: what PRODUCES the applied change, in words, so no table or figure can present the offset as a learned correction.
SOURCE = {"A2": "learned correction", "S1": "learned correction",
          "A2O": "learned correction plus the fixed 0.25 m down-offset",
          "S1O": "learned correction plus the fixed 0.25 m down-offset",
          "PIDO": "the fixed 0.25 m down-offset ONLY -- no policy acts, NOT a learned correction"}


def prefire(trace):
    """Per ceiling-triggered event: pre-fire climb rate and time to the peak height above the tip."""
    rows = []
    for (leg, k), g in trace.groupby(["leg", "event_idx"]):
        pre = g[(g.t_rel_s >= -1.0) & (g.t_rel_s <= 0.0)]
        post = g[(g.t_rel_s >= 0.0) & g.in_breakaway.astype(str).str.lower().eq("true")]
        if pre.empty or post.empty:
            continue
        up = -post.raw_below_m.astype(float)
        rows.append({"leg": leg, "event_idx": k, "climb_pre_mps": float(pre.f_vfr_climb_mps.astype(float).median()),
                     "t_peak_s": float(post.t_rel_s.iloc[int(np.nanargmax(up.values))]), "sim_t_fire": float(g[g.t_rel_s >= 0].sim_time_s.iloc[0])})
    return pd.DataFrame(rows)


def applied_alt(legs_root, ev):
    """Median applied change to the altitude command over the second before each fire, from the leg's flight_state.
    For PIDO this change IS the fixed offset, not a learned correction; the caller labels it."""
    out = []
    for leg, g in ev.groupby("leg"):
        fs = glob.glob(os.path.join(legs_root, leg, "env*", "flight_state_ep*.csv.gz"))
        if not fs:
            continue
        d = pd.read_csv(fs[0], usecols=["sim_time_s", "res_applied_2"], compression="gzip")
        for t in g.sim_t_fire:
            w = d[(d.sim_time_s >= t - 1.0) & (d.sim_time_s <= t)]
            if len(w):
                out.append(float(w.res_applied_2.median()))
    return out


def label(arm, v):
    """The placeholder text for one arm: the measured median in meters PLUS, in words, what produced it (SOURCE)."""
    if arm == "PID":
        return "0.00 m (no policy and no offset: zero by construction)"
    return ("%.2f m (%s)" % (float(np.median(v)), SOURCE[arm])) if v else "no fires (%s)" % SOURCE[arm]


def selftest():
    t = pd.DataFrame({"leg": "L", "event_idx": 0, "t_rel_s": np.arange(-2.0, 3.0, 0.5), "sim_time_s": np.arange(100.0, 105.0, 0.5),
                      "f_vfr_climb_mps": [0.1, 0.2, 0.4, 0.4, 0.4, 0.2, 0.0, -0.5, -1.0, -1.0],
                      "raw_below_m": [-1.5, -1.7, -1.9, -2.0, -2.05, -2.10, -2.0, -1.8, -1.5, -1.2],
                      "in_breakaway": [False, False, False, False, True, True, True, True, True, True], "_guard_altitude_ceiling_fire_count": 1})
    r = prefire(t)
    ok = len(r) == 1 and abs(r.climb_pre_mps.iloc[0] - 0.4) < 1e-9 and abs(r.t_peak_s.iloc[0] - 0.5) < 1e-9
    lp = label("PIDO", [-0.25, -0.25, -0.24])            # the offset must be named and a learned correction denied
    ok = ok and lp.startswith("-0.25 m") and "down-offset" in lp and "NOT a learned correction" in lp
    ok = ok and "learned correction" in label("A2", [-0.3]) and "offset" not in label("A2", [-0.3])
    ok = ok and label("PID", []).startswith("0.00 m") and "zero by construction" in label("PID", [])
    print("SELFTEST", "PASS" if ok else "FAIL", r.to_dict("records"))
    return 0 if ok else 1


def collect(run_dir):
    """Every ceiling-triggered breakaway across the twelve corpora: pre-fire rows and the per-arm medians of the
    applied altitude-command change."""
    tl, frames, res = os.path.join(run_dir, "ticklevel"), [], {}
    ev_all = pd.read_csv(os.path.join(tl, "ABORT_EVENTS.csv"))
    ev_all = ev_all[(ev_all.kind == "breakaway") & (ev_all.trigger_reason == "guard_altitude_ceiling")]
    for p in sorted(glob.glob(os.path.join(tl, "RECOVERY_TRACE_*.csv.gz"))):
        tag = os.path.basename(p)[len("RECOVERY_TRACE_"):-len(".csv.gz")]
        tr = pd.read_csv(p).merge(ev_all[ev_all.tag == tag][["leg", "event_idx"]], on=["leg", "event_idx"])
        r = prefire(tr)
        r["tag"] = tag
        frames.append(r)
        arm = ARM.get(tag.rstrip("0123456789"))
        if arm and len(r):
            res.setdefault(arm, []).extend(applied_alt(os.path.join(run_dir, "shims", tag, "flat"), r))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ev, res = collect(a.run_dir)
    T = {}
    if len(ev):
        T = {"BRK.climbrate": "%.2f m/s" % ev.climb_pre_mps.median(), "BRK.arrest_s": "%.1f s" % ev.t_peak_s.median(), "BRK.n_events": int(len(ev))}
    for arm in list(SOURCE) + ["PID"]:
        T["BRK.cmd.%s" % arm] = label(arm, res.get(arm, []))
        T["BRK.cmdsource.%s" % arm] = SOURCE.get(arm, "zero by construction")
    vd = os.path.join(a.run_dir, "verdicts")
    os.makedirs(vd, exist_ok=True)
    ev.to_csv(os.path.join(vd, "PREFIRE_EVENTS.csv"), index=False)
    json.dump(T, open(os.path.join(vd, "PREFIRE_PLACEHOLDERS.json"), "w"), indent=1)
    print(T)
    return 0


if __name__ == "__main__":
    sys.exit(main())
