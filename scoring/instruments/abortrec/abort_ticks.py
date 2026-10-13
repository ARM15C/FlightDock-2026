#!/usr/bin/env python3
# THE PER-TICK PASS -- the only part of this dispatch that opens the 20 Hz telemetry.
# RESPONSIBILITY: for each leg, (a) read the TRUTH geometry at each event's own time so every event row can say
#   whether it happened inside the 10 m window of record, and (b) cut a +-30 s trace about each emergency
#   breakaway entry so the recovery can be READ rather than inferred from the state machine's constants.
#   WHY THE EVENTS THEMSELVES COME FROM ELSEWHERE: the event list is the banked AMPSAFETY-20260903 summary,
#   whose per-leg breakaway count equals the readout's guards.total on all 3,600 legs (measured 2026-09-04).
#   This pass ADDS geometry and traces to a proven event list; it does not re-detect events, so the two
#   instruments cannot drift apart.
#   THE TRACE SPAN IS SIM TIME, NOT TICKS: dt is 0.05 s (20 Hz, measured), so +-30 s is about 1,201 ticks, but a
#   leg whose clock stalls yields fewer and that is recorded in n_ticks rather than padded.
#   A BREAKAWAY WHOSE WINDOW RUNS OFF EITHER END OF THE LEG IS KEPT, TRUNCATED, AND FLAGGED -- dropping it would
#   silently bias the recovery read toward breakaways that had room to recover.
#   T1: `up_m` = -raw_below_m is height above the BOOM TIP; crossing 2.00 m is a safety-EXCLUSION-ZONE breach.
# PUBLIC API: TRACE_COLS; _armed(d, t0, t1); _trace(...); leg_pass(args) -> (geom rows, trace rows)
# SIDE-EFFECTS: none -- pure computation over one leg. The WRITING half lives in abort_pass.py, split out
#   at this file's declared soft cap rather than raising it (caps are split, never raised).
# SIZE: soft 85 / hard 100 code-LoC; function cap 25. (Soft was declared at 70 before the guard-arming read
#   existed; recalibrated ONCE to 85 and stated. The HARD cap is not raised: that one is split.)
# COVERAGE: limb A3 in abort_limbs.py -- every breakaway in the event table has a trace, every trace's t_rel_s
#   spans the requested window unless the leg itself ended, and every event's range is finite or explicitly null.
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402
import abort_events as E       # noqa: E402

_STATES = A.formation_states()   # the states in which the guards are evaluated at all, read from the source

_GEOM = ["%s_%s" % (c, g) for c in ("raw", "env", "ctrl")
         for g in ("below_m", "behind_m", "cross_m", "range_m")]
TRACE_COLS = (["sim_time_s", "fsm_state", "_safety_emergency_active", "cmd_speed_mps", "cmd_alt_msl_m",
               "l_vfr_tas_mps", "l_vfr_gs_mps", "f_vfr_alt_m", "f_vfr_climb_mps", "f_vfr_throttle_pct",
               "_guard_altitude_ceiling_fire_count", "_guard_hard_overtake_fire_count"] + _GEOM)


def _ticks(leg_dir):
    """One leg's ticks, numeric where numeric. Refuses an empty frame rather than summarising it as clean."""
    paths = sorted(glob.glob(os.path.join(leg_dir, "env*", "flight_state_ep*.csv.gz")))
    if not paths:
        raise ValueError("no telemetry under %s" % leg_dir)
    have = set(pd.read_csv(paths[0], nrows=0).columns)
    use = [c for c in TRACE_COLS if c in have]
    d = pd.concat([pd.read_csv(p, usecols=use, low_memory=False) for p in paths], ignore_index=True)
    for c in use:
        if c != "fsm_state":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if d.empty:
        raise ValueError("telemetry present but ZERO rows under %s" % leg_dir)
    return d.sort_values("sim_time_s").reset_index(drop=True)


def _at(d, t):
    """Truth and estimator geometry at the tick nearest event time t. Nearest, not interpolated: a tick IS the
    measurement, and interpolating across a 0.05 s gap would invent a value the producer never wrote."""
    i = int(np.abs(d.sim_time_s.to_numpy(float) - float(t)).argmin())
    r = d.iloc[i]
    out = {"tick_idx_at_event": i, "t_tick_s": float(r.sim_time_s), "fsm_state_at_event": r.get("fsm_state")}
    for c in _GEOM:
        out[c + "_at_event"] = (float(r[c]) if c in d.columns and np.isfinite(r[c]) else None)
    rng = out.get("raw_range_m_at_event")
    out["inside_10m"] = (None if rng is None else bool(rng <= A.WINDOW_M))
    out["up_m_at_event"] = (None if out.get("raw_below_m_at_event") is None else -out["raw_below_m_at_event"])
    return out


def _armed(d, t0, t1):
    """How much of a truth event's span the guard was ARMED for. Both guards are wrapped in
    `if self.state in FORMATION_STATES:`, so outside those states they are never asked and a truth crossing
    there cannot be a MISSED abort. armed_max_run_s is the longest CONTIGUOUS armed stretch, which is what the
    longitudinal guard's persist gate actually needs -- 10 s of arming split into two 5 s pieces never fires."""
    seg = d.loc[(d.sim_time_s >= t0) & (d.sim_time_s <= t1)]
    if seg.empty:
        return {"armed_ticks": 0, "span_ticks": 0, "armed_frac": None, "armed_max_run_s": 0.0}
    a = seg.fsm_state.astype(str).isin(_STATES).to_numpy()
    t = seg.sim_time_s.to_numpy(float)
    best, edges = 0.0, np.flatnonzero(np.diff(np.concatenate(([False], a, [False]))))
    for i, j in zip(edges[0::2], edges[1::2]):
        best = max(best, float(t[j - 1] - t[i]))
    return {"armed_ticks": int(a.sum()), "span_ticks": int(len(a)),
            "armed_frac": float(a.mean()), "armed_max_run_s": round(best, 3)}


def _trace(d, t, e, tag, name, k):
    """The +-30 s cut about one breakaway. A window that runs off either end of the leg is KEPT, truncated and
    flagged -- dropping it would bias the recovery read toward breakaways that had room to recover."""
    seg = d.loc[(t >= e["t0"] - A.TRACE_SPAN_S) & (t <= e["t1"] + A.TRACE_SPAN_S)].copy()
    seg.insert(0, "t_rel_s", seg.sim_time_s - e["t0"])
    seg.insert(0, "event_idx", k), seg.insert(0, "leg", name), seg.insert(0, "tag", tag)
    seg["in_breakaway"] = (seg.sim_time_s >= e["t0"]) & (seg.sim_time_s <= e["t1"])
    seg["truncated_pre"] = bool(t[0] > e["t0"] - A.TRACE_SPAN_S)
    seg["truncated_post"] = bool(t[-1] < e["t1"] + A.TRACE_SPAN_S)
    return seg


def leg_pass(args):
    """One leg -> (event geometry rows, trace rows). A failure is RECORDED on the leg, never dropped."""
    tag, leg_dir, row = args
    name = row["leg"]
    try:
        d = _ticks(leg_dir)
    except Exception as e:
        return [{"tag": tag, "leg": name, "error": str(e)[:200]}], []
    ev, geo, tr = E.leg_events(row), [], []
    t = d.sim_time_s.to_numpy(float)
    for k, e in enumerate(ev):
        g = dict({"tag": tag, "leg": name, "event_idx": k, "kind": e["kind"]}, **_at(d, e["t0"]))
        if not e["is_fire"] and e["kind"] != "breakaway":
            g.update(_armed(d, e["t0"], e["t1"]))
        geo.append(g)
        if e["kind"] != "breakaway":
            continue
        tr.append(_trace(d, t, e, tag, name, k))
    return geo, tr
