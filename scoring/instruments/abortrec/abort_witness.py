#!/usr/bin/env python3
# THE WITNESS READERS -- a CAP-DRIVEN STRUCTURAL SPLIT out of abort_events.py, moving no logic.

#   per-leg summary -- the ceiling guard's per-fire witnesses, the hard-overtake fires, and the emergency
#   breakaway episodes -- and attribute every breakaway to the guard that caused it.
#   WHY THE OVERTAKE FIRE TIME COMES FROM THE BREAKAWAY: the controller keeps a per-fire witness list for the
#   CEILING guard only (GUARDWIT); the hard-overtake guard has a counter and nothing else. Its fire time is
#   therefore read from the breakaway the fire caused, whose reason field is exactly what the banked readout's
#   guards.by_type counts -- so the two instruments cannot disagree about how many there were.
#   THE ONE-TICK JOIN, MEASURED NOT ASSUMED: the controller stamps the TRANSITION time while the fsm_state
#   column flips on the NEXT tick, so 52 of 2,596 episodes sit exactly one 0.05 s tick after their event
#   (measured 2026-09-04, all 52 at exactly one tick). The join is nearest-within-one-tick and REFUSES on
#   anything further -- an unattributed breakaway cannot be scored, and must not be quietly filed as one.
#   T1: a fire is a guard fire; crossing 2.00 m is a safety-EXCLUSION-ZONE breach, never a "collision".
# PUBLIC API: TICK_S; _ceiling(row); _overtake(row); _trigger(row, t0); breakaway_rows(row)
# SIDE-EFFECTS: none. Pure reads over one banked summary row.
# SIZE: soft 45 / hard 60 code-LoC; function cap 25.
# COVERAGE: limb A1 in abort_limbs.py -- all 2,596 breakaways attribute to a reason, and the per-leg totals
#   reconcile to the banked readout's guards.by_type on all 3,600 legs.
TICK_S = 0.05                    # the producer's telemetry cadence, measured (20 Hz)


def _ceiling(row):
    """The ceiling guard's per-fire witnesses. t_sim_s is the fire time; the channel field records WHICH
    channel the guard compared, which is the perception half of a false abort."""
    return [{"t0": f["t_sim_s"], "t1": f["t_sim_s"], "peak_up_m": None, "min_range_m": None,
             "compared_below_m": f.get("compared_below_m"), "guard_channel": f.get("channel"),
             "fused_below_m": f.get("fused_below_m"), "gps_below_m": f.get("gps_below_m"),
             "behind_m": f.get("behind_m")}
            for f in row.get("ceiling_fires", []) if f.get("t_sim_s") is not None]


def _overtake(row):
    """Hard-overtake fires. The controller keeps no per-fire witness list for this guard (only a counter), so
    the fire TIME comes from the breakaway the fire caused -- safety_emergency_events, whose reason field the
    banked readout's guards.by_type is itself a count of."""
    return [{"t0": e["sim_time_s"], "t1": e["sim_time_s"], "from_state": e.get("from_state")}
            for e in row.get("safety_emergency_events", []) if e.get("reason") == "guard_hard_overtake"]


def _trigger(row, t0):
    """Which guard caused this breakaway. The controller stamps the TRANSITION time while the state column
    flips on the NEXT tick, so 52 of 2,596 episodes sit exactly one 0.05 s tick later than their event
    (measured 2026-09-04). The join is therefore nearest-within-one-tick, and an episode that matches nothing
    REFUSES rather than being filed as an unattributed breakaway."""
    best, dt = None, None
    for e in row.get("safety_emergency_events", []):
        d = abs(float(e["sim_time_s"]) - float(t0))
        if dt is None or d < dt:
            best, dt = e, d
    if best is None or dt > TICK_S + 1e-6:
        raise SystemExit("REFUSE rc=5: breakaway at t=%.3f on %s matches no safety_emergency_event (nearest "
                         "%s s away) -- an unattributed breakaway cannot be scored" % (t0, row["leg"], dt))
    return {"trigger_reason": best.get("reason"), "trigger_from_state": best.get("from_state"),
            "trigger_join_dt_s": round(dt, 4)}


def breakaway_rows(row):
    """Every emergency-breakaway episode as an event row, each carrying the guard that triggered it."""
    return [{"kind": "breakaway", "axis": "both", "is_fire": False, "t0": e["t0"], "t1": e["t1"],
             **_trigger(row, e["t0"]),
             "dur_s": e["dwell_s"], "exit_state": e.get("exit_state"),
             "dwell_meets_min": e.get("dwell_meets_min"), "exit_is_intercept": e.get("exit_is_intercept"),
             "behind_at_exit_m": e.get("behind_at_exit_m"), "up_at_exit_m": e.get("up_at_exit_m"),
             "t_to_behind15_s": e.get("t_to_behind15_s"),
             "t_to_up_under_bar_s": e.get("t_to_up_under_bar_s"),
             "cmd_speed_minus_lead_tas_mean": e.get("cmd_speed_minus_lead_tas_mean"),
             "cmd_alt_mean_m": e.get("cmd_alt_mean_m")}
            for e in row.get("recovery", [])]
