#!/usr/bin/env python3
# WHAT THE BREAKAWAY IS DESIGNED TO DO, AND WHAT IT DID.
# RESPONSIBILITY: reduce the +-30 s traces to one row per emergency breakaway, comparing the measured manoeuvre
#   against the architecture's own contract -- every constant READ from formation_config, never paraphrased.
#   THE CONTRACT, quoted from baseline_formation_controller._handle_emergency_breakaway:
#     command = self._windff_lead_gs_air(lead_state.ground_speed_ms) - NOPASS_HARD_SLOWDOWN_DELTA_MPS
#     altitude = PRECONTACT_ALT_MSL
#     exit when dwell_ok (>= ABORT_MIN_DWELL_S) AND behind_ok (>= ABORT_EXIT_BEHIND_MIN_M)
#                AND violation_cleared (geom.behind_m >= 0), then transition to INTERCEPT.
#   THE EXIT GATE IS MEASURED ON ITS OWN CHANNEL: the handler tests `geom.behind_m`, which is the FUSED
#   channel, and the tick that satisfies the gate is already stamped as the transition. Reading truth at the
#   last in-state tick therefore understates conformance badly (12.3 % vs 100.0 %, measured 2026-09-04).
#   Truth is reported beside it, because the gap between the two is the perception quantity of interest.
#   THE FINDING THAT FALLS OUT OF READING IT: all three exit gates are LONGITUDINAL. There is no vertical exit
#   criterion anywhere in the handler, so a breakaway triggered by the ALTITUDE-ceiling guard has no condition
#   that requires the aircraft to be back under the 2.00 m bar before it exits. Whether the excursion cleared
#   is therefore MEASURED here, never assumed from the state machine.
#   T2: the commanded separation is built from an AIR-frame lead reference, so the conformance comparator is
#   the leader's TAS. The ground-speed form of the same difference is a wind artefact and is carried only as a
#   labelled column, never as the contract read. Wind is never added to an airspeed.
#   T1: 2.00 m is the safety-EXCLUSION-ZONE bar; a breakaway is a guard fire, never a "collision".
# PUBLIC API: contract_row(seg, ev); _commanded(seg, inside); _exit_gate(seg, t, behind, t_exit)
#   (the driver that walks the traces lives in abort_recovery_run.py -- the same reduction-vs-I/O split this
#    dispatch already uses for abort_ticks / abort_pass.)
# SIDE-EFFECTS: none -- pure reduction of one trace segment. The writing half is abort_recovery_run.py.
# SIZE: soft 75 / hard 95 code-LoC; function cap 25.
# COVERAGE: limb A8 in abort_limbs.py -- one summary row per breakaway event in ABORT_EVENTS.csv, no more and
#   no fewer, and every contract column is null-or-finite rather than silently zero.
import os
import sys

import numpy as np
import pandas as pd  # noqa: F401 (callers pass pandas objects; kept for the type contract)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402


def _first_after(t, mask, t_exit):
    """Seconds from exit until `mask` first holds. None means it never held inside the trace window, which is
    a different statement from 0.0 and must not be written as one."""
    w = np.flatnonzero(mask & (t >= t_exit))
    return (float(t[w[0]] - t_exit) if w.size else None)


def _commanded(seg, inside):
    """What was COMMANDED during the breakaway, against the handler's two constants. T2: the contract is built
    from an AIR-frame lead reference, so the conformance read differences against the leader's TAS; the
    ground-speed form of the same difference is a wind artefact and is labelled as one."""
    if not inside.any():
        return {}
    cmd, tas = seg.cmd_speed_mps.to_numpy(float), seg.l_vfr_tas_mps.to_numpy(float)
    gs = seg.l_vfr_gs_mps.to_numpy(float)
    return {"cmd_minus_lead_tas_mean": float(np.nanmean((cmd - tas)[inside])),
            "cmd_minus_lead_gs_mean_ARTEFACT": float(np.nanmean((cmd - gs)[inside])),
            "contract_speed_delta_mps": -A.SEP_DELTA_MPS,
            "cmd_alt_mean_msl_m": float(np.nanmean(seg.cmd_alt_msl_m.to_numpy(float)[inside])),
            "contract_alt_msl_m": A.HOLD_ALT_MSL}


def _entry(ev, t, inside):
    """The identity and dwell half of the row, split out of contract_row at its declared function cap."""
    t_exit = float(t[inside].max()) if inside.any() else float(ev["t1"])
    return t_exit, {"tag": ev["tag"], "leg": ev["leg"], "event_idx": ev["event_idx"],
                    "trigger_reason": ev["trigger_reason"], "dwell_s": ev["dur_s"],
                    "dwell_meets_min": bool(ev["dur_s"] >= A.DWELL_MIN_S),
                    "contract_dwell_min_s": A.DWELL_MIN_S, "exit_state": ev["exit_state"],
                    "exit_is_intercept": ev["exit_state"] == "INTERCEPT"}


def _exit_gate(seg, t, behind, t_exit):
    """THE EXIT GATE IS READ ON THE CHANNEL THE HANDLER ITSELF READS, AT THE TICK THE TRANSITION TAKES EFFECT.
    `geom.behind_m` is the FUSED channel, and the tick that satisfies the gate is already labelled as the
    transition -- so measuring truth at the last in-state tick reads the gate a fraction early and on the
    wrong channel. Measured 2026-09-04: last-in-state on truth reads 12.3 % over the 15.0 m gate, while
    first-out-of-state on the fused channel reads 100.0 %. The contract is honoured exactly; the earlier
    number was an artefact of the reader, not a breach by the controller. Truth travels beside it, because
    the gap between the two IS the perception-vs-truth quantity this dispatch is for."""
    out = t > t_exit
    r = {"behind_at_exit_m": (float(behind[t <= t_exit][-1]) if (t <= t_exit).any() else None),
         "behind_fused_first_out_m": (float(seg.ctrl_behind_m.to_numpy(float)[out][0]) if out.any() else None),
         "behind_truth_first_out_m": (float(behind[out][0]) if out.any() else None),
         "exit_gate_channel": "fused (ctrl_behind_m) -- the channel _handle_emergency_breakaway reads"}
    r["behind_meets_exit_gate"] = (None if r["behind_fused_first_out_m"] is None
                                   else bool(r["behind_fused_first_out_m"] >= A.EXIT_BEHIND_M))
    return r


def contract_row(seg, ev):
    """One breakaway measured against the handler's own constants."""
    t = seg.sim_time_s.to_numpy(float)
    up = -seg.raw_below_m.to_numpy(float)
    behind = seg.raw_behind_m.to_numpy(float)
    inside = seg.in_breakaway.to_numpy(bool)
    pre = t < float(ev["t0"])
    t_exit, r = _entry(ev, t, inside)
    r.update({"trace_ticks": int(len(seg)), "truncated_pre": bool(seg.truncated_pre.iloc[0]),
              "truncated_post": bool(seg.truncated_post.iloc[0])})
    r.update(_commanded(seg, inside))
    r["up_at_entry_m"] = float(up[pre][-1]) if pre.any() else None
    r["up_max_in_breakaway_m"] = float(np.nanmax(up[inside])) if inside.any() else None
    r["up_at_exit_m"] = float(up[t <= t_exit][-1]) if (t <= t_exit).any() else None
    r.update(_exit_gate(seg, t, behind, t_exit))
    # THE VERTICAL QUESTION THE STATE MACHINE NEVER ASKS: was the aircraft under the bar when it exited, and
    # if not, how long after exit did it take? A null here means it never cleared inside the +-30 s window.
    r["under_bar_at_exit"] = (None if r["up_at_exit_m"] is None else bool(r["up_at_exit_m"] <= A.BAR))
    r["t_to_under_bar_after_exit_s"] = _first_after(t, up <= A.BAR, t_exit)
    r["t_to_behind15_after_exit_s"] = _first_after(t, behind >= A.EXIT_BEHIND_M, t_exit)
    r["up_mean_30s_after_exit_m"] = (float(np.nanmean(up[(t > t_exit) & (t <= t_exit + A.TRACE_SPAN_S)]))
                                     if ((t > t_exit) & (t <= t_exit + A.TRACE_SPAN_S)).any() else None)
    r["contract_exit_gates"] = "longitudinal only: dwell >= %.1f s AND behind >= %.1f m AND behind >= 0" % (
        A.DWELL_MIN_S, A.EXIT_BEHIND_M)
    return r
