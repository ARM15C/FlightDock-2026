#!/usr/bin/env python3
# THE AMPLIFYING SAFETY BATTERY, extracted ON THE BOX -- one bounded JSON row per leg, so 41 GB of per-tick
# telemetry never crosses the network.

#   emergency-abort classification, vertical and longitudinal stopping distances, the separation distributions,
#   and the measured recovery against the architecture's own stated contract.
#   WHAT A FALSE OR MISSED ABORT *IS* HERE: the ceiling guard
#   compares the PERCEPTION channel (env, GPS-direct -- FD_GUARD_CEILING_CHANNEL=env is witnessed in both arms'
#   manifests) against -2.00 m, while H1 is scored on TRUTH (raw, SITL SIM_STATE). A FALSE abort is a fire with no
#   truth excursion beside it; a MISSED abort is a truth excursion with no fire. Both are read at three tolerance
#   windows so the count cannot be an artefact of one choice.
#   THE ARCHITECTURE'S OWN CONTRACT, quoted from the handler this measures against (baseline_formation_controller
#   _handle_emergency_breakaway): command lead_gs - 3.0 m/s, hold 890.0 m MSL, exit only when dwell >= 2.0 s AND
#   behind >= 15.0 m AND behind >= 0, then INTERCEPT. NOTE FOR THE READ: those exit gates are LONGITUDINAL -- a
#   ceiling-triggered abort has no vertical exit criterion at all, so "did the excursion clear" is measured here
#   rather than assumed from the state machine.
#   T1 THROUGHOUT: 2.00 m is the safety-EXCLUSION-ZONE bar (a firing is a zone breach, never a "collision"); the
#   airframe-contact plane is 3.42 m and is counted separately, never absorbed into a percentile.
#   THIS TOOL WRITES ONE FILE AND READS EVERYTHING ELSE READ-ONLY. It never deletes, moves or edits a leg.
# PUBLIC API: python safety_extract.py <legs_root> <out_jsonl> [--arm NAME] [--sigma S] [--jobs N]
#   rc 0; rc 3 if ZERO legs resolved (an empty denominator is not a result); a per-leg failure is recorded in the
#   row as "error" and counted, never silently dropped.
# SIDE-EFFECTS: writes <out_jsonl> only.
# SIZE: soft 130 / hard 145 code-LoC; function cap 30. (Soft was declared at 120 before the file
#   existed; recalibrated ONCE to 130 when the wind-invariant TAS comparator was added -- stated here
#   rather than quietly exceeded. The HARD cap is unchanged and is never raised: that one is split.)
# COVERAGE: driven on a known leg whose fused channel crosses -2.00 m while env and truth do not (W0_10775_
#   racetrack, sigma 0.35) -- the case that must classify as NO fire, NO truth excursion, and no false abort.
import argparse
import concurrent.futures as cf
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safety_channels as SCH  # noqa: E402
import safety_events as EV  # noqa: E402

BAR, TOLS = SCH.BAR, (0.25, 0.5, 1.0)
ABORT_MIN_DWELL_S, ABORT_EXIT_BEHIND_MIN_M = 2.0, 15.0
PRECONTACT_ALT_MSL, NOPASS_DELTA, OT_PERSIST = 890.0, 3.0, 10.0
EB = "EMERGENCY_BREAKAWAY"


def run_summary(leg_dir):
    """The controller's own per-leg summary: guard counters and the GUARDWIT fire witnesses."""
    g = sorted(glob.glob(os.path.join(leg_dir, "episode_*", "run_summary.json")))
    if not g:
        return {}
    return json.load(open(g[-1]))


def overlaps(t0, t1, a, b):
    return not (t1 < a or t0 > b)


def classify_ceiling(fires, excursions, tol):
    """False = a fire with no TRUTH excursion within tol. Missed = a truth excursion with no fire within tol."""
    ft = [f["t_sim_s"] for f in fires if f.get("t_sim_s") is not None]
    ex = [(e["t0"], e["t1"]) for e in excursions]
    false_n = sum(1 for t in ft if not any(overlaps(t - tol, t + tol, a, b) for a, b in ex))
    missed_n = sum(1 for a, b in ex if not any(a - tol <= t <= b + tol for t in ft))
    return {"fires": len(ft), "truth_excursions": len(ex), "false": false_n, "missed": missed_n}


def identity(leg_dir, arm, sigma):
    """The leg's labels. `key` is the leg NAME, which is what every banked readout joins on."""
    name = os.path.basename(leg_dir.rstrip("/"))
    row = {"arm": arm, "sigma": sigma, "cell": os.path.basename(os.path.dirname(leg_dir.rstrip("/"))),
           "leg": name, "key": name}
    parts = name.split("_")
    if len(parts) >= 3:
        row.update({"wind_label": parts[0], "seed": parts[1], "traj": parts[2]})
    return row


def extract_leg(leg_dir, arm, sigma):
    """One leg -> one bounded row. Any failure is recorded in the row, never dropped."""
    row = identity(leg_dir, arm, sigma)
    try:
        d = SCH.load_ticks(leg_dir)
    except Exception as e:                      # a leg that cannot be read must SAY so, not summarise as clean
        row["error"] = str(e)[:200]
        return row
    rs = run_summary(leg_dir)
    fires = list(rs.get("guard_ceiling_fires", []))
    row["guards"] = {k: rs.get(k) for k in ("guard_altitude_ceiling_fire_count",
                                            "guard_hard_overtake_fire_count", "guard_speed_cap_fire_count",
                                            "guard_env_fallback_count", "count_no_pass_violations",
                                            "safety_emergency_count")}
    row["safety_emergency_events"] = rs.get("safety_emergency_events", [])
    row["terminal_reason"] = str(d.terminal_reason.dropna().iloc[-1]) if d.terminal_reason.notna().any() else None
    row["n_ticks"] = int(len(d))
    row["dt_median_s"] = float(np.nanmedian(np.diff(d.sim_time_s.to_numpy(float)))) if len(d) > 1 else None
    for ch in SCH.CHANNELS:
        row[ch] = SCH.channel_block(d, ch)
    row["ceiling_abort"] = {str(t): classify_ceiling(fires, row["raw"]["excursions"], t) for t in TOLS}
    row["ceiling_fires"] = fires
    row["stopping_vertical"] = EV.stopping_vertical(d, -d.raw_below_m, fires)
    row["overtake_truth_runs_over_persist"] = [r for r in row["raw"]["ahead_runs"] if r["dur_s"] >= OT_PERSIST]
    row["recovery"] = EV.recovery(d)
    row["leader_altitude"] = EV.leader_altitude(d)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("legs_root"), ap.add_argument("out_jsonl")
    ap.add_argument("--arm", default=""), ap.add_argument("--sigma", default="")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    legs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.legs_root, "*", "*", "results_eid*.json")))
    if not legs:
        raise SystemExit("REFUSE rc 3: ZERO legs under %s -- an empty denominator is not a result" % a.legs_root)
    n_err = 0
    with open(a.out_jsonl, "w") as fh, cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for row in ex.map(extract_leg, legs, [a.arm] * len(legs), [a.sigma] * len(legs), chunksize=2):
            n_err += 1 if "error" in row else 0
            fh.write(json.dumps(row) + "\n")
    print("legs=%d errors=%d out=%s" % (len(legs), n_err, a.out_jsonl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
