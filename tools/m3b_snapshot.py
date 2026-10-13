#!/usr/bin/env python3
# =============================================================================
# m3b_snapshot.py — shared M3b-window perception-vs-truth snapshot bundle.
# RESPONSIBILITY: For ONE leg, compute the M3b docking snapshot consumed by BOTH
#   the records report (#3) and the metrics-CSV summarizer (#2), over the
#   truth-side M3b window (truth in the boom-tip box |behind|<=0.7,|below|<=0.7,
#   |cross|<=0.5 m), joined truth<-consumed on sim_time_s (truth-left, 0.06 s):
#     * per-axis RMS / p95 / p99 on BOTH truth-side and consumed-side positions,
#       plus the perception-vs-truth disparity Δ (signed med + RMS/p95/p99);
#     * M3b dwell on truth side, perception side, and simultaneous (f9) + >=3.0 flags;
#     * relative closure velocity (perception vc_behind_mps + truth Δbehind/Δt);
#     * QuadPlane TRUE airspeed = follower EAS (VFR_HUD) x eas2tas (parsed from the
#       controller's logged baro eas2tas; ISA-from-altitude fallback) — NOT EAS;
#     * safety guards fired (success_metrics violations + run_summary fire counts).
# PUBLIC API:
#   m3b_snapshot(leg_dir) -> dict
#   flatten(snapshot) -> dict # flat {col: val} for CSV
#   main(argv) -> int # CLI: <leg_dir> [--json OUT] | --selftest
# SIDE-EFFECTS: reads leg artifacts (read-only); optional JSON write. NO network,
#   NO process spawn, NO env mutation. Computes NO metric inside any env; ingests
#   NO truth-side field into any actor/critic/reward/obs/policy path.
# CAPS: file soft 200 / hard 240 LoC; per-function soft 45 LoC.
# COVERAGE: --selftest (synthetic leg fixture) + python -m py_compile.
# =============================================================================
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvt_disparity_census import TOL_S, load_truth_dedup  # noqa: E402

AX = ("behind", "below", "cross")
M3B = {"behind": 0.7, "below": 0.7, "cross": 0.5}
TRU = {a: f"raw_truth_{a}_m" for a in AX}
CONS = {a: f"{a}_m" for a in AX}
LEADER_TAS = 24.7


def _stats(v):
    if len(v) == 0:
        return {"rms": None, "p95": None, "p99": None}
    a = v.abs()
    return {"rms": round(float(np.sqrt((v ** 2).mean())), 3),
            "p95": round(float(a.quantile(.95)), 3),
            "p99": round(float(a.quantile(.99)), 3)}


def _load_json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _parse_eas2tas(leg):
    p = os.path.join(leg, "controller_1x.log")
    if not os.path.exists(p):
        return None
    with open(p, errors="ignore") as fh:
        for ln in fh:
            m = re.search(r"baro eas2tas LIVE:.*eas2tas=([0-9.]+)", ln)
            if m:
                return float(m.group(1))
    return None


def _isa_eas2tas(alt_m):
    if alt_m is None or not np.isfinite(alt_m):
        return None
    sigma = ((288.15 - 0.0065 * alt_m) / 288.15) ** 4.256
    return round(float(1.0 / np.sqrt(sigma)), 5)


def _dwells(leg):
    sm = _load_json(os.path.join(leg, "success_metrics.json"))
    tr = _load_json(os.path.join(leg, "run_summary_env_truth.json"))
    fg = _load_json(os.path.join(leg, "fivegoal_grade.json"))
    perc = (sm.get("milestones", {}).get("m3b_tight_contact", {}) or {}).get("best_dwell_s")
    truth, simul = tr.get("m3b_best_dwell_s"), fg.get("f9_longest_s")
    ge3 = lambda x: bool(x is not None and x >= 3.0)
    return {"truth_m3b_s": truth, "perceived_m3b_s": perc, "simultaneous_f9_s": simul,
            "truth_ge3": ge3(truth), "perceived_ge3": ge3(perc), "simultaneous_ge3": ge3(simul)}


def _safety(leg):
    sm = _load_json(os.path.join(leg, "success_metrics.json")).get("safety", {})
    rs = _load_json(os.path.join(leg, "run_summary.json"))
    viol = [{"code": v.get("code"), "t_s": round(v.get("first_time_s") or 0.0, 1),
             "sev": v.get("severity")} for v in sm.get("violations", [])]
    gk = ("guard_altitude_ceiling_fire_count", "guard_hard_overtake_fire_count",
          "safety_emergency_count", "dc_vert_abort_count", "dc_airspeed_tripwire_count",
          "diagnostic_suppressed_breakaways_total", "guard_speed_cap_fire_count")
    return {"hard_fail": sm.get("hard_fail"), "violations": viol,
            "guard_counts": {k: rs.get(k) for k in gk}}


def _airspeed_closure(win, leg):
    vc = win["vc_behind_mps"] if "vc_behind_mps" in win else pd.Series(dtype=float)
    eas = win["follower_airspeed_mps"] if "follower_airspeed_mps" in win else pd.Series(dtype=float)
    alt = float(win["follower_alt_msl_m"].median()) if ("follower_alt_msl_m" in win and len(win)) else None
    e2t, src = _parse_eas2tas(leg), "baro_log"
    if e2t is None:
        e2t, src = _isa_eas2tas(alt), "isa_fallback"
    eas_med = round(float(eas.median()), 3) if len(eas) else None
    tas = round(eas_med * e2t, 3) if (eas_med is not None and e2t) else None
    tcl = None
    if len(win) >= 3:
        tcl = round(float(np.median(np.gradient(win[TRU["behind"]].values,
                                                win["sim_time_s"].values))), 3)
    closure = {"perception_vc_behind_mps_med": round(float(vc.median()), 3) if len(vc) else None,
               "perception_vc_behind_mps_absp95": round(float(vc.abs().quantile(.95)), 3) if len(vc) else None,
               "truth_behind_rate_mps_med": tcl}
    air = {"follower_eas_mps_med": eas_med, "eas2tas": e2t, "eas2tas_src": src,
           "follower_tas_mps": tas, "leader_tas_mps": LEADER_TAS}
    return closure, air


def m3b_snapshot(leg):
    tr, meta = load_truth_dedup(leg)
    sa = pd.read_csv(os.path.join(leg, "speed_arbitration.csv"), low_memory=False)
    keep = [c for c in ["behind_m", "below_m", "cross_m", "vc_behind_mps",
                        "follower_airspeed_mps", "follower_alt_msl_m"] if c in sa.columns]
    m = pd.merge_asof(tr, sa[["sim_time_s"] + keep].sort_values("sim_time_s"),
                      on="sim_time_s", tolerance=TOL_S, direction="nearest")
    tin = ((m[TRU["behind"]].abs() <= M3B["behind"]) & (m[TRU["below"]].abs() <= M3B["below"])
           & (m[TRU["cross"]].abs() <= M3B["cross"]))
    win = m[tin].dropna(subset=[TRU["behind"], CONS["behind"]])
    axes = {}
    for ax in AX:
        tv, cv = win[TRU[ax]], win[CONS[ax]]
        dv = cv - tv
        smed = round(float(dv.median()), 3) if len(dv) else None
        axes[ax] = {"truth": _stats(tv), "consumed": _stats(cv),
                    "disparity": {"signed_med": smed, **_stats(dv)}}
    closure, air = _airspeed_closure(win, leg)
    return {"leg_dir": leg, "window": "truth_M3b", "n": int(len(win)),
            "truth_points_distinct": meta["truth_points_distinct"],
            "axes": axes, "dwell": _dwells(leg), "closure": closure,
            "airspeed": air, "safety": _safety(leg)}


def flatten(s):
    out = {"m3b_n": s["n"]}
    for ax in AX:
        a = s["axes"][ax]
        for ch in ("truth", "consumed"):
            for st in ("rms", "p95", "p99"):
                out[f"m3b_{ax}_{ch}_{st}"] = a[ch][st]
        for st in ("signed_med", "rms", "p95", "p99"):
            out[f"m3b_{ax}_disp_{st}"] = a["disparity"][st]
    d, c, a, sf = s["dwell"], s["closure"], s["airspeed"], s["safety"]
    out.update({"m3b_dwell_truth_s": d["truth_m3b_s"], "m3b_dwell_perc_s": d["perceived_m3b_s"],
                "m3b_dwell_simul_f9_s": d["simultaneous_f9_s"], "m3b_dwell_truth_ge3": d["truth_ge3"],
                "m3b_dwell_perc_ge3": d["perceived_ge3"], "m3b_dwell_simul_ge3": d["simultaneous_ge3"]})
    out.update({"m3b_clos_perc_vc_med": c["perception_vc_behind_mps_med"],
                "m3b_clos_perc_vc_absp95": c["perception_vc_behind_mps_absp95"],
                "m3b_clos_truth_rate_med": c["truth_behind_rate_mps_med"]})
    out.update({"m3b_foll_eas_med": a["follower_eas_mps_med"], "m3b_eas2tas": a["eas2tas"],
                "m3b_eas2tas_src": a["eas2tas_src"], "m3b_foll_tas": a["follower_tas_mps"],
                "m3b_leader_tas": a["leader_tas_mps"]})
    g = sf["guard_counts"]
    out.update({"m3b_hard_fail": sf["hard_fail"],
                "m3b_safety_codes": ";".join(f"{v['code']}@{v['t_s']}s" for v in sf["violations"]) or "none",
                "m3b_guard_alt_ceiling": g.get("guard_altitude_ceiling_fire_count"),
                "m3b_guard_hard_overtake": g.get("guard_hard_overtake_fire_count"),
                "m3b_safety_emergency": g.get("safety_emergency_count"),
                "m3b_dc_vert_abort": g.get("dc_vert_abort_count"),
                "m3b_suppressed_breakaways": g.get("diagnostic_suppressed_breakaways_total")})
    return out


FLAT_COLUMNS = list(flatten({
    "n": 0, "axes": {a: {"truth": {"rms": 0, "p95": 0, "p99": 0},
                         "consumed": {"rms": 0, "p95": 0, "p99": 0},
                         "disparity": {"signed_med": 0, "rms": 0, "p95": 0, "p99": 0}} for a in AX},
    "dwell": {"truth_m3b_s": 0, "perceived_m3b_s": 0, "simultaneous_f9_s": 0,
              "truth_ge3": 0, "perceived_ge3": 0, "simultaneous_ge3": 0},
    "closure": {"perception_vc_behind_mps_med": 0, "perception_vc_behind_mps_absp95": 0,
                "truth_behind_rate_mps_med": 0},
    "airspeed": {"follower_eas_mps_med": 0, "eas2tas": 0, "eas2tas_src": 0,
                 "follower_tas_mps": 0, "leader_tas_mps": 0},
    "safety": {"hard_fail": 0, "violations": [], "guard_counts": {}}}).keys())


def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    rows = []
    for i in range(40):
        t = 100.0 + i * 0.1
        rows.append({"tick_idx": i, "sim_time_s": t, "raw_truth_behind_m": 0.1,
                     "raw_truth_below_m": 0.05, "raw_truth_cross_m": 0.0})
    with open(os.path.join(d, "paired_tick_log_env0.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    sa = "sim_time_s,state,behind_m,below_m,cross_m,vc_behind_mps,follower_airspeed_mps,follower_alt_msl_m\n"
    for i in range(40):
        sa += f"{100.0+i*0.1},DOCKING_CREEP,0.15,0.06,0.01,-0.02,22.7,890\n"
    open(os.path.join(d, "speed_arbitration.csv"), "w").write(sa)
    s = m3b_snapshot(d)
    assert s["n"] == 40, s["n"]
    assert s["axes"]["behind"]["disparity"]["signed_med"] == 0.05, s["axes"]["behind"]
    assert s["airspeed"]["eas2tas_src"] == "isa_fallback"
    assert s["airspeed"]["follower_tas_mps"] > 22.7  # TAS > EAS
    assert "m3b_behind_disp_p99" in flatten(s)
    print("SELFTEST PASS")
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    if len(argv) < 2:
        print("usage: m3b_snapshot.py <leg_dir> [--json OUT] | --selftest", file=sys.stderr)
        return 2
    s = m3b_snapshot(argv[1])
    print(json.dumps(s, indent=2, default=str))
    if "--json" in argv:
        json.dump(s, open(argv[argv.index("--json") + 1], "w"), indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
