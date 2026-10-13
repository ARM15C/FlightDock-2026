#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: A-9 split-window H1 scorer (T-352, BEHAVIORAL; operator
#   signature H1ADJ-A2B2-20260809). Per-leg v3 adjudication row:
#     H1-D truth peak-up over DOCKING rows, bar 2.00 m (gse.H1_BAR_M)
#     H1-R per-recovery-window truth crest, absolute ceiling 2.50 m
#     hard_fail_v3 = real_overtake OR bust_h1d OR bust_h1r OR collision
#     B2 co-report, TWO fields (both reporting only — the terminal remains
#       a hard fail either way):
#       collision_terminal_envelope_refuted — the CORRECT computation
#         (T-437): terminal fired but the collision ENVELOPE predicate
#         (cg<3.0 AND below<-2.0 AND -2.0<=behind<=5.0, 3-tick sustain;
#         collision_envelope.py) never fired on the TRUTH channel.
#       collision_terminal_truth_refuted — SUPERSEDED DEFECTIVE PROXY,
#         kept byte-identical so banked artifacts replay (whole-leg truth
#         peak <= 2.00 m stands in for the envelope; WRONG on D1_43285,
#         T-435). Never cite it as a collision adjudication.
#   RECOVERY rows are rows whose OWN fsm_state column reads
#   EMERGENCY_BREAKAWAY — the FSM's per-tick record in the paired CSV, never
#   a scoring predicate (T-340 rule). Contiguous recovery rows form one
#   window per guard firing; crest delta above up-at-firing is co-reported
#   per window (the ARM-F stopping-distance vertical metric).
#
#   VENUE DEVIATION, DISCLOSED: A-9.3 names gate_safety_events.py, but that
#   file measures 298 code-LoC against its caps (300/310), and caps are
#   never raised without an — so the v3 scorer lands here
#   as a SIBLING module and gate_safety_events.py is NOT touched (v2 and all
#   wire fields untouched by construction, satisfying A-9.3's actual terms).
#   Every v2 field is co-reported verbatim from gse.score_leg.
# PUBLIC API: score_leg_v3(leg_dir) -> dict; h1_windowed_aggregate(rows) ->
#   dict; CLI: --leg-dir DIR | --selfcheck.
# SIDE-EFFECTS: none (pure reader; CLI prints JSON). Selfcheck uses a
#   tempdir, removed in finally.
# CAPS: file soft 300 / hard 380 code-LoC; function cap 70 (the selfcheck
#   battery is deliberately in-module — pytest CANNOT run while a SITL leg
#   is flying, the gse precedent — and _scan_windows keeps one row-scan in
#   one function rather than splitting a single pass across two).
# COVERAGE: tests/unit/test_gate_h1_windows.py + in-module --selfcheck (the
#   only verification available while a SITL leg is flying).
# =============================================================================
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collision_envelope as cenv  # noqa: E402 (T-437 envelope predicate)
import gate_safety_events as gse  # noqa: E402 (v2 scorer; never modified)

H1R_CEILING_M = 2.50          # A-9.2, operator-set (H1ADJ-A2B2-20260809)
RECOVERY_STATE = "EMERGENCY_BREAKAWAY"
WINDOWS_FRAME = "boom_tip_truth_decisive_rowwise_fsm_state"


def _scan_windows(leg_dir):
    """Row-wise H1-D / H1-R decomposition. Returns None when no valid row;
    {'no_state_column': True} when rows exist but fsm_state is absent —
    the leg is NOT decomposable and must fail CLOSED, never silently
    fall back to the whole-leg number."""
    csvs = sorted(glob.glob(os.path.join(leg_dir, "env*",
                                         "paired_channels_ep*.csv")))
    peak_dock = None
    runs, cur = [], None
    ticks = state_seen = 0
    ec = cenv.EnvelopeCounter()            # T-437: truth-channel envelope
    for path in csvs:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                try:                       # validity mirrors gse._scan_csvs
                    up = -float(r["raw_below_m"])
                    t = float(r["sim_time_s"])
                except (TypeError, ValueError, KeyError):
                    continue
                ticks += 1
                ec.feed(r.get("raw_behind_m"), r.get("raw_cross_m"),
                        r.get("raw_below_m"))
                st = r.get("fsm_state")
                if st is None:
                    continue
                state_seen += 1
                if st == RECOVERY_STATE:
                    if cur is None:
                        cur = {"t_enter_s": t, "up_at_firing_m": up,
                               "crest_m": up, "t_exit_s": t}
                    else:
                        cur["crest_m"] = max(cur["crest_m"], up)
                        cur["t_exit_s"] = t
                else:
                    if cur is not None:
                        runs.append(cur)
                        cur = None
                    peak_dock = up if peak_dock is None else max(peak_dock, up)
        if cur is not None:                # leg ended inside recovery
            runs.append(cur)
            cur = None
    if ticks == 0:
        return None
    if state_seen == 0:
        return {"no_state_column": True, "ticks": ticks}
    for w in runs:
        w["crest_delta_m"] = round(w["crest_m"] - w["up_at_firing_m"], 4)
        w["duration_s"] = round(w["t_exit_s"] - w["t_enter_s"], 3)
        for k in ("t_enter_s", "up_at_firing_m", "crest_m", "t_exit_s"):
            w[k] = round(w[k], 4)
    return {"no_state_column": False, "ticks": ticks,
            "peak_dock_m": peak_dock, "recovery_windows": runs,
            "envelope": {"ticks_in_zone": ec.ticks_in_zone,
                         "max_consecutive": ec.max_consecutive,
                         "fires": ec.fires}}


def score_leg_v3(leg_dir):
    """One leg -> v3 row = every v2 field verbatim + the A-9 windowed
    fields. Missing evidence or a non-decomposable CSV => instrument_gap
    (fail-closed at the caller, the gse convention)."""
    row = dict(gse.score_leg(leg_dir))       # v2 co-reported, untouched
    row["h1r_ceiling_m"] = H1R_CEILING_M
    row["windows_frame"] = WINDOWS_FRAME
    if row.get("instrument_gap"):
        return row
    w = _scan_windows(leg_dir)
    if w is None or w.get("no_state_column"):
        row["instrument_gap"] = True
        row["gap_detail"] = {"windows": False,
                             "why": ("no_valid_rows" if w is None
                                     else "fsm_state_column_absent")}
        return row
    crests = [r["crest_m"] for r in w["recovery_windows"]]
    bust_h1d = (w["peak_dock_m"] is not None
                and w["peak_dock_m"] > gse.H1_BAR_M)
    bust_h1r = any(c > H1R_CEILING_M for c in crests)
    refuted = bool(row["collision_terminal"]
                   and row["truth_peak_up_m"] <= gse.H1_BAR_M)
    row.update({
        "truth_peak_up_docking_m": (round(w["peak_dock_m"], 4)
                                    if w["peak_dock_m"] is not None else None),
        "recovery_windows": w["recovery_windows"],
        "recovery_crest_max_m": (round(max(crests), 4) if crests else None),
        "bust_h1d": bool(bust_h1d),
        "bust_h1r": bool(bust_h1r),
        "collision_terminal_truth_refuted": refuted,   # SUPERSEDED proxy
        "collision_terminal_envelope_refuted": bool(
            row["collision_terminal"] and not w["envelope"]["fires"]),
        "envelope_max_consecutive": w["envelope"]["max_consecutive"],
        "hard_fail_v3": bool(row["real_overtake"] or bust_h1d or bust_h1r
                             or row["collision_terminal"]),
    })
    return row


def h1_windowed_aggregate(rows):
    """Gate-level A-9 fields from per-leg v3 rows (skips gap legs).
    exceed_legs_v3 is the A-9.5 conjunct: H1-D bust OR H1-R bust — the
    quantity a launch gate asserting `exceed_2m_legs==0` must read on
    panels governed by A-9. Ack bar applies to h1d p95 (A-9.5)."""
    ok = [r for r in rows if not r.get("instrument_gap")]
    dock = [r["truth_peak_up_docking_m"] for r in ok
            if r.get("truth_peak_up_docking_m") is not None]
    crests = [c["crest_m"] for r in ok for c in r.get("recovery_windows", [])]
    return {
        "h1d_truth_peak_up_m": ({
            "p50": round(gse.pct(dock, .5), 3),
            "p95": round(gse.pct(dock, .95), 3),
            "p99": round(gse.pct(dock, .99), 3),
            "max": round(max(dock), 3), "n_legs": len(dock)} if dock else None),
        "h1r_recovery_crests_m": ({
            "max": round(max(crests), 3), "n_windows": len(crests),
            "exceed_ceiling": sum(1 for c in crests if c > H1R_CEILING_M)}
            if crests else {"max": None, "n_windows": 0, "exceed_ceiling": 0}),
        "exceed_legs_v3": sum(1 for r in ok
                              if r.get("bust_h1d") or r.get("bust_h1r")),
        "hard_fail_v3_legs": sum(1 for r in ok if r.get("hard_fail_v3")),
        "collision_terminals_truth_refuted": sum(         # SUPERSEDED proxy
            1 for r in ok if r.get("collision_terminal_truth_refuted")),
        "collision_terminals_envelope_refuted": sum(
            1 for r in ok if r.get("collision_terminal_envelope_refuted")),
        "n_legs_scored": len(ok), "n_legs_gap": len(rows) - len(ok),
        "h1_bar_m": gse.H1_BAR_M, "h1r_ceiling_m": H1R_CEILING_M,
        "windows_frame": WINDOWS_FRAME,
    }


def _mk_leg(root, name, rows, terminal="TIMEOUT"):
    """Synthetic leg matching the REAL evidence layout (gse-compatible)."""
    d = os.path.join(root, name, "env1")
    os.makedirs(d)
    hdr = ("sim_time_s,raw_behind_m,raw_cross_m,raw_below_m,raw_range_m,"
           "ctrl_below_m,fsm_state\n")
    open(os.path.join(d, "paired_channels_ep0001.csv"), "w").write(hdr + rows)
    if terminal is not None:
        json.dump({"episodes": [{"terminal_type": terminal}]},
                  open(os.path.join(root, name, "results_eid1.json"), "w"))
    e = os.path.join(root, name, "episode_000")
    os.makedirs(e)
    open(os.path.join(e, "TRANSITION_SNAPSHOT.log"), "w").write("")
    json.dump({"terminal_sim_time_s": 999.0},
              open(os.path.join(e, "run_summary_finalized.json"), "w"))
    json.dump({"breakaway_events": 0},
              open(os.path.join(root, name, "safety_summary.json"), "w"))
    return os.path.join(root, name)


def _sc_rows():
    """Synthetic legs for the selfcheck battery. DC=DOCKING_CREEP rows,
    EB=EMERGENCY_BREAKAWAY rows; up = -raw_below_m."""
    dc = "{t},0.5,0.0,{bl},4.0,{bl},DOCKING_CREEP\n"
    eb = "{t},0.5,0.0,{bl},4.0,{bl},EMERGENCY_BREAKAWAY\n"
    mk = lambda pat, t, up: pat.format(t=t, bl=-up)
    # (a) docking bust 2.5 — v3 fails on H1-D
    a = mk(dc, 1.0, 2.5) + mk(dc, 1.2, 0.5)
    # (b) THE ADJUDICATED CASE: docking clean, recovery crest 2.30
    #     (>2.00, <=2.50) — v2 FAILS this leg, v3 PASSES it
    b = mk(dc, 1.0, 0.5) + mk(eb, 1.2, 2.30) + mk(dc, 1.4, 0.5)
    # (c) recovery crest 2.60 > ceiling — v3 fails on H1-R
    c = mk(dc, 1.0, 0.5) + mk(eb, 1.2, 2.60) + mk(dc, 1.4, 0.5)
    # (d) boundary: crest exactly at the ceiling passes; +0.01 busts
    d0 = mk(dc, 1.0, 0.5) + mk(eb, 1.2, 2.50)
    d1 = mk(dc, 1.0, 0.5) + mk(eb, 1.2, 2.51)
    # (e) two separate firings -> two windows, deltas from each own edge
    e = (mk(dc, 1.0, 0.5) + mk(eb, 2.0, 1.0) + mk(eb, 2.2, 1.8)
         + mk(dc, 3.0, 0.4) + mk(eb, 4.0, 0.9) + mk(eb, 4.2, 2.1))
    # (f) truth-refuted terminal: whole-leg truth max 1.98 < 2.00
    f = mk(dc, 1.0, 1.98) + mk(dc, 1.2, 0.5)
    # (g) THE 43285 SHAPE (T-437): whole-leg peak 2.05 busts the ceiling,
    #     but only ONE envelope tick — proxy says genuine, envelope refutes
    g = mk(dc, 1.0, 2.05) + mk(dc, 1.2, 0.5)
    # (gg) GENUINE envelope collision: 3 consecutive in-zone ticks
    gg = mk(dc, 1.0, 2.05) + mk(dc, 1.05, 2.05) + mk(dc, 1.1, 2.05)
    # (fx) a mid-run unparseable behind SPLITS a would-be sustain (2+2)
    bad = "{t},notanum,0.0,{bl},4.0,{bl},DOCKING_CREEP\n"
    fx = (mk(dc, 1.0, 2.05) + mk(dc, 1.05, 2.05) + mk(bad, 1.1, 2.05)
          + mk(dc, 1.15, 2.05) + mk(dc, 1.2, 2.05))
    return a, b, c, d0, d1, e, f, g, gg, fx


def _sc_score(t):
    """Build + score the synthetic battery; returns (rows dict, gap leg)."""
    a, b, c, d0, d1, e, f, g, gg, fx = _sc_rows()
    S = {}
    legs = [("a", a), ("b", b), ("c", c), ("d0", d0), ("d1", d1), ("e", e),
            ("f", f), ("g", g), ("gg", gg), ("fx", fx), ("b2", b)]
    for name, rows in legs:
        term = ("SAFETY_COLLISION" if name in ("f", "g", "gg", "fx")
                else "TIMEOUT")
        S[name] = score_leg_v3(_mk_leg(t, name, rows, terminal=term))
    # no-fsm-state leg: reuse gse's own selfcheck header (no state col)
    d = os.path.join(t, "h", "env1")
    os.makedirs(d)
    open(os.path.join(d, "paired_channels_ep0001.csv"), "w").write(
        "sim_time_s,raw_behind_m,raw_cross_m,raw_below_m,raw_range_m,"
        "ctrl_below_m\n1.0,0.5,0.0,0.5,4.0,0.5\n")
    json.dump({"episodes": [{"terminal_type": "TIMEOUT"}]},
              open(os.path.join(t, "h", "results_eid1.json"), "w"))
    eh = os.path.join(t, "h", "episode_000")
    os.makedirs(eh)
    open(os.path.join(eh, "TRANSITION_SNAPSHOT.log"), "w").write("")
    S["h"] = score_leg_v3(os.path.join(t, "h"))
    return S


def _selfcheck():
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="ghw_sc_")
    try:
        S = _sc_score(t)
        A, B, C, D0, D1 = S["a"], S["b"], S["c"], S["d0"], S["d1"]
        E, F, G, GG, FX = S["e"], S["f"], S["g"], S["gg"], S["fx"]
        H, B2 = S["h"], S["b2"]
        agg = h1_windowed_aggregate([A, B, C, D0, D1, E, F, G, GG, FX])
        checks = [
            ("i  docking bust 2.5 => bust_h1d + hard_fail_v3",
             A["bust_h1d"] and A["hard_fail_v3"]
             and A["truth_peak_up_docking_m"] == 2.5),
            ("ii ADJUDICATED SPLIT: recovery 2.30 => v2 FAILS, v3 PASSES",
             B["hard_fail_v2"] and not B["hard_fail_v3"]
             and not B["bust_h1d"] and not B["bust_h1r"]
             and B["recovery_crest_max_m"] == 2.3),
            ("iii recovery 2.60 > ceiling => bust_h1r + hard_fail_v3",
             C["bust_h1r"] and C["hard_fail_v3"] and not C["bust_h1d"]),
            ("iv boundary: 2.50 passes, 2.51 busts",
             not D0["bust_h1r"] and D1["bust_h1r"]),
            ("v  two firings => two windows, per-window deltas",
             len(E["recovery_windows"]) == 2
             and E["recovery_windows"][0]["crest_delta_m"] == 0.8
             and E["recovery_windows"][1]["crest_delta_m"] == 1.2),
            ("vi B2 refuted co-report fires ONLY when truth never crossed",
             F["collision_terminal_truth_refuted"]
             and not G["collision_terminal_truth_refuted"]
             and F["hard_fail_v3"] and G["hard_fail_v3"]),
            ("vii missing fsm_state => instrument_gap, fail CLOSED",
             H.get("instrument_gap") is True
             and H["gap_detail"]["why"] == "fsm_state_column_absent"),
            ("viii v2 fields co-reported verbatim",
             B["truth_peak_up_m"] == 2.3 and B["truth_ceiling_bust"]
             and "hard_fail_v2" in B),
            ("ix aggregate: exceed_legs_v3 + refuted counts + n",
             agg["exceed_legs_v3"] == 6      # A,G,GG,FX (h1d) + C,D1 (h1r)
             and agg["hard_fail_v3_legs"] == 7  # A,C,D1 + F,G,GG,FX terms
             and agg["collision_terminals_truth_refuted"] == 1     # F only
             and agg["collision_terminals_envelope_refuted"] == 3  # F,G,FX
             and agg["n_legs_scored"] == 10),
            ("x  determinism", B == B2),
            ("xi THE 43285 SHAPE: proxy says genuine, ENVELOPE refutes "
             "(1 tick < 3-sustain); fields disagree by design",
             not G["collision_terminal_truth_refuted"]
             and G["collision_terminal_envelope_refuted"]
             and G["envelope_max_consecutive"] == 1),
            ("xii genuine 3-tick envelope collision: NEITHER field refutes",
             not GG["collision_terminal_truth_refuted"]
             and not GG["collision_terminal_envelope_refuted"]
             and GG["envelope_max_consecutive"] == 3
             and GG["hard_fail_v3"]),
            ("xiii unparseable behind SPLITS the sustain run (fail-closed)",
             FX["collision_terminal_envelope_refuted"]
             and FX["envelope_max_consecutive"] == 2),
            ("xiv 43168 shape: refuted under BOTH fields (0 ticks)",
             F["collision_terminal_truth_refuted"]
             and F["collision_terminal_envelope_refuted"]
             and F["envelope_max_consecutive"] == 0),
        ]
        for n, p in checks:
            print(("PASS " if p else "FAIL ") + n)
        print("selfcheck:", "PASS" if all(p for _, p in checks) else "FAIL")
        return 0 if all(p for _, p in checks) else 1
    finally:
        shutil.rmtree(t, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-dir")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return _selfcheck()
    if not a.leg_dir:
        ap.error("--leg-dir required (or --selfcheck)")
    print(json.dumps(score_leg_v3(a.leg_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
