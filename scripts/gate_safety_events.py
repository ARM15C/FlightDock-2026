#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Per-leg REAL-safety-event scoring for gate adjudication
#   . Scores a banked eval leg from its paired
#   truth/fused channel CSVs + TRANSITION_SNAPSHOT.log: real CG-frame
#   overtake (close-formation gated, turn-projection artifacts excluded),
#   H1 signed peak upward deviation (truth decisive, fused reported;
#   operator-elected boom-tip frame), collision, and the v2 hard-fail
#   conjunction.
# COLLISION LIMB: reads the ENV'S OWN episode
#   terminal via leg_terminal_reader (results_eid JSON primary / metrics CSV
#   fallback) — the channel that actually FIRES. The old translog token count
#   (`collision_rows`) is a GUARD THAT CANNOT FIRE (the translog vocabulary is
#   FSM states; SAFETY_COLLISION is an env terminator) and is DEPRECATED:
#   LEGACY-NEVER-FIRED, co-reported as a diagnostic only, never in the
#   criterion. Deletion requires its own DELETION dispatch. Unreadable
#   terminal source => instrument_gap (fail closed, the existing convention). Adjudication-side READER — evidence is scored at read time,
#   so banked legs re-score under new criteria without re-flying. This is the
#   evidence-scoring sibling module curriculum_eval_spawner.py's cap note
#   requested; the spawner's safety_summary.json remains the LEGACY channel
#   and is reconciled against, never trusted as the v2 verdict.
# PUBLIC API: score_leg(leg_dir) -> dict; h1_aggregate(rows) -> dict;
#   pct(vals, q); CLI: --leg-dir DIR | --selfcheck.
# CAPTURE COMPLETENESS: score_leg also emits
#   `capture_terminal_s` (the RUNTIME's own terminal sim time) and
#   `capture_complete` (did the recording reach it?). These exist because the
#   T-270 short-capture screen inferred evidence loss from SPAN ALONE and so
#   could not tell "the recording was cut" (evidence loss) from "the flight
#   ended early" (a real outcome). Resolved HERE, beside capture_span_s, so
#   one derived quantity has one writer. FAIL-CLOSED: if the terminal time
#   cannot be read, completeness is UNPROVEN and capture_complete is False —
#   never optimistically True. score_leg emits them; it does NOT act on them
#   (the screen in curriculum_gate_runner.py is the sole consumer).
# SIDE-EFFECTS: none (pure reader; CLI prints JSON to stdout).
#   Selfcheck uses a tempdir, removed in finally.
# CAPS: file soft 300 / hard 310 LoC; function cap 60.
#   (soft 220 -> 240 + hard 250 -> 290 by (S-3);
#   soft 240 -> 300 + hard 290 -> 310 by (S-10,
#   +24 measured vs my <=16 budget — the FOURTH consecutive underestimate,
#   on the record). Soft 300 = alarm on any further growth. The completeness test belongs beside the span it
#   qualifies; a separate module would put ONE quantity in TWO files, which
#   is the defect class S-3/S-4 close. The hard raise was elected AFTER the
#   measured cost came in at +56 lines (218 -> 274), not the +10-12 first
#   estimated: +23 production instrument, +30 --selfcheck battery, +3
#   constants. The selfcheck battery is deliberately KEPT in-module because
#   pytest CANNOT be run while a SITL leg is flying, so --selfcheck is the
#   only verification available in exactly the window it is needed.)
# COVERAGE: tests/unit/test_gate_safety_events.py (+ runner battery) +
#   analysis/e_series_20260729/b1_smoke.py (interim, no-pytest window).
# FRAME FACTS: behind_m/below_m are BOOM-TIP relative; the tip sits
#   BOOM_HORIZONTAL_PROJECTION_M aft of the leader CG, so
#   aft_of_leader_CG = raw_behind_m + BOOM_AFT_M. H1 bar = 2.00 m LITERAL
#   (elected frame; R-1, commit 892ea917 — the value then equalled
#   -GUARD_ALT_CEILING_BELOW_M). DECOUPLED at GUARD-ELECT-588 (2026-08-15):
#   the guard now margin-arms at -1.90, but the H1 hypothesis bar is
#   criterion-frozen at 2.00 and must NEVER follow the guard constant.
# =============================================================================
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

_FD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _cand in (os.path.join(_FD, "tests"),
              "/home/ubuntu/FlightDock/tests"):   # canonical (runner's FD)
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
import formation_config as _fc  # noqa: E402 (read-only constant source)
import leg_terminal_reader as _ltr  # noqa: E402 (S-10: terminal channel)
import leg_demoted_reader as _ldr  # noqa: E402 (TERMDEMOTE-767)

BOOM_AFT_M = float(_fc.BOOM_HORIZONTAL_PROJECTION_M)          # 4.31
H1_BAR_M = 2.00   # criterion-frozen H1 bar; decoupled from the guard
#   constant at GUARD-ELECT-588 (guard margin-arms at -1.90; see header)
CLOSE_RANGE_M, CLOSE_CROSS_M = 30.0, 10.0    # close-formation gate (declared)
CENTER_BAR_M, CENTER_PERSIST_S = -1.0, 0.10  # formation_success_metrics:201-203
_BREAKAWAY_RE = re.compile(r"new=EMERGENCY_BREAKAWAY")
_COLLISION_RE = re.compile(r"SAFETY_COLLISION")   # LEGACY-NEVER-FIRED (S-10)
# S-3 capture-completeness tolerance, sized from the MEASURED separation, not
# chosen for convenience: complete captures measure delta = 0.000 s (all 12 D1
# legs, CAPTURE_TRUNCATION_RCA), while the observed wrapper-kill shapes measure
# delta in the HUNDREDS of seconds (38-39 s and 334 s captures against ~600 s
# windows). 1.0 s sits ~2 orders of magnitude clear of both populations and
# well above the 0.25 s frame-gap convention. One fact, one name.
CAPTURE_COMPLETE_TOL_S = 1.0
# Terminal-time channels in PREFERENCE ORDER. The finalized summary is the
# runtime's own sha-pinned account of when the episode ended; the unfinalized
# summary carries the same number under a different key and is the fallback.
_TERM_T_KEYS = (("episode_*/run_summary_finalized.json", "terminal_sim_time_s"),
                ("episode_*/run_summary.json", "summary_terminal_sim_time_s"))


def pct(vals, q):
    """Linear-interp percentile on a small list (signed; caller never abs())."""
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (k - f) * (s[f + 1] - s[f])


def _r3(v, neg=False):
    """T-455: round-to-3 or None; neg=True reports max(0, -v) instead."""
    return None if v is None else round(max(0.0, -v) if neg else v, 3)


def _scan_csvs(leg_dir):
    """All paired-channel ticks -> per-leg extrema/flags, or None (gap)."""
    csvs = sorted(glob.glob(os.path.join(leg_dir, "env*",
                                         "paired_channels_ep*.csv")))
    min_rb = min_cb = 1e18
    min_aft_close = 1e18     # E-4: overtake magnitude (close-gated min aft-of-CG)
    # T-455: two truth-side quantities that were parsed here and thrown away.
    # Definitions are taken VERBATIM from tools/gate_leg_report.py:136 so the
    # project carries ONE definition of each, not two:
    #   min_rg -> min over ALL ticks of raw_range_m (truth 3D
    #                       boom-tip range). NOT close-gated, matching the
    #                       reference instrument's `min_range_m`.
    #   min_behind_close -> close-gated min of raw_behind_m; reported as
    #                       max(0, -min) = peak excursion PAST the boom-tip
    #                       no-pass line (the ARM-B/F longitudinal item).
    # Both are REPORTED ONLY — no criterion in this file consumes them.
    min_rg = min_bc = 1e18
    excl = ticks = 0
    ot_any = ot_center = False
    run_start = None
    t_lo = t_hi = None       # T-271: capture span for the completeness screen
    for path in csvs:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    rb = float(r["raw_behind_m"])
                    rx = float(r["raw_cross_m"])
                    rg = float(r["raw_range_m"])
                    bl = float(r["raw_below_m"])
                    t = float(r["sim_time_s"])
                except (TypeError, ValueError, KeyError):
                    continue
                ticks += 1
                t_lo = t if t_lo is None else min(t_lo, t)
                t_hi = t if t_hi is None else max(t_hi, t)
                min_rb = min(min_rb, bl)
                min_rg = min(min_rg, rg)          # T-455 (all ticks)
                try:
                    min_cb = min(min_cb, float(r["ctrl_below_m"]))
                except (TypeError, ValueError, KeyError):
                    pass
                aft = rb + BOOM_AFT_M
                close = rg <= CLOSE_RANGE_M and abs(rx) <= CLOSE_CROSS_M
                if close:
                    min_aft_close = min(min_aft_close, aft)
                    min_bc = min(min_bc, rb)                       # T-455
                if aft < 0 and not close:
                    excl += 1        # turn-geometry projection artifact tick
                if close and aft < 0:
                    ot_any = True    # criterion A: gated pass of leader CG
                if close and aft < CENTER_BAR_M:   # criterion B: codebase bar
                    run_start = t if run_start is None else run_start
                    if t - run_start >= CENTER_PERSIST_S:
                        ot_center = True
                else:
                    run_start = None
    if ticks == 0:
        return None
    return {"ticks": ticks, "min_raw_below": min_rb,
            "min_ctrl_below": (min_cb if min_cb < 1e17 else None),
            "min_aft_close_m": (min_aft_close if min_aft_close < 1e17
                                else None),
            "real_overtake_any": ot_any, "real_overtake_center": ot_center,
            "excluded_artifact_ticks": excl,
            "min_raw_range_m": (min_rg if min_rg < 1e17 else None),
            "min_behind_close_m": (min_bc if min_bc < 1e17 else None),
            "capture_span_s": round(t_hi - t_lo, 3),
            "capture_t_hi": round(t_hi, 3)}   # S-3: last row the capture saw


def _terminal_sim_time(leg_dir):
    """The RUNTIME's own terminal sim time for this leg, or None if it cannot
    be read. Max over episodes (a multi-episode leg ends when its LAST episode
    does). Returning None is meaningful: it means completeness is UNPROVEN,
    and score_leg turns that into capture_complete=False (fail closed)."""
    for pat, key in _TERM_T_KEYS:
        vals = []
        for p in sorted(glob.glob(os.path.join(leg_dir, pat))):
            try:
                v = json.load(open(p)).get(key)
            except (OSError, ValueError):
                continue
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            return round(max(vals), 3)
    return None


def _scan_translogs(leg_dir):
    """Guard-firing / collision RAW row counts, or None if no translog."""
    logs = glob.glob(os.path.join(leg_dir, "episode_*",
                                  "TRANSITION_SNAPSHOT.log"))
    if not logs:
        return None
    brk = coll = 0
    for lg in logs:
        for ln in open(lg, errors="replace"):
            if _BREAKAWAY_RE.search(ln):
                brk += 1
            if _COLLISION_RE.search(ln):
                coll += 1
    return {"guard_firings_legacy": brk, "collision_rows": coll}


def score_leg(leg_dir):
    """One leg -> v2 safety-event score. Missing evidence => instrument_gap
    (fail-closed at the caller: evidence missing is never a pass)."""
    ch = _scan_csvs(leg_dir)
    tl = _scan_translogs(leg_dir)
    terms = _ltr.terminal_types(leg_dir)
    if ch is None or tl is None or terms is None:
        return {"instrument_gap": True,
                "gap_detail": {"paired_csv": ch is not None,
                               "translog": tl is not None,
                               "terminal_source": terms is not None},
                "boom_aft_m": BOOM_AFT_M, "h1_bar_m": H1_BAR_M}
    summary = None
    p = os.path.join(leg_dir, "safety_summary.json")
    if os.path.exists(p):
        try:
            summary = json.load(open(p))
        except (OSError, ValueError):
            summary = None
    term_t = _terminal_sim_time(leg_dir)
    truth_up = -ch["min_raw_below"]
    fused_up = (-ch["min_ctrl_below"]
                if ch["min_ctrl_below"] is not None else None)
    real_ot = ch["real_overtake_any"]
    bust = truth_up > H1_BAR_M
    # TERMDEMOTE-767 limbs, built by leg_demoted_reader -- the ONE WRITER of
    # the terminal-OR-demoted-event disjunction. The predicate and both keys
    # are UNCHANGED; only their construction moved. Adding a scored terminal
    # now edits SCORED_TERMINALS there and grows neither this function (58 of
    # a 60-line cap) nor this file (310 of a 310-line hard cap).
    limbs = _ldr.terminal_limbs(leg_dir, terms)
    return {
        "instrument_gap": False,
        "real_overtake": real_ot,
        # E-4: the measured quantity behind a real_overtake reason —
        # peak in-front-of-leader-CG excursion, close-gated (m; 0 if never)
        "real_overtake_peak_infront_m": (
            round(max(0.0, -ch["min_aft_close_m"]), 4)
            if ch["min_aft_close_m"] is not None else None),
        "real_overtake_center": ch["real_overtake_center"],
        "criteria_agree": real_ot == ch["real_overtake_center"],
        "excluded_artifact_ticks": ch["excluded_artifact_ticks"],
        "truth_peak_up_m": round(truth_up, 4),
        "fused_peak_up_m": (round(fused_up, 4)
                            if fused_up is not None else None),
        "truth_ceiling_bust": bust,
        # T-455 — REPORTED ONLY, no criterion below reads either. Both are
        # truth-channel (raw_*) and both mirror tools/gate_leg_report.py:136
        # exactly, so the two instruments must agree to 1e-3 on any leg.
        # min_boom_tip_range_truth_m answers the operator's item-3 question on
        # the verdict path; max_infront_boomtip_close_m is the ARM-B/F
        # longitudinal stopping-distance item (range PAST the no-pass line).
        "min_boom_tip_range_truth_m": _r3(ch["min_raw_range_m"]),
        "max_infront_boomtip_close_m": _r3(ch["min_behind_close_m"], neg=True),
        **limbs,
        "terminal_types": terms,
        "collision_rows": tl["collision_rows"],   # LEGACY-NEVER-FIRED diag
        "guard_firings_legacy": tl["guard_firings_legacy"],
        "legacy_summary_reconciled": (
            summary is not None
            and summary.get("breakaway_events") == tl["guard_firings_legacy"]),
        "hard_fail_v2": bool(real_ot or bust or limbs["collision"]
                             or limbs["nopass"] or limbs["contact"]),
        "capture_span_s": ch["capture_span_s"],
        # S-3: did the recording reach the runtime's own end-of-episode?
        # term_t is None => UNPROVEN => False. Never optimistically True.
        "capture_terminal_s": term_t,
        "capture_complete": (term_t is not None
                             and term_t - ch["capture_t_hi"]
                             <= CAPTURE_COMPLETE_TOL_S),
        "ticks": ch["ticks"], "boom_aft_m": BOOM_AFT_M, "h1_bar_m": H1_BAR_M,
    }


def h1_aggregate(rows):
    """Gate-level H1 fields from per-leg rows (skips gap legs). SIGNED."""
    out = {}
    for key, name in (("truth_peak_up_m", "h1_truth_peak_up_m"),
                      ("fused_peak_up_m", "h1_fused_peak_up_m")):
        vals = [r[key] for r in rows
                if r.get(key) is not None]
        out[name] = ({"p50": round(pct(vals, .5), 3),
                      "p95": round(pct(vals, .95), 3),
                      "p99": round(pct(vals, .99), 3),
                      "max": round(max(vals), 3),
                      "exceed_2m_legs": sum(1 for v in vals if v > H1_BAR_M),
                      "n_legs": len(vals)} if vals else None)
    out["h1_bar_m"] = H1_BAR_M
    out["h1_frame"] = "boom_tip_truth_decisive_fused_reported"
    return out


def _completeness_cases(mk):
    """S-3 detectors for capture completeness. Every case is asserted in BOTH
    directions: a rule that can only say 'complete' is not a discriminator."""
    rows = "1.0,0.5,0.0,0.5,4.0,0.5\n600.0,0.5,0.0,0.5,4.0,0.5\n"
    done = score_leg(mk("cc_done", rows, term_t=600.0))
    cut = score_leg(mk("cc_cut", rows, term_t=900.0))      # 300 s lost
    noss = score_leg(mk("cc_noss", rows))                  # no run_summary
    at = score_leg(mk("cc_at", rows, term_t=600.0 + CAPTURE_COMPLETE_TOL_S))
    over = score_leg(mk("cc_over", rows,
                        term_t=600.0 + CAPTURE_COMPLETE_TOL_S + 0.01))
    multi = score_leg(mk("cc_multi", rows, term_t=[100.0, 600.0]))
    return [
        ("vii capture reaching terminal => complete",
         done["capture_complete"] and done["capture_terminal_s"] == 600.0),
        ("viii capture stopping 300 s early => INCOMPLETE",
         not cut["capture_complete"] and cut["capture_terminal_s"] == 900.0),
        ("ix no run_summary => UNPROVEN => fail closed (not True)",
         noss["capture_terminal_s"] is None
         and noss["capture_complete"] is False),
        ("x boundary: delta == tol complete, tol+eps INCOMPLETE",
         at["capture_complete"] and not over["capture_complete"]),
        ("xi multi-episode terminal = MAX over episodes",
         multi["capture_terminal_s"] == 600.0 and multi["capture_complete"]),
    ]


def _terminal_cases(mk):
    """S-10: the collision limb reads the channel that FIRES (env terminal),
    and an unreadable terminal source fails CLOSED as instrument_gap."""
    rows = "1.0,0.5,0.0,0.5,4.0,0.5\n"
    coll = score_leg(mk("tc_coll", rows, terminal="SAFETY_COLLISION"))
    nosrc = score_leg(mk("tc_nosrc", rows, terminal=None))
    return [
        ("xii collision TERMINAL => hard_fail_v2 (the limb can fire)",
         coll["hard_fail_v2"] and coll["collision_terminal"]
         and coll["terminal_types"] == ["SAFETY_COLLISION"]
         and coll["collision_rows"] == 0),
        ("xiii no terminal source => instrument_gap, fail closed",
         nosrc.get("instrument_gap") is True
         and nosrc["gap_detail"]["terminal_source"] is False),
    ]


def _selfcheck():
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="gse_sc_")
    hdr = ("sim_time_s,raw_behind_m,raw_cross_m,raw_below_m,raw_range_m,"
           "ctrl_below_m\n")

    def mk(name, rows, translog=True, breaks=0, term_t=None,
           terminal="TIMEOUT"):
        d = os.path.join(t, name, "env1")
        os.makedirs(d)
        if terminal is not None:
            json.dump({"episodes": [{"terminal_type": terminal}]},
                      open(os.path.join(t, name, "results_eid1.json"), "w"))
        open(os.path.join(d, "paired_channels_ep0001.csv"), "w").write(
            hdr + rows)
        if translog:
            e = os.path.join(t, name, "episode_000")
            os.makedirs(e)
            open(os.path.join(e, "TRANSITION_SNAPSHOT.log"), "w").write(
                "[TRANSITION] t=1 old=HOLD_CONTACT new=EMERGENCY_BREAKAWAY "
                "reason=guard_hard_overtake\n" * breaks)
        for i, tv in enumerate(term_t if isinstance(term_t, list)
                               else [] if term_t is None else [term_t]):
            e = os.path.join(t, name, "episode_%03d" % i)
            os.makedirs(e, exist_ok=True)
            json.dump({"terminal_sim_time_s": tv},
                      open(os.path.join(e, "run_summary_finalized.json"), "w"))
        json.dump({"breakaway_events": breaks},
                  open(os.path.join(t, name, "safety_summary.json"), "w"))
        return os.path.join(t, name)

    ot = score_leg(mk("ot", "1.0,-6.0,0.0,1.0,7.0,1.0\n"
                            "1.2,-6.5,0.0,1.0,7.5,1.0\n"))
    art = score_leg(mk("art", "1.0,-8.9,144.4,17.6,145.7,17.0\n"
                              "1.2,0.5,0.1,0.5,4.0,0.5\n", breaks=2))
    ceil = score_leg(mk("ceil", "1.0,0.5,0.0,-2.5,4.0,-2.6\n"))
    fused = score_leg(mk("fused", "1.0,0.5,0.0,-1.5,4.0,-2.3\n"))
    gap = score_leg(mk("gap", "", translog=True))
    agg = h1_aggregate([ot, art, ceil, fused])
    checks = [
        ("i overtake: both criteria + agree",
         ot["hard_fail_v2"] and ot["real_overtake"]
         and ot["real_overtake_center"] and ot["criteria_agree"]),
        ("ii artifact: no fail, excluded=1, legacy firings reconciled",
         not art["hard_fail_v2"] and art["excluded_artifact_ticks"] == 1
         and art["guard_firings_legacy"] == 2
         and art["legacy_summary_reconciled"]),
        ("iii truth ceiling bust fails", ceil["hard_fail_v2"]
         and ceil["truth_peak_up_m"] == 2.5),
        ("iv fused-only excursion passes (truth decisive)",
         not fused["hard_fail_v2"] and fused["fused_peak_up_m"] == 2.3),
        ("v empty csv => instrument_gap", gap["instrument_gap"]),
        ("vi aggregate max/exceed", agg["h1_truth_peak_up_m"]["max"] == 2.5
         and agg["h1_truth_peak_up_m"]["exceed_2m_legs"] == 1),
    ] + _completeness_cases(mk) + _terminal_cases(mk)
    for n, p in checks:
        print(("PASS " if p else "FAIL ") + n)
    shutil.rmtree(t, ignore_errors=True)
    print("selfcheck:", "PASS" if all(p for _, p in checks) else "FAIL")
    return 0 if all(p for _, p in checks) else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-dir")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return _selfcheck()
    if not a.leg_dir:
        ap.error("--leg-dir required (or --selfcheck)")
    print(json.dumps(score_leg(a.leg_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
