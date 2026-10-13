#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: re-read ONE banked exam corpus under the OPERATOR'S RULED
#   SAFETY WINDOW (2026-08-31): an altitude excursion counts as a safety event
#   ONLY inside the terminal docking phase, i.e. on ticks within
#   WINDOW_RANGE_M (3.0 m) of the boom tip -- the safety exclusion zone -- and
#   H1's p95 is computed on that windowed per-leg peak. Outside the window an
#   excursion is REPORTED and affects NO safety or M3b score (class C58).
#   Overtakes are UNCHANGED: the operator ruled the line stays at any
#   close-gated pass of the leader CG, so this file READS `real_overtake`
#   from the banked readout and never recomputes it (class C58b).
#   ADDS columns beside the banked v2 numbers; overwrites no verdict.
# PUBLIC API: rescore(legs_root, readout_path, window_m=3.0) -> dict;
#   main(argv) -> int.  CLI: --legs-root DIR --readout JSON [--out JSON]
#   [--window-m F] | --selftest.  Exit 0 ok / 1 validation FAILED / 2 usage.
# SIDE-EFFECTS: reads only; writes --out when given; --selftest uses a tempdir.
# CAPS: file soft 110 / hard 140 code-LoC; function cap 45.
# COVERAGE: --selftest (both directions: an in-window peak counts, an
#   identical peak outside the window does not) PLUS the mandatory
#   REPRODUCTION limb -- the whole-leg path must reproduce the banked
#   readout's per-cell `zone_violations` and `h1_truth_peak_up_m.p95`
#   EXACTLY, or rescore() raises and nothing is published (class C55).
# =============================================================================
import argparse
import csv
import glob
import gzip
import json
import os
import sys

_SCRIPTS = "/home/ubuntu/FlightDock/scripts"
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import gate_safety_events as gse  # noqa: E402  (signed pct + H1 bar)

WINDOW_RANGE_M = 3.0          # operator-ruled safety exclusion zone, from tip
BAR_M = gse.H1_BAR_M          # 2.00 m, read from the instrument, never retyped


def _leg_peaks(leg_dir, window_m):
    """(whole_leg_peak_up, windowed_peak_up) in metres; None when unreadable."""
    g = glob.glob(os.path.join(leg_dir, "env*", "flight_state_ep*.csv.gz"))
    whole = win = None
    for path in sorted(g):
        with gzip.open(path, "rt") as fh:
            for r in csv.DictReader(fh):
                try:
                    up = -float(r["raw_below_m"])
                    rng = float(r["raw_range_m"])
                except (TypeError, ValueError, KeyError):
                    continue
                whole = up if whole is None else max(whole, up)
                if rng <= window_m:
                    win = up if win is None else max(win, up)
    return whole, win


def _agg(peaks):
    v = [p for p in peaks if p is not None]
    if not v:
        return None
    return {"p95": round(gse.pct(v, .95), 3), "max": round(max(v), 3),
            "exceed_2m_legs": sum(1 for x in v if x > BAR_M), "n_legs": len(v)}


def rescore(legs_root, readout_path, window_m=WINDOW_RANGE_M):
    d = json.load(open(readout_path))
    banked = {r["leg"]: r for r in d["legs"]}
    cells, rows = {}, []
    for leg, r in banked.items():
        p = os.path.join(legs_root, leg)
        whole, win = _leg_peaks(p, window_m) if os.path.isdir(p) else (None, None)
        rows.append({"leg": leg, "cell": r["cell"], "whole": whole, "win": win,
                     "banked_whole": r.get("truth_peak_up_m"),
                     "passed": bool(r.get("passed")),
                     "real_overtake": bool(r.get("real_overtake"))})
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)
    out = {"legs_root": legs_root, "readout": readout_path, "window_m": window_m,
           "bar_m": BAR_M, "cells": {}, "legs": rows}
    for cell, rs in sorted(cells.items()):
        out["cells"][cell] = {
            "n": len(rs), "k_success": sum(1 for r in rs if r["passed"]),
            "overtake_legs": sum(1 for r in rs if r["real_overtake"]),
            "whole": _agg([r["whole"] for r in rs]),
            "windowed": _agg([r["win"] for r in rs])}
    _validate(out, d)
    return out


def _validate(out, banked_readout):
    """LIMB 1 REPRODUCTION: the whole-leg path must recover the banked
    readout's per-cell zone_violations and p95 EXACTLY. LIMB 2: per-leg
    whole-leg peak must equal the banked per-leg value (4 dp). Either
    failure raises -- a divergent rescore is never published (C55)."""
    problems = []
    for leg in out["legs"]:
        b, w = leg["banked_whole"], leg["whole"]
        if b is not None and w is not None and abs(round(w, 4) - b) > 5e-4:
            problems.append(f"{leg['leg']}: whole {w:.4f} != banked {b:.4f}")
    by_cell = {}
    for leg in out["legs"]:
        by_cell.setdefault(leg["cell"], []).append(leg["banked_whole"])
    for cell, s in out["cells"].items():
        bc = banked_readout["cells"].get(cell) or {}
        bh = bc.get("h1_truth_peak_up_m") or {}
        # reproduce the banked aggregates FROM the banked per-leg peaks: same
        # legs + same aggregation, free of gz-vs-readout rounding (4 dp limb
        # above ties gz to banked per leg)
        rep = _agg(by_cell.get(cell, []))
        if rep and bh:
            if rep["exceed_2m_legs"] != bc.get("zone_violations"):
                problems.append(f"{cell}: violations {rep['exceed_2m_legs']}"
                                f" != banked {bc.get('zone_violations')}")
            if abs(rep["p95"] - bh.get("p95", -1)) > 1e-9:
                problems.append(f"{cell}: p95 {rep['p95']} != banked {bh.get('p95')}")
    out["validation"] = {"problems": problems, "reproduced": not problems}
    if problems:
        raise SystemExit("REPRODUCTION FAILED (nothing published):\n  "
                         + "\n  ".join(problems[:12]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs-root")
    ap.add_argument("--readout")
    ap.add_argument("--out")
    ap.add_argument("--window-m", type=float, default=WINDOW_RANGE_M)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        import windowed_rescore_selftest as st
        return st.run()
    if not (a.legs_root and a.readout):
        ap.error("--legs-root and --readout required")
    res = rescore(a.legs_root, a.readout, a.window_m)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    for cell, s in sorted(res["cells"].items()):
        print("%-10s n=%2d k=%2d ot=%d | whole p95=%s viol=%s | WINDOWED p95=%s viol=%s"
              % (cell, s["n"], s["k_success"], s["overtake_legs"],
                 s["whole"]["p95"], s["whole"]["exceed_2m_legs"],
                 s["windowed"]["p95"], s["windowed"]["exceed_2m_legs"]))
    print("REPRODUCTION:", "PASS" if res["validation"]["reproduced"] else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
