#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: mechanical per-leg detail report for a gate/eval legs dir —
#   the operator's standing directive (08-05, after the FIFTH false FAIL of
#   the campaign): every stage verdict ships per-leg detail, and every FAIL
#   must sit next to its measured quantities so an instrument-gap fail looks
#   exactly as anomalous as it is. Promoted from the 08-05 scratchpad
#   extractor that caught the T-273 missed-bridge false FAIL.
#
# PER LEG: M3b dwell episodes >=3.0 s on truth/perceived/intersection masks
#   (gap-tolerant <=0.25 s, convention of record; gapped runs disclosed) +
#   best boom-tip range + below/vsep(CG) stats + guard tokens + in-front
#   excursions (all-ticks AND close-gated, T-159 artifact rule) + T1
#   contact-plane crossings (conservative proxy, NOT physical contact) +
#   verdict join when an adjudication JSON is given. PANEL: abort
#   classification (v2 criterion: false/missed aborts).
# S-13: the panel real-event clause reads the ENV
#   TERMINAL via leg_terminal_reader — the old guard_tokens grep for
#   SAFETY_COLLISION could NEVER match (translog vocabulary is FSM states).
#   AI-1/AI-2/AI-3 numbers from BEFORE this commit were collision-blind and
#   must not be quoted. AI-3 caveat unchanged: `ev` is the SCORER'S OWN
#   predicate, so missed_abort_legs measures guard-vs-scorer disagreement,
#   a labelled proxy — see CURRICULUM_ROADMAP 6.3.1.
#
# PUBLIC API: leg_detail(leg_dir) -> dict; panel_rollup(legs) -> dict;
#   main(argv) -> int. CLI: --legs-dir DIR [--adjudication JSON]
#   [--leg-glob PATTERN] [--out JSONL]. Exit 0 on success, 2 on usage/IO.
# SIDE-EFFECTS: reads only; writes --out (JSONL) if given, else stdout.
# CAPS: file soft 210 / hard 250 LoC; function cap 70 (leg_detail carries
#   the whole single-pass CSV scan at 67 — splitting it would thread six
#   accumulators through helper signatures for no reader benefit).
# COVERAGE: tests/unit/test_gate_leg_report.py.
# =============================================================================
import argparse
import csv
import glob
import json
import os
import re
import sys

_SCRIPTS = "/home/ubuntu/FlightDock/scripts"
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import leg_terminal_reader as _ltr  # noqa: E402 (S-13: terminal channel)
import leg_demoted_reader as _ldr  # noqa: E402 (TERMDEMOTE-767)

BOOM_VERT_M = 4.31          # leader CG above boom tip (GEOMETRY_OF_RECORD r2)
T1_PLANE_M = -3.42          # contact plane (conservative proxy)
GAP_TOL_S = 0.25            # dwell convention of record
DWELL_S = 3.0
CLOSE_RANGE_M, CLOSE_CROSS_M = 30.0, 10.0
H1_BAR_M = 2.0
GUARD_RE = re.compile(r"(EMERGENCY_BREAKAWAY|SAFETY_COLLISION|BELLY_CLAMP|"
                      r"GUARD_[A-Z_]+|BREAKAWAY_[A-Z_]+)")


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (k - f) * (s[f + 1] - s[f])


def _episodes(rows, key):
    """[(dur_s, n_gaps_gt_tol)] for contiguous key==True runs."""
    eps, start, last_t, gaps = [], None, None, 0
    for t, flags in rows:
        if flags.get(key):
            if start is None:
                start, gaps = t, 0
            elif t - last_t > GAP_TOL_S:
                gaps += 1
            last_t = t
        elif start is not None:
            eps.append((last_t - start, gaps))
            start = None
    if start is not None:
        eps.append((last_t - start, gaps))
    return eps


# TERMDEMOTE-767. Extracted rather than inlined: adding these six lines
# inside leg_detail took it to 72 against its function cap of 70, which
# check_loc_caps caught. A cap breach is fixed by extraction, never by moving
# the cap. Reporting-only -- no panel criterion keys on these; without them a
# demoted leg reads as a clean TIMEOUT on the panel.
def _demoted_fields(leg_dir):
    """Did each demotable condition fire, terminal or not (None if unread)."""
    return {"collision_condition_fired":
            _ldr.condition_fired(leg_dir, "SAFETY_COLLISION"),
            "airspeed_condition_fired":
            _ldr.condition_fired(leg_dir, "SAFETY_AIRSPEED")}


def leg_detail(leg_dir):
    o = {"leg": os.path.basename(leg_dir.rstrip("/")),
         # S-13: the env's own terminal(s); None = unreadable (disclosed, and
         # panel_rollup treats unknown as NOT-an-event only with the count of
         # unknowns reported so the gap cannot hide)
         "terminal_types": _ltr.terminal_types(leg_dir),
         **_demoted_fields(leg_dir)}
    rows, below, rng = [], [], []
    min_behind_all = min_behind_close = 1e18
    t1_cross, prev_below = 0, None
    for p in sorted(glob.glob(os.path.join(leg_dir, "env*",
                                           "paired_channels_ep*.csv"))):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    t = float(r["sim_time_s"])
                    bl = float(r["raw_below_m"])
                    bh = float(r["raw_behind_m"])
                    cx = float(r["raw_cross_m"])
                    rg = float(r["raw_range_m"])
                except (TypeError, ValueError, KeyError):
                    continue
                tm = r.get("raw_m3b") in ("1", "True", "true")
                cm = r.get("ctrl_m3b") in ("1", "True", "true")
                rows.append((t, {"truth": tm, "perc": cm,
                                 "both": tm and cm}))
                below.append(bl)
                rng.append(rg)
                min_behind_all = min(min_behind_all, bh)
                if rg <= CLOSE_RANGE_M and abs(cx) <= CLOSE_CROSS_M:
                    min_behind_close = min(min_behind_close, bh)
                if prev_below is not None and prev_below > T1_PLANE_M >= bl:
                    t1_cross += 1
                prev_below = bl
    if not rows:
        o["error"] = "no paired CSV rows"
        return o
    rows.sort(key=lambda x: x[0])
    o["ticks"] = len(rows)
    o["capture_span_s"] = round(rows[-1][0] - rows[0][0], 1)
    for key in ("truth", "perc", "both"):
        eps = _episodes(rows, key)
        durs = sorted((round(e[0], 2) for e in eps if e[0] >= DWELL_S),
                      reverse=True)
        o["dwell_" + key] = {
            "episodes_ge3s": len(durs),
            "min": min(durs) if durs else None,
            "med": round(_pct(durs, .5), 2) if durs else None,
            "max": max(durs) if durs else None,
            "all": durs[:8],
            "gapped_runs": sum(1 for e in eps
                               if e[0] >= DWELL_S and e[1] > 0),
            "best_any": round(max((e[0] for e in eps), default=0.0), 2)}
    o["min_range_m"] = round(min(rng), 3)
    o["below_min_med_max"] = [round(min(below), 2),
                              round(_pct(below, .5), 2),
                              round(max(below), 2)]
    o["vsep_cg_min_med_max"] = [round(v + BOOM_VERT_M, 2)
                                for v in o["below_min_med_max"]]
    o["max_infront_boomtip_m"] = round(max(0.0, -min_behind_all), 3)
    o["max_infront_boomtip_close_m"] = (
        round(max(0.0, -min_behind_close), 3)
        if min_behind_close < 1e17 else 0.0)
    o["t1_plane_crossings"] = t1_cross
    guards = {}
    for lg in glob.glob(os.path.join(leg_dir, "episode_*",
                                     "TRANSITION_SNAPSHOT.log")):
        for ln in open(lg, errors="replace"):
            for m in GUARD_RE.finditer(ln):
                guards[m.group(1)] = guards.get(m.group(1), 0) + 1
    o["guard_tokens"] = guards
    return o


def panel_rollup(legs):
    """v2-criterion abort classification across the panel. S-13: the
    collision term reads the ENV TERMINAL (the old guard_tokens grep matched
    a token real translogs cannot contain — a clause that could not fire).
    Legs with an unreadable terminal are counted in `terminal_unknown_legs`
    so the gap is denominator-visible, never silent."""
    fa = ma = fired = real = unk = 0
    for o in legs:
        v = o.get("verdict") or {}
        brk = o.get("guard_tokens", {}).get("EMERGENCY_BREAKAWAY", 0)
        terms = o.get("terminal_types")
        unk += 1 if terms is None else 0
        ev = bool(v.get("real_overtake")) or \
            (v.get("truth_peak_up_m") or 0) > H1_BAR_M or \
            "SAFETY_COLLISION" in (terms or [])
        fired += 1 if brk else 0
        real += 1 if ev else 0
        fa += 1 if (brk and not ev) else 0
        ma += 1 if (ev and not brk) else 0
    return {"legs": len(legs), "legs_with_abort": fired,
            "legs_with_real_event": real,
            "false_abort_legs": fa, "missed_abort_legs": ma,
            "terminal_unknown_legs": unk}


def main(argv=None):
    ap = argparse.ArgumentParser(description="per-leg gate detail report")
    ap.add_argument("--legs-dir", required=True)
    ap.add_argument("--adjudication", default=None)
    ap.add_argument("--leg-glob", default="*")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not os.path.isdir(args.legs_dir):
        print(f"ERROR: not a directory: {args.legs_dir}", file=sys.stderr)
        return 2
    vrows = {}
    if args.adjudication:
        adj = json.load(open(args.adjudication))
        vrows = {r["leg"]: r for r in adj.get("legs", [])}
    out = []
    for d in sorted(glob.glob(os.path.join(args.legs_dir, args.leg_glob))):
        if not os.path.isdir(d):
            continue
        o = leg_detail(d)
        v = vrows.get(o["leg"])
        if v is not None:
            o["verdict"] = {k: v.get(k) for k in
                            ("passed", "fail_reasons", "evidence_problems",
                             "gradable", "intersection_v3_s", "real_overtake",
                             "truth_peak_up_m", "fused_peak_up_m",
                             "hard_fails", "capture_span_s")}
        out.append(o)
    lines = [json.dumps(o, sort_keys=True) for o in out]
    lines.append(json.dumps({"panel": panel_rollup(out)}, sort_keys=True))
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
