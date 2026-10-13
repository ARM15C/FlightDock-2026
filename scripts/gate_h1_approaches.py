#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: the H1 APPROACH scorer -- criterion `v4_approach_range_armed`,
#   implementing the operator window ruling of 2026-08-11:
#     a ceiling bust counts only while within 10 m 3D RANGE of the boom tip
#     (INCLUSIVE); approaches are SEGMENTED at breakaways; leg = WORST approach.
#   Busts count REGARDLESS of FSM state, including EMERGENCY_BREAKAWAY -- the
#   guard's own climb close to the boom is a real bust, and the breakaway
#   interval belongs to the approach that triggered it. Every v2 field is
#   co-reported verbatim (score_leg is called, not reimplemented).
#
#   NOT A REPLACEMENT FOR A-9 (`gate_h1_windows.py`). A-9 EXCLUDES breakaway
#   rows from H1-D and ceilings recovery crests separately; this ruling COUNTS
#   them inside 10 m. They disagree by design -- [MEASURED, T-435] on
#   D1_43084_straight A-9's H1-D is 1.96 m (no bust) and this criterion gives
#   2.29 m (bust). Never fold one into the other.
#
#   Sibling module, not an edit to gate_safety_events.py: that file measures
#   298 code-LoC against its declared soft cap of 300. Caps are SPLIT, never
#   raised -- the same pressure that produced gate_h1_windows.py.
#
#   Row validity MIRRORS gate_safety_events._scan_csvs:116-127 (all five of
#   sim_time_s / raw_behind_m / raw_cross_m / raw_range_m / raw_below_m must
#   parse) so this scorer reproduces the T-435 audit bit-exactly. Upward
#   deviation = -below_m (gse:213). Range is the WRITER's own range_m =
#   sqrt(behind^2+cross^2+below^2) (env_formation_sitl.py:5413-5415).
#   Prereg: analysis/t436h1scorer_20260811T141500Z/PREREG_T436.md.
# PUBLIC API: score_leg_v4(leg_dir) -> dict; h1_approach_aggregate(rows) ->
#   dict; approach_row_fields(sev) -> dict; approach_reasons(sev) -> list;
#   ARM_RANGE_M; APPROACHES_FRAME. CLI: --leg-dir DIR | --selfcheck.
# SIDE-EFFECTS: reads leg CSVs read-only; prints JSON on the CLI path. No
#   writes, no RNG, no env toggles, no child process, no SITL, no network.
# CAPS: file soft 240 / hard 300 code-LoC; function LoC cap 60.
#   DECLARED at 200/260 in the T-436 prereg draft and REVISED to 240/300
#   BEFORE this file ever landed, because the module also took ownership of
#   the row-field and reason PROJECTIONS (approach_row_fields /
#   approach_reasons). Those live here rather than in the runner both for
#   layering -- the module that produces a field owns its projection -- and
#   because putting them in curriculum_gate_runner.py would have breached
#   ITS declared function cap. No established cap was raised: caps are
#   SPLIT, never raised, and this is a first declaration on a new file.
# COVERAGE: tests/unit/test_gate_h1_approaches.py + the in-module --selfcheck
#   (kept in-module deliberately: pytest CANNOT be run while a SITL leg is
#   flying, so the gate scorers must be checkable without it). Both exercise
#   every rule in BOTH directions.
# =============================================================================
"""H1 approach scorer: 10 m-armed, breakaway-segmented, worst-approach."""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # T-354 trap
import gate_safety_events as gse  # noqa: E402

ARM_RANGE_M = 10.0          # INCLUSIVE (<=)
RECOVERY_STATE = "EMERGENCY_BREAKAWAY"
APPROACHES_FRAME = "boom_tip_truth_armed10m_segmented_worst_approach"


def _blank(t):
    return {"t_start": t, "t_end": t, "n_rows": 0, "n_armed_rows": 0,
            "armed_peak_up_m": None, "range_at_armed_peak_m": None,
            "unarmed_peak_up_m": None, "ended_by": None}


def _close(a, ended_by):
    a["ended_by"] = ended_by
    for k in ("t_start", "t_end", "armed_peak_up_m",
              "range_at_armed_peak_m", "unarmed_peak_up_m"):
        if a[k] is not None:
            a[k] = round(a[k], 4)
    a["armed_bust"] = bool(a["armed_peak_up_m"] is not None
                           and a["armed_peak_up_m"] > gse.H1_BAR_M)
    return a


def _scan_approaches(leg_dir):
    """Segment at EMERGENCY_BREAKAWAY runs; per approach take the peak upward
    deviation ARMED at range <= 10 m. Returns None when no valid row;
    {'no_state_column': True} when rows exist but fsm_state is absent -- the
    leg is NOT decomposable and must fail CLOSED, never silently fall back to
    the whole-leg number."""
    out, cur, in_eb = [], None, False
    ticks = state_seen = 0
    for path in sorted(glob.glob(os.path.join(leg_dir, "env*",
                                              "paired_channels_ep*.csv"))):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                try:                    # validity mirrors gse._scan_csvs
                    t = float(r["sim_time_s"])
                    up = -float(r["raw_below_m"])
                    rg = float(r["raw_range_m"])
                    float(r["raw_behind_m"]), float(r["raw_cross_m"])
                except (TypeError, ValueError, KeyError):
                    continue
                ticks += 1
                st = r.get("fsm_state")
                if st is None:
                    continue
                state_seen += 1
                if in_eb and st != RECOVERY_STATE:
                    out.append(_close(cur, "BREAKAWAY"))
                    cur = None
                in_eb = st == RECOVERY_STATE
                if cur is None:
                    cur = _blank(t)
                cur["t_end"] = t
                cur["n_rows"] += 1
                if cur["unarmed_peak_up_m"] is None:
                    cur["unarmed_peak_up_m"] = up
                cur["unarmed_peak_up_m"] = max(cur["unarmed_peak_up_m"], up)
                if rg <= ARM_RANGE_M:
                    cur["n_armed_rows"] += 1
                    if (cur["armed_peak_up_m"] is None
                            or up > cur["armed_peak_up_m"]):
                        cur["armed_peak_up_m"] = up
                        cur["range_at_armed_peak_m"] = rg
    if ticks == 0:
        return None
    if state_seen == 0:
        return {"no_state_column": True, "ticks": ticks}
    if cur is not None:
        out.append(_close(cur, "BREAKAWAY_NO_EXIT" if in_eb else "LEG_END"))
    return {"no_state_column": False, "ticks": ticks, "approaches": out}


def score_leg_v4(leg_dir):
    """One leg -> v4 row = every v2 field verbatim + the approach fields."""
    row = dict(gse.score_leg(leg_dir))          # v2 co-reported, untouched
    row["h1a_arm_range_m"] = ARM_RANGE_M
    row["approaches_frame"] = APPROACHES_FRAME
    if row.get("instrument_gap"):
        return row
    w = _scan_approaches(leg_dir)
    if w is None or w.get("no_state_column"):
        row["instrument_gap"] = True
        row["gap_detail"] = {"approaches": False,
                             "why": ("no_valid_rows" if w is None
                                     else "fsm_state_column_absent")}
        return row
    aps = w["approaches"]
    pks = [a["armed_peak_up_m"] for a in aps
           if a["armed_peak_up_m"] is not None]
    worst = max(pks) if pks else None
    row.update({
        "approaches": aps, "n_approaches": len(aps),
        "truth_peak_up_armed_m": worst,
        "range_at_armed_peak_m": next(
            (a["range_at_armed_peak_m"] for a in aps
             if a["armed_peak_up_m"] == worst), None),
        "bust_h1a": bool(worst is not None and worst > gse.H1_BAR_M),
        "hard_fail_v4": bool(row["real_overtake"] or row["collision"]
                             or (worst is not None
                                 and worst > gse.H1_BAR_M)),
    })
    return row


def approach_row_fields(sev):
    """v4 per-leg fields for the gate row (per-leg detail mandate). The caller
    adds these ONLY under v4, so pre-v4 verdicts stay byte-identical."""
    return {k: sev.get(k) for k in (
        "hard_fail_v4", "bust_h1a", "truth_peak_up_armed_m",
        "range_at_armed_peak_m", "n_approaches", "approaches",
        "h1a_arm_range_m", "approaches_frame")}


def approach_reasons(sev):
    """The v4 performance reason, carrying its MEASURED quantity AND the range
    it was scored at (E-4 rule) -- the range is what makes a bust reviewable
    against the 10 m arming ruling."""
    if not sev.get("bust_h1a"):
        return []
    base = f"h1a_bust={sev['truth_peak_up_armed_m']:.2f}m>bar2.00"
    rng = sev.get("range_at_armed_peak_m")
    return [base if rng is None else f"{base}@range{rng:.2f}m"]


def h1_approach_aggregate(rows):
    """Gate-level v4 fields from per-leg rows (skips gap legs). exceed_legs_v4
    is the conjunct a launch gate asserting `exceed_2m_legs==0` must read on
    panels governed by this criterion. Legs that never came within the arming
    range contribute NO armed peak and are excluded from the percentiles --
    n_legs_armed states that denominator explicitly."""
    ok = [r for r in rows if not r.get("instrument_gap")]
    v = [r["truth_peak_up_armed_m"] for r in ok
         if r.get("truth_peak_up_armed_m") is not None]
    out = {"h1a_arm_range_m": ARM_RANGE_M, "approaches_frame": APPROACHES_FRAME,
           "exceed_legs_v4": sum(1 for r in ok if r.get("bust_h1a")),
           "n_legs_graded": len(ok), "n_legs_armed": len(v),
           "n_legs_never_armed": len(ok) - len(v),
           "n_approaches_total": sum(r.get("n_approaches", 0) for r in ok)}
    # gse.pct takes a FRACTION (.5/.95/.99), not a percent -- same call
    # convention as gse.h1_aggregate:261-263 and ghw:149-151.
    out["h1_truth_peak_up_armed_m"] = ({
        "p50": round(gse.pct(v, .5), 3), "p95": round(gse.pct(v, .95), 3),
        "p99": round(gse.pct(v, .99), 3), "max": round(max(v), 4),
        "exceed_2m_legs": sum(1 for x in v if x > gse.H1_BAR_M),
        "n_legs": len(v)} if v else None)
    return out


def _rows(specs):
    """Synthetic CSV rows: specs = [(fsm_state, up_m, range_m), ...]."""
    return [{"sim_time_s": i * 0.05, "raw_below_m": -up, "raw_range_m": rg,
             "raw_behind_m": 1.0, "raw_cross_m": 0.0, "fsm_state": st}
            for i, (st, up, rg) in enumerate(specs)]


def _write(tmp, specs):
    d = os.path.join(tmp, "env0")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "paired_channels_ep0001.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_rows(specs)[0]))
        w.writeheader()
        w.writerows(_rows(specs))
    return tmp


def _check_arming(td):
    """Arming BOTH directions, plus the range and bar boundaries."""
    D = "DOCKING_CREEP"
    s = _scan_approaches(_write(os.path.join(td, "a"),
                                [(D, 2.5, 9.9)] * 5))["approaches"]
    assert len(s) == 1 and s[0]["armed_bust"], "in-window bust must FIRE"
    s = _scan_approaches(_write(os.path.join(td, "b"),
                                [(D, 2.5, 10.1)] * 5))["approaches"]
    assert not s[0]["armed_bust"] and s[0]["n_armed_rows"] == 0, \
        "the same bust outside 10 m must REFUSE"
    s = _scan_approaches(_write(os.path.join(td, "c"),
                                [(D, 2.5, 10.0)] * 5))["approaches"]
    assert s[0]["armed_bust"], "boundary 10.0 m is INCLUSIVE, must ARM"
    s = _scan_approaches(_write(os.path.join(td, "d"),
                                [(D, 1.99, 5.0)] * 5))["approaches"]
    assert not s[0]["armed_bust"], "1.99 m must not bust a 2.00 m bar"


def _check_segmentation(td):
    """Segmentation, bust attribution, and both terminal shapes."""
    D, C = "DOCKING_CREEP", RECOVERY_STATE
    seg = [(D, 2.5, 5.0)] * 3 + [(C, 2.6, 5.0)] * 2 + [(D, 0.5, 5.0)] * 4
    s = _scan_approaches(_write(os.path.join(td, "e"), seg))["approaches"]
    assert len(s) == 2, f"breakaway must split the leg, got {len(s)}"
    assert s[0]["armed_bust"] and not s[1]["armed_bust"], \
        "the bust must attach to approach 0, not approach 1"
    assert s[0]["armed_peak_up_m"] == 2.6, \
        "the breakaway climb BELONGS to the approach that triggered it"
    assert s[0]["ended_by"] == "BREAKAWAY" and s[1]["ended_by"] == "LEG_END"
    s = _scan_approaches(_write(os.path.join(td, "f"),
                                [(D, 1.0, 5.0)] * 6))["approaches"]
    assert len(s) == 1, "a leg with no breakaway is ONE approach"
    s = _scan_approaches(_write(os.path.join(td, "g"),
                                [(D, 1.0, 5.0)] * 3
                                + [(C, 1.2, 5.0)] * 3))["approaches"]
    assert len(s) == 1 and s[0]["ended_by"] == "BREAKAWAY_NO_EXIT", \
        "a leg ending inside a breakaway keeps its final approach"


def _check_fail_closed(td):
    """No fsm_state column => gap, NEVER a silent whole-leg fallback."""
    rs = _rows([("DOCKING_CREEP", 1.0, 5.0)] * 3)
    for r in rs:
        del r["fsm_state"]
    d2 = os.path.join(td, "h", "env0")
    os.makedirs(d2)
    with open(os.path.join(d2, "paired_channels_ep0001.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    assert _scan_approaches(os.path.join(td, "h"))["no_state_column"], \
        "a missing fsm_state column must FAIL CLOSED, not fall back"


def selfcheck(verbose=True):
    """Every rule, BOTH directions. Returns True or raises."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _check_arming(td)
        _check_segmentation(td)
        _check_fail_closed(td)
    # >=3 armed legs so the INTERPOLATING branch of gse.pct is exercised --
    # a 1-element list short-circuits and would hide a percent/fraction
    # units error in the call convention.
    ag = h1_approach_aggregate(
        [{"truth_peak_up_armed_m": x, "bust_h1a": x > gse.H1_BAR_M,
          "n_approaches": 1} for x in (1.0, 1.5, 2.4, 2.6)]
        + [{"truth_peak_up_armed_m": None, "bust_h1a": False,
            "n_approaches": 1}])
    assert ag["exceed_legs_v4"] == 2 and ag["n_legs_armed"] == 4 \
        and ag["n_legs_never_armed"] == 1, "never-armed legs are disclosed"
    h = ag["h1_truth_peak_up_armed_m"]
    assert h["p50"] == 1.95 and h["max"] == 2.6 and h["n_legs"] == 4, \
        f"percentiles must take a FRACTION, not a percent: {h}"
    if verbose:
        print("SELFCHECK PASS: arming both directions + inclusive boundary, "
              "segmentation, breakaway-climb attribution, worst-approach, "
              "no-breakaway, NO_EXIT, fsm_state fail-closed, aggregate")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="H1 approach scorer (v4)")
    ap.add_argument("--leg-dir")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return 0 if selfcheck() else 3
    if not a.leg_dir:
        ap.error("--leg-dir or --selfcheck required")
    print(json.dumps(score_leg_v4(a.leg_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
