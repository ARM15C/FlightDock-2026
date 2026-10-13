#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Curriculum-v3.1 SEALED-GATE adjudication core (P1-a slice:
#   offline adjudication + A7/RC3 retention contract + SectionC zero-shot
#   restriction checks + dry-run eval-command emission). Live eval spawning,
#   train.py pause hooks, and leg_meta/safety_summary WRITERS land in P1-b —
#   this module only READS leg evidence. Per-leg pass verdicts come from the
#   FROZEN instrument chain loaded BY PATH (t0_adjudicate.boot_pass, decisive
#   key `intersection`, + oc_exact.wilson_lb) — never re-implemented here.
# BARS (v3.1 amendment, defaults pinned by unit test):
#   mid12: N=12 balanced 6/6, pass = k>=10 AND >=4/6 per trajectory (A1/base Q3)
#   anchor30: N=30 balanced 15/15, pass = k>=27 (Wilson LB descriptive, A3)
#   cert50: N=50 balanced 25/25, PRIMARY pooled one-sided Wilson LB>=0.80
#            (== k>=45); per-traj floor >=21/25 SECONDARY descriptive with
#            LB(21/25)=0.6875 disclosed, NEVER gates (A4/RC9)
#   retention screen: N=6/stratum balanced 3/3, trigger = k<=4, DIAGNOSTIC (RC3)
#   retention confirm: N=12 balanced 6/6 disjoint seeds, REPLACES screen,
#            confirmed failure = k<=9; 2 confirmed strikes => demote (RC3)
# EVIDENCE RULES:
#   leg pass = intersection AND safety conjunct per the spec-selected
#   criterion, echoed in the verdict:
#   - "v2_real_events" (DEFAULT): hard_fail_v2 == False from
#     gate_safety_events.score_leg — real CG-frame overtake OR truth-side
#     H1 ceiling bust OR collision. Legacy guard-firing count is
#     always co-reported per leg (field `hard_fails`).
#   - "v1_guard_firings": legacy — hard_fails==0 from safety_summary.json.
#   E-4 EVIDENCE/PERFORMANCE SPLIT (the 5-false-FAIL class, mechanized):
#     PERFORMANCE fail_reasons each carry their MEASURED quantity
#     (e.g. intersection_dwell_fail:best=1.20s). EVIDENCE problems (unbridged
#     paired/ layout, dead/null instrument, missing CSVs/translog/meta) make
#     the leg INDETERMINATE: gradable=false, EXCLUDED from k/n, named in
#     protocol_problems => invalid_protocol => panel fails CLOSED until the
#     leg is replaced or repaired. Evidence-absence is never a counted
#     performance miss ("evidence missing means INCONCLUSIVE, not GREEN" —
#     and not RED either). A second independent reader (naive both-mask
#     dwell from the leg's own env CSVs) annotates CONTRADICTION when the
#     frozen chain returned null over rich data — it can stop false FAILs,
#     never mint a PASS. INDETERMINATE_SETUP is a RESERVED evidence class:
#     no criterion fires until the 43435 RCA pre-registers one.
#   Every verdict emits H1 fields (signed truth/fused peak-up + gate p50/p95/
#   p99/max/exceedances) in BOTH modes. Gradable legs: missing/corrupt
#   performance evidence still fails closed; never resampled.
#   Retention only: RC3 carve-out — abort_pre_control=true =>
#   excluded-with-disclosure. Balance violation or wrong N =>
#   invalid_protocol (fail-closed, passed=false).
# PUBLIC API: leg_verdict, gate_adjudicate, retention_adjudicate,
#   zeroshot_check, split_evidence, main. CLI: --mode
#   {gate,retention,zeroshot,check} --legs-dir --spec JSONFILE --out JSONFILE
#   [--dry-run-spawn].
# SIDE-EFFECTS: writes --out only; loads frozen modules read-only.
# CAPS: file soft 320 / hard 360 LoC; function cap 80 (RAISED 60 -> 80 by
#   E-4, DECLARED: leg_verdict carries the whole evidence/performance split
#   in one read — splitting it would smear the reason taxonomy across
#   helpers and hide the fail-closed ordering the tests pin).
# LAYERING (T-353, STRUCTURAL): evidence ADMISSIBILITY ("may this leg be
#   graded?") lives in the sibling gate_evidence_rules.py and is re-exported
#   here under the original names; this module owns ADJUDICATION ("did it
#   pass?"). Dependency is one-way: runner -> rules, never back.
# COVERAGE: tests/unit/test_curriculum_gate_runner.py (deliberately UNCHANGED
#   across the T-353 split — a passing untouched test is the equivalence
#   evidence, alongside the byte-level pre/post harness on 14 real legs).
# =============================================================================
import argparse
import importlib.util
import json
import os
import sys

# T-354 latent-fragility fix, disclosed: this module NEVER put its own
# directory on sys.path — `import gate_safety_events` resolved only when a
# co-resident import had already done so (worked by session accident, failed
# under spec_from_file_location in isolation). Siblings (gate_h1_windows,
# gate_safety_events) already self-insert; now so does the runner.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gate_evidence_rules as _ger  # noqa: E402 T-353: evidence layer
import gate_h1_approaches as gha  # noqa: E402 T-436: H1 approach scorer
import gate_h1_windows as ghw  # noqa: E402 T-354: A-9 split-window scorer
import gate_safety_events as gse  # noqa: E402 B-1: real-safety-event scoring

FD = "/home/ubuntu/FlightDock"
# T-354: "v3_windowed" = the SIGNED A-9 criterion —
# hard_fail_v3 from gate_h1_windows.score_leg_v3 (H1-D docking bar 2.00 OR
# H1-R recovery ceiling 2.50 OR real overtake OR collision terminal), with
# the B2 truth-refuted co-report. STRICTLY OPT-IN via spec: the default stays
# v2_real_events and the default output is byte-identical (harness-pinned) —
# pre-A-9 panels are adjudicated exactly as before.
# T-436: `v4_approach_range_armed` implements the operator H1 window ruling of
# 2026-08-11 (bust arms within 10 m 3D range of the boom tip; approaches
# SEGMENTED at breakaways; leg = WORST approach). It is NOT v3: A-9 EXCLUDES
# breakaway rows, v4 COUNTS them inside 10 m. STRICTLY OPT-IN via spec; the
# default stays v2_real_events and default output is byte-identical.
SAFETY_CRITERIA = ("v2_real_events", "v1_guard_firings", "v3_windowed",
                   "v4_approach_range_armed")
T0_ADJ = os.path.join(FD, "analysis/t0adj_prebuild_20260712T150000Z/t0_adjudicate.py")
OC_EXACT = os.path.join(FD, "analysis/sgd1_gate_memo_20260710T153822Z/oc_exact.py")
TRAJS = ("straight", "racetrack")
BARS = {"mid12": dict(n=12, per_traj=6, k_pass=10, traj_floor=4),
        "anchor30": dict(n=30, per_traj=15, k_pass=27, traj_floor=None),
        "cert50": dict(n=50, per_traj=25, lb_bar=0.80,
                       traj_floor_desc=21)}          # floor descriptive only
RET = dict(screen_n=6, screen_traj=3, screen_trigger=4,
           confirm_n=12, confirm_traj=6, confirm_fail=9, strikes_demote=2)
# T-353 (STRUCTURAL): the EVIDENCE-ADMISSIBILITY layer — the capture screen,
# the unbridged refusal, the naive second reader and their constants — now
# lives in gate_evidence_rules.py. It moved VERBATIM (zero behavior delta,
# proven by a byte-level pre/post equivalence harness on 14 real banked legs
# plus this module's UNCHANGED unit test). Re-exported below under the exact
# ORIGINAL names so every caller and every test resolves as before: the
# split is a file boundary, not an interface change.
MIN_CAPTURE_S = _ger.MIN_CAPTURE_S
SETUP_SCREEN_REASON = _ger.SETUP_SCREEN_REASON
NAIVE_GAP_TOL_S = _ger.NAIVE_GAP_TOL_S
NAIVE_DWELL_S = _ger.NAIVE_DWELL_S
_unbridged = _ger.unbridged
_naive_m3b_best = _ger.naive_m3b_best
split_evidence = _ger.split_evidence
_evidence_problem = _ger.evidence_problem
split_short_captures = _ger.split_short_captures
_short_problem = _ger.short_problem


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frozen():
    return _load(T0_ADJ, "t0_adjudicate"), _load(OC_EXACT, "oc_exact")


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _v3_row_fields(sev):
    """T-354: A-9 per-leg fields (per-leg detail mandate). Keys are ABSENT
    under the default criterion so pre-A-9 verdicts stay byte-identical."""
    return {k: sev.get(k) for k in (
        "hard_fail_v3", "bust_h1d", "bust_h1r", "truth_peak_up_docking_m",
        "recovery_crest_max_m", "recovery_windows",
        "collision_terminal_truth_refuted",          # SUPERSEDED proxy
        "collision_terminal_envelope_refuted",       # T-437 correct field
        "envelope_max_consecutive", "h1r_ceiling_m")}


def _safety_reasons(sev, safety_criteria):
    """The safety PERFORMANCE reasons for one leg, each carrying its measured
    quantity (E-4 rule). The H1 limb is criterion-selected; overtake and
    collision limbs are common to every criterion. Extracted from
    leg_verdict by T-436 to keep it inside its declared function cap --
    ordering and text are unchanged, which the untouched runner tests pin."""
    reasons = []
    if sev["real_overtake"]:
        peak = sev.get("real_overtake_peak_infront_m")
        reasons.append("real_overtake:peak_infront="
                       + (f"{peak:.2f}m" if peak is not None else "n/a"))
    if safety_criteria == "v3_windowed":
        reasons += _v3_window_reasons(sev)         # A-9: H1-D / H1-R limbs
    elif safety_criteria == "v4_approach_range_armed":
        reasons += gha.approach_reasons(sev)       # T-436: armed worst-approach
    elif sev["truth_ceiling_bust"]:
        reasons.append(f"truth_ceiling_bust={sev['truth_peak_up_m']:.2f}m")
    if sev["collision"]:
        n = sev.get("terminal_types", []).count("SAFETY_COLLISION")
        reasons.append(f"collision:terminal=SAFETY_COLLISION:n={n}")
    return reasons


def _score_leg_for(leg_dir, safety_criteria):
    """B-1/T-354/T-436: the safety scorer selected by the criterion. v3 and v4
    rows are strict SUPERSETS of v2 rows (both co-report every v2 field
    verbatim), so every v2 field path downstream is unchanged."""
    if safety_criteria == "v3_windowed":
        return ghw.score_leg_v3(leg_dir)
    if safety_criteria == "v4_approach_range_armed":
        return gha.score_leg_v4(leg_dir)
    return gse.score_leg(leg_dir)


def _v3_window_reasons(sev):
    """A-9 performance reasons, each carrying its MEASURED quantity (E-4
    rule). Replaces the single truth_ceiling_bust limb under v3_windowed."""
    reasons = []
    if sev.get("bust_h1d"):
        reasons.append(
            f"h1d_bust={sev['truth_peak_up_docking_m']:.2f}m>bar2.00")
    if sev.get("bust_h1r"):
        reasons.append(f"h1r_bust:crest={sev['recovery_crest_max_m']:.2f}m"
                       f">ceiling{sev['h1r_ceiling_m']:.2f}")
    return reasons


def leg_verdict(leg_dir, adj, safety_criteria="v2_real_events"):
    """One leg -> pass/fail + provenance, E-4 split: `fail_reasons` =
    PERFORMANCE misses, each carrying its measured quantity;
    `evidence_problems` = the instruments could not measure the flight
    (gradable=false -> INDETERMINATE at the panel, excluded from k/n).
    Evidence legs can never PASS and are never a counted performance miss.

    B-1: safety conjunct selected by `safety_criteria`; H1 + real-event
    fields emitted in BOTH modes; legacy hard_fails always co-reported."""
    assert safety_criteria in SAFETY_CRITERIA, safety_criteria
    meta = _read_json(os.path.join(leg_dir, "leg_meta.json"))
    safety = _read_json(os.path.join(leg_dir, "safety_summary.json"))
    bp = adj.boot_pass(leg_dir)
    # T-354: v3 rows are a strict SUPERSET of v2 rows (v2 fields co-reported
    # verbatim by score_leg_v3), so every v2 field path below is unchanged.
    sev = _score_leg_for(leg_dir, safety_criteria)
    hard_fails = None if safety is None else safety.get("hard_fails")
    reasons, evidence = [], []
    if _unbridged(leg_dir):
        evidence.append("INDETERMINATE_UNBRIDGED")
    if meta is None or meta.get("trajectory") not in TRAJS:
        evidence.append("leg_meta_missing_or_bad")
    if safety_criteria == "v1_guard_firings":
        if hard_fails is None:
            evidence.append("safety_summary_missing")
        elif hard_fails > 0:
            reasons.append(f"safety_hard_fails={hard_fails}")
    elif sev.get("instrument_gap"):
        evidence.append("safety_instrument_gap")
    else:
        reasons += _safety_reasons(sev, safety_criteria)
    naive = None
    if not bp["intersection"]:
        v3s = bp["intersection_v3_s"]
        naive = _naive_m3b_best(leg_dir)
        if bp.get("intersection_live"):
            reasons.append("intersection_dwell_fail:best="
                           f"{(v3s if v3s is not None else 0.0):.2f}s")
        elif naive is not None and not evidence:
            # bridge present + the leg's own data readable, yet the frozen
            # chain's contact-window row filter found NOTHING. Two shapes:
            # naive also ~zero => the leg measurably never reached the
            # contact window (the 43435 shape) — a PERFORMANCE zero; naive
            # shows a real dwell the frozen chain could not see => genuine
            # CONTRADICTION, evidence-class HOLD for a human.
            if naive >= NAIVE_DWELL_S:
                evidence.append("CONTRADICTION_frozen_null_naive_best="
                                f"{naive:.2f}s")
            else:
                reasons.append("intersection_dwell_fail:best=0.00s"
                               f";no_contact_rows;naive_best={naive:.2f}s")
        else:
            # the frozen chain measured NOTHING and there is no readable
            # leg-side data to check it against — evidence, not performance
            evidence.append("instrument_null_dwell")
            if naive is not None and naive >= NAIVE_DWELL_S:
                evidence.append("CONTRADICTION_frozen_null_naive_best="
                                f"{naive:.2f}s")
    row = {"leg": os.path.basename(leg_dir.rstrip("/")),
           "trajectory": (meta or {}).get("trajectory"),
           "seed": (meta or {}).get("seed"),
           "abort_pre_control": bool((meta or {}).get("abort_pre_control")),
           "intersection_v3_s": bp["intersection_v3_s"],
           "intersection_live": bool(bp.get("intersection_live")),
           "naive_m3b_best_s": naive,
           "hard_fails": hard_fails,
           "real_overtake": sev.get("real_overtake"),
           "truth_peak_up_m": sev.get("truth_peak_up_m"),
           "fused_peak_up_m": sev.get("fused_peak_up_m"),
           "hard_fail_v2": sev.get("hard_fail_v2"),
           "instrument_gap": bool(sev.get("instrument_gap")),
           "capture_span_s": sev.get("capture_span_s"),
           "capture_complete": sev.get("capture_complete"),   # S-4 screen
           "evidence_problems": evidence, "gradable": not evidence,
           "passed": not reasons and not evidence,
           "fail_reasons": reasons}
    if safety_criteria == "v3_windowed":
        row.update(_v3_row_fields(sev))
    if safety_criteria == "v4_approach_range_armed":
        row.update(gha.approach_row_fields(sev))
    return row


def _leg_dirs(legs_dir):
    return sorted(os.path.join(legs_dir, d) for d in os.listdir(legs_dir)
                  if os.path.isdir(os.path.join(legs_dir, d)))


def _traj_counts(rows):
    per = {t: [0, 0] for t in TRAJS}                      # [n, k]
    for r in rows:
        if r["trajectory"] in per:
            per[r["trajectory"]][0] += 1
            per[r["trajectory"]][1] += int(r["passed"])
    return per


def gate_adjudicate(rows, gate_type, wilson_lb, safety_criteria="v2_real_events"):
    """Sealed gate on a fixed-N batch. Fail-closed on protocol violations."""
    bar = BARS[gate_type]
    rows, evid_excl = split_evidence(rows)             # E-4: evidence first
    rows, short_excl = split_short_captures(rows)
    # S-4: `and r["passed"]` keeps this field's VALUE true to its NAME. Under
    # the old screen every kept short leg had necessarily passed, so the two
    # agreed by construction; the corrected screen also keeps short legs that
    # FAILED, and listing those under `..._kept_passed` would be a false
    # statement in an evidence artifact. They are not hidden: they are counted
    # in n/k and carry capture_span_s + capture_complete in `legs`.
    short_kept = [r["leg"] for r in rows
                  if r.get("capture_span_s") is not None
                  and r["capture_span_s"] < MIN_CAPTURE_S and r["passed"]]
    n, k = len(rows), sum(r["passed"] for r in rows)
    per = _traj_counts(rows)
    problems = []
    if evid_excl:
        problems.append(_evidence_problem(evid_excl))
    if short_excl:
        problems.append(_short_problem(short_excl))
    if n != bar["n"]:
        problems.append(f"N={n}!={bar['n']}")
    if any(per[t][0] != bar["per_traj"] for t in TRAJS):
        problems.append("trajectory_balance_violation")
    if any(r["trajectory"] not in TRAJS for r in rows):
        problems.append("unknown_trajectory_leg")
    out = {"gate_type": gate_type, "n": n, "k": k,
           "per_trajectory": {t: {"n": per[t][0], "k": per[t][1]} for t in TRAJS},
           "wilson_lb_one_sided_95": round(wilson_lb(k, n), 6) if n else None,
           "safety_criteria": safety_criteria,      # B-1: selector echoed
           "invalid_protocol": bool(problems), "protocol_problems": problems,
           "evidence_excluded": [r["leg"] for r in evid_excl],   # E-4
           "short_capture_excluded": [r["leg"] for r in short_excl],
           "short_capture_kept_passed": short_kept,   # T-270 §7.1 disclosure
           "legs": rows + short_excl + evid_excl}
    out.update(gse.h1_aggregate(rows))              # B-1: H1 always emitted
    if safety_criteria == "v3_windowed":
        # T-354: A-9.5 — exceed_legs_v3 (H1-D OR H1-R bust) is the quantity
        # a launch gate asserting `exceed_2m_legs==0` reads on A-9 panels.
        out.update(ghw.h1_windowed_aggregate(rows))
    if safety_criteria == "v4_approach_range_armed":
        # T-436: exceed_legs_v4 is the quantity a launch gate asserting
        # `exceed_2m_legs==0` reads on panels governed by the window ruling.
        out.update(gha.h1_approach_aggregate(rows))
    if problems:
        out["passed"] = False
        return out
    if gate_type == "cert50":
        out["primary_pooled_lb_bar"] = bar["lb_bar"]
        out["passed"] = out["wilson_lb_one_sided_95"] >= bar["lb_bar"]
        out["per_traj_floor_descriptive"] = {
            t: {"k": per[t][1], "floor": bar["traj_floor_desc"],
                "met": per[t][1] >= bar["traj_floor_desc"],
                "lb_disclosed": round(wilson_lb(bar["traj_floor_desc"], 25), 4),
                "note": "SECONDARY engineering floor, never gates (RC9)"}
            for t in TRAJS}
    else:
        ok = k >= bar["k_pass"]
        if bar["traj_floor"] is not None:
            ok = ok and all(per[t][1] >= bar["traj_floor"] for t in TRAJS)
        out["passed"] = bool(ok)
    return out


def retention_adjudicate(strata, phase, prior_strikes, wilson_lb):
    """RC3 operating contract. strata = {name: [leg rows]}. Screen is
    DIAGNOSTIC (trigger list); confirm REPLACES screen and mints strikes."""
    is_scr = phase == "screen"
    n_req = RET["screen_n"] if is_scr else RET["confirm_n"]
    t_req = RET["screen_traj"] if is_scr else RET["confirm_traj"]
    thresh = RET["screen_trigger"] if is_scr else RET["confirm_fail"]
    out = {"phase": phase, "replaces_screen": not is_scr, "strata": {},
           "triggered": [], "confirmed_strikes": [], "invalid_protocol": False}
    for name, rows in strata.items():
        rows, evid_excl = split_evidence(rows)            # E-4 split
        rows, short_excl = split_short_captures(rows)     # T-271 screen
        usable = [r for r in rows if not r["abort_pre_control"]]
        excluded = len(rows) - len(usable)                # RC3 carve-out
        n, k = len(usable), sum(r["passed"] for r in usable)
        per = _traj_counts(usable)
        problems = []
        if evid_excl:
            problems.append(_evidence_problem(evid_excl))
        if short_excl:
            problems.append(_short_problem(short_excl))
        if n != n_req:
            problems.append(f"N={n}!={n_req}")
        if any(per[t][0] != t_req for t in TRAJS):
            problems.append("trajectory_balance_violation")
        hit = n and k <= thresh
        out["strata"][name] = {
            "n": n, "k": k, "excluded_pre_control": excluded,
            "wilson_lb_one_sided_95": round(wilson_lb(k, n), 6) if n else None,
            "protocol_problems": problems,
            ("triggered" if is_scr else "confirmed_failure"): bool(hit)}
        if problems:
            out["invalid_protocol"] = True
        elif hit:
            (out["triggered"] if is_scr else out["confirmed_strikes"]).append(name)
    strikes = int(prior_strikes) + (0 if is_scr else len(out["confirmed_strikes"]))
    out["strikes_total"] = strikes
    out["demote"] = (not is_scr) and (not out["invalid_protocol"]) \
        and strikes >= RET["strikes_demote"]
    out["note"] = ("screen is DIAGNOSTIC ONLY; confirmation on disjoint seeds "
                   "REPLACES the screen, never pooled (RC3)")
    return out


def zeroshot_check(ladder, anchors, skipped, candidate, probe_k=None,
                   wilson_lb=None):
    """SectionC restrictions. Returns allowed + reasons; probe 6/6 only
    AUTHORIZES SCHEDULING the full sealed gate (LB 0.689 certifies nothing)."""
    reasons = []
    if candidate in anchors:
        reasons.append("anchor_never_skippable")
    i = ladder.index(candidate)
    if any(nb in anchors for nb in ladder[max(0, i - 1):i + 2] if nb != candidate):
        reasons.append("adjacent_to_anchor")
    if i > 0 and ladder[i - 1] in skipped:
        reasons.append("consecutive_skip")
    out = {"candidate": candidate, "allowed": not reasons, "reasons": reasons}
    if probe_k is not None:
        out["probe"] = {"k": probe_k, "n": 6,
                        "schedules_full_gate": not reasons and probe_k == 6,
                        "lb_disclosed": round(wilson_lb(6, 6), 4) if wilson_lb
                        else None,
                        "note": "instrumentation-only; certifies nothing"}
    return out


def main():
    ap = argparse.ArgumentParser(description="Curriculum v3.1 sealed-gate runner (P1-a)")
    ap.add_argument("--mode", required=True,
                    choices=("gate", "retention", "zeroshot", "check"))
    ap.add_argument("--legs-dir", default=None)
    ap.add_argument("--spec", default=None, help="JSON spec file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run-spawn", action="store_true",
                    help="print eval spawn commands, spawn NOTHING (P1-b wires live)")
    args = ap.parse_args()
    adj, oc = _frozen()
    spec = _read_json(args.spec) if args.spec else {}
    if args.mode == "check":
        lb = oc.wilson_lb
        pins = (round(lb(45, 50), 4), round(lb(27, 30), 4),
                round(lb(21, 25), 4), round(lb(6, 6), 3))
        # instrument-exact: LB(45,50)=0.80846 (doc's "0.8084" is a truncation)
        ok = pins == (0.8085, 0.7745, 0.6875, 0.689)
        print(f"wilson pins {pins} {'OK' if ok else 'MISMATCH'}")
        raise SystemExit(0 if ok else 1)
    if args.dry_run_spawn:
        for i in range(int((spec or {}).get("n", 0))):
            print(f"DRY: eval_drl_checkpoint.py --episode {i} "
                  f"--checkpoint {spec.get('checkpoint', '<ckpt>')} (P1-b wires live)")
        raise SystemExit(0)
    if args.mode == "gate":
        crit = (spec or {}).get("safety_criteria", "v2_real_events")
        rows = [leg_verdict(d, adj, crit) for d in _leg_dirs(args.legs_dir)]
        out = gate_adjudicate(rows, spec["gate_type"], oc.wilson_lb, crit)
    elif args.mode == "retention":
        crit = (spec or {}).get("safety_criteria", "v2_real_events")
        strata = {name: [leg_verdict(os.path.join(args.legs_dir, name, d), adj,
                                     crit)
                         for d in sorted(os.listdir(os.path.join(args.legs_dir, name)))]
                  for name in spec["strata"]}
        out = retention_adjudicate(strata, spec["phase"],
                                   spec.get("prior_strikes", 0), oc.wilson_lb)
    else:
        out = zeroshot_check(spec["ladder"], spec["anchors"],
                             spec.get("skipped", []), spec["candidate"],
                             spec.get("probe_k"), oc.wilson_lb)
    txt = json.dumps(out, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
