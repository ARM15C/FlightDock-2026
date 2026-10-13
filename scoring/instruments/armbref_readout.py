#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: read the two flown-but-unscored ARM-B reference ladders (BASELINE-025 at
#   sigma 0.25 and BASELINE-035 at sigma 0.35, 600 legs each, winds 0-8 m/s at 1 m/s) into the
#   readout their OWN SIGNED preregs registered BEFORE the legs flew -- BASELINE035_PREREG SS8
#   and BASELINE025_PREREG SS8-SS9 -- and, across the two, the dose-response table SS8 calls
#   "the first like-for-like sigma comparison at 1 m/s resolution".
#   MODE: BENCHMARK. DESCRIPTIVE ONLY (BASELINE025_PREREG SS9.6): ARM-B is a reference
#   description of one controller. It tests no hypothesis, confirms nothing, and enters no
#   alpha-graph node on its own account. Every bound still NAMES its node (SS9.5).
#   The success predicate is the project's ONE predicate, unchanged: dwell_of_record >= 3.0 s
#   AND truth peak <= 2.00 m, read from the instrument's OWN verdict, never re-derived from the
#   rounded column. Scoring is DELEGATED to curriculum_gate_runner via armbsig_readout; this
#   file adds only the ladder walk, the SS9.3 clustering layer, the CLEAN/ALL bracket and the
#   pairing. Nothing pinned is reimplemented, and every pinned instrument is DIGEST-VERIFIED.
#   A peak above 2.00 m is a safety-EXCLUSION-ZONE violation (T1), never a "collision".
#   NOT A TRUTH PATH: reads FINISHED leg telemetry offline. Imports nothing from src/drl; no
#   metric is computed inside the env; nothing it writes is read by an actor, critic, reward,
#   replay buffer, observation or policy target.
#   SPLIT, NEVER RAISED: this file crossed its HARD cap twice while being written, so the
#   REPORT (armbref_render.py) and the two registered COMPARISONS (armbref_predict.py) moved to
#   siblings, functions unchanged. DISCLOSED: the approved plan authorized five new files; this
#   is a four-module instrument plus its selfcheck, contract and unit test -- seven. The extra
#   two are cap-driven structural splits that move no logic, the remedy the plan itself named.
# PUBLIC API: PINNED ; verify_pins() ; first_tick_wind(leg_dir) ; guard_fires(leg_dir) ;
#   score_one(leg_dir, meta, adj) ; walk(legs_root, manifest, adj) -> (rows, missing) ;
#   rung_stats(rows, sigma, wilson_lb, expected_n, missing) ; main(argv=None) -> int
# SIDE-EFFECTS: reads leg dirs and manifests READ-ONLY; writes --out JSON and a sibling
#   _TABLE.md. No env mutation, no subprocess, no network, no SITL, no artifact modified.
# SIZE: soft 300 / hard 340 code-LoC; function cap 45.
# COVERAGE: armbref_selfcheck.py L5-L12 + tests/unit/test_armbref_readout.py. EXECUTED.
# =============================================================================
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FD = "/home/ubuntu/FlightDock"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(FD, "scripts"))
sys.path.insert(0, os.path.join(FD, "analysis/armbsig_20260824T0030Z"))
import armbref_stats as S                          # noqa: E402
import armbsig_readout as AR                       # noqa: E402
import armbref_render as R                        # noqa: E402
import armbref_predict as P                       # noqa: E402

#: Every instrument this readout leans on, pinned by sha256. A pinned instrument that has
#: MOVED is a different instrument, and the readout refuses rc 5 rather than score with it.
PINNED = {
    "analysis/armbsig_20260824T0030Z/armbsig_readout.py": "761c20fb5392bdd4",  # re-pinned 2026-08-31: T5 commit 3ae02b86e (equivalence smoke 70/0 drift)
    "scripts/gate_safety_events.py": "52b07e0217429890",
    "scripts/curriculum_gate_runner.py": "8a0fe3449c5bd286",
    "analysis/benchmark_n100_20260718T190000Z/extractors/safety_supp_lib.py": "6bc12d573c1ea5a3",
    "analysis/benchmark_n100_20260718T190000Z/runner/benchmark_adjudicate.py": "998bf48c2335bfa5",
    "analysis/c006_campaign_20260728T110000Z/p13b_fault_rca/faultclass_census.py":
        "7f92be0f7d978b8b",
}
FAULTCLASS = os.path.join(FD, "analysis/c006_campaign_20260728T110000Z/p13b_fault_rca/"
                              "faultclass_census.py")
H1_BAR_M, DWELL_BAR_S, ATTRITION_BAR = 2.00, 3.0, 0.145
BOOT, BOOT_SEED = 10000, 0
LABELS = {"channel": "truth_decisive_fused_reported", "venue": "as_flown",
          "frame": "boom_tip_truth_decisive_fused_reported"}


def verify_pins():
    """Refuse rather than score with an instrument that has moved (failure mode (i))."""
    bad = [p for p, d in PINNED.items() if not S.digest(os.path.join(FD, p)).startswith(d)]
    return bad


def first_tick_wind(leg_dir):
    """Wind from the LEG'S OWN FIRST TICK -- never the rung label, never config.py. Rows are
    consumed in FILE ORDER and never sorted or deduped (data hazard 1)."""
    for p in sorted(glob.glob(os.path.join(leg_dir, "env*", "paired_channels_ep*.csv"))):
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                v = row.get("resolved_wind_max_ms")
                return None if v in (None, "") else float(v)
    return None


def metrics_wind_sigma(leg_dir):
    """The leg's own metrics row: wind and COMMANDED sigma, for cross-check only."""
    for p in sorted(glob.glob(os.path.join(leg_dir, "metrics_*.csv"))):
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                w = row.get("wind_speed_ms")
                return (None if w in (None, "") else float(w)), row.get("turbulence_sigma_ms")
    return None, None


def guard_fires(leg_dir):
    """Per-type guard interventions from the leg's own transition log. Every `reason=guard_*`
    token is counted GENERICALLY so a type this code has never seen is never silently dropped;
    emergency breakaways are counted separately because a guard fire and an abort differ."""
    out, aborts = {}, 0
    for p in sorted(glob.glob(os.path.join(leg_dir, "episode_*", "TRANSITION_SNAPSHOT.log"))):
        with open(p, errors="replace") as fh:
            for ln in fh:
                if "new=EMERGENCY_BREAKAWAY" in ln:
                    aborts += 1
                i = ln.find("reason=guard_")
                if i >= 0:
                    t = ln[i + 7:].split()[0]
                    out[t] = out.get(t, 0) + 1
    return {"by_type": out, "total": sum(out.values()), "emergency_breakaways": aborts}


def _fault_label(leg_dir):
    try:
        c = S._load(FAULTCLASS, "faultclass_census").classify_leg(leg_dir)
        return c.get("label") or c.get("class") or "UNREADABLE"
    except Exception as exc:                                       # noqa: BLE001
        return "CLASSIFIER_ERROR:%s" % type(exc).__name__


def _wind_check(leg_dir, meta):
    """Cross-check the three wind sources. Disagreement REFUSES the leg (failure mode (g))."""
    tick, (mw, msig) = first_tick_wind(leg_dir), metrics_wind_sigma(leg_dir)
    want = meta["wind_ms"]
    bad = [s for s, v in (("first_tick", tick), ("metrics", mw))
           if v is not None and abs(v - want) > 1e-6]
    return {"manifest_wind_ms": want, "first_tick_wind_ms": tick, "metrics_wind_ms": mw,
            "metrics_commanded_sigma": msig, "sources_disagree": bad,
            "wind_source_of_record": "leg_first_tick_resolved_wind_max_ms"}


def score_one(leg_dir, meta, adj):
    """One leg -> the pinned verdict row plus the ladder's own provenance fields."""
    row = AR.score_leg(leg_dir, {"cell": meta["cell"], "sigma": meta["sigma_ms"],
                                 "wind_label": meta["label"], "wind_ms": meta["wind_ms"],
                                 "seed": meta["seed"], "trajectory": meta["traj"]}, adj)
    row["wind_check"] = _wind_check(leg_dir, meta)
    row["sigma_gate_state"] = _gate_state(leg_dir)
    row["fault_class"] = _fault_label(leg_dir)
    row["clean"] = row["fault_class"] == "CLEAN"
    row["guards"] = guard_fires(leg_dir)
    row["exceeds_2m"] = (None if row.get("truth_peak_up_m") is None
                         else bool(row["truth_peak_up_m"] > H1_BAR_M))
    return row


def walk(legs_root, manifest, adj):
    """Walk the MANIFEST (the one writer), not the directory: a leg the manifest asked for and
    the disk does not have is MISSING and stays in the denominator -- absence is not zero."""
    rows, missing = [], []
    for leg in manifest["legs"]:
        d = os.path.join(legs_root, leg["cell"],
                         "%s_%d_%s" % (leg["label"], leg["seed"], leg["traj"]))
        if not os.path.isdir(d):
            missing.append({"cell": leg["cell"], "seed": leg["seed"], "trajectory": leg["traj"]})
            continue
        rows.append(score_one(d, leg, adj))
    return rows, missing


def _gate_state(leg_dir):
    """The realized-sigma gate's OWN state word. The calm tier writes CALM_CONFIRMED and no
    `sigma_ned` at all, so a reader that keys on sigma_ned alone sees the S0 control rung as
    UNSAMPLED when it is in fact positively witnessed (prereg SS6.1)."""
    p = os.path.join(leg_dir, "sigma_gate.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get("state")
    except Exception:                                          # noqa: BLE001
        return "UNREADABLE"


def _calm_block(rows):
    """SS6.1 / SS13 rule 3: the still-air rung is gated by the CALM_CONFIRMED POSITIVE
    assertion, never by a band around sigma 0 (which is degenerate -- band(0.00) = 0.0000).
    Unwitnessed, the rung is 'asserted but UNWITNESSED' and is NOT used as a control."""
    st = _count(rows, "sigma_gate_state")
    ok = st.get("CALM_CONFIRMED", 0)
    return {"states": st, "n_calm_confirmed": ok, "n_legs": len(rows),
            "s0_control_usable": bool(rows) and ok == len(rows),
            "rule": "prereg SS6.1: --paired-csv on S0 and no other rung; an S0 leg that DELIVERS "
                    "wind REFUSES; absent/unreadable stream is UNSAMPLED",
            "if_unwitnessed": "asserted but UNWITNESSED -- not used as a control (SS13 rule 3)"}


def _sigma_na(half, rows):
    """At sigma 0 the band comparison is DEGENERATE, so the realized-sigma limb does not apply.
    The reasons it produced are REMOVED and RECORDED -- never silently dropped."""
    dropped = [r for r in half["inconclusive_reasons"] if "sigma" in r]
    half["inconclusive_reasons"] = [r for r in half["inconclusive_reasons"] if "sigma" not in r]
    half["INCONCLUSIVE"] = bool(half["inconclusive_reasons"])
    half["sigma_limb"] = {"applicable": False, "reason": "band(0.00) = 0.0000 is degenerate; the "
                                                        "still-air rung is judged by SS6.1's "
                                                        "CALM_CONFIRMED assertion instead",
                          "reasons_withdrawn": dropped, "calm": _calm_block(rows)}
    return half


def _by_seed(rows, key):
    out = {}
    for r in rows:
        v = r.get(key)
        if v is not None:
            out.setdefault(r["seed"], []).append(float(v))
    return list(out.values())


def _clustered(rows):
    """The SS9.3 layer: realised design effect + seed-clustered intervals, reported BESIDE the
    leg-level figure so neither extreme is assumed."""
    graded = [r for r in rows if r.get("gradable")]
    succ = _by_seed([dict(r, s=float(r["passed"])) for r in graded], "s")
    n, k = len(graded), sum(1 for r in graded if r["passed"])
    de = S.design_effect(succ)
    out = {"n_legs": n, "k": k, "n_seeds": len(succ), "design_effect": de,
           "wilson_leg_level_two_sided_95": S.wilson2(k, n),
           "wilson_rao_scott_at_n_eff": (S.rao_scott_wilson(k, n, de["deff"]) if de["deff"]
                                         else None),
           "rate_ci_seed_cluster_boot": S.cluster_boot_stat(succ, "mean", BOOT, 0.95, BOOT_SEED)}
    for name, key, stat in (("dwell_median_s", "intersection_v3_s", "median"),
                            ("truth_peak_p95_m", "truth_peak_up_m", "p95"),
                            ("fused_peak_p95_m", "fused_peak_up_m", "p95")):
        cl = _by_seed(graded, key)
        out[name + "_ci_seed_cluster_boot"] = (S.cluster_boot_stat(cl, stat, BOOT, 0.95,
                                                                   BOOT_SEED) if cl else None)
        out[name + "_design_effect"] = S.design_effect(cl) if cl else None
    return out


def _bracket(rows, sigma, wilson_lb, expected_n, missing):
    """CLEAN (primary) and ALL (bound), each with its own n. Never one alone."""
    clean = [r for r in rows if r.get("clean")]
    w = rows[0]["wind_ms"] if rows else None
    halves = {"ALL": AR.cell_stats(rows, sigma, wilson_lb, expected_n, missing),
              "CLEAN": AR.cell_stats(clean, sigma, wilson_lb, len(clean), 0)}
    for h in halves.values():
        h.update({"wind_ms": w, "sigma": sigma})
        if sigma == 0.0:
            _sigma_na(h, rows if h is halves["ALL"] else clean)
    return {"ALL": halves["ALL"], "CLEAN": halves["CLEAN"],
            "fault_incidence": {"n_legs": len(rows), "n_clean": len(clean),
                                "n_faulted": len(rows) - len(clean),
                                "labels": _count(rows, "fault_class")},
            "screen_note": "the CLEAN screen is outcome-correlated and inflates whatever rate "
                           "it is pointed at; the pair brackets the truth"}


def _count(rows, key):
    out = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return out


def _hazards(rows):
    """Truth-side-decisive abort accounting: a hazard is defined from TRUTH GEOMETRY ALONE and
    is computed independently of any guard signal, because a guard must never be evaluated by a
    predicate derived from itself. The +-1.0 s window the SAP asks for needs tick-level abort
    times this corpus does not carry, so the window is NOT MEASURED and the leg-level counts
    below are labelled as such rather than presented as the windowed statistic."""
    haz = [r for r in rows if r.get("exceeds_2m") or r.get("real_overtake")]
    ab = [r for r in rows if r["guards"]["emergency_breakaways"] > 0]
    hs, abs_ = {id(r) for r in haz}, {id(r) for r in ab}
    return {"n_true_hazard_legs": len(haz), "n_legs_with_abort": len(ab),
            "hazard_without_abort_legs": sum(1 for r in haz if id(r) not in abs_),
            "abort_without_hazard_legs_PHANTOM": sum(1 for r in ab if id(r) not in hs),
            "phantom_note": "PHANTOM is not 'unnecessary' -- the prevented-event counterfactual "
                            "is unrecoverable",
            "window_s": None, "windowed_rates": "NOT MEASURED (no tick-level abort times)",
            "resolution": "leg_level_not_time_windowed",
            "predicate": "truth_peak_up_m > 2.00 OR real_overtake; guard-signal independent"}


def rung_stats(rows, sigma, wilson_lb, expected_n=None, missing=0):
    st = _bracket(rows, sigma, wilson_lb, expected_n, missing)
    graded = [r for r in rows if r.get("gradable")]
    exc = [r for r in graded if r.get("exceeds_2m")]
    st.update({
        "clustered": _clustered(rows),
        "h1_exceedance": {"k_exceeding": len(exc), "n_graded": len(graded), "bar_m": H1_BAR_M,
                          "cp_upper_one_sided_95": (round(S.cp_upper(len(exc), len(graded)), 6)
                                                    if graded else None),
                          "method": "clopper_pearson_exact_upper", "alpha_node":
                              "descriptive_no_alpha_0.05"},
        "guards": _guard_roll(rows), "aborts": _hazards(rows),
        "wind_disagreements": [r["leg"] for r in rows if r["wind_check"]["sources_disagree"]],
        "labels": dict(LABELS, n_legs=len(rows), sigma=sigma),
    })
    return st


def _guard_roll(rows):
    by, tot, ab = {}, 0, 0
    for r in rows:
        for t, c in r["guards"]["by_type"].items():
            by[t] = by.get(t, 0) + c
        tot += r["guards"]["total"]
        ab += r["guards"]["emergency_breakaways"]
    return {"by_type": by, "total": tot, "emergency_breakaways": ab,
            "note": "per-type, never a single total"}


def _pins_block():
    return {p: {"declared_prefix": d, "measured": S.digest(os.path.join(FD, p))[:16]}
            for p, d in PINNED.items()}


def _cells(rows, manifest, missing, oc):
    by, exp, miss = {}, {}, {}
    for r in rows:
        by.setdefault(r["cell"], []).append(r)
    for leg in manifest["legs"]:
        exp[leg["cell"]] = exp.get(leg["cell"], 0) + 1
    for m in missing:
        miss[m["cell"]] = miss.get(m["cell"], 0) + 1
    return {c: rung_stats(rs, rs[0]["sigma"], oc.wilson_lb, exp.get(c), miss.get(c, 0))
            for c, rs in by.items()}


def _emit(js, out):
    txt = R.render(js)
    if out:
        json.dump(js, open(out, "w"), indent=1, default=str)
        open(os.path.splitext(out)[0] + "_TABLE.md", "w").write(txt)
    sys.stdout.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description="ARM-B reference ladder readout (reads only)")
    ap.add_argument("--legs-root")
    ap.add_argument("--manifest")
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--out")
    ap.add_argument("--dose-response", nargs=2, metavar=("READOUT_035", "READOUT_025"))
    a = ap.parse_args(argv)
    bad = verify_pins()
    if bad:
        print("READOUT REFUSED: pinned instrument(s) MOVED: %s" % bad, file=sys.stderr)
        return 5
    if a.dose_response:
        j35, j25 = (json.load(open(p)) for p in a.dose_response)
        dr = P.dose_response(j35, j25)
        if a.out:
            json.dump(dr, open(a.out, "w"), indent=1, default=str)
        print(json.dumps(dr["per_wind"], indent=1, default=str))
        return 0
    if not (a.legs_root and a.manifest and a.sigma is not None):
        ap.error("--legs-root, --manifest and --sigma are required unless --dose-response")
    adj, oc = AR.cgr._frozen()
    man = json.load(open(a.manifest))
    rows, missing = walk(a.legs_root, man, adj)
    cells = _cells(rows, man, missing, oc)
    js = {"dispatch": "ARMBREF-%s" % man["dispatch"], "mode": "BENCHMARK",
          "instrument": AR.CRIT, "sigma": a.sigma, "n_legs_scored": len(rows),
          "expected_total": man["n_legs"], "manifest_sha_source": man.get("source_table_sha256"),
          "pinned_instruments": _pins_block(), "cells": cells,
          "predictions": P.predictions(cells, a.sigma), "dispositions": R.dispositions(rows, missing),
          "caveats": R.CAVEATS, "missing_legs": missing, "legs": rows}
    _emit(js, a.out)
    dis = [r["leg"] for r in rows if r["wind_check"]["sources_disagree"]]
    if dis:
        print("READOUT REFUSED: %d leg(s) whose wind sources DISAGREE (first: %s)"
              % (len(dis), dis[0]), file=sys.stderr)
        return 6
    if len(rows) != man["n_legs"] or missing:
        print("READOUT REFUSED: %d of %d manifest legs MISSING -- a corpus short of its "
              "manifest is INCONCLUSIVE, never a verdict" % (len(missing), man["n_legs"]),
              file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
