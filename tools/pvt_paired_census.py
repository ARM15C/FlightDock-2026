#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: T-344 perception-vs-truth census in the PAIRED SAME-TICK
#   frame, under provenance discipline. Reports the G3 estimand (signed median
#   + |p50|/|p95|/|p99|) for ctrl (FUSED), env (ENV_GPS_DIRECT) and the
#   truth-free ctrl-minus-env contrast, on ALL THREE axes always, in three
#   views (ctrl-fresh filtered = PRIMARY, unfiltered, align-mode-0
#   sensitivity), with denominators asserted rather than inferred. Also
#   contrasts T-93 CLEAN vs FAULT using labels RECOMPUTED by the pinned T-315
#   classifier -- banked labels are read only to print a disagreement cross-tab.
#
#   *** THIS IS NOT ESTIMATOR B. *** The canonical P-vs-T estimator by operator
#   directive (2026-06-12) is tools/pvt_disparity_census.py, which requires
#   paired_tick_log_env0.jsonl. That artifact is absent from every panel in
#   scope (newest anywhere 2026-07-04), so B is structurally unrunnable here
#   and the two CANNOT be cross-calibrated from banked data. This tool uses the
#   labelled NON-CANONICAL paired same-tick frame (precedent T-194, T-126).
#   No output of this tool may be differenced against an Estimator-B or G6
#   number; side-by-side quoting under a banner only.
#
#   Column semantics resolved at the WRITER (src/drl/env_formation_sitl.py):
#     delta_ctrl_raw_<ax>_m = ctrl - raw (:5699-5702) = FUSED - TRUTH
#     delta_env_raw_<ax>_m = env - raw (:5695-5698) = GPS - TRUTH
#     raw_* = RAW_TRUTH_DIRECT (fusion bypassed); env_* = ENV_GPS_DIRECT
#     (NOT fusion output); ctrl_* = controller _geom_cache, ctrl_source=FUSED.
#   Positive delta = perceived reads LARGER than truth.
# PUBLIC API: load_leg(leg_dir), denominators(df), channel_stats(v, min_n),
#   leg_census(leg_dir, min_n), panel_census(leg_dirs, label, expected_n,
#   min_n), panel_medians(panel, view), selfcheck(), main(argv=None).
#   Statistics (contrasts/Holm/Hodges-Lehmann) live in tools/pvt_paired_stats.py,
#   split out when this file crossed its soft cap -- the cap was NOT raised.
# SIDE-EFFECTS: reads leg artifacts READ-ONLY; writes exactly one JSON at
#   --out-json plus stdout/stderr. --selfcheck writes only inside a
#   TemporaryDirectory it removes. No env mutation, no subprocess, no network,
#   no SITL, no artifact ever modified.
# CAPS: file soft 340 / hard 420 code-LoC; function LoC cap 45.
# COVERAGE: --selfcheck, 9 groups S1-S9, EVERY group with a disagree arm;
#   fail-closed -- no report is emitted unless selfcheck passes.
# =============================================================================
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pvt_paired_stats import contrasts, selfcheck_stats

AXES = ("behind", "below", "cross")
CHANNELS = ("ctrl", "env")
ESTIMATOR = "PAIRED_SAME_TICK_NONCANONICAL"
MIN_N_DEFAULT = 200
KNOWN_ALIGN_MODES = (-1, 0, 1, 2, 3, 4, 5)
REPO = pathlib.Path(__file__).resolve().parents[1]
CLASSIFIER = REPO / ("analysis/c006_campaign_20260728T110000Z/p13b_fault_rca/"
                     "faultclass_census.py")
CLASSIFIER_PIN = ("7f92be0f7d978b8b6669aa6931898fc6e038e3da1df8bbf3c65b92e6f"
                  "867f30e")
G3_ORACLE = REPO / "analysis/bwcol_20260808T2100Z/bw_census.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pinned_classifier():
    """Import the T-315 classifier under its A-7 signed sha pin. Mismatch is
    rc=5 refusal -- NEVER a silent fallback to the retired banked labels."""
    if not CLASSIFIER.is_file():
        raise SystemExit(5)
    got = hashlib.sha256(CLASSIFIER.read_bytes()).hexdigest()
    if got != CLASSIFIER_PIN:
        sys.stderr.write(f"CLASSIFIER PIN DRIFT: {got}\n")
        raise SystemExit(5)
    return _load_module(CLASSIFIER, "faultclass_census")


def _resolve_paired(leg_dir):
    """Same ordered search as the pinned classifier -- hardcoded path FIRST so
    a leg readable there resolves identically (T-131)."""
    d = str(leg_dir)
    for cand in ([os.path.join(d, "paired", "env0",
                               "paired_channels_ep0001.csv")]
                 + sorted(glob.glob(os.path.join(
                     d, "env*", "paired_channels_ep*.csv")))
                 + sorted(glob.glob(os.path.join(
                     d, "*", "env*", "paired_channels_ep*.csv")))):
        if os.path.isfile(cand):
            return cand
    return None


def load_leg(leg_dir):
    """-> (DataFrame, provenance). Never drops rows; marks them."""
    d = pathlib.Path(leg_dir)
    prov = {"leg": d.name, "paired_csv": None,
            "estimator_b_available": bool(
                glob.glob(str(d / "**" / "paired_tick_log_env*.jsonl"),
                          recursive=True))}
    if prov["estimator_b_available"]:
        sys.stderr.write(
            f"[ESTIMATOR-B AVAILABLE] {d.name}: this leg carries a tick log; "
            "the CANONICAL estimator tools/pvt_disparity_census.py applies "
            "and should be preferred for it.\n")
    p = _resolve_paired(d)
    if p is None:
        return None, prov
    files = sorted(glob.glob(os.path.join(os.path.dirname(p),
                                          "paired_channels_ep*.csv")))
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)
    prov["paired_csv"] = os.path.relpath(p, REPO)
    prov["n_files"] = len(files)
    num = ([f"delta_{c}_raw_{a}_m" for c in CHANNELS for a in AXES]
           + [f"raw_{a}_m" for a in AXES]
           + ["ctrl_cache_is_stale", "ctrl_behind_m", "raw_align_mode",
              "env_align_mode"])
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, prov


def _mode_hist(df, col):
    """Histogram that CANNOT hide an unknown mode: anything outside the known
    set lands in UNKNOWN, and the total is asserted against the row count."""
    out = {}
    if col not in df.columns:
        return {"ABSENT": int(len(df))}
    for v, n in df[col].value_counts(dropna=False).items():
        key = (str(int(v)) if pd.notna(v) and int(v) in KNOWN_ALIGN_MODES
               else ("NaN" if pd.isna(v) else f"UNKNOWN_{v}"))
        out[key] = out.get(key, 0) + int(n)
    assert sum(out.values()) == len(df), "align-mode histogram lost rows"
    return out


def denominators(df):
    """Asserted denominators. UNSAMPLED is never CLEAN and never a zero."""
    stale = df.get("ctrl_cache_is_stale")
    ctrl_null = int(df["ctrl_behind_m"].isna().sum()) \
        if "ctrl_behind_m" in df.columns else None
    out = {"rows_total": int(len(df)),
           "rows_ctrl_stale": int((stale == 1).sum()) if stale is not None
           else None,
           "rows_ctrl_fresh": int((stale == 0).sum()) if stale is not None
           else None,
           "rows_ctrl_null": ctrl_null,
           "raw_align_mode": _mode_hist(df, "raw_align_mode"),
           "env_align_mode": _mode_hist(df, "env_align_mode")}
    for c in ("ctrl_source", "env_geom_source", "raw_geom_source"):
        out[c] = ({str(k): int(v) for k, v in
                   df[c].value_counts(dropna=False).items()}
                  if c in df.columns else "ABSENT")
    # F1-alt witness: truth/GPS channels must NOT be fusion output.
    out["f1alt_ok"] = not any(
        "FUSED" in str(k) for c in ("env_geom_source", "raw_geom_source")
        if c in df.columns for k in df[c].dropna().unique())
    return out


def channel_stats(v, min_n=MIN_N_DEFAULT):
    """G3 estimand. Returns None (-> UNSAMPLED) below min_n -- never zeros."""
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    if v.size < min_n:
        return None
    a = np.abs(v)
    return {"n": int(v.size),
            "signed_median": round(float(np.median(v)), 4),
            "abs_p50": round(float(np.percentile(a, 50)), 4),
            "abs_p95": round(float(np.percentile(a, 95)), 4),
            "abs_p99": round(float(np.percentile(a, 99)), 4),
            "rms": round(float(np.sqrt(np.mean(v ** 2))), 4),
            "estimator": ESTIMATOR}


def _view_masks(df):
    fresh = (df["ctrl_cache_is_stale"] == 0) & df["ctrl_behind_m"].notna() \
        if "ctrl_cache_is_stale" in df.columns else pd.Series(True,
                                                              index=df.index)
    m0 = (df["raw_align_mode"] == 0) if "raw_align_mode" in df.columns \
        else pd.Series(True, index=df.index)
    return {"filtered": fresh, "unfiltered": pd.Series(True, index=df.index),
            "mode0_sensitivity": m0}


def _blocks(df, mask, min_n, disp=None):
    """One view's stats. ctrl/env truth-referenced + truth-free ctrl-minus-env.
    Behind keys are suffixed _CONFOUNDED_descriptive by construction."""
    sub, out = df[mask], {}
    for ch in CHANNELS:
        for ax in AXES:
            col = f"delta_{ch}_raw_{ax}_m"
            if col not in sub.columns:
                continue
            key = f"{ch}_{ax}" + ("_CONFOUNDED_descriptive"
                                  if ax == "behind" else "")
            out[key] = channel_stats(sub[col], min_n)
    for ax in AXES:
        c, e = f"delta_ctrl_raw_{ax}_m", f"delta_env_raw_{ax}_m"
        if c in sub.columns and e in sub.columns:
            key = f"ctrl_minus_env_{ax}" + ("_CONFOUNDED_descriptive"
                                            if ax == "behind" else "")
            out[key] = channel_stats(sub[c] - sub[e], min_n)
    if disp is not None and "raw_behind_m" in sub.columns:
        adj = sub["delta_ctrl_raw_behind_m"] + disp
        out["ctrl_behind_displacement_adjusted"] = channel_stats(adj, min_n)
    return out


def leg_census(leg_dir, min_n=MIN_N_DEFAULT):
    df, prov = load_leg(leg_dir)
    if df is None or len(df) == 0:
        return {"leg": prov["leg"], "status": "UNSAMPLED",
                "why": "no_paired_csv", "provenance": prov}
    fc = _pinned_classifier()
    cls = fc.classify_leg(str(leg_dir))
    disp = cls.get("disp_median_m")
    row = {"leg": prov["leg"], "status": "OK", "provenance": prov,
           "class_recomputed": cls.get("label"),
           "disp_median_m": disp, "ticks_est": cls.get("ticks_est"),
           "denominators": denominators(df), "views": {}}
    masks = _view_masks(df)
    for name, m in masks.items():
        row["views"][name] = _blocks(df, m, min_n,
                                     disp if name == "filtered" else None)
    return row


def panel_census(leg_dirs, label, expected_n=None, min_n=MIN_N_DEFAULT):
    """Fail-closed on count mismatch: flagged, never silently truncated."""
    rows = [leg_census(d, min_n) for d in leg_dirs]
    ok = [r for r in rows if r["status"] == "OK"]
    out = {"panel": label, "estimator": ESTIMATOR,
           "legs_found": len(rows), "legs_expected": expected_n,
           "legs_ok": len(ok),
           "legs_excluded": [{"leg": r["leg"], "why": r.get("why")}
                             for r in rows if r["status"] != "OK"],
           "count_ok": (expected_n is None or len(rows) == expected_n),
           "class_counts": {}, "rows": rows}
    for r in rows:
        k = r.get("class_recomputed") or "UNSAMPLED"
        out["class_counts"][k] = out["class_counts"].get(k, 0) + 1
    return out


def panel_medians(panel, view="filtered"):
    """Median-over-legs of each block field. Skips UNSAMPLED (None) blocks."""
    agg = {}
    for r in panel["rows"]:
        for key, blk in r.get("views", {}).get(view, {}).items():
            if blk:
                agg.setdefault(key, []).append(blk)
    return {k: {f: round(float(np.median([b[f] for b in v])), 4)
                for f in ("signed_median", "abs_p50", "abs_p95", "abs_p99")}
            | {"n_legs": len(v), "estimator": ESTIMATOR}
            for k, v in agg.items()}


def _sc_equivalence(g3):
    """S1 estimand equivalence with the banked-D1 oracle + its disagree arm."""
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({f"delta_{c}_raw_{a}_m": rng.normal(0.1, 0.5, n)
                       for c in CHANNELS for a in AXES})
    df["ctrl_cache_is_stale"] = 0
    df["ctrl_behind_m"] = rng.normal(5, 1, n)
    ref, mine = g3.g3_table(df), _blocks(df, _view_masks(df)["unfiltered"], 10)
    for ax in AXES:
        k = f"ctrl_{ax}" + ("_CONFOUNDED_descriptive" if ax == "behind" else "")
        for f in ("signed_median", "abs_p50", "abs_p95", "abs_p99"):
            assert mine[k][f] == ref[f"ctrl_{ax}"][f], f"S1 mismatch {k}.{f}"
    df2 = df.copy()
    df2.loc[0, "delta_ctrl_raw_below_m"] = 99.0
    m2 = _blocks(df2, _view_masks(df2)["unfiltered"], 10)
    assert m2["ctrl_below"]["abs_p99"] != ref["ctrl_below"]["abs_p99"], \
        "S1 disagree arm failed: comparison is vacuous"
    return "S1 estimand-equivalence PASS (+disagree)"


def _sc_stale():
    """S2a/S2b filter moves the answer in BOTH directions; S3 denominators."""
    n = 600
    base = np.concatenate([np.full(n // 2, 0.1), np.full(n // 2, 0.1)])
    df = pd.DataFrame({f"delta_{c}_raw_{a}_m": base.copy()
                       for c in CHANNELS for a in AXES})
    df["ctrl_behind_m"] = 5.0
    df["ctrl_cache_is_stale"] = np.concatenate(
        [np.zeros(n // 2), np.ones(n // 2)])
    df.loc[df.index[n // 2:], "delta_ctrl_raw_below_m"] = 9.0
    f = _blocks(df, _view_masks(df)["filtered"], 10)
    u = _blocks(df, _view_masks(df)["unfiltered"], 10)
    assert f["ctrl_below"]["abs_p95"] < u["ctrl_below"]["abs_p95"], "S2a"
    assert f["env_below"]["abs_p95"] == u["env_below"]["abs_p95"], \
        "S2b filter is mis-scoped: it moved the env channel"
    # Disagree arm: outliers live ONLY in the fresh half and are frequent
    # enough to clear p95 there (20/300) but not in the pooled set (20/600).
    df2 = df.copy()
    df2["delta_ctrl_raw_below_m"] = 0.1
    df2.loc[df2.index[:20], "delta_ctrl_raw_below_m"] = 9.0
    f2 = _blocks(df2, _view_masks(df2)["filtered"], 10)
    u2 = _blocks(df2, _view_masks(df2)["unfiltered"], 10)
    assert f2["ctrl_below"]["abs_p95"] > u2["ctrl_below"]["abs_p95"], \
        "S2 disagree arm: filtered must be able to come out HIGHER"
    df3 = df.copy()
    df3["ctrl_cache_is_stale"] = 1
    assert _blocks(df3, _view_masks(df3)["filtered"], 10)["ctrl_below"] is None,\
        "S3: 100% stale must be UNSAMPLED, never a number"
    return "S2a/S2b/S3 stale-filter + denominators PASS (both directions)"


def _sc_truthfree():
    """S7: ctrl_minus_env is invariant to a truth displacement; truth-
    referenced stats shift by exactly that displacement."""
    n, D = 500, 1.25
    rng = np.random.default_rng(11)
    ctrl = rng.normal(0.2, 0.3, n)
    env = rng.normal(0.0, 0.1, n)
    df = pd.DataFrame({"delta_ctrl_raw_behind_m": ctrl,
                       "delta_env_raw_behind_m": env,
                       "ctrl_behind_m": 5.0, "ctrl_cache_is_stale": 0})
    b1 = _blocks(df, _view_masks(df)["unfiltered"], 10)
    df2 = df.copy()          # truth shifts by -D => both deltas shift by +D
    df2["delta_ctrl_raw_behind_m"] += D
    df2["delta_env_raw_behind_m"] += D
    b2 = _blocks(df2, _view_masks(df2)["unfiltered"], 10)
    k, t = "ctrl_behind_CONFOUNDED_descriptive", \
        "ctrl_minus_env_behind_CONFOUNDED_descriptive"
    assert abs((b2[k]["signed_median"] - b1[k]["signed_median"]) - D) < 1e-6, \
        "S7: truth-referenced stat did not shift by D"
    assert b2[t]["signed_median"] == b1[t]["signed_median"], \
        "S7: ctrl_minus_env is NOT truth-free"
    return "S7 truth-free invariance PASS (+shift arm)"


def _sc_guards():
    """S6 classifier pin; S8 Estimator-B guard fires; S9 unknown align mode."""
    fc = _pinned_classifier()
    assert fc.label_for(1.0) == "FAULT" and fc.label_for(0.0) == "CLEAN"
    assert fc.label_for(0.35) == "BORDERLINE"
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td) / "leg"
        (d / "env0").mkdir(parents=True)
        _, prov = load_leg(d)
        assert prov["estimator_b_available"] is False, "S8 false-positive"
        (d / "env0" / "paired_tick_log_env0.jsonl").write_text("{}\n")
        _, prov2 = load_leg(d)
        assert prov2["estimator_b_available"] is True, \
            "S8: Estimator-B guard CANNOT fire"
    df = pd.DataFrame({"raw_align_mode": [0, 0, 5, 4, 7, 7]})
    h = _mode_hist(df, "raw_align_mode")
    assert h.get("UNKNOWN_7") == 2 and h.get("5") == 1 and h.get("4") == 1, \
        "S9: unknown align mode was dropped"
    return "S6/S8/S9 pin + estimator-B guard + align-mode PASS"


def selfcheck():
    g3 = _load_module(G3_ORACLE, "bw_census")
    for fn in (lambda: _sc_equivalence(g3), _sc_stale, _sc_truthfree,
               selfcheck_stats, _sc_guards):
        print("  " + fn())
    print("SELFCHECK PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", action="append", default=[],
                    help="NAME=GLOB[:expected_n] (repeatable)")
    ap.add_argument("--out-json")
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck()
    selfcheck()                      # fail-closed: no report without it
    out = {"estimator": ESTIMATOR, "classifier_pin": CLASSIFIER_PIN,
           "not_estimator_b": True, "panels": {}}
    for spec in a.panel:
        name, _, rest = spec.partition("=")
        pat, _, exp = rest.rpartition(":")
        if not pat:
            pat, exp = rest, ""
        legs = [d for d in sorted(glob.glob(pat)) if os.path.isdir(d)]
        p = panel_census(legs, name, int(exp) if exp.isdigit() else None,
                         a.min_n)
        p["panel_medians_filtered"] = panel_medians(p, "filtered")
        p["panel_medians_unfiltered"] = panel_medians(p, "unfiltered")
        if p["legs_ok"] >= 6:
            p["contrasts"] = contrasts(p, a.alpha)
        out["panels"][name] = p
        print(f"[{name}] legs={p['legs_found']} ok={p['legs_ok']} "
              f"expected={p['legs_expected']} count_ok={p['count_ok']} "
              f"classes={p['class_counts']}")
        if not p["count_ok"]:
            sys.stderr.write(f"PANEL COUNT MISMATCH: {name}\n")
    if a.out_json:
        pathlib.Path(a.out_json).write_text(json.dumps(out, indent=1))
        print(f"wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
