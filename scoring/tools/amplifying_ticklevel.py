#!/usr/bin/env python3
# amplifying_ticklevel.py -- ANALYSIS_PLAN_COMPARATIVE.md section 5, the TICK-LEVEL half: false and missed emergency
#   aborts with their arming condition (ABORTREC per-leg table, primary tolerance 1.0 s), stopping distance after an
#   intervention (vertical: rise after the fire, time to peak, time back under the bar; longitudinal: furthest
#   distance ahead of the no-pass line per ahead-run), recovery behavior after a breakaway against the contract
#   (ABORTREC recovery summary), and TIME TO DOCK read per tick from each captured leg's flight record: M2 reached =
#   the FIRST tick inside the M2 pre-contact box (thresholds read from the leg's own success_metrics.json, truth
#   channel, flight_state tick file; a 3.0 s hold is NOT required -- measured 2026-09-16, 77 of 720 captured legs
#   pass through M2 without holding 3 s, and 23 of them hold it only after a later breakaway). A captured leg that
#   never enters the box before M3b (5 of 720 measured, all within 2.2-3.1 m of M2) is measured from its CLOSEST
#   approach to M2 before M3b and flagged `m2_marker = closest approach`; the count is printed. M3b achieved = the
#   sim time at which the FROZEN dwell instrument's mask
#   (analysis/g1v3_dwell_20260710T214619Z/dwell_intersection.py, loaded BY PATH, unedited: paired_channels rows via its
#   own `load`, ctrl_m3b AND raw_m3b shifted by its cross-correlation lag, 0.2 s gap tolerance) first sustains a
#   run of 3.0 s. REPRODUCTION LIMB: the instrument's own longest run (mask_dwell) recomputed here must EQUAL the
#   board's intersection_v3_s to 3 dp on every captured leg, else REFUSE. Decides nothing. "stage" never "rung".
# PUBLIC API: aborts(pl); stopping(summaries); recovery(rs); ttd_leg(job); time_to_dock(board, roots, workers);
#   main(argv=None) -> int. CLI: --run-dir DIR [--out DIR] [--ticklevel-dir DIR] [--summaries-dir DIR] [--workers 8]
#   [--only TAG,...] [--skip-ttd] [--selftest]
# SIDE-EFFECTS: reads <ticklevel>/ABORT_PER_LEG.csv, RECOVERY_SUMMARY.csv, <summaries>/<tag>.jsonl.gz and captured
#   legs' env*/flight_state_ep*.csv.gz (read-only); writes ABORTS, STOPPING, RECOVERY, TIME_TO_DOCK (+ per-leg) under
#   <out> (default <run-dir>/comparative/). Spawns a process pool for the tick reads. No network.
# SIZE: soft 210 / hard 240 code-lines; function cap 30.
# COVERAGE: --selftest: a synthetic tick record with a planted M2 hold and M3b dwell yields the planted time to dock;
#   a tick record whose dwell disagrees with the board REFUSES; the abort table carries the arming denominators.
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hypothesis_verdicts as V  # noqa: E402
import amplifying_tables as T  # noqa: E402

VALS = ["min", "mean", "median", "p95", "max"]
TOL, HOLD = "1.0", 3.0
V3_DWELL = "/home/ubuntu/FlightDock/analysis/g1v3_dwell_20260710T214619Z/dwell_intersection.py"


def instrument():
    """The frozen dwell instrument, loaded by path exactly as the adjudicator of record loads it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dwell_intersection", V3_DWELL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_run(times, mask, hold, gap):
    """sim time at which a gap-tolerant run of `mask` (the instrument's own run rule) first reaches `hold` s."""
    rs = lt = None
    for t, ok in zip(times, mask):
        if ok:
            if rs is None or (lt is not None and t - lt > gap):
                rs = t
            lt = t
            if lt - rs >= hold:
                return float(t)
        else:
            rs = lt = None
    return None
STEM = {"A2P": "A2P", "CSV": "CSVO", "ARMB": "ARMB"}
OUT = ["."]                                                   # set by main(); where the pre-refusal per-leg file lands # the banked AMPSAFETY file stems (six corpora only)


def aborts(pl):
    """Per corpus: fires, armed truth events, false and missed aborts at the primary tolerance, with denominators."""
    g = pl.groupby("tag", sort=False)
    d = pd.DataFrame({"legs": g.size(), "altitude-guard fires": g.vertical_fires.sum(),
                      "truth excursions over 2.00 m (all)": g.vertical_truth_events_all.sum(),
                      "truth excursions while the guard was armed": g.vertical_truth_events.sum(),
                      "false aborts (fire, no truth excursion, tol %s s)" % TOL: g["vertical_false_tol" + TOL].sum(),
                      "missed aborts (armed excursion, no fire, tol %s s)" % TOL: g["vertical_missed_tol" + TOL].sum(),
                      "legs with a false abort": g["vertical_false_tol" + TOL].apply(lambda s: int((s > 0).sum())),
                      "legs with a missed abort": g["vertical_missed_tol" + TOL].apply(lambda s: int((s > 0).sum())),
                      "breakaways": g.n_breakaways.sum()}).reset_index().rename(columns={"tag": "corpus"})
    d["false-abort rate (% of fires)"] = (100 * d.iloc[:, 5] / d.iloc[:, 2].replace(0, np.nan)).round(1)
    d["missed-abort rate (% of armed excursions)"] = (100 * d.iloc[:, 6] / d.iloc[:, 4].replace(0, np.nan)).round(1)
    d["arming condition"] = "missed aborts counted only on excursions where the guard was eligible to fire (ABORTREC abort_eligible.py)"
    return d


def load_summaries(sdir, tags):
    rows = []
    for tag in tags:
        p = os.path.join(sdir, tag + ".jsonl.gz")
        if not os.path.exists(p):
            p = os.path.join(sdir, "%s_%s.jsonl.gz" % (STEM.get(tag[:-3], tag[:-3]), "0.25" if tag.endswith("025") else "0.35"))
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt") as fh:
            for line in fh:
                d = json.loads(line)
                d["corpus"] = tag
                rows.append(d)
    return rows


def stopping(summaries):
    """Vertical: per ceiling fire; longitudinal: per ahead-run (meters ahead of the no-pass line)."""
    vert, lon = [], []
    for d in summaries:
        for e in d.get("stopping_vertical") or []:
            vert.append({"corpus": d["corpus"], "leg": d["leg"], "rise after fire (m)": e.get("overshoot_m"),
                         "time to peak (s)": e.get("t_to_peak_s"), "time back under bar (s)": e.get("t_to_clear_bar_s")})
        for r in (d.get("raw") or {}).get("ahead_runs") or []:
            lon.append({"corpus": d["corpus"], "leg": d["leg"], "ahead of no-pass line (m)": r.get("max_ahead_m"), "duration (s)": r.get("dur_s")})
    agg = {"n": "size", **{v: (lambda s, v=v: np.percentile(s.dropna(), 95) if v == "p95" else getattr(s.dropna(), v)()) for v in VALS}}
    out = []
    for name, df, col in (("vertical, fire to END OF LEG (pinned instrument): rise after fire (m)", pd.DataFrame(vert), "rise after fire (m)"),
                          ("vertical, fire to END OF LEG (pinned instrument): time to peak (s)", pd.DataFrame(vert), "time to peak (s)"),
                          ("vertical, fire to END OF LEG (pinned instrument): time back under bar (s)", pd.DataFrame(vert), "time back under bar (s)"),
                          ("longitudinal: ahead of no-pass line (m)", pd.DataFrame(lon), "ahead of no-pass line (m)")):
        if len(df):
            x = df.groupby("corpus", sort=False)[col].agg(**agg).reset_index().assign(quantity=name)
            out.append(x)
    return pd.concat(out, ignore_index=True).round(3) if out else pd.DataFrame()


def stopping_bounded(tl, tags):
    """Vertical stopping INSIDE the breakaway, per ceiling-triggered breakaway, from RECOVERY_TRACE. The pinned
    safety_events.stopping_vertical searches from the fire to the END OF THE LEG, so a later excursion in the same
    leg inflates its rise and time to peak (measured 2026-09-18). Truth = raw_below_m; perception = ctrl_below_m."""
    ev = pd.read_csv(os.path.join(tl, "ABORT_EVENTS.csv"), low_memory=False)
    ev = ev[(ev.kind == "breakaway") & (ev.trigger_reason == "guard_altitude_ceiling")][["tag", "leg", "event_idx"]]
    rows = []
    for tag in tags:
        tr = pd.read_csv(os.path.join(tl, "RECOVERY_TRACE_%s.csv.gz" % tag), low_memory=False)
        tr = tr.merge(ev[ev.tag == tag][["leg", "event_idx"]], on=["leg", "event_idx"])
        tr = tr[tr.t_rel_s >= 0.0]
        for (leg, k), g in tr.groupby(["leg", "event_idx"], sort=False):
            post = g[g.in_breakaway.astype(str).str.lower().eq("true")]
            if post.empty:
                continue
            r = {"corpus": tag}
            for ch, col in (("truth", "raw_below_m"), ("perception", "ctrl_below_m")):
                up, t = -post[col].to_numpy(float), post.t_rel_s.to_numpy(float)
                j, under = int(np.nanargmax(up)), np.flatnonzero(up <= 2.0)
                r["rise after fire, %s (m)" % ch] = up[j] - up[0]
                r["time to peak, %s (s)" % ch] = t[j] - t[0]
                r["time back under bar, %s (s)" % ch] = (t[under[0]] - t[0]) if under.size else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = {"n": "size", **{v: (lambda s, v=v: np.percentile(s.dropna(), 95) if v == "p95" else getattr(s.dropna(), v)()) for v in VALS}}
    out = [df.groupby("corpus", sort=False)[c].agg(**agg).reset_index().assign(quantity="vertical, inside the breakaway: " + c)
           for c in df.columns if c != "corpus"]
    return pd.concat(out, ignore_index=True).round(3)


def recovery(rs):
    g = rs.groupby("tag", sort=False)
    return pd.DataFrame({"breakaway events": g.size(), "dwell in abort state, median (s)": g.dwell_s.median(),
                         "dwell meets contract minimum (%)": 100 * g.dwell_meets_min.mean(), "exit to INTERCEPT (%)": 100 * g.exit_is_intercept.mean(),
                         "commanded speed minus leader TAS, mean (m/s)": g.cmd_minus_lead_tas_mean.mean(), "contract speed delta (m/s)": g.contract_speed_delta_mps.first(),
                         "commanded altitude, mean (m MSL)": g.cmd_alt_mean_msl_m.mean(), "contract altitude (m MSL)": g.contract_alt_msl_m.first(),
                         "under the bar at exit (%)": 100 * g.under_bar_at_exit.mean(), "time back under bar after exit, median (s)": g.t_to_under_bar_after_exit_s.median(),
                         "time to 15 m behind after exit, median (s)": g.t_to_behind15_after_exit_s.median()}).reset_index().rename(columns={"tag": "corpus"}).round(2)



def ttd_leg(job):
    tag, leg, tick_path, sm_path, leg_dir, board_dwell = job
    th = json.load(open(sm_path))["thresholds"]["milestones"]
    d = pd.read_csv(tick_path, low_memory=False, usecols=["sim_time_s", "raw_behind_m", "raw_below_m", "raw_cross_m"])
    m2 = ((d.raw_behind_m - th["m2_behind_target"]).abs() <= th["m2_behind_tol"]) & ((d.raw_below_m - th["m2_below_target"]).abs() <= th["m2_below_tol"]) & (d.raw_cross_m.abs() <= th["m2_cross_tol"])
    t2, marker = (float(d.sim_time_s[m2].iloc[0]), "box entry") if m2.any() else (None, "closest approach")
    V3 = instrument()
    rows = V3.load(leg_dir)
    times = [float(r["sim_time_s"]) for r in rows]
    pm, tm = [V3.T(r["ctrl_m3b"]) for r in rows], [V3.T(r["raw_m3b"]) for r in rows]
    lhat = V3.xcorr_lag([float(r["ctrl_behind_m"]) for r in rows], [float(r["raw_behind_m"]) for r in rows])
    mask = V3.shifted_and(pm, tm, lhat)
    t3 = first_run(times, mask, th["m3b_dwell_s"], V3.GAP)
    if t3 is not None and (t2 is None or t2 > t3):                       # never in the box before M3b: closest approach before M3b
        pre = d[d.sim_time_s < t3]
        dist = np.sqrt((pre.raw_behind_m - th["m2_behind_target"]) ** 2 + (pre.raw_below_m - th["m2_below_target"]) ** 2 + pre.raw_cross_m ** 2)
        t2, marker, closest = float(pre.sim_time_s[dist.idxmin()]), "closest approach", float(dist.min())
    else:
        closest = 0.0
    return {"m2_marker": marker, "closest_to_m2_m": round(closest, 2),"corpus": tag, "leg": leg, "m2_achieved_s": t2, "m3b_achieved_s": t3, "time_to_dock_s": (t3 - t2) if (t2 is not None and t3 is not None) else None,
            "tick_dwell_s": round(V3.mask_dwell(times, mask), 3), "board_dwell_s": board_dwell, "lag_ticks": lhat}


def time_to_dock(board, roots, workers):
    b = T.arm_sigma(board)
    ok = b[b.passed.astype(bool)]
    jobs = []
    for r in ok.itertuples():
        legdir = glob.glob(os.path.join(roots[r.corpus], "**", r.leg), recursive=True)
        tick = glob.glob(os.path.join(legdir[0], "env*", "flight_state_ep*.csv.gz")) if legdir else []
        sm = glob.glob(os.path.join(legdir[0], "episode_*", "success_metrics.json")) if legdir else []
        if not tick or not sm:
            raise SystemExit("REFUSE: captured leg %s/%s lacks a tick record or success_metrics.json" % (r.corpus, r.leg))
        jobs.append((r.corpus, r.leg, tick[0], sm[0], legdir[0], float(r.intersection_v3_s)))
    with mp.Pool(workers) as pool:
        per = pd.DataFrame(pool.map(ttd_leg, jobs, chunksize=4))
    bad = per[(per.tick_dwell_s < HOLD - 1e-9) | ((per.tick_dwell_s - per.board_dwell_s).abs() > 0.0015)]
    if len(bad):
        print(bad.head(10).to_string())
        raise SystemExit("REFUSE rc 9: %d captured legs fail the dwell reproduction (instrument run == board intersection_v3_s to 3 dp)" % len(bad))
    per.to_csv(os.path.join(OUT[0], "TIME_TO_DOCK_PER_LEG_RAW.csv"), index=False)   # written BEFORE any refusal, for the read
    print("time to dock: %d captured legs; M2 marker = closest approach on %d" % (len(per), int((per.m2_marker == "closest approach").sum())))
    if per.time_to_dock_s.isna().any() or (per.time_to_dock_s < 0).any():
        raise SystemExit("REFUSE: %d captured legs have no M2 marker before M3b" % int((per.time_to_dock_s.isna() | (per.time_to_dock_s < 0)).sum()))
    per = per.merge(ok[["corpus", "leg", "arm", "sig", "stage", "trajectory"]], on=["corpus", "leg"])
    agg = {"n": "size", **{v: (lambda s, v=v: np.percentile(s, 95) if v == "p95" else getattr(s, v)()) for v in VALS}}
    by = lambda keys, lab: per.groupby(keys, sort=False).time_to_dock_s.agg(**agg).reset_index().assign(**lab)
    t = pd.concat([by(["arm", "sig", "stage", "trajectory"], {}), by(["arm", "sig", "trajectory"], {"stage": "all stages incl. S0"}),
                   by(["arm", "sig"], {"stage": "all stages incl. S0", "trajectory": "both"})], ignore_index=True)
    return t.round({v: 1 for v in VALS}), per


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("--run-dir", "--out", "--ticklevel-dir", "--summaries-dir", "--only"):
        ap.add_argument(k)
    ap.add_argument("--workers", type=int, default=8), ap.add_argument("--skip-ttd", action="store_true"), ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    import rescore_run as R
    board = pd.read_csv(os.path.join(a.run_dir, "board", "BOARD_W10.csv"), low_memory=False)
    out, tl = a.out or os.path.join(a.run_dir, "comparative"), a.ticklevel_dir or os.path.join(a.run_dir, "ticklevel")
    os.makedirs(out, exist_ok=True)
    OUT[0] = out
    tags = a.only.split(",") if a.only else list(R.CORPORA)
    tables = {"ABORTS": aborts(pd.read_csv(os.path.join(tl, "ABORT_PER_LEG.csv"), low_memory=False)),
              "STOPPING": pd.concat([stopping(load_summaries(a.summaries_dir or os.path.join(tl, "summaries"), tags)),
                                     stopping_bounded(tl, tags)], ignore_index=True),
              "RECOVERY": recovery(pd.read_csv(os.path.join(tl, "RECOVERY_SUMMARY.csv")))}
    if not a.skip_ttd:
        tables["TIME_TO_DOCK"], per = time_to_dock(board[board.corpus.isin(tags)], {t: R.resolve(t) for t in tags}, a.workers)
        per.to_csv(os.path.join(out, "TIME_TO_DOCK_PER_LEG.csv"), index=False)
    for k, df in tables.items():
        T.write(df, out, k)
        print("%s %d rows" % (k, len(df)))
    return 0


def selftest():
    t = list(np.arange(0, 20, 0.05))
    mask = [(3.0 <= x < 4.0) or (10.0 <= x) for x in t]               # a 1 s run, then a run from 10 s on
    assert abs(first_run(t, mask, 3.0, 0.2) - 13.0) < 0.051 and first_run(t, [False] * len(t), 3.0, 0.2) is None
    V3 = instrument(); assert abs(V3.GAP - 0.2) < 1e-12 and callable(V3.mask_dwell)
    pl = pd.DataFrame({"tag": ["A", "A"], "vertical_fires": [2, 0], "vertical_truth_events_all": [1, 1], "vertical_truth_events": [1, 0],
                       "vertical_false_tol1.0": [1, 0], "vertical_missed_tol1.0": [0, 0], "n_breakaways": [2, 0]})
    a = aborts(pl); assert a.iloc[0]["false-abort rate (% of fires)"] == 50.0 and a.iloc[0]["legs with a false abort"] == 1
    assert bounded_limb()
    print("SELFTEST PASS")
    return 0


def bounded_limb():
    """One ceiling breakaway: truth peaks 0.5 s after the fire INSIDE the breakaway (2.10 m), then a LATER, higher
    excursion (2.90 m) after the breakaway ends. The bounded rise must be 0.10 m at 0.5 s, not 0.90 m."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tr = np.round(np.arange(-1.0, 6.0, 0.25), 2)
        up = np.where(tr < 0, 1.9, np.where(tr <= 0.5, 2.0 + 0.2 * tr, np.where(tr < 3.0, 1.5, 2.9)))
        pd.DataFrame({"tag": "A", "leg": "L", "event_idx": 0, "t_rel_s": tr, "raw_below_m": -up, "ctrl_below_m": -(up + 0.3),
                      "in_breakaway": (tr >= 0) & (tr < 3.0)}).to_csv(os.path.join(td, "RECOVERY_TRACE_A.csv.gz"), index=False)
        pd.DataFrame({"tag": ["A"], "leg": ["L"], "event_idx": [0], "kind": ["breakaway"],
                      "trigger_reason": ["guard_altitude_ceiling"]}).to_csv(os.path.join(td, "ABORT_EVENTS.csv"), index=False)
        s = stopping_bounded(td, ["A"]).set_index("quantity")["max"]
    p = "vertical, inside the breakaway: "
    return (abs(s[p + "rise after fire, truth (m)"] - 0.10) < 1e-9 and abs(s[p + "time to peak, truth (s)"] - 0.5) < 1e-9
            and abs(s[p + "time back under bar, truth (s)"] - 0.0) < 1e-9 and abs(s[p + "time back under bar, perception (s)"] - 0.75) < 1e-9)


if __name__ == "__main__":
    sys.exit(main())
