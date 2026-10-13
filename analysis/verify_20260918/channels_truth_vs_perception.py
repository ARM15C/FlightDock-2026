#!/usr/bin/env python3
# =============================================================================

#   should include both truth and perception"). NOT in the signed plans; decides nothing.
# WHAT: inside the aerial-docking terminal-approach corridor (ticks with truth range <= 10.0 m), for all twelve
#   corpora, the THREE logged views of the QuadPlane's position relative to the boom-tip, side by side:
#     truth raw_* (the simulator's true positions; the channel of record)
#     perception ctrl_* (what the controller reads; the altitude guard's default channel, per
#                                    FD_GUARD_CEILING_CHANNEL default 'ctrl' in tests/baseline_formation_controller.py)
#   Table 1 per arm, sigma and axis: the VALUE on each view (median [p5, p95] over ticks) and each perception
#            view's error against truth (bias, RMSE, mean absolute error). Height = -below (m above the boom-tip).
#   Table 2 at each altitude-guard fire inside the corridor: height above the tip on the three views at the fire
#            tick (median), and the share of fires with each view above 2.00 m.
#   Quantiles over ticks are computed from per-leg tick samples pooled with numpy (every corridor tick is used for
#   the error sums; value quantiles use every 5th corridor tick to bound memory, stated in the output).
# CLI: --extract [--workers 12] | --report SIDE-EFFECTS: writes CHANNELS_*.csv / CHANNELS_REPORT.md under OUT only.
# =============================================================================
import argparse
import concurrent.futures as cf
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offset_delivery import OUT, RUN, TAGS, arm, md  # noqa: E402

AXES = {"height above the tip": ("below", -1.0), "behind the tip": ("behind", 1.0), "cross-track": ("cross", 1.0)}
VIEWS = {"truth": "raw", "perception": "ctrl"}   # two channels only
COLS = ["raw_range_m", "_guard_altitude_ceiling_fire_count"] + ["%s_%s_m" % (v, a) for v in VIEWS.values()
                                                                  for a, _ in AXES.values()]
STEP = 5


def one(job):
    tag, leg, path = job
    d = pd.read_csv(path, compression="gzip", usecols=COLS, low_memory=False)
    inc = (d.raw_range_m <= 10.0).to_numpy()
    k = d[inc]
    sums, samp = [], {}
    for ax, (col, sgn) in AXES.items():
        t = sgn * k["raw_%s_m" % col]
        for view, pre in VIEWS.items():
            v = sgn * k["%s_%s_m" % (pre, col)]
            samp[(ax, view)] = v.to_numpy(dtype=np.float32)[::STEP]
            if pre != "raw":
                e = (v - t).dropna()
                sums.append({"tag": tag, "axis": ax, "view": view, "n": len(e), "s": e.sum(), "sq": (e ** 2).sum(),
                             "ab": e.abs().sum()})
    fires = []
    cnt = d["_guard_altitude_ceiling_fire_count"].fillna(0).to_numpy()
    for i in np.flatnonzero(np.diff(cnt) > 0) + 1:
        if inc[i]:
            fires.append({"tag": tag, "leg": leg, **{view: -d["%s_below_m" % pre].iloc[i] for view, pre in VIEWS.items()}})
    return tag, sums, samp, fires


def extract(workers):
    jobs = [(t, p.split(os.sep)[-3], p) for t in TAGS
            for p in sorted(glob.glob(os.path.join(RUN, "shims", t, "flat", "W*", "env*", "flight_state_ep*.csv.gz")))]
    if len(jobs) != 12 * 540:
        raise SystemExit("REFUSE: expected 6480 wind-stage flight_state files, found %d" % len(jobs))
    sums, fires, pool = [], [], {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for tag, s, samp, f in ex.map(one, jobs, chunksize=20):
            sums += s
            fires += f
            for key, arr in samp.items():
                pool.setdefault((tag,) + key, []).append(arr)
    q = []
    for (tag, ax, view), arrs in pool.items():
        a = np.concatenate(arrs)
        a = a[~np.isnan(a)]
        q.append({"tag": tag, "axis": ax, "view": view, "ticks_sampled": len(a), "median": np.median(a),
                  "p5": np.percentile(a, 5), "p95": np.percentile(a, 95)})
    pd.DataFrame(q).to_csv(os.path.join(OUT, "CHANNELS_VALUES.csv"), index=False)
    pd.DataFrame(sums).groupby(["tag", "axis", "view"])[["n", "s", "sq", "ab"]].sum().reset_index().to_csv(
        os.path.join(OUT, "CHANNELS_ERRORS.csv"), index=False)
    pd.DataFrame(fires).to_csv(os.path.join(OUT, "CHANNELS_FIRES.csv"), index=False)
    print("EXTRACT DONE legs=%d fires=%d" % (len(jobs), len(fires)))


def relabel():
    """The 2026-09-18 extract carried three views; keep truth and perception (ctrl) and rewrite the intermediates."""
    for f, how in (("CHANNELS_VALUES.csv", "view"), ("CHANNELS_ERRORS.csv", "view"), ("CHANNELS_FIRES.csv", "cols")):
        d = pd.read_csv(os.path.join(OUT, f))
        if how == "view" and "perception (fused)" in set(d["view"]):
            d = d[d["view"].isin(["truth", "perception (fused)"])].replace({"view": {"perception (fused)": "perception"}})
        elif how == "cols" and "perception (fused)" in d.columns:
            d = d.drop(columns=["perception (GPS-direct)"]).rename(columns={"perception (fused)": "perception"})
        d.to_csv(os.path.join(OUT, f), index=False)


def report():
    relabel()
    V = pd.read_csv(os.path.join(OUT, "CHANNELS_VALUES.csv"))
    E = pd.read_csv(os.path.join(OUT, "CHANNELS_ERRORS.csv"))
    F = pd.read_csv(os.path.join(OUT, "CHANNELS_FIRES.csv"))
    E["bias"], E["rmse"], E["mae"] = E.s / E.n, np.sqrt(E.sq / E.n), E.ab / E.n
    rows = []
    for t in TAGS:
        for ax in AXES:
            r = {"arm": arm(t), "sigma": "0.25" if t.endswith("25") else "0.35", "axis (m)": ax}
            for view in VIEWS:
                v = V[(V.tag == t) & (V.axis == ax) & (V["view"] == view)].iloc[0]
                r[view + ": median [p5, p95]"] = "%.2f [%.2f, %.2f]" % (v["median"], v.p5, v.p95)
            for view in list(VIEWS)[1:]:
                e = E[(E.tag == t) & (E.axis == ax) & (E["view"] == view)].iloc[0]
                r[view + " minus truth: bias / RMSE / MAE"] = "%+.3f / %.3f / %.3f" % (e.bias, e.rmse, e.mae)
            r["corridor ticks"] = int(E[(E.tag == t) & (E.axis == ax)].n.max())
            rows.append(r)
    T1 = pd.DataFrame(rows)
    g = F.groupby("tag")
    T2 = pd.DataFrame({"arm": [arm(t) for t in TAGS], "sigma": ["0.25" if t.endswith("25") else "0.35" for t in TAGS],
                       "fires": [len(g.get_group(t)) if t in g.groups else 0 for t in TAGS]})
    for view in VIEWS:
        T2["height at fire, %s: median (m)" % view] = [round(g.get_group(t)[view].median(), 3) if t in g.groups else np.nan
                                                       for t in TAGS]
    for view in VIEWS:
        T2["fires with %s above 2.00 m (%%)" % view] = [round(100 * (g.get_group(t)[view] > 2.0).mean(), 1)
                                                         if t in g.groups else np.nan for t in TAGS]
    T1.to_csv(os.path.join(OUT, "CHANNELS_TABLE1_VALUES_AND_ERRORS.csv"), index=False)
    T2.to_csv(os.path.join(OUT, "CHANNELS_TABLE2_AT_GUARD_FIRE.csv"), index=False)
    txt = ["# Truth against perception inside the terminal-approach corridor - POST-PLAN; decides nothing",
           "", "Nine wind stages; corridor = ticks with truth range <= 10.0 m. Truth = raw_*; perception = ctrl_*",
           "(what the controller reads). Value quantiles use every %dth corridor tick; error statistics use every" % STEP,
           "corridor tick. Positive bias on 'height above the tip' = the view reads the QuadPlane HIGHER than truth.", "",
           "## 1. Values on truth and perception, and perception's error against truth", "", md(T1), "",
           "## 2. At each altitude-guard fire inside the corridor", "", md(T2), ""]
    open(os.path.join(OUT, "CHANNELS_REPORT.md"), "w").write("\n".join(txt))
    pd.set_option("display.width", 320)
    pd.set_option("display.max_colwidth", 60)
    print(T1[T1["axis (m)"] == "height above the tip"].drop(columns=["axis (m)"]).to_string(index=False))
    print()
    print(T2.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.extract:
        extract(a.workers)
    if a.report:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
