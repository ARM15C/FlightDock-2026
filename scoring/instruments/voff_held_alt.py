#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: limb L4 of PREREG_VERTOFF ("did the offset FIRE?") -- the per-leg
#   HELD altitude of the follower relative to the boom tip, read PER TICK from the
#   banked paired-channels CSV, inside the ruled window (truth range <= 3.0 m).
#   Reported as a per-leg MEDIAN, never a peak aggregate (class C66: an aggregate
#   of per-leg maxima is not a held value -- the 09-01 lesson that reframed the
#   altitude problem, analysis/heldalt_20260901T0000Z). A VOFF wave whose held
#   point did not move DOWN by about its offset versus the same-seed control did
#   not deliver the lever: INCONCLUSIVE, never GREEN.
# SIGN CONVENTION, READ NOT ASSUMED: `raw_below_m` is POSITIVE when the follower
#   is BELOW the tip (first tick of every leg reads ~+9 m at 15 m range: the
#   follower starts low). Held altitude here is ABOVE-the-tip = -raw_below_m, the
#   same sign as the H1 peak channel (positive = toward the leader belly).
# PUBLIC API: leg_held(csv_path, window_m) -> dict ; census(legs_root, window_m)
#   -> {leg: dict} ; CLI --legs-root DIR --out JSON [--window-m 3.0] [--selftest]
# SIDE-EFFECTS: reads CSVs; writes one JSON when --out is given. Spawns nothing,
#   imports no training code, no AWS SDK. Truth channel only (raw_*): never a
#   policy input, never a reward term.
# CAPS: file soft 70 / hard 90 code lines; functions <= 25 code lines.
# COVERAGE: --selftest drives a synthetic CSV both ways (in-window rows only are
#   used; the median is the median; out-of-window rows are ignored; a missing
#   column refuses).
# =============================================================================
import argparse
import csv
import json
import os
import statistics
import sys
import tempfile

WIN_M = 3.0
PAIRED = os.path.join("paired", "env0", "paired_channels_ep0001.csv")


def leg_held(path, window_m=WIN_M):
    above = []
    with open(path) as fh:
        rd = csv.DictReader(fh)
        for need in ("raw_range_m", "raw_below_m"):
            if need not in rd.fieldnames:
                raise RuntimeError("column %s absent in %s" % (need, path))
        for r in rd:
            try:
                if float(r["raw_range_m"]) <= window_m:
                    above.append(-float(r["raw_below_m"]))
            except ValueError:
                continue
    if not above:
        return {"n_win_ticks": 0, "median_above_m": None, "mean_above_m": None}
    return {"n_win_ticks": len(above),
            "median_above_m": round(statistics.median(above), 4),
            "mean_above_m": round(sum(above) / len(above), 4)}


def census(legs_root, window_m=WIN_M):
    out = {}
    for name in sorted(os.listdir(legs_root)):
        p = os.path.join(legs_root, name, PAIRED)
        if os.path.isfile(p):
            out[name] = leg_held(p, window_m)
    if not out:
        raise RuntimeError("no leg carries %s under %s" % (PAIRED, legs_root))
    return out


def _selftest():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "L", *PAIRED.split(os.sep))
    os.makedirs(os.path.dirname(p))
    rows = [(2.0, 0.5), (2.5, 0.3), (1.0, 0.7), (9.0, -5.0), (2.9, "x")]
    with open(p, "w") as fh:
        fh.write("sim_time_s,raw_range_m,raw_below_m\n")
        for i, (rg, bl) in enumerate(rows):
            fh.write("%d,%s,%s\n" % (i, rg, bl))
    got = census(d)["L"]
    ok = got == {"n_win_ticks": 3, "median_above_m": -0.5, "mean_above_m": -0.5}
    open(p, "w").write("sim_time_s,raw_range_m\n1,2.0\n")
    try:
        leg_held(p); refused = False
    except RuntimeError:
        refused = True
    print("SELFTEST", "PASS" if ok and refused else "FAIL", got, "refuses_missing_col", refused)
    return 0 if ok and refused else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs-root"); ap.add_argument("--out")
    ap.add_argument("--window-m", type=float, default=WIN_M)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    doc = {"legs_root": a.legs_root, "window_m": a.window_m,
           "sign": "median_above_m = -raw_below_m; positive = ABOVE the boom tip",
           "legs": census(a.legs_root, a.window_m)}
    if a.out:
        json.dump(doc, open(a.out, "w"), indent=1)
    meds = [v["median_above_m"] for v in doc["legs"].values() if v["median_above_m"] is not None]
    print("%d legs, %d with window ticks, corpus median-of-medians %.3f m" % (
        len(doc["legs"]), len(meds), statistics.median(meds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
