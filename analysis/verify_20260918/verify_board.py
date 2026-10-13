#!/usr/bin/env python3
# =============================================================================

#   record of run_20260918T2208Z BEFORE anything is built on it.
#   Step 7 denominators and pairing: 600 legs per corpus, 540 wind-stage legs, 180 strong-wind legs, 60 per stage,
#           300/300 trajectories, seed block 10775-10804 with 20 legs per seed, one row per (seed, stage, trajectory)
#           key, the SAME key set in all twelve corpora, sigma only 0.00 / 0.25 / 0.35, gradable / clean / fault counts.
#   Step 8 exact reproduction against the 09-03 board (analysis/top2_20260903T1900Z/TOP2_BOARD_W10.csv) on the six
#           corpora it holds (A2P025/035, CSV025/035, ARMB025/035), leg by leg on (corpus, key), for every column the
#           two boards share. Floats are compared exactly and with |diff| <= 1e-9; the max |diff| is printed.
#   Step 9 every leg whose value differs is written out with both values; the success-flag change between the exam's
#           flag and the flag of record is tabulated against the corridor and whole-leg peaks, so each change carries
#           its cause or is listed as UNEXPLAINED.
#   Every number is computed here with pandas / numpy from the two CSVs; nothing is typed in.
# CLI: python verify_board.py --run-dir DIR --out DIR
# SIDE-EFFECTS: reads the two boards READ-ONLY; writes CSV / txt under --out only.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import os
import sys

import numpy as np
import pandas as pd

OLD = (_REPO + "/results/TOP2_BOARD_W10_20260903.csv")
SEEDS = set(range(10775, 10805))
BAR = 2.00


def step7(b):
    rows, ok = [], True
    keys = {c: set(g.key) for c, g in b.groupby("corpus")}
    ref = next(iter(keys.values()))
    for c, g in b.groupby("corpus"):
        w = g[g.wind_rung]
        r = {"corpus": c, "n": len(g), "unique_keys": g.key.nunique(), "wind_stage_n": len(w),
             "strong_wind_n": int(g.sw.sum()), "sw_winds": sorted(g[g.sw].wind_ms.unique().tolist()),
             "S0_n": int((g.wind_label == "S0").sum()), "stages": g.cell.nunique(),
             "min_per_stage": int(g.groupby("cell").size().min()), "max_per_stage": int(g.groupby("cell").size().max()),
             "straight": int((g.trajectory == "straight").sum()), "racetrack": int((g.trajectory == "racetrack").sum()),
             "seeds_ok": set(g.seed) == SEEDS, "legs_per_seed": sorted(g.groupby("seed").size().unique().tolist()),
             "sigmas": sorted(g.sigma.unique().tolist()), "same_keys_as_first": keys[c] == ref,
             "gradable": int(g.gradable.sum()), "clean": int(g.clean.sum()),
             "fault_classes": g.fault_class.value_counts().to_dict()}
        good = (r["n"] == 600 and r["unique_keys"] == 600 and r["wind_stage_n"] == 540 and r["strong_wind_n"] == 180
                and r["sw_winds"] == [6.0, 7.0, 8.0] and r["S0_n"] == 60 and r["stages"] == 10
                and r["min_per_stage"] == r["max_per_stage"] == 60 and r["straight"] == r["racetrack"] == 300
                and r["seeds_ok"] and r["legs_per_seed"] == [20] and r["same_keys_as_first"]
                and set(r["sigmas"]) <= {0.0, 0.25, 0.35} and len(r["sigmas"]) == 2)
        r["PASS"] = good
        ok = ok and good
        rows.append(r)
    return pd.DataFrame(rows), ok


def step8(b, old):
    shared = [c for c in old.columns if c in b.columns and c not in ("corpus", "key")]
    m = old.merge(b, on=["corpus", "key"], how="outer", suffixes=("_old", "_new"), indicator=True)
    out, diffs = [], []
    for c in shared:
        a, n = m[c + "_old"], m[c + "_new"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(n) and a.dtype != bool:
            both_nan = a.isna() & n.isna()
            ad = (a.astype(float) - n.astype(float)).abs()
            same = both_nan | (ad <= 1e-9)
            mx = float(np.nanmax(ad.values)) if ad.notna().any() else 0.0
        else:
            same = a.astype(str) == n.astype(str)
            mx = np.nan
        d = m.loc[~same & (m._merge == "both"), ["corpus", "key", c + "_old", c + "_new"]]
        out.append({"column": c, "legs_compared": int((m._merge == "both").sum()), "legs_differ": len(d),
                    "max_abs_diff": mx})
        if len(d):
            diffs.append(d.rename(columns={c + "_old": "old", c + "_new": "new"}).assign(column=c))
    orphans = m[m._merge != "both"].groupby(["_merge", "corpus"], observed=True).size()
    return pd.DataFrame(out), (pd.concat(diffs) if diffs else pd.DataFrame()), orphans


def cause(r):
    """Why the flag of record differs from the exam's flag (plan: condition 2 read inside the corridor, not the leg)."""
    if r.passed and not r.passed_exam:
        return "record PASS / exam FAIL: " + ("peak over bar only outside the corridor (whole %.3f > 2.00 >= corridor %s)"
                                              % (r.whole, "%.3f" % r.win if pd.notna(r.win) else "none")
                                              if r.whole > BAR and not (r.win > BAR) else "UNEXPLAINED")
    if r.passed_exam and not r.passed:
        return "record FAIL / exam PASS: " + ("corridor peak %.3f > 2.00" % r.win if r.win > BAR
                                              else "overtake" if r.real_overtake else "UNEXPLAINED")
    return "same"


def step9(b):
    ch = b[b.passed != b.passed_exam].copy()
    ch["cause"] = ch.apply(cause, axis=1)
    return ch[["corpus", "key", "passed_exam", "passed", "win", "whole", "real_overtake", "fail_reasons", "cause"]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    b = pd.read_csv(os.path.join(a.run_dir, "board", "BOARD_W10.csv"))
    old = pd.read_csv(OLD) if os.path.exists(OLD) else None
    t7, ok7 = step7(b)
    t7.to_csv(os.path.join(a.out, "STEP7_DENOMINATORS.csv"), index=False)
    print("STEP 7:", "PASS" if ok7 else "FAIL", "| corpora", len(t7), "| rows", len(b))
    print(t7.drop(columns=["fault_classes", "sw_winds", "legs_per_seed"]).to_string(index=False))
    print("fault classes:", {r.corpus: r.fault_classes for r in t7.itertuples()})
    if old is not None:
        t8, d8, orph = step8(b, old)
        t8.to_csv(os.path.join(a.out, "STEP8_REPRODUCTION_BY_COLUMN.csv"), index=False)
        d8.to_csv(os.path.join(a.out, "STEP8_DIFFERING_LEGS.csv"), index=False)
        print("\nSTEP 8: 09-03 board", OLD, "| rows", len(old))
        print("legs present in only one board, by corpus:\n", orph.to_string())
        print(t8.to_string(index=False))
    else:
        print("\nSTEP 8 skipped: the earlier board is not part of this package")
    t9 = step9(b)
    t9.to_csv(os.path.join(a.out, "STEP9_FLAG_CHANGES.csv"), index=False)
    print("\nSTEP 9: legs whose flag of record differs from the exam flag:", len(t9))
    print(t9.groupby(["corpus", t9.cause.str.split(":").str[0]]).size().to_string())
    print("UNEXPLAINED:", int(t9.cause.str.contains("UNEXPLAINED").sum()))
    return 0 if ok7 else 1


if __name__ == "__main__":
    sys.exit(main())
