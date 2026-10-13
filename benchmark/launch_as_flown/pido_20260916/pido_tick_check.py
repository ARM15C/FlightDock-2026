#!/usr/bin/env python3
# RESPONSIBILITY: the ZERO-CORRECTION tick check for a flown PIDO corpus. Reads every FINISHED leg's (results file present;
#   a half-written record from a flying leg is never read -- measured 09-17 03:05Z: gzip EOFError on a live leg) per-tick flight-state record
#   (flight_state_ep*.csv[.gz], written by src/drl/flight_state_recorder.py) and asserts, on EVERY tick of EVERY leg, that
#   the applied speed residual (res_applied_0) and heading residual (res_applied_1) are identically zero and the applied
#   altitude residual (res_applied_2) lies in [-offset, 0] (the gated down-offset ramps in through an EMA; nothing else may
#   move it). A tick that fails marks its leg FAIL; a corpus with any FAIL leg, or fewer legs than --expect-legs, reads
#   INCONCLUSIVE. This is the instrument PREREG_PIDO.md section 5 names; it decides no hypothesis.
# PUBLIC API: --legs-root <exam out-dir> --offset 0.25 [--expect-legs 600] [--tol 1e-6] --out <json>
#   rc 0 PASS. 1 INCONCLUSIVE (a failing leg, a missing record, or a short corpus). 2 no legs found. 3 usage/IO
#   --selftest builds a passing and a failing synthetic corpus and asserts rc 0 then 1 (fixture both directions).
# SIDE-EFFECTS: writes one JSON (the selftest uses a temp dir, removed). Reads only. Imports pandas/numpy (the analysis rule); nothing from src/.
# SIZE: soft 60 / hard 80 code-lines; function cap 30.
# COVERAGE: --selftest (both directions); the JSON per corpus is the record.
import argparse, glob, gzip, json, os, shutil, sys, tempfile
import numpy as np
import pandas as pd
COLS = ["res_applied_0", "res_applied_1", "res_applied_2"]

def leg_dirs(root):
    """FINISHED legs only (a results file exists): a leg still flying holds an open, half-written record."""
    return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(root, "B*", "*", "results_eid*.json"))})

def check_leg(leg, offset, tol):
    files = sorted(glob.glob(os.path.join(leg, "env*", "flight_state_ep*.csv*")))
    if not files: return {"verdict": "NO_RECORD", "n_ticks": 0}
    try:
        df = pd.concat([pd.read_csv(f, usecols=lambda c: c in COLS) for f in files], ignore_index=True)
    except (OSError, EOFError, ValueError) as e:
        return {"verdict": "UNREADABLE", "n_ticks": 0, "error": type(e).__name__}
    if any(c not in df.columns for c in COLS): return {"verdict": "NO_COLUMNS", "n_ticks": int(len(df))}
    d = df[COLS].apply(pd.to_numeric, errors="coerce").dropna()
    a0, a1, a2 = (np.abs(d["res_applied_0"]).max(), np.abs(d["res_applied_1"]).max(), d["res_applied_2"])
    ok = len(d) > 0 and a0 <= tol and a1 <= tol and a2.min() >= -offset - tol and a2.max() <= tol
    return {"verdict": "PASS" if ok else "FAIL", "n_ticks": int(len(d)), "max_abs_speed_res": float(a0) if len(d) else None,
            "max_abs_heading_res": float(a1) if len(d) else None, "alt_res_min": float(a2.min()) if len(d) else None,
            "alt_res_max": float(a2.max()) if len(d) else None}

def run(root, offset, expect, tol, out):
    legs = leg_dirs(root)
    if not legs: print("no legs with a flight-state record under %s" % root); return 2
    res = {os.path.relpath(l, root): check_leg(l, offset, tol) for l in legs}
    n_pass = sum(r["verdict"] == "PASS" for r in res.values())
    verdict = "PASS" if n_pass == len(res) and len(res) >= expect else "INCONCLUSIVE"
    doc = {"legs_root": root, "offset_m": offset, "tol": tol, "expect_legs": expect, "n_legs": len(res), "n_pass": n_pass,
           "n_fail": sum(r["verdict"] == "FAIL" for r in res.values()), "n_no_record": sum(r["verdict"] in ("NO_RECORD", "NO_COLUMNS", "UNREADABLE") for r in res.values()),
           "verdict": verdict, "legs": res}
    json.dump(doc, open(out, "w"), indent=1)
    print("TICK CHECK %s: legs %d, pass %d, fail %d, no record %d -> %s" % (verdict, len(res), n_pass, doc["n_fail"], doc["n_no_record"], out))
    return 0 if verdict == "PASS" else 1

def selftest():
    td = tempfile.mkdtemp(); rcs = []
    for tag, hdg in (("good", 0.0), ("bad", 0.01)):
        root = os.path.join(td, tag)
        for i in range(3):
            d = os.path.join(root, "B25-W0", "W0_1077%d_straight" % i, "env%d" % i); os.makedirs(d)
            open(os.path.join(os.path.dirname(d), "results_eid0001.json"), "w").write("{}")
            with gzip.open(os.path.join(d, "flight_state_ep0001.csv.gz"), "wt") as fh:
                fh.write("tick_idx,res_applied_0,res_applied_1,res_applied_2,res_source\n")
                for t in range(50): fh.write("%d,0.0,%s,%.4f,x\n" % (t, hdg if (tag == "bad" and i == 2 and t == 49) else 0.0, -0.25 * min(1.0, t / 30.0)))
        rcs.append(run(root, 0.25, 3, 1e-6, os.path.join(td, tag + ".json")))
    rcs.append(run(os.path.join(td, "good"), 0.25, 600, 1e-6, os.path.join(td, "short.json")))
    shutil.rmtree(td, ignore_errors=True); print("selftest rcs (want [0, 1, 1]):", rcs); return 0 if rcs == [0, 1, 1] else 1

def main(argv=None):
    ap = argparse.ArgumentParser(description="PIDO zero-correction tick check")
    ap.add_argument("--legs-root"); ap.add_argument("--offset", type=float, default=0.25); ap.add_argument("--expect-legs", type=int, default=600)
    ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest: return selftest()
    if not a.legs_root or not a.out: ap.print_usage(); return 3
    return run(a.legs_root, a.offset, a.expect_legs, a.tol, a.out)

if __name__ == "__main__":
    sys.exit(main())
