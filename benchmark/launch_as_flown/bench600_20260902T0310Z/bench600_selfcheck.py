#!/usr/bin/env python3
# RESPONSIBILITY: the COVERAGE file for the BENCH600 twins -- every declared delta driven BOTH
#   directions on dev, spawning no SITL and touching no box. Limbs:
#     H1-H5 bench600_sigma_hook: sigma 0.25 takes the parent path (band 0.0799, no --paired-csv);
#            sigma 0.00 with a readable paired CSV forwards --paired-csv and runs the calm witness;
#            calm witness rc 5 => 17; paired CSV absent => 16 with NO gate spawned; witness rc 0 => 0.
#     T1-T6 bench600_exam_table: both ladders 600 rows / 25 deep at width 24; B25R refuses; drifted
#            copy refuses; S0 row without paired_csv, a 0.20 wind row, a 0.00 row off S0 all refuse.
#     M1-M7 bench600_manifest_check: both signed runs pass; sibling table 33; wrong policy 32;
#            wrong dose 34; unknown run 31; absent manifest 35.
#     O1-O3 bench600_overlay_exam: PREFLIGHT re-pointed; a knob on the argv refuses 14; the
#            preflight twin execs bench600_exam_wave.py and is executable (the 19:00Z lesson).
#     W1-W2 wave twin: SIGMA_HOOK and the table gate resolve to the bench600 files by path.
#     F1-F11 bench600_fire.sh (in the sibling bench600_selfcheck_fire.py -- split at the cap): unknown RUN
#            15, wrong pin 12, census file absent 11, tampered SHAS 13, buffer reachable 13, all gates
#            pass => "GATES PASS" fixture stop rc 0, no tmux; D7 signature gate driven FOUR ways (F7a
#            unsigned MUST be 16, F7b signed MUST be 0, F7c wrong sha 16, F7d absent bank 17 -- no limb
#            accepts a range, which is how the first D7 shipped as a no-op); D8 resume F10/F11.
# PUBLIC API: python bench600_selfcheck.py (rc 0 = every limb passed; each limb printed)
# SIDE-EFFECTS: temp dirs only; monkeypatches subprocess.run INSIDE this process for the hook
#   limbs; runs the fire twin with BENCH600_FIXTURE=1 (stops before the launch line). No SITL.
# SIZE: soft 110 / hard 140 code-lines; function cap 40.
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

FD = "/home/ubuntu/FlightDock"
B = os.path.join(FD, "analysis/bench600_20260902T0310Z")
PROT = os.path.join(FD, "analysis/b25r_20260830T0040Z/stage/A2/protected")
T25 = os.path.join(FD, "analysis/armblad025_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD025.json")
T35 = os.path.join(FD, "analysis/armblad035_20260830T1600Z/SEED_CELL_TABLE_ARMBLAD035.json")
FAILS = []


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(B, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def limb(tag, cond, msg):
    print(" %s %s: %s" % ("PASS" if cond else "FAIL", tag, msg))
    if not cond:
        FAILS.append(tag)


def hook_limbs():
    H = load("bench600_sigma_hook")
    calls, rcs = [], []

    def fake(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rcs.pop(0))
    td = tempfile.mkdtemp()
    log = os.path.join(td, "lane.log")
    leg = os.path.join(td, "leg")
    os.makedirs(os.path.join(leg, "env3"))
    real, subprocess.run = subprocess.run, fake
    try:
        rcs[:] = [0]
        rc = H.gate(leg, 3, 0.25, log)
        limb("H1", rc == 0 and "--paired-csv" not in calls[-1] and "0.0799" in calls[-1],
             "sigma 0.25 -> parent path, band 0.0799, no --paired-csv")
        rcs[:] = [0]
        rc = H.gate(leg, 3, 0.0, log)
        limb("H2", rc == 16 and len(calls) == 1, "sigma 0.00 with the paired CSV absent -> 16, no gate spawned")
        open(os.path.join(leg, "env3", "paired_channels_ep0001.csv"), "w").write("resolved_wind_max_ms\n0.0\n")
        rcs[:] = [0, 0]
        rc = H.gate(leg, 3, 0.0, log)
        limb("H3", rc == 0 and "--paired-csv" in calls[-2] and calls[-1][1].endswith("armblad025_calm_witness.py"),
             "sigma 0.00 + CSV -> --paired-csv forwarded, calm witness run, rc 0")
        rcs[:] = [0, 5]
        limb("H4", H.gate(leg, 3, 0.0, log) == 17, "calm witness rc 5 (tier never ran) -> 17")
        rcs[:] = [3, 3]
        limb("H5", H.gate(leg, 3, 0.0, log) == 3, "calm tier ASSERT_FAIL (wind delivered on a calm cell) -> 3")
    finally:
        subprocess.run = real
        shutil.rmtree(td, ignore_errors=True)


def table_limbs():
    T = load("bench600_exam_table")
    for tag, t, key in (("T1", T25, "B25"), ("T2", T35, "B35")):
        ln = T.lanes(t, 24)
        depth = {}
        for l in ln:
            depth[l["env_id"]] = depth.get(l["env_id"], 0) + 1
        limb(tag, len(ln) == 600 and set(depth.values()) == {25} and T.expected_by_sigma(t) == {"B00": 60, key: 540},
             "%s: 600 lanes, 25 deep x 24, B00 60 + %s 540" % (os.path.basename(t), key))

    def refuses(fn):
        try:
            fn()
            return False
        except T.Refusal:
            return True
    limb("T3", refuses(lambda: T.table_rows(os.path.join(FD, "analysis/armbsig_20260824T0030Z/SEED_CELL_TABLE_ARMBSIG_B25R.json"))),
         "grid B25R (signed elsewhere) refuses")
    rows = json.load(open(T25))["legs"]
    td = tempfile.mkdtemp()
    p = os.path.join(td, "drift.json")
    json.dump({"legs": rows[:-1]}, open(p, "w"))
    limb("T4", refuses(lambda: T.table_rows(p)), "drifted copy (sha unlisted) refuses")
    shutil.rmtree(td, ignore_errors=True)
    b1 = [dict(r) for r in rows]
    b1[0]["paired_csv"] = False
    b2 = [dict(r) for r in rows]
    b2[100]["sigma_ms"] = 0.20
    b3 = [dict(r) for r in rows]
    b3[100]["sigma_ms"] = 0.0
    limb("T5", all(T.sigma_refusal(b, 0.25) for b in (b1, b2, b3)) and T.sigma_refusal(rows, 0.25) is None,
         "S0 w/o paired_csv, a 0.20 wind row, a 0.00 row off S0 all refuse; the genuine table passes")
    limb("T6", "0.20 is PURGED" in T.sigma_refusal(b2, 0.25), "the T5 refusal text names the purge")


def manifest_limbs():
    M = load("bench600_manifest_check")
    mf = os.path.join(B, "WAVE_MANIFEST_BENCH600.json")
    ck, vn = os.path.join(PROT, "model_final.zip"), os.path.join(PROT, "vecnormalize_final.pkl")
    cases = (("M1", "H1H2", ck, T25, 0.75, 0), ("M2", "H3", ck, T35, 0.75, 0), ("M3", "H1H2", ck, T35, 0.75, 33),
             ("M4", "H1H2", vn, T25, 0.75, 32), ("M5", "H1H2", ck, T25, 1.00, 34), ("M6", "H9", ck, T25, 0.75, 31))
    for tag, run, c, t, d, want in cases:
        rc = M.check(mf, run, c, t, d)[0]
        limb(tag, rc == want, "run %s ckpt %s table %s dose %.2f -> rc %d (want %d)"
             % (run, os.path.basename(c)[:10], os.path.basename(t)[-8:-5], d, rc, want))
    limb("M7", M.check("/nonexistent", "H1H2", ck, T25, 0.75)[0] == 35, "absent manifest -> 35")


def overlay_wave_limbs():
    O = load("bench600_overlay_exam")
    ov = O.load()
    limb("O1", ov.PREFLIGHT == os.path.join(B, "bench600_preflight.sh"), "PREFLIGHT re-pointed at the twin")
    limb("O2", O.main(["--dose", "0.75", "--ckpt", "x", "--out-dir", "/x", "--forward-bias-ms", "0.2"]) == 14,
         "a knob on the argv refuses 14 before the parent parses it")
    pf = open(os.path.join(B, "bench600_preflight.sh")).read()
    limb("O3", 'exec python3 "$B/bench600_exam_wave.py" "$@"' in pf and os.access(os.path.join(B, "bench600_preflight.sh"), os.X_OK),
         "preflight twin execs the wave twin and is executable")
    w = open(os.path.join(B, "bench600_exam_wave.py")).read()
    limb("W1", 'os.path.join(HERE, "bench600_sigma_hook.py")' in w, "wave twin SIGMA_HOOK -> bench600_sigma_hook.py")
    limb("W2", 'os.path.join(HERE, "bench600_exam_table.py")' in w and 'os.path.join(HERE, "armbsig' not in w,
         "wave twin table gate -> bench600_exam_table.py; no parent sibling resolved via HERE")


def fire_limbs():
    load("bench600_selfcheck_fire").fire_limbs(limb)   # F1-F6 live in the sibling (cap split, never raised)


def main():
    for f in (hook_limbs, table_limbs, manifest_limbs, overlay_wave_limbs, fire_limbs):
        f()
    print("bench600_selfcheck: %s (%d failing limb(s))" % ("PASS" if not FAILS else "FAIL " + ",".join(FAILS), len(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
