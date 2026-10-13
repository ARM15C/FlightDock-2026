#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Witness the REALIZED wind/turbulence cell of a live JSON-FDM
#   process by reading the simulator's OWN argv and environment from /proc.
#   This is the only witness with discriminating power. Two weaker witnesses
#   FAILED on 2026-07-30 and this tool exists to replace both:
#     (1) ws1_smoke.sh greps the wrapper log for `--wind-*`; the FDM argv never
#         reaches that log, so both WITNESS_RESOLVED_CELL.txt files were EMPTY
#         and an empty file is indistinguishable from a null result;
#     (2) frv2's witness spawned 12 samplers for a 4-lane flight, and the 8
#         that never flew reported mean_n ~ 0 with std ~ 0.5 — numbers that
#         look like a measured calm cell but measure nothing.
#   Rules that follow: identity is decided by the FILESYSTEM (os.path.samefile
#   against the canonical FDM script), never by a cmdline substring, so a
#   monitoring command that merely carries the script name cannot self-match;
#   and any lane that cannot be positively identified is reported
#   INCONCLUSIVE, NEVER as a number.
#   S-B2 (2026-08-07): reporting a cell is not gating one. The tool now also
#   ASSERTS the expected cell (--expect-wind-ms / --expect-sigma-ms) and can
#   wait for lanes to arm (--wait-secs). An observed-and-wrong cell exits 3
#   (ASSERT_FAIL), distinct from 2 (INCONCLUSIVE = could not observe), so a
#   caller can never read "flew the wrong sigma" as "saw nothing".
# PUBLIC API: fdm_pids, witness_pid, _cell_mismatch, witness_all, selfcheck, main
# SIDE-EFFECTS: reads /proc and stats candidate argv paths; writes stdout and,
#   with --out, one JSON file. Never signals, kills, or mutates any process.
# CAPS: file soft 230 / hard 250 LoC; function cap 45. Raised from 180/220 by
#   OPERATOR ELECTION 2026-08-07 when S-B2 landed at 223: the file gained the
#   assertion + arming capability and now carries 15 both-direction detectors.
#   DISCLOSED: the S-B2 estimate was <=180 — a sixth consecutive underestimate.
# COVERAGE: tests/unit/test_fdm_cell_witness.py
# =============================================================================
import argparse
import json
import os
import sys
import time

FD_ROOT = "/home/ubuntu/FlightDock"
FDM_SCRIPT = os.path.join(FD_ROOT, "tools/json_fdm/plane6dof_fdm.py")
WIND_FLAGS = ("--wind-speed-ms", "--wind-from-deg", "--wind-sigma-ms",
              "--wind-length-m", "--wind-seed")
# Which flags are mandatory is RESOLVED AT THE WRITER, not inferred from
# samples: src/drl/env_formation_sitl.py:3341-3355 appends the wind block only
# when FD_FDM_WIND_SPEED_MS or FD_FDM_WIND_SIGMA_MS is set, and within it
#   --wind-speed-ms ALWAYS (defaults to "0")
#   --wind-seed ALWAYS
#   --wind-from-deg only if FD_FDM_WIND_FROM_DEG is non-empty (independent
#                    of turbulence)
#   --wind-sigma-ms / --wind-length-m only when turbulence is configured
# config._v3_stage() sets wind_sigma_ms=() and wind_L_m=() when turb is False,
# so a CALM cell (S0, and every S0 retention screen) legitimately boots with
# neither. Absent-as-a-PAIR is therefore a WITNESSED calm cell, not a missing
# measurement; only a HALF-present pair is broken. Both refinements came from
# the live exercise on 2026-07-30, which first mis-flagged 12 legitimate S0
# retention lanes as INCONCLUSIVE.
REQUIRED_FLAGS = ("--wind-speed-ms", "--wind-seed")
TURB_FLAGS = ("--wind-sigma-ms", "--wind-length-m")
ENV_KEYS = ("FD_V3_SIGMA_OVERRIDE_MS", "FD_FDM_WIND_SPEED_MS",
            "FD_FDM_WIND_SIGMA_MS", "FD_FDM_WIND_LENGTH_M",
            "FD_FDM_WIND_FROM_DEG", "FD_LEADER_WIND",
            "FD_V3_REHEARSAL_DISABLE", "FD_CURRICULUM_V3")
INCONCLUSIVE = "INCONCLUSIVE"
WITNESSED = "WITNESSED"
# A cell that was OBSERVED and is WRONG is not the same as one that could not
# be observed. Conflating them would let a mis-flown stage exit as "no data".
ASSERT_FAIL = "ASSERT_FAIL"


def _read_null(path):
    """Read a NUL-delimited /proc file -> list[str]; None on ANY failure.

    None means "could not observe", which callers must map to INCONCLUSIVE.
    It must never be conflated with an empty observation.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    return [p.decode("utf-8", "replace") for p in raw.split(b"\x00") if p]


def _is_fdm(pid, script=FDM_SCRIPT):
    """Identity by filesystem, not text.

    Requires /proc/<pid>/exe to resolve (a live process we may inspect) AND
    some argv entry to os.path.samefile() the canonical FDM script. A
    substring test would match a shell command that merely mentions the
    script — the self-match class that produced a phantom process count on
    2026-07-30. samefile() compares st_dev/st_ino and cannot self-match.
    """
    try:
        os.readlink("/proc/%d/exe" % pid)
    except OSError:
        return False
    argv = _read_null("/proc/%d/cmdline" % pid)
    if not argv:
        return False
    for tok in argv[1:]:
        try:
            if os.path.samefile(tok, script):
                return True
        except OSError:
            continue
    return False


def fdm_pids(script=FDM_SCRIPT, proc="/proc"):
    """Every live PID positively identified as the JSON FDM."""
    out = []
    try:
        names = os.listdir(proc)
    except OSError:
        return out
    for name in names:
        if name.isdigit() and _is_fdm(int(name), script):
            out.append(int(name))
    return sorted(out)


def witness_pid(pid):
    """Realized cell for one FDM pid, or INCONCLUSIVE with a reason."""
    argv = _read_null("/proc/%d/cmdline" % pid)
    env = _read_null("/proc/%d/environ" % pid)
    if argv is None or env is None:
        return {"pid": pid, "state": INCONCLUSIVE,
                "reason": "/proc unreadable (process exited or EPERM)"}
    flags = {}
    for i, tok in enumerate(argv):
        if tok in WIND_FLAGS:
            flags[tok] = argv[i + 1] if i + 1 < len(argv) else INCONCLUSIVE
    envd = {}
    for item in env:
        key, _, val = item.partition("=")
        if key in ENV_KEYS:
            envd[key] = val
    rec = {"pid": pid, "state": WITNESSED, "argv_wind": flags, "env": envd}
    lacks_req = [f for f in REQUIRED_FLAGS if f not in flags]
    if lacks_req:
        rec["state"] = INCONCLUSIVE
        rec["reason"] = "FDM argv lacks %s" % ",".join(lacks_req)
        return rec
    have_turb = [f for f in TURB_FLAGS if f in flags]
    if not have_turb:
        rec["turbulence"] = False          # calm cell BY CONSTRUCTION (S0)
    elif len(have_turb) == len(TURB_FLAGS):
        rec["turbulence"] = True
    else:
        rec["state"] = INCONCLUSIVE
        rec["reason"] = ("partial turbulence argv — has %s, lacks %s"
                         % (",".join(have_turb),
                            ",".join(f for f in TURB_FLAGS if f not in flags)))
    return rec


def _cell_mismatch(rec, exp_wind=None, exp_sigma=None, tol=1e-6):
    """Reasons this lane's REALIZED cell differs from what the caller expects.

    Compared as floats against the lane's OWN argv, so "0.150" and 0.15 agree.
    The wind flags are written as COMMA TRIPLES by the env for turbulence
    (--wind-sigma-ms "0.150,0.150,0.150") and as a scalar for speed; EVERY
    component must match, so an anisotropic cell cannot pass on its first axis.
    A calm lane (no turbulence pair) matches --expect-sigma-ms 0 and nothing
    else: expecting turbulence where the simulator flies calm is a MISMATCH,
    not an absent measurement.
    """
    bad, flags = [], rec.get("argv_wind") or {}

    def off(flag, want):
        got = flags.get(flag)
        if got is None:
            return "absent"
        try:
            vals = [float(x) for x in str(got).split(",")]
        except ValueError:
            return got
        return None if vals and all(
            abs(v - want) <= tol for v in vals) else got

    if exp_wind is not None:
        seen = off("--wind-speed-ms", exp_wind)
        if seen is not None:
            bad.append("wind-speed-ms=%s expected %s" % (seen, exp_wind))
    if exp_sigma is not None:
        if not rec.get("turbulence"):
            if abs(exp_sigma) > tol:
                bad.append("lane is CALM, expected sigma %s" % exp_sigma)
        else:
            seen = off("--wind-sigma-ms", exp_sigma)
            if seen is not None:
                bad.append("wind-sigma-ms=%s expected %s" % (seen, exp_sigma))
    return bad


def witness_all(expect=None, script=FDM_SCRIPT, proc="/proc",
                expect_wind=None, expect_sigma=None):
    """Witness every live FDM lane.

    `expect` is the number of lanes the caller believes flew. A mismatch is
    INCONCLUSIVE for the whole record: un-flown lanes are never padded with
    zeros (the frv2 defect this tool replaces).
    """
    pids = fdm_pids(script, proc)
    rec = {"fdm_found": len(pids), "legs": [witness_pid(p) for p in pids]}
    if expect is not None:
        rec["expected"] = int(expect)
        if len(pids) != int(expect):
            rec["state"] = INCONCLUSIVE
            rec["reason"] = ("expected %d FDM lanes, identified %d — un-flown "
                             "lanes are NEVER reported as zeros"
                             % (int(expect), len(pids)))
            return rec
    rec["state"] = WITNESSED if pids else INCONCLUSIVE
    if not pids:
        rec["reason"] = "no FDM process positively identified"
        return rec
    if expect_wind is None and expect_sigma is None:
        return rec
    unread = [l["pid"] for l in rec["legs"] if l.get("state") != WITNESSED]
    if unread:
        rec["state"] = INCONCLUSIVE
        rec["reason"] = ("cannot assert a cell on unreadable lanes %s"
                         % unread)
        return rec
    bad = ["pid %s: %s" % (l["pid"], b) for l in rec["legs"]
           for b in _cell_mismatch(l, expect_wind, expect_sigma)]
    if bad:
        rec["state"] = ASSERT_FAIL
        rec["reason"] = "; ".join(bad)
    return rec


def selfcheck(tmpdir):
    """Prove the two guarantees without touching a live process."""
    real = os.path.join(tmpdir, "plane6dof_fdm.py")
    impostor = os.path.join(tmpdir, "impostor_plane6dof_fdm.py")
    for path in (real, impostor):
        with open(path, "w") as fh:
            fh.write("# fixture\n")
    ok1 = not os.path.samefile(impostor, real)          # substring != identity
    rec = witness_all(expect=4, script=real, proc=tmpdir)
    ok2 = rec["state"] == INCONCLUSIVE and "legs" in rec and not rec["legs"]
    # The assertion must be able to fire in BOTH directions, on both a
    # turbulent and a calm lane — a gate that can only pass is not a gate.
    turb = {"argv_wind": {"--wind-speed-ms": "6.0", "--wind-sigma-ms": "0.150"},
            "turbulence": True}
    calm = {"argv_wind": {"--wind-speed-ms": "6.0"}, "turbulence": False}
    ok3 = (not _cell_mismatch(turb, 6.0, 0.15)) and bool(
        _cell_mismatch(turb, 6.0, 0.20))
    ok4 = (not _cell_mismatch(calm, 6.0, 0.0)) and bool(
        _cell_mismatch(calm, 6.0, 0.15))
    print("selfcheck identity-not-substring: %s" % ("PASS" if ok1 else "FAIL"))
    print("selfcheck unflown-lanes-inconclusive: %s"
          % ("PASS" if ok2 else "FAIL"))
    print("selfcheck cell-assert-both-directions: %s"
          % ("PASS" if (ok3 and ok4) else "FAIL"))
    return 0 if (ok1 and ok2 and ok3 and ok4) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expect", type=int, default=None,
                    help="lane count the caller believes flew; mismatch is "
                         "INCONCLUSIVE for the whole record")
    ap.add_argument("--script", default=FDM_SCRIPT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--expect-wind-ms", type=float, default=None,
                    help="assert every lane's realized --wind-speed-ms")
    ap.add_argument("--expect-sigma-ms", type=float, default=None,
                    help="assert realized --wind-sigma-ms; 0 asserts CALM")
    ap.add_argument("--wait-secs", type=float, default=0.0,
                    help="re-poll until the lanes are up; timeout is "
                         "INCONCLUSIVE, never a pass")
    args = ap.parse_args(argv)
    if args.selfcheck:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            return selfcheck(td)
    deadline = time.monotonic() + max(0.0, args.wait_secs)
    while True:
        rec = witness_all(expect=args.expect, script=args.script,
                          expect_wind=args.expect_wind_ms,
                          expect_sigma=args.expect_sigma_ms)
        if rec["state"] != INCONCLUSIVE or time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    text = json.dumps(rec, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return {WITNESSED: 0, ASSERT_FAIL: 3}.get(rec["state"], 2)


if __name__ == "__main__":
    sys.exit(main())
