#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: R-2 — the per-leg REALIZED-σ gate that A-3.2 requires, and
#   the durable stream it must read. Two jobs, both load-bearing:
#
#   (1) HARVEST. `plane6dof_fdm.py` opens --log-jsonl in mode "w" and the env
#       relaunches the FDM from inside reset(), so logs/fdm_env{k}.jsonl is
#       TRUNCATED on every restart and a post-leg reader sees only the LAST
#       life. This copies the in-window slice to a per-leg file BEFORE it can
#       be destroyed. Same shape as wse_widen_driver.sh:160-182, which already
#       emits the exact {fdm_timestamp, wind_ned} schema leg_sigma parses.
#
#   (2) GATE. The flown F6 witness reported σ as "(info)", East-only, and its
#       caller swallowed every rc with `|| log`. That is how 210 ARM-B legs flew
#       under-delivered σ printing OK. This REFUSES.
#
#   NOT A SECOND ESTIMATOR. The σ maths is eval_sigma_witness.gate ->
#   s1417_sigma_witness.leg_sigma under a sha256 pin. Four copies of this
#   arithmetic already exist in the tree; this adds none.
# A11-2: calm tier in
#   the sibling leg_sigma_calm.py. When --paired-csv shows the leg's OWN
#   resolved_wind_max_ms == 0, the tier-1 band comparison is replaced by the
#   positive CALM_CONFIRMED assertion; the harvester distinguishes
#   "parsed-but-wind-free" (windless: calm evidence) from "did not parse"
#   (malformed) — the change §A-11.2 owed at implementation.
# PUBLIC API: harvest(src, dst, t0, t1) -> dict · gate_leg(leg_dir, env_id,
#   commanded, band, t0, t1, logs_dir=None, paired_csv=None) -> dict ·
#   selfcheck() -> int · main(argv) -> int.
#   EXIT CODES (shared with tools/fdm_cell_witness.py so one caller branch
#   handles both): 0 MEASURED_PASS · 2 UNSAMPLED|INCONCLUSIVE · 3 ASSERT_FAIL.
#   **2 AND 3 ARE BOTH REFUSALS** — A-3.2: "UNSAMPLED is a refusal, never a
#   pass." A sha-pin failure is a refusal too, never a skip.
# SIDE-EFFECTS: reads logs/fdm_env{N}.jsonl; WRITES <leg>/fdm_witness.jsonl and
#   <leg>/sigma_gate.json. No subprocess, no network, no signals, no SITL.
# CAPS: file soft 200 / hard 260 code-LoC; function LoC cap 40.
# COVERAGE: tests/unit/test_leg_sigma_gate.py (R-2 dispatch) + the caps pin.
# =============================================================================
import argparse
import importlib.util
import json
import os
import sys
import tempfile

WITNESS = ("/home/ubuntu/FlightDock/analysis/evalsigma_20260807T1130Z/"
           "eval_sigma_witness.py")
WITNESS_NAME = "fdm_witness.jsonl"
VERDICT_NAME = "sigma_gate.json"
T0_DEFAULT, T1_DEFAULT = 60.0, 360.0          # wse_widen_driver.sh:71
BAND_DEFAULT = 0.03                            # A-3.2, BENCHMARK-A3-20260807
RC_PASS, RC_INCONCLUSIVE, RC_ASSERT = 0, 2, 3


def _witness(path=None):
    """Import the eval-side witness. A pin failure must REFUSE, not skip."""
    p = path or WITNESS
    spec = importlib.util.spec_from_file_location("_evalsigma_for_gate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                # SystemExit on sha drift
    return mod


def stream_path(env_id, logs_dir=None):
    """Deterministic: env_formation_sitl.py:3625-3638 writes exactly here.

    Keyed on the env_id the driver itself passed as --env-id, so parallel
    slots sharing one logs/ dir stay attributable with no glob and no mtime
    guessing.
    """
    base = logs_dir or os.path.join("/home/ubuntu/FlightDock", "logs")
    return os.path.join(base, "fdm_env%s.jsonl" % env_id)


def harvest(src, dst, t0=T0_DEFAULT, t1=T1_DEFAULT):
    """Copy the in-window slice to a DURABLE per-leg file.

    Never raises: a harvest failure must surface as a refusal downstream, not
    as an exception that aborts a leg the driver would otherwise have scored.
    Malformed lines are skipped and counted, never guessed at.
    """
    kept = bad = windless = 0
    t_first = t_last = None
    try:
        with open(src, errors="replace") as fh, open(dst, "w") as out:
            for line in fh:
                try:
                    rec = json.loads(line)
                    t = float(rec["fdm_timestamp"])
                except (ValueError, KeyError, TypeError):
                    bad += 1
                    continue
                w = rec.get("wind_ned")
                if w is None:
                    # A11-2: parsed as an FDM record but carries NO wind
                    # field — calm EVIDENCE, distinct from malformation.
                    windless += 1 if t0 <= t < t1 else 0
                    continue
                if not (t0 <= t < t1) or not isinstance(w, list) or len(w) != 3:
                    continue
                out.write(json.dumps({"fdm_timestamp": t, "wind_ned": w}) + "\n")
                kept += 1
                t_first = t if t_first is None else t_first
                t_last = t
    except OSError as exc:
        return {"kept": 0, "malformed": bad, "windless": windless,
                "why": "unreadable: %s" % exc, "src": src, "dst": dst}
    return {"kept": kept, "malformed": bad, "windless": windless,
            "t_first": t_first, "t_last": t_last, "src": src, "dst": dst,
            "why": ""}


def _commanded(value):
    """Accept '0.30' or '0.30,0.30,0.30'; always compared PER AXIS."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = [p for p in str(value).split(",") if p.strip()]
    return float(parts[0]) if len(parts) == 1 else [float(p) for p in parts]


def _calm():
    """Import the A11-2 sibling from this file's own directory."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "leg_sigma_calm.py")
    spec = importlib.util.spec_from_file_location("_leg_sigma_calm", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gate_leg(leg_dir, env_id, commanded, band=BAND_DEFAULT,
             t0=T0_DEFAULT, t1=T1_DEFAULT, logs_dir=None, _mod=None,
             paired_csv=None):
    """Harvest, then gate the DURABLE copy. Fail-closed at every branch."""
    os.makedirs(leg_dir, exist_ok=True)
    dst = os.path.join(leg_dir, WITNESS_NAME)
    hv = harvest(stream_path(env_id, logs_dir), dst, t0, t1)
    if paired_csv is not None:
        cm = _calm()
        cw = cm.resolved_wind_max(paired_csv)
        if cw is not None and cw == 0.0:      # positive, readable zero ONLY
            return cm.gate_calm(hv, dst)
    try:
        mod = _mod or _witness()
    except SystemExit as exc:                   # sha pin drift => REFUSE
        return {"state": "ASSERT_FAIL", "passed": False, "harvest": hv,
                "why": "witness pin refused: %s" % exc, "rc": RC_ASSERT}
    if hv["kept"] == 0:
        # No durable samples: distinguish "could not read" from "read, empty".
        state = "UNSAMPLED" if hv["why"].startswith("unreadable") \
            else "INCONCLUSIVE"
        return {"state": state, "passed": False, "harvest": hv,
                "why": hv["why"] or "no samples in window", "rc": RC_INCONCLUSIVE}
    realized = mod.realized_sigma_ned(dst, t0, t1)
    verdict = mod.gate(realized, _commanded(commanded), band)
    verdict["harvest"] = hv
    verdict["rc"] = (RC_PASS if verdict["passed"] else
                     RC_ASSERT if verdict["state"] == "ASSERT_FAIL"
                     else RC_INCONCLUSIVE)
    return verdict


def _write_fixture(path, sig, n=20000, hz=100.0, t_start=0.0, seed=7):
    """Deterministic AR(1) fixture in the FDM's own schema."""
    import math
    import random
    rng = random.Random(seed)
    a, prev = math.exp(-1.0 / (0.5 * hz)), [0.0, 0.0, 0.0]
    with open(path, "w") as fh:
        for k in range(n):
            row = []
            for i in range(3):
                step = math.sqrt(1.0 - a * a) * sig[i]
                prev[i] = a * prev[i] + step * rng.gauss(0.0, 1.0)
                row.append(prev[i])
            fh.write(json.dumps({"fdm_timestamp": t_start + k / hz,
                                 "wind_ned": row}) + "\n")


def _cases(td, mod):
    """Every direction stated; none assumed."""
    logs = os.path.join(td, "logs")
    os.makedirs(logs, exist_ok=True)
    _write_fixture(os.path.join(logs, "fdm_env3.jsonl"), (0.30, 0.30, 0.30),
                   n=40000, t_start=0.0)
    good = gate_leg(os.path.join(td, "legA"), 3, "0.30,0.30,0.30",
                    logs_dir=logs, _mod=mod)
    out = [("delivered 0.30 vs commanded 0.30 PASSES", good["state"],
            "MEASURED_PASS"),
           ("the durable copy was written", os.path.isfile(
               os.path.join(td, "legA", WITNESS_NAME)), True),
           ("harvest kept only the in-window slice",
            good["harvest"]["kept"] <= 30001, True),
           ("commanded 0.20 REFUSES the same leg (band 0.03)",
            gate_leg(os.path.join(td, "legB"), 3, "0.20", logs_dir=logs,
                     _mod=mod)["state"], "ASSERT_FAIL"),
           ("commanded 0.40 REFUSES it too (both directions)",
            gate_leg(os.path.join(td, "legC"), 3, "0.40", logs_dir=logs,
                     _mod=mod)["state"], "ASSERT_FAIL"),
           ("ABSENT stream is UNSAMPLED, never a pass",
            gate_leg(os.path.join(td, "legD"), 99, "0.30", logs_dir=logs,
                     _mod=mod)["state"], "UNSAMPLED"),
           ("UNSAMPLED rc is a REFUSAL (2), not 0",
            gate_leg(os.path.join(td, "legE"), 99, "0.30", logs_dir=logs,
                     _mod=mod)["rc"], RC_INCONCLUSIVE)]
    # A stream that exists but lies wholly OUTSIDE the window => INCONCLUSIVE,
    # which must stay DISTINCT from UNSAMPLED: "saw nothing" and "could not
    # read" are different facts and neither is a pass.
    _write_fixture(os.path.join(logs, "fdm_env4.jsonl"), (0.30, 0.30, 0.30),
                   n=500, t_start=900.0)
    off = gate_leg(os.path.join(td, "legF"), 4, "0.30", logs_dir=logs, _mod=mod)
    out += [("out-of-window stream is INCONCLUSIVE", off["state"],
             "INCONCLUSIVE"),
            ("INCONCLUSIVE != UNSAMPLED", off["state"] == "UNSAMPLED", False),
            ("neither ever passes", off["passed"], False),
            ("the REAL ARM-B under-delivery 0.2598 would REFUSE at band 0.02",
             mod.gate({"sigma_ned": [0.2598] * 3, "n": 999}, 0.300,
                      0.02)["state"], "ASSERT_FAIL"),
            ("the serial reference 0.2970 PASSES at band 0.02",
             mod.gate({"sigma_ned": [0.2970] * 3, "n": 999}, 0.300,
                      0.02)["state"], "MEASURED_PASS")]
    return out


def selfcheck():
    mod, bad = _witness(), 0
    td = tempfile.mkdtemp()
    cases = _cases(td, mod) + _calm().calm_cases(td, gate_leg, _write_fixture)
    for name, got, want in cases:
        ok = got == want
        print(" %s %s: %s want=%s" % ("OK " if ok else "BAD", name, got, want))
        bad += 0 if ok else 1
    print("SELFCHECK", "PASS" if not bad else "FAIL (%d)" % bad)
    return 0 if not bad else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description="R-2 per-leg realized-sigma gate")
    ap.add_argument("--leg")
    ap.add_argument("--env-id")
    ap.add_argument("--commanded")
    ap.add_argument("--band", type=float, default=BAND_DEFAULT)
    ap.add_argument("--t0", type=float, default=T0_DEFAULT)
    ap.add_argument("--t1", type=float, default=T1_DEFAULT)
    ap.add_argument("--logs-dir")
    ap.add_argument("--paired-csv", help="A11-2: the leg's OWN paired-channels"
                    " CSV; calm tier entered only on resolved_wind_max_ms==0")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck()
    if not a.leg or a.env_id is None or a.commanded is None:
        ap.error("--leg, --env-id and --commanded are required")
    rec = gate_leg(a.leg, a.env_id, a.commanded, a.band, a.t0, a.t1,
                   a.logs_dir, paired_csv=a.paired_csv)
    try:
        with open(os.path.join(a.leg, VERDICT_NAME), "w") as fh:
            json.dump(rec, fh, sort_keys=True, indent=1)
    except OSError as exc:
        print("verdict unwritable: %s" % exc, file=sys.stderr)
    print(json.dumps({k: v for k, v in rec.items() if k != "harvest"},
                     sort_keys=True))
    return rec["rc"]


if __name__ == "__main__":
    sys.exit(main())
