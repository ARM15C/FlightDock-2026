#!/usr/bin/env python3
"""armblad035_venue_limb -- S9: EXECUTE the EVAL env contract against the venue preamble.

RESPONSIBILITY
  COPY of baseline025_venue_limb.py with one path changed (L2: the parent flew as-is).
  This limb exists because of a measured failure, not a hypothetical one. The parent's first launch
  attempt (2026-08-26 19:32:29Z) inlined the venue into the driver and omitted
  `source scripts/eval_venue_env.sh` plus four warmup/safety knobs. All 24 slots HALTED with
  "EVAL env contract violated before leg 1" and 6 violations. Nothing was burned -- zero SITL,
  zero seeds, zero steps, wall 0 s -- because preflight_contract.py is fail-closed and runs
  before leg 1. But NOTHING IN THE TEST SUITE COULD HAVE CAUGHT IT: every limb up to that point
  read files and built tables, and the venue is an ENVIRONMENT, which only exists when something
  sources it.

  So S9 sources the preamble in a SUBSHELL and runs the real contract checker inside it. A text
  assertion ("the driver mentions eval_venue_env.sh") would pass on a preamble that sourced the
  file and then unset one of its variables; executing the contract cannot be fooled that way.

  BOTH DIRECTIONS. The positive: sourcing the preamble satisfies the contract. The negative: the
  same subshell with ONE required variable unset must REFUSE -- otherwise the limb is testing
  that a checker exists, not that it works (RB-20: a check that cannot fail is not a gate).

PUBLIC API: s9_venue_satisfies_eval_contract() -> (id, ok, detail)
SIDE-EFFECTS: spawns two short-lived bash subshells that export variables and run
  preflight_contract.py READ-ONLY. Writes nothing, flies nothing, contacts no box.
CAPS: soft 45 / hard 65 code-lines; function cap 30.
COVERAGE: registered by armblad035_selfcheck.all_limbs(), which the unit test executes.
"""
import os
import subprocess

FD = "/home/ubuntu/FlightDock"
HERE = os.path.dirname(os.path.abspath(__file__))
PREAMBLE = os.path.join(HERE, "armblad035_venue_preamble.sh")
CONTRACT = os.path.join(FD, "analysis/armbdesc_20260820T1300Z/preflight_contract.py")
PY = os.path.join(FD, "flightdock_native_venv/bin/python")
#: One variable the EVAL contract requires and the generated venue file supplies. Unsetting it
#: is the negative control: the contract MUST refuse.
CANARY = "FD_DEMOTE_TERMINALS"


def _run(extra):
    cmd = 'set +u; source "%s" >/dev/null 2>&1; %s "%s" "%s"' % (PREAMBLE, extra, PY, CONTRACT)
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=FD)


def s9_venue_satisfies_eval_contract():
    ok = _run("")
    bad = _run("unset %s;" % CANARY)
    detail = ("preamble satisfies the EVAL contract (rc %d); the same venue with %s unset "
              "REFUSES (rc %d)" % (ok.returncode, CANARY, bad.returncode))
    if ok.returncode != 0:
        detail = "PREAMBLE FAILS THE CONTRACT: %s" % (ok.stdout + ok.stderr)[-300:]
    return ("S9_venue_EXECUTES_the_eval_contract",
            ok.returncode == 0 and bad.returncode != 0, detail)


if __name__ == "__main__":
    i, o, d = s9_venue_satisfies_eval_contract()
    print(" %-42s %s %s" % (i, "PASS" if o else "FAIL", d))
    raise SystemExit(0 if o else 1)
