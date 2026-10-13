#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: assert the EVAL env contract ONCE, before a wave commits any
#   leg. Exists because on 2026-08-20 wave 1 launched and all 76 legs failed
#   individually with NO_PAIRED_CSV; the real cause was three D7_REQUIRED
#   warmup knobs the driver's frozen parent predates (ARMED-ELECT-737,
#   GATETOL-746, both signed 2026-08-19). The contract worked -- fail-closed,
#   nothing banked -- but it fired 76 times, AFTER the wave had committed.
#
# WHY NOT A LIST OF KNOBS IN THE DRIVER. Hand-listing is the SEEDALLOC-759
#   defect class: a hardcoded list cannot see a requirement signed after it was
#   written, which is exactly what happened. This asks the contract itself, so
#   a knob signed next month is covered without editing anything here.
#
#   A dryrun never reaches eval_drl_checkpoint, so nothing else in the launch
#   path exercises the contract before seeds are committed.
#
# PUBLIC API: main() -> int
# SIDE-EFFECTS: imports src.drl.env_contract and reads os.environ. Prints one
#   line on success, the violation banner on failure. Writes nothing.
# EXIT: 0 contract clean · 12 violation (caller must HALT the wave, never
#   continue) · 3 the contract module itself could not be imported.
# CAPS: file soft 45 / hard 60 code-lines; function cap 35.
# COVERAGE: exercised BOTH directions by the driver-level lanes recorded in the
#   ARMBDESC-B commit: clean -> 0; knob unset -> 12; knob set to a WRONG VALUE
#   -> 12; R5 align_center dropped -> 12.
# =============================================================================
"""Assert the EVAL env contract before a wave commits. See header above."""
import os
import sys

FD = os.environ.get("FLIGHTDOCK_ROOT", "/home/ubuntu/FlightDock")


def main():
    src = os.path.join(FD, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from drl import env_contract as ec
    except Exception as e:
        sys.stderr.write("PREFLIGHT: cannot import env_contract: %s\n" % e)
        return 3
    try:
        ec.assert_contract(ec.EVAL)
    except Exception as e:                    # EnvContractViolation and kin
        sys.stderr.write("%s\n" % e)
        return 12
    print("[env_contract] EVAL role clean before leg 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
