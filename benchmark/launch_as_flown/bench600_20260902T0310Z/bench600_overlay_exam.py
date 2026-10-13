#!/usr/bin/env python3
# RESPONSIBILITY: the BENCH600 exam driver -- the flown crrl_overlay_exam.py (guardspike818, the
#   KNOBSWEEP/H3C050/VOFFQ lineage) with ONE declared delta: PREFLIGHT points at
#   bench600_preflight.sh, so the wave that flies is the twin carrying the S0 calm-tier hook.
#   Everything else (patch-and-revert of the floor, the only-declared diff, OVERLAY_WITNESS.json,
#   the ledger-named custody refusal, --dry-run) is the parent's own code, imported and called.
#   This is the pattern the parent's OWN live fixtures use (crrl_overlay_limbs_live.py:34 sets
#   ov.PREFLIGHT); it is safe here because PREFLIGHT is read in the PARENT process when the argv
#   is built (exam_argv) and then exec'd as a subprocess by path -- unlike the wave's slot
#   fan-out, which is why the wave itself had to be a file twin.
# PUBLIC API: python bench600_overlay_exam.py <every crrl_overlay_exam.py argument>
# SIDE-EFFECTS: exactly the parent's. No knob flag is ever forwarded by this dispatch (E1: no
#   exam knob); a knob on the argv is REFUSED here before the parent parses it.
# SIZE: soft 25 / hard 40 code-lines; function cap 20.
# COVERAGE: bench600_selfcheck.py -- PREFLIGHT re-pointed (read back), knob argv refused rc 14,
#   --dry-run of the exact live argv on the ARMBLAD025 table rc 0 with the config revert witnessed.
import os
import sys

FD = "/home/ubuntu/FlightDock"
OV_DIR = os.path.join(FD, "analysis/guardspike818_20260825T1700Z")
PREFLIGHT = os.path.join(FD, "analysis/bench600_20260902T0310Z/bench600_preflight.sh")
KNOBS = ("--clamp-behind-positive-ms", "--forward-bias-ms", "--vert-offset-down-m", "--beta")


def load():
    sys.path.insert(0, OV_DIR)
    import crrl_overlay_exam as ov
    ov.PREFLIGHT = PREFLIGHT
    return ov


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    bad = [a for a in argv if a.split("=")[0] in KNOBS]
    if bad:
        print("BENCH600 REFUSED rc 14: election E1 is 'no exam knob'; argv carries %s" % bad)
        return 14
    return load().main(argv)


if __name__ == "__main__":
    sys.exit(main())
