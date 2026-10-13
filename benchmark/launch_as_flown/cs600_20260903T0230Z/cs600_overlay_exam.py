#!/usr/bin/env python3
# CS600 TWIN of analysis/bench600_20260902T0310Z/bench600_overlay_exam.py with ONE declared delta: the knob rule is
#   "exactly the DECLARED flag and nothing else" instead of "no knob". The declared flag arrives in env CS600_KNOBFLAG
#   (empty for the parent-CAND-S run; "--vert-offset-down-m 0.25" for the CANDSSW-leg1 run), written by the fire script
#   from the manifest gate's KNOBFLAG line (empty, "--vert-offset-down-m 0.25", or "--clamp-behind-positive-ms 0.50").
#   Any knob token on the argv that is not the declared one refuses rc 14; a
#   declared flag that is ABSENT from the argv also refuses rc 14. PREFLIGHT re-points at bench600_preflight.sh exactly
#   as the parent does (the S0 calm-tier wave twin; policy-agnostic, reused by path). Neither parent is edited.
# PUBLIC API: CS600_KNOBFLAG="<flag or empty>" python cs600_overlay_exam.py <every crrl_overlay_exam.py argument>
# SIDE-EFFECTS: exactly the overlay parent's. SIZE: soft 30 / hard 45 code-lines; function cap 20.
# COVERAGE: cs600_selfcheck.py -- undeclared knob refused 14; declared-but-absent refused 14; empty declaration with a
#   clean argv passes to the parent (dry-run); declared voff present passes to the parent (dry-run).
import os, sys
FD = "/home/ubuntu/FlightDock"
OV_DIR = os.path.join(FD, "analysis/guardspike818_20260825T1700Z")
PREFLIGHT = os.path.join(FD, "analysis/bench600_20260902T0310Z/bench600_preflight.sh")
KNOBS = ("--clamp-behind-positive-ms", "--forward-bias-ms", "--vert-offset-down-m", "--beta")

def load():
    sys.path.insert(0, OV_DIR)
    import crrl_overlay_exam as ov
    ov.PREFLIGHT = PREFLIGHT
    return ov

def knob_pairs(argv):
    return {argv[i]: (argv[i + 1] if i + 1 < len(argv) else None) for i, a in enumerate(argv) if a in KNOBS}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    declared = os.environ.get("CS600_KNOBFLAG", "").split()
    want = {declared[0]: declared[1]} if len(declared) == 2 else {}
    got = knob_pairs(argv)
    if {k: (float(v) if v else None) for k, v in got.items()} != {k: float(v) for k, v in want.items()}:
        print("CS600 REFUSED rc 14: argv knobs %s != declared %s" % (got, want)); return 14
    return load().main(argv)

if __name__ == "__main__":
    sys.exit(main())
