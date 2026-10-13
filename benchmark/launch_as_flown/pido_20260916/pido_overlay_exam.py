#!/usr/bin/env python3
# PIDO TWIN of analysis/cs600_20260903T0230Z/cs600_overlay_exam.py (parent NOT edited) with ONE declared delta: the declared
#   knobs are a SET, not a single flag. env PIDO_KNOBFLAG carries "<flag> <value> [<flag> <value> ...]" (written by the fire
#   script from the manifest gate's KNOBFLAG line: "--vert-offset-down-m 0.25 --action-scales 0.00"); the argv must carry
#   EXACTLY that set -- a missing declared knob, an extra knob, or a different value refuses rc 14. The knob vocabulary adds
#   --action-scales (the PIDO-20260916 lever). PREFLIGHT re-points at bench600_preflight.sh exactly as the parent does.
# PUBLIC API: PIDO_KNOBFLAG="<flags>" python pido_overlay_exam.py <every crrl_overlay_exam.py argument>
# SIDE-EFFECTS: exactly the overlay parent's. SIZE: soft 30 / hard 45 code-lines; function cap 20.
# COVERAGE: pido_selfcheck.py -- G2 undeclared knob 14; G3 one declared knob absent 14; G4 declared pair plus an extra 14;
#   G5 the declared pair on a clean argv passes to the parent (dry-run rc 0, witness carries action_scales 0.0).
import os, sys
FD = "/home/ubuntu/FlightDock"
OV_DIR = os.path.join(FD, "analysis/guardspike818_20260825T1700Z")
PREFLIGHT = os.path.join(FD, "analysis/bench600_20260902T0310Z/bench600_preflight.sh")
KNOBS = ("--clamp-behind-positive-ms", "--forward-bias-ms", "--vert-offset-down-m", "--beta", "--action-scales")

def load():
    sys.path.insert(0, OV_DIR)
    import crrl_overlay_exam as ov
    ov.PREFLIGHT = PREFLIGHT
    return ov

def knob_pairs(argv):
    return {argv[i]: (argv[i + 1] if i + 1 < len(argv) else None) for i, a in enumerate(argv) if a in KNOBS}

def as_floats(d):
    return {k: (float(v) if v not in (None, "") else None) for k, v in d.items()}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    want = knob_pairs(os.environ.get("PIDO_KNOBFLAG", "").split())
    got = knob_pairs(argv)
    if as_floats(got) != as_floats(want) or None in as_floats(got).values():
        print("PIDO REFUSED rc 14: argv knobs %s != declared %s" % (got, want)); return 14
    return load().main(argv)

if __name__ == "__main__":
    sys.exit(main())
