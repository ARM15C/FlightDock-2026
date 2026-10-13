#!/usr/bin/env python3
# PIDO TWIN of analysis/s1abl_20260915T0400Z/s1abl_sig_gate.py with ONE declared delta: the run token prefix is
#   PIDO- (the parent is NOT edited). One prereg covers both runs, signed once with run token ALL. (parent header follows)
# RESPONSIBILITY: decide whether ONE PIDO prereg is genuinely SIGNED, and refuse if it is not.
#   Two conditions, both required: (1) a NON-SPACE character immediately after the run token, which
#   the underscore placeholder cannot satisfy; (2) a 64-hex sha on the line that EQUALS this run's
#   banked pre-signature blob sha in PREREG_SHAS.json. So the signature is bound to the blob it signed,
#   not to a prefix anyone could leave lying in a template.
#   Fail-closed everywhere: an absent or unreadable bank REFUSES; an unknown run REFUSES.
# PUBLIC API: SIG_RX(run) -> compiled regex; check(prereg, run, shas) -> (rc, msg); main(argv) -> int
#   rc 0 signed AND bound to the banked blob sha. 16 not validly signed (absent line, placeholder,
#   no sha, or a sha that is not this run's). 17 the sha bank is absent/unreadable or lacks the run.
# SIDE-EFFECTS: none. Reads two files, writes nothing, spawns nothing, imports neither src/ nor torch.
# SIZE: soft 40 / hard 55 code-lines; function cap 25.
# COVERAGE: pido_selfcheck.py -- F7a unsigned placeholder MUST be 16 (never a range), F7b a correctly
#   signed fixture MUST be 0, F7c a signature naming the WRONG sha MUST be 16, F8 absent prereg MUST be 16,
#   F7d absent sha bank MUST be 17.
import argparse
import json
import os
import re
import sys


def SIG_RX(run):
    """Column 0, the run token, then a NON-SPACE (kills the '- ____' placeholder), then a 64-hex sha."""
    return re.compile(r"^OPERATOR-SIGNATURE: PIDO-%s-\S.*?\b([0-9a-f]{64})\b" % re.escape(run), re.M)


def banked_sha(shas, run):
    try:
        with open(shas, encoding="utf-8") as fh:
            runs = json.load(fh)["runs"]
    except (OSError, KeyError, ValueError) as e:
        return None, "sha bank absent/unreadable at %s (%s)" % (shas, type(e).__name__)
    if run not in runs:
        return None, "sha bank has no entry for run %r (has %s)" % (run, sorted(runs))
    return runs[run], None


def check(prereg, run, shas):
    want, why = banked_sha(shas, run)
    if want is None:
        return 17, "GATE D7: " + why
    if not os.path.isfile(prereg):
        return 16, "GATE D7: prereg absent at %s -- nothing flies unsigned" % prereg
    with open(prereg, encoding="utf-8") as fh:
        found = SIG_RX(run).findall(fh.read())
    if not found:
        return 16, ("GATE D7: %s carries no VALID signature for PIDO-%s. A column-0 line naming "
                    "the 64-hex prereg sha is required; the unsigned placeholder does not count."
                    % (os.path.basename(prereg), run))
    if want not in found:
        return 16, ("GATE D7: the signature on %s names sha %s.., but run %s was banked at %s.. -- "
                    "the signature does not bind this blob" % (os.path.basename(prereg), found[0][:12], run, want[:12]))
    return 0, "SIGNED: PIDO-%s bound to prereg sha %s.." % (run, want[:12])


def main(argv=None):
    ap = argparse.ArgumentParser(description="PIDO prereg signature gate (sha-bound)")
    for f in ("--prereg", "--run", "--shas"):
        ap.add_argument(f, required=True)
    a = ap.parse_args(argv)
    rc, msg = check(a.prereg, a.run, a.shas)
    print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
