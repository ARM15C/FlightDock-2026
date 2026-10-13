#!/usr/bin/env python3
# =============================================================================

#   preflight that RUN_ORDER.md step 0 used to run as an inline `python -c`: (1) process census from /proc -- zero
#   simulators, zero MAVProxy, zero pytest on this box (never kills anything); (2) free disk on /home/ubuntu at or above
#   the floor; (3) all twelve corpora resolve to exactly one exam tree (rescore_run.resolve, the same check the run
#   itself enforces). Prints one line per check and a final PRECHECK PASS / PRECHECK FAIL line.
# PUBLIC API: census() -> dict; disk_free_gb(path) -> float; main(argv=None) -> int.
#   CLI: [--min-free-gb 20] [--selftest]. rc 0 PASS, rc 1 FAIL.
# SIDE-EFFECTS: none. Reads /proc and the corpora directory listing only.
# SIZE: soft 90 / hard 120 code-LoC; function cap 30.
# COVERAGE: --selftest: the census pattern matcher counts a simulator and a pytest command line, ignores this
#   script's own command line, and the corpus table holds twelve entries.
# =============================================================================
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore_run as R  # noqa: E402

#: substrings that identify a process which must not be running during the re-score.
PATTERNS = {"simulator": ("arduplane", "arducopter", "sim_vehicle", "ardupilot/build"),
            "mavproxy": ("mavproxy",), "pytest": ("pytest",)}


def classify(cmdline):
    """The census class of one command line, or None. This script's own command line never counts."""
    low = cmdline.lower()
    if "precheck.py" in low:
        return None
    for name, pats in PATTERNS.items():
        if any(p in low for p in pats):
            return name
    return None


def census():
    counts = {k: 0 for k in PATTERNS}
    me = os.getpid()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == me:
            continue
        try:
            raw = open("/proc/%s/cmdline" % pid, "rb").read()
        except OSError:
            continue
        kind = classify(raw.replace(b"\0", b" ").decode("utf-8", "replace"))
        if kind:
            counts[kind] += 1
    return counts


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def selftest():
    ok = classify("/x/arduplane --model quadplane") == "simulator" and classify("python -m pytest tests") == "pytest"
    ok = ok and classify("python precheck.py --selftest") is None and classify("bash -l") is None
    ok = ok and len(R.CORPORA) == 12
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ok = True
    c = census()
    print("processes:", c)
    ok = ok and not any(c.values())
    free = disk_free_gb("/home/ubuntu")
    print("free disk on /home/ubuntu: %.1f GB (floor %.1f GB)" % (free, a.min_free_gb))
    ok = ok and free >= a.min_free_gb
    for tag in R.CORPORA:
        try:
            print("corpus %-8s -> %s" % (tag, R.resolve(tag)))
        except SystemExit as e:
            print("corpus %-8s -> %s" % (tag, e))
            ok = False
    print("PRECHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
