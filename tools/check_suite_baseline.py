#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: gate a pytest run against the pre-existing-failure baseline
#   by TEST-ID SET, not by count. Built for T-38 (2026-07-23) after the count
#   form was found to be the only comparison anyone had actually been running:
#   "38 == 38" passes unchanged when one test is repaired and a different one
#   breaks, which is the same weak-gate pattern this project has now hit four
#   times in one session.
#
#   THREE OUTCOMES, and only one of them is green:
#     NEW failures -> RED. Test IDs failing now that are not in the
#                            baseline. This is the regression signal.
#     FIXED failures -> AMBER. Baseline IDs no longer failing. Not a
#                            regression, but the baseline is now STALE and must
#                            be re-captured, so it is reported loudly rather
#                            than silently tolerated.
#     exact set match -> GREEN.
#
# PUBLIC API: compare(baseline_ids, current_ids) -> dict; main(argv=None) -> int
#   CLI: check_suite_baseline.py --baseline PATH [--run | --from-file PATH]
#   Exit 0 GREEN, 1 NEW failures, 2 baseline stale (fixed-only), 3 usage/IO.
# SIDE-EFFECTS: with --run, invokes pytest as a subprocess. Reads only.
# CAPS: file soft 135 / hard 155 lines; functions <= 55.
#   Re-declared 2026-08-18 (SUITEGATE) as a PRECONDITION of the
#   returncode fail-closed below, not retro-fitted after a breach.
# COVERAGE: tests/unit/test_check_suite_baseline.py — new-only, fixed-only,
#   both, and exact-match, each asserted on the EXIT CODE as well as the dict,
#   because a checker whose verdict never reaches the shell is a printer.
# =============================================================================
import argparse
import json
import os
import re
import subprocess
import sys

_FAILED = re.compile(r"^FAILED\s+(\S+)")
#: A real completed session always prints a count summary. Its ABSENCE is
#: how "0 failures" is told apart from "0 tests executed".
_RAN = re.compile(r"\d+\s+(passed|failed|error)")


def parse_ids(text: str):
    """Test IDs from pytest -q output. Strips the ' - <error>' suffix."""
    out = set()
    for line in text.splitlines():
        m = _FAILED.match(line.strip())
        if m:
            out.add(m.group(1).split(" - ")[0].strip())
    return out


def load_baseline(path: str):
    if path.endswith(".json"):
        return set(json.load(open(path))["failure_ids"])
    return {l.strip() for l in open(path) if l.strip()
            and not l.lstrip().startswith("#")}


def compare(baseline_ids, current_ids) -> dict:
    new = sorted(current_ids - baseline_ids)
    fixed = sorted(baseline_ids - current_ids)
    verdict = ("RED_NEW_FAILURES" if new
               else ("AMBER_BASELINE_STALE" if fixed else "GREEN"))
    return {"n_baseline": len(baseline_ids), "n_current": len(current_ids),
            "new_failures": new, "fixed_since_baseline": fixed,
            "count_would_have_passed": len(baseline_ids) == len(current_ids),
            "verdict": verdict}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--run", action="store_true",
                    help="invoke pytest tests/unit/ -q")
    ap.add_argument("--from-file", help="pytest output captured earlier")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if not (a.run or a.from_file):
        print("need --run or --from-file", file=sys.stderr)
        return 3
    try:
        if a.run:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/unit/", "-q"],
                capture_output=True)
            text = proc.stdout.decode(errors="replace")
            # FAIL CLOSED on a session that never RAN. MEASURED HOLE
            # (2026-08-18): this path read stdout only. With the empty v3
            # baseline, a session that produced no FAILED lines -- a SUITEGATE
            # refusal (rc 9), a collection error, an internal error -- compared
            # as {} vs {} and returned GREEN rc 0 from a suite that never
            # executed. rc plus a summary-line check is what separates
            # "0 failures" from "0 tests".
            if proc.returncode not in (0, 1) or not _RAN.search(text):
                sys.stderr.write(text[-2000:] + "\n")
                sys.stderr.write(
                    "REFUSING: pytest did not complete a real run (rc=%d). "
                    "No verdict is possible; this is NOT green.\n"
                    % proc.returncode)
                return 3
        else:
            text = open(a.from_file, errors="replace").read()
        res = compare(load_baseline(a.baseline), parse_ids(text))
    except Exception as e:
        print("baseline check FAILED CLOSED: %s: %s" % (type(e).__name__, e),
              file=sys.stderr)
        return 3
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print(json.dumps(res, indent=1))
    if res["verdict"] == "RED_NEW_FAILURES":
        print("\nNEW failures vs baseline — this is a regression:",
              file=sys.stderr)
        for t in res["new_failures"]:
            print("  " + t, file=sys.stderr)
        if res["count_would_have_passed"]:
            print("\nNOTE: the failure COUNT is unchanged. A count-only gate "
                  "would have passed this.", file=sys.stderr)
        return 1
    if res["verdict"] == "AMBER_BASELINE_STALE":
        print("\nNo new failures, but %d baseline failure(s) are now FIXED — "
              "re-capture the baseline." % len(res["fixed_since_baseline"]),
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
