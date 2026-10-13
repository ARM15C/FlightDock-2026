#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: SPAWNGUARD-782. Decide, from the spawner's OWN accounting
#   line, whether a leg actually flew -- because rc cannot.
#   curriculum_eval_spawner.py is idempotent by leg directory: a seed whose
#   directory already exists is SKIPPED and the process still exits 0, printing
#   {"legs": 1, "spawned": 0, "skipped_existing": 1, "legs_failed": 0, ...}.
#   MEASURED TWICE on 2026-08-21: it made a control leg look flown when it
#   never ran, and it would have let ARMBDESC wave 1 reuse 35 legs flown under
#   a superseded episode-termination rule while reporting success.
# PUBLIC API: verdict(text) -> (rc, message); main(argv) -> int
#   rc 0 every requested leg flew (spawned == legs, no failures)
#   rc 2 SKIPPED_EXISTING -- a stale leg dir was reused
#   rc 3 NO ACCOUNTING LINE -- unreadable or absent; fail-closed, never "fine"
#   rc 4 FAILED LEG -- the spawner itself reported legs_failed
# SIDE-EFFECTS: reads one lane log, prints one line. Writes nothing, kills
#   nothing, never touches the leg.
# FAIL DIRECTION: closed. Absence of evidence is rc 3, not rc 0.
# CAPS: file soft 70 / hard 90 code-lines; function cap 30.
# COVERAGE: --selfcheck exercises all four verdicts, both directions.
# =============================================================================
import argparse
import json
import sys


def verdict(text):
    """Last JSON object in the lane log is the spawner's accounting line."""
    line = None
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("{") and '"spawned"' in s:
            line = s
    if line is None:
        return 3, "NO ACCOUNTING LINE: the spawner printed no {...spawned...}"
    try:
        d = json.loads(line)
    except ValueError as e:
        return 3, "NO ACCOUNTING LINE: unparseable (%s)" % e
    legs = d.get("legs")
    spawned = d.get("spawned")
    failed = d.get("legs_failed") or 0
    skipped = d.get("skipped_existing") or 0
    if legs is None or spawned is None:
        return 3, "NO ACCOUNTING LINE: missing legs/spawned in %s" % line
    if failed:
        return 4, "FAILED LEG: legs_failed=%d in %s" % (failed, line)
    if skipped or spawned != legs:
        return 2, ("SKIPPED_EXISTING: spawned=%d of legs=%d, skipped=%d -- a "
                   "stale leg directory was reused; rc would have been 0"
                   % (spawned, legs, skipped))
    return 0, "spawned=%d of legs=%d, nothing skipped" % (spawned, legs)


def selfcheck():
    ok = '{"legs": 1, "spawned": 1, "skipped_existing": 0, "legs_failed": 0}'
    skip = '{"legs": 1, "spawned": 0, "skipped_existing": 1, "legs_failed": 0}'
    fail = '{"legs": 1, "spawned": 1, "skipped_existing": 0, "legs_failed": 1}'
    cases = [
        ("flew", "noise\n" + ok, 0),
        ("skipped (THE bug)", "noise\n" + skip, 2),
        ("failed", fail, 4),
        ("no line", "just log text\n", 3),
        ("unparseable", '{"spawned": ', 3),
        ("missing keys", '{"spawned": 1}', 3),
        ("last line wins", skip + "\n" + ok, 0),
        ("last line wins, other way", ok + "\n" + skip, 2),
    ]
    for name, text, want in cases:
        got, msg = verdict(text)
        assert got == want, "%s: want rc %d got %d (%s)" % (name, want, got, msg)
        print(" PASS %-26s rc=%d %s" % (name, got, msg[:60]))
    print("selfcheck OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane-log")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck()
    try:
        text = open(a.lane_log, encoding="utf-8", errors="replace").read()
    except OSError as e:
        print("[spawnguard] rc 3 unreadable lane log: %s" % e)
        return 3
    rc, msg = verdict(text)
    print("[spawnguard] rc %d %s" % (rc, msg))
    return rc


if __name__ == "__main__":
    sys.exit(main())
