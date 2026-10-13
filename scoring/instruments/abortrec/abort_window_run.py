#!/usr/bin/env python3
# THE WINDOW PASS RUNNER -- a CAP-DRIVEN STRUCTURAL SPLIT out of abort_window.py, moving no logic.

#   write WINDOW_TICKS_<TAG>.jsonl. Same reduction-vs-I/O boundary this dispatch already uses twice.
#   REFUSES on a corpus that does not resolve exactly 600 legs: a reproduction limb can pass by finding
#   nothing, so the denominator is asserted before any counting begins.
# PUBLIC API: run(tag, out_dir, jobs); python abort_window_run.py <TAG> [out_dir] [jobs]
# SIDE-EFFECTS: writes WINDOW_TICKS_<TAG>.jsonl under out_dir. Reads telemetry read-only.
# SIZE: soft 35 / hard 50 code-LoC; function cap 25.
# COVERAGE: limb A10 in abort_limbs.py -- 600 rows per corpus, zero errors, and the never-within count equal
#   to the gustapproach approach-pull census.
import concurrent.futures as cf
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402
import abort_window as WD      # noqa: E402


def run(tag, out_dir, jobs=10):
    """The window pass for one corpus. An unreadable leg is recorded and counted, never silently dropped."""
    legs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(A.LEGS_ROOT[tag], "*")))
    if len(legs) != 600:
        raise SystemExit("REFUSE rc=3: %s resolves %d legs, not 600" % (tag, len(legs)))
    work = [(tag, os.path.join(A.LEGS_ROOT[tag], n), n) for n in legs]
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex, \
            open(os.path.join(out_dir, "WINDOW_TICKS_%s.jsonl" % tag), "w") as fh:
        rows = list(ex.map(WD.leg_window, work, chunksize=4))
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_err = sum(1 for r in rows if "error" in r)
    never = sum(1 for r in rows if not r.get("ever_within_window"))
    print("%s legs=%d errors=%d never_within_%.0fm=%d" % (tag, len(rows), n_err, A.WINDOW_M, never))
    return never


if __name__ == "__main__":
    d = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    run(sys.argv[1], d, int(sys.argv[3]) if len(sys.argv) > 3 else 10)
