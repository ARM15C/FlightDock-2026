#!/usr/bin/env python3
# THE PER-TICK PASS RUNNER -- a CAP-DRIVEN STRUCTURAL SPLIT out of abort_ticks.py, moving no logic.

#   the two artifacts it produces -- the +-30 s recovery traces and the per-event truth geometry. It exists
#   because abort_ticks.py reached 83 of its declared soft 80 lines when the trace writer was added, and the
#   standing rule is that a cap is SPLIT, never raised.
#   A LEG THAT CANNOT BE READ IS RECORDED AS AN ERROR ROW AND COUNTED, never silently dropped: an unreadable
#   leg summarising as "no events" is a silent failure toward good news.
# PUBLIC API: run(tag, out_dir, jobs) -> trace row count; python abort_pass.py <TAG> [out_dir] [jobs]
# SIDE-EFFECTS: writes RECOVERY_TRACE_<TAG>.csv.gz and EVENT_GEOM_<TAG>.jsonl under out_dir. Reads telemetry
#   read-only; never deletes, moves or edits a leg.
# SIZE: soft 40 / hard 55 code-LoC; function cap 25.
# COVERAGE: limb A3 in abort_limbs.py -- trace count equals the breakaway count, and error rows are zero.
import concurrent.futures as cf
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402
import abort_ticks as TK       # noqa: E402


def run(tag, out_dir, jobs=12):
    """The pass for one corpus. Writes the trace file and the per-event geometry beside it."""
    rows = list(A.summary_rows(tag))
    work = [(tag, os.path.join(A.LEGS_ROOT[tag], r["leg"]), r) for r in rows]
    geo, tr = [], []
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        for g, s in ex.map(TK.leg_pass, work, chunksize=4):
            geo.extend(g), tr.extend(s)
    with open(os.path.join(out_dir, "EVENT_GEOM_%s.jsonl" % tag), "w") as fh:
        for g in geo:
            fh.write(json.dumps(g) + "\n")
    trace = pd.concat(tr, ignore_index=True) if tr else pd.DataFrame(columns=["tag", "leg", "event_idx"])
    trace.to_csv(os.path.join(out_dir, "RECOVERY_TRACE_%s.csv.gz" % tag), index=False, compression="gzip")
    n_err = sum(1 for g in geo if "error" in g)
    print("%s legs=%d event_geom=%d errors=%d traces=%d trace_rows=%d"
          % (tag, len(rows), len(geo), n_err, len(tr), len(trace)))
    return len(trace)


if __name__ == "__main__":
    d = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    run(sys.argv[1], d, int(sys.argv[3]) if len(sys.argv) > 3 else 12)
