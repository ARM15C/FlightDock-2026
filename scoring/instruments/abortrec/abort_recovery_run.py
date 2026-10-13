#!/usr/bin/env python3
# THE RECOVERY DRIVER -- a CAP-DRIVEN STRUCTURAL SPLIT out of abort_recovery.py, moving no logic.

#   the pair to abort_recovery.contract_row, and write RECOVERY_SUMMARY.csv. It exists because abort_recovery.py
#   passed its declared soft cap when the exit-gate channel correction was added, and a cap is SPLIT, never
#   raised. The reduction-vs-I/O boundary is the same one this dispatch already uses for abort_ticks/abort_pass.
#   A BREAKAWAY WITH NO TRACE IS A REFUSAL, not a skipped row: the two artifacts must describe the same events.
# PUBLIC API: main(in_dir); python abort_recovery_run.py [dir]
# SIDE-EFFECTS: writes RECOVERY_SUMMARY.csv under in_dir. Reads this dispatch's own traces read-only.
# SIZE: soft 35 / hard 50 code-LoC; function cap 25.
# COVERAGE: limb A8 in abort_limbs.py -- one summary row per breakaway event, no more and no fewer.
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A          # noqa: E402
import abort_recovery as RV       # noqa: E402


def main(in_dir):
    ev = pd.read_csv(os.path.join(in_dir, "ABORT_EVENTS.csv"))
    ev = ev[ev.kind == "breakaway"]
    rows = []
    for tag in A.TAGS:
        tr = pd.read_csv(os.path.join(in_dir, "RECOVERY_TRACE_%s.csv.gz" % tag))
        idx = {k: g for k, g in tr.groupby(["leg", "event_idx"])}
        for _, e in ev[ev.tag == tag].iterrows():
            seg = idx.get((e["leg"], e["event_idx"]))
            if seg is None:
                raise SystemExit("REFUSE rc=6: breakaway %s/%s#%d has no trace" % (tag, e["leg"], e["event_idx"]))
            rows.append(RV.contract_row(seg, e))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(in_dir, "RECOVERY_SUMMARY.csv"), index=False)
    print("recovery rows=%d (breakaway events=%d)" % (len(out), len(ev)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
