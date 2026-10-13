#!/usr/bin/env python3
# =============================================================================

#   decides nothing. Truth is always compared against Perception.
# WHAT: per leg, the peak height above the boom-tip on TRUTH (-raw_below_m) and on PERCEPTION (-ctrl_below_m, what the
#   controller reads) in three windows, all selected on the TRUTH range (the same ticks for both channels):
#     corridor raw_range_m <= 10.0 whole leg every tick within 3 m raw_range_m <= 3.0
#   Window rule COPIED from the pinned analysis/windowed_20260831T1900Z/windowed_rescore.py `_leg_peaks`: a tick whose
#   raw_below_m or raw_range_m does not parse is skipped for every window.
#   VALIDATION, ENFORCED (refuses): truth corridor / 3 m / whole-leg peaks equal the board's win / win3 / whole on all
#   7,200 legs (null exactly where the board is null; |diff| <= 1e-4 m, the board's stored precision), and the
#   whole-leg Perception peak equals the board's `fused_peak_up_m` (the readout's perception column) on >= 99.5 % of
#   legs to 1e-3 m - reported exactly either way.
# CLI: --extract [--workers 12] | --report SIDE-EFFECTS: writes PERCEPTION_PEAKS_PER_LEG.csv under OUT only.
# =============================================================================
import argparse
import concurrent.futures as cf
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offset_delivery import OUT, RUN, TAGS  # noqa: E402

WINDOWS = {"corridor": 10.0, "within 3 m": 3.0, "whole leg": None}
COLS = ["raw_range_m", "raw_below_m", "ctrl_below_m"]
PATH = os.path.join(OUT, "PERCEPTION_PEAKS_PER_LEG.csv")


def one(job):
    tag, path = job
    d = pd.read_csv(path, compression="gzip", usecols=COLS, low_memory=False)
    d = d[d.raw_below_m.notna() & d.raw_range_m.notna()]
    row = {"corpus": tag, "leg": path.split(os.sep)[-3]}
    for name, w in WINDOWS.items():
        k = d if w is None else d[d.raw_range_m <= w]
        for ch, col in (("truth", "raw_below_m"), ("perception", "ctrl_below_m")):
            v = -k[col].dropna()
            row["%s_%s" % (ch, name.replace(" ", "_"))] = float(v.max()) if len(v) else np.nan
    return row


def extract(workers):
    jobs = [(t, p) for t in TAGS for p in sorted(glob.glob(os.path.join(RUN, "shims", t, "flat", "*", "env*", "flight_state_ep*.csv.gz")))]
    if len(jobs) != 7200:
        raise SystemExit("REFUSE: expected 7200 flight_state files, found %d" % len(jobs))
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=20))
    pd.DataFrame(rows).to_csv(PATH, index=False)
    print("EXTRACT DONE legs=%d" % len(rows))


def validate():
    L = pd.read_csv(PATH)
    b = pd.read_csv(os.path.join(RUN, "board", "BOARD_W10.csv"), low_memory=False, usecols=["corpus", "leg", "win", "win3", "whole", "fused_peak_up_m"])
    m = b.merge(L, on=["corpus", "leg"], how="outer", indicator=True)
    if not (m._merge == "both").all() or len(m) != 7200:
        raise SystemExit("REFUSE: per-leg join is not 7,200 one-to-one rows")
    for bc, lc in (("win", "truth_corridor"), ("win3", "truth_within_3_m"), ("whole", "truth_whole_leg")):
        same_null = (m[bc].isna() == m[lc].isna()).all()
        dmax = float((m[bc] - m[lc]).abs().max())
        print("truth %-9s == board %-5s on %d legs: same nulls %s, max |diff| %.2g m" % (lc.split("_", 1)[1], bc, len(m), same_null, dmax))
        if not same_null or dmax > 1e-4:
            raise SystemExit("REFUSE: truth %s does not reproduce the board's %s" % (lc, bc))
    agree = ((m.fused_peak_up_m - m.perception_whole_leg).abs() <= 1e-3)
    print("perception whole leg == board fused_peak_up_m (1e-3 m) on %d of %d legs (%.3f %%); max |diff| %.3g m"
          % (agree.sum(), len(m), 100 * agree.mean(), (m.fused_peak_up_m - m.perception_whole_leg).abs().max()))
    if agree.mean() < 0.995:
        raise SystemExit("REFUSE: whole-leg perception disagrees with the readout's perception column on > 0.5 % of legs")
    print("VALIDATION PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.extract:
        extract(a.workers)
    if a.report:
        validate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
