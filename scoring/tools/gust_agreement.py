#!/usr/bin/env python3
# =============================================================================

#   lists: for each pair of corpora that a plan compares within seed, the share of paired wind-stage legs whose
#   REALIZED per-axis sigma (the driver's own sigma_gate record, `sigma_ned`, carried per leg in the board) agrees
#   to six decimals, and a per-leg agreement flag so the verdict script can recompute every paired statistic on
#   the agreeing subset. Method of the 2026-09-04 gust_share.py instrument (compare what the FDM produced, never
#   the seeding code); S0 legs have no realized sigma and are excluded, by design. Decides nothing.
# PUBLIC API: PAIRS; realized(df) -> DataFrame[key, corpus, n, e, d]; agreement(board) -> (summary, per_leg);
#   main(argv=None) -> int. CLI: --run-dir DIR [--selftest]
# SIDE-EFFECTS: reads <run-dir>/board/BOARD_W10.csv; writes GUST_AGREEMENT.csv (summary) and
#   GUST_AGREEMENT_LEGS.csv (per leg, per pair) under <run-dir>/board/. Spawns nothing.
# SIZE: soft 90 / hard 110 code-LoC; function cap 30.
# COVERAGE: --selftest: two synthetic corpora agreeing on 2 of 3 labels give 2/3; a pair with an empty join REFUSES.
# =============================================================================
import argparse
import ast
import sys

import itertools
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hypothesis_verdicts as V  # noqa: E402

#: (left, right) corpora compared within seed; both sides at one turbulence intensity. EVERY pair of the six arms is
#: listed, so the comparative plan's all-against-all tables are covered too.
PAIRS = [(V.corpus(a, s), V.corpus(b, s)) for s in (0.25, 0.35) for a, b in itertools.combinations(V.ORDER, 2)]


def realized(df):
    """Per wind-stage leg: the realized N/E/D sigma from the sigma_gate record (None for S0 and unsampled legs)."""
    def ned(g):
        g = g if isinstance(g, dict) else (ast.literal_eval(g) if isinstance(g, str) and g else {})
        v = g.get("ned")
        return v if isinstance(v, list) and len(v) == 3 else [np.nan] * 3
    x = df[df.wind_label != "S0"]
    arr = np.array([ned(g) for g in x.sigma_gate], dtype=float)
    return pd.DataFrame({"key": x.key.values, "corpus": x.corpus.values, "n": arr[:, 0], "e": arr[:, 1], "d": arr[:, 2]})


def agreement(board):
    r = realized(board)
    rows, legs = [], []
    for a, b in PAIRS:
        m = r[r.corpus == a].merge(r[r.corpus == b], on="key", suffixes=("_a", "_b"))
        if m.empty:
            raise SystemExit("REFUSE: pair %s/%s joins ZERO legs" % (a, b))
        agree = ((m.n_a.round(6) == m.n_b.round(6)) & (m.e_a.round(6) == m.e_b.round(6)) & (m.d_a.round(6) == m.d_b.round(6)))
        rows.append({"left": a, "right": b, "n_pairs": len(m), "agree": int(agree.sum()),
                     "share_pct": round(100.0 * agree.mean(), 2), "unsampled": int(m[["n_a", "n_b"]].isna().any(axis=1).sum())})
        legs.append(pd.DataFrame({"left": a, "right": b, "key": m.key, "agree": agree.values}))
    return pd.DataFrame(rows), pd.concat(legs, ignore_index=True)


def selftest():
    global PAIRS
    mk = lambda tag, vals: pd.DataFrame({"key": ["W12|1|s", "W12|2|s", "W12|3|s"], "corpus": tag, "wind_label": "W12",
                                         "sigma_gate": [{"ned": v} for v in vals]})
    a = mk("A2P025", [[0.25, 0.2, 0.3], [0.251, 0.2, 0.3], [0.25, 0.2, 0.3]])
    b = mk("ARMB025", [[0.25, 0.2, 0.3], [0.250, 0.2, 0.3], [0.25, 0.2, 0.3]])
    saved, PAIRS = PAIRS, [("A2P025", "ARMB025")]
    try:
        s, l = agreement(pd.concat([a, b]))
        ok = int(s.agree.iloc[0]) == 2 and len(l) == 3 and s.n_pairs.iloc[0] == 3
        PAIRS = [("A2P025", "S1A25")]
        try:
            agreement(pd.concat([a, b]))
            ok = False                      # an empty join must refuse, never report a share
        except SystemExit:
            pass
    finally:
        PAIRS = saved
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    bd = os.path.join(a.run_dir, "board")
    board = pd.read_csv(os.path.join(bd, "BOARD_W10.csv"), low_memory=False)
    s, l = agreement(board)
    s.to_csv(os.path.join(bd, "GUST_AGREEMENT.csv"), index=False)
    l.to_csv(os.path.join(bd, "GUST_AGREEMENT_LEGS.csv"), index=False)
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
