#!/usr/bin/env python3
"""Board loader and paired cross-check shared by the analysis scripts: board() returns the scoring board of record with the seed block checked; check() compares a same-seed pair against the comparative pair table of record."""
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, (_REPO + "/analysis/verify_20260918/figures"))
import figure_style as S  # noqa: E402

V = S.V
HERE = os.path.dirname(os.path.abspath(__file__))


def board():
    b = pd.read_csv(os.path.join(S.RUN, "board", "BOARD_W10.csv"), low_memory=False)
    for c in ("passed", "sw", "wind_rung"):
        b[c] = b[c].astype(str).str.lower().eq("true")
    for c, g in b.groupby("corpus"):
        if sorted(g.seed.unique()) != S.SEED_BLOCK:
            raise SystemExit("REFUSE: corpus %s is not seed block 10775-10804" % c)
    return b


def check(pairs, sig, x, y, pool, n, r):
    a, b = (x, y) if S.ARMS.index(x) < S.ARMS.index(y) else (y, x)
    p = pairs[(pairs.sigma == sig) & (pairs.arm_a == V.NAME[a]) & (pairs.arm_b == V.NAME[b]) & (pairs.pool == pool)]
    sign = 1 if a == x else -1
    disc = "%d / %d" % ((r["b"], r["c"]) if sign == 1 else (r["c"], r["b"]))
    if len(p) != 1 or r["n"] != n or round(sign * float(p.diff_pp.iloc[0]), 2) != round(r["diff"], 2) or disc != p.discordant.iloc[0]:
        raise SystemExit("REFUSE: %s-%s %s sigma %s does not reproduce ARM_PAIRS.csv" % (x, y, pool, sig))

