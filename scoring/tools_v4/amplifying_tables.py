#!/usr/bin/env python3
# amplifying_tables.py -- ANALYSIS_PLAN_COMPARATIVE.md section 5, the BOARD-derived half: safety counters by stage,
#   dwell of successful legs (mean/median/max/p95), corridor distributions, and the H1 channel-sensitivity rows
#   (corridor 10 m / 3 m / whole leg / perception whole leg). The tick-level half (aborts, stopping distance,
#   recovery, and TIME TO DOCK read per tick -- the per-leg metrics file's milestone times were found unreliable on
#   177 captured legs, 2026-09-16) is amplifying_ticklevel.py. Decides nothing. All arithmetic pandas/numpy.
#   "stage" never "rung"; a peak over 2.00 m is a safety-EXCLUSION-ZONE violation (T1).
# PUBLIC API: safety_by_stage(board); dwell(board); corridor(board);
#   channel_sensitivity(board); main(argv=None) -> int. CLI: --run-dir DIR [--out DIR] [--selftest]
# SIDE-EFFECTS: reads <run-dir>/board/BOARD_W10.csv; writes SAFETY_BY_STAGE, DWELL, CORRIDOR_DIST,
#   CHANNEL_SENSITIVITY (.csv + .md) under <out>
#   (default <run-dir>/comparative/). Spawns nothing. No network.
# SIZE: soft 170 / hard 200 code-lines; function cap 30.
# COVERAGE: --selftest: a synthetic board yields the expected row counts and a dwell p95 computed by numpy on the
#   same values.
import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hypothesis_verdicts as V  # noqa: E402
import compare_arms as C  # noqa: E402
import p95_boot as PB  # noqa: E402

VALS = ["min", "mean", "median", "p95", "max"]
DIST = {"n": "size", "min": "min", "mean": "mean", "median": "median", "p95": lambda s: np.percentile(s, 95), "max": "max"}


def arm_sigma(board):
    """arm name and sigma for every board row from its corpus tag (the verdict tool's own mapping, inverted)."""
    inv = {V.corpus(a, s): (C.NAME[a], s) for a in C.ARMS for s in (0.25, 0.35)}
    miss = set(board.corpus) - set(inv)
    if miss:
        raise SystemExit("REFUSE: unknown corpus tags %s" % sorted(miss))
    b = board.copy()
    b["arm"], b["sig"] = zip(*b.corpus.map(inv))
    b["stage"] = b.wind_label.map(lambda s: "S0 calm control stage" if s == "S0" else "%d m/s" % (int(s[1:]) // 2))
    return b


def safety_by_stage(board):
    b = arm_sigma(board)
    rows = []
    for (arm, sig, st), x in b.groupby(["arm", "sig", "stage"], sort=False):
        g = [ast.literal_eval(s) if isinstance(s, str) else (s or {}) for s in x.guards]
        alt = [int(d.get("by_type", {}).get("guard_altitude_ceiling", 0)) for d in g]
        ov = [int(d.get("by_type", {}).get("guard_hard_overtake", 0)) for d in g]
        ent = x.dropna(subset=["win"])
        rows.append({"arm": arm, "sigma": sig, "stage": st, "n": len(x), "captured": int(x.passed.astype(bool).sum()),
                     "entered corridor": len(ent), "never reached corridor": len(x) - len(ent),
                     "legs over 2.00 m in corridor": int((ent.win > V.BAR).sum()), "contact-plane crossings, whole leg": int((x.whole > V.CONTACT).sum()),
                     "overtake legs": int(x.real_overtake.astype(bool).sum()), "altitude-guard fires": sum(alt), "legs with an altitude fire": sum(1 for v in alt if v),
                     "overtake-guard fires": sum(ov), "legs with an overtake fire": sum(1 for v in ov if v)})
    return pd.DataFrame(rows)


def dwell(board):
    b = arm_sigma(board)
    ok = b[b.passed.astype(bool)]
    d = ok.groupby(["arm", "sig", "stage"], sort=False).intersection_v3_s.agg(**DIST).reset_index()
    pooled = ok[ok.wind_rung].groupby(["arm", "sig"], sort=False).intersection_v3_s.agg(**DIST).reset_index().assign(stage="nine wind stages pooled")
    return pd.concat([d, pooled], ignore_index=True).round({v: 2 for v in VALS})



def corridor(board):
    """Per-leg peak upward deviation, legs that entered the corridor. The p95 carries its seed-cluster bootstrap
    interval, resampled over the seed labels of the WHOLE pool (legs that never entered included)."""
    b = arm_sigma(board)
    ent = b.dropna(subset=["win"])
    keys = ["arm", "sig", "stage"]
    d = ent.groupby(keys, sort=False).win.agg(**DIST).reset_index()
    ov = ent.groupby(keys, sort=False).real_overtake.apply(lambda s: int(s.astype(bool).sum()))
    d["overtake legs"] = [int(ov[k]) for k in zip(d.arm, d.sig, d.stage)]   # every row, pooled rows included
    ci = {k: PB.fmt(*PB.ci(x, "win")) for k, x in b.groupby(keys, sort=False)}
    d["p95 bootstrap 95 %"] = [ci[k] for k in zip(d.arm, d.sig, d.stage)]
    pooled = ent[ent.wind_rung].groupby(["arm", "sig"], sort=False).win.agg(**DIST).reset_index().assign(stage="nine wind stages pooled")
    cip = {k: PB.fmt(*PB.ci(x, "win")) for k, x in b[b.wind_rung].groupby(["arm", "sig"], sort=False)}
    pooled["p95 bootstrap 95 %"] = [cip[k] for k in zip(pooled.arm, pooled.sig)]
    return pd.concat([d, pooled], ignore_index=True).round({v: 3 for v in VALS}).rename(columns={"n": "legs entering corridor"})


CHANNELS = {"p95 corridor 10 m (of record)": "win", "p95 within 3 m of the tip": "win3",
            "p95 whole leg (entire episode), truth": "whole", "p95 whole leg (entire episode), perception": "fused_peak_up_m"}


def channel_sensitivity(board):
    """Every p95 of the per-leg peak upward deviation, in any channel, with the same seed-cluster bootstrap interval."""
    b = arm_sigma(board)
    rows = []
    for (arm, sig), x in b[b.wind_rung].groupby(["arm", "sig"], sort=False):
        row = {"arm": arm, "sigma": sig}
        for name, col in CHANNELS.items():
            row[name] = round(PB.p95(x[col]), 3)
            row[name + ", bootstrap 95 %"] = PB.fmt(*PB.ci(x, col))
        row.update({"legs in 3 m pool": int(x.win3.notna().sum()), "legs in 10 m pool": int(x.win.notna().sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def write(df, out, name):
    df.to_csv(os.path.join(out, name + ".csv"), index=False)
    open(os.path.join(out, name + ".md"), "w").write(V.md_table(df) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir"), ap.add_argument("--out"), ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    board = pd.read_csv(os.path.join(a.run_dir, "board", "BOARD_W10.csv"), low_memory=False)
    out = a.out or os.path.join(a.run_dir, "comparative")
    os.makedirs(out, exist_ok=True)
    for name, df in (("SAFETY_BY_STAGE", safety_by_stage(board)), ("DWELL", dwell(board)),
                     ("CORRIDOR_DIST", corridor(board)), ("CHANNEL_SENSITIVITY", channel_sensitivity(board))):
        write(df, out, name)
        print("%s %d rows" % (name, len(df)))
    return 0


def selftest():
    rows = []
    for arm in C.ARMS:
        for sig in (0.25, 0.35):
            for lab in ["S0"] + ["W%d" % w for w in range(0, 17, 2)]:
                for seed in (10775, 10776):
                    rows.append({"corpus": V.corpus(arm, sig), "leg": "%s_%d_straight" % (lab, seed), "wind_label": lab, "seed": seed, "trajectory": "straight",
                                 "wind_rung": lab != "S0", "sw": lab in ("W12", "W14", "W16"), "passed": True, "intersection_v3_s": 3.0 + seed % 10,
                                 "win": 1.5, "win3": 1.4, "whole": 1.6, "fused_peak_up_m": 1.7, "real_overtake": False, "guards": "{'by_type': {}}"})
    board = pd.DataFrame(rows)
    PB.BOOT = 50
    d = dwell(board); s = board[board.wind_rung & (board.corpus == "A2P025")].intersection_v3_s
    assert abs(d[(d.stage == "nine wind stages pooled") & (d.arm == C.NAME["A2"]) & (d.sig == 0.25)].p95.iloc[0] - round(np.percentile(s, 95), 2)) < 1e-9
    n = len(C.ARMS)                                   # six arms from 2026-09-17; ten wind-label groups per arm-sigma
    assert len(safety_by_stage(board)) == n * 2 * 10 and len(corridor(board)) == n * 2 * 11 and len(channel_sensitivity(board)) == n * 2
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
