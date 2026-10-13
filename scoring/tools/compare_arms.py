#!/usr/bin/env python3
# compare_arms.py -- ANALYSIS_PLAN_COMPARATIVE.md sections 3, 4 and 6: every arm-against-arm paired difference at
#   each turbulence intensity, the sigma 0.25 -> 0.35 master table per arm, the S0 control-stage table and the one
#   gust-agreement table. Decides nothing. Every method is imported from hypothesis_verdicts.py (Wilson rate
#   interval, Newcombe method-10 paired interval, seed-cluster bootstrap on a p95); the H2 strong-wind bound and the
#   H3 decrease are REPRINTED from this run's PLACEHOLDERS.json, never recomputed. "stage" never "rung"; a peak over 2.00 m
#   is a safety-EXCLUSION-ZONE violation (T1).
# PUBLIC API: ARMS; pair_rows(board) -> DataFrame; master(board, tok) -> DataFrame; s0_table(board) -> DataFrame;
#   gust_table(board) -> DataFrame; main(argv=None) -> int. CLI: --run-dir DIR [--out DIR] [--selftest]
# SIDE-EFFECTS: reads <run-dir>/board/BOARD_W10.csv and <run-dir>/verdicts/PLACEHOLDERS.json; writes ARM_PAIRS.csv,
#   MASTER_TABLE.csv, S0_TABLE.csv, GUST_AGREEMENT_TABLE.csv (+ .md twins) and COMPARATIVE_PLACEHOLDERS.json under
#   <out> (default <run-dir>/comparative/). Spawns nothing. No network.
# SIZE: soft 200 / hard 240 code-lines; function cap 30.
# COVERAGE: --selftest: a synthetic TWELVE-corpus board (_fixture) yields C(6,2) = 15 pairs per sigma, a master row
#   per arm whose decrease equals the placeholder it reprints, the PID-baseline and PIDO rows carrying "no verdict" while
#   the four FlightDock rows carry the H3 verdict statistic, and a gust table whose share equals the planted
#   agreement on the pairs that involve the PID-baseline.
import argparse
import ast
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hypothesis_verdicts as V  # noqa: E402
import gust_agreement as G  # noqa: E402
import p95_boot as PB  # noqa: E402

#: order of record. The PID-baseline sits ahead of PIDO and behind
#: every FlightDock arm, so a policy-vs-baseline pair still reads policy minus PID-baseline. These tables are
#: AMPLIFYING: every arm is compared with every other, including the like-for-like pairs the offset makes possible
#: (S1O vs PIDO, A2O vs PIDO), and NOTHING here grades a hypothesis -- the H2 verdict is against the PID-baseline
#: alone and is only reprinted in the `role` column.
ARMS = ["A2", "A2O", "S1", "S1O", "PID", "PIDO"]
NAME = dict(V.NAME)
POOLS = {"nine wind stages": lambda d: d.wind_rung, "strong-wind (6-8 m/s)": lambda d: d.sw}
VERDICT_ROW = {(p, "PID", "strong-wind (6-8 m/s)", 0.25) for p in ("A2", "A2O", "S1", "S1O")}


def wilson(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return "%.1f %% [%.1f, %.1f]" % (100.0 * k / n, 100 * lo, 100 * hi)


def two_sided(r):
    """Two-sided 95 % Newcombe interval on a paired difference from paired()'s 2x2 counts, in pp."""
    lo = V.newcombe_lb(r["a"], r["b"], r["c"], r["d"], 0.025)
    hi = -V.newcombe_lb(r["a"], r["c"], r["b"], r["d"], 0.025)
    return lo, hi


def p95_diff(board, ca, cb):
    """Pooled corridor p95 of arm a minus arm b on legs that entered the corridor, seed-cluster bootstrap interval."""
    x = board[(board.corpus.isin([ca, cb])) & board.wind_rung]
    return PB.diff_ci(x[x.corpus == ca], x[x.corpus == cb], "win")


def pair_rows(board, tok=None):
    rows = []
    for sigma in (0.25, 0.35):
        for a, b in itertools.combinations(ARMS, 2):
            ca, cb = V.corpus(a, sigma), V.corpus(b, sigma)
            for pool, mask in list(POOLS.items()) + [(lab, (lambda l: lambda d: d.wind_label == l)(lab))
                                                     for lab in sorted(board[board.wind_rung].wind_label.unique(), key=lambda s: int(s[1:]))]:
                r = V.paired(board, ca, cb, mask)
                lo, hi = two_sided(r)
                rows.append({"sigma": sigma, "arm_a": NAME[a], "arm_b": NAME[b], "pool": pool, "n_pairs": r["n"],
                             "k_a": r["ka"], "k_b": r["kb"], "diff_pp": round(r["diff"], 2), "ci95": "[%.1f, %.1f]" % (lo, hi),
                             "discordant": "%d / %d" % (r["b"], r["c"]),
                             "role": "H2 verdict row (reprinted)" if (a, b, pool, sigma) in VERDICT_ROW else "exploratory",
                             "h2_one_sided_lb95_reprinted": (tok or {}).get("H2.lbw95.%s" % a, "") if (a, b, pool, sigma) in VERDICT_ROW else ""})
            pa, pb, d, lo, hi = p95_diff(board, ca, cb)
            rows.append({"sigma": sigma, "arm_a": NAME[a], "arm_b": NAME[b], "pool": "nine wind stages, terminal-approach-corridor p95 (m)",
                         "n_pairs": np.nan, "k_a": round(pa, 3), "k_b": round(pb, 3), "diff_pp": round(d, 3),
                         "ci95": "[%.3f, %.3f]" % (lo, hi), "discordant": "", "role": "exploratory", "h2_one_sided_lb95_reprinted": ""})
    return pd.DataFrame(rows)


def guard_counts(x):
    g = [ast.literal_eval(s) if isinstance(s, str) else (s or {}) for s in x.guards]
    alt = sum(int(d.get("by_type", {}).get("guard_altitude_ceiling", 0)) for d in g)
    ov = sum(int(d.get("by_type", {}).get("guard_hard_overtake", 0)) for d in g)
    return alt, ov


def arm_block(board, tok, arm, sigma):
    c, x = V.corpus(arm, sigma), V.wind(board, arm, sigma)
    sw = x[x.sw]
    alt, ov = guard_counts(x)
    ent = x.dropna(subset=["win"])
    return {"capture rate, nine wind stages": wilson(int(x.passed.astype(bool).sum()), len(x)),
            "capture rate, strong-wind (6-8 m/s)": wilson(int(sw.passed.astype(bool).sum()), len(sw)),
            "terminal-approach-corridor p95 (m)": "%s %s" % tuple(tok["H1.%s.%s.s%s" % (k, arm, "025" if sigma == 0.25 else "035")] for k in ("p95", "p95ci")),
            "legs over 2.00 m in the terminal-approach corridor": int((ent.win > V.BAR).sum()), "legs entering the terminal-approach corridor": len(ent),
            "overtake legs": int(x.real_overtake.astype(bool).sum()), "altitude-guard fires": alt,
            "overtake-guard fires": ov}


def master(board, tok):
    rows = []
    for arm in ARMS:
        a, b = arm_block(board, tok, arm, 0.25), arm_block(board, tok, arm, 0.35)
        for q in a:
            rows.append({"arm": NAME[arm], "quantity": q, "sigma 0.25": a[q], "sigma 0.35": b[q],
                         "change": "" if isinstance(a[q], str) else b[q] - a[q]})
        src = "H3 verdict statistic (reprinted)" if arm in V.POL else "same rule, reported (no verdict)"
        rows.append({"arm": NAME[arm], "quantity": "capture-rate decrease sigma (σ) 0.25 -> 0.35 (pp), nine wind stages",
                     "sigma 0.25": "", "sigma 0.35": "", "change": "%s %s -- %s" % (tok["H3.decrease.%s" % arm], tok["H3.decrease.ci.%s" % arm], src)})
        rows.append({"arm": NAME[arm], "quantity": "capture-rate decrease sigma (σ) 0.25 -> 0.35 (pp), strong-wind (6-8 m/s)",
                     "sigma 0.25": "", "sigma 0.35": "", "change": "%s -- reprinted" % tok["H3.decrease.sw.%s" % arm]})
        pa, pb, d, lo, hi = p95_diff(board, V.corpus(arm, 0.35), V.corpus(arm, 0.25))
        rows.append({"arm": NAME[arm], "quantity": "terminal-approach-corridor p95 increase sigma (σ) 0.25 -> 0.35 (m), bootstrap 95 %",
                     "sigma 0.25": round(pb, 3), "sigma 0.35": round(pa, 3), "change": "%.3f [%.3f, %.3f]" % (d, lo, hi)})
    return pd.DataFrame(rows)


def s0_table(board):
    rows = []
    for arm in ARMS:
        for sigma, lab, name in ((0.25, "S0", "S0 calm control stage (turbulence off)"), (0.25, "W0", "0 m/s mean-wind stage, sigma (σ) 0.25"),
                                 (0.35, "W0", "0 m/s mean-wind stage, sigma (σ) 0.35")):
            x = board[(board.corpus == V.corpus(arm, sigma)) & (board.wind_label == lab)]
            if len(x) != 60:
                raise SystemExit("REFUSE: %s %s %s holds %d legs, not 60" % (arm, sigma, lab, len(x)))
            alt, ov = guard_counts(x)
            ent = x.dropna(subset=["win"])
            rows.append({"arm": NAME[arm], "stage": name, "n": 60, "capture rate": wilson(int(x.passed.astype(bool).sum()), 60),
                         "terminal-approach-corridor p95 (m)": round(PB.p95(ent.win), 3), "p95 bootstrap 95 %": PB.fmt(*PB.ci(x, "win")),
                         "legs over 2.00 m": int((ent.win > V.BAR).sum()), "overtake legs": int(x.real_overtake.astype(bool).sum()),
                         "altitude-guard fires": alt, "overtake-guard fires": ov})
    return pd.DataFrame(rows)


def gust_table(board):
    """Every arm pair at each sigma: agreement share, the nine-stage paired difference on all legs and on agreeing legs."""
    saved = G.PAIRS
    G.PAIRS = [(V.corpus(a, s), V.corpus(b, s)) for s in (0.25, 0.35) for a, b in itertools.combinations(ARMS, 2)]
    try:
        summ, legs = G.agreement(board)
    finally:
        G.PAIRS = saved
    out = []
    for r in summ.itertuples():
        keys = legs[(legs.left == r.left) & (legs.right == r.right) & legs.agree].key
        full = V.paired(board, r.left, r.right, lambda d: d.wind_rung)
        sub = V.paired(board[board.key.isin(keys)], r.left, r.right, lambda d: d.wind_rung)
        out.append({"left": r.left, "right": r.right, "n_pairs": r.n_pairs, "agree_share_pct": r.share_pct,
                    "diff_all_pp": round(full["diff"], 2), "ci95_all": "[%.1f, %.1f]" % two_sided(full),
                    "diff_agreeing_pp": round(sub["diff"], 2), "ci95_agreeing": "[%.1f, %.1f]" % two_sided(sub), "n_agreeing": sub["n"]})
    return pd.DataFrame(out)


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
    tok = json.load(open(os.path.join(a.run_dir, "verdicts", "PLACEHOLDERS.json")))
    out = a.out or os.path.join(a.run_dir, "comparative")
    os.makedirs(out, exist_ok=True)
    tables = {"ARM_PAIRS": pair_rows(board, tok), "MASTER_TABLE": master(board, tok), "S0_TABLE": s0_table(board), "GUST_AGREEMENT_TABLE": gust_table(board)}
    for k, df in tables.items():
        write(df, out, k)
    m = tables["MASTER_TABLE"]
    ct = {"CMP.master.%s.%s" % (r.arm.split()[0].replace("-", ""), r.quantity.split(",")[0].replace(" ", "_")): str(r.change) for r in m.itertuples() if r.change != ""}
    json.dump(ct, open(os.path.join(out, "COMPARATIVE_PLACEHOLDERS.json"), "w"), indent=1)
    print("comparative tables: " + ", ".join("%s %d rows" % (k, len(v)) for k, v in tables.items()))
    return 0


def _fixture():
    """A synthetic twelve-corpus board plus the placeholder file the comparative tables reprint from."""
    r = np.random.default_rng(1)
    rows = []
    for arm in ARMS:
        for sigma in (0.25, 0.35):
            for lab in ["S0"] + ["W%d" % w for w in range(0, 17, 2)]:
                for seed in range(10775, 10805):
                    for tr in ("straight", "racetrack"):
                        rows.append({"corpus": V.corpus(arm, sigma), "wind_label": lab, "seed": seed, "trajectory": tr,
                                     "key": "%s_%d_%s" % (lab, seed, tr), "wind_rung": lab != "S0", "sw": lab in ("W12", "W14", "W16"),
                                     "passed": bool(r.random() < (0.7 if sigma == 0.25 else 0.5)), "win": float(r.normal(1.5, 0.3)),
                                     "real_overtake": False, "guards": "{'by_type': {'guard_hard_overtake': 1}, 'total': 1}",
                                     "sigma_gate": "{'ned': [0.25, 0.25, %s]}" % ("0.26" if (arm == "PID" and seed % 10 == 0) else "0.25") if lab != "S0" else "{'ned': None}"})
    board = pd.DataFrame(rows)
    tok = {"H1.p95.%s.s%s" % (a, s): 1.5 for a in ARMS for s in ("025", "035")}
    tok.update({"H1.p95ci.%s.s%s" % (a, s): "[1.4, 1.6]" for a in ARMS for s in ("025", "035")})
    for a in ARMS:
        tok.update({"H3.decrease.%s" % a: 20.0, "H3.decrease.ci.%s" % a: "[15.0, 25.0]", "H3.decrease.sw.%s" % a: 21.0})
    PB.BOOT = 50
    tok.update({"H2.lbw95.A2": 3.5, "H2.lbw95.S1": -12.0})
    return board, tok


def selftest():
    board, tok = _fixture()
    n = len(ARMS)
    npairs = n * (n - 1) // 2
    p = pair_rows(board, tok); assert p.sigma.nunique() == 2 and (p.groupby("sigma").size() == npairs * (2 + 9 + 1)).all(), p.groupby("sigma").size()
    assert (p.role == "H2 verdict row (reprinted)").sum() == 4 and {3.5, -12.0} <= set(p[p.role != "exploratory"].h2_one_sided_lb95_reprinted)
    m = master(board, tok); assert len(m) == n * (8 + 3)
    dec = m[m.quantity.str.startswith("capture-rate decrease sigma (σ) 0.25 -> 0.35 (pp), nine wind stages")]
    assert len(dec) == n and dec.change.str.startswith("20.0").all(), dec
    assert (dec[dec.arm.isin([NAME["PID"], NAME["PIDO"]])].change.str.contains("no verdict")).all() \
        and (dec[~dec.arm.isin([NAME["PID"], NAME["PIDO"]])].change.str.contains("H3 verdict")).all()
    s = s0_table(board); assert len(s) == 3 * n
    g = gust_table(board); pid = g.left.str.startswith("ARMB") | g.right.str.startswith("ARMB")
    assert len(g) == 2 * npairs and int(pid.sum()) == 2 * (n - 1) and (g[pid].agree_share_pct == 90.0).all() \
        and (g[~pid].agree_share_pct == 100.0).all(), g.agree_share_pct.unique()
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
