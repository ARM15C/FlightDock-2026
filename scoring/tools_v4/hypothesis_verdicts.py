#!/usr/bin/env python3
# =============================================================================

#   placeholder registry from the board of record, under the signed plans:
#   H1 -- pooled p95 of the per-leg peak upward deviation inside the aerial-docking terminal approach corridor over
#         the nine sigma-0.25 wind stages (legs that entered the corridor), bar 2.00 m; seed-cluster bootstrap CI.
#   H2 -- same-seed paired strong-wind (6, 7, 8 m/s mean wind) difference vs the PID-baseline, decided on the
#         COMPUTED DIFFERENCE, bar +15 pp; the two-sided 95 % seed-cluster bootstrap is the
#         range of record beside it and the Wilson-based paired range (Newcombe 1998, method 10) is a comparison.
#         ONE METHOD PER HYPOTHESIS: no Wald bound, no alpha-0.04 node, no
#         Cohen's h, no design effect. Rates carry Wilson 95 % intervals; paired differences the Wilson-based
#         paired interval; the p95 its seed-cluster bootstrap.
#   H3 -- DECREASE (sigma 0.25 rate minus sigma 0.35 rate, positive) in the pooled nine-stage safe-capture rate on
#         the same seeds, bar 35 pp; the two-sided 95 % Wilson-based paired interval on the decrease beside it.
#   Every statistic is also computed on the exam's whole-leg flag (`passed_exam`, the reproduction of the banked
#   record) and, where GUST_AGREEMENT_LEGS.csv exists, on the gust-agreeing subset. All arithmetic pandas / numpy /
#   scipy / statsmodels; nothing typed, nothing carried from an earlier document. "stage" never "rung"; a peak
#   over the bar is a safety-EXCLUSION-ZONE violation (T1); turbulence 0.25 / 0.35 only (T5).
# PUBLIC API: Z1; h1(board, arm, sigma); paired(board, a, b, mask, col); h2(board, pol); h3(board, arm);
#   placeholders(board, agree) -> dict; main(argv=None) -> int. CLI: --run-dir DIR [--selftest]
# SIDE-EFFECTS: reads <run-dir>/board/*.csv; writes <run-dir>/verdicts/HYPOTHESIS_VERDICTS.csv, .md and PLACEHOLDERS.json. Spawns nothing.
# SIZE: soft 230 / hard 270 code-LoC; function cap 45.
# COVERAGE: --selftest: a synthetic board with known discordant pairs reproduces the paired SE, the alpha1 bound
#   and the H3 decrease sign; a board whose paired join is empty REFUSES.
# =============================================================================
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p95_boot as PB  # noqa: E402 (the ONE p95 + seed-cluster bootstrap implementation; fixed seed, fresh per call)
import boot_rates as BR  # noqa: E402 (v4: the same bootstrap for capture rates and paired differences)

BAR, CONTACT, H2_BAR, H3_BAR = 2.00, 3.42, 15.0, 35.0
# Arm keys = the operator's shorthand (2026-09-16, PIDO added 2026-09-17): A2, A2O, S1 (no offset), S1O (+0.25 m
# offset), PID, PIDO (PID-baseline + 0.25 m offset). Values = corpus-tag prefixes.
ARM = {"A2": "A2P", "A2O": "A2V", "S1": "S1A", "S1O": "CSV", "PID": "ARMB", "PIDO": "PIDO"}
NAME = {"A2": "FlightDock-A2 (no offset)", "A2O": "FlightDock-A2O (+0.25 m offset)", "S1": "FlightDock-S1 (no offset)",
        "S1O": "FlightDock-S1O (+0.25 m offset)", "PID": "PID-baseline", "PIDO": "PIDO (PID-baseline, +0.25 m offset)"}
ORDER = ["A2", "A2O", "S1", "S1O", "PID", "PIDO"]
POL = ["A2", "A2O", "S1", "S1O"]          # every FlightDock configuration receives its own verdict on each hypothesis
NOVERDICT = ["PID", "PIDO"]               # reported with the same statistic, never judged
SHORT_SUF = ("S1A", "A2V", "PIDO")        # corpus tags written without the leading zero: S1A25, A2V25, PIDO25
STAGE_MS = {"W%d" % (2 * i): float(i) for i in range(9)}


def corpus(arm, sigma):
    """A2P025 / CSV025 / ARMB025; the ablation and PIDO corpora are tagged S1A25 / A2V25 / PIDO25 (no leading 0)."""
    suf = {0.25: "025", 0.35: "035"}[sigma]
    return ARM[arm] + (suf[1:] if ARM[arm] in SHORT_SUF else suf)


def wind(board, arm, sigma):
    x = board[(board.corpus == corpus(arm, sigma)) & board.wind_rung]
    if len(x) != 540:
        raise SystemExit("REFUSE: %s %s holds %d wind-stage legs, not 540" % (arm, sigma, len(x)))
    return x


def h1(board, arm, sigma):
    x = wind(board, arm, sigma)
    ent = x[x.win.notna()]
    p95 = PB.p95(ent.win)
    per = ent.groupby("wind_label").win.apply(PB.p95)
    over = per[per > BAR]
    return {"p95": round(p95, 3), "ci": PB.fmt(*PB.ci(x, "win")), "whole_ci": PB.fmt(*PB.ci(x, "whole")),
            "n_entered": len(ent), "never": int(x.win.isna().sum()),
            "stages_over": "%d of 9" % len(over),
            "stage_detail": "; ".join("%s (%.1f m/s mean wind, %.3f m)" % (k, STAGE_MS[k], v) for k, v in over.items()) or "none",
            "legs_over": int((ent.win > BAR).sum()), "contact": int((ent.win > CONTACT).sum()),
            "contact_whole": int((x.whole > CONTACT).sum()), "overtake": int(x.real_overtake.astype(bool).sum()),
            "p95_whole": round(PB.p95(x.whole), 3), "verdict": "MET" if p95 <= BAR else "NOT MET"}


def newcombe_lb(a, b, c, d, alpha, paired=True):
    """Newcombe (1998) method 10: the Wilson-based lower bound for a difference p1 - p2, one-sided at alpha.
    a = both succeed, b = policy only, c = PID only, d = neither. phi is the phi coefficient of the 2x2; 0 when a
    margin is empty. Returned in percentage points.
    paired=False sets phi to 0, which is the Wilson-based bound for two INDEPENDENT proportions: the same arithmetic
    with the same-seed pairing thrown away. It is reported as a comparison so the value of the paired design is
    visible; it never decides anything, because these legs ARE paired on (seed, stage, trajectory)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = proportion_confint(a + b, n, alpha=2 * alpha, method="wilson")
    l2, u2 = proportion_confint(a + c, n, alpha=2 * alpha, method="wilson")
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = ((a * d - b * c) / np.sqrt(den) if den > 0 else 0.0) if paired else 0.0
    return 100.0 * ((p1 - p2) - np.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))


def paired(board, a, b, mask, col="passed"):
    """Same-seed pairing on (stage, seed, trajectory). An empty join REFUSES (class C55)."""
    m = board[(board.corpus == a) & mask(board)][["key", col, "seed"]].merge(
        board[(board.corpus == b) & mask(board)][["key", col]], on="key", suffixes=("_a", "_b"))
    if m.empty:
        raise SystemExit("REFUSE: the paired join %s/%s produced ZERO rows" % (a, b))
    pa, pb = m[col + "_a"].astype(bool), m[col + "_b"].astype(bool)
    n, bb, cc = len(m), int((pa & ~pb).sum()), int((~pa & pb).sum())
    diff = 100.0 * (pa.sum() - pb.sum()) / n
    aa, dd = int((pa & pb).sum()), int((~pa & ~pb).sum())
    return {"n": n, "ka": int(pa.sum()), "kb": int(pb.sum()), "diff": diff, "a": aa, "b": bb, "c": cc, "d": dd,
            "lbw95": newcombe_lb(aa, bb, cc, dd, 0.05), "keys": m.key.values, "pa": pa.values, "pb": pb.values,
            "seed": m.seed.values}          # v4: the seed label of every pair, for the seed-cluster bootstrap


def h2(board, pol, col="passed", keys=None):
    sw = lambda d: d.sw
    r = paired(board, corpus(pol, 0.25), "ARMB025", sw, col)
    if keys is not None:
        r = paired(board[board.key.isin(keys)], corpus(pol, 0.25), "ARMB025", sw, col)
    w2 = lambda k, n: "[%.1f, %.1f]" % tuple(100 * v for v in proportion_confint(k, n, alpha=0.05, method="wilson"))
    nine = paired(board, corpus(pol, 0.25), "ARMB025", lambda d: d.wind_rung, col)
    sd = r["seed"]
    blo, bhi = BR.diff_ci(r)                                    # v4 range of record
    nlo, nhi = newcombe_lb(r["a"], r["b"], r["c"], r["d"], 0.025), -newcombe_lb(r["a"], r["c"], r["b"], r["d"], 0.025)
    return {"k": r["ka"], "kpid": r["kb"], "n": r["n"], "diff": round(r["diff"], 2), "lbw95": round(r["lbw95"], 2),
            "ci": "[%+.1f, %+.1f]" % (blo, bhi), "ci_newcombe": "[%+.1f, %+.1f]" % (nlo, nhi),
            "disc": "%d / %d" % (r["b"], r["c"]), "rate": "%.1f %% [%.1f, %.1f]" % ((100 * r["ka"] / r["n"],) + BR.rate_ci(sd, r["pa"])),
            "pidrate": "%.1f %% [%.1f, %.1f]" % ((100 * r["kb"] / r["n"],) + BR.rate_ci(sd, r["pb"])),
            "rate_wilson": "%.1f %% %s" % (100 * r["ka"] / r["n"], w2(r["ka"], r["n"])),
            "pidrate_wilson": "%.1f %% %s" % (100 * r["kb"] / r["n"], w2(r["kb"], r["n"])),
            "pooled9": "%+.1f pp, lower bound %+.1f" % (nine["diff"], nine["lbw95"]),
            "verdict": "MET" if 100 * (r["ka"] - r["kb"]) >= H2_BAR * r["n"] else "NOT MET"}


def h3(board, arm, col="passed", keys=None):
    a, b = corpus(arm, 0.25), corpus(arm, 0.35)
    r = paired(board if keys is None else board[board.key.isin(keys)], a, b, lambda d: d.wind_rung, col)
    dec = r["diff"]                                             # sigma 0.25 minus sigma 0.35: positive = decrease
    lo = newcombe_lb(r["a"], r["b"], r["c"], r["d"], 0.025)     # two-sided 95 %: lower bound of the decrease
    hi = -newcombe_lb(r["a"], r["c"], r["b"], r["d"], 0.025)    # upper bound = minus the lower bound of the reversed difference
    swr = paired(board, a, b, lambda d: d.sw, col)
    blo, bhi = BR.diff_ci(r)                                    # v4 range of record; Newcombe kept for comparison
    return {"k25": r["ka"], "k35": r["kb"], "decrease": round(dec, 2), "ci": "[%.1f, %.1f]" % (blo, bhi),
            "ci_newcombe": "[%.1f, %.1f]" % (lo, hi),
            "disc": "%d / %d" % (r["b"], r["c"]), "decrease_sw": round(swr["diff"], 2), "verdict": "MET" if dec <= H3_BAR else "NOT MET"}


def placeholders(board, agree=None):
    T = {}
    for arm in ORDER:
        for sig, tag in ((0.25, "s025"), (0.35, "s035")):
            r = h1(board, arm, sig)
            T.update({"H1.p95.%s.%s" % (arm, tag): "%.3f m" % r["p95"], "H1.p95ci.%s.%s" % (arm, tag): r["ci"],
                      "H1.stagesover.%s.%s" % (arm, tag): r["stages_over"], "H1.stageover.detail.%s" % arm: r["stage_detail"] if sig == 0.25 else T.get("H1.stageover.detail.%s" % arm),
                      "H1.legsover.%s.%s" % (arm, tag): r["legs_over"], "H1.contact.%s.%s" % (arm, tag): r["contact"],
                      "H1.nocorridor.%s.%s" % (arm, tag): r["never"], "H1.p95whole.%s.%s" % (arm, tag): "%.3f m" % r["p95_whole"],
                      "H1.p95wholeci.%s.%s" % (arm, tag): r["whole_ci"], "H3.p95ci.%s.%s" % (arm, tag): r["ci"],
                      "SAFE.overtake.%s.%s" % (arm, tag): r["overtake"], "H3.p95.%s.%s" % (arm, tag): "%.3f m" % r["p95"],
                      "H3.stagesover.%s.%s" % (arm, tag): r["stages_over"], "H3.legsover.%s.%s" % (arm, tag): r["legs_over"]})
            if arm in POL:
                T["H1.verdict.%s" % arm] = "%s %s H1." % (NAME[arm], "met" if r["verdict"] == "MET" else "did not meet") if sig == 0.25 else T["H1.verdict.%s" % arm]
        s0 = board[(board.corpus == corpus(arm, 0.25)) & (board.wind_label == "S0")]
        T["S0.k.%s" % arm] = int(s0.passed.sum())
        # Calm-control repeatability ("swing"): the stage is flown once inside the arm's sigma 0.25 corpus and again
        # inside its sigma 0.35 corpus -- same 60 legs, same 30 seeds x 2 trajectories, turbulence off both times.
        # The swing is the SECOND flight minus the FIRST, in points. paired() returns first minus second, hence -diff.
        rep = paired(board, corpus(arm, 0.25), corpus(arm, 0.35), lambda d: d.wind_label == "S0")
        if rep["n"] != 60:
            raise SystemExit("REFUSE: %s calm control stage pairs %d legs, expected 60" % (arm, rep["n"]))
        # paired() gives (first - second); the swing is (second - first), so the interval is the NEGATED, SWAPPED pair.
        d_lo, d_hi = newcombe_lb(rep["a"], rep["b"], rep["c"], rep["d"], 0.025), -newcombe_lb(rep["a"], rep["c"], rep["b"], rep["d"], 0.025)
        lo, hi = -d_hi, -d_lo
        if not lo <= -rep["diff"] <= hi:
            raise SystemExit("REFUSE: %s calm swing %.1f is outside its own interval [%.1f, %.1f]" % (arm, -rep["diff"], lo, hi))
        T.update({"S0.swing.%s" % arm: "%+.1f" % -rep["diff"], "S0.swing.ci.%s" % arm: "[%+.1f, %+.1f]" % (lo, hi),
                  "S0.k2.%s" % arm: rep["kb"], "S0.flipped.%s" % arm: rep["b"] + rep["c"]})
    for arm in ORDER:
        r = h3(board, arm)
        T.update({"H3.k.%s.s025" % arm: r["k25"], "H3.k.%s.s035" % arm: r["k35"], "H3.decrease.%s" % arm: "%.1f pp" % r["decrease"],
                  "H3.decrease.ci.%s" % arm: r["ci"], "H3.decrease.ci.newcombe.%s" % arm: r["ci_newcombe"],
                  "H3.decrease.sw.%s" % arm: "%.1f pp" % r["decrease_sw"], "H3.disc.%s" % arm: r["disc"]})
        if arm in POL:
            T["H3.verdict.%s" % arm] = "%s %s H3." % (NAME[arm], "met" if r["verdict"] == "MET" else "did not meet")
    h2_placeholders(T, board, agree)
    for arm in NOVERDICT:                       # the comparator's own H2 difference, so every arm has the same cell
        r = paired(board, corpus(arm, 0.25), "ARMB025", lambda d: d.sw)
        T["H2.diff.%s" % arm] = "%+.2f pp" % r["diff"]
        T["H2.diff.ci.%s" % arm] = "[%+.1f, %+.1f]" % BR.diff_ci(r)
    T.update(PB.stamp())
    for h in ("H1", "H2", "H3"):
        met = [" met " in T["%s.verdict.%s" % (h, p)] for p in POL]
        T["%s.verdict.both" % h] = "%d of the four FlightDock configurations met %s" % (sum(met), h)
        T["%s.verdict.all" % h] = "; ".join(T["%s.verdict.%s" % (h, p)].rstrip(".") for p in POL) + "."
    return T


def h2_placeholders(T, board, agree):
    """H2 verdict statistic per FlightDock arm, and the H2 plan's section-5 diagnostics (decide nothing)."""
    wil = lambda k, n: "%.1f %% [%.1f, %.1f]" % ((100.0 * k / n,) + tuple(100 * v for v in proportion_confint(k, n, alpha=0.05, method="wilson")))
    for pol in POL:
        for col, suf in (("passed", ""), ("passed_exam", ".exam")):
            r = h2(board, pol, col)
            T.update({"H2.k.%s.sw%s" % (pol, suf): r["k"], "H2.k.PID.sw%s" % suf: r["kpid"], "H2.diff.%s%s" % (pol, suf): "%+.2f pp" % r["diff"],
                      "H2.diff.ci.%s%s" % (pol, suf): r["ci"], "H2.diff.ci.newcombe.%s%s" % (pol, suf): r["ci_newcombe"],
                      "H2.rate.wilson.%s.sw%s" % (pol, suf): r["rate_wilson"], "H2.pidrate.wilson.sw%s" % suf: r["pidrate_wilson"],
                      "H2.lbw95.%s%s" % (pol, suf): "%+.2f pp" % r["lbw95"], "H2.disc.%s%s" % (pol, suf): r["disc"],
                      "H2.rate.%s.sw%s" % (pol, suf): r["rate"], "H2.pidrate.sw%s" % suf: r["pidrate"], "H2.pooled9.%s%s" % (pol, suf): r["pooled9"]})
            if pol in POL and col == "passed":
                T["H2.verdict.%s" % pol] = "%s %s H2." % (NAME[pol], "met" if r["verdict"] == "MET" else "did not meet")
        T["H2.kwhole.%s.sw" % pol] = T["H2.k.%s.sw.exam" % pol]
    for arm in ORDER:                           # strong-wind rate overall (no-verdict arms) and by trajectory (every arm)
        x = wind(board, arm, 0.25)
        x = x[x.sw]
        if arm in NOVERDICT:                    # reported with the same statistic, no verdict
            T["H2.swrate.%s" % arm] = BR.rate(x.seed.to_numpy(), x.passed.astype(bool))
            T["H2.swrate.wilson.%s" % arm] = wil(int(x.passed.sum()), len(x))
        for traj, g in x.groupby("trajectory"):
            T["H2.traj.%s.%s" % (arm, traj)] = "%d of %d, %s" % (int(g.passed.sum()), len(g), wil(int(g.passed.sum()), len(g)))
    if agree is not None:                       # strong-wind pool: all pairs beside the gust-agreeing pairs
        for pol in POL:
            k = agree[(agree.left == corpus(pol, 0.25)) & (agree.right == "ARMB025") & agree.agree].key.values
            full, sub = h2(board, pol), h2(board, pol, keys=k)
            T["H2.lbw95.%s.gustagree" % pol] = "%+.2f pp (lower bound %+.2f pp) on the %d of %d strong-wind pairs that agree (%.1f %%)" % (
                sub["diff"], sub["lbw95"], sub["n"], full["n"], 100.0 * sub["n"] / full["n"])


def md_table(df):
    """Markdown table without the optional tabulate dependency. An empty cell prints "n/a", never a bare "nan"."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("n/a" if (v != v or v is None or v == "") else str(v) for v in r) + " |")
    return "\n".join(lines)


def selftest():
    keys = ["W12|%d|s" % (10775 + i) for i in range(20)]
    sd = [10775 + i for i in range(20)]
    pa = [True] * 12 + [False] * 8
    pb = [True] * 6 + [False] * 6 + [True] * 4 + [False] * 4     # a = 6 both, b = 6 policy only, c = 4 PID only, d = 4
    board = pd.concat([pd.DataFrame({"key": keys, "seed": sd, "corpus": "A2P025", "passed": pa, "sw": True, "wind_rung": True}),
                       pd.DataFrame({"key": keys, "seed": sd, "corpus": "ARMB025", "passed": pb, "sw": True, "wind_rung": True})])
    r = paired(board, "A2P025", "ARMB025", lambda d: d.sw)
    ok = abs(r["diff"] - 10.0) < 1e-9 and (r["a"], r["b"], r["c"], r["d"]) == (6, 6, 4, 4) and -100 < r["lbw95"] < 10.0
    ok = ok and list(r["seed"]) == sd and BR.diff_ci({"seed": sd, "pa": r["pa"], "pb": r["pa"]}) == (0.0, 0.0)
    ok = ok and abs(newcombe_lb(6, 6, 4, 4, 0.05) - r["lbw95"]) < 1e-9
    try:
        paired(board, "A2P025", "CSV025", lambda d: d.sw)
        ok = False
    except SystemExit:
        pass
    PB.BOOT = 50
    T = placeholders(_full_fixture())
    ok = ok and T["H3.p95ci.A2.s035"] == T["H1.p95ci.A2.s035"] and T["H1.p95wholeci.PIDO.s025"].startswith("[")
    ok = ok and T["BOOT.seed"] == PB.SEED and T["H2.traj.PID.straight"].startswith("63 of 90") and "H2.swrate.PIDO" in T
    # 2026-09-18: the exam-flag pass overwrote H2.pidrate.sw (key had no flag suffix); record and exam rates differ here
    ok = ok and T["H2.pidrate.sw"] == T["H2.swrate.PID"] and T["H2.pidrate.sw"] != T["H2.pidrate.sw.exam"]
    ok = ok and T["H2.diff.ci.A2"].startswith("[") and T["H3.decrease.ci.newcombe.A2"].startswith("[") and "H2.swrate.wilson.PIDO" in T
    print("SELFTEST", "PASS" if ok else "FAIL", "lbw95=%.2f" % r["lbw95"])
    return 0 if ok else 1


def _full_fixture():
    """A synthetic twelve-corpus board (600 legs each) that drives placeholders() end to end."""
    rows = []
    for arm in ORDER:
        for sigma in (0.25, 0.35):
            for lab in ["S0"] + list(STAGE_MS):
                for seed in range(10775, 10805):
                    for i, tr in enumerate(("straight", "racetrack")):
                        win = 1.0 + 0.01 * (seed % 30) + (0.2 if sigma == 0.35 else 0.0)
                        rows.append({"corpus": corpus(arm, sigma), "wind_label": lab, "seed": seed, "trajectory": tr,
                                     "key": "%s|%d|%s" % (lab, seed, tr), "wind_rung": lab != "S0", "sw": lab in ("W12", "W14", "W16"),
                                     "passed": seed % 10 < (7 if sigma == 0.25 else 5), "passed_exam": seed % 10 < 6,
                                     "win": win, "whole": win + 0.1, "real_overtake": False})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    bd = os.path.join(a.run_dir, "board")
    board = pd.read_csv(os.path.join(bd, "BOARD_W10.csv"), low_memory=False)
    for c in ("passed", "passed_exam", "sw", "wind_rung", "real_overtake"):
        board[c] = board[c].astype(str).str.lower().eq("true")
    ag = os.path.join(bd, "GUST_AGREEMENT_LEGS.csv")
    T = placeholders(board, pd.read_csv(ag) if os.path.exists(ag) else None)
    vd = os.path.join(a.run_dir, "verdicts")
    os.makedirs(vd, exist_ok=True)
    json.dump(T, open(os.path.join(vd, "PLACEHOLDERS.json"), "w"), indent=1, default=str)
    # Every arm carries its MET / NOT MET mark. POL arms' marks are the VERDICTS of record;
    # the PID-baseline's and PIDO's are shown for comparison and judge nothing, hence the "(reference)" tag.
    def mark(h, a):
        ok = {"H1": lambda: float(T["H1.p95.%s.s025" % a].split()[0]) <= BAR,
              "H2": lambda: float(T["H2.diff.%s" % a].split()[0]) >= H2_BAR,
              "H3": lambda: float(T["H3.decrease.%s" % a].split()[0]) <= H3_BAR}[h]()
        return ("MET" if ok else "NOT MET") + ("" if a in POL else " (reference)")

    stat = {"H1": lambda a: "%s %s -- %s" % (T["H1.p95.%s.s025" % a], T["H1.p95ci.%s.s025" % a], mark("H1", a)),
            "H2": lambda a: "%s %s -- %s" % (T["H2.diff.%s" % a], T["H2.diff.ci.%s" % a], mark("H2", a)),
            "H3": lambda a: "%s %s -- %s" % (T["H3.decrease.%s" % a], T["H3.decrease.ci.%s" % a], mark("H3", a))}
    desc = {"H1": ("corridor p95 of the peak upward deviation, nine sigma (σ) 0.25 stages pooled", "<= 2.00 m"),
            "H2": ("paired strong-wind (6-8 m/s) difference vs the PID-baseline [two-sided 95 % seed-cluster bootstrap]", ">= +15 pp"),
            "H3": ("decrease in the pooled safe-capture rate, sigma (σ) 0.25 -> 0.35", "<= 35 pp")}
    rows = [(h, desc[h][0], desc[h][1]) + tuple(stat[h](a) for a in ORDER) + (T["%s.verdict.all" % h],) for h in ("H1", "H2", "H3")]
    tab = pd.DataFrame(rows, columns=["hypothesis", "deciding statistic", "threshold"] + [NAME[a] for a in ORDER] + ["verdict"])
    tab.to_csv(os.path.join(vd, "HYPOTHESIS_VERDICTS.csv"), index=False)
    open(os.path.join(vd, "HYPOTHESIS_VERDICTS.md"), "w").write("Channel: truth, inside the aerial-docking terminal-approach corridor (<= 10.0 m of the boom-tip). "
                                                        "Every arm carries its MET / NOT MET mark; the PID-baseline's and PIDO's are tagged (reference) because they are "
                                                        "reported arms and judge nothing. H2 is measured AGAINST the PID-baseline, so the PID-baseline's own H2 value is "
                                                        "+0.00 pp against itself and PIDO's is the reading of the 0.25 m offset alone.\n\n"
                                                        + md_table(tab) + "\n")
    print(tab.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
