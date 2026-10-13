#!/usr/bin/env python3
# =============================================================================
# HYPOTHESIS ALTERNATIVES, v4. EXPLORATORY: changes nothing of
#   record. Reads board/BOARD_W10.csv of the 09-18 scoring run and recomputes the three hypotheses under the alternative
#   bars examined. THE COMPUTED VALUE DECIDES every verdict here; intervals are
#   information. Sigma set confirmed by the operator (T5): sigma 0.25 for H1 and H2, sigma 0.25 -> 0.35 for H3; the
#   operator also asked for an H2 reading at sigma 0.35.
#   H1: corridor percentile of the peak upward deviation (truth, corridor <= 10 m, nine wind stages), p95 AND p90,
#       bar <= 2.00 m, each with the seed-cluster bootstrap interval (30 seeds, PB.BOOT resamples, percentile, PB.SEED).
#   H2: same-seed paired difference in docking-envelope captures against the PID-baseline, strong-wind pool (180 pairs)
#       AND nine stages pooled (540 pairs), at sigma 0.25 AND sigma 0.35; bars >= +15 pp, >= +5 pp, > 0 pp; the
#       Newcombe (1998) two-sided 95 % interval AND a seed-cluster bootstrap two-sided 95 % interval.
#   H3: decrease in captures sigma 0.25 -> sigma 0.35 on the same arm, nine stages (540 pairs) and strong wind (180);
#       bars <= 35 pp and <= 40 pp; both intervals.
#   The PID-baseline and PIDO get NO verdict; their values are shown as references.
# SELF-CHECK (the script stops if it fails, so a misread board can never produce a table): the p95, its interval, the
#   strong-wind H2 difference and the H3 decrease and interval equal verdicts/PLACEHOLDERS.json; seed block 10775-10804.
# Bar comparisons on capture counts are made in INTEGERS (100*(ka-kb) against bar*n), so a tie cannot be lost to rounding.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, (_REPO + "/analysis/pvalue_supplement_20260920"))
import pvalue_supplement as P  # noqa: E402

S, V = P.S, P.V
import compare_arms as CA  # noqa: E402
import p95_boot as PB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "v4_20260920"
POOLS = [("strong wind (6-8 m/s)", "sw"), ("nine wind stages", "wind_rung")]
H1_BAR, H2_BARS, H3_BARS = 2.00, (15, 5, 0), (35, 40)


def pct_ci(df, qs):
    """Percentiles qs of df.win and their seed-cluster bootstrap intervals, PB's own labels, groups and picks."""
    labels = np.sort(df.seed.unique())
    g = PB._groups(df, "win", labels)
    vals = np.array([np.percentile(np.concatenate([g[i] for i in row]), qs) for row in PB._picks(len(labels))])
    v = df.win.dropna().to_numpy(dtype=float)
    return np.percentile(v, qs), np.percentile(vals, 2.5, axis=0), np.percentile(vals, 97.5, axis=0)


def boot_diff(bd, r, corp_a):
    """Seed-cluster bootstrap two-sided 95 % interval on a paired difference (pp): whole seeds, PB's picks."""
    seed = bd[bd.corpus == corp_a].set_index("key").seed.reindex(r["keys"]).to_numpy()
    labels = np.sort(np.unique(seed))
    if list(labels) != S.SEED_BLOCK:
        raise SystemExit("STOP: paired keys do not cover seed block 10775-10804")
    d = np.array([(r["pa"][seed == s].sum() - r["pb"][seed == s].sum()) for s in labels], dtype=float)
    n = np.array([(seed == s).sum() for s in labels], dtype=float)
    picks = PB._picks(len(labels))
    vals = 100.0 * d[picks].sum(axis=1) / n[picks].sum(axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def h1_rows(bd, ph):
    rows = []
    for sig, tag in ((0.25, "s025"), (0.35, "s035")):
        for arm in S.ARMS:
            x = V.wind(bd, arm, sig)
            (p90, p95), lo, hi = pct_ci(x, [90, 95])
            if "%.3f m" % round(p95, 3) != ph["H1.p95.%s.%s" % (arm, tag)] or PB.fmt(lo[1], hi[1]) != ph["H1.p95ci.%s.%s" % (arm, tag)]:
                raise SystemExit("STOP: H1 p95 %s %s does not reproduce the scoring run" % (arm, sig))
            ent = x.win.dropna()
            for q, p, l, h in (("p95", p95, lo[1], hi[1]), ("p90", p90, lo[0], hi[0])):
                rows.append({"sigma": sig, "arm": V.NAME[arm], "statistic": "corridor " + q, "value_m": round(p, 3), "boot_lo": round(l, 3), "boot_hi": round(h, 3),
                             "legs_in_corridor": len(ent), "legs_over_2.00m": int((ent > H1_BAR).sum()), "pct_legs_over": round(100.0 * (ent > H1_BAR).mean(), 1),
                             "value_at_or_under_2.00": bool(p <= H1_BAR), "judged": arm in V.POL and sig == 0.25})
    return pd.DataFrame(rows)


def h2_rows(bd, ph):
    rows = []
    for sig in (0.25, 0.35):
        for pool, col in POOLS:
            for arm in [a for a in S.ARMS if a != "PID"]:
                r = V.paired(bd, V.corpus(arm, sig), V.corpus("PID", sig), lambda d, c=col: d[c])
                if sig == 0.25 and col == "sw" and arm in V.POL and "%+.2f pp" % r["diff"] != ph["H2.diff.%s" % arm]:
                    raise SystemExit("STOP: H2 %s does not reproduce the scoring run" % arm)
                nlo, nhi = CA.two_sided(r)
                blo, bhi = boot_diff(bd, r, V.corpus(arm, sig))
                num = 100 * (r["ka"] - r["kb"])
                row = {"sigma": sig, "pool": pool, "arm": V.NAME[arm], "pairs": r["n"], "arm_captures": r["ka"], "pid_captures": r["kb"],
                       "arm_rate": round(100.0 * r["ka"] / r["n"], 1), "pid_rate": round(100.0 * r["kb"] / r["n"], 1), "diff_pp": round(r["diff"], 2),
                       "newcombe_lo": round(nlo, 1), "newcombe_hi": round(nhi, 1), "boot_lo": round(blo, 1), "boot_hi": round(bhi, 1),
                       "judged": arm in V.POL}
                for bar in H2_BARS:
                    row["value_clears_%+d" % bar] = bool(num > 0) if bar == 0 else bool(num >= bar * r["n"])
                    row["tie_at_%+d" % bar] = bool(num == bar * r["n"])
                rows.append(row)
    return pd.DataFrame(rows)


def h3_rows(bd, ph):
    rows = []
    for pool, col in POOLS[::-1]:
        for arm in S.ARMS:
            a, b = V.corpus(arm, 0.25), V.corpus(arm, 0.35)
            r = V.paired(bd, a, b, lambda d, c=col: d[c])
            nlo, nhi = CA.two_sided(r)
            if col == "wind_rung" and ("%.1f pp" % round(r["diff"], 2) != ph["H3.decrease.%s" % arm] or "[%.1f, %.1f]" % (nlo, nhi) != ph["H3.decrease.ci.%s" % arm]):
                raise SystemExit("STOP: H3 %s does not reproduce the scoring run" % arm)
            blo, bhi = boot_diff(bd, r, a)
            num = 100 * (r["ka"] - r["kb"])
            row = {"pool": pool, "arm": V.NAME[arm], "pairs": r["n"], "captures_s025": r["ka"], "captures_s035": r["kb"],
                   "rate_s025": round(100.0 * r["ka"] / r["n"], 1), "rate_s035": round(100.0 * r["kb"] / r["n"], 1), "decrease_pp": round(r["diff"], 2),
                   "newcombe_lo": round(nlo, 1), "newcombe_hi": round(nhi, 1), "boot_lo": round(blo, 1), "boot_hi": round(bhi, 1), "judged": arm in V.POL}
            for bar in H3_BARS:
                row["value_at_or_under_%d" % bar] = bool(num <= bar * r["n"])
                row["tie_at_%d" % bar] = bool(num == bar * r["n"])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    bd = P.board()
    ph = json.load(open(os.path.join(S.RUN, "verdicts", "PLACEHOLDERS.json")))
    out = {"H1": h1_rows(bd, ph), "H2": h2_rows(bd, ph), "H3": h3_rows(bd, ph)}
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print("SELF-CHECK PASS: p95, H2 strong-wind difference and H3 decrease reproduce the 09-18 scoring run; %s" % PB.stamp())
    for k, df in out.items():
        df.to_csv(os.path.join(HERE, "%s_ALTERNATIVES_%s.csv" % (k, TAG)), index=False)
        print("\n==== %s ====" % k)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
