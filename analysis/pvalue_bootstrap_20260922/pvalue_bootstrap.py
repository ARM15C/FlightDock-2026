#!/usr/bin/env python3
# =============================================================================
# SEED-CLUSTER BOOTSTRAP P-VALUES for the revised hypotheses (after the review meeting:
#   "GO with both option A and B. I want to see how they differ."). EXPLORATORY; changes nothing of record.
# Readings (revised hypotheses, all nine wind stages pooled, 0-8 m/s, same seeds 10775-10804):
#   H1 sigma 0.25, corridor p95 of the truth peak upward deviation <= 2.00 m. H0: p95 >= 2.00 m.
#   H2 sigma 0.25, same-seed paired difference in captures vs PID-baseline > 0. H0: difference <= 0 pp.
#   H3 decrease in captures sigma 0.25 -> 0.35, same arm, <= 35 pp. H0: decrease >= 35 pp.
# Method: the SAME resamples as every interval of record (whole seeds, PB._picks: 10,000 resamples, fixed seed 20260915,
#   percentile method). One-sided bootstrap p = share of resamples on the H0 side of the bar (ties count toward H0).
#   Option A: two-sided 95 % interval clears the bar <=> one-sided alpha 0.025.
#   Option B: one-sided 95 % bound (two-sided 90 % interval) clears the bar <=> one-sided alpha 0.05.
# SELF-CHECK, ENFORCED: every 2.5/97.5 interval reproduces the v4 alternatives CSVs; seed block 10775-10804. Any miss stops the script.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import os
import sys

import numpy as np
import pandas as pd

W = (_REPO + "/analysis/whatif_20260920")
sys.path.insert(0, W)
import hypothesis_alternatives_v4 as HA  # noqa: E402

S, V, PB, P = HA.S, HA.V, HA.PB, HA.P
HERE = os.path.dirname(os.path.abspath(__file__))
JUDGED = ["A2", "A2O", "S1", "S1O"]


def v4(k):
    return pd.read_csv(os.path.join(W, "%s_ALTERNATIVES_v4_20260920.csv" % k))


def p95_vals(df):
    labels = np.sort(df.seed.unique())
    if list(labels) != S.SEED_BLOCK:
        raise SystemExit("STOP: H1 pool does not cover seed block 10775-10804")
    g = PB._groups(df, "win", labels)
    vals = np.array([np.percentile(np.concatenate([g[i] for i in row]), 95) for row in PB._picks(len(labels))])
    return float(np.percentile(df.win.dropna().to_numpy(dtype=float), 95)), vals


def diff_vals(bd, r, corp_a):
    seed = bd[bd.corpus == corp_a].set_index("key").seed.reindex(r["keys"]).to_numpy()
    labels = np.sort(np.unique(seed))
    if list(labels) != S.SEED_BLOCK:
        raise SystemExit("STOP: paired keys do not cover seed block 10775-10804")
    d = np.array([(r["pa"][seed == s].sum() - r["pb"][seed == s].sum()) for s in labels], dtype=float)
    n = np.array([(seed == s).sum() for s in labels], dtype=float)
    picks = PB._picks(len(labels))
    return 100.0 * d[picks].sum(axis=1) / n[picks].sum(axis=1)


def row(h, arm, value, vals, bar, upper_is_good, dp, ref_lo, ref_hi, extra=None):
    q = lambda x: float(np.percentile(vals, x))
    lo95, hi95, lo90, hi90 = q(2.5), q(97.5), q(5), q(95)
    rdp = 3 if h == "H1" else 1  # the v4 CSVs store H1 to 3 dp, H2/H3 to 1 dp
    if (round(lo95, rdp), round(hi95, rdp)) != (ref_lo, ref_hi):
        raise SystemExit("STOP: %s %s interval [%.4f, %.4f] does not reproduce v4 [%s, %s]" % (h, arm, lo95, hi95, ref_lo, ref_hi))
    if upper_is_good:  # H2: good side is above the bar
        p = float(np.mean(vals <= bar)); a_ok, b_ok, pe = lo95 > bar, lo90 > bar, value > bar
    else:              # H1, H3: good side is at or under the bar
        p = float(np.mean(vals >= bar)); a_ok, b_ok, pe = hi95 <= bar, hi90 <= bar, value <= bar
    out = {"hypothesis": h, "arm": arm, "judged": arm in JUDGED, "value": round(value, dp), "bar": bar,
           "ci95_lo": round(lo95, dp), "ci95_hi": round(hi95, dp), "ci90_lo": round(lo90, dp), "ci90_hi": round(hi90, dp),
           "boot_p_one_sided": p, "point_estimate_clears": bool(pe),
           "A_interval_clears": bool(a_ok), "A_p_below_0.025": bool(p < 0.025),
           "B_interval_clears": bool(b_ok), "B_p_below_0.05": bool(p < 0.05)}
    out.update(extra or {})
    return out


def main():
    bd = P.board()
    h1v, h2v, h3v = v4("H1"), v4("H2"), v4("H3")
    rows = []
    for arm in S.ARMS:
        x = V.wind(bd, arm, 0.25)
        val, vals = p95_vals(x)
        ref = h1v[(h1v.sigma == 0.25) & (h1v.arm == V.NAME[arm]) & (h1v.statistic == "corridor p95")].iloc[0]
        rows.append(row("H1", arm, val, vals, 2.00, False, 3, ref.boot_lo, ref.boot_hi))
    for arm in [a for a in S.ARMS if a != "PID"]:
        r = V.paired(bd, V.corpus(arm, 0.25), V.corpus("PID", 0.25), lambda d: d.wind_rung)
        vals = diff_vals(bd, r, V.corpus(arm, 0.25))
        ref = h2v[(h2v.sigma == 0.25) & (h2v.pool == "nine wind stages") & (h2v.arm == V.NAME[arm])].iloc[0]
        rows.append(row("H2", arm, r["diff"], vals, 0.0, True, 2, ref.boot_lo, ref.boot_hi,
                        {"pairs": r["n"], "discordant_arm_only": r["b"], "discordant_pid_only": r["c"]}))
    for arm in S.ARMS:
        r = V.paired(bd, V.corpus(arm, 0.25), V.corpus(arm, 0.35), lambda d: d.wind_rung)
        vals = diff_vals(bd, r, V.corpus(arm, 0.25))
        ref = h3v[(h3v.pool == "nine wind stages") & (h3v.arm == V.NAME[arm])].iloc[0]
        rows.append(row("H3", arm, r["diff"], vals, 35.0, False, 2, ref.boot_lo, ref.boot_hi, {"pairs": r["n"]}))
    df = pd.DataFrame(rows)
    df["arm"] = df.arm.map(lambda a: {"PID": "PID-baseline"}.get(a, a))
    df["A_interval_vs_p_agree"] = df["A_interval_clears"] == df["A_p_below_0.025"]
    df["B_interval_vs_p_agree"] = df["B_interval_clears"] == df["B_p_below_0.05"]
    df.to_csv(os.path.join(HERE, "PVALUE_BOOTSTRAP_20260922.csv"), index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    print("SELF-CHECK PASS: all intervals reproduce v4;", PB.stamp())
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
