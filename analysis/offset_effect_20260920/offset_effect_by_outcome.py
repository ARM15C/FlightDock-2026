#!/usr/bin/env python3
# =============================================================================
# OFFSET EFFECT BY OUTCOME. AMPLIFYING; decides nothing. For each controller family
#   (A2 -> A2O, S1 -> S1O, PID-baseline -> PIDO): the change from adding the 0.25 m offset in
#   (1) docking-envelope captures (same-seed paired difference, two-sided 95 % Newcombe 1998 interval) and
#   (2) the corridor p95 of the peak upward deviation (seed-cluster bootstrap interval, the tools' own PB.diff_ci),
#   at sigma 0.25 and 0.35, for the nine wind stages pooled and the strong-wind pool (6-8 m/s), for all legs and split by
#   trajectory (straight, racetrack).
# CLASS RULE (stated, mine): "helped"/"hurt" only when the 95 % interval excludes zero; otherwise "no clear change".
#   Captures: up = helped. Corridor p95: down = helped.
# VALIDATION, ENFORCED (refuses): all-trajectory capture rows reproduce comparative/ARM_PAIRS.csv (difference, pairs,
#   discordant); all-trajectory nine-stage p95 rows reproduce ARM_PAIRS.csv (difference and interval); the straight and
#   racetrack capture counts sum to the all-trajectory counts; seed block 10775-10804.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import os
import sys

import pandas as pd

sys.path.insert(0, (_REPO + "/analysis/pvalue_supplement_20260920"))
import pvalue_supplement as P  # noqa: E402

S, V = P.S, P.V
import compare_arms as CA  # noqa: E402
import p95_boot as PB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FAM = [("A2O", "A2"), ("S1O", "S1"), ("PIDO", "PID")]
POOLS = [("nine wind stages", "wind_rung"), ("strong-wind (6-8 m/s)", "sw")]


def klass(lo, hi, up_is_good):
    if lo > 0:
        return "helped" if up_is_good else "hurt"
    if hi < 0:
        return "hurt" if up_is_good else "helped"
    return "no clear change"


def main():
    bd = P.board()
    trajs = sorted(bd.trajectory.dropna().unique())
    pairs = pd.read_csv(os.path.join(S.RUN, "comparative", "ARM_PAIRS.csv"))
    rows = []
    for sig in (0.25, 0.35):
        for pool, col in POOLS:
            for on, off in FAM:
                tot = {}
                for tr in ["all"] + trajs:
                    mask = (lambda d, c=col, t=tr: d[c] & ((d.trajectory == t) if t != "all" else True))
                    r = V.paired(bd, V.corpus(on, sig), V.corpus(off, sig), mask)
                    if tr == "all":
                        P.check(pairs, sig, on, off, pool, r["n"], r)
                    tot[tr] = (r["n"], r["ka"], r["kb"])
                    lo, hi = CA.two_sided(r)
                    rows.append({"outcome": "captures (pp)", "sigma": sig, "pool": pool, "trajectory": tr, "controller": V.NAME[off], "legs_or_pairs": r["n"],
                                 "offset_off": round(100.0 * r["kb"] / r["n"], 1), "offset_on": round(100.0 * r["ka"] / r["n"], 1),
                                 "change": round(r["diff"], 2), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1), "class": klass(lo, hi, True)})
                    x = bd[bd.corpus.isin([V.corpus(on, sig), V.corpus(off, sig)]) & bd[col]]
                    if tr != "all":
                        x = x[x.trajectory == tr]
                    pa, pb, d, plo, phi = PB.diff_ci(x[x.corpus == V.corpus(on, sig)], x[x.corpus == V.corpus(off, sig)], "win")
                    if tr == "all" and col == "wind_rung":
                        q = pairs[(pairs.sigma == sig) & (pairs.arm_a == V.NAME[off]) & (pairs.arm_b == V.NAME[on]) & pairs.pool.str.contains("p95")]
                        if len(q) != 1 or round(-float(q.diff_pp.iloc[0]), 3) != round(d, 3) or q.ci95.iloc[0] != "[%.3f, %.3f]" % (-phi, -plo):
                            raise SystemExit("REFUSE: p95 %s %s does not reproduce ARM_PAIRS.csv (%s vs %.3f [%.3f, %.3f])" % (on, sig, q.to_dict("records"), d, plo, phi))
                    rows.append({"outcome": "corridor p95 (m)", "sigma": sig, "pool": pool, "trajectory": tr, "controller": V.NAME[off],
                                 "legs_or_pairs": int(x[x.corpus == V.corpus(on, sig)].win.notna().sum()),
                                 "offset_off": round(pb, 3), "offset_on": round(pa, 3), "change": round(d, 3), "ci_lo": round(plo, 3), "ci_hi": round(phi, 3),
                                 "class": klass(plo, phi, False)})
                if tuple(sum(tot[t][i] for t in trajs) for i in range(3)) != tot["all"]:
                    raise SystemExit("REFUSE: trajectories do not sum to the pool for %s %s %s" % (on, sig, pool))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "OFFSET_EFFECT_BY_OUTCOME.csv"), index=False)
    print("VALIDATION PASS: %d rows; trajectories %s" % (len(df), trajs))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
