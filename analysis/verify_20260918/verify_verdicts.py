#!/usr/bin/env python3
# =============================================================================

#   deciding numbers of the signed plans from the board of record with code written separately from
#   hypothesis_verdicts.py, and compare to that tool's PLACEHOLDERS.json.
#   H1 p95 (numpy linear interpolation) of the corridor peak `win`, nine wind stages at sigma 0.25, legs with a peak.
#   H2 strong-wind (6-8 m/s, sigma 0.25) pairs on (seed, stage, trajectory) key against the PID-baseline; paired
#       difference; one-sided 95 % lower bound = lower limit of the two-sided 90 % Newcombe (1998) method-10 interval,
#       Wilson limits from statsmodels.proportion_confint(method="wilson").
#   H3 pooled nine-stage rate at sigma 0.25 minus at sigma 0.35, flag of record `passed`.
#   Also recomputes the PID-baseline and PIDO strong-wind rates both ways (flag of record, exam flag) to settle which
#   one the tool's two PID keys carry.
# CLI: python verify_verdicts.py --run-dir DIR --out DIR SIDE-EFFECTS: writes CSV under --out only.
# =============================================================================
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

ARMS = {"A2": ("A2P025", "A2P035"), "A2O": ("A2V25", "A2V35"), "S1": ("S1A25", "S1A35"),
        "S1O": ("CSV025", "CSV035"), "PID": ("ARMB025", "ARMB035"), "PIDO": ("PIDO25", "PIDO35")}


def newcombe10(a, b, c, d, conf):
    """Two-sided `conf` interval for p1 - p2 on paired binary data (Newcombe 1998, method 10)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = proportion_confint(a + b, n, alpha=1 - conf, method="wilson")
    l2, u2 = proportion_confint(a + c, n, alpha=1 - conf, method="wilson")
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / np.sqrt(den) if den > 0 else 0.0
    dd = p1 - p2
    lo = dd - np.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = dd + np.sqrt((p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2)
    return dd, lo, hi


def num(s):
    return float(re.search(r"[-+]?\d+\.\d+", s).group())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    b = pd.read_csv(os.path.join(a.run_dir, "board", "BOARD_W10.csv"))
    ph = json.load(open(os.path.join(a.run_dir, "verdicts", "PLACEHOLDERS.json")))
    w = b[b.wind_rung]
    rows = []
    for arm, (c25, c35) in ARMS.items():
        x25, x35 = w[w.corpus == c25], w[w.corpus == c35]
        pk = x25.win.dropna().to_numpy()
        h1 = float(np.percentile(pk, 95, method="linear"))
        rows.append(("H1 p95 (m)", arm, len(pk), h1, num(ph["H1.p95.%s.s025" % arm])))
        r25, r35 = 100 * x25.passed.mean(), 100 * x35.passed.mean()
        rows.append(("H3 decrease (pp)", arm, len(x25), r25 - r35, num(ph["H3.decrease.%s" % arm])))
        if arm in ("PID", "PIDO"):
            s = x25[x25.sw]
            rows.append(("strong-wind rate, flag of record (%)", arm, len(s), 100 * s.passed.mean(),
                         num(ph["H2.swrate.%s" % arm])))
            if arm == "PID":
                rows.append(("strong-wind rate, placeholder pidrate.sw (%)", arm, len(s), 100 * s.passed.mean(), num(ph["H2.pidrate.sw"])))
            rows.append(("strong-wind rate, exam flag (%)", arm, len(s), 100 * s.passed_exam.mean(), np.nan))
            continue
        m = x25[x25.sw].merge(w[(w.corpus == "ARMB025") & w.sw], on="key", suffixes=("", "_pid"))
        f, g = m.passed.astype(bool), m.passed_pid.astype(bool)
        cnt = [int((f & g).sum()), int((f & ~g).sum()), int((~f & g).sum()), int((~f & ~g).sum())]
        dd, lo, _ = newcombe10(*cnt, conf=0.90)
        rows.append(("H2 paired difference (pp)", arm, len(m), 100 * dd, num(ph["H2.diff.%s" % arm])))
        rows.append(("H2 one-sided 95 % lower bound (pp)", arm, len(m), 100 * lo, num(ph["H2.lbw95.%s" % arm])))
    t = pd.DataFrame(rows, columns=["quantity", "arm", "n", "independent", "tool"])
    t["abs_diff"] = (t.independent - t.tool).abs()
    # the tool prints H1 to 0.001 m, H2 to 0.01 pp, H3 and rates to 0.1: agreement means within half a unit
    unit = t.quantity.map(lambda q: 0.0005 if "H1" in q else 0.005 if "H2" in q else 0.05)
    t["agrees"] = np.where(t.tool.isna(), "n/a", np.where(t.abs_diff <= unit + 1e-12, "yes", "NO"))
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "STEP12_INDEPENDENT_RECOMPUTE.csv"), index=False)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nz for one-sided 95 %% = two-sided 90 %%: %.6f" % norm.ppf(0.95))
    print("tool H2.pidrate.sw = %s; tool H2.k.PID.sw = %s; tool H2.k.PID.sw.exam = %s"
          % (ph["H2.pidrate.sw"], ph["H2.k.PID.sw"], ph["H2.k.PID.sw.exam"]))
    bad = t[t.agrees == "NO"]
    print("DISAGREEMENTS:", len(bad))
    return 1 if len(bad) else 0


if __name__ == "__main__":
    sys.exit(main())
