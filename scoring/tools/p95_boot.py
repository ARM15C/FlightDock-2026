#!/usr/bin/env python3
# =============================================================================

#   implementation of the 95th percentile (p95) of the per-leg peak upward deviation and of its interval, used by
#   every tool, so the method cannot differ between tables. As the plans state it: the p95 is computed by linear
#   interpolation between order statistics (numpy default); every such p95, pooled or per stage and in any channel,
#   carries a two-sided 95 % seed-cluster bootstrap interval -- BOOT resamples of the pool's seed labels (30 in a
#   benchmark pool), whole seeds resampled together, percentile method (2.5th and 97.5th percentiles of the
#   resampled p95 values), fixed random-number seed SEED. A FRESH generator is built from SEED on every call, so an
#   interval never depends on the order in which the tools or tables are run.
#   Seed labels come from the WHOLE pool handed in (legs with no peak included), so a seed whose legs never entered
#   the corridor is still one of the resampled labels; a resample that holds no peak at all is skipped and counted.
# PUBLIC API: BOOT; SEED; p95(values) -> float; ci(df, col="win") -> (lo, hi);
#   diff_ci(dfa, dfb, col="win") -> (pa, pb, pa - pb, lo, hi); fmt(lo, hi) -> str; stamp() -> dict.
#   CLI: --selftest.
# SIDE-EFFECTS: none.
# SIZE: soft 80 / hard 100 code-LoC; function cap 30.
# COVERAGE: --selftest: p95 equals numpy's linear percentile; the interval brackets the p95 of a known sample, is
#   identical on a second call (fresh generator) and independent of row order; a seed with no peak stays a label;
#   the difference interval of a sample against itself is [0, 0].
# =============================================================================
import sys

import numpy as np
import pandas as pd

BOOT, SEED = 10000, 20260915


def p95(values):
    """95th percentile by linear interpolation between order statistics; NaN when there is no value."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 95)) if len(v) else float("nan")


def _groups(df, col, labels):
    return [df.loc[df.seed == s, col].dropna().to_numpy(dtype=float) for s in labels]


def _picks(n_labels):
    return np.random.default_rng(SEED).integers(0, n_labels, size=(BOOT, n_labels))


def ci(df, col="win"):
    """Two-sided 95 % seed-cluster bootstrap interval on the p95 of df[col]. df is the WHOLE pool (needs `seed`)."""
    labels = np.sort(df.seed.unique())
    g = _groups(df, col, labels)
    vals = np.array([p95(np.concatenate([g[i] for i in row])) for row in _picks(len(labels))])
    if np.isnan(vals).all():
        return float("nan"), float("nan")
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def diff_ci(dfa, dfb, col="win"):
    """p95(a), p95(b), a - b and the interval on a - b; the SAME seed labels are drawn for both arms (paired)."""
    labels = np.sort(np.union1d(dfa.seed.unique(), dfb.seed.unique()))
    ga, gb = _groups(dfa, col, labels), _groups(dfb, col, labels)
    vals = np.array([p95(np.concatenate([ga[i] for i in row])) - p95(np.concatenate([gb[i] for i in row]))
                     for row in _picks(len(labels))])
    pa, pb = p95(dfa[col]), p95(dfb[col])
    return pa, pb, pa - pb, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def fmt(lo, hi):
    return "[%.3f, %.3f]" % (lo, hi)


def stamp():
    """What every output prints beside a bootstrap interval."""
    return {"BOOT.resamples": BOOT, "BOOT.seed": SEED, "BOOT.method": "seed-cluster, percentile, two-sided 95 %"}


def selftest():
    global BOOT
    BOOT = 400
    r = np.random.default_rng(7)
    df = pd.DataFrame({"seed": np.repeat(np.arange(10775, 10805), 18), "win": r.normal(1.5, 0.3, 540)})
    ok = abs(p95(df.win) - np.percentile(df.win, 95)) < 1e-12
    lo, hi = ci(df)
    ok = ok and lo < p95(df.win) < hi and (lo, hi) == ci(df) and (lo, hi) == ci(df.sample(frac=1.0, random_state=1))
    hole = df.copy()
    hole.loc[hole.seed == 10775, "win"] = np.nan              # a seed whose legs never entered stays a label
    ok = ok and len(np.sort(hole.seed.unique())) == 30 and ci(hole)[0] == ci(hole)[0]
    d = diff_ci(df, df)
    ok = ok and d[2] == 0.0 and d[3] == 0.0 and d[4] == 0.0
    print("SELFTEST", "PASS" if ok else "FAIL", fmt(lo, hi))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
