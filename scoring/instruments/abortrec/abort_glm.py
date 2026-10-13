#!/usr/bin/env python3
# THE CORRELATION READ.
# RESPONSIBILITY: fit the abort outcomes against wind, venue turbulence, trajectory and arm, with standard
#   errors clustered on the FLIGHT SEED, and emit one tidy coefficient table.
#   WHY CLUSTERED ON SEED: the three arms fly the SAME 30 flight seeds (block 10775-10804) at every rung, so a
#   leg is not an independent draw -- a seed that is hard for one arm tends to be hard for all three. Ordinary
#   errors would understate the uncertainty. Clustering on the seed label is the correction, and "seed" here
#   means the FLIGHT seed, never a random-draw stream.
#   TWO FAMILIES, BOTH REPORTED: Binomial on "did this leg have any such event" answers whether the event
#   OCCURS more often; Poisson on the count answers whether it occurs MORE TIMES per leg. They can disagree --
#   a rare arm with many events per affected leg is a real pattern -- so neither is dropped.
#   S0 IS EXCLUDED FROM THE FIT AND REPORTED SEPARATELY: S0 is the zero-disturbance control stage (zero wind
#   AND zero turbulence), so it carries no venue sigma and would enter a turbulence factor as a third level
#   that is not a venue. Its rates are printed beside the fit, never inside it.
#   T2: wind_ms is the commanded WIND rung, a property of the air mass. It is never added to an airspeed; both
#   aircraft fly 24.7 m/s TAS at every rung and the margin against any airspeed limit is the same at all of them.
#   CONSTANT-SIGMA (measured, not T2): turbulence sigma does NOT grow with the wind rung -- it is constant
#   across rungs by construction and by measurement. wind_ms and sigma are therefore separate regressors, and
#   no coefficient here may be read as "more wind means bigger gusts".
# PUBLIC API: OUTCOMES; fit_one(df, outcome, family); main(in_dir)
# SIDE-EFFECTS: writes ABORT_GLM.csv under in_dir. Reads ABORT_PER_LEG.csv read-only.
# SIZE: soft 70 / hard 90 code-LoC; function cap 25.
# COVERAGE: limb A5 in abort_limbs.py -- every fit converges, every cluster count equals 30 seeds, and the
#   Binomial and Poisson tables carry the same regressor set.
import os
import sys

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402

TOL = A.TOL_PRIMARY
#: outcome name -> (per-leg count column, human sentence). The count column drives BOTH families: the Binomial
#: reads it as "> 0", the Poisson reads it as the count itself.
OUTCOMES = {
    "vertical_false_abort": ("vertical_false_tol%s" % TOL, "altitude-ceiling guard fired with no truth excursion"),
    "vertical_missed_abort": ("vertical_missed_tol%s" % TOL, "truth excursion inside 10 m with no guard fire"),
    "longitudinal_false_abort": ("longitudinal_false_tol%s" % TOL, "hard-overtake guard fired with no truth pass"),
    "longitudinal_missed_abort": ("longitudinal_missed_tol%s" % TOL, "truth pass held past the guard's persist gate with no fire"),
    "ceiling_fire": ("vertical_fires", "altitude-ceiling guard fired at all"),
    "overtake_fire": ("longitudinal_fires", "hard-overtake guard fired at all"),
    "breakaway": ("n_breakaways", "the aircraft entered an emergency breakaway"),
}
FORMULA = "%s ~ wind_ms + C(sigma) + C(trajectory) + C(policy)"


def fit_one(df, col, family):
    """One GLM, seed-clustered. Returns a tidy frame; a fit that does not converge is REPORTED, never silently
    replaced by a fit that did."""
    d = df.copy()
    d["y"] = (d[col] > 0).astype(int) if family == "binomial" else d[col].astype(int)
    fam = sm.families.Binomial() if family == "binomial" else sm.families.Poisson()
    try:
        m = smf.glm(FORMULA % "y", data=d, family=fam).fit(
            cov_type="cluster", cov_kwds={"groups": d["seed"]})
    except Exception as e:                       # a refusal to converge is a finding, not a blank cell
        return pd.DataFrame([{"term": "FIT_FAILED", "note": str(e)[:200]}])
    out = pd.DataFrame({"term": m.params.index, "coef": m.params.values, "se": m.bse.values,
                        "z": m.tvalues.values, "p": m.pvalues.values,
                        "ci_lo": m.conf_int()[0].values, "ci_hi": m.conf_int()[1].values})
    out["n_obs"], out["n_clusters"] = int(m.nobs), int(d["seed"].nunique())
    out["events_total"], out["legs_with_event"] = int(d[col].sum()), int((d[col] > 0).sum())
    return out


def main(in_dir):
    df = pd.read_csv(os.path.join(in_dir, "ABORT_PER_LEG.csv"))
    pool = df[df.wind_label != "S0"].copy()          # the nine wind rungs; S0 is the control stage
    rows = []
    for name, (col, sentence) in OUTCOMES.items():
        for family in ("binomial", "poisson"):
            t = fit_one(pool, col, family)
            t.insert(0, "family", family), t.insert(0, "outcome", name)
            t["reads_as"], t["tolerance_s"] = sentence, TOL
            rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(in_dir, "ABORT_GLM.csv"), index=False)
    print("fits=%d rows=%d legs_in_fit=%d seeds=%d failed=%d"
          % (len(OUTCOMES) * 2, len(out), len(pool), pool.seed.nunique(), int((out.term == "FIT_FAILED").sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
