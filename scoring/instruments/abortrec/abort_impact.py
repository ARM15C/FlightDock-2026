#!/usr/bin/env python3
# DID THE INTERVENTION COST THE DOCK?
# RESPONSIBILITY: within ONE arm at ONE venue, ask whether a leg that suffered a given intervention docked less
#   often than a leg that did not -- stratified by the FLIGHT seed, so a seed that is hard for every arm cannot
#   masquerade as an effect of the intervention.
#   WHY STRATIFIED AND NOT PAIRED: the campaign's existing pairing is arm-vs-arm on (wind label, seed,
#   trajectory), and within a single arm that key names exactly one leg, so there is no partner to pair with.
#   The within-arm question is therefore answered by CONDITIONING on the seed: a Cochran-Mantel-Haenszel read
#   over the 30 seed strata (statsmodels StratifiedTable), reported beside the unstratified Fisher exact test
#   so the reader can see whether conditioning moved it.
#   "SEED" HERE IS THE FLIGHT SEED LABEL (block 10775-10804), never a random-draw stream.
#   THIS IS AN ASSOCIATION, NOT A CAUSE, and the table says so on its face: the same turbulence that provokes a
#   guard fire also makes the dock harder, so a negative association is expected even if the abort itself cost
#   nothing. What the table CAN settle is the opposite direction -- an intervention followed by an undiminished
#   dock rate is evidence the abort was cheap.
#   A CELL WITH TOO FEW EXPOSED LEGS IS REPORTED AS SUCH AND NOT TESTED: a p-value from four legs is noise
#   wearing a decimal point.
# PUBLIC API: EXPOSURES; read_one(d, col); main(in_dir)
# SIDE-EFFECTS: writes ABORT_M3B_IMPACT.csv under in_dir. Reads ABORT_PER_LEG.csv read-only.
# SIZE: soft 80 / hard 90 code-LoC; function cap 25. (Soft was declared at 70 before the file existed;
#   recalibrated ONCE to 80 when the boundary-odds-ratio flag was added -- stated here rather than quietly
#   exceeded, and following the same convention as the sibling instrument safety_extract.py. The HARD cap is
#   unchanged and is never raised: that one is split.)
# COVERAGE: limb A7 in abort_limbs.py -- exposed + unexposed equals the cell's leg count for every row, no
#   row reports a test when the exposed count is under the declared minimum, and every 0/infinite odds ratio
#   carries or_is_boundary rather than passing as an estimate.
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abort_common as A       # noqa: E402

MIN_EXPOSED = 10               # below this the association is reported as a count only, never as a test
TOL = A.TOL_PRIMARY
EXPOSURES = {
    "a FALSE vertical abort": "vertical_false_tol%s" % TOL,
    "a MISSED vertical abort": "vertical_missed_tol%s" % TOL,
    "a FALSE longitudinal abort": "longitudinal_false_tol%s" % TOL,
    "a MISSED longitudinal abort": "longitudinal_missed_tol%s" % TOL,
    "an altitude-ceiling guard fire": "vertical_fires",
    "a hard-overtake guard fire": "longitudinal_fires",
    "an emergency breakaway": "n_breakaways",
}


def _strata(d, exposed):
    """One 2x2 per flight seed: rows exposed/not, columns docked/not. Strata with no variation contribute
    nothing to the CMH statistic and are kept rather than dropped, so the seed count stays honest."""
    out = []
    for _, g in d.groupby("seed"):
        e, s = exposed.loc[g.index], g.m3b_success.to_numpy(bool)
        out.append(np.array([[int((e & s).sum()), int((e & ~s).sum())],
                             [int((~e & s).sum()), int((~e & ~s).sum())]]))
    return out


def read_one(d, col):
    """The dock rate with and without the exposure, and whether the difference survives conditioning on seed."""
    exposed = d[col] > 0
    n_e, n_u = int(exposed.sum()), int((~exposed).sum())
    row = {"legs": len(d), "legs_exposed": n_e, "legs_unexposed": n_u,
           "dock_rate_exposed_pct": (100.0 * d.m3b_success[exposed].mean() if n_e else None),
           "dock_rate_unexposed_pct": (100.0 * d.m3b_success[~exposed].mean() if n_u else None)}
    row["difference_pp"] = (None if row["dock_rate_exposed_pct"] is None or row["dock_rate_unexposed_pct"] is None
                            else row["dock_rate_exposed_pct"] - row["dock_rate_unexposed_pct"])
    if n_e < MIN_EXPOSED or n_u < MIN_EXPOSED:
        row["tested"] = False
        row["note"] = "fewer than %d legs on one side -- counted, not tested" % MIN_EXPOSED
        return row
    tab = [[int((exposed & d.m3b_success).sum()), int((exposed & ~d.m3b_success).sum())],
           [int((~exposed & d.m3b_success).sum()), int((~exposed & ~d.m3b_success).sum())]]
    row["fisher_or"], row["fisher_p"] = fisher_exact(tab)
    st = StratifiedTable(_strata(d, exposed))
    row["cmh_or"] = float(st.oddsratio_pooled)
    row["cmh_p"] = float(st.test_null_odds().pvalue)
    row["cmh_or_ci_lo"], row["cmh_or_ci_hi"] = [float(x) for x in st.oddsratio_pooled_confint()]
    row["n_seed_strata"], row["tested"] = int(d.seed.nunique()), True
    return _boundary(row, int(d.m3b_success[exposed].sum()), n_e)


def _boundary(row, docked, n_e):
    """A pooled odds ratio of exactly 0 or infinity is a BOUNDARY, not an estimate: every exposed leg docked, or
    none did. Reporting it as a number would publish a certainty the data cannot carry, so it is flagged here
    and the exact test beside it is what the reader uses."""
    row["or_is_boundary"] = not np.isfinite(row["cmh_or"]) or row["cmh_or"] == 0.0
    if row["or_is_boundary"]:
        row["note"] = ("a boundary, not an estimate: %d of %d exposed legs docked -- read the exact test, and "
                       "the confidence interval is unbounded" % (docked, n_e))
    return row


def main(in_dir):
    df = pd.read_csv(os.path.join(in_dir, "ABORT_PER_LEG.csv"))
    rows = []
    for tag, g in df.groupby("tag"):
        for pool in ("all_ten_rungs", "wind_rungs"):
            d = g if pool == "all_ten_rungs" else g[g.wind_label != "S0"]
            for label, col in EXPOSURES.items():
                rows.append(dict({"tag": tag, "policy": A.POLICY[tag], "sigma": A.SIGMA[tag], "pool": pool,
                                  "exposure": label, "tolerance_s": TOL,
                                  "channel": "truth (raw), window <= %.1f m" % A.WINDOW_M,
                                  "seed_block": "%d-%d" % A.SEED_BLOCK}, **read_one(d, col)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(in_dir, "ABORT_M3B_IMPACT.csv"), index=False)
    print("impact rows=%d tested=%d not_tested=%d" % (len(out), int(out.tested.sum()), int((~out.tested).sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
