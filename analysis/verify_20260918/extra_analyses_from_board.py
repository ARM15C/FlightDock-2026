#!/usr/bin/env python3
# =============================================================================

#   nothing. Success = the flag of record (board `passed`). Methods are IMPORTED, never re-typed:
#     single rate -> two-sided Wilson 95 % (compare_arms.wilson)
#     paired change -> two-sided 95 % Newcombe (1998) (compare_arms.two_sided on hypothesis_verdicts.paired)
#     breakaway cost -> seed-stratified CMH from the pinned tick-level instrument (ticklevel/abort_impact.read_one),
#                         re-run here on the flag of record because ABORT_M3B_IMPACT.csv carries the EXAM flag
#                         (m3b_success == passed_exam on 7,200 of 7,200 legs, measured 2026-09-18).
#   Row 14 calm control stage flown twice (in the sigma 0.25 corpus and the sigma 0.35 corpus; identical leg keys).
#   Row 13 success by trajectory, nine wind stages, per arm and sigma.
#   Row 20 docking with and without an emergency breakaway, nine wind stages.
#   A star (*) marks a two-sided interval that excludes zero, or a CMH p below 0.05.
# CLI: (none) SIDE-EFFECTS: writes the six files named in OUTS under OUT only.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, (_REPO + "/scoring/tools"))
from offset_delivery import OUT, RUN, md  # noqa: E402
import hypothesis_verdicts as V  # noqa: E402
import compare_arms as CA  # noqa: E402

ARMS = [("FlightDock-A2", "A2P025", "A2P035"), ("FlightDock-A2O", "A2V25", "A2V35"), ("FlightDock-S1", "S1A25", "S1A35"),
        ("FlightDock-S1O", "CSV025", "CSV035"), ("PID-baseline (comparator)", "ARMB025", "ARMB035"), ("PIDO", "PIDO25", "PIDO35")]
spec = importlib.util.spec_from_file_location("abort_impact", os.path.join(RUN, "ticklevel", "abort_impact.py"))
AI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AI)


def board():
    b = pd.read_csv(os.path.join(RUN, "board", "BOARD_W10.csv"), low_memory=False)
    b["passed"] = b.passed.astype(str).eq("True")
    b["wind_rung"] = b.wind_rung.astype(str).eq("True")
    return b


def star_iv(lo, hi):
    return "[%+.1f, %+.1f]%s" % (lo, hi, " *" if lo > 0 or hi < 0 else "")


def repeatability(b):
    s0 = lambda d: d.wind_label == "S0"
    rows = []
    for name, c25, c35 in ARMS:
        r = V.paired(b, c25, c35, s0)
        if r["n"] != 60:
            raise SystemExit("REFUSE: %s calm stage pairs %d legs, expected 60" % (name, r["n"]))
        lo, hi = CA.two_sided(r)
        rows.append({"arm": name, "paired legs": r["n"], "first flight (in the sigma 0.25 corpus)": "%d of 60, %s" % (r["ka"], CA.wilson(r["ka"], 60)),
                     "second flight (in the sigma 0.35 corpus)": "%d of 60, %s" % (r["kb"], CA.wilson(r["kb"], 60)),
                     "change, second minus first (pp)": "%+.1f" % -r["diff"], "change: two-sided 95 % Newcombe (pp)": star_iv(-hi, -lo),
                     "legs that flipped": r["b"] + r["c"], "success lost": r["b"], "success gained": r["c"], "legs identical in outcome": r["a"] + r["d"]})
    return pd.DataFrame(rows)


def trajectory(b):
    rows = []
    for name, c25, c35 in ARMS:
        for sig, c in (("0.25", c25), ("0.35", c35)):
            x = b[(b.corpus == c) & b.wind_rung]
            st, rt = x[x.trajectory == "straight"], x[x.trajectory == "racetrack"]
            rows.append({"arm": name, "sigma": sig, "straight": "%d of %d, %s" % (st.passed.sum(), len(st), CA.wilson(int(st.passed.sum()), len(st))),
                         "racetrack": "%d of %d, %s" % (rt.passed.sum(), len(rt), CA.wilson(int(rt.passed.sum()), len(rt))),
                         "straight minus racetrack (pp)": "%+.1f" % (100.0 * (st.passed.mean() - rt.passed.mean()))})
    return pd.DataFrame(rows)


def breakaway_cost(b):
    pl = pd.read_csv(os.path.join(RUN, "ticklevel", "ABORT_PER_LEG.csv"), low_memory=False)
    pl = pl.merge(b[["corpus", "leg", "passed", "wind_rung"]], left_on=["tag", "leg"], right_on=["corpus", "leg"], how="inner")
    if len(pl) != 7200:
        raise SystemExit("REFUSE: per-leg abort join has %d rows, expected 7200" % len(pl))
    pl["m3b_success"] = pl.passed
    rows = []
    for name, c25, c35 in ARMS:
        for sig, c in (("0.25", c25), ("0.35", c35)):
            d = pl[(pl.tag == c) & pl.wind_rung].reset_index(drop=True)
            r = AI.read_one(d, "n_breakaways")
            p = r.get("cmh_p", np.nan)
            if r.get("tested") and p == 0.0:   # the double-precision p underflowed; report its magnitude from log sf
                st = AI.StratifiedTable(AI._strata(d, d["n_breakaways"] > 0)).test_null_odds()
                p = "< 1e%d" % int(np.floor(chi2.logsf(st.statistic, 1) / np.log(10)) + 1)
            rows.append({"arm": name, "sigma": sig, "legs": r["legs"], "legs with a breakaway": r["legs_exposed"],
                         "docked, legs with a breakaway (%)": "" if r["dock_rate_exposed_pct"] is None else "%.1f" % r["dock_rate_exposed_pct"],
                         "docked, legs without (%)": "" if r["dock_rate_unexposed_pct"] is None else "%.1f" % r["dock_rate_unexposed_pct"],
                         "difference (pp)": "" if r["difference_pp"] is None else "%+.1f" % r["difference_pp"],
                         "seed-stratified CMH p": (p + " *" if isinstance(p, str) else "%.2g%s" % (p, " *" if p < 0.05 else "")) if r.get("tested") else "",
                         "CMH pooled odds ratio [95 %]": ("%.2f [%.2f, %.2f]" % (r["cmh_or"], r["cmh_or_ci_lo"], r["cmh_or_ci_hi"])) if r.get("tested") else "",
                         "note": r.get("note", "") if isinstance(r.get("note", ""), str) else ""})
    return pd.DataFrame(rows)


NOTES = {
    "CALM_STAGE_REPEATABILITY": ["Calm control stage (sigma 0.00, no wind), flown once inside each arm's sigma 0.25 corpus and once inside its sigma 0.35 corpus, on the same 30 seeds x 2 trajectories.",
                                 "Rates: two-sided Wilson 95 %. Change: two-sided 95 % paired Newcombe (1998). * = the interval excludes zero. Success = flag of record."],
    "SUCCESS_BY_TRAJECTORY": ["Nine wind stages, 270 legs per trajectory per corpus. Rates: two-sided Wilson 95 %. Success = flag of record."],
    "BREAKAWAY_COST": ["Nine wind stages, 540 legs per corpus. Exposure = at least one emergency breakaway in the leg.",
                       "CMH = Cochran-Mantel-Haenszel over the 30 seed strata (statsmodels StratifiedTable), from the pinned tick-level instrument, re-run on the flag of record.",
                       "* = CMH p < 0.05. A side with fewer than 10 legs is counted, not tested."]}


def main():
    b = board()
    for stem, fn in (("CALM_STAGE_REPEATABILITY", repeatability), ("SUCCESS_BY_TRAJECTORY", trajectory), ("BREAKAWAY_COST", breakaway_cost)):
        t = fn(b)
        t.to_csv(os.path.join(OUT, stem + ".csv"), index=False)
        open(os.path.join(OUT, stem + ".md"), "w").write("# %s - POST-PLAN; decides nothing\n\n" % stem.replace("_", " ").lower().capitalize()
                                                         + md(t) + "\n\n" + "\n".join("- " + n for n in NOTES[stem]) + "\n")
        pd.set_option("display.width", 320)
        pd.set_option("display.max_colwidth", 60)
        print("==", stem), print(t.to_string(index=False)), print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
