#!/usr/bin/env python3
# =============================================================================

#   truth by the signed plan. Truth is always compared against Perception.
# HOW: every table is built by its OWN tool function, imported unedited (amplifying_tables, compare_arms,
#   build_ablation_table, hypothesis_verdicts), twice:
#     truth on the board as it is -> must reproduce the original file cell for cell, or REFUSE;
#     perception on a copy of the board whose win / win3 / whole columns hold the Perception peaks of
#                 PERCEPTION_PEAKS_PER_LEG.csv (step A: -ctrl_below_m on the same truth-range-selected ticks).
#   The companion file keeps the table's key columns and, for every cell that differs between the two builds, gives
#   "<column>, truth", "<column>, perception" and "<column>, perception minus truth" (first number of each cell).
#   Rows whose cells are all identical (success rates, guard counts) are left out of the companion.
#   The originals are never written.
# CLI: (none) SIDE-EFFECTS: writes <TABLE>_TRUTH_AND_PERCEPTION.csv/.md under OUT only.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, (_REPO + "/scoring/tools"))
from offset_delivery import OUT, RUN, md  # noqa: E402
import amplifying_tables as AT  # noqa: E402
import build_ablation_table as AB  # noqa: E402
import hypothesis_verdicts as V  # noqa: E402
import compare_arms as CA  # noqa: E402

SWAP = {"win": "perception_corridor", "win3": "perception_within_3_m", "whole": "perception_whole_leg"}
NUM = re.compile(r"[-+]?\d+\.?\d*")


def boards():
    raw = pd.read_csv(os.path.join(RUN, "board", "BOARD_W10.csv"), low_memory=False)
    P = pd.read_csv(os.path.join(OUT, "PERCEPTION_PEAKS_PER_LEG.csv"))
    m = raw[["corpus", "leg"]].merge(P, on=["corpus", "leg"], how="left")
    if m[list(SWAP.values())].shape[0] != len(raw) or m.perception_whole_leg.isna().any():
        raise SystemExit("REFUSE: step A peaks do not cover every board leg")
    per = raw.copy()
    for bc, pc in SWAP.items():
        per[bc] = m[pc].to_numpy()
    return raw, per


def boolify(b):
    b = b.copy()
    for c in ("passed", "passed_exam", "sw", "wind_rung", "real_overtake"):
        b[c] = b[c].astype(str).str.lower().eq("true")
    return b


def tok_for(board, tok):
    """The H1 placeholders the master table reprints, recomputed on `board` with the verdict tool's own h1()."""
    t = dict(tok)
    for arm in CA.ARMS:
        for sig, tag in ((0.25, "s025"), (0.35, "s035")):
            r = V.h1(board, arm, sig)
            t["H1.p95.%s.%s" % (arm, tag)], t["H1.p95ci.%s.%s" % (arm, tag)] = "%.3f m" % r["p95"], r["ci"]
    return t


def same_as_file(df, path):
    a = pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=False)
    b = pd.read_csv(path, dtype=str, keep_default_na=False)
    return a.shape == b.shape and (a.values == b.values).all()


def first(x):
    m = NUM.search(str(x))
    return float(m.group()) if m else np.nan


def side_by_side(t, p, keys):
    t, p = t.astype(str).reset_index(drop=True), p.astype(str).reset_index(drop=True)
    diff_cols = [c for c in t.columns if c not in keys and (t[c] != p[c]).any()]
    rows = (t[diff_cols] != p[diff_cols]).any(axis=1)
    out = t.loc[rows, keys].copy()
    for c in diff_cols:
        out[c + ", truth"], out[c + ", perception"] = t.loc[rows, c], p.loc[rows, c]
        d = p.loc[rows, c].map(first) - t.loc[rows, c].map(first)
        out[c + ", perception minus truth"] = d.map(lambda v: "" if np.isnan(v) else ("%+d" % v if float(v).is_integer() and abs(v) >= 1 else "%+.3f" % v))
    return out


def main():
    raw, per = boards()
    rawb, perb = boolify(raw), boolify(per)
    tok = json.load(open(os.path.join(RUN, "verdicts", "PLACEHOLDERS.json")))
    if tok_for(rawb, tok) != tok:
        raise SystemExit("REFUSE: the verdict tool's h1() on the board does not reproduce PLACEHOLDERS.json")
    tok_p = tok_for(perb, tok)
    comp, verd = os.path.join(RUN, "comparative"), os.path.join(RUN, "verdicts")
    TABLES = {"SAFETY_BY_STAGE": (lambda b, t: AT.safety_by_stage(b), comp, ["arm", "sigma", "stage"], False),
              "CORRIDOR_DIST": (lambda b, t: AT.corridor(b), comp, ["arm", "sig", "stage"], False),
              "CHANNEL_SENSITIVITY": (lambda b, t: AT.channel_sensitivity(b), comp, ["arm", "sigma"], False),
              "S0_TABLE": (lambda b, t: CA.s0_table(b), comp, ["arm", "stage"], False),
              "ARM_PAIRS": (lambda b, t: CA.pair_rows(b, t), comp, ["sigma", "arm_a", "arm_b", "pool"], False),
              "MASTER_TABLE": (lambda b, t: CA.master(b, t), comp, ["arm", "quantity"], False),
              "ABLATION_TABLE": (lambda b, t: AB.build(b)[0], verd, ["corpus", "configuration", "sigma"], True)}
    for name, (fn, d, keys, boolean) in TABLES.items():
        bt, bp = (rawb, perb) if boolean else (raw, per)
        t, p = fn(bt, tok), fn(bp, tok_p)
        if not same_as_file(t, os.path.join(d, name + ".csv")):
            raise SystemExit("REFUSE: the truth build of %s does not reproduce the original file" % name)
        s = side_by_side(t, p, keys).rename(columns=lambda c: c.replace("(entire episode), truth, ", "(entire episode), "))
        s.to_csv(os.path.join(OUT, name + "_TRUTH_AND_PERCEPTION.csv"), index=False)
        open(os.path.join(OUT, name + "_TRUTH_AND_PERCEPTION.md"), "w").write(
            "# %s: truth and Perception side by side - POST-PLAN; decides nothing\n\n" % name.replace("_", " ").lower().capitalize()
            + md(s) + "\n\n- Truth = raw_*, the original table reproduced cell for cell. Perception = ctrl_* on the same ticks. "
            "Only rows and columns where the two differ are shown.\n")
        print("%-20s truth build == original file; %d rows x %d columns differ" % (name, len(s), (len(s.columns) - len(keys)) // 3))
    h1_companion(rawb, perb)
    return 0


def h1_companion(rawb, perb):
    rows = []
    for arm in CA.ARMS:
        t, p = V.h1(rawb, arm, 0.25), V.h1(perb, arm, 0.25)
        rows.append({"arm": CA.NAME[arm], "graded": "yes, on truth" if arm in V.POL else "no (reported)",
                     "corridor p95, truth (m)": "%.3f %s" % (t["p95"], t["ci"]), "corridor p95, perception (m) - not graded": "%.3f %s" % (p["p95"], p["ci"]),
                     "perception minus truth (m)": "%+.3f" % (p["p95"] - t["p95"]),
                     "stages over 2.00 m, truth": t["stages_over"], "stages over 2.00 m, perception": p["stages_over"],
                     "legs over 2.00 m, truth": t["legs_over"], "legs over 2.00 m, perception": p["legs_over"]})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "HYPOTHESIS_VERDICTS_H1_TRUTH_AND_PERCEPTION.csv"), index=False)
    open(os.path.join(OUT, "HYPOTHESIS_VERDICTS_H1_TRUTH_AND_PERCEPTION.md"), "w").write(
        "# H1 statistic: truth (graded) and Perception (not graded) side by side - POST-PLAN\n\n" + md(s)
        + "\n\n- Sigma 0.25, nine wind stages, corridor <= 10.0 m on the truth range. H1 is graded on truth only (signed plan). "
        "Intervals: two-sided 95 % seed-cluster bootstrap (p95_boot.py).\n")
    print("HYPOTHESIS_VERDICTS H1 companion written")


if __name__ == "__main__":
    sys.exit(main())
