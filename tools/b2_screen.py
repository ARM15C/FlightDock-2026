#!/usr/bin/env python3
# =============================================================================
# #513 B2 boot screen (PREREG_513 + ADDENDUM_1). Classifies a leg's leader-sample
#   alignment from rows STRICTLY BEFORE the first DOCKING_CREEP row, so the
#   screen cannot see any corridor outcome (PREREG_513 §3 -- outcome-blindness
#   is structural, not promised).
#
# TWO MARKERS, UNION RULE (PREREG_513 §2, declared before any performance was
#   computed): discard iff M-A OR M-B fires.
#     M-A median(rm_age2_s * 1000) >= 25 ms over mode-0 rows in the window
#     M-B median(ctrl_behind_m - raw_behind_m) >= 1.0 m over the same rows
#   Union is chosen because a false positive costs a DISCARDED BOOT, never an
#   injected 1.235 m -- the asymmetry that ranks B2 above the vetoed B1.
#
# NOT A FIX, NOT ENABLED ANYWHERE. This module computes and reports. Enabling
#   screening in any certification run is a separate and any verdict
#   computed under an enabled screen must state the screened population.
#
# PUBLIC API: screen_leg(leg_dir) -> LegScreen; census(leg_dirs) -> DataFrame
#   CLI: b2_screen.py --corpus DIR [DIR ...] --out PATH
# CAPS: file soft 150 / hard 175 LoC; per-function <= 45.
# COVERAGE: tests/unit/test_b2_screen.py -- synthetic fixtures incl. the KA-4
#   non-identifying negative control.
# =============================================================================
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

AGE2_THRESH_MS = 25.0      # sits in the empty 6.2x gap between the two modes
BIAS_THRESH_M = 1.0        # #500: |bias| >= 1.0 separated 4/4 fails vs 0/26
MIN_MODE0_FRACTION = 0.20  # ADDENDUM_1 A-2: per-leg sufficiency, not corpus HALT
MAX_EXCLUDED = 4           # of 42; declared in ADDENDUM_1 before the count was known
CREEP = "DOCKING_CREEP"


@dataclass
class LegScreen:
    leg: str
    usable: bool = False
    reason: str = ""
    n_window: int = 0
    n_mode0: int = 0
    mode0_fraction: float = 0.0
    age2_med_ms: Optional[float] = None
    bias_med_m: Optional[float] = None
    m_a: bool = False
    m_b: bool = False
    fires: bool = False
    cls_window: Optional[str] = None
    cls_full: Optional[str] = None
    extra: dict = field(default_factory=dict)


def _paired_csv(leg_dir):
    hits = sorted(glob.glob(os.path.join(
        leg_dir, "paired", "env*", "paired_channels_ep*.csv")))
    return hits[0] if hits else None


def _classify(age2_med_ms):
    if age2_med_ms is None or not np.isfinite(age2_med_ms):
        return None
    return "EXPRESSING" if age2_med_ms >= AGE2_THRESH_MS else "CLEAN"


def _median_or_none(series):
    v = series.dropna()
    return float(np.median(v.values)) if len(v) else None


def screen_leg(leg_dir):
    """Compute the outcome-blind screen for one leg. Never raises on a bad leg:
    an unusable leg is returned as usable=False with a reason, so the census can
    report exclusions by name rather than dying on the first bad input."""
    name = os.path.basename(os.path.normpath(leg_dir))
    r = LegScreen(leg=name)
    path = _paired_csv(leg_dir)
    if path is None:
        r.reason = "no paired CSV"
        return r
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        r.reason = "unreadable: %s: %s" % (type(e).__name__, e)
        return r
    for col in ("fsm_state", "raw_align_mode", "rm_age2_s",
                "ctrl_behind_m", "raw_behind_m"):
        if col not in df.columns:
            r.reason = "missing column %s" % col
            return r

    creep = np.flatnonzero((df["fsm_state"].astype(str) == CREEP).values)
    if creep.size == 0:
        r.reason = "no %s row -- window undefined" % CREEP
        return r
    win = df.iloc[:int(creep[0])]
    r.n_window = len(win)

    m0 = win[win["raw_align_mode"].values == 0]
    r.n_mode0 = len(m0)
    r.mode0_fraction = (r.n_mode0 / r.n_window) if r.n_window else 0.0
    if r.mode0_fraction < MIN_MODE0_FRACTION:
        r.reason = ("DATA-INSUFFICIENT: mode-0 fraction %.3f < %.2f"
                    % (r.mode0_fraction, MIN_MODE0_FRACTION))
        return r

    r.age2_med_ms = _median_or_none(m0["rm_age2_s"] * 1000.0)
    r.bias_med_m = _median_or_none(m0["ctrl_behind_m"] - m0["raw_behind_m"])
    r.m_a = bool(r.age2_med_ms is not None and r.age2_med_ms >= AGE2_THRESH_MS)
    # T-32: ABSOLUTE value, restoring the CITED #500
    # rule ("|bias| >= 1.0 separated 4/4 fails vs 0/26" — the comment at
    # BIAS_THRESH_M always said abs; the implementation had dropped it). The
    # measured phenomenon is one-signed NEGATIVE (#504: expressing-boot
    # bias_FRESH in [-1.412, -1.116] vs clean [-0.201, +0.007]), so the signed
    # `>= +1.0` test could NEVER fire on the defect it was built to catch —
    # 0/20 firings on F-2 [MEASURED, T25_PRECONDITION.md]. With abs(), the
    # union is no longer a decorative-conjunct single-marker rule (#513
    # lesson): on F-2 the corrected m_b additionally catches env1/ep5
    # (bias -1.209) and env2/ep5 (bias -1.013) — BOTH dwell-0.00 episodes the
    # age2 marker missed, consistent with #506's "marker under-counts 2 of 10".
    r.m_b = bool(r.bias_med_m is not None and abs(r.bias_med_m) >= BIAS_THRESH_M)
    r.fires = r.m_a or r.m_b                       # UNION, per PREREG_513 §2
    r.cls_window = _classify(r.age2_med_ms)

    full0 = df[df["raw_align_mode"].values == 0]
    r.cls_full = _classify(_median_or_none(full0["rm_age2_s"] * 1000.0))
    r.usable = True
    return r


def census(leg_dirs):
    rows = [screen_leg(d) for d in leg_dirs]
    return pd.DataFrame([vars(x) for x in rows]).drop(columns=["extra"])


def _summary(df):
    use = df[df["usable"]]
    exc = df[~df["usable"]]
    expr = use[use["cls_full"] == "EXPRESSING"]
    clean = use[use["cls_full"] == "CLEAN"]
    agree = int((use["cls_window"] == use["cls_full"]).sum())
    return {
        "legs_total": int(len(df)),
        "legs_usable": int(len(use)),
        "excluded": [{"leg": r.leg, "reason": r.reason} for r in exc.itertuples()],
        "excluded_within_bar": bool(len(exc) <= MAX_EXCLUDED),
        "n_expressing_full": int(len(expr)),
        "n_clean_full": int(len(clean)),
        "fires_on_expressing": int(expr["fires"].sum()),
        "fires_on_clean": int(clean["fires"].sum()),
        "catch_m_a_only": int(expr["m_a"].sum()),
        "catch_m_b_only": int(expr["m_b"].sum()),
        "KA3_window_agreement": "%d/%d" % (agree, len(use)),
        "KA3_disagreements": [r.leg for r in use.itertuples()
                              if r.cls_window != r.cls_full],
        "KA4_identifying": bool(len(expr) and len(clean)
                                and int(expr["fires"].sum()) > int(clean["fires"].sum())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirs = []
    for c in a.corpus:
        dirs += [d for d in sorted(glob.glob(os.path.join(c, "*")))
                 if os.path.isdir(d) and _paired_csv(d)]
    df = census(dirs)
    df.to_csv(a.out, index=False)
    s = _summary(df)
    json.dump(s, open(a.out.replace(".csv", "_summary.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
