#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: T-93 fault screen, RULE v2 (SCREENPOINT-720). Labels an eval
#   leg CLEAN / BORDERLINE / FAULT / UNSAMPLED from the aligner's along-track
#   displacement, exactly as v1 does — except the statistic is the **MEAN**,
#   not the median.
#
#   WHY THE STATISTIC CHANGED. Fix A (FD_RAW_ALIGN_CENTER, elected
#   2026-08-18T21:04:46Z) centres each leg's age lattice, which drives the
#   MEDIAN tick to 0.0 BY CONSTRUCTION. Measured: ticks_median == 0.0 on all 14
#   cohort legs in both arms and on all 43 replay legs with the flag ON — v1
#   reads CLEAN everywhere and can no longer quarantine anything. Centering
#   does NOT zero the mean, so residual systematic bias survives there
#   (measured spread -0.277 .. +0.477) and is what this rule screens on.
#
#   NOT a fraction-of-displaced-rows rule. That was the shape FIXA-696
#   proposed; measured, it INVERTS post-Fix (it tracks dispersion about each
#   leg's own centre, so the best-corrected legs carry the HIGHEST fractions)
#   and would quarantine precisely the legs Fix A repaired.
#
#   v1 (analysis/c006_campaign_.../faultclass_census.py) is NOT modified and is
#   NOT superseded for historical legs: it stays frozen, keeps its SHA pin in
#   tools/pvt_paired_census.py, and pre-epoch labels keep meaning what they
#   meant. Every result here carries `rule_version` so v1 and v2 labels are
#   never pooled across the definition change.
#
# PUBLIC API: RULE_VERSION, FAULT_MIN_MEAN_TICKS, CLEAN_MAX_MEAN_TICKS,
#   label_for_bias, classify_leg, resolve_paired_csv
# SIDE-EFFECTS: reads leg CSVs from disk. Writes nothing, spawns nothing,
#   signals nothing, never mutates the environment. OUTCOME-BLIND by contract:
#   reads only rm_* columns, never a leg's pass/fail.
# CAPS: file soft 110 / hard 140 LoC (code lines); function cap 45.
# COVERAGE: tests/unit/test_faultclass_bias_v2.py
# =============================================================================
import glob
import importlib.util
import math
import os
import csv
import statistics as st

RULE_VERSION = "bias_v2"

_V1_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis/c006_campaign_20260728T110000Z/p13b_fault_rca/"
    "faultclass_census.py")


def _v1():
    """Load the frozen v1 classifier for its CONSTANTS only. Importing rather
    than restating them means TICK_DT, R_EARTH and MIN_ROWS cannot drift
    between the two rules — only the statistic differs, which is the whole
    point of the change."""
    spec = importlib.util.spec_from_file_location("faultclass_census", _V1_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_C = _v1()
R_EARTH = _C.R_EARTH
TICK_DT = _C.TICK_DT
MIN_ROWS = _C.MIN_ROWS

# SCREENPOINT-720 pre-registration (analysis/screenpoint_20260818T2200Z).
# Anchored OUTSIDE this analysis: 1 tick = CRUISE_SPEED_MS * TICK_DT =
# 24.7 * 0.05 = 1.235 m, against the M3b behind tolerance of 0.7 m.
FAULT_MIN_MEAN_TICKS = 0.45   # ~0.556 m == 79% of the 0.7 m tolerance
CLEAN_MAX_MEAN_TICKS = 0.15   # ~0.185 m == 26% of the 0.7 m tolerance


def label_for_bias(mean_ticks):
    """Label from the DIMENSIONLESS mean tick bias. Pure, so the rule can be
    exercised in both directions with no leg on disk — a property v1 has and
    this keeps. ABSOLUTE: a leg that over-corrects BACKWARD is as contaminated
    as one that over-advances, and a signed rule would miss it."""
    if abs(mean_ticks) <= CLEAN_MAX_MEAN_TICKS:
        return "CLEAN"
    if abs(mean_ticks) >= FAULT_MIN_MEAN_TICKS:
        return "FAULT"
    return "BORDERLINE"


def resolve_paired_csv(leg_dir):
    """Ordered search, hardcoded path FIRST — the T-131 contract, mirrored from
    v1 so a leg readable by one rule is readable by the other. os.path.isfile
    (not lexists) is load-bearing: a dangling link stays UNSAMPLED."""
    for cand in (os.path.join(leg_dir, "paired", "env0",
                              "paired_channels_ep0001.csv"),
                 *sorted(glob.glob(os.path.join(
                     leg_dir, "env*", "paired_channels_ep*.csv"))),
                 *sorted(glob.glob(os.path.join(
                     leg_dir, "*", "env*", "paired_channels_ep*.csv")))):
        if os.path.isfile(cand):
            return cand
    return None


def _rows(path):
    """Per-row along-track displacement in meters and in dimensionless ticks.
    Normalised by the leg's OWN measured groundspeed, so no wind table, stage
    label or TAS assumption enters."""
    disp, ticks = [], []
    for r in csv.DictReader(open(path)):
        try:
            la, lo = float(r["rm_lead_lat"]), float(r["rm_lead_lon"])
            laa, loa = float(r["rm_lead_lat_al"]), float(r["rm_lead_lon_al"])
            vn, ve = float(r["rm_lead_vn"]), float(r["rm_lead_ve"])
        except (ValueError, KeyError):
            continue
        sp = math.hypot(vn, ve)
        if sp < 1.0:
            continue
        dn = math.radians(laa - la) * R_EARTH
        de = math.radians(loa - lo) * R_EARTH * math.cos(math.radians(la))
        along = (dn * vn + de * ve) / sp
        disp.append(along)
        ticks.append(along / (sp * TICK_DT))
    return disp, ticks


def classify_leg(leg_dir):
    """Same call signature and same fail-closed contract as v1: anything
    unreadable is UNSAMPLED, never CLEAN. `ticks_est` (the MEDIAN) is retained
    unchanged so v1/v2 readers can be proven equivalent on the same leg; only
    `mean_ticks` decides the label."""
    name = os.path.basename(leg_dir.rstrip("/"))
    base = {"leg": name, "rule_version": RULE_VERSION}
    path = resolve_paired_csv(leg_dir)
    if path is None:
        return dict(base, label="UNSAMPLED", why="no_paired_csv")
    try:
        disp, ticks = _rows(path)
    except OSError:
        return dict(base, label="UNSAMPLED", why="csv_unreadable")
    if len(disp) < MIN_ROWS:
        return dict(base, label="UNSAMPLED",
                    why="rows=%d<%d" % (len(disp), MIN_ROWS))
    mean_t = sum(ticks) / len(ticks)
    return dict(base, label=label_for_bias(mean_t),
                mean_ticks=round(mean_t, 4),
                disp_mean_m=round(sum(disp) / len(disp), 3),
                disp_median_m=round(st.median(disp), 3),
                ticks_est=round(st.median(ticks), 3),
                n_rows=len(disp))
