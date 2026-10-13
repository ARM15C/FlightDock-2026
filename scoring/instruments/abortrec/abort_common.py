#!/usr/bin/env python3
# THE ABORT AND RECOVERY INSTRUMENT -- shared vocabulary.
# RESPONSIBILITY: name the six corpora, bind each to its banked per-leg summary AND its banked readout, and
#   carry every threshold this dispatch compares against -- each READ FROM the producer at import, never retyped.
#   WHAT IS NEW HERE AND WHAT IS NOT: AMPSAFETY-20260903 already delivered the VERTICAL false/missed abort, the
#   guard-fire rates, the breakaway battery and the post-exit recovery timings. This dispatch does not recompute
#   them; it reads that instrument's banked per-leg summaries as its event source. What is genuinely new is the
#   LONGITUDINAL false/missed pair, the per-EVENT table with its 10 m flag, the per-rung split, the GLM, the
#   within-arm seed-paired false-abort -> M3b read, and the +-30 s recovery traces.
#   TOLERANCE WINDOWS: the banked set is 0.25 / 0.5 / 1.0 s; 2.0 s is ADDED as a
#   fourth, never as a replacement. PRIMARY for the document is 1.0 s -- the delivered number and the widest of the
#   banked set. 2.0 s is the sensitivity bound. Every table prints all four.
#   CHANNEL (R5.1): events are classified against the TRUTH channel (raw_*), which is what the H1 verdict's
#   scorer reads. The ceiling guard compares the ESTIMATOR channel (env; FD_GUARD_CEILING_CHANNEL=env witnessed
#   in both arms' manifests, 2026-09-03), so a FALSE vertical abort is BY CONSTRUCTION a perception-vs-truth
#   event: the estimator saw a hazard the truth channel did not.
#   T1 THROUGHOUT: 2.00 m is the safety-EXCLUSION-ZONE bar. A guard firing is a zone breach or a guard fire --
#   never a "collision", "crash" or "contact". The airframe-contact plane is 3.42 m above the boom tip.
# PUBLIC API: TAGS, SUMMARY, READOUT, LEGS_ROOT, POLICY, SIGMA, TOLS, TOL_PRIMARY, WINDOW_M, BAR, CONTACT,
#   OT_PERSIST_S, DWELL_MIN_S, EXIT_BEHIND_M, SEP_DELTA_MPS, HOLD_ALT_MSL, TRACE_SPAN_S, formation_states(),
#   readout_legs(tag), summary_rows(tag), wind_map(sigma), SEED_BLOCK.
# SIDE-EFFECTS: none beyond reading banked files read-only. SIZE: soft 60 / hard 80 code-LoC; function cap 20.
# COVERAGE: limb A0 in abort_limbs.py -- every tag resolves 600 legs in BOTH its summary and its readout, and
#   every constant below equals the producer's own value at run time.
import gzip
import json
import os
import re
import sys

ROOT = "/home/ubuntu/FlightDock"
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "analysis/ch4pull_20260904T1605Z"))
import formation_config as FC       # noqa: E402 the producer of every threshold below
import ch4_common as C4             # noqa: E402 the corpus->legs_root/readout binding already proven

#: The six corpora of record. The summary stem is the ampsafety arm name, matched to the tag by a per-leg
#: guard-count fingerprint (1.000 agreement on all 600 legs of all six, measured 2026-09-04) -- never by name.
TAGS = ("A2P025", "A2P035", "CSV025", "CSV035", "ARMB025", "ARMB035")
_STEM = {"A2P": "A2P", "CSV": "CSVO", "ARMB": "ARMB"}
AMP = os.path.join(ROOT, "analysis/ampsafety_20260903T1300Z/summaries")
SUMMARY = {t: os.path.join(AMP, "%s_%s.jsonl.gz" % (_STEM[t[:-3]], "0.25" if t.endswith("025") else "0.35"))
           for t in TAGS}
READOUT = {t: C4.READOUT[t] for t in TAGS}
LEGS_ROOT = {t: C4.LEGS_ROOT[t] for t in TAGS}
POLICY = {t: C4.POLICY[t] for t in TAGS}
SIGMA = {t: C4.SIGMA[t] for t in TAGS}
SEED_BLOCK = (10775, 10804)

#: Tolerance windows. 1.0 s is PRIMARY (the banked, delivered number); 2.0 s is the added sensitivity bound.
TOLS = (0.25, 0.5, 1.0, 2.0)
TOL_PRIMARY = 1.0
WINDOW_M = 10.0                 # the window of record
TRACE_SPAN_S = 30.0             # +-30 s about each breakaway entry

#: Every threshold READ from the producer. A retyped constant is how a derivation inherits a parent literal.
BAR = -float(FC.GUARD_ALT_CEILING_BELOW_M)          # 2.00 m above the boom tip, as height
CONTACT = 3.42                                      # airframe-contact plane, docs/GEOMETRY_OF_RECORD.md
OT_PERSIST_S = float(FC.GUARD_OVERTAKE_PERSIST_S)   # the no-pass line must be held this long to fire
DWELL_MIN_S = float(FC.ABORT_MIN_DWELL_S)
EXIT_BEHIND_M = float(FC.ABORT_EXIT_BEHIND_MIN_M)
SEP_DELTA_MPS = float(FC.NOPASS_HARD_SLOWDOWN_DELTA_MPS)
HOLD_ALT_MSL = float(FC.PRECONTACT_ALT_MSL)


def formation_states():
    """The states in which the two guards are EVALUATED AT ALL, read out of the controller's own source rather
    than re-typed -- a derivation that re-types a parent literal inherits its drift. Outside this set both
    guards are structurally disarmed (baseline_formation_controller: `if self.state in FORMATION_STATES:`
    wraps Guard 1 and Guard 2), so a truth crossing there is NOT a missed abort: the guard was never asked.
    Refuses rather than returning a guess if the block cannot be found."""
    src = open(os.path.join(ROOT, "tests/baseline_formation_controller.py")).read()
    i = src.index("FORMATION_STATES = {")
    names = re.findall(r"ControllerState\.([A-Z_]+)", src[i:src.index("}", i)])
    if not names:
        raise SystemExit("REFUSE rc=9: FORMATION_STATES could not be read from the controller source")
    return set(names)


def readout_legs(tag):
    """The banked readout's per-leg rows, keyed by leg name. Carries guards.by_type (the reproduction target),
    M3b success (`passed`), wind_ms, trajectory and seed -- every covariate the GLM reads."""
    return {L["leg"]: L for L in json.load(open(READOUT[tag]))["legs"]}


def summary_rows(tag):
    """The banked ampsafety per-leg rows for one corpus, streamed. The EVENT SOURCE of this dispatch."""
    with gzip.open(SUMMARY[tag], "rt") as fh:
        for line in fh:
            yield json.loads(line)


def wind_map(sigma):
    """label -> wind_ms, read from the SIGNED seed-cell table for that venue."""
    return C4.wind_map("025" if float(sigma) == 0.25 else "035")
