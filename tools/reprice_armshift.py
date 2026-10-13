#!/usr/bin/env python3
"""ARMSHIFT-817 buffer reprice: the delta from the SUPERSEDED vertical point to the ELECTED one.

RESPONSIBILITY: one stateless PAIR delta, held in its own module because adding it inline pushed
`reprice_terms.py` past its HARD cap of 160 code lines. No cap was raised; the file was split
(the `reprice_terms` / `reramp_leg_scope` precedent). `reprice_terms.py` imports the fn and
registers it in TERMS, so there is still exactly one term registry.

WHY THIS IS NOT `--term e5v_armshift`. That term moved the two ARMS only, so it may read the
superseded half's WEIGHTS off live config. ARMSHIFT-817 moves arms AND weights together
(0.80 -> 1.40 m, W_ENDPOINT_RAMP 0.20 -> 1.00, W_CEILRATE 0.5 -> 1.25), so its superseded half
must restate BOTH frozen -- reading live weights there would silently price the old arm at the
NEW weights and the delta would be wrong in a way nothing downstream could see.

The ELECTED half comes from the LIVE functions, so this term REQUIRES config to already carry the
elected point when it runs. `elected_is_live()` is the fail-closed assertion for that; the caller
must invoke it (the launcher's pre-reprice import witness does), because a reprice run against an
unpatched config would produce a delta of exactly zero and look like a clean no-op.

PUBLIC API: SUPERSEDED, ELECTED, armshift817_delta(prev_below_m, below_m), elected_is_live(),
    constants()
SIDE-EFFECTS: none -- pure functions; reads config, never writes it.
SIZE: soft 60 / hard 80 LoC; function cap 30.
COVERAGE: tests/unit/test_armshift_817.py -- the frozen superseded restatement is cross-pinned
    against the LIVE functions under a config patched back to the superseded point (a drifted
    restatement fails there), the delta is driven both directions, and elected_is_live() is
    driven to BOTH verdicts.
"""
from src.drl import config
from src.drl.reward_mrrl import ceilrate_penalty, endpoint_ramp_penalty

#: (arm_m, W_ENDPOINT_RAMP, W_CEILRATE) -- the point a pre-817 buffer was priced at.
SUPERSEDED = (0.80, 0.20, 0.5)
#: The elected point (ARMSHIFT-817). Config must carry these when the term runs.
ELECTED = (1.40, 1.00, 1.25)


def _superseded_half(prev_below_m, below_m):
    """The two vertical endpoint terms restated at the FROZEN superseded arm AND weights."""
    arm, w_ep, w_cr = SUPERSEDED
    up = -below_m
    total = -w_ep * max(0.0, up - arm)
    if up > arm:
        up_rate = (up - (-prev_below_m)) / config.CEILRATE_DT_S
        total += -w_cr * max(0.0, up_rate)
    return total


def armshift817_delta(prev_below_m, below_m):
    """(ramp + rate at the ELECTED point) - (ramp + rate at the SUPERSEDED point)."""
    elected = (endpoint_ramp_penalty(below_m)
               + ceilrate_penalty(prev_below_m, below_m))
    return elected - _superseded_half(prev_below_m, below_m)


def elected_is_live():
    """Fail closed: the elected half is the LIVE functions, so config MUST carry the elected point.

    Against an unpatched config this term returns 0.0 everywhere -- a silent no-op that reads as a
    clean reprice. Raises rather than returning False so a caller cannot ignore it by accident.
    """
    live = (config.W_ENDPOINT_RAMP_START_M, config.W_ENDPOINT_RAMP, config.W_CEILRATE)
    if live != ELECTED or config.CEILRATE_START_M != ELECTED[0]:
        raise AssertionError(
            "ARMSHIFT-817 reprice needs config AT THE ELECTED POINT "
            f"{ELECTED} (both arms {ELECTED[0]}); config reads arms "
            f"({config.W_ENDPOINT_RAMP_START_M}, {config.CEILRATE_START_M}), "
            f"W_ENDPOINT_RAMP {config.W_ENDPOINT_RAMP}, W_CEILRATE {config.W_CEILRATE}")
    return True


def constants():
    return {"W_ENDPOINT_RAMP": config.W_ENDPOINT_RAMP, "W_CEILRATE": config.W_CEILRATE,
            "CEILRATE_DT_S": config.CEILRATE_DT_S,
            "ELECTED_ARM_M": config.W_ENDPOINT_RAMP_START_M,
            "SUPERSEDED_ARM_M": SUPERSEDED[0],
            "SUPERSEDED_W_ENDPOINT_RAMP": SUPERSEDED[1],
            "SUPERSEDED_W_CEILRATE": SUPERSEDED[2]}
