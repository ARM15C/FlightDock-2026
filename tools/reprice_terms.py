# =============================================================================
# RESPONSIBILITY: the TERM REGISTRY for the offline replay-buffer reprice --
#   which stateless reward term each --term selects, which observation axes it
#   decodes, the norm each axis is DECLARED to carry, and the fail-closed
#   checks that must pass before any reward is touched. Split out of
#   tools/recompute_buffer_rewards.py by R1ARM-723 when adding the
#   "precontact" term pushed that file THROUGH its hard cap: raising a hard
#   cap to fit code is the move that is not available, so the file split
#   instead. The boundary is real, not cosmetic -- this module is PURE
#   (no disk, no argv, no pickles) and the caller keeps all the I/O.
#
#   Two term SHAPES:
#     PAIR fn(prev, curr) on ONE axis -- ceiling, nopass, e5_ramp
#     POINT fn(*curr) on EVERY axis -- precontact
#
#   Every delta calls the SAME stateless functions the live env uses
#   (src/drl/reward_mrrl.py) -- one source of truth, bit-exact with the live
#   call site. A full from-scratch recompute is impossible (stateful terms)
#   and is deliberately NOT attempted; "precontact" qualifies precisely
#   BECAUSE it is flat in time (no dwell, no history).
# PUBLIC API: TERMS, axes(term), axis_index(term), assert_decode_is_sane().
# SIDE-EFFECTS: NONE. Reads config only; never touches disk or the env.
# CAPS: file soft 130 / hard 160 code-LoC; functions <= 60.
#   MEASURED 121 at the split. Stated, not quietly exceeded.
# COVERAGE: tests/unit/test_recompute_buffer_precontact.py (the POINT term and
#   its guards), tests/unit/test_recompute_buffer_rewards_nopass.py (PAIR).
# =============================================================================
import numpy as np

from src.drl import config
from src.drl.env_formation_sitl import check_contact_m3b
from src.drl.reward_mrrl import (ceiling_shaping_reward, ceilrate_penalty,
                                 endpoint_ramp_penalty, nopass_shaping_reward,
                                 precontact_reward)
from tests import formation_config as _fc
from tools.reprice_armshift import armshift817_delta, constants as _a817_constants

_PRECONTACT_BEHIND_M = _fc.PRECONTACT_BEHIND_M
_PRECONTACT_BELOW_M = _fc.PRECONTACT_BELOW_M
_CONTACT_BEHIND_M = _fc.CONTACT_BEHIND_M
_CONTACT_BELOW_M = _fc.CONTACT_BELOW_M


def _e5_ramp_delta(prev_below_m, below_m):
    """E5/S20SPLIT reprice: the F2 ramp delta ALONE, applied at s' exactly as
    the live wiring applies it. NEVER use --term ceiling for this reprice:
    ceiling_shaping_reward now CONTAINS the ramp, so re-running it on the
    already-priced 2term buffer would re-add asym + PBRS (double-count).
    `prev_below_m` is accepted for the _TERMS calling convention and unused —
    the ramp is an endpoint term with no potential half."""
    del prev_below_m
    return endpoint_ramp_penalty(below_m)


# E5V-758 (SIGNED 2026-08-20): both vertical arms moved 1.20 -> 0.80. A buffer
# priced at the 1.20 arms (47bd1992 chain) needs the ARM-SHIFT delta:
#   (ramp + rate at the ELECTED arm) - (ramp + rate at the SUPERSEDED 1.20).
# The new halves come from the LIVE functions (config now carries 0.80); the
# superseded halves restate the frozen 1.20 formulas verbatim and are
# cross-pinned against the live functions under a patched config by
# tests/unit/test_e5v_758.py — a drifted restatement fails there, not here.
_E5V_SUPERSEDED_ARM_M = 1.20


def _e5v_armshift_delta(prev_below_m, below_m):
    new = (endpoint_ramp_penalty(below_m)
           + ceilrate_penalty(prev_below_m, below_m))
    up = -below_m
    old_ramp = -config.W_ENDPOINT_RAMP * max(0.0, up - _E5V_SUPERSEDED_ARM_M)
    old_rate = 0.0
    if up > _E5V_SUPERSEDED_ARM_M:
        up_rate = (up - (-prev_below_m)) / config.CEILRATE_DT_S
        old_rate = -config.W_CEILRATE * max(0.0, up_rate)
    return new - (old_ramp + old_rate)


# R1ARM-723 reprice: the WEIGHTED r_precontact a transition earns under the
# ARMED v3 ladder. A buffer priced before the arming holds rewards from a
# different reward function; training on it un-repriced fits the critic to a
# mixture of two definitions.
#   REPRICEABLE because r_precontact is FLAT IN TIME -- it pays from the first
# tick in band and reads no dwell counter, so unlike r_dwell it is fully
# determined by the endpoint state. That is what puts it inside this file's
# "stateless terms" responsibility rather than outside it.
#   in_contact_m3b is DECODED and passed, never assumed False. The bands are
# provably disjoint (contact needs |behind_m| <= 0.7 m; this band needs
# behind_m in [9.3, 10.7] m), and _assert_decode_is_sane MEASURES that on the
# real buffer instead of trusting the proof.
def _precontact_delta(behind_err, cross_err, below_err,
                      behind_m, cross_m, below_m):
    return config.R1_W_PRECONTACT * precontact_reward(
        behind_err, cross_err, below_err,
        check_contact_m3b(behind_m, cross_m, below_m))
# OVSHAPE-20260816 generalization: the recompute is AXIS-PARAMETERIZED. Each
# term binds an obs feature NAME (never a bare index), the stateless delta
# function, and the constants echoed into the report. The index is resolved
# from config.OBS_FEATURES_V42 at run time and the 50.0 norm asserted, so a
# wrong-axis invocation fails closed instead of decoding garbage.
TERMS = {
    "ceiling": dict(
        axes=(("below_m", 50.0),), fn=ceiling_shaping_reward,
        constants=lambda: {
            "W_ABOVE_ASYM": config.W_ABOVE_ASYM,
            "W_CEILING": config.W_CEILING,
            "CEILING_RAMP_START_M": config.CEILING_RAMP_START_M,
            "gamma": config.CEILING_SHAPE_GAMMA}),
    "nopass": dict(
        axes=(("behind_m", 50.0),), fn=nopass_shaping_reward,
        constants=lambda: {
            "W_INFRONT_ASYM": config.W_INFRONT_ASYM,
            "W_NOPASS": config.W_NOPASS,
            "NOPASS_RAMP_START_M": config.NOPASS_RAMP_START_M,
            "NOPASS_CAP_M": config.NOPASS_CAP_M,
            "gamma": config.NOPASS_SHAPE_GAMMA}),
    # E5/S20SPLIT (REPRICE_CONTRACT.md): the incremental delta between the
    # pre-E5 and post-E5 ceiling definitions, for buffers ALREADY priced with
    # the old ceiling term. No gamma in the echo: the term has no PBRS half.
    "e5_ramp": dict(
        axes=(("below_m", 50.0),), fn=_e5_ramp_delta,
        constants=lambda: {
            "W_ENDPOINT_RAMP": config.W_ENDPOINT_RAMP,
            "W_ENDPOINT_RAMP_START_M": config.W_ENDPOINT_RAMP_START_M}),
    # E5R Clause 1 (SIGNED 2026-08-19; dt AMENDED 0.1->0.25 same day,
    # DTFIX-735): the incremental ceilrate delta for buffers ALREADY priced
    # with the E5-era ceiling sum. PAIR term -- the rate needs the (prev,
    # curr) below endpoint pair. NEVER --term ceiling on an already-priced
    # buffer: same double-count trap _e5_ramp_delta pins above.
    "ceilrate": dict(
        axes=(("below_m", 50.0),), fn=ceilrate_penalty,
        constants=lambda: {
            "W_CEILRATE": config.W_CEILRATE,
            "CEILRATE_START_M": config.CEILRATE_START_M,
            "CEILRATE_DT_S": config.CEILRATE_DT_S}),
    # E5V-758 (SIGNED 2026-08-20): arm-shift delta for buffers ALREADY priced
    # at the 1.20 arms. PAIR term (the rate half needs the endpoint pair).
    # NEVER --term e5_ramp or --term ceilrate after this — both would re-add
    # a full term instead of the shift (the double-count trap above).
    "e5v_armshift": dict(
        axes=(("below_m", 50.0),), fn=_e5v_armshift_delta,
        constants=lambda: {
            "W_ENDPOINT_RAMP": config.W_ENDPOINT_RAMP,
            "W_CEILRATE": config.W_CEILRATE,
            "CEILRATE_DT_S": config.CEILRATE_DT_S,
            "ELECTED_ARM_M": config.W_ENDPOINT_RAMP_START_M,
            "SUPERSEDED_ARM_M": _E5V_SUPERSEDED_ARM_M}),
    # ARMSHIFT-817: arms AND weights move together, so its superseded half must
    # restate BOTH frozen -- it may NOT reuse e5v_armshift, which reads live
    # weights for the old half. Delta lives in tools/reprice_armshift.py (this
    # file is at its HARD cap; the term was SPLIT OUT, no cap was raised).
    # REQUIRES config already at the elected point: run reprice_armshift
    # .elected_is_live() first or the delta is a silent 0.0 everywhere.
    "armshift817": dict(
        axes=(("below_m", 50.0),), fn=armshift817_delta,
        constants=_a817_constants),
    # R1ARM-723. POINT term: the fn takes the s' value of EVERY axis, not a
    # (prev, curr) pair on one axis -- r_precontact has no potential half and
    # needs three error axes plus the three boom-tip axes for the contact rail.
    # Each axis declares its own norm; they are NOT all 50.0.
    "precontact": dict(
        point=True, fn=_precontact_delta,
        axes=(("behind_error_m", 10.0), ("cross_error_m", 5.0),
              ("below_error_m", 10.0), ("behind_m", 50.0),
              ("cross_m", 10.0), ("below_m", 50.0)),
        constants=lambda: {
            "R1_W_PRECONTACT": config.R1_W_PRECONTACT,
            "R_PRECONTACT_CAP": config.R_PRECONTACT_CAP,
            "R_PRECONTACT_TIER": config.R_PRECONTACT_TIER,
            "MILESTONE_THRESHOLDS_TIER": list(
                config.MILESTONE_THRESHOLDS[config.R_PRECONTACT_TIER]),
            "PRECONTACT_TARGET": [_PRECONTACT_BEHIND_M, _PRECONTACT_BELOW_M],
            "CONTACT_TARGET": [_CONTACT_BEHIND_M, _CONTACT_BELOW_M]}),
}


def axis_index(term):
    """THE single-axis accessor. Kept because banked campaign code depends on
    it -- analysis/.../w1419_gb_verify.py gates on the report key it feeds --
    and a multi-axis term is a usage error here, not a wider return type."""
    ax = axes(term)
    if len(ax) != 1:
        raise ValueError("%s is a multi-axis term; use _axes()" % term)
    return ax[0][0]


# [(index, norm)] for a term's axes, fail-closed on drift. The DECLARED norm is
# asserted against the live one PER AXIS -- that is what lets a term read the
# error axes (10.0/5.0/10.0) while still refusing to decode if a norm moves.
def axes(term):
    out = []
    for feature, want in TERMS[term]["axes"]:
        idx = list(config.OBS_FEATURES_V42).index(feature)  # raises if absent
        norm = float(config.OBS_NORMS[idx])
        if abs(norm - want) > 1e-9:
            raise ValueError("obs norm for %s is %.4f, declared %.4f — the "
                             "decode assumption is broken; refusing"
                             % (feature, norm, want))
        out.append((idx, norm))
    return out


# Validate the DECODE against a relation the delta never computes, and REPORT
# the frame split. NOT a tautology: behind_error_m and behind_m are SEPARATE
# obs features, so the TARGET they differ by is recoverable, and it must be one
# of the two the env can command. Wrong indices or norms imply a target that is
# neither, and fail here instead of silently repricing garbage.
#
# * MEASURED 2026-08-19, and it CORRECTS AN EARLIER CLAIM OF MINE. The target
# is FSM-DEPENDENT (env_formation_sitl._get_phase_targets): HOLD_PRECONTACT and
# earlier command the 10 m standoff; DOCKING_CREEP and later command the boom
# tip (0.0, 0.45). `behind_error_m` is therefore a FRAME-SWITCHING quantity,
# NOT a fixed offset from behind_m. On the carried s19-final buffer 67.5% of
# rows are in the CONTACT frame and 10.5% in the PRECONTACT one. I had claimed
# the contact rail "can never fire" from a geometry argument that assumed the
# standoff frame always; it fires on 10,825 rows. The audit REPORTS the split
# and the rail count so the claim is measured per buffer, never assumed again.
def assert_decode_is_sane(nxt, ax):
    (bi, bn), _, (li, ln), (Bi, Bn), (Ci, Cn), (Li, Ln) = ax
    b_err, b_m = nxt[..., bi] * bn, nxt[..., Bi] * Bn
    l_err, l_m = nxt[..., li] * ln, nxt[..., Li] * Ln
    # A row can only report its frame when BOTH error axes are off the clip
    # rail -- MEASURED: 39 rows of the carried buffer saturate `below_error_m`
    # while `behind_error_m` is live, and a behind-only mask mis-flags them.
    # Saturated rows are >= 11 m outside a 0.7 m tolerance on that axis, so
    # `n` >> 1 and the term pays 0 there; that is ASSERTED below, not assumed.
    live = ((np.abs(nxt[..., bi]) < 1.1 - 1e-6)
            & (np.abs(nxt[..., li]) < 1.1 - 1e-6))
    tb, tl = (b_m - b_err)[live], (l_m - l_err)[live]
    ok = np.zeros(tb.shape, bool)
    for w_b, w_l in ((_CONTACT_BEHIND_M, _CONTACT_BELOW_M),
                     (_PRECONTACT_BEHIND_M, _PRECONTACT_BELOW_M)):
        ok |= (np.abs(tb - w_b) < 0.5) & (np.abs(tl - w_l) < 0.5)
    if not ok.all():
        raise ValueError(
            "decode check FAILED: %d of %d unsaturated rows imply a "
            "(target_behind, target_below) that is neither the precontact "
            "standoff nor the contact target -- obs indices or norms are "
            "wrong; refusing" % (int((~ok).sum()), int(ok.size)))
    # Clipped rows are priced without a recoverable frame. That is only sound
    # because a clipped error axis is provably OUT of the band -- guaranteed
    # at IMPORT by _assert_clip_reach_exceeds_band(), not re-tested per row.
    pre = np.abs((b_m - b_err) - _PRECONTACT_BEHIND_M) < 0.5
    contact = np.vectorize(check_contact_m3b, otypes=[bool])(
        b_m, nxt[..., Ci] * Cn, nxt[..., Li] * Ln)
    return {"rows": int(b_m.size),
            "frame_unrecoverable_saturated": int((~live).sum()),
            "frame_precontact_rows": int((live & pre).sum()),
            "frame_contact_rows": int((live & ~pre).sum()),
            "contact_m3b_rows": int(contact.sum())}


def _assert_clip_reach_exceeds_band():
    """A CLIPPED error axis must be provably OUTSIDE the tier-4 band.

    The reprice prices clipped rows without recovering their frame, which is
    sound only if clipping always lands a row out of band. Each error axis
    reaches OBS_CLIP * norm meters; the band needs `tol`. Checked at IMPORT so
    it cannot be true today and silently false after a norm election -- and so
    it is one assertion over constants rather than a per-row scan of 200k
    rows. MEASURED headroom under R1b tier-1 tolerances: 11/2.0 = 5.5x,
    5.5/2.0 = 2.75x, 11/2.5 = 4.4x.
    """
    tol = config.MILESTONE_THRESHOLDS[config.R_PRECONTACT_TIER]
    for (feature, norm), t in zip(TERMS["precontact"]["axes"][:3], tol):
        if norm * config.OBS_CLIP <= t:
            raise ValueError(
                "%s clips at %.2f m, inside its %.2f m tolerance: a clipped "
                "row could sit IN band with an unknown true value, so the "
                "reprice can no longer price clipped rows" % (
                    feature, norm * config.OBS_CLIP, t))


_assert_clip_reach_exceeds_band()
