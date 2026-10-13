# =============================================================================

import os  # L-DIAG: needed for env-var driven diagnostic flags

# =============================================================================

# C++ handoff configuration
CPP_HANDOFF_ENABLE_RANGE_M = 30.0  # Hand off to C++ below this range
CPP_HANDOFF_DISABLE_RANGE_M = 40.0  # Return to Python above this (hysteresis)
CPP_HANDOFF_ENABLED = True  # Master switch for C++ handoff feature
CPP_HANDOFF_WARMUP_S = 2.0  # Sim-time dwell after PASS before engaging C++ (for always_on)

# --- Tranche C: Formation-mode altitude authority experiment ---
# When True, FORM_ENABLE is forced to 0 in close-in states to test whether
# C++ formation waypoint altitude causes the DC vertical climb artifact.
FORM_ENABLE_CLOSE_IN_DISABLE = False  # Default OFF (safe) — Run B confirmed culprit
FORM_ENABLE_CLOSE_IN_DISABLE_STATES = {
    "ALT_CAP_HOLD_RANGE",
    "DOCKING_CREEP",
}
# Safety abort: if cross-track exceeds this for > dwell_s, re-enable C++
FORM_ENABLE_EXPERIMENT_CROSS_ABORT_M = 2.5
FORM_ENABLE_EXPERIMENT_CROSS_ABORT_DWELL_S = 10.0

# =============================================================================
# CONTROLLER VERSION
# =============================================================================
CONTROLLER_VERSION = "BASELINE_v2.92_v9.10_GATES_0-2"

# =============================================================================
# PRECISION MODE + SENSOR STRATEGY + COUNTERS
# =============================================================================
# Close-range precision mode thresholds
PRECISION_MODE_THRESHOLD_M = 30.0  # Enter precision mode below this range
VERY_PRECISION_MODE_THRESHOLD_M = 15.0  # Extra-fine control below this range

# Micro-adjustment thresholds for precision mode
MICRO_HEADING_DELTA_DEG = 0.05  # Minimum heading change in precision mode
MICRO_SPEED_DELTA_MS = 0.05  # Minimum speed change in precision mode

# Sensor mode options
SENSOR_MODE_FUSED = "fused"  # Default: GPS+UWB fusion
SENSOR_MODE_UWB_ONLY = "uwb_only"  # UWB-only inside close range
SENSOR_MODE_UWB_DOMINANT = "uwb_dominant"  # De-weight GPS inside close range

# Cross-track filtering for close range (reduces jitter)
CROSS_TRACK_FILTER_ALPHA = 0.3  # EMA filter coefficient (lower = more smoothing)
CROSS_RATE_FILTER_ALPHA = 0.2  # EMA filter for cross rate (extra smoothing)

# =============================================================================
# SENSOR MODE WIRING - UWB VALIDITY CRITERIA
# =============================================================================
# These constants define when UWB data is considered valid and usable.
# If any criterion fails, we fall back to GPS/fused.

UWB_STALE_THRESHOLD_S = 0.5  # Max age for UWB data before considered stale
UWB_RANGE_MIN_M = 0.5  # Minimum valid UWB range (below = likely error)
UWB_RANGE_MAX_M = 150.0  # Maximum valid UWB range (beyond = unreliable)
UWB_JUMP_THRESHOLD_M = 10.0  # Max range jump between consecutive samples
UWB_NAN_IS_INVALID = True  # Treat NaN UWB values as invalid

# UWB-dominant blending weights (Test 308)
# As range decreases from threshold to very_close, w_uwb ramps up
UWB_DOMINANT_WEIGHT_FAR = 0.5  # UWB weight at close_range_threshold
UWB_DOMINANT_WEIGHT_NEAR = 0.9  # UWB weight at very_precision threshold

# Fallback warning rate limit (don't spam logs)
SENSOR_FALLBACK_LOG_INTERVAL_S = 5.0  # Min seconds between fallback warnings

# =============================================================================
# MISSION PARAMETERS
# =============================================================================
# Altitude configuration
PRECONTACT_ALT_MSL = 890.0  # Precontact altitude MSL (10m below leader)
CONTACT_ALT_MSL = 900.0  # Contact altitude MSL (match leader for boom)
# Tranche F: Aligned with 45-deg boom model. 6.1 * sin(45deg) = 4.31m below leader CG.
BOOM_TIP_BELOW_OFFSET_M = 4.31  # Vertical component of boom tip below leader CG
BOOM_TIP_OFFSET_M = BOOM_TIP_BELOW_OFFSET_M  # Legacy alias (used by geometry.py)
# Horizontal (along-track) component of the boom tip behind leader CG.
# 45-deg boom: 6.1 * cos(45deg) ~= 4.31m, equal to the vertical component by symmetry.
# CANONICAL home for the controller (resolved via `from formation_config import *`)
# and the truth observer; rounded 4.31 matches the observer/env_method_c convention.
# NOTE (flagged, NOT reconciled here): BOOM_TIP_NOMINAL_BEHIND_M (below, = 4.3134,
# the exact 6.1*cos45) is a derived sibling used by env_formation_sitl/success_metrics
# — a pre-existing ~3.4mm naming/value split for the same physical quantity.
BOOM_HORIZONTAL_PROJECTION_M = 4.31  # Boom-tip horizontal projection (45-deg, rounded)
HOME_ALT_MSL = 584.1  # Home altitude MSL (will be updated from GPS)

# Speed configuration
CRUISE_SPEED_MS = 24.7  # Nominal cruise speed (m/s)
SAFE_FLOOR_MS = 19.0  # Minimum safe airspeed (m/s)
SAFE_MAX_MS = 31.0  # Maximum safe airspeed (m/s)

# Flying Boom parameters (from DHC-2 Beaver)
BOOM_STOWED_LENGTH_M = 2.4  # Stowed boom length
BOOM_MAX_LENGTH_M = 6.1  # Fully extended boom length
BOOM_UPDATE_RATE_HZ = 5.0  # Boom gimbal/extension update rate

# =============================================================================
# TRANCHE D: BOOM GEOMETRY MODEL (45 deg belly-mounted telescoping boom)
# =============================================================================
# FlightDock design parameter — NOT ATP-3.3.4.5 standard 30 deg.
# Boom is modeled as STATIC (fixed nominal deployed pose) in Tranche D.
# Dynamic boom model (articulation + extension telemetry) is a future deliverable.
import math as _math

BOOM_LENGTH_EXTENDED_M = 6.1  # fully extended
BOOM_LENGTH_STOWED_M = 2.4  # stowed
BOOM_MOUNT_ANGLE_DEG = 45.0  # below horizontal from belly hatch
BOOM_GIMBAL_RANGE_DEG = 20.0  # pitch and yaw, whole-boom articulation

# Computed nominal tip position in leader CG frame
BOOM_TIP_NOMINAL_BEHIND_M = BOOM_LENGTH_EXTENDED_M * _math.cos(
    _math.radians(BOOM_MOUNT_ANGLE_DEG)
)  # 4.31
BOOM_TIP_NOMINAL_BELOW_M = BOOM_LENGTH_EXTENDED_M * _math.sin(
    _math.radians(BOOM_MOUNT_ANGLE_DEG)
)  # 4.31
BOOM_TIP_NOMINAL_CROSS_M = 0.0

# Phase distance definitions (relative to boom tip)
STERN_CONVERSION_RANGE_M = 200.0  # Stern conversion target range

# Lateral tolerance (used for ALTITUDE_CAPTURE deviation check)
LATERAL_TOLERANCE_M = 2.0  # Lateral tolerance for both phases

# =============================================================================
# BODY-FRAME ENVELOPE DEFINITIONS
# =============================================================================
# Problem: 3D range gating is physically inconsistent when follower is
# intentionally 10m below leader. "10m range" is unattainable if you're
# 10m below AND trying to be behind the boom.
# Solution: Define precontact/contact as leader-body-frame envelopes.

# Precontact position targets (in leader body frame)
PRECONTACT_BEHIND_M = 10.0  # Target distance behind boom tip
PRECONTACT_CROSS_M = 0.0  # Target lateral offset (centered)
PRECONTACT_BELOW_M = 10.0  # Target altitude below leader (for boom)

# Precontact tolerances: the bare legacy aliases are DERIVED from the OUTER
# tier below (see "TWO-TIER ENVELOPES") and are defined there, not here.
# They used to be literals at this point in the file, and
# PRECONTACT_BELOW_TOL_M went stale at 3.0 when TEST 400 raised the OUTER
# value to 4.0 - the same re-fork that T-386 removed from geometry.py.

# Contact position targets (at boom tip)
CONTACT_BEHIND_M = 0.0  # At boom tip (0m behind)
CONTACT_CROSS_M = 0.0  # Centered laterally
CONTACT_BELOW_M = 0.45  # Terminal plane 0.45m below tip: measured E_peak 2.0m vs guard -2.0
# requires this margin (creep retune 2026-06-12, analysis/b2pp_closeout_and_creep_retune_scope_20260612T082503Z §5)

# Contact tolerances (tighter than precontact)
CONTACT_BEHIND_TOL_M = 0.5  # ±0.5m behind tolerance (used in summary output)

# =============================================================================
# TWO-TIER ENVELOPES (OUTER = capture/stay, INNER = dwell accumulate)
# =============================================================================
# OUTER envelopes: Used for state capture (REJOIN->HOLD) and stay-in-state logic
# INNER envelopes: Used for dwell timer accumulation (the actual requirement)

# P1: Wider ENTRY envelope (capture-only, not stay/dwell)
PRECONTACT_BEHIND_ENTRY_TOL_M = 4.0  # ±4.0m - widened for native Linux TECS tracking (cloud adaptation)
PRECONTACT_CROSS_ENTRY_TOL_M = 8.0  # ±8.0m - wide lateral capture
PRECONTACT_BELOW_ENTRY_TOL_M = 5.0  # ±5.0m - wide altitude capture

# Precontact OUTER tolerances (keep existing values for capture/stay)
PRECONTACT_BEHIND_OUTER_TOL_M = 1.5  # ±1.5m - same as original PRECONTACT_BEHIND_TOL_M
PRECONTACT_CROSS_OUTER_TOL_M = 2.0  # ±2.0m - same as original
# TEST 400: Increased from 3.0m to 4.0m to accommodate observed p2p_below of 6.8-7.5m
# (±3.4-3.75m oscillations). This provides buffer against TECS altitude tracking lag.
PRECONTACT_BELOW_OUTER_TOL_M = 4.0  # ±4.0m - widened for altitude oscillations

# FSMDWELL-503 remedy R-2b, elected 2026-08-14 (FSMDWELL-R2B-20260814).
# DEFAULT-OFF: with the flag unset every value here is exactly what flew.
#
# The defect: this file declares "OUTER = capture/stay, INNER = dwell accumulate"
# (:156-159), which requires OUTER >= INNER. On the behind axis it is INVERTED --
# INNER 2.0 vs OUTER 1.5 -- so in |behind_err| in (1.5, 2.0] a leg accrues dwell
# credit AND outer-exit grace at the same time. The exit fires first (1.0 s) and
# clears the credit, making the 5.0 s PRECONTACT_HOLD_DURATION_S structurally
# unreachable in that band. Offline replay over banked legs: this repair fixes
# 3/3 stall legs and preserves the passing control; repointing the dead v9.9
# grace constant fixes 1/3 and widening OUTER cross fixes 0/3 (analysis/
# fsmdwell503_rca_20260814T1700Z/FSMDWELL_RCA.md). 2.0 is the minimal sufficient
# value measured by sweep -- exactly INNER, the smallest value that makes this
# file's own declared intent true.
#
# PLACEMENT IS LOAD-BEARING: this override sits BEFORE the T-389 alias block
# below so PRECONTACT_BEHIND_TOL_M derives from the repaired value. Overriding
# AFTER the aliases is the FD_CONTACT_MATCH_M3B trap (:352-356), where the bare
# alias keeps the pre-override value -- see the docstring of
# tests/unit/test_tolerance_single_source.py::
# test_formation_config_own_legacy_aliases_track_its_outer_tier.
#
# SCOPE, disclosed because this constant is deliberately single-source (T-389):
# the flag widens the behind tolerance for EVERY consumer, not just the FSM.
#   * formation_envelope.py:71 / geometry.py:591 is_in_precontact_outer_envelope
#     -- the FSM stay/ejection envelope. THIS IS THE INTENT.
#   * env_formation_sitl.py:4713 T-391 warmup acceptance gate -- becomes MORE
#     PERMISSIVE: an episode may hand off up to 2.0 m off-station longitudinally
#     instead of 1.5 m. Side effect, not the intent; the paired validation panel
#     measures it (R2B_PANEL_PREREG.md).
#   * drl_controller_ref.py:198 _maybe_latch_m2 -- same widening on the M2 latch.
#   * env_formation_sitl.py:5084 env_in_station -- descriptive witness only.
#   * baseline_formation_controller.py:9876 run_self_tests() nesting check flips
#     from FAIL to PASS with the flag ON (it asserts OUTER >= INNER, the very
#     invariant this repairs); manual --self-test CLI only -- disclosed.
# DEFAULT FLIPPED TO ON 2026-08-15 (this session: "ensure
# and triple-check that FD_FSM_NESTING_REPAIR is defaulted to ON"). Basis:
# offline replay 3/3 · 12-leg paired panel zero flight cost (R2BVERDICT-551) ·
# retro-audit 9/300 banked legs flip incl. failed gate leg W14_43233
# (RETRO-552). Only the literal "0" disables (fail-safe direction is now ON);
# the OFF spelling exists for A/B panels and the historical replay venue.
FD_FSM_NESTING_REPAIR = os.environ.get("FD_FSM_NESTING_REPAIR", "1").strip() != "0"
if FD_FSM_NESTING_REPAIR:
    PRECONTACT_BEHIND_OUTER_TOL_M = 2.0  # == INNER; repairs the tier inversion

# T-389: legacy bare aliases, DERIVED here so they can never re-fork from the
# OUTER tier. PRECONTACT_BELOW_TOL_M was 3.0 against an OUTER of 4.0 until
# now; its only live reader is run_self_tests() in the frozen baseline
# controller (:9609), where the compared error is 0.0 and the result is
# identical either way => zero functional delta. Pinned by
# tests/unit/test_tolerance_single_source.py.
PRECONTACT_BEHIND_TOL_M = PRECONTACT_BEHIND_OUTER_TOL_M
PRECONTACT_CROSS_TOL_M = PRECONTACT_CROSS_OUTER_TOL_M
PRECONTACT_BELOW_TOL_M = PRECONTACT_BELOW_OUTER_TOL_M

# Precontact INNER tolerances (tighter for dwell requirement)
# TEST 400: Aligned with M2 milestone tolerances to enable state progression
# ChatGPT-5 Assessment RC1: INNER was ±1.0m, M2 was ±2.0m - mismatch caused dwell failure
PRECONTACT_BEHIND_INNER_TOL_M = 2.0  # ±2.0m - match M2 milestone
PRECONTACT_CROSS_INNER_TOL_M = 2.0  # ±2.0m - match M2 milestone
PRECONTACT_BELOW_INNER_TOL_M = 2.5  # ±2.5m - keep as-is (already adequate)

# Contact OUTER tolerances (keep existing values for capture/stay)
CONTACT_BEHIND_OUTER_TOL_M = 0.5  # ±0.5m - same as original CONTACT_BEHIND_TOL_M
CONTACT_CROSS_OUTER_TOL_M = 0.5  # ±0.5m - same as original
CONTACT_BELOW_OUTER_TOL_M = 1.0  # ±1.0m - same as original

# Contact INNER tolerances (Stage 1 precision - corrected to ±0.20m all axes)
CONTACT_BEHIND_INNER_TOL_M = 0.20  # ±0.20m - longitudinal (behind leader)
CONTACT_CROSS_INNER_TOL_M = 0.20  # ±0.20m - lateral (corrected from 0.10)
CONTACT_BELOW_INNER_TOL_M = 0.20  # ±0.20m - vertical (corrected from 0.40)

# =============================================================================
# CONTACT CAPTURE GATE (Vc + cross-rate stability before HOLD_CONTACT)
# =============================================================================
CONTACT_VC_MAX_MS = 0.30  # Max closure rate to enter contact hold (P1: 0.20 * 1.5)
CONTACT_VC_STABLE_TIME_S = 1.0  # Must be stable for this long
CONTACT_CROSS_RATE_MAX_MS = 0.3  # Max lateral rate for contact capture
CONTACT_VC_GATE_LOG_INTERVAL_S = 2.0  # Rate limit for gate status logging

# =============================================================================
# MULTI-SEGMENT CLOSURE TAPER (gentler near target)
# =============================================================================
CLOSURE_TAPER_VERY_NEAR_ERR_M = 2.0  # Below this, use very-near closure limit
CLOSURE_TAPER_ULTRA_NEAR_ERR_M = 0.5  # Below this, use ultra-near (nearly stopped)
MAX_CLOSURE_RATE_VERY_NEAR_MS = 0.225  # Max closure at 2m error (P1: 0.15 * 1.5)
MAX_CLOSURE_RATE_ULTRA_NEAR_MS = 0.05  # Max closure at 0.5m error

# --- CSV2 crosswind-dock: active anticipatory closure-rate braking ---
# (controller flag FD_CLOSURE_BRAKE_ENABLE, default OFF; byte-identical when OFF).
# Counters gust-driven airspeed-tracking overshoot PAST the boom-tip by commanding
# active deceleration (airspeed below leader ground speed) when the realized closure
# rate exceeds a safe stopping-distance profile v_safe=sqrt(2*a*gap) within the
# terminal engage zone. The existing FINAL_APPROACH_SCHEDULE already tapers the
# positive closure command; this adds the missing ACTIVE braking the min-cap cannot.
# Starting points; sized by SITL A/B in R2. NOT a per-stage toggle (continuous law).
CLOSURE_BRAKE_ENGAGE_M = 3.0       # engage active braking when behind-gap < this (m)
CLOSURE_BRAKE_A_MS2 = 0.5          # safe-stop decel: v_safe = sqrt(2*a*gap) (m/s^2)
CLOSURE_BRAKE_GAIN = 1.0           # active decel per (m/s) of over-closing beyond v_safe
CLOSURE_BRAKE_MAX_MS = 2.0         # max commanded decel below leader ground speed (m/s)
CLOSURE_BRAKE_MIN_GAP_M = 0.5      # do NOT engage below this gap: FD-006 + station-keeping own
                                   # that zone, and v_safe=sqrt(2*a*gap)->0 there would over-fire
                                   # on calm station-keeping jitter (R2 calm-regression fix)
CLOSURE_BRAKE_OVER_DEADBAND_MS = 0.3  # fire only when closure_rate exceeds v_safe by this margin
                                      # (m/s) -> ignores calm jitter; brake acts on the excess

# --- #336 subtractive cross-gate decouple + behind deadband (controller flag
# FD_SPD_CROSS_DECOUPLE, default OFF; byte-identical when OFF). Matches the #335
# [MEASURED] binder (CROSS_GATE caps airspeed BELOW leader 24.7 via perceived-GS
# noise + evc->0 under lateral imperfection => follower recedes => limit-cycle).
# Active ONLY in DOCKING_CREEP / HOLD_CONTACT. Two effects, both in compute_speed_cmd:
#   (D2 floor) the climb clamp can freeze approach (hold exactly leader speed,
#                 zero closure) but can NEVER command below the leader's TRUE 24.7.
#   (D1 deadband) zero the along-track P-term inside +/-DB so the D-term nulls the
#                 closing rate at contact (converge to exactly leader speed).
# #341 v10.1: ADOPTED default-ON (kill-switch FD_SPD_CROSS_DECOUPLE=0). Validated
# across #336/#337/#338/#340 (11 flights) + #341 defaults-only re-fly. Constituent
# of the SHA-pinned v10.1 MRRL reference (adopted with park-bias as ONE config).
SPD_CROSS_DECOUPLE_ENABLE = os.environ.get("FD_SPD_CROSS_DECOUPLE", "1").strip() == "1"
SPD_BEHIND_DEADBAND_M = 0.35  # along-track deadband (m); research band 0.3-0.4, < 0.7 M3b box
# --- #340 F1 park-bias (FD_SPD_PARK_BIAS; nests INSIDE the #336 D1 lever gate =>
# inert unless SPD_CROSS_DECOUPLE_ENABLE AND state is DC/HC). Shifts the P-path
# park point FORWARD by SPD_PARK_BIAS_M so that under the #338 [MEASURED]
# contact-window truth-aft offset (perception under-reads behind by 0.3-0.8m on
# some seeds) BOTH perception and truth land in the M3b box. P-term-path ONLY:
# closure schedule/decel/cap/stuck consumers of the raw behind_error are untouched.
# #341 v10.1: ADOPTED default-ON (kill-switch FD_SPD_PARK_BIAS=0). #340 converted
# both #338 fail-seeds (F7 4.90s / F404 3.45s), 0 hard-fails. Second v10.1 constituent.
SPD_PARK_BIAS_ENABLE = os.environ.get("FD_SPD_PARK_BIAS", "1").strip() == "1"
SPD_PARK_BIAS_M = float(os.environ.get("FD_SPD_PARK_BIAS_M", "0.35"))  # m fwd; research band 0.3-0.4

# Range-dependent MIN_CLOSURE floor (prevent forced closure at target)
# Increased minimum closure floors to prevent 30m stall
MIN_CLOSURE_FLOOR_FAR_MS = 0.50  # Min closure when behind_err > 20m
MIN_CLOSURE_FLOOR_NEAR_MS = 0.20  # Min closure when 5m < behind_err < 20m
MIN_CLOSURE_FLOOR_VERY_NEAR_MS = 0.12  # Min closure when 1m < behind_err < 5m (P1: 0.08 * 1.5)
MIN_CLOSURE_FLOOR_ULTRA_NEAR_MS = 0.0  # No forced closure when behind_err < 1m

# =============================================================================
# v9.2 Patch C: CLOSE-IN CLOSURE SCHEDULE
# =============================================================================
# Piecewise linear schedule for desired closure rate based on behind error.
# Replaces fixed max_closure taper in close range with a range-dependent
# target. A physics-based braking clamp prevents overshoot at all ranges.
CLOSURE_SCHEDULE_BREAKPOINTS = [
    # (behind_err_m, target_closure_ms)
    (15.0, 1.5),
    (5.0, 0.35),  # K2: 0.50→0.35 — align with FINAL_APPROACH 5m
    (3.0, 0.25),  # N-R revert: 0.18→0.25 (restore Tranche M closure energy)
    (1.0, 0.15),  # N-R revert: 0.10→0.15
    (0.3, 0.05),  # N-R revert: 0.03→0.05
]
CLOSURE_SCHEDULE_FINAL_MS = 0.0  # At/below smallest breakpoint
CLOSURE_BRAKING_A_MAX_MS2 = (
    0.12  # Max deceleration for braking clamp (RC-2: realistic for TECS T=5s)
)
CLOSURE_BRAKING_A_MAX_DOCKING_MS2 = (
    0.25  # Docking states: higher authority for tighter position hold
)
CLOSURE_SCHEDULE_ENABLE_RANGE_M = 25.0  # Only use schedule below this range

# =============================================================================
# HOLD_CONTACT PI CONTROLLER (replaces inline proportional law)
# =============================================================================
HOLD_CONTACT_KP = 0.30  # Proportional gain on behind_error
HOLD_CONTACT_KI = 0.05  # Integral gain (slow wind-up)
HOLD_CONTACT_KD = 0.20  # Derivative gain (uses closure_rate)
HOLD_CONTACT_I_CLAMP_MS = 0.4  # Max integrator contribution (m/s)

# =============================================================================
# CROSS-TRACK PI HEADING BIAS (replaces HOLD_CONTACT-only integrator)
# =============================================================================
# Problem: Persistent cross-track error (~0.5m) prevents INNER dwell accumulation.
# Solution: PI bias that activates during DOCKING_CREEP (not just HOLD_CONTACT)
# so cross_err is already near zero when entering contact envelope.
# Gating: Enable below certain range AND abs(cross_err) not wildly off.
CROSS_BIAS_ENABLE_RANGE_M = 30.0  # Start PI below this range (matches close-range)
CROSS_BIAS_ENABLE_MAX_CROSS_M = 2.0  # Only integrate if cross_err below this
CROSS_BIAS_ENABLE_NEAR_CONTACT_M = 5.0  # Extra gating: behind_err < this for creep
CROSS_BIAS_KP = 0.02  # Proportional gain (deg/m) — P1 baseline
CROSS_BIAS_KI = 0.08  # Integral gain (deg/m·s)
CROSS_BIAS_I_CLAMP_DEG = 2.0  # Max integrator heading bias (±deg) — P1 baseline
CROSS_BIAS_I_DECAY = 0.995  # Slower decay when not integrating
CROSS_BIAS_I_DECAY_TAU_S = 9.98  # Time constant: -0.05/ln(0.995) ≈ 9.98s

# --- CSV2_R7_1 rejoin cross-track crab integral (flag FD_CROSS_CRAB_ENABLE, default OFF) ---
# ILOS-style integral extended into REJOIN_PRECONTACT/HOLD_PRECONTACT to NULL the
# steady-state cross offset under a continuous mean crosswind (pure-P only balances it).
# Separate integrator state (_cross_i_rejoin_deg); calm-inert via deadband + sustained-tick
# trigger; deterministic tick-dt; leader-yaw turn-freeze. Inert unless the flag is ON.
CROSS_BIAS_REJOIN_I_CLAMP_DEG = 6.0       # Max rejoin crab (±deg). Covers S1 crab ≈4.1°
                                          # (asin(2.0·sin45/20)); does NOT cover S3 ≈8.1°
                                          # (asin(4.0·sin45/20)) — S3 full nulling NOT claimed.
CROSS_BIAS_REJOIN_ENABLE_MAX_CROSS_M = 25.0  # Engage up to this |cross|; > documented S1
                                          # pure-P equilibrium ≈18 m so the integrator is
                                          # NOT disabled in the very failure mode it fixes.
CROSS_BIAS_REJOIN_DEADBAND_M = 0.75       # Integrate only when |cross| one-sided beyond this
                                          # for N ticks. Below M2 tol (2 m); above calm noise
                                          # (~0.2 m) → calm-inert.
CROSS_BIAS_REJOIN_SUSTAIN_N_TICKS = 10    # Consecutive same-sign-beyond-deadband ticks before
                                          # engaging (≈0.5 s at 20 Hz): confirms sustained drift.
CROSS_BIAS_REJOIN_TURN_FREEZE_DEG_S = 3.0  # Freeze integrator when |leader yaw-rate| exceeds
                                          # this (clears straight-leg noise ~1 deg/s; triggers
                                          # before racetrack turns ~15+ deg/s).
CROSS_BIAS_REJOIN_TURN_BLEED_TAU_S = 1.5   # Lever A: when frozen specifically by a leader turn,
                                          # bleed the rejoin integrator with this FAST time
                                          # constant (vs the slow CROSS_BIAS_I_DECAY_TAU_S ≈9.98 s
                                          # for other freezes) to shed turn-onset windup before it
                                          # carries a stale crab through the turn.

# OUTER-exit grace time (prevents one-sample boundary exits)
# #459 Candidate A (R7.1, D1-authorized): sized to 0.5 s (from a dead 0.25) to bridge the
# sub-second overtake blips that churn HOLD_CONTACT<->DOCKING_CREEP; DEAD unless
# FD_CONTACT_OUTER_EXIT_GRACE_ENABLE=1 (the grace wiring reads it), so this value change is
# byte-identical when the flag is OFF. Self-test range [0.1,1.0] honoured.
CONTACT_OUTER_EXIT_GRACE_S = 0.5  # Must be outside OUTER this long to exit (#459)
CONTACT_OUTER_CATASTROPHIC_MULT = 2.0  # Immediate exit if error > mult * tolerance
# #459 Candidate A master gate: OFF => the HC->DC exit is instantaneous (byte-identical to
# the frozen baseline). ON => temporal exit-grace hysteresis + catastrophic backstop.
# D-A ADOPTED default-ON (operator authorization 2026-07-16: "I authorize D-A: Candidate A
# default-ON"). Evidence: #463 A/B PRIMARY PASS 3/3 (v3 3.05/8.35/10.25 s, churn 56->26) vs
# same-day matched CTL 0/3; #466 refuted the envelope-match alternative 0/3 (CA dominates
# every seed). Explicit FD_CONTACT_OUTER_EXIT_GRACE_ENABLE=0 is the runtime kill-switch.
FD_CONTACT_OUTER_EXIT_GRACE_ENABLE = (
    os.environ.get("FD_CONTACT_OUTER_EXIT_GRACE_ENABLE", "1").strip() == "1")

# #466: contact-HOLD envelope matched to the M3b certification box.
# OFF (default) => legacy 0.5/0.5/1.0 above — behavior-identical unset. ON => behind/below
# take the M3b values (0.7/0.7; cross already matched at 0.5), deleting the 0.5-0.7 behind
# churn band (#440/#463) and ending contact-holds up to 1.0 m ABOVE the tip (belly axis).
# Catastrophic exit radius is PINNED at ABS 1.0 m: the R7.1 controller
# computes MULT*BEHIND_TOL, so MULT here is DERIVED (1.0/0.7), no longer a semantic 2x.
# (run_self_tests()'s MULT>=1.5 expectation encodes the old semantic; it runs only under
# the manual --self-test CLI and would print 1 FAIL there with this flag ON — disclosed.)
FD_CONTACT_MATCH_M3B = os.environ.get("FD_CONTACT_MATCH_M3B", "0").strip() == "1"
if FD_CONTACT_MATCH_M3B:
    CONTACT_BEHIND_OUTER_TOL_M = 0.7
    CONTACT_BELOW_OUTER_TOL_M = 0.7
    CONTACT_OUTER_CATASTROPHIC_MULT = 1.0 / CONTACT_BEHIND_OUTER_TOL_M  # product == 1.0 ABS

# v9.10 Gate 2: SYMMETRIC DEBOUNCING - entry AND exit grace
# Entry grace: require sustained time INSIDE envelope before entering HOLD
# Exit grace: require sustained time OUTSIDE envelope before exiting HOLD
# This prevents single-sample boundary transitions in both directions.

HOLD_ENTRY_GRACE_S = (
    0.5  # Must be inside OUTER continuously to enter HOLD (shorter - easier to enter)
)
HOLD_EXIT_GRACE_S = (
    1.0  # Must be outside OUTER continuously to exit HOLD (longer - harder to exit)
)

# Legacy exit grace (still used for outer envelope exit)
# P9.0 v9.9: Increased from 1.0s to 2.0s to fix REJOIN↔HOLD dithering at 3×
# ChatGPT-5 Assessment: 3× bounces REJOIN↔HOLD every ~1.25s, never reaches 5s dwell.
# Requires sustained violation before state transition.
PRECONTACT_OUTER_EXIT_GRACE_S = 2.0  # Must be outside OUTER this long to exit

# TEST 400: INNER envelope exit grace (matches OUTER pattern)
# ChatGPT-5 Assessment: Single-sample spikes were resetting dwell timer to zero
PRECONTACT_INNER_EXIT_GRACE_S = 0.25  # Must be outside INNER this long to reset dwell

# P9.0 v9.9: HYSTERESIS BANDS FOR PRECONTACT STATE TRANSITIONS
# Problem: State oscillates between REJOIN and HOLD due to noise at boundary.
# Solution: Enter at tight thresholds, exit only at looser thresholds.
PRECONTACT_HYSTERESIS_ENABLED = True
PRECONTACT_ENTRY_BEHIND_M = 12.0  # Enter HOLD when behind <= this
PRECONTACT_EXIT_BEHIND_M = 14.0  # Exit HOLD only when behind > this
PRECONTACT_ENTRY_CROSS_M = 2.0  # Enter HOLD when |cross| <= this
PRECONTACT_EXIT_CROSS_M = 3.0  # Exit HOLD only when |cross| > this

# =============================================================================
# TEST 401: CLIMB GUARD (Fix 4 - explicit progression guard)
# =============================================================================
# Problem: At 5x speedup, no state-machine oscillation occurs to naturally damp
# approach velocity. The controller transitions to ALTITUDE_CAPTURE too fast,
# overshooting and triggering S1_OVERTAKE.
# Solution: Require low behind-closure for a stable period before climb transition.

CLIMB_GUARD_VC_MAX_BEHIND_MS = 0.3  # Must have vc_behind below this
CLIMB_GUARD_STABLE_S = 1.0  # Must be stable for this duration (sim-time)
CLIMB_GUARD_LOG_INTERVAL_S = 2.0  # Rate limit for guard logging

# v9.2 Patch B: Climb closure clamp - prevent climb+close coupling
# During DOCKING_CREEP, clamp closure rate when altitude error is large.
# This prevents the autopilot from combining aggressive climb with aggressive
# closure, which leads to S1_OVERTAKE at high speedups.
CLIMB_CLAMP_ALT_ERR_M = 1.0  # Clamp closure if |below_m| > this
CLIMB_CLAMP_MAX_CLOSURE_MS = 0.2  # Max closure rate (m/s) while climbing
CLIMB_CLAMP_LOG_INTERVAL_S = 2.0  # Rate limit for clamp logging

# =============================================================================
# P1 v9.4: Altitude Capture Gating (boom-tip vertical alignment)
# =============================================================================
# Problem: DOCKING_CREEP starts while follower is still ~10m below boom tip.
# CLIMB_CLAMP then dominates (triggers when |below_m| > 1.0m), limiting closure
# to lead_gs + 0.2 m/s, causing stall at ~8-9m.
# Solution: Add ALT_CAP_HOLD_RANGE sub-state that holds range (~10m behind)
# while capturing vertical alignment to boom tip.

ALT_CAP_BELOW_TARGET_M = 0.0  # Target: aligned with boom tip vertically
ALT_CAP_BELOW_TOL_M = 1.0  # Must be within ±1.0m of target
ALT_CAP_VZ_TOL_MPS = 0.3  # Relative vertical rate must be ≤0.3 m/s
ALT_CAP_STABLE_S = 1.5  # Must hold stable for 1.5s sim-time
ALT_CAP_GRACE_S = 0.25  # Grace period before resetting stable dwell
ALT_CAP_HOLD_BEHIND_TARGET_M = 10.0  # Hold behind at ~10m during altitude capture
ALT_CAP_HOLD_BEHIND_TOL_M = 1.5  # Behind tolerance during hold
ALT_CAP_HOLD_CROSS_TOL_M = 2.0  # Cross tolerance during hold
ALT_CAP_LOG_INTERVAL_S = 2.0  # Rate limit for altitude capture logging


# Emergency breakaway + control law constants (DO NOT REMOVE)
NOPASS_GUARD_SAFETY_DOWN_SLEW = 8.0  # P5.2: Faster down-slew when safety active (m/s/s)
NOPASS_HARD_SLOWDOWN_DELTA_MPS = 3.0  # Slowdown vs leader when behind_m < 0

# v9.10 Gate 1: EMERGENCY_BREAKAWAY exit conditions
# ABORT must hold for minimum time, exit only when safely behind
ABORT_MIN_DWELL_S = 2.0  # Minimum sim-time in ABORT before exit allowed
ABORT_EXIT_BEHIND_MIN_M = 15.0  # Don't exit ABORT until behind >= this (2.5x CE_GUARD outer range)
ABORT_EXIT_STATE = "INTERCEPT"  # State to transition to after safe separation

# Chunk 9: Minimal guards — protect against the two real hazards
# GUARDRESTORE-756 (operator SIGNED 2026-08-20; record:
# analysis/guardrestore_20260820T0500Z/GUARDRESTORE_756.md). The EVAL system
# definition returns to -2.0: guard activation actuates throttle and controls,
# so a -1.75 trigger in an eval does not merely OBSERVE the controllers, it
# INTERVENES in them. -1.75 was a TRAINING-campaign device (bias the policy
# lower) and stays available ONLY through the env override below; the training
# launch contract (env_contract.TRAIN_REQUIRED, R6, exact "-1.75") REFUSES a
# launch without it, so training cannot silently revert to -2.0 either.
# Promotion exams run inside the training process and inherit -1.75 (operator
# elected "exam stays -1.75"). NEVER pool panels across this venue change:
# every panel through 2026-08-20 (incl. P3/BASE900) is -1.75-era.
# GUARD-ELECT-714's arrest-lag analysis (archive below) remains the measured
# record of what -2.0 costs in prevention (44% vs 90% at -1.75); the operator
# reaffirmed -2.0 with that analysis on the table.
GUARD_ALT_CEILING_BELOW_M = float(
    os.environ.get("FD_GUARD_CEILING_BELOW_M", "-2.0")
)  # Breakaway if follower exceeds this above boom-tip ref (envelope min: -1.15m)
# ---- decision archive (read before touching this constant again) ----
# GUARD-ELECT-714 (signed 2026-08-18) elected -1.75, superseding ONLY the
# threshold clause of GUARD-ELECT-588. 588 elected -1.90 on DETECTION
# (did the guard fire on a busting leg?) and was correct on that
# criterion. But the failure mode is ARREST LAG: the guard fires, the
# abort descends as designed, and a QuadPlane still cannot reverse
# vertical velocity instantly — so a bust is routinely DETECTED and NOT
# PREVENTED. Measured over 225 banked legs / 33 bust events, comparing
# the margin at fire against the EMPIRICAL arrest distribution
# (ARREST_WIDEN near-ceiling, n=8, 0.00-0.17 m):
#     -1.90 14.50/33 prevented (44%), 1 certain, 6 UNPREVENTABLE
#     -1.80 26.25/33 (80%), 15 certain, 1 unpreventable
#     -1.75 29.62/33 (90%), 23 certain, 0 unpreventable <-
# -1.75 is the FIRST threshold at which every bust has a fire with
# usable margin. Cost measured, not assumed: legs that actually fired
# EMERGENCY_BREAKAWAY passed 66/69 = 96% vs 31/38 = 82% for legs that
# never aborted — an abort does not cost a leg its pass, so the
# false-fire rate is not a cost in the currency that matters.
# Floor: the docking envelope minimum is -1.15 m, so -1.75 preserves
# 0.60 m of clearance; and below ~1.60 prevention goes NON-MONOTONIC
# (32.38 at 1.60 but 29.88 at 1.55) as the 2 s refractory consumes an
# early fire — a region this analysis does not support.
# The H1 scoring bar stays 2.00 (gate_safety_events.H1_BAR_M, DECOUPLED
# and unmoved by 588/714/756). Rollback of 756: delete the env-read and
# restore the -1.75 literal (requires a new operator signature; the
# GUARDRESTORE-756 detector makes a silent revert impossible).
GUARD_ALT_CEILING_PERSIST_S = (
    0.0  # GUARD-ELECT-588 (signed 2026-08-15, supersedes E1/T-341's 0.5):
    # single-tick semantics — peak-class H1 events dwell ~1 tick beyond the
    # bar, so ANY persistence >0 re-opens the miss class (SWEEP-588: p0.10
    # drops a real -2.07 m bust). Rollback: 0.5 (T-341 value).
)
GUARD_OVERTAKE_PERSIST_S = (
    10.0  # 10s continuous behind_m<0 = genuine overtake (envelope max: 3.85s)
)
GUARD_SPEED_CAP_RANGE_M = 1.0  # Boom-tip range threshold for soft speed cap
GUARD_SPEED_CAP_VC_MPS = 1.5  # Max closure rate near boom tip (m/s)


# =================================================================
# Option A: Two-Phase DOCKING_CREEP
# =================================================================
DC_PHASED_ENABLE = True  # Feature flag

# Sub-phase names
DC_SUBPHASE_VERT_STABILIZE = "DC_PHASE1_VERT_STABILIZE"
DC_SUBPHASE_CREEP_CLOSE = "DC_PHASE2_CREEP_CLOSE"

# Phase 1 → Phase 2 stability gate (wider to avoid deadlock at 1x)
DC_PHASE1_STABLE_BELOW_ABS_M = 1.00  # N0-revert: 1.25 regressed dc_below_err_p95 (0.38→0.985m)
DC_PHASE1_STABLE_CROSS_ABS_M = 1.00  # |cross| must be < this to advance
DC_PHASE1_DWELL_S = 2.5  # Must sustain for this many sim-seconds

# Phase 2 → Phase 1 fallback (hysteresis — wider than entry gate)
DC_PHASE2_ABORT_BELOW_ABS_M = 1.50  # Revert if |below| exceeds this
DC_PHASE2_ABORT_CROSS_ABS_M = 1.50  # Revert if |cross| exceeds this

# Fix 2: Phase 2 abort grace period (bounded)
DC_PHASE2_ABORT_GRACE_S = 1.0  # N-R revert: 2.0→1.0 (restore Tranche M)
DC_PHASE2_ABORT_HARD_BELOW_M = 3.0  # Immediate fallback if |below| exceeds (no grace)
DC_PHASE2_ABORT_HARD_RANGE_M = 3.25  # Immediate fallback if range below (matches CE inner)

# A1C1: Behind-gated corridor violation (replaces hard range fallback in Phase2)
DC_PHASE2_CORRIDOR_BEHIND_GATE_M = 1.00  # Only check corridor when behind < this
DC_PHASE2_CORRIDOR_CROSS_ABS_M = 1.00  # Cross misalignment limit inside corridor
DC_PHASE2_CORRIDOR_BELOW_ABS_M = 1.00  # Below misalignment limit inside corridor

# A1C2: Re-entry cooldown (prevents rapid Phase2/Phase1 oscillation)
DC_PHASE2_REENTRY_COOLDOWN_S = 0.50  # N-R revert: 1.50→0.50 (restore Tranche M)

# Phase 2 sink-rate trigger
DC_PHASE2_SINK_TRIG_MPS = -0.30  # Sustained sink rate threshold
DC_PHASE2_SINK_TRIG_HOLD_S = 0.50  # Must persist this long to trigger

# Fallback behavior
DC_PHASE2_CLOSURE_FREEZE_ON_LOSS = True  # Freeze closure (leader_gs) on fallback

# =================================================================
# Option E: Dynamic TECS_SPDWEIGHT (state-scoped, ramped)
# =================================================================
TECS_SPDWEIGHT_ENABLE = True  # Feature flag
TECS_SPDWEIGHT_CRUISE = 1.00  # Default/cruise value
TECS_SPDWEIGHT_ALT_CAP = 0.40  # Fix 1: Altitude bias during ALT_CAP_HOLD_RANGE (match DC Phase 1)
TECS_SPDWEIGHT_DC_PHASE1 = 0.40  # Mild altitude bias during stabilize
TECS_SPDWEIGHT_DC_PHASE2 = 0.30  # K-B revert: 0.35→0.30 (K-B5 regressed altitude authority)
TECS_SPDWEIGHT_HOLD_CONTACT = 0.10  # Altitude priority in contact hold
TECS_SPDWEIGHT_RAMP_S = 1.5  # Linear ramp time (seconds)
TECS_SPDWEIGHT_WRITE_MAX_HZ = 2.0  # Max PARAM_SET rate
TECS_SPDWEIGHT_READBACK_TOL = 0.05  # Acceptable readback mismatch
TECS_SPDWEIGHT_FAILSAFE_DISABLE_ON_MISMATCH = True  # Disable on readback failure

# K-B6: Faster TECS response for docking speed tracking
TECS_TIME_CONST_OVERRIDE = 4.0  # K-B6: 5.0→4.0 (faster energy→throttle response)
TECS_TIME_CONST_OVERRIDE_ENABLE = (
    False  # K-B revert: K-B6 (4.0s) regressed altitude; disable global override
)
TECS_TIME_CONST_ZONE_GATED = False  # N-R revert: True→False (disable zone-gated TECS override)


# =============================================================================
# Rec 5: BACKLOG TIME METRIC + BACKPRESSURE
# =============================================================================
RUNTIME_BACKLOG_SNAPSHOT_THRESHOLD_S = 0.50  # Trigger snapshot above this
BACKPRESSURE_DROP_THRESHOLD_S = 1.0  # Fast-forward queue above this
BACKPRESSURE_MIN_SPEEDUP = 5.0  # Only enable backpressure at 5x+

# =============================================================================
# ALTITUDE_CAPTURE PHASE CONSTANTS
# =============================================================================
# Problem: During DOCKING_CREEP, the controller commands altitude climb from
# 890m (precontact) to 900m (contact) while also closing distance. The autopilot
# (TECS) struggles to climb during precision speed control, resulting in
# insufficient vertical progress.
# Solution: Add explicit ALTITUDE_CAPTURE phase between HOLD_PRECONTACT and
# DOCKING_CREEP that performs the 10m climb while maintaining standoff distance.
# Only proceed to DOCKING_CREEP when altitude is captured.

ALT_CAPTURE_TOLERANCE_M = 1.5  # Must be within ±1.5m of target altitude
ALT_CAPTURE_STABLE_TIME_S = 3.0  # Must hold altitude for 3s to proceed
ALT_CAPTURE_VERT_RATE_MAX_MS = 0.3  # Vertical rate must be below this (m/s)
ALT_CAPTURE_LOG_INTERVAL_S = 2.0  # Rate limit for altitude capture logging

# =============================================================================
# ALTITUDE_CAPTURE STANDOFF
# =============================================================================

ALT_CAPTURE_STANDOFF_M = 50.0  # Distance behind during climb
ALT_CAPTURE_BEHIND_EXIT_TOL_M = 15.0  # Exit tolerance relative to standoff
# Per-subphase SPDWEIGHT for optimal control
# Test 321 proved that SPDWEIGHT=0.0 disables speed commands entirely, causing overtake
# ArduPilot TECS_SPDWEIGHT: 0.0=pitch for altitude only, 1.0=balanced, 2.0=pitch for speed
# - Backout phase needs balanced (1.0) so speed commands work to open separation
# - Climb phase needs more altitude authority (0.7) so pitch can climb effectively
ALT_CAPTURE_TECS_SPDWEIGHT_BACKOUT = 1.0  # Balanced: speed control for separation
ALT_CAPTURE_TECS_SPDWEIGHT_CLIMB = 0.7  # Altitude bias: more pitch for climb
ALT_CAPTURE_TECS_SPDWEIGHT_NORMAL = 1.0  # TECS weight outside capture (balanced)
ALT_CAPTURE_TECS_SET_INTERVAL_S = 2.0  # Rate limit for set_param calls (sim-seconds)
ALT_CAPTURE_TIMEOUT_S = 90.0  # Timeout if altitude capture fails

# QASSIST during ALTITUDE_CAPTURE
# Problem: TECS cannot climb 890m->900m while station-keeping at 24.7 m/s
# Evidence: Test 320 showed only 0.5m altitude gain in 24 seconds
# Solution: Temporarily raise Q_ASSIST_SPEED above cruise to enable quad motors
# ArduPilot QASSIST: "assistance enabled when airspeed < Q_ASSIST_SPEED"
# Current cruise: 24.7 m/s, default Q_ASSIST_SPEED = 15 → QASSIST disabled
# During capture: set Q_ASSIST_SPEED = 30 → QASSIST enabled (24.7 < 30)
ALT_CAPTURE_QASSIST_SPEED_CAPTURE = 30.0  # Enable QASSIST at cruise (m/s)
ALT_CAPTURE_QASSIST_SPEED_NORMAL = 15.0  # Default from quadplane_formation_mav_clean.parm
ALT_CAPTURE_QASSIST_SET_INTERVAL_S = 2.0  # Rate limit for set_param calls (sim-seconds)

# =============================================================================
# ALTITUDE_CAPTURE - TECS SPEED CONTROL + TWO-PHASE + COLLISION GUARD
# =============================================================================

# Two-phase ALTITUDE_CAPTURE subphase constants
ALT_CAPTURE_SUBPHASE_BACKOUT = "STANDOFF_BACKOUT"  # Phase A: achieve standoff
ALT_CAPTURE_SUBPHASE_CLIMB = "STANDOFF_CLIMB"  # Phase B: climb to altitude

# Standoff backout phase tolerances
ALT_CAPTURE_STANDOFF_TOL_M = 5.0  # Behind error tolerance for standoff achieved
ALT_CAPTURE_STANDOFF_STABLE_S = 2.0  # Must hold standoff for this duration
ALT_CAPTURE_BACKOUT_SPEED_MARGIN = 1.5  # Slow below leader by this during backout (m/s)

# Collision guard constants (safety override)
ALT_CAPTURE_COLLISION_GUARD_M = 20.0  # If behind_m drops below this, force slowdown
ALT_CAPTURE_CLOSURE_WARN_MS = 0.5  # Warn if closure rate exceeds this near leader
ALT_CAPTURE_COLLISION_LOG_INTERVAL_S = 2.0  # Rate limit for collision warnings

# Speed tracking diagnostics
ALT_CAPTURE_SPEED_ERR_WARN_MS = 1.0  # Warn if |IAS - spd_cmd| > this
ALT_CAPTURE_SPEED_ERR_PERSIST_S = 5.0  # After this many seconds of error
ALT_CAPTURE_DIAG_LOG_INTERVAL_S = 3.0  # Diagnostic logging interval

# QASSIST near-target cutoff and overshoot guardrails
# Root cause: QASSIST enabled during STANDOFF_BACKOUT when
# cmd_alt=890m, causing sustained ~1m/s climb that overpowered TECS altitude hold.
# Fix: QASSIST ON only in STANDOFF_CLIMB, with auto-cutoff near target.
ALT_CAPTURE_QASSIST_CUTOFF_BELOW_M = 0.8  # Disable QASSIST when within this of target
ALT_CAPTURE_OVERSHOOT_ABORT_M = 2.0  # Abort ALTITUDE_CAPTURE if exceeds by this
ALT_CAPTURE_OVERSHOOT_PERSIST_S = 1.5  # Must persist this long before abort
ALT_CAPTURE_OVERSHOOT_LOG_INTERVAL_S = 2.0  # Rate limit for overshoot warnings

# =============================================================================
# CLOSURE-CAP BRAKE + NO-PASS SAFETY + LEADER SPEED ESTIMATOR
# =============================================================================
# Cap positive closure rate and convert to a speed cap that's still >= leader
# speed while behind. Only allow below-leader speeds very close to contact.

CLOSURE_CAP_ENABLE_RANGE_M = 10.0  # K-B0: Give TECS 15s tracking window
CLOSURE_CAP_SCHEDULE = [  # (behind_m, max_vc_mps) breakpoints
    (10.0, 0.80),  # K-B1: gentle cap at entry, allows closing
    (8.0, 0.60),  # K-B1: begin tapering
    (5.0, 0.45),  # K-B1: 0.50→0.45 — aligned with schedule
    (3.0, 0.30),  # K-B1: 0.35→0.30 — tighter at critical zone
    (1.5, 0.20),  # K-B1: 0.25→0.20
    (0.5, 0.15),  # At 0.5m: allow 0.15 m/s closure
    (0.0, 0.05),  # At contact: minimal creep
]
CLOSURE_CAP_BRAKE_GAIN = 1.0  # K-B4: Proportional gain for excess closure braking
CLOSURE_CAP_BELOW_LEADER_THRESHOLD_M = 0.3  # Only go below leader inside this
CLOSURE_CAP_LOG_INTERVAL_S = 2.0  # Rate limit for closure cap logging

# --- Hard No-Pass / Breakaway Constants (D2) ---
# DELETED 2026-08-16: OVERTAKE_ABORT_AHEAD_M (-5.0)
# and OVERTAKE_ABORT_PERSIST_S (1.0) removed — ZERO consumers in the live tree,
# established by census (OVGUARD-CENSUS-602). They described an overtake abort
# no code ever read; the only live overtake response is the breakaway guard at
# GUARD_OVERTAKE_PERSIST_S. Detector: test_dead_overtake_constants_removed.py.
# DELETED 2026-08-16: NO_PASS_ABORT_LOG_INTERVAL_S
# (1.0) removed — declaration was the only live-tree reference (full-tree grep
# + the DELDEAD-604 AST scan, tag state 163a9535). It rate-limited abort
# warnings for the overtake abort that DELDEAD-604 removed — nothing ever
# emitted them. Detector: test_dead_overtake_constants_removed.py.

# --- Leader Effective Speed Estimator (B) ---
LEADER_SPEED_EST_MIN_DT_S = 0.2  # Min dt for valid speed estimate
LEADER_SPEED_EST_MAX_DT_S = 5.0  # Max dt for valid speed estimate
LEADER_SPEED_EST_EWMA_ALPHA = 0.2  # EWMA smoothing coefficient
LEADER_SPEED_EST_LOG_INTERVAL_S = 5.0  # Rate limit for speed estimator logging

# --- Closure Computation (C) ---
CLOSURE_RATE_EWMA_ALPHA = 0.3  # EWMA smoothing for closure rate

# Range-based Vc for gate/D-term (bypasses fusion KF velocity bias)
RANGE_VC_WINDOW_S = 2.0  # Sliding window for range differentiation
RANGE_VC_EWMA_ALPHA = 0.3  # Smoothing factor for range-diff Vc

# --- TECS Restore Retry (F) ---
TECS_RESTORE_MAX_RETRIES = 3  # Max retry attempts for TECS restore
TECS_RESTORE_RETRY_DELAY_S = 0.4  # Delay between retries

# --- DIAG324 Rate Limit (G) ---
DIAG324_LOG_INTERVAL_S = 2.0  # Rate limit for comprehensive creep diagnostic

# =============================================================================
# CONTACT_INNER_CAPTURE_GATE (OPTIONAL INNER stability before entry)
# =============================================================================
# Problem: Entering HOLD_CONTACT with cross_err already outside INNER causes
# immediate timer reset. Original solution gated entry on INNER stability.
# Test 310: Made optional - restored two-tier semantics where OUTER is entry gate.
CONTACT_INNER_STABLE_TIME_S = 0.8  # Must be in INNER for this duration
CONTACT_INNER_GATE_LOG_INTERVAL_S = 2.0  # Rate limit gate status logging
# Feature flag: Set True to require INNER stability before HOLD_CONTACT entry.
# Default False restores two-tier behavior (OUTER for entry, INNER for dwell).
REQUIRE_CONTACT_INNER_BEFORE_HOLD = False

# =============================================================================
# LEADER TURN GATING (don't attempt contact during turns)
# =============================================================================
LEAD_YAW_RATE_TURN_FILTER_TAU_S = 1.0  # Low-pass filter time constant
# Test 310: Lowered thresholds to detect typical leader turns
LEAD_YAW_RATE_TURN_ON_DPS = 0.8  # Turn detected if filtered rate > this
LEAD_YAW_RATE_TURN_OFF_DPS = 0.4  # Turn ends if filtered rate < this
TURN_GATE_STRAIGHT_TIME_S = 5.0  # Must be straight this long for DOCKING_CREEP
TURN_GATE_LOG_INTERVAL_S = 2.0  # Rate limit for turn gate logging

# =============================================================================
# FORMATION HEADING GUIDANCE (replaces LOS bearing at close range)
# =============================================================================
# Problem: LOS bearing (atan2(-cross, behind)) amplifies jitter near target.
# Solution: Formation guidance = lead_hdg + PD cross correction + slew limiter.

# Heading slew limiter (real deg/s rate limit, not just clamp on correction)
# This prevents tiny cross noise from jerking commanded heading
HEADING_SLEW_VERY_CLOSE_DPS = 1.0  # Max heading change rate inside 30m
HEADING_SLEW_CLOSE_DPS = 2.5  # Max heading change rate 30-50m
HEADING_SLEW_MEDIUM_DPS = 5.0  # Max heading change rate 50-100m
HEADING_SLEW_FAR_DPS = 25.0  # Max heading change rate >100m

# =============================================================================
# RANGE-DEPENDENT COMMAND RATE LIMITING
# =============================================================================
# Problem: Current DO_CHANGE_SPEED/DO_REPOSITION thresholds too coarse for
# close-range fine control.
# Solution: Tighter thresholds inside 30m, looser outside.

# Very close range (<30m) - fine control
VERY_CLOSE_REPOSITION_INTERVAL_S = 0.12  # Fast heading updates
VERY_CLOSE_HEADING_DELTA_DEG = 0.4  # Small heading change threshold
VERY_CLOSE_SPEED_INTERVAL_S = 0.10  # K4: 0.20→0.10 — double update rate
VERY_CLOSE_SPEED_DELTA_MS = 0.04  # K4: 0.08→0.04 — finer delta threshold

# Close range (30-50m) - transition
CLOSE_REPOSITION_INTERVAL_S = 0.20  # Moderate heading updates
CLOSE_HEADING_DELTA_DEG = 0.8  # Moderate heading threshold
CLOSE_SPEED_INTERVAL_S = 0.30  # Moderate speed updates
CLOSE_SPEED_DELTA_MS = 0.15  # Small speed threshold

# Hold durations (seconds) - Test 310: Reduced for faster docking cycles
PRECONTACT_HOLD_DURATION_S = 5.0  # Hold at precontact for 5s
CONTACT_HOLD_DURATION_S = 5.0  # Hold at contact for 5s (Test 311)

# Control parameters
CLOSURE_RATE_CREEP_MS = 1.0  # Creep closure rate (m/s) - user specified

# Lateral control gains (Test 301 tuning)
# - P-gain reduced 50% from 0.35 to prevent oscillation overshoot
# - D-gain added to dampen oscillations (cross_rate feedback)
CROSS_CORRECTION_GAIN = 0.175  # Lateral P-gain (reduced from 0.35 per Test 300 analysis)
CROSS_D_GAIN = 0.5  # Lateral D-gain for oscillation damping
MAX_HEADING_RATE_DPS = 25.0  # Maximum heading rate (deg/s) at long range
CLOSE_RANGE_HEADING_RATE_DPS = 5.0  # Maximum heading rate at close range (<100m)
CLOSE_RANGE_THRESHOLD_M = 100.0  # Range threshold for reduced heading rate

# =============================================================================
# AIRSPEED CONTROL - TAPERED APPROACH LAW
# =============================================================================
# Leader reference speed (TAS at leader altitude)
LEADER_TARGET_TAS_MS = 24.7  # Leader's target TAS (m/s) - from mission profile
DENSITY_CORRECTION_ENABLE = (
    False  # SITL treats DO_CHANGE_SPEED as TAS; EAS correction causes 1x approach stall
)

# Tapered closure control (replaces constant margin)
# - Far: allow faster closure; Near: gentle approach
CLOSURE_TAPER_FAR_RANGE_M = 150.0  # Range beyond which max closure applies
CLOSURE_TAPER_NEAR_RANGE_M = 20.0  # Range at which closure should be minimal
MAX_CLOSURE_RATE_FAR_MS = 5.0  # Max closure when far (m/s)
# Increased from 0.5 to 1.0 m/s to prevent 30m stall
MAX_CLOSURE_RATE_NEAR_MS = 1.0  # Max closure when near (m/s) - doubled from 0.5

# Mid-range closure boost (fix 50m stall with linear taper)
# Problem: Linear 20-150m taper gave only 1.2 m/s at 40m behind_error - insufficient
# Solution: Add mid-range bracket with more aggressive closure rate
CLOSURE_TAPER_MID_RANGE_M = 50.0  # Mid-range threshold for 3-segment taper
# Increased from 2.5 to 3.0 m/s for more aggressive mid-range closure
MAX_CLOSURE_RATE_MID_MS = 3.0  # Max closure at mid-range (50m behind_error)

# Speed command slew rate limiting (prevents TECS excitation)
# M3 S3: Reduced from 2.0→1.5 to reduce TECS excitation from speed command discontinuities
SPEED_CMD_SLEW_RATE_MS_PER_S = (
    1.5  # Max speed change per second (m/s/s) (P1: was 1.0, M3: was 2.0)
)

# Overtaking protection: DELETED 2026-08-16.
# OVERTAKE_PROTECTION_MARGIN_M (3.0) and OVERTAKE_SPEED_REDUCTION_MS (1.0) had
# ZERO consumers in the live tree — nothing ever implemented this "slow down
# hard when closer than target" protection. Detector:
# test_dead_overtake_constants_removed.py.

# Command rate limiting (reduce MAVLink spam)
DO_CHANGE_SPEED_MIN_INTERVAL_S = 0.5  # Min interval between speed commands
DO_CHANGE_SPEED_DELTA_THRESHOLD = 0.3  # Send if speed changes by more than this
DO_REPOSITION_MIN_INTERVAL_S = 0.2  # Min interval between reposition commands

# =============================================================================
# ALTITUDE COMMAND CONTROL + WATCHDOG
# =============================================================================
# Problem: Test 311 showed altitude command was sent but not tracked by TECS.
# QuadPlane stayed at 910m instead of commanded 890m, causing below_err ~ -20m.
# Solution: Make altitude command robust with forced sends and fallback commands.

# Altitude command delta - force DO_REPOSITION if altitude cmd changes by this much
ALT_CMD_DELTA_M = 0.5  # Force resend if alt_cmd changes by > 0.5m

# Altitude error forcing - force DO_REPOSITION if altitude error persists
ALT_ERR_FORCE_M = 5.0  # Force resend if |fol_alt - alt_cmd| > 5m
ALT_ERR_FORCE_TIME_S = 2.0  # For more than 2 seconds

# Altitude error warning - alert if altitude tracking fails
ALT_ERR_WARNING_M = 10.0  # Warn if |fol_alt - alt_cmd| > 10m
ALT_ERR_WARNING_TIME_S = 5.0  # For more than 5 seconds
ALT_ERR_WARNING_LOG_INTERVAL_S = 5.0  # Rate limit warning logs

# Fallback altitude command rate (MAV_CMD_GUIDED_CHANGE_ALTITUDE)
ALT_FALLBACK_INTERVAL_S = 2.5  # Send fallback altitude cmd every 2.5s

# ALT_FORCE log rate limiting (reduce log spam)
ALT_FORCE_LOG_INTERVAL_S = 2.0  # Only log ALT_FORCE every 2 seconds

# Vertical gating - prevent REJOIN_PRECONTACT from stalling
ALT_GATE_THRESHOLD_M = 6.0  # Gate if |below_err| > 6m
ALT_GATE_TIME_S = 10.0  # For more than 10s
ALT_GATE_RANGE_M = 150.0  # Only gate if range < 150m
ALT_GATE_STANDOFF_M = 50.0  # Add 50m to behind target when gated

# High-side descent ramp parameters (P1: graduated descent to avoid ALT_GATE deadlock)
HIGH_SIDE_RAMP_START_M = 150.0  # Begin descent at 150m behind (just before ALT_GATE range)
HIGH_SIDE_RAMP_END_M = 80.0  # Complete descent by 80m behind (well above 60m cutoff)

# STERN_CONVERSION minimum dwell (bleed off energy before REJOIN)
STERN_CONVERSION_MIN_DWELL_S = 5.0  # Minimum time in STERN_CONVERSION
STERN_CONVERSION_STABILITY_CROSS_M = 15.0  # Max cross-track to consider stable

# =============================================================================
# STERN_CONVERSION HANDOFF FIXES
# =============================================================================
# Problem: Test 304 got stuck in STERN_CONVERSION because the range window
# (180-220m) was too narrow - once range dropped below 180m, transition blocked.
# Solution: One-sided gate + forced handoff at very close range + timeout.

# One-sided range gate: transition allowed if range <= 200m + this margin
STERN_HANDOFF_MAX_OVER_M = 30.0  # Allow transition if range <= 230m

# Force handoff if we get very close (can't wait forever for cross to settle)
FORCE_REJOIN_RANGE_M = 150.0  # Force transition below this range
FORCE_REJOIN_CROSS_LIMIT_M = 25.0  # Allow forced handoff with this much cross

# Timeout: don't stay in STERN_CONVERSION forever
STERN_CONVERSION_MAX_TIME_S = 60.0  # Force transition after this time

# Blocked gate diagnostic logging rate limit
BLOCKED_GATE_LOG_INTERVAL_S = 3.0  # Log blocked gates every N seconds

# Overtake protection is enabled/disabled via enable_overtake_protect parameter.

# Overtake protection log rate limit (prevent 20Hz spam)
OVERTAKE_LOG_INTERVAL_S = 5.0  # Max 1 log per this many seconds

# =============================================================================
# SPEED COMMAND PIPELINE FIXES
# =============================================================================
# Problem: Test 305 showed OVERTAKE_PROTECT logging but speed never changed.
# Root cause: Deadband (0.1 m/s) > per-tick slew delta (0.05 m/s at 20Hz).
# Solution: Remove deadband-after-slew, use emergency slew for OVERTAKE_PROTECT.

# Emergency slew rate for OVERTAKE_PROTECT (much faster than normal 1.0 m/s/s)
OVERTAKE_SLEW_RATE_MS_PER_S = 5.0  # 5x faster decel when overshooting

# Force speed send threshold - if target changes by this much, bypass min-interval
FORCE_SPEED_SEND_DELTA_MS = 1.0  # Force send if target drops >1 m/s

# Precontact VC gate - speedup-invariant (v9.2 Patch A)
# Uses hysteresis + exit grace + dt-capped credit accumulation to eliminate
# sample-rate sensitivity. Three bands:
#   PASS (vc <= PASS_MAX): accumulate credit
#   HYS (PASS_MAX < vc < RESET_MAX): hold credit, clear violation
#   RESET (vc >= RESET_MAX): start violation timer, decrement after grace
PRECONTACT_VC_PASS_MAX_MS = 0.50  # Pass condition (accumulate credit)
PRECONTACT_VC_RESET_MAX_MS = 0.70  # Hysteresis reset band (> pass)
PRECONTACT_VC_STABLE_REQ_S = 1.50  # Required stable credit (sim seconds)
PRECONTACT_VC_EXIT_GRACE_S = 0.25  # Sustained violation before decrement
PRECONTACT_VC_FILTER_TAU_S = 3.00  # EWMA tau for slow behind derivative (s)

# P7.3 v9.7: VC gate decay system (replaces hard reset with decay)
# Enables 5x to accumulate credit by using time-based grace + gradual decay
# instead of immediate credit decrement on sustained RESET band.
VC_GATE_DECAY_RATE_PER_S = 0.5  # Credit decay rate in RESET band (credit/s)
VC_GATE_RESET_GRACE_S = 0.25  # Time in RESET band before decay starts (sim-s)

# P7.4 v9.7: dt spike handling (stabilizes integrators at high speedup)
# Prevents dt spikes from destabilizing closure deficit integrator and
# stuck integral boost. Safety down-slew uses full dt (must bind immediately).
DT_SPIKE_THRESHOLD_S = 0.20  # dt >= this is a spike (normal is ~0.05s at 20Hz)
DT_SPIKE_INTEGRATOR_FREEZE = True  # Freeze integrators on spike tick

# P7.2 v9.7: Vertical coupling (replaces binary climb clamp)
# Ramps below_target from PRECONTACT (10m) to CONTACT (0m) as behind_m decreases.
# Holds range (vc_cap=0) when vertical error is huge, proportional gate otherwise.
VERT_RAMP_START_BEHIND_M = 12.0  # Begin ramping below_target at this behind_m
VERT_RAMP_END_BEHIND_M = 2.0  # Ramp complete (target=0) at this behind_m
VERT_ERR_HOLD_RANGE_M = 3.0  # Hold range (vc_cap=0) if |below_err| > this
VERT_ERR_GATE_SCALE_M = 1.5  # Proportional gate scaling when err < HOLD threshold

# Coupled below-target ramp (DOCKING_CREEP altitude accommodation)
COUPLED_BELOW_TARGET_ENABLE = True  # Master enable
COUPLED_BELOW_TARGET_ENABLE_STATES = ["DOCKING_CREEP"]  # Conservative: DC only
COUPLED_BELOW_TARGET_RATE_LIMIT_MPS = 0.40  # Clamp d(below_target)/dt
COUPLED_BELOW_SIGN_MODE = "AUTO_DETECT"  # Runtime sign safety check

# P7.1 v9.7: Final approach schedule (tapering closure for contact)
# Piecewise linear schedule for closure rate based on behind distance.
# Used in DOCKING_CREEP as the single source of truth for closure limiting.
# M3 S3: Reduced top 3 entries to limit TECS excitation; added (0.6, 0.04) for finer M3a resolution
FINAL_APPROACH_SCHEDULE = [
    (8.0, 0.40),  # K1: 0.60→0.40 — reduce entry momentum
    (5.0, 0.25),  # K1: 0.35→0.25 — start decelerating earlier
    (3.0, 0.18),  # N-R revert: 0.14→0.18
    (2.0, 0.15),  # N-R revert: 0.10→0.15
    (1.0, 0.12),  # N-R revert: 0.08→0.12
    (0.7, 0.08),  # N-R revert: 0.06→0.08
    (0.6, 0.04),  # 0.6-0.7m: unchanged
    (0.3, 0.04),  # N-R revert: 0.03→0.04
    (0.0, 0.03),  # N-R revert: 0.02→0.03
]
M3A_DWELL_GRACE_S = 0.25  # Grace before resetting M3a dwell timer
M3B_DWELL_GRACE_S = 0.25  # Grace before resetting M3b dwell timer
# #460 Candidate B (R7.1 metric-side, D1-authorized): M3b dwell bounded-EXCURSION tolerance.
# Default-OFF => the duration-only grace above is unchanged (byte-identical). When ON, the M3b
# grace survives an out-of-corridor excursion iff it is BOTH shorter than D AND stays under M
# |behind| (the magnitude gate M is new — the current grace has none). Adds dwell margin without
# any control-command change (observer/metric only). Cap M <= 1.5 (M >= 2.0 false-certifies the
# p95 departure tail). Sizing per ALONGTRACK_MARGIN_R7_REAUTH: D=0.5 s, M=1.0 m.
FD_M3B_EXCURSION_TOL_ENABLE = (
    os.environ.get("FD_M3B_EXCURSION_TOL_ENABLE", "0").strip() == "1")
M3B_DWELL_EXCURSION_D_S = 0.5   # tolerated excursion duration when armed
M3B_DWELL_EXCURSION_MAG_M = 1.0  # tolerated |behind| excursion magnitude (cap <= 1.5)

# Docking waypoint sequencing (DO NOT REMOVE)
DOCKING_HOLD_BEHIND_M = 8.0  # Phase-1 trail offset (hold station)
DOCKING_TRAIL_RAMP_RATE_MPS = 0.3  # Phase-2 trail ramp rate toward contact

# =============================================================================
# DIAGNOSTIC OVERLAYS (SITL experiments only)
# =============================================================================
# In production (no env vars set), all are False → full guard stack is active.
# Override via env var: FLIGHTDOCK_DIAG_BREAKAWAY_SHADOW=1, etc.
DIAG_BREAKAWAY_SHADOW = (
    os.environ.get("FLIGHTDOCK_DIAG_BREAKAWAY_SHADOW", "") == "1"
)  # Global: block ALL breakaway transitions
DIAG_GEOMETRY_TERMINATOR = (
    os.environ.get("FLIGHTDOCK_DIAG_GEOM_TERM", "0") == "1"
)  # Hard-stop on invalid geometry (default OFF — CZ-EMERGENCY provides safety floor)


# =============================================================================
# P8.3 v9.8: NEAR-PRECONTACT CLOSURE CAP
# =============================================================================
# Problem: At 5× speedup, vc_slow stays at 1.5-2.0 m/s in REJOIN_PRECONTACT,
# causing the VC gate to never pass (requires vc < 0.5 m/s for 1.5s).
# Solution: When close to M2 envelope (behind in [8, 15]m AND cross/below
# in-family), cap closure rate to allow VC gate accumulation.
NEAR_PRECONTACT_ENABLE_BEHIND_MAX_M = 15.0  # Enable cap when behind < this
NEAR_PRECONTACT_ENABLE_BEHIND_MIN_M = 8.0  # Enable cap when behind > this
NEAR_PRECONTACT_ENABLE_CROSS_TOL_M = 5.0  # Enable cap when |cross| < this
NEAR_PRECONTACT_ENABLE_BELOW_TOL_M = 5.0  # Enable cap when below in tolerance
NEAR_PRECONTACT_VC_CAP_MPS = 0.4  # Max closure rate when cap active

# Station-keeping speed gain for HOLD_PRECONTACT (softer than rejoin)
HOLD_STATION_KEEPING_GAIN = 0.12  # Speed correction per m behind error

# ISA atmospheric model constants (for TAS -> EAS conversion)
# Used by compute_isa_density_ratio() and tas_to_eas() functions
ISA_TEMPERATURE_LAPSE_RATE = 0.0065  # K/m
ISA_SEA_LEVEL_TEMP = 288.15  # K

# =============================================================================
# CLOSE-RANGE FINE CONTROL (< 30m)
# =============================================================================
# Problem: Test 303 showed min range 12.7m but couldn't close to 10m target.
# Root cause: Kp=0.05 only generates 0.5 m/s at 10m error - insufficient closure.
# Solution: Range-scaled gains - higher Kp at close range for tighter control.

# Range thresholds for close-range mode
VERY_CLOSE_RANGE_M = 30.0  # Below this, use fine control mode
FINE_CONTROL_RANGE_M = 50.0  # Transition zone for gain scaling

# Speed control: Range-scaled Kp
# - Far (>50m): Kp = 0.05 (conservative, prevents overshoot)
# - Near (<30m): Kp = 0.15 (aggressive, ensures closure to target)
# Increased P-gains to prevent 30m stall
SPEED_KP_FAR = 0.08  # Proportional gain at far range
SPEED_KP_NEAR = 0.30  # Proportional gain at close range (M3b: 0.20→0.30 for dwell stability)
SPEED_KD = 0.3  # Derivative gain (damping) - unchanged

# =============================================================================
# STUCK RECOVERY (integral boost when Vc near zero despite positive behind_error)
# =============================================================================
# Stuck detection: If position-derivative Vc is near zero despite positive
# behind_error, apply integral boost to break out of stall.
STUCK_VC_THRESHOLD_MS = 0.15  # Consider "stuck" if |actual_vc| < 0.15 m/s
STUCK_BEHIND_ERR_MIN_M = 5.0  # Only trigger stuck recovery if behind_err > 5m
STUCK_TIME_THRESHOLD_S = 10.0  # Must be stuck for 10s before recovery kicks in
STUCK_INTEGRAL_GAIN = 0.02  # Integral gain: adds 0.02 m/s per second stuck
STUCK_INTEGRAL_MAX_MS = 2.0  # Max integral boost (m/s)
STUCK_INTEGRAL_DECAY_RATE = 0.1  # Decay rate when not stuck (m/s per second)
STUCK_RECOVERY_LOG_INTERVAL_S = 5.0  # Rate limit recovery logs

# =============================================================================
# v9.2 Patch D: ANTI-STALL CLOSURE DEFICIT INTEGRATOR
# =============================================================================
# Replaces P2 close-range closure integrator. Integrates deficit between
# desired closure (from schedule) and measured closure. More conservative
# gain with explicit safety: resets near target, on S1 risk, and decays
# when inactive. Prevents range plateau at 10-15m without overshoot.
CLOSURE_DEFICIT_KI = 0.05  # Integral gain (1/s)
CLOSURE_DEFICIT_I_MAX = 0.15  # Max integrator output (m/s) (RC-3: halved to limit windup at 3x/5x)
CLOSURE_DEFICIT_MIN_BEHIND_M = 0.5  # Only integrate when behind > this
CLOSURE_DEFICIT_MIN_ERROR_MS = 0.1  # Only integrate when deficit > this
CLOSURE_DEFICIT_DECAY_TAU_S = 2.0  # Decay time constant when inactive

# Leader speed reference selection
# When the estimated leader speed differs significantly from LEADER_TARGET_TAS_MS,
# use the estimate to avoid commanding insufficient closure
LEADER_SPEED_EST_THRESHOLD_MS = 1.0  # Use estimate if |est - target| > 1 m/s

# Fine heading control at very close range (<30m)
VERY_CLOSE_HEADING_RATE_DPS = 1.0  # Max heading rate inside 30m
FINE_CONTROL_HEADING_RATE_DPS = 2.5  # Max heading rate in transition zone (30-50m)

# Lateral control: Range-scaled D-gain
# At close range, increase D-gain to dampen oscillations more aggressively
CROSS_D_GAIN_NEAR = 0.8  # D-gain at close range (vs 0.5 at far range)

# Geometry gates for state transitions
STERN_ASPECT_TOLERANCE_DEG = 15.0  # Acceptable aspect angle for stern

# Fighter intercept geometry (stern conversion)
LATERAL_OFFSET_M = 800.0  # Lateral offset during head-on approach (m)
LEAD_TURN_TA_DEG = 150.0  # Target aspect for lead turn (deg)
LEAD_TURN_SPEED_MS = 30.0  # Airspeed during lead turn (m/s)
OFFSET_SIDE = 1.0  # +1 = offset to right of target, -1 = left
LEAD_TURN_MIN_RANGE_M = 2000.0  # Min range before lead turn (m)

# Dynamic offset side selection
# If QuadPlane starts on opposite side from OFFSET_SIDE, flip to avoid crossing
DYNAMIC_OFFSET_THRESHOLD_M = 500.0  # Only flip if |cross_m| > this threshold
STERN_CONV_SPEED_BOOST_RANGE_M = 200.0  # Range error threshold for speed boost (m)

# Weather check parameters
WEATHER_PARAMS = ["SIM_WIND_SPD", "SIM_WIND_DIR", "SIM_WIND_TURB"]

# FW Transition parameters (robust transition from VTOL to Fixed-Wing)
FW_TRANSITION_MIN_AIRSPEED_MPS = 12.5  # Achievable at high speedup
FW_TRANSITION_SUSTAIN_S = 3.0  # Sustain airspeed for this long
FW_TRANSITION_TIMEOUT_S = 60.0  # Timeout per attempt
FW_TRANSITION_RETRIES = 3  # Number of retry attempts
FW_STABILIZE_TIME_S = 5.0  # Time to stabilize after FW confirmed

# Pre-flight validation parameters
PREFLIGHT_LEAD_TIMEOUT_S = 10.0  # Wait for leader telemetry
PREFLIGHT_MIN_LEAD_MESSAGES = 5  # Minimum leader messages to validate

# Message rate configuration (higher rates for better control)
# ---------------------------------------------------------------------------
# #487 (2026-07-20): ATTITUDE rate made env-gated for the #488 rate-ladder RCA.
#   WHY: the SimTimeAuthority's "now" is ZOH-held from the last ATTITUDE
#   (sim_time_authority.py:262); now_s() does NOT interpolate (352-359). Its
#   step was MEASURED at 50.0 ms (~97% of steps, 14 legs) and 50 ms x 24.7 m/s
#   = 1.235 m == the two-level behind-bias quantum measured in #485, in BOTH
#   arms with opposite sign. L_add was REFUTED as the source (#486 Addendum A1:
#   doubling L_add to 100 ms moved the age floor 50->100 but left the bias level
#   at +1.25 m). This constant is the ONLY origin of the 20 Hz request
#   (baseline_formation_controller.py:3277/3280 -- the timebase_canary path is
#   absent from the tree, so the fallback is the live path).
#   SET_MESSAGE_INTERVAL is per-link, so an external re-request cannot reach the
#   controller's own link; the config constant is the only clean knob.
#   UNSET => 20.0 => functionally identical to every prior campaign.
#   Out-of-range / unparseable fails SAFE to 20.0, never to a silent 0 (which
#   would DISABLE the stream and stall the time authority).
FD_ATTITUDE_RATE_MIN_HZ = 5.0
FD_ATTITUDE_RATE_MAX_HZ = 50.0


def _resolve_attitude_rate_hz(raw=None):
    """FD_ATTITUDE_RATE_HZ -> float in [5,50]; anything else -> 20.0 (safe)."""
    if raw is None:
        raw = os.environ.get("FD_ATTITUDE_RATE_HZ", "20.0")
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return 20.0
    if not (FD_ATTITUDE_RATE_MIN_HZ <= v <= FD_ATTITUDE_RATE_MAX_HZ):
        return 20.0
    return v


MSG_RATE_ATTITUDE_HZ = _resolve_attitude_rate_hz()  # ATTITUDE message rate (#487 gated)
MSG_RATE_VFR_HUD_HZ = 10.0  # VFR_HUD message rate
MSG_RATE_POSITION_HZ = 10.0  # GLOBAL_POSITION_INT rate
