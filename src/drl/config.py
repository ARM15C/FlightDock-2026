"""FlightDock DRL Configuration — single source of truth for all hyperparameters.

Observation space, action space, reward RRL, PPO/TD3/SAC hyperparameters,
curriculum stages, and process management. Import safety bounds from the
baseline formation_config to avoid duplication.
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Dict, List

# --- Baseline imports (via __init__.py sys.path setup) ---
import formation_config as _fc

# =============================================================================
# [FINAL SETUP: APPLIES]
# OBSERVATION SPACE
# =============================================================================
OBS_DIM = 42
              # (sensor_mode_id, uwb_valid), 31 (dt_s), 34 (speedup_factor), 39-40 (wind channels, structurally
              # disconnected — fork (a)).
OBS_DIM_LEGACY = 39
# D6.1 named-feature index map of record (42-D). The schema hash derives from this tuple, so any
# reordering/resize changes the hash and trips consumers LOUDLY at import — never a silent mis-
# index.
OBS_FEATURES_V42: tuple = (
    "behind_m", "cross_m", "below_m", "range_m",
    "closure_rate_ms", "vel_cross_m_s", "vel_below_m_s",
    "behind_error_m", "cross_error_m", "below_error_m",
    "follower_airspeed_ms", "follower_alt_msl_m", "follower_heading_deg",
    "follower_pitch_deg", "follower_roll_deg", "follower_vd_ms",
    "leader_airspeed_ms", "leader_heading_deg", "leader_alt_msl_m",
    "leader_vd_ms", "leader_yaw_rate_dps",
    "heading_error_deg", "bearing_to_leader_deg", "target_aspect_deg",
    "speed_delta_ms", "max_closure_ms",
    "phase_id", "in_inner_envelope", "in_outer_envelope",
    "continuous_contact_m3b_dwell_s", "milestone_tier",
    "prev_speed_residual", "prev_heading_residual", "prev_altitude_residual",
    "wake_hazard_intensity",
    "ttc_front_boundary_s", "closure_rate_delta_ms2",
    "aft_boundary_distance", "behind_drift_rate", "aft_boundary_ttc",
    "longitudinal_accel", "follower_throttle_pct",
)
import hashlib as _hashlib  # noqa: E402
OBS_SCHEMA_V42_SHA256: str = _hashlib.sha256(
    "|".join(OBS_FEATURES_V42).encode()).hexdigest()

OBS_NORMS: List[float] = [
    # 0-3: Relative position
    50.0,  # behind_m
    10.0,  # cross_m
    50.0,  # below_m
    100.0,  # range_m
    # 4-6: Relative velocity
    5.0,  # closure_rate_ms
    3.0,  # vel_cross_m_s
    3.0,  # vel_below_m_s
    # 7-9: Envelope errors
    10.0,  # behind_error_m
    5.0,  # cross_error_m
    10.0,  # below_error_m
    # 10-15: Follower state
    31.0,  # follower_airspeed_ms
    500.0,  # follower_alt_msl_m
    180.0,  # follower_heading_deg
    30.0,  # follower_pitch_deg
    45.0,  # follower_roll_deg
    5.0,  # follower_vd_ms
    # 16-20: Leader state
    31.0,  # leader_airspeed_ms
    180.0,  # leader_heading_deg
    500.0,  # leader_alt_msl_m
    5.0,  # leader_vd_ms
    10.0,  # leader_yaw_rate_dps (LP-filtered)
    # 21-23: Heading geometry
    180.0,  # heading_error_deg
    # bearing_to_leader_deg (saturates >198° by design; leader is in forward hemisphere during DRL
    # phase)
    180.0,
    180.0,  # target_aspect_deg
    # 24-25: Speed & closure context
    5.0,  # speed_delta_ms
    5.0,  # max_closure_ms
    # 26-28: Phase & envelope flags
    3.0,  # phase_id
    1.0,  # in_inner_envelope (binary)
    1.0,  # in_outer_envelope (binary)
    3.0,  # continuous_contact_m3b_dwell_s (C3h: was 30.0 for accumulated)
    4.0,  # current_milestone_tier
    1.0,  # prev_speed_residual_applied
    3.0,  # prev_heading_residual_applied
    1.0,  # prev_altitude_residual_applied
    # 34: Wake turbulence hazard
    1.0,  # wake_hazard_intensity (already 0.0–1.0, norm=1.0)
    # 35-36: C3i anticipation features (raw clamped values, OBS_NORMS handles normalization)
    5.0,  # ttc_front_boundary_s (time-to-front-M3b-overshoot, clamped [-5, 5])
    5.0,  # closure_rate_delta_ms2 (closure accel/decel, clamped [-5, 5])
    1.0,
    5.0,
    10.0,  # aft_boundary_ttc (sentinel 10.0 when drifting away)
    # 40-41: C3o leading indicators for double-integrator anticipatory control
    5.0,  # longitudinal_accel (d(airspeed)/dt, m/s², clamped [-5, 5])
    100.0,  # follower_throttle_pct (0-100%, actuator state of thrust axis)
]

assert len(OBS_NORMS) == OBS_DIM, f"OBS_NORMS length {len(OBS_NORMS)} != OBS_DIM {OBS_DIM}"

OBS_CLIP = 1.1  # Clip normalised obs to ±1.1

# =============================================================================
# [FINAL SETUP: APPLIES]
# ACTION SPACE
# =============================================================================
ACTION_DIM = 3
# Action order: [speed (m/s), heading (deg), altitude (m)] (actuator mode). Heading residual is
# bounded by CLAMP_HEADING_NEGATIVE_DEG / CLAMP_HEADING_POSITIVE_DEG below.
ACTION_SCALES: List[float] = [1.0, 1.0, 1.0]  # speed m/s, heading deg, altitude m

# Reference-level position offsets (RFL mode); cross-axis restored alongside ACTION_DIM=3. Action
# order under reference mode: [behind (m), cross (m), below (m)].
ACTION_SCALES_REFERENCE: List[float] = [1.0, 1.0, 1.0]  # behind m, cross m, below m
RESIDUAL_MODE: str = "actuator"  # "actuator" (C3o default) | "reference" (C3r RFL)
RFL_ACTIVATION_TIER: int = 3  # milestone tier gate: offsets active only at M3a_fine+
FORCE_ZERO_RESIDUAL: bool = False
INJECT_ZERO_ACTION: bool = False  # C3o identity proof: inject [0,0,0] through normal path

# =============================================================================
# C3o: COORDINATED INTERVENTION STACK
# =============================================================================
# [FINAL SETUP: APPLIES (barrier reward is live inside the M3b box, weight 0.30)]
# V1: CBF-style barrier reward
W_BARRIER: float = 0.30  # C3p: recalibrated (0.1 → 0.30)
W_BARRIER_OUTSIDE: float = 0.0
CBF_VIOLATION_SCALE: float = 2.0  # penalty multiplier for boundary violations
BARRIER_PROXIMITY_GATE: float = 1.5  # gate at 1.5x boundary in error space
BARRIER_VELOCITY_EMA: float = 0.3  # velocity filter alpha (reduce finite-diff noise)
# [FINAL SETUP: NOT USED, forward bias parked at 0.0 (disabled)]
# V3: Forward bias offset
FORWARD_BIAS_MS: float = 0.0          # C3o-B: 0.5 m/s (0.0 = disabled)
BIAS_SMOOTH_ALPHA: float = 0.15  # EMA alpha for C1-continuous bias activation
# [FINAL SETUP: NOT USED, speed EMA parked at 1.0 (passthrough) and slew at 100 (unlimited)]
# V4: Speed conditioning on policy delta
EMA_ALPHA_SPEED: float = 1.0
SPEED_SLEW_RATE_MPS2: float = 100.0
# [FINAL SETUP: APPLIES (speed and heading residual clamps)]
# C3X Intervention 2: asymmetric behind-axis (speed) clamp at Site (ii) in residual_action.py.
# Constrains the policy speed delta to [-1.0, +0.10] m/s before integration with the baseline PID
# setpoint.
CLAMP_BEHIND_NEGATIVE_MS: float = -1.0
CLAMP_BEHIND_POSITIVE_MS: float = 0.10
# Symmetric ±3 deg conservative bound on the per-tick heading residual amplitude. The downstream
# baseline rate-limiter (HEADING_RATE_*_DPS) governs achieved-rate and is the harder gate; this
# clamp constrains the instantaneous policy delta added to baseline.heading_deg.
CLAMP_HEADING_NEGATIVE_DEG: float = -3.0
CLAMP_HEADING_POSITIVE_DEG: float = 3.0
# [FINAL SETUP: NOT USED, CAPS lambda parked at 0.0]
# V2: CAPS temporal smoothness (deferred)
CAPS_ACTOR_LAMBDA: float = 0.0  # Global default aligned with CLI (--caps-actor-lambda default=0.0).

# Safety bounds (imported from baseline)
SAFE_FLOOR_MS: float = _fc.SAFE_FLOOR_MS  # 19.0
SAFE_MAX_MS: float = _fc.SAFE_MAX_MS  # 31.0
# ±20 m from leader altitude (precontact is ~14.31m below)
ALT_OFFSET_MAX_M: float = 20.0

# [FINAL SETUP: APPLIES]
# Belly-collision PRE-TRAINING safety clamp (#390 C1): asymmetric, phase-gated climb-RATE limiter
# on the residual altitude COMMAND in residual_action.py. FD_BELLY_CLAMP_ENABLE=0 is the explicit
# opt-OUT (validation reruns of pre-flip corpora only — NEVER for training or benchmark legs).
BELLY_CLAMP_ENABLE: bool = _os.environ.get("FD_BELLY_CLAMP_ENABLE", "1") == "1"
BELLY_CLAMP_VZ_MS: float = 0.75
BELLY_CLAMP_RANGE_M: float = 5.0
# #452 (#390 Build-A): the AUTHORITATIVE clamp is at the env command chokepoint
# (env_formation_sitl._apply_belly_clamp at both _send_commands sites), gated on these docking-
# phase FSM states — the ONLY locus that sees the realized altitude command in BOTH residual modes
# (reference-mode offset is applied inside the frozen controller and is invisible to
# residual_action). The FSM-state gate SUPERSEDES the BELLY_CLAMP_RANGE_M proxy for the env clamp.
BELLY_CLAMP_STATES: frozenset = frozenset(
    {"ALT_CAP_HOLD_RANGE", "DOCKING_CREEP", "HOLD_CONTACT"})
# Injected at the env chokepoint because the residual action channel measurably does NOT drive the
# realized altitude command in docking states (BELLYMARGIN_RESULT_R1). Default-unset (0) => byte-
# identical no-op.
FAULTINJ_VZ_MS: float = float(_os.environ.get("FD_FAULTINJ_VZ", "0") or 0.0)
FAULTINJ_CAP_M: float = 25.0
# EMERGENCY_BREAKAWAY alone must NOT terminate — breakaway->direct- rejoin cycles recover
# routinely; INTERCEPT entry = formation lost. The latch arms on the first ARM_STATES sighting
# after reset so warmup/startup residue can never fire it.
FORMATION_LOST_TERM_ENABLE: bool = (
    _os.environ.get("FD_FORMATION_LOST_TERM", "0") == "1")
FORMATION_LOST_STATES: tuple = ("INTERCEPT",)
FORMATION_LOST_ARM_STATES: tuple = (
    "REJOIN_PRECONTACT", "HOLD_PRECONTACT", "ALT_CAP_HOLD_RANGE",
    "ALTITUDE_CAPTURE", "DOCKING_CREEP", "HOLD_CONTACT")

# Heading rate-limiting (range-dependent)
HEADING_RATE_CLOSE_DPS: float = 1.0  # Max deg/s when range < threshold
HEADING_RATE_FAR_DPS: float = 5.0  # Max deg/s when range >= threshold
HEADING_RATE_RANGE_THRESHOLD_M: float = 30.0

# =============================================================================
# [FINAL SETUP: APPLIES]
# REWARD — RRL
# =============================================================================
# Proximity anisotropic scaling (M3b envelope normalisers)
PROX_BEHIND_SCALE: float = 0.7
PROX_CROSS_SCALE: float = 0.5
PROX_BELOW_SCALE: float = 0.7

R_PROX_M2_DECAY: float = 0.2  # Multiplier for r_prox when inside M2 range
R_PROX_M2_RANGE_THRESHOLD_M: float = 15.0  # Boom-tip range threshold in meters

# =========================================================================
# [FINAL SETUP: APPLIES]
# Campaign 3: Zone-based reward modulation
# =========================================================================
# P0-A: State-dependent R_reference penalty
# [TUNING-START] W_REF_INNER=5.0 from residual minimization framework.
# Actual optimal value depends on FlightDock reward magnitude scale.
ZONE_BASED_WREF_ENABLED: bool = True
W_REF_INNER: float = 5.0
W_REF_OUTER: float = 0.1
# [TUNING-START] Center at 0.7 m = M3b tight contact envelope boundary. Steepness k=15 gives ~95%
# transition within ±0.2 m of boundary.
W_REF_SIGMOID_CENTER_M: float = 0.7
W_REF_SIGMOID_STEEPNESS: float = 15.0

# Changes proximity-to-dwell ratio from ~4.4:1 to ~0.44:1 inside the docking envelope.
R_PROX_ENVELOPE_SUPPRESSION: float = 0.1

# [TUNING-START] 0.1 as starting point for FlightDock action scale.
W_ACTION_RATE_DEFAULT: float = 0.1

# P3-CRRL: PID-delta proportional residual clipping [LITERATURE] Staessens et al. (2023).
CRRL_BETA: float = 0.3
CRRL_FLOOR: float = 0.01  # Minimum clip bound per axis (prevents zero-clip)
CRRL_ENABLED: bool = True

# Law: A_i(s) = min(A_static_i, beta_i*|du_PID,i(s)| + 1_terminal(s)*A_min_i), per-sign A-/A+.
# Default OFF => the legacy symmetric CRRL path above is byte-identical.
CRRL_BPC_ENABLED: bool = _os.environ.get("FD_CRRL_BPC", "1") == "1"
CRRL_BPC_TERMINAL_STATES: frozenset = frozenset({"DOCKING_CREEP", "HOLD_CONTACT"})
# Order = action layout [speed, heading, alt]; units = u_pid_delta units (m/s, deg, m).
CRRL_BPC_FLOOR_POS: tuple = (0.0, 0.0, 0.0)  # P-6 E3: FLOOR-ZERO era (was 0.25/0.10)
CRRL_BPC_FLOOR_NEG: tuple = (0.25, 0.0, 0.75)   # FLOORELECT-822 (signed 2026-08-28): speed-NEG brake 0.25, alt-NEG descent 0.75, terminal window only
CRRL_BPC_BETA: tuple = (0.3, 0.3, 0.3)          # [CAND] P-4 reward-scale re-check may revise
# Default = CRRL_BPC_BETA => bounds byte-identical when the knob is unset.
# FD_CRRL_BPC_BETA_HDG_NEG overrides ONLY the heading component of the negative side (P4_HNB015
# probe cells).
CRRL_BPC_BETA_NEG: tuple = (
    CRRL_BPC_BETA[0],
    float(_os.environ.get("FD_CRRL_BPC_BETA_HDG_NEG", "") or CRRL_BPC_BETA[1]),
    CRRL_BPC_BETA[2],
)


@dataclass(frozen=True)
class StageWeights:
    """Per-stage reward component weights (10 weighted + r_approach + r_terminal unweighted)."""
    w_ref: float
    w_prox: float
    w_dwell: float
    w_safe: float
    w_smooth: float
    w_wake: float
    w_m3a_coarse: float = 1.0
    w_m3a_fine: float = 1.0
    w_contact_dwell: float = 0.0
    w_action_rate: float = 0.0
    w_exit_penalty: float = 1.0  # C3i: increased from 0.3 (C3h was 560x weaker than r_approach)
    w_overspeed: float = 0.25  # C3k: rebalanced from 1.0 (C3j 6.5:1 ratio -> ~2.1:1)
    # Default 0.0 => INERT.
    w_precontact: float = 0.0
    w_vsat: float = 0.0


# [FINAL SETUP: APPLIES]
STAGE_WEIGHTS: Dict[int, StageWeights] = {
    # Path 1: w_ref 0.2→0.5 (reference model now commands full approach trajectory, so r_ref HELPS
    # rather than anchoring at M2 station-keeping)
    0: StageWeights(
        w_ref=0.5, w_prox=0.5, w_dwell=1.0, w_safe=5.0,
        w_smooth=0.1, w_wake=0.5, w_m3a_coarse=1.0, w_m3a_fine=1.0,
        w_contact_dwell=0.3, w_action_rate=0.1),
    1: StageWeights(
        w_ref=0.5, w_prox=1.5, w_dwell=1.5, w_safe=5.0,
        w_smooth=0.1, w_wake=0.5, w_m3a_coarse=1.0, w_m3a_fine=1.0,
        w_contact_dwell=0.5, w_action_rate=0.1),
    2: StageWeights(
        w_ref=0.3, w_prox=2.0, w_dwell=2.0, w_safe=5.0,
        w_smooth=0.1, w_wake=1.0, w_m3a_coarse=1.0, w_m3a_fine=1.0,
        w_contact_dwell=0.5, w_action_rate=0.1),
    3: StageWeights(
        w_ref=0.3, w_prox=2.0, w_dwell=2.5, w_safe=5.0,
        w_smooth=0.1, w_wake=1.0, w_m3a_coarse=1.0, w_m3a_fine=1.0,
        w_contact_dwell=0.5, w_action_rate=0.1),
    4: StageWeights(
        w_ref=0.3, w_prox=2.0, w_dwell=2.5, w_safe=5.0,
        w_smooth=0.1, w_wake=1.0, w_m3a_coarse=1.0, w_m3a_fine=1.0,
        w_contact_dwell=0.5, w_action_rate=0.1),
    5: StageWeights(
        w_ref=0.3, w_prox=2.0, w_dwell=2.5, w_safe=5.0,
        w_smooth=0.1, w_wake=1.0, w_m3a_coarse=1.0, w_m3a_fine=1.0,
        w_contact_dwell=0.5, w_action_rate=0.1),
}

# R6 Stage 0 reward rebalance — default-off; gated on OPTION_C_R6_STAGE0_REBALANCE=1.
R6_STAGE0_REBALANCE_ACTIVE: bool = _os.environ.get("OPTION_C_R6_STAGE0_REBALANCE") == "1"
if R6_STAGE0_REBALANCE_ACTIVE:
    _r6_w_m3a = float(_os.environ.get("R6_W_M3A_COARSE", "2.0"))
    _r6_w_dwell = float(_os.environ.get("R6_W_DWELL", "1.5"))
    STAGE_WEIGHTS[0] = _dc_replace(STAGE_WEIGHTS[0], w_m3a_coarse=_r6_w_m3a, w_dwell=_r6_w_dwell)

# [FINAL SETUP: NOT USED, nothing reads the M3a fine-dwell terms]
R_M3A_COARSE_BASE: float = 0.05  # Flat per-tick reward inside M3a coarse
R_M3A_FINE_BASE: float = 0.10  # Base per-tick reward inside M3a fine
R_M3A_FINE_DWELL_COEFF: float = 0.02  # Dwell bonus coefficient for M3a fine
# [FINAL SETUP: APPLIES]
R_DWELL_COEFF: float = 1.00

CONTACT_DWELL_RAMP_CAP: float = 1.0

# Eliminates buzzing attractor by making sustained dwell disproportionately more valuable than
# transient contact.
CONTACT_DWELL_EXPONENT: float = 2.0

# C3j I3: linear ramp floor for r_contact_dwell. Increased from 0.15 (C3i) to 0.30 (C3j) to make
# dwell reward competitive with approach reward inside M3b.
DWELL_RAMP_FLOOR: float = 0.30

# Exits below LOWER_GATE are noise/boundary crossings and are not penalized.
EXIT_DWELL_THRESHOLD_S: float = 1.5
EXIT_DWELL_LOWER_GATE_S: float = 0.2
R_EXIT_PENALTY: float = -0.30
# C3i I2: behind-axis conditioned exit penalty multiplier. Behind-axis exits (forward overshoot)
# get 5x base penalty.
EXIT_BEHIND_MULTIPLIER: float = 5.0  # C3q: reinstated (was 5.0 in C3m, reverted C3n, reinstated under velocity gate)

# C3q V1: Sigmoid velocity gate sharpness for directional barrier reward. At k=15: modulation=0.9
# at +0.15 m/s, 0.5 at 0.0, 0.1 at -0.15 m/s.
VELOCITY_GATE_K: float = 15.0

W_HOLD_FWD: float = 0.5

# The endpoint form is what keeps both terms bit-exactly recomputable from stored replay-buffer
# observations (obs[2] = below_m / OBS_NORMS[2]); a per-sub-tick form would not be. Default-OFF
# via FD_CEILING_SHAPE_ENABLE (flag-off byte-identical).
CEILING_RAMP_START_M: float = 1.4  # ramp arms at this above-boom offset; the
                                    # bar it approaches is |COLLISION_ABOVE_BOOM_M| = 2.0 (the H1 bar)
W_ABOVE_ASYM: float = 0.10  # linear above-boom penalty per metre (asym term)
W_CEILING: float = 1.00  # quadratic ramp weight (potential term)
CEILING_SHAPE_GAMMA: float = 0.995  # PBRS discount — MUST equal the trainer gamma (TD3 0.995)

# A RAMPED endpoint term, ADDED BESIDE W_ABOVE_ASYM — it does not replace it.
W_ENDPOINT_RAMP_START_M: float = 0.80  # arms here, above the boom-tip

# Descent is never charged; below the arming height nothing is charged.
W_CEILRATE: float = 0.5  # penalty per (m/s) of climb, above the arm
CEILRATE_START_M: float = 0.80  # E5V-758: arms at the same height as E5
CEILRATE_DT_S: float = 0.25

W_ENDPOINT_RAMP: float = 0.20  # linear penalty per metre beyond it

# Capped, the max is 2.30 (0.07x) and the deep tail is carried by the LINEAR asym term alone.
# Default-OFF via FD_NOPASS_SHAPE_ENABLE (flag-off byte-identical).
NOPASS_RAMP_START_M: float = 1.4  # ramp arms at this in-front offset; the
                                    # bar it approaches is |COLLISION_BEHIND_MIN| = 2.0
W_INFRONT_ASYM: float = 0.10  # linear in-front penalty per metre (asym term)
W_NOPASS: float = 1.00  # quadratic ramp weight (potential term)
NOPASS_CAP_M: float = 0.6  # ramp-argument saturation = bar - ramp start
NOPASS_SHAPE_GAMMA: float = 0.995  # PBRS discount — MUST equal the trainer gamma (TD3 0.995)

DWELL_CAP_M3A_FINE: float = 0.20  # Saturates at ~50s continuous M3a fine dwell
# Saturates ~9s. Paired with R_DWELL_COEFF 1.00.
DWELL_CAP_M3B: float = 10.0

# Form: LINEAR in the normalized tier-4 error, 1.0 at the hold point, 0.0 at the tier-4 outer edge
# -- continuous across the band boundary. Cap 3.0 is a STRUCTURAL RAIL, not an active constraint:
# the linear raw term maxes at 1.0, so the cap cannot bind at w_precontact=1.0.
R_PRECONTACT_CAP: float = 3.0
# Weight, cap, linear form and the in_contact_m3b hard-zero are UNCHANGED; raw max is still 1.0,
# so the 1.0 < 3.0 < 10.0 rail ordering is byte-identical.
R_PRECONTACT_TIER: int = 1

R_APPROACH_THRESHOLD_M: float = 25.0
R_APPROACH_MAX: float = 0.30  # Path 1: 0.60→0.30 (reference model already commands approach)

# C3i I1: suppress r_approach inside M3b (eliminates buzzing incentive) Hard gate: r_approach =
# 0.0 when in_contact_m3b=True. C3j: DISABLED — replaced by smooth sigmoid taper
# (APPROACH_TAPER_*).
APPROACH_SUPPRESSION_INSIDE_M3B: bool = False

# C3j-1: Smooth sigmoid approach taper aligned to baseline ULTRA_NEAR boundary. Replaces the C3i
# hard gate with a smooth keep factor: approach_taper_keep = 1/(1+exp(-k*(behind_m - midpoint)))
# At midpoint, keep = 0.5.
APPROACH_TAPER_MIDPOINT: float = 10.0  # meters — baseline NEAR->ULTRA_NEAR transition
APPROACH_TAPER_STEEPNESS: float = 1.0  # sigmoid steepness parameter k

# C3j-2: Dynamic overspeed penalty in terminal corridor. Mirrors baseline's NEAR->ULTRA_NEAR
# deceleration profile.
OVERSPEED_CORRIDOR_OUTER: float = 10.0  # meters — corridor starts here
OVERSPEED_CORRIDOR_INNER: float = 0.7  # meters — corridor ends at M3b boundary
OVERSPEED_V_MIN: float = 0.05  # m/s — speed limit at inner boundary
OVERSPEED_V_MAX: float = 1.0  # m/s — speed limit at outer boundary

# [FINAL SETUP: NOT USED, nothing reads PASSTHROUGH_PENALTY]
# Applied per-tick when the L2 norm of the raw action vector is below threshold.
PASSTHROUGH_PENALTY: float = -0.03
# [FINAL SETUP: APPLIES]
PASSTHROUGH_L2_THRESHOLD: float = 0.05

# A 200K buffer fills in 10K steps. 400K provides ~20K steps of history (reasonable for TD3).
BUFFER_SIZE_HIGH_P: int = 400_000  # Used when n_envs > 8
BUFFER_SIZE_DEFAULT: int = 200_000  # Used when n_envs <= 8

# Stage-dependent dwell threshold for r_dwell activation, Early stages: low threshold to
# reward any M3b contact. Later stages: full 3.0s required to match SUCCESS_DWELL_S.
DWELL_REWARD_THRESHOLD_S: Dict[int, float] = {
    0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2,
}

# Terminal reward values.
R_TERMINAL_SUCCESS: float = 20.0
R_TERMINAL_FAIL: float = -10.0
R_TERMINAL_TIMEOUT: float = 0.0
R_TERMINAL_SAFETY: float = -10.0  # Safety termination (same magnitude as FAIL)

# Episode safety termination thresholds Range safety is now stage-dependent (see
# STAGE_SAFETY_RANGE below)
SAFETY_ALT_FLOOR_AGL_M: float = 30.0  # Terminate if alt AGL < this
SAFETY_DESCENT_RATE_MAX_MS: float = 8.0  # Terminate if Vd > this (down positive)

TAS_MIN_SAFE: float = 20.0  # m/s — stall margin (tighter than SAFE_FLOOR_MS=19)
TAS_MAX_SAFE: float = 30.0  # m/s — VNE margin (tighter than SAFE_MAX_MS=31)
AIRSPEED_VIOLATION_TICKS: int = 40  # 2.0s at 20 Hz → grace for gust recovery

COLLISION_CG_DISTANCE_MIN: float = 3.0  # meters
COLLISION_BEHIND_MIN: float = -2.0  # meters (ahead of leader limit)
COLLISION_BEHIND_MAX: float = 5.0  # meters (behind leader limit)
COLLISION_ABOVE_BOOM_M: float = -2.0  # meters (must be > 2m ABOVE boom-tip to trigger)
COLLISION_SUSTAINED_TICKS: int = 3  # 0.15s at 20Hz — filter single-tick transients
# Per-tick fallback to _last_geom when truth is unavailable. Default OFF = byte-identical;
# enabling in TRAINING additionally needs a G4 review (truth-conditioned episode boundary reaches
# the reward's SAFETY_ branch).
COLLISION_TRUTH_CHANNEL_ENABLE: bool = (
    _os.environ.get("FD_COLLISION_TRUTH_CHANNEL", "0") == "1")
# Consumed ONLY by the learner entry (train.py launch preconditions); the eval/gate path never
# reads it, so eval behavior is unchanged.
COLLISION_TRUTH_CHANNEL_TRAIN_ACK: str = _os.environ.get(
    "FD_COLLISION_TRUTH_CHANNEL_TRAIN_ACK", "").strip()

# Which SAFETY terminals are DEMOTED -- recorded as an event while the episode FLIES ON -- instead
# of ending the episode. Default empty == byte-identical to the historical path.
DEMOTABLE_TERMINALS: frozenset = frozenset(
    ("SAFETY_AIRSPEED", "SAFETY_COLLISION"))

# TELEMETRY FRESHNESS. THE ACTUAL MECHANISM, established by reading the cache LIFECYCLE and not
# the assignment line. tests/baseline_formation_controller.py sets _last_vfr_hud / _last_attitude
# / _last_position on ARRIVAL and NEVER clears or expires them.
TELEM_MAX_AGE_S: float = 0.5
# Default ON: this closes a fail-open, it does not add a feature. The kill switch exists so the
# rollback path is an env var and never a revert.
TELEM_FRESHNESS_ENABLE: bool = (
    _os.environ.get("FD_TELEM_FRESHNESS", "1").strip() == "1")
DEMOTE_TERMINALS: frozenset = frozenset(
    _n for _n in (_s.strip() for _s
                  in _os.environ.get("FD_DEMOTE_TERMINALS", "").split(","))
    if _n)
_demote_unknown = DEMOTE_TERMINALS - DEMOTABLE_TERMINALS
if _demote_unknown:
    raise ValueError(
        "FD_DEMOTE_TERMINALS names terminals that are not demotable: %s "
        "(demotable: %s). TERMDEMOTE-767 implements demotion at exactly two "
        "return sites, so a name outside that set would do nothing while the "
        "launch log claims otherwise." % (sorted(_demote_unknown),
                                          sorted(DEMOTABLE_TERMINALS)))
del _demote_unknown

# SAFETY_CONTACT -- genuine airframe contact. The QuadPlane top touches the leader belly at
# below_m < -3.42.
CONTACT_PLANE_BELOW_M: float = -3.42  # below_m < this == airframe contact
SAFETY_TERMINALS_KNOWN: frozenset = frozenset(
    ("SAFETY_CONTACT", "SAFETY_NOPASS"))
SAFETY_TERMINALS: frozenset = frozenset(
    _n for _n in (_s.strip() for _s
                  in _os.environ.get("FD_SAFETY_TERMINALS", "").split(","))
    if _n)
_st_unknown = SAFETY_TERMINALS - SAFETY_TERMINALS_KNOWN
if _st_unknown:
    raise ValueError(
        "FD_SAFETY_TERMINALS names unknown terminals: %s (known: %s). "
        "NOPASSTERM-770 implements exactly two return sites, so a name outside "
        "that set would do nothing while the launch log claims otherwise."
        % (sorted(_st_unknown), sorted(SAFETY_TERMINALS_KNOWN)))
del _st_unknown
NOPASS_TERM_AHEAD_M: float = float(
    _os.environ.get("FD_NOPASS_TERM_AHEAD_M", "2.0"))  # meters AHEAD of tip
NOPASS_TERM_PERSIST_S: float = float(
    _os.environ.get("FD_NOPASS_TERM_PERSIST_S", "2.0"))  # continuous seconds
NOPASS_TERM_PERSIST_TICKS: int = max(
    1, int(round(NOPASS_TERM_PERSIST_S * 20.0)))  # 20 Hz, as AIRSPEED_VIOLATION_TICKS
# Without it a LEAD_TURN rotates the leader's body frame across a follower 13 m off-axis and 19 m
# away at ~7 m/s, which reads as "13 m ahead" and ended 2 of 24 smoke legs for a non-hazard.
NOPASS_TERM_OFFAXIS_M: float = float(
    _os.environ.get("FD_NOPASS_TERM_OFFAXIS_M", "3.0"))  # sqrt(cross^2+below^2)

# This pair carries the same refusal into the UNCONDITIONAL launch-precondition cluster.
RAW_ALIGN_CENTER_ENABLE: bool = (
    _os.environ.get("FD_RAW_ALIGN_CENTER", "0").strip() == "1")
RAW_ALIGN_CENTER_TRAIN_ACK: str = _os.environ.get(
    "FD_RAW_ALIGN_CENTER_TRAIN_ACK", "").strip()

# Command rate limiting — match standalone controller's very-close-range rates to prevent TECS
# oscillation from 20Hz re-planning
CMD_REPOSITION_MIN_INTERVAL_S: float = 0.12  # standalone: VERY_CLOSE_REPOSITION_INTERVAL_S
CMD_REPOSITION_HDG_DELTA_DEG: float = 0.4  # standalone: VERY_CLOSE_HEADING_DELTA_DEG
CMD_SPEED_MIN_INTERVAL_S: float = 0.10  # standalone: VERY_CLOSE_SPEED_INTERVAL_S
CMD_SPEED_DELTA_THRESHOLD: float = 0.04  # standalone: VERY_CLOSE_SPEED_DELTA_MS

# Altitude forcing guards — match standalone controller,
CMD_ALT_TARGET_DELTA_M: float = 0.5  # standalone: ALT_CMD_DELTA_M
CMD_ALT_ERR_FORCE_M: float = 5.0  # standalone: ALT_ERR_FORCE_M
CMD_ALT_ERR_FORCE_TIME_S: float = 2.0  # standalone: ALT_ERR_FORCE_TIME_S

# Range safety zone — quadratic penalty (v5: replaces linear Approach F) Below SOFT_WARN_RANGE_M:
# no penalty. Above: quadratic ramp to max penalty.
SOFT_WARN_RANGE_M: float = 200.0  # Warn threshold (was SOFT_RESET_RANGE_M=300)
HARD_RANGE_MAX_M: float = 800.0  # Hard termination threshold (unchanged)
RANGE_PENALTY_QUADRATIC_MAX: float = 5.0  # Max penalty at HARD_RANGE_MAX_M

# Stage-dependent safety range overrides (wider for early exploration stages).
STAGE_SAFETY_RANGE: Dict[int, float] = {
    0: 800.0,  # Stage 0: very wide — let agent explore, baseline keeps it safe
    1: 600.0,  # Stage 1: still wide with mild wind
    2: 400.0,  # Stage 2: tighter, agent should be competent
    3: 300.0,  # Stage 3: production-like tightness
    4: 300.0,  # Stage 4: heavy-DR / sensor-degradation; circle deferred
    5: 300.0,  # Stage 5: worst-case stretch; circle deferred
}

# =============================================================================
# [FINAL SETUP: NOT USED, training used TD3]
# PPO HYPERPARAMETERS
# PRESERVED: PPO config retained for defense comparison (--algorithm ppo). Primary algorithm is
# TD3 (see TD3Config below).
# =============================================================================


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters — Option F unified config for all stages.

    v12: Unified to single config. Previously split into PPO_STAGE_0
    (ent=0.0, clip=0.05) and PPO_STAGE_1_PLUS (ent=0.005, clip=0.10).
    Campaign 004 showed that ent_coef=0.0 traps policy at zero-residual
    local optimum. Single config with mild exploration (ent=0.005,
    clip=0.15) allows learning from Stage 0.
    """
    n_steps: int = 900  # 4Hz × 225s episode = 900 decisions/env/rollout
    # NOT USED — overridden in train.py as n_steps * n_envs (full-batch).
    batch_size: int = 900
    n_epochs: int = 3
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.15  # v12: was 0.10; wider to escape zero-residual trap
    # C005b: 4x increase (was 0.005 — entropy bonus was 2e-6 of value_loss).
    ent_coef: float = 0.02
    # v14: halved to rebalance actor/critic gradient (pv_ratio was 0.003-0.006).
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    normalize_advantage: bool = True
    lr_initial: float = 1e-4  # C005b: constant for Stage 0 (was 3e-4 decaying to 1e-5)
    # DEFERRED: Linear decay disabled in v14 for Stage 0 stability. Revisit for Stage 1+ if needed.
    lr_final: float = 1e-4
    net_arch: Any = field(default_factory=lambda: dict(
        pi=[256, 256], vf=[256, 256]))  # C005b: separate actor/critic MLPs


# Single PPO config for all curriculum stages.
PPO_CONFIG = PPOConfig()


# =============================================================================
# [FINAL SETUP: APPLIES]
# TD3 HYPERPARAMETERS (v18 — off-policy pivot)
# =============================================================================

@dataclass(frozen=True)
class TD3Config:
    """TD3 hyperparameters for RRL residual learning.

    TD3 (Twin Delayed DDPG) uses deterministic policy with additive noise for
    exploration, twin critics (Q1, Q2) to mitigate overestimation, and delayed
    policy updates. The off-policy replay buffer enables experience reuse,
    giving 5-20x sample efficiency vs PPO for expensive SITL episodes.

    Noise sigma is per-dimension: [speed, heading, altitude].
    Speed residual (±1 m/s scaled) needs less exploration than heading (±3°).
    Altitude is tightest (±1 m) — minimal noise to avoid safety terminations.
    """
    gamma: float = 0.995  # Discount — match PPO for fair comparison
    tau: float = 0.005
    # NOTE: On resume, SB3 restores lr from checkpoint. Use --learning-rate CLI flag to override.
    learning_rate: float = 5e-5
    buffer_size_stage0: int = 200_000
    buffer_size_stage1_plus: int = 200_000  # F1: same 200K in later stages (values only; stage-key flatten = F4, separate)
    batch_size: int = 256  # Gradient batch from replay buffer
    learning_starts: int = 4000  # Pure random exploration (~4.4 episodes), per-env basis
    learning_starts_total_cap: int = 16_000  # F2: hardware-independent cap on learning_starts*n_envs (first gradient at the same transition count on any box)
    train_freq: int = 1  # Train every step (train_freq=(1,"step"))
    gradient_steps: int = 4  # Legacy reference at n_envs=4; effective value derived via grads_per_transition (F3)
    grads_per_transition: float = 1.0  # F3: hardware-decoupled UTD — effective gradient_steps = round(grads_per_transition*n_envs)
    policy_delay: int = 2
    noise_sigma_speed: float = 0.15  # σ for speed residual noise
    noise_sigma_heading: float = 0.10
    noise_sigma_altitude: float = 0.10  # σ for altitude residual noise
    # Target smoothing noise σ (halved for narrow residual action space, v19 P1)
    target_policy_noise: float = 0.1
    # Target noise clipping bound (halved, consistent with target_policy_noise, v19 P1)
    target_noise_clip: float = 0.25
    # HISTORICAL: v19 P1 sizing rationale, at the THEN-current widths. HISTORICAL: actor 128, critic
    # HISTORICAL: 256; actor mapped 39D→3D simple, critic 42D→1D over 8-component reward is harder)
    net_arch: Any = field(default_factory=lambda: dict(pi=[128, 128], qf=[256, 256]))


TD3_CONFIG = TD3Config()


# =============================================================================
# [FINAL SETUP: NOT USED, training used TD3]
# SAC HYPERPARAMETERS (v18 — off-policy fallback)
# =============================================================================

@dataclass(frozen=True)
class SACConfig:
    """SAC hyperparameters — fast fallback if TD3 exploration proves insufficient.

    SAC (Soft Actor-Critic) uses a stochastic policy with automatic entropy
    tuning. The ``ent_coef="auto"`` setting learns the entropy coefficient α
    via dual gradient descent, targeting entropy ≈ −dim(action) = −3.

    Shares most infrastructure with TD3 (replay buffer, VecNormalize, callbacks).
    Primary difference: stochastic policy enables automatic exploration without
    manual noise tuning, at the cost of slightly higher variance in actions.
    """
    gamma: float = 0.995
    tau: float = 0.005
    # NOTE: On resume, SB3 restores lr from checkpoint. Use --learning-rate CLI flag to override.
    learning_rate: float = 5e-5  # C006 P2: 1e-4→5e-5, matching TD3
    buffer_size_stage0: int = 50_000
    buffer_size_stage1_plus: int = 100_000
    batch_size: int = 256
    learning_starts: int = 4000
    train_freq: int = 1
    gradient_steps: int = 4
    # Automatic entropy tuning (dual gradient descent)
    ent_coef: str = "auto"
    target_entropy: str = "auto"  # Default: -dim(action) = -3
    net_arch: Any = field(default_factory=lambda: dict(
        pi=[128, 128], qf=[256, 256]))  # v19 P1: asymmetric, matching TD3


SAC_CONFIG = SACConfig()


# =============================================================================
# [FINAL SETUP: APPLIES]
# OFF-POLICY MONITORING THRESHOLDS (v18)
# =============================================================================
# VecNormalize stats are frozen after this many episodes so that replay buffer
# samples remain consistently normalized.
VECNORM_CALIBRATION_EPISODES: int = 30

# Zero-residual alert: triggers if action L2 norm stays below threshold for
# ZERO_RESIDUAL_ALERT_EPISODES consecutive episodes (policy stuck at baseline passthrough).
ZERO_RESIDUAL_ALERT_THRESHOLD: float = 0.05
ZERO_RESIDUAL_ALERT_EPISODES: int = 5

# Q-divergence alert: if critic_loss > Q_DIVERGENCE_MULTIPLIER × mean_episode_reward, critic is
# overfitting or Q-values are diverging. Imperfect proxy — refine to track actual Q-value
# magnitudes before Phase D production campaigns.
Q_DIVERGENCE_MULTIPLIER: float = 500.0
# Divergence is about GROWTH RATE, not level — a rate-of-change conjunct is the registered follow-
# on.
Q_ABSOLUTE_THRESHOLD: float = 500.0
# L9.3 advisory magnitude flag (non-halting, fires once per run).
Q_MAX_EARLY_WARNING_THRESHOLD: float = 3500.0
Q_MAX_EARLY_WARNING_LOOKBACK: int = 25

# Checkpoint-gated rollback configuration (H3)
ROLLBACK_EVAL_INTERVAL_EPISODES: int = 15
ROLLBACK_EVAL_EPISODES: int = 10
ROLLBACK_GRACE_EPISODES: int = 25
ROLLBACK_TRIGGER_RATIO: float = 0.50
ROLLBACK_COOLDOWN_EPISODES: int = 50
MAX_ROLLBACKS: int = 1
ROLLBACK_BEST_CHECKPOINT_DIR: str = "best_policy"

# Values below are GENERATOR-DERIVED by tools/h12_calibration.py from the banked healthy corpus
# (margin rule on its face: threshold = 1.5x max healthy L25 rounded up to 5; floor = 3x max
# healthy rise rounded up to 0.01) and pinned by tests/unit/drl/test_h12_conjunction.py — never
# hand-edit one without regenerating the other.
H12_LOSS_THRESHOLD: float = 65.0
H12_MIN_RISE_FRAC: float = 0.10

EARLY_STOP_ENABLED: bool = True
EARLY_STOP_METRIC: str = "mean_dwell"  # achieved_rate SATURATES on C006
# A guard that cannot fire is not a guard.
EARLY_STOP_WINDOW_EPISODES: int = 6  # was 10
EARLY_STOP_DELTA_FRAC: float = 0.5  # fire bar: value <= R* * (1 - delta)
EARLY_STOP_PATIENCE_WINDOWS: int = 3  # consecutive degraded windows
EARLY_STOP_MIN_TIMESTEPS: int = 15000  # no firing before this budget
EARLY_STOP_FLOOR_WINDOWS: int = 3  # 0 disables floor mode
EARLY_STOP_FLOOR_EPS: float = 0.2  # metric units (s for mean_dwell)


# =============================================================================
# [FINAL SETUP: APPLIES as the container the evaluator indexes; the V3 ladder rebinds it below]
# CURRICULUM
# =============================================================================

@dataclass(frozen=True)
class CurriculumStage:
    """Configuration for one curriculum stage."""
    max_rollouts: int
    wind_max_ms: float
    wind_turb: float
    gps_accuracy: float  # v1: SIM_GPS_ACCURACY (reported, no-op); v2: GPS CEP target (m)
    m3b_early_exit_rate: float  # Early-exit threshold (M3b rate)
    warmup_target: str = "M2"  # "M1" = REJOIN_PRECONTACT, "M2" = HOLD_PRECONTACT
    action_scale: float = 1.0  # Multiplier on ACTION_SCALES (0.3=gentle, 1.0=full)
    # D2 (FD_CURRICULUM_V2) additive fields — defaults keep every v1 entry byte-identical (these are
    # ignored on the v1 code path).
    total_timesteps: int = 0  # v2 per-stage timestep budget (advance trigger)
    gps_dropout_enabled: bool = False  # v2 GPS dropout windows (S4a+ only)
    gps_dropout_max_per_episode: int = 0
    wind_from_deg: float = 0.0  # meteorological FROM bearing (v3: 180.0 fixed)
    wind_sigma_ms: tuple = ()  # FDM Dryden sigma triple (m/s); => no turbulence args
    wind_L_m: tuple = ()  # FDM Dryden length-scale triple (m; 8785C full-length)
    wind_fixed: bool = False  # True => wind_max_ms IS the episode wind (no uniform draw)
    stage_label: str = ""


# Warmup is a simulation infrastructure limitation (ArduPilot SITL cannot restore mid-flight
# state), NOT a training variable. Using M2 for all stages ensures: 1.
CURRICULUM_STAGES: Dict[int, CurriculumStage] = {
    0: CurriculumStage(
        max_rollouts=20, wind_max_ms=0.0, wind_turb=0.0,
        gps_accuracy=0.3, m3b_early_exit_rate=0.5,
        warmup_target="M2", action_scale=1.0),
    1: CurriculumStage(
        max_rollouts=20, wind_max_ms=2.0, wind_turb=0.1,
        gps_accuracy=0.3, m3b_early_exit_rate=0.5,
        warmup_target="M2", action_scale=1.0),
    2: CurriculumStage(
        max_rollouts=34, wind_max_ms=5.0, wind_turb=0.3,
        gps_accuracy=0.3, m3b_early_exit_rate=0.5,
        warmup_target="M2", action_scale=1.0),
    3: CurriculumStage(
        max_rollouts=15, wind_max_ms=8.0, wind_turb=0.6,
        gps_accuracy=0.3, m3b_early_exit_rate=0.5,
        warmup_target="M2", action_scale=1.0),
    4: CurriculumStage(
        max_rollouts=15, wind_max_ms=8.0, wind_turb=0.6,
        gps_accuracy=1.0, m3b_early_exit_rate=0.5,
        warmup_target="M2", action_scale=1.0),
    5: CurriculumStage(
        max_rollouts=15, wind_max_ms=8.0, wind_turb=0.8,
        gps_accuracy=2.0, m3b_early_exit_rate=0.5,
        warmup_target="M2", action_scale=1.0),
}

EARLY_EXIT_MIN_ROLLOUTS: int = 3  # Don't early-exit before this many rollouts

# =============================================================================
# [FINAL SETUP: NOT USED, no runtime code reads a condition profile]
# CONDITION PROFILES
# =============================================================================
# Named profiles for runtime conditions, distinct from CURRICULUM_STAGES.
# Validation/canary/smoke entry points default to CLEAN_STRAIGHT_ZERO_NOISE; training entry points
# require explicit --profile flag (see train.py). Profile field semantic: - wind_max_ms /
# wind_turb / gps_accuracy: same as CurriculumStage - leader_profile: "straight" | "racetrack" |
# "circle" - leader_omega_deg_s: only meaningful when leader_profile == "circle" gps_accuracy
# interpretation: - SIM_GPS_ACCURACY (the *reported* HDOP to ArduPilot's onboard EKF), NOT
# SIM_GPS_NOISE (physical noise on GPS measurements).
CONDITION_PROFILES: Dict[str, Dict[str, Any]] = {
    "clean_straight_zero_noise": {
        "wind_max_ms": 0.0,
        "wind_turb": 0.0,
        "gps_accuracy": 0.05,  # Patch A0: 0.05 (not 0.0; not 0.3)
        "leader_profile": "straight",
        "leader_omega_deg_s": None,
    },
    "stage0_canonical": {
        "wind_max_ms": 0.0,
        "wind_turb": 0.0,
        "gps_accuracy": 0.3,  # Stage 0 default (legacy compat)
        "leader_profile": "racetrack",
        "leader_omega_deg_s": None,
    },
    "curriculum": {
        # Sentinel: defer to CURRICULUM_STAGES + sample_domain_randomization. Selected via
        # --profile=curriculum on training entry points.
        "wind_max_ms": None,
        "wind_turb": None,
        "gps_accuracy": None,
        "leader_profile": None,
        "leader_omega_deg_s": None,
    },
}

DEFAULT_VALIDATION_PROFILE: str = "clean_straight_zero_noise"

# =============================================================================
# [FINAL SETUP: APPLIES]
# ENVIRONMENT
# =============================================================================
EPISODE_DURATION_S: float = 900.0
# training, promotion exams and benchmark evals alike.
TICK_HZ: int = 20  # DRL control frequency (sub-tick rate)
TICK_DT_S: float = 1.0 / TICK_HZ  # Nominal tick period (0.05 s)
ACTION_REPEAT: int = 5  # 20Hz / 5 = 4Hz DRL decisions

# SITL port base (env_id * PORT_STRIDE for isolation)
PORT_BASE: int = 14551
PORT_STRIDE: int = 100

# Port bases for multi-process SITL isolation
SITL_SERIAL_PORT_BASE: int = 5760  # ArduPlane SITL serial0=tcp: base
PORT_BASE_LEADER: int = 14550  # Leader broadcaster UDP base
PORT_BASE_TIMESYNC: int = 14553  # Timesync UDP base

# JSON-lockstep backend (#309 P1; read ONLY when FD_TRAIN_BACKEND=json — inert on native) NOTE:
# the leader engage gate (FD_ENGAGE_SIM_TIME_S) is NOT armed by the env — sim-time gates are
# default-OFF for representative runs (standing policy; the census docked UNGATED with IC-hold).
# P2 arms it deliberately via parent env.
FDM_PORT_BASE: int = 9002  # external JSON FDM UDP bind base (env_id * PORT_STRIDE offset)
JSON_HOME_ALT_AMSL: float = 584.0  # true field elevation (launcher --home / FDM --home-alt-amsl).
                                       # Native capture path untouched.
JSON_LEADER_START_OFFSET_M: float = 39550.0  # census-substrate spawn geometry, CALIBRATED for the
                                       # The j6 value 38850 lands 683 m deeper in this assembly, outside the instant-LEAD_TURN criterion;
                                       # the 9.26 km auto placement is far outside the validated cell.
JSON_LEADER_HDG_DEG: float = 180.0
                                       # the starting default across EVERYTHING — no tail chases). Racetrack provably ignores --hdg
                                       # (heading solely from PatternState :2062-2115) so passing it unconditionally is inert there.

# Wall-clock stall detector threshold
STALL_DETECTOR_THRESHOLD_S: float = 30.0

# Each attempt's timebase settle is a LOTTERY, not a deterministic check: the 4xl sustains ~1.58x
# sim/wall at --speedup 2.0 (p05 1.49 / p50 1.58 / p95 1.66, n=416) while QUALITY_GATE_MIN=0.95
# demands rate_err <= 5%, i.e. rate_med in [1.90, 2.10]. rate_med is the median of a rolling
# 20-sample window (sim_time_authority.RATE_WINDOW_SIZE), so a settle passes only when jitter
# lifts that median over the bar inside the 30 s window. Ten leave ~5e-6.
MAX_WARMUP_RETRIES: int = 10  # Max attempts for launch+warmup before giving up
WARMUP_RETRY_DELAY_S: float = 5.0  # Seconds to wait between retry attempts
# WARMUP_TIMEOUT_S — the wall-clock LIVENESS watchdog per attempt (frozen- sim protection only;
# never the scientific budget). Overrides are validation-leg escape hatches and FAIL CLOSED: they
# require FD_ALLOW_WARMUP_OVERRIDE=1, finite positive bounded values, and the wall value must stay
# < FLIGHTDOCK_WATCHDOG_STALE_MIN × 60 (default 900 s) so a hung warmup triggers TimeoutError +
# retry before the watchdog kills the process.
def _warmup_budget_env(_key: str, _default: float) -> float:
    _raw = _os.environ.get(_key)
    if _raw is None or _raw.strip() == "":
        return _default
    if _os.environ.get("FD_ALLOW_WARMUP_OVERRIDE", "0") != "1":
        raise ValueError(
            f"{_key}={_raw!r} is set but FD_ALLOW_WARMUP_OVERRIDE != 1 — "
            "warmup-budget overrides are validation-only (production guard)")
    try:
        _v = float(_raw)
    except ValueError as _e:
        raise ValueError(f"{_key} invalid literal: {_raw!r}") from _e
    if not (0.0 < _v <= 86400.0):  # NaN/inf/zero/negative all fail this
        raise ValueError(f"{_key} out of range (0, 86400]: {_raw!r}")
    return _v


WARMUP_TIMEOUT_S: float = _warmup_budget_env("FD_WARMUP_WALL_TIMEOUT_S", 600.0)
WARMUP_MAX_SIM_S: float = _warmup_budget_env("FD_WARMUP_MAX_SIM_S", 900.0)
_watchdog_stale_s = float(_os.environ.get("FLIGHTDOCK_WATCHDOG_STALE_MIN", "15")) * 60.0
if WARMUP_TIMEOUT_S >= _watchdog_stale_s:
    raise ValueError(
        f"FD_WARMUP_WALL_TIMEOUT_S={WARMUP_TIMEOUT_S} must be < watchdog stale "
        f"interval {_watchdog_stale_s}s (raise FLIGHTDOCK_WATCHDOG_STALE_MIN "
        "coherently in the same launch env)")

# DO_REPOSITION carrot-point distance (meters ahead of follower)
CARROT_DISTANCE_M: float = 1000.0

# Default max closure rate for obs vector context
DEFAULT_MAX_CLOSURE_MS: float = 1.5

# Idle-drift fix (v7 — SIGSTOP/SIGCONT)
AHRS_EKF_TYPE_DRL: int = 10  # Ground-truth position, bypass EKF
FS_GCS_ENABLE_DRL: int = 0  # Disable GCS failsafe during SIGSTOP pauses
IDLE_DRIFT_SIGSTOP_ENABLED: bool = True

# Episode termination C006 B2: disabled at Stage 0 — removes +20 replay buffer attractor
SUCCESS_DWELL_S: float = float('inf')

# Reserved for Stage 1+ curriculum gating. Not consumed at runtime — SUCCESS_DWELL_S above is the
# active constant.
SUCCESS_DWELL_S_BY_STAGE: Dict[int, float] = {
    0: float('inf'),  1: 3.0,  2: 3.0,  3: 3.0,  4: 3.0,  5: 3.0,
}

# =============================================================================
# [FINAL SETUP: NOT USED, curriculum V2 is off by default; V3 is the ladder of record]
# CURRICULUM V2 (D2 — FD_CURRICULUM_V2; ADDITIVE, default-OFF, byte-identical
# when off). 11 sub-stages S0a..S5a; timestep-budgeted; degraded GPS realized as a seeded slow
# GLTCH random-walk + dropout (env-side ladder, never touches the fuser). When the flag is ON the
# v1 names are rebound to these tables so the frozen reward_mrrl (reads config.STAGE_WEIGHTS
# directly), the env, and the manager all pick up v2 values transparently.
# =============================================================================
STAGE_LABELS: Dict[int, str] = {
    0: "S0a", 1: "S0b", 2: "S1a", 3: "S1b", 4: "S2a", 5: "S2b",
    6: "S3a", 7: "S3b", 8: "S4a", 9: "S4b", 10: "S5a",
}
MAX_STAGE_INDEX_V2: int = 10
CUMULATIVE_TIMESTEPS_V2 = [
    200_000, 400_000, 800_000, 1_300_000, 1_900_000, 2_500_000,
    3_300_000, 4_100_000, 5_100_000, 6_300_000, 7_800_000,
]

CURRICULUM_STAGES_V2: Dict[int, CurriculumStage] = {
    0:  CurriculumStage(max_rollouts=10**9, wind_max_ms=0.0, wind_turb=0.0, gps_accuracy=0.1, m3b_early_exit_rate=0.5, total_timesteps=200_000),
    1:  CurriculumStage(max_rollouts=10**9, wind_max_ms=1.0, wind_turb=0.1, gps_accuracy=0.1, m3b_early_exit_rate=0.5, total_timesteps=200_000),
    2:  CurriculumStage(max_rollouts=10**9, wind_max_ms=2.0, wind_turb=0.1, gps_accuracy=0.1, m3b_early_exit_rate=0.5, total_timesteps=400_000),
    3:  CurriculumStage(max_rollouts=10**9, wind_max_ms=2.5, wind_turb=0.1, gps_accuracy=0.1, m3b_early_exit_rate=0.5, total_timesteps=500_000),
    4:  CurriculumStage(max_rollouts=10**9, wind_max_ms=3.0, wind_turb=0.2, gps_accuracy=0.2, m3b_early_exit_rate=0.5, total_timesteps=600_000),
    5:  CurriculumStage(max_rollouts=10**9, wind_max_ms=3.5, wind_turb=0.2, gps_accuracy=0.2, m3b_early_exit_rate=0.5, total_timesteps=600_000),
    6:  CurriculumStage(max_rollouts=10**9, wind_max_ms=4.0, wind_turb=0.3, gps_accuracy=0.3, m3b_early_exit_rate=0.5, total_timesteps=800_000),
    7:  CurriculumStage(max_rollouts=10**9, wind_max_ms=4.5, wind_turb=0.3, gps_accuracy=0.5, m3b_early_exit_rate=0.5, total_timesteps=800_000),
    8:  CurriculumStage(max_rollouts=10**9, wind_max_ms=5.0, wind_turb=0.4, gps_accuracy=0.5, m3b_early_exit_rate=0.5, total_timesteps=1_000_000, gps_dropout_enabled=True, gps_dropout_max_per_episode=1),
    9:  CurriculumStage(max_rollouts=10**9, wind_max_ms=6.0, wind_turb=0.4, gps_accuracy=0.7, m3b_early_exit_rate=0.5, total_timesteps=1_200_000, gps_dropout_enabled=True, gps_dropout_max_per_episode=2),
    10: CurriculumStage(max_rollouts=10**9, wind_max_ms=6.0, wind_turb=0.5, gps_accuracy=1.0, m3b_early_exit_rate=0.5, total_timesteps=1_500_000, gps_dropout_enabled=True, gps_dropout_max_per_episode=3),
}

# Re-keyed reward weights (inherits any R6 Stage-0 rebalance already applied to STAGE_WEIGHTS[0]);
# old-S0 band -> {0,1}, S1 -> {2,3}, S2 -> {4,5}, S3 -> {6..10}.
STAGE_WEIGHTS_V2: Dict[int, StageWeights] = {
    0: STAGE_WEIGHTS[0], 1: STAGE_WEIGHTS[0],
    2: STAGE_WEIGHTS[1], 3: STAGE_WEIGHTS[1],
    4: STAGE_WEIGHTS[2], 5: STAGE_WEIGHTS[2],
    6: STAGE_WEIGHTS[3], 7: STAGE_WEIGHTS[3], 8: STAGE_WEIGHTS[3],
    9: STAGE_WEIGHTS[3], 10: STAGE_WEIGHTS[3],
}
# Safety termination range (monotone-by-wind; termination only, not reward).
STAGE_SAFETY_RANGE_V2: Dict[int, float] = {
    0: 800.0, 1: 600.0, 2: 400.0, 3: 400.0, 4: 300.0, 5: 300.0,
    6: 300.0, 7: 300.0, 8: 300.0, 9: 300.0, 10: 300.0,
}
DWELL_REWARD_THRESHOLD_S_V2: Dict[int, float] = {i: 0.2 for i in range(11)}
SUCCESS_DWELL_S_BY_STAGE_V2: Dict[int, float] = {
    0: float('inf'), 1: float('inf'), **{i: 3.0 for i in range(2, 11)},
}

# GPS degradation ladder constants (env-side OU GLTCH walk + dropout scheduler).
GPS_OU_SIGMA_FACTOR: float = 1.1774  # per-axis sigma = CEP / 1.1774 (2-D median ~ CEP)
GPS_OU_TAU_S: float = 20.0  # OU drift timescale (10-30 s -> survives EKF3 as drift)
GPS_DEG_PER_M: float = 8.99e-6  # meters -> degrees latitude (matches gltch_*.parm)
GPS_UPDATE_HZ: int = 5  # GLTCH re-read at the GPS output rate
GPS_DROPOUT_MIN_S: float = 1.0
GPS_DROPOUT_MAX_S: float = 3.0

# #379 (P0.3): constant leader-datalink transport delay. The RUNTIME knob is the
# FD_LEADER_TX_DELAY_S env var read by tests/leader_broadcast_fixed.py (sender-side FIFO delay
# line — one injection point, so the frozen controller and the DRL obs see CONSISTENTLY delayed
# leader data; unset/0 = inert).
LEADER_TX_DELAY_S_DEFAULT: float = 0.0

# --- Flag-gated rebind: ON => v1 names point to the v2 tables (transparent to frozen reward_mrrl
# + env + manager); OFF => byte-identical v1. ---
FD_CURRICULUM_V2: bool = _os.environ.get("FD_CURRICULUM_V2", "0") == "1"
CUMULATIVE_TIMESTEPS = None  # v1: rollout-budgeted (no timestep ladder)
if FD_CURRICULUM_V2:
    CURRICULUM_STAGES = dict(CURRICULUM_STAGES_V2)
    STAGE_WEIGHTS = dict(STAGE_WEIGHTS_V2)
    STAGE_SAFETY_RANGE = dict(STAGE_SAFETY_RANGE_V2)
    DWELL_REWARD_THRESHOLD_S = dict(DWELL_REWARD_THRESHOLD_S_V2)
    SUCCESS_DWELL_S_BY_STAGE = dict(SUCCESS_DWELL_S_BY_STAGE_V2)
    CUMULATIVE_TIMESTEPS = list(CUMULATIVE_TIMESTEPS_V2)

# =============================================================================
# [FINAL SETUP: APPLIES (curriculum V3 ladder of record)]
# CURRICULUM V3 (P1-b — FD_CURRICULUM_V3; wind ladder per CURRICULUM_V3.
# Values MIRROR tools/curriculum_v3_totals.stage_table() (the RC5 generator of record stays
# single-source): the cross-pin unit test (tests/unit/drl/test_curriculum_v3.py) asserts field-by-
# field equality so the literals here can never drift from the generator. Advance authority:
# SEALED GATES ONLY (curriculum_gate_runner verdict -> CurriculumManager.promote()).
# =============================================================================
# W-H: sigma override knob. The CurriculumManager
# REWRITES FD_FDM_WIND_* every reset (smoke-r1 lesson), so a sigma probe MUST override at the
# cell-table source, not the env vars. Comma triple; parse failure or wrong arity = raise (fail-
# closed, never a silent calm-air placebo).
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
_V3_SIGMA_DEFAULT = "0.10,0.10,0.10"
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
_V3_SIGMA = tuple(
    float(x) for x in _os.environ.get(
        "FD_V3_SIGMA_OVERRIDE_MS", _V3_SIGMA_DEFAULT).split(","))
if len(_V3_SIGMA) != 3:
    raise ValueError(
        f"FD_V3_SIGMA_OVERRIDE_MS must be a comma triple, got {_V3_SIGMA!r}")
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
_V3_L = (300.0, 300.0, 300.0)  # 8785C full-length convention (OU tau = L/V)
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
_V3_CEP_M = 0.25  # GPS CEP via the v2 OU-GLTCH env ladder

# --- SIGMA POLICY: ONE FACT, ONE NAME ------------------------------------- The set of sigma
# values the curriculum may train on, and the set sealed for H3. WHY THESE ARE NAMED.
V3_CURRICULUM_SIGMAS: tuple = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30)
# T5 governs EXAM venues only and is untouched.
V3_HOLDOUT_SIGMAS: tuple = (0.35, 0.40, 0.50, 0.80)

# WHY IT EXISTS. Prose cannot be diffed against prose; this dict can, and the detector diffs it.
# [FINAL SETUP: NOT USED, nothing reads it]
HYPOTHESIS_VENUES: dict = {
    "H2": {"sigma": 0.25, "winds": (6.0, 7.0, 8.0), "band": "light",
           "never_train": False,
           "signature": "H2REPIN-20260828 (operator: light = sigma 0.25); was 0.30 under ARMBTOP-CAP8-20260817 + A-15.3"},
    # Preregister it.
    "H3": {"sigma": 0.35, "winds": (6.0, 7.0, 8.0), "band": "moderate",
           "never_train": True,
           "signature": "ARMBSIG-810-A7-20260824 (sigma 0.35 + bar -35 pp; "
                        "supersedes SIGMAELECT-801-20260822's 0.40) + "
                        "ARMBTOP-CAP8-20260817 §4 (+A1-20260817, +A2-20260818) (winds)"},
}


def _v3_stage(label: str, wind_ms: float, turb: bool,
              sigma_ms: tuple = None) -> CurriculumStage:
    return CurriculumStage(
        max_rollouts=10**9, wind_max_ms=wind_ms,
        wind_turb=0.0,  # SIM_WIND_TURB knob is inert on the json backend; the
                        # REAL turbulence is the FDM Dryden sigma below.
        gps_accuracy=_V3_CEP_M, m3b_early_exit_rate=0.5,
        warmup_target="M2", action_scale=1.0,
        wind_from_deg=180.0,
        wind_sigma_ms=(_V3_SIGMA if sigma_ms is None else sigma_ms)
        if turb else (),
        wind_L_m=_V3_L if turb else (),
        wind_fixed=True, stage_label=label)


# Sigma is the SECOND curriculum axis; single-factor attribution is preserved because sigma steps
# happen at FIXED wind (D1 @ 6.0, D2 @ 8.0) and wind steps at FIXED sigma.
C1_STAGE_INDEX_V3: int = 22
# C1 remains the H2 anchor at 22.
MAX_STAGE_INDEX_V3: int = 39  # A3: L4/L2/L0 27-29, P1a/P1/L3/P2a 30-33; A2A0..A2A25 34-39
# A2A-20260901: 33 -> 39 for the six appended A2A rungs.
PROMOTION_CEILING_V3: int = C1_STAGE_INDEX_V3
# (label, wind_ms, sigma) per stage index.
_V3_LADDER_SPEC: tuple = (
    ("S0", 0.0, 0.10),
    *((f"W{i - 1}", 0.5 * (i - 1), 0.10) for i in range(1, 14)),
    ("D1a", 6.0, 0.15),  # sigma step @ 6.0 (1/2)
    ("D1", 6.0, 0.20),  # sigma step @ 6.0 (2/2)
    *((f"W{i - 3}", 0.5 * (i - 3), 0.20) for i in range(16, 20)),
    ("D2a", 8.0, 0.25),  # sigma step @ 8.0 (1/2)
    ("D2b", 8.0, 0.30),  # sigma step @ 8.0 (2/2)
    # The 0.20 -> 0.30 step at wind 8.0 became two steps of 0.05, for the same reason S-2 split the
    # 0.10 -> 0.20 step at wind 6.0: D2 attempt_001 failed 7/12 with FOUR ceiling busts and the policy
    # parked ON the 2.00 m bar.
    ("C1", 8.0, 0.30),
    # APPEND-ONLY — placing these AFTER C1 keeps every historical stage index valid (never pool across
    # a definition change), keeps C1 = 22 = the H2 anchor, and keeps C1's per-cell sampler (range(2,
    # C1_STAGE_INDEX_V3)) untouched.
    ("D1b", 6.0, 0.25),  # sigma step @ 6.0 (3/3)
    ("R13", 6.5, 0.25),  # wind ramp @ sigma 0.25
    ("R14", 7.0, 0.25),  # wind ramp @ sigma 0.25
    ("R15", 7.5, 0.25),  # wind ramp @ sigma 0.25
    # No floor-1.00 training exists anywhere below 6.0 m/s.
    ("L4", 4.0, 0.25),  # low-wind @ sigma 0.25
    ("L2", 2.0, 0.25),  # low-wind @ sigma 0.25
    # L0 closes it.
    ("L0", 0.0, 0.25),  # low-wind @ sigma 0.25
    # P1a/P1/P2a are the SAME cells as D1a/D1/D2a on CLEAN ledgers, which also keeps floor-1.00
    # training out of the floor-(0,0,0) stages' ledgers (never pool across a definition change).
    ("P1a", 6.0, 0.15),
    ("P1", 6.0, 0.20),
    ("L3", 3.0, 0.25),
    ("P2a", 8.0, 0.25),
    ("A2A0",  0.0, 0.0),  # still air anchor
    ("A2A8",  8.0, 0.0),  # max examined wind, zero turbulence
    ("A2A10", 8.0, 0.10),  # sigma sweep @ 8.0
    ("A2A15", 8.0, 0.15),  # sigma sweep @ 8.0
    ("A2A20", 8.0, 0.20),
    ("A2A25", 8.0, 0.25),  # the exam venue condition
)
STAGE_LABELS_V3: Dict[int, str] = {
    i: spec[0] for i, spec in enumerate(_V3_LADDER_SPEC)}
# Per-stage sigma triples — the DEFAULT under the diagonal. A set FD_V3_SIGMA_OVERRIDE_MS wins
# GLOBALLY (probe/eval semantics preserved; guarded by the env_contract ack, not by this table).
V3_SIGMA_IS_OVERRIDE: bool = "FD_V3_SIGMA_OVERRIDE_MS" in _os.environ
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
V3_SIGMA_BY_STAGE: Dict[int, tuple] = {
    i: (_V3_SIGMA if V3_SIGMA_IS_OVERRIDE else (spec[2],) * 3)
    for i, spec in enumerate(_V3_LADDER_SPEC)}
V3_REHEARSAL_START_INDEX: int = 3
V3_ANCHOR_STAGES: tuple = (0, 9, 22)
# Pinned by tests/unit/test_ladder_doc_in_sync.py:: test_h2_decisive_cell_is_anchor_gated, which
# asserts the CELL, not the index.

CURRICULUM_STAGES_V3: Dict[int, CurriculumStage] = {
    # A2A-20260901: the turbulence flag is `not stage 0 AND sigma > 0`, not the bare `i != 0` it was.
    # For every pre-A2A rung this is IDENTICAL (indices 1..33 all carry sigma >= 0.10), so no existing
    # stage moves.
    i: _v3_stage(spec[0], spec[1], i != 0 and spec[2] > 0.0,
                 sigma_ms=V3_SIGMA_BY_STAGE[i])
    for i, spec in enumerate(_V3_LADDER_SPEC)
}

# Pool is BINDING + INCLUSIVE (RC5 generator).
STAGE_NOMINAL_TIMESTEPS_V3: Dict[int, int] = {
    0: 60_000,
    **{i: 36_000 for i in range(1, 10)},
    # S-2: upper bound 24 -> 25 with the D1a insert (WS3/WS4/WS5 + D1a/D1/D2).
    **{i: 48_000 for i in range(10, 22)},
    22: 160_000,
    **{i: 48_000 for i in range(23, 34)},
    # A2A-20260901: same 48,000 nominal as every appended sibling.
    **{i: 48_000 for i in range(34, 40)},
}
STAGE_CAP_TIMESTEPS_V3: Dict[int, int] = {
    0: 120_000,
    **{i: 96_000 for i in range(1, 22)},
    22: 320_000,
    **{i: 72_000 for i in range(23, 27)},
    **{i: 72_000 for i in range(27, 34)},
    # A2A-20260901: six appended rungs, caps 72,000 each = 432,000, funded by the NAMED pool rider
    # A2ACAP-20260901 (paired 1:1).
    **{i: 72_000 for i in range(34, 40)},
}
# It is a NAMED LITERAL, not slack. Left implicit, the zero-slack pool identity (spend + ladder
# caps + reserve + extensions == effective pool) would simply stop closing, and a broken identity
# teaches nobody anything -- it just stops being checked.
CAP8_RELEASED_RESERVE_V3: int = 0

# Deliberately NOT folded into STAGE_CAP_TIMESTEPS_V3 / STAGE_NOMINAL_ TIMESTEPS_V3.
STAGE_EXTENSION_TIMESTEPS_V3: Dict[int, int] = {
    15: 189_994,
    18: 48_000,
    # Kept as a VISIBLE SUM rather than collapsed to 102,549 so an auditor sees TWO elections with two
    # bases, never one number that quietly grew -- the same discipline POOL_EXTENSION_TIMESTEPS_V3
    # enforces by keying on names. That is inherited from attempt 1 as amended, and it binds only
    # AFTER the exam has fired at the halfway point.
    20: 49_212 + 53_337 + 96_000 + 96_000 + 192_000 + 52_000 + 96_000 + 2_400_000,  # = 3,034,549 (S20EXT..S20EXT7 + WEEKPOOL)
    # P1a (leg 1) is a 96,000 plan -- freeze eats ~40k -- so its margin is 52,000. Paired with the
    # NOMMARGIN pool delta below.
    **{i: 4_000 for i in (24, 25, 32)},
    # Stage 23 is NOT here: its explicit 60,000 below already gives nominal_eff 108,000 > 92,000.
    26: 4_000 + 44_000,  # R15 -- CANDSSW second pass behind A2SW
    27: 4_000 + 44_000,  # L4
    28: 4_000 + 44_000,  # L2
    29: 4_000 + 44_000,  # L0
    33: 4_000 + 44_000,  # P2a
    # Leg 1 of this ladder unfroze at step 40,004 (convergence), so a 48,000 plan would leave ~8,000
    # learning steps; the plan is 96,000. Plan, not prior: the selfchecks derive a stage's PRIOR from
    # ^S31EXT-named riders, and there is none.
    31: 4_000 + 48_000 + 2_400_000,
    # Without this rider a fresh 96,000 re-fly reaches nominal_eff 100,000 at LOCAL step 62,048 and
    # gate-pauses 600 min. Visible SUM (S20EXT pattern): two elections, two bases. 52,000 = A3 nominal
    # margin (96,000 plan + 4,000) 51,222 = S30EXT: the voided attempt's BANKED consumption =>
    # nominal_eff = 48,000 + 89,952 = 151,222; the re-fly lands at 51,222 + 96,000 = 147,222 -- margin
    # EXACTLY 4,000, the ladder-wide invariant. => cap_eff = 72,000 + 89,952 = 175,222 (unreachable by
    # the plan).
    30: 52_000 + 51_222,
    # Not a climb. WHY 40,000 MORE THAN THE PLAN NEEDS: the actor freeze holds >= 40,000 of every
    # 96,000-step plan, so an arm buys only ~56,000 steps of real policy learning.
    19: 140_000,  # = 140,000 (S19EXT)
    # MECHANISM-ONLY extension for the leg-3 RE-FLY. The re-fly plans another 48,000, so F9 sees
    # 96,000 and the pre-extension nominal_eff of 52,000 refuses it (rc 5) -- correctly: the leg would
    # file a gate request at ~step 0 and gate-pause up to 600 minutes. nominal_eff = 48,000 + 52,000 =
    # 100,000, which leaves the house-standard F9 margin of EXACTLY 4,000 over the 96,000 the re-fly
    # reaches (see assert_budget's docstring: "every RERAMP stage carries a 4,000-step margin for
    # exactly this"). cap_eff = 72,000 + 52,000 = 124,000, unreachable by construction: the launcher
    # bounds the leg at --total-timesteps 48,000, so real consumption stops at 96,000.
    23: 60_000,
    # A2A-20260901: the SAME 4,000 nominal margin every appended sibling carries, so decide() cannot
    # file a gate request at plan end and idle the box (the S19EXT lesson). Six stages x 4,000 =
    # 24,000, paired 1:1 with the NAMED pool rider A2ANOM-20260901 -- if the stage half ever landed
    # without its pool half, the zero-slack identity in test_pool_extension is what refuses.
    **{i: 4_000 for i in range(34, 40)},
}

LEARNER_POOL_INCLUSIVE_V3: int = 1_796_000

# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
POOL_EXTENSION_TIMESTEPS_V3: Dict[str, int] = {
    "POOLRAISE-20260813": 683_829,
    "POOLRESERVE-20260813": 340_977,
    # Mechanism-only: the paired cap claim is unreachable (the s18b launcher bounds run-local steps to
    # land at the ORIGINAL 96,000 stage cap), so this delta funds zero actual steps by construction.
    "S18EXT-20260815": 48_000,
    "S20EXT-20260817": 49_212,
    # It is still not an allowance: decide() halts on the per-stage cap independently of the pool.
    "S20EXT2-20260819": 53_337,
    # Like S20EXT2, NOT mechanism-only: attempt 3 flies the full 96,000, so this delta funds real
    # steps and the paired cap claim is reachable.
    "S20EXT3-20260820": 96_000,
    "S20EXT4-20260821": 96_000,  # paired pool delta for the attempt-4 stage extension (standing rider)
    # NOT mechanism-only: both arms fly their full plans, so this delta funds real steps and the
    # paired cap claim is reachable.
    "S20EXT5-20260823": 192_000,
    # Spend and pool move in lockstep, so POOLRESERVE headroom is unchanged.
    "S19EXT-20260823": 140_000,
    # NOT mechanism-only: the leg flies its full plan, so this delta funds real steps.
    "TREMED-813-20260825": 52_000,
    # NOT mechanism-only: the funded legs fly their full plans, so this delta funds real steps.
    "GUARDSPIKE-818-20260825": 96_000,
    "S23EXT-20260825": 48_000,
    "S23EXT2-20260826": 8_000,
    # Deliberately NOT S31EXT-named: it is plan, not prior (see STAGE_EXTENSION above).
    "E2FREEZE-RERAMP811-20260826": 48_000,
    "WEEKPOOL-20260829": 4_800_000,
    # The four APPENDED stage caps (23..26) are NOT here — they are funded by CAP8_RELEASED_RESERVE_V3
    # going 288,000 -> 0, a transfer inside the identity, not a raise.
    "LOWWIND-RERAMP811": 288_000,  # A3: 4 x 72,000 (L4, L2, L0, L3)
    # PSTAGES (A3): the three re-fly cells on clean ledgers, 3 x 72,000.
    "PSTAGES-RERAMP811": 216_000,
    # NOMMARGIN (A3): 10 x 4,000 + 52,000 -- the gate-request margins above.
    "NOMMARGIN-RERAMP811": 92_000,
    # NOT mechanism-only: the re-fly flies its full 96,000, so this delta funds real steps.
    "S30EXT-RERAMP811": 51_222,
    # PAIRED 1:1 with the five +44,000 stage extensions above = 220,000. The identity is the
    # authority; a wrong premise stated plainly beats a tidy one.
    "CANDSSW-20260831": 220_000,
    # TWO terms, each paired 1:1 with the stage-side claim it funds, because they fund different
    # things and a single summed number would hide which: A2ACAP 6 x 72,000 = 432,000 -- the six
    # appended stage CAPS (34..39) A2ANOM 6 x 4,000 = 24,000 -- their nominal MARGINS (gate-pause
    # guard) Named, never widened: the launcher checks pool riders BY NAME, and the zero-slack
    # identity in test_pool_extension refuses if either half is missing.
    "A2ACAP-20260901": 432_000,
    "A2ANOM-20260901": 24_000,
}


def effective_pool_v3() -> int:
    """The binding inclusive pool: base + every SIGNED raise election.

    THE single definition — every consumer (decide()'s pool branch, the G12
    launch gate) calls this rather than summing for itself, so the runtime
    guard and the launch gate can never disagree about the ceiling.
    """
    return LEARNER_POOL_INCLUSIVE_V3 + sum(POOL_EXTENSION_TIMESTEPS_V3.values())
V3_GATE_TYPE: Dict[int, str] = {
    i: ("anchor30" if i in V3_ANCHOR_STAGES else "mid12")
    for i in range(MAX_STAGE_INDEX_V3 + 1)
}

# Escape hatch (rollback path, default absent): FD_V3_PROMOTION_ACK_DISABLE=1 restores auto-
# promotion; read at use, never cached.
V3_PROMOTION_ACK_STAGE_MIN: int = 6
V3_PROMOTION_ACK_DIRNAME: str = "promotion_acks"

# This gates PROMOTION ONLY — it never alters, re-scores, or suppresses a verdict.
V3_H1_ACK_MAX_EXCEED: int = 1
V3_H1_ACK_P95_MAX_M: float = 1.95

# Scoped to the V3 ladder ONLY: STAGE_WEIGHTS (v1) and STAGE_WEIGHTS_V2 keep w_precontact = 0.0,
# so every banked v1/v2 run stays byte-reproducible and no label is pooled across the definition
# change (feedback_never_pool_across_a_definition_change). ROLLBACK: set R1_W_PRECONTACT = 0.0 —
# one line, restores the inert ladder.
R1_W_PRECONTACT: float = 1.0

# ONE object shared by every v3 stage — the aliasing is the single-baseline invariant, not an
# accident, so it is built once and bound 26 times exactly as before. THE DEADBAND is the down-
# mid/down-hard boundary the measurement names.
VSAT_DEADBAND: float = 0.75
W_VSAT_A2A: float = 0.5


# Had they differed, (a) is the SAFER one: under (b) a rehearsal episode would be scored by a
# different rule than the rung it flies inside, handing the value function contradictory returns
# for similar states. SCOPE, stated plainly: this arms the term on EVERY v3 stage, not only the
# A2A rungs -- that is what (a) means and what the single-baseline lock intends (ONE reward
# definition for the v3 ladder).
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
_STAGE_WEIGHTS_V3_VECTOR: StageWeights = _dc_replace(
    STAGE_WEIGHTS[0], w_precontact=R1_W_PRECONTACT, w_vsat=W_VSAT_A2A)
# Every value is FLAT across the ladder, so extending the range adds no new tuning decision -- but
# leaving them at C1 would have left stages 23..26 with NO weights, NO safety range and NO dwell
# thresholds, i.e. a KeyError (or a silent v1 fallback) the first time a re-ramp leg launched.
# Detector: drl/test_curriculum_v3.py::test_flag_on_rebind covers every stage key.
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
STAGE_WEIGHTS_V3: Dict[int, StageWeights] = {
    i: _STAGE_WEIGHTS_V3_VECTOR for i in range(MAX_STAGE_INDEX_V3 + 1)}
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
STAGE_SAFETY_RANGE_V3: Dict[int, float] = {
    i: 300.0 for i in range(MAX_STAGE_INDEX_V3 + 1)}
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
DWELL_REWARD_THRESHOLD_S_V3: Dict[int, float] = {
    i: 0.2 for i in range(MAX_STAGE_INDEX_V3 + 1)}
# [FINAL SETUP: NOT USED, not read by runtime code; referenced only inside this file]
SUCCESS_DWELL_S_BY_STAGE_V3: Dict[int, float] = {
    i: float("inf") for i in range(MAX_STAGE_INDEX_V3 + 1)}

FD_CURRICULUM_V3: bool = _os.environ.get("FD_CURRICULUM_V3", "0") == "1"
if FD_CURRICULUM_V3 and FD_CURRICULUM_V2:
    raise RuntimeError(
        "FD_CURRICULUM_V2 and FD_CURRICULUM_V3 are mutually exclusive — unset one.")
if FD_CURRICULUM_V3:
    CURRICULUM_STAGES = dict(CURRICULUM_STAGES_V3)
    STAGE_WEIGHTS = dict(STAGE_WEIGHTS_V3)
    STAGE_SAFETY_RANGE = dict(STAGE_SAFETY_RANGE_V3)
    DWELL_REWARD_THRESHOLD_S = dict(DWELL_REWARD_THRESHOLD_S_V3)
    SUCCESS_DWELL_S_BY_STAGE = dict(SUCCESS_DWELL_S_BY_STAGE_V3)
    CUMULATIVE_TIMESTEPS = None  # gates are the SOLE advance authority (belt-and-braces)
    # V3 rides the v2-class ENV machinery for the GPS CEP realization (OU-GLTCH ladder):
    # env_formation_sitl.py:705 gates that machinery on THIS module attribute. The V2 TABLE rebind
    # above did NOT run (mutual exclusion), so flipping the attribute here activates only the env-side
    # ladder.
    FD_CURRICULUM_V2 = True

# Milestone tier thresholds for Stage 0 (precontact station-keeping quality). IMPORTANT: These
# measure error from the PRECONTACT TARGET (10m behind/below boom-tip), NOT from the boom-tip
# itself.
MILESTONE_THRESHOLDS: Dict[int, tuple] = {
    1: (2.0, 2.0, 2.5),  # Precontact coarse — within ±2m of hold point
    2: (1.5, 1.5, 1.5),  # Precontact medium — decent station-keeping
    3: (1.0, 0.8, 1.0),  # Precontact fine — good station-keeping
    4: (0.7, 0.5, 0.7),  # Precontact strict — tight station-keeping
}
# MILESTONE_NAMES removed in v15 audit — never referenced outside config.py

# =============================================================================
# [FINAL SETUP: APPLIES]
# TRUE FLIGHTDOCK CONTACT MILESTONES (boom-tip reference)
# =============================================================================
# These measure proximity to the actual boom-tip (target = 0,0,0 in the
# geometry frame), matching the baseline SITL harness (formation_success_metrics.py). The
# precontact MILESTONE_THRESHOLDS above measure error from the 10m standoff hold point — a
# different reference frame.

CONTACT_M3A_BEHIND_TOL: float = 1.2  # ±1.2m behind boom-tip
CONTACT_M3A_BELOW_TOL: float = 1.2  # ±1.2m below boom-tip
CONTACT_M3A_CROSS_TOL: float = 1.0  # ±1.0m cross-track
CONTACT_M3A_DWELL_S: float = 2.0  # 2.0s continuous

CONTACT_M3B_BEHIND_TOL: float = 0.7  # ±0.7m behind boom-tip
CONTACT_M3B_BELOW_TOL: float = 0.7  # ±0.7m below boom-tip
CONTACT_M3B_CROSS_TOL: float = 0.5  # ±0.5m cross-track
CONTACT_M3B_DWELL_S: float = 3.0  # 3.0s continuous

# Boom geometry (6.1m boom at 45° below leader CG)
BOOM_TIP_BEHIND_M: float = 4.313  # 6.1 * cos(45°)
BOOM_TIP_BELOW_M: float = 4.313  # 6.1 * sin(45°)

# M2 Precontact (boom-tip reference, NOT delta — follower targets 10m standoff)
ACTUAL_M2_BEHIND_TARGET: float = 10.0  # follower targets 10m behind boom-tip
ACTUAL_M2_BELOW_TARGET: float = 10.0  # follower targets 10m below boom-tip
ACTUAL_M2_BEHIND_TOL: float = 2.0  # ±2m
ACTUAL_M2_BELOW_TOL: float = 2.0  # ±2m
ACTUAL_M2_CROSS_MAX: float = 2.0  # <2m
ACTUAL_M2_DWELL_S: float = 3.0  # 3.0s continuous

# M3a Coarse (boom-tip delta)
CONTACT_M3A_COARSE_BEHIND_TOL: float = 2.0
CONTACT_M3A_COARSE_BELOW_TOL: float = 2.0
CONTACT_M3A_COARSE_CROSS_TOL: float = 1.5
CONTACT_M3A_COARSE_DWELL_S: float = 1.0

# =============================================================================
# [FINAL SETUP: APPLIES]
# POLICY INITIALIZATION (Approach G — zero-policy warm-start)
# =============================================================================
# In RRL, action=[0,0,0] means "follow baseline exactly". We initialize the
# policy to output near-zero actions with low exploration noise so that: 1. First-rollout episodes
# last the full 4500 steps (no SAFETY_RANGE triggers) 2.
LOG_STD_INIT: float = -0.7  # σ ≈ 0.50; entropy +0.42 nats (was -0.24 at -1.5)
ACTION_NET_GAIN: float = 0.01  # ortho_init gain for action_net (≈ zero output)

# =============================================================================
# [FINAL SETUP: APPLIES]
# PROCESS MANAGEMENT
# =============================================================================
PID_FILE_PATH: str = "/tmp/flightdock_training.pid"
HEARTBEAT_FILE_PATH: str = "/tmp/flightdock_heartbeat"

HEARTBEAT_PER_WORKER_TEMPLATE: str = "/tmp/flightdock_hb_env{env_id}"
# Watchdog threshold: 15 min (set via FLIGHTDOCK_WATCHDOG_STALE_MIN env var in
# training_watchdog.sh)

# Timeouts
SIGTERM_TIMEOUT_S: float = 10.0
VECENV_CLOSE_TIMEOUT_S: float = 30.0


# Positive = command LOWER (more clearance under the 2.00 m guard). Placed at END OF FILE
# deliberately: launchers assert the CRRL floor line BY NUMBER (config.py:308) at wheels-up, so
# nothing may be inserted above it.
VERT_OFFSET_DOWN_M: float = 0.0       # VERTOFF-20260902: 0.0 = disabled
