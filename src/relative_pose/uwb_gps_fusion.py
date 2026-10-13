"""
UWB + GPS Relative Pose Fusion Module

This module provides sensor fusion between GPS-derived relative positioning and
UWB (Ultra-Wideband) range measurements for accurate relative pose estimation
between the follower QuadPlane and the lead DHC-2 Beaver.

Created: 2025-11-26
Author: FlightDock Team
Reference: DEEP_DIVE_ANALYSIS_V4.md

Architecture:
    GPS (5 Hz) ──┬──► Kalman Filter ──► Fused (behind, cross, below)
    UWB (10 Hz) ─┘

The UWB sensor provides accurate range-only measurements that help correct
GPS multipath errors and improve cross-track accuracy during formation flight.
"""

import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# T-403 UWB gate sidecar (default-OFF, write-only). The three-way import is
# DEFENSIVE ON PURPOSE: the flight path imports THIS module BARE inside a
# `try/except ImportError` that silently degrades the controller to GPS-only
# geometry (tests/baseline_formation_controller.py:82-91, tests/geometry.py:35-49).
# An ImportError escaping from here would therefore disable UWB fusion on EVERY
# leg with nothing but a printed warning. It must be impossible to raise.
try:
    from . import uwb_gate_diag as _gate_diag      # package import (tests, tools)
except ImportError:
    try:
        import uwb_gate_diag as _gate_diag         # bare import (flight path)
    except ImportError:
        _gate_diag = None                          # absent -> feature simply off

# Configure logging
logger = logging.getLogger(__name__)


def _valid_sigma_axes(v) -> bool:
    """C1 (T-358): exactly 3 finite, strictly-positive floats. Anything else
    is refused -- a partial or non-positive noise model must never half-apply
    (a non-positive R can drive S<=0 and a degenerate gain)."""
    try:
        if v is None or len(v) != 3:
            return False
        return all(math.isfinite(float(x)) and float(x) > 0.0 for x in v)
    except (TypeError, ValueError):
        return False


def _parse_gps_sigma_axes(raw: str):
    """C1 (T-358): parse FD_FUSION_GPS_SIGMA_AXES='behind,cross,below'.
    Empty/unset -> None (default scalar path, byte-identical). Malformed ->
    None + ONE loud log line. Never raises: a bad env var must not take down
    the controller, and must not silently half-apply either."""
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    try:
        vals = tuple(float(p) for p in parts)
    except (TypeError, ValueError):
        vals = None
    if not _valid_sigma_axes(vals):
        logger.error(
            "C1 HONEST-R: FD_FUSION_GPS_SIGMA_AXES=%r is not 3 positive "
            "comma-separated floats (behind,cross,below) -- FAILING CLOSED to "
            "the scalar gps_sigma_m default",
            raw,
        )
        return None
    return vals


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Default measurement noise standard deviations
GPS_POS_SIGMA_M = 2.0  # GPS position noise [m]
GPS_VEL_SIGMA_M_S = 0.3  # GPS velocity noise [m/s]
UWB_RANGE_SIGMA_M = 0.10  # UWB range noise [m] assumed-R (substrate re-tune 2026-06-28; conservative vs 0.05 emitter true sigma; DW1000/DW3000)

# Process noise (model uncertainty)
PROCESS_ACCEL_SIGMA_M_S2 = 1.0  # Process acceleration noise [m/s^2]

# Innovation gate (chi-squared threshold for outlier rejection)
INNOVATION_GATE_SIGMA = 3.0  # Reject measurements > 3 sigma from prediction

# ADAPTIVE PROCESS NOISE (ADDED 2025-11-27 per intercept_rejoin analysis)
# During rapid closure, increase process noise so filter tracks faster
# UPDATED 2025-11-27: Lowered threshold from 3.0 to 2.0 for earlier Q boost
ADAPTIVE_Q_RANGE_RATE_THRESH = 2.0  # m/s - if |Vc| > this, boost Q (was 3.0)
ADAPTIVE_Q_BOOST_FACTOR = 3.0  # UPDATED 2025-11-27: Increased from 2.0 to reduce ~14m bias lag

# DWNA Q-RECALIBRATION knob (#789) -- REFUTED [MEASURED 2026-06-18]; retained default-OFF
# as the reproducible instrument for the refutation (and an optional SITL leg). The spine-
# collapse root is multi-axis process-noise (Q) STARVATION (q_pos=0.5*sigma^2*dt^4 with
# process_accel_sigma=1.0 is ~5e-5 m^2/step -- too small to balance the (1-K) GPS shrink
# vs R=4.0, so P->0, K->0, UWB gated out). The research-first PRIMARY (wrp72qzzb) proposed
# repairing the Q MODEL by raising process_accel_sigma ("sigma-raise") to land steady-state
# GPS gain K in [0.1,0.3]. The MANDATED offline covariance trace (tools/qrecal_calibrate.py)
# REFUTES this at the faithful ~20 Hz tick: the GPS update independently observes VELOCITY
# (R_vel=0.3^2=0.09), pinning P_vel, so the velocity->position re-inflation is strangled and
# only the dt^4 q_pos term reaches position -> reaching K=0.15 needs sigma_a = 139-259 m/s^2
# (14-26 g) across the 20->14.66 Hz envelope -- physically absurd for a formation QuadPlane
# (maneuvers ~0.1-1 g). The ADOPTED lever is instead the additive position random-walk
# pos_rw_m2_s (injects position variance DIRECTLY, ~dt; K=0.18 in band; dt-robust). This
# knob stays for reproducibility; the constant below is in the REFUTED region (K~0.12<<band)
# -- no physical sigma_a works. qrecal-OFF (default) -> the legacy 0.5 coeff +
# process_accel_sigma path is byte-identical. Env FD_FUSION_QRECAL_ENABLE /
# FD_FUSION_QRECAL_ACCEL_SIGMA when the constructor args are None.
QRECAL_ACCEL_SIGMA_DEFAULT = 200.0  # m/s^2 -- REFUTED region (K~0.12 << band); no viable value

# ADAPTIVE-β (clean-margin A1, FD_UWB_BETA_ADAPT_ENABLE; default-ON, adopted baseline).
# Drives the PUSKF per-axis update-fraction β DYNAMICALLY so ONE estimator wins BOTH
# regimes (A0 calibration `analysis/beta_adapt_a0_20260620T020227Z`): a CUSUM detector
# on the BEHIND GPS-vs-UWB disagreement m=(-innov)*H[behind] gates cross/below β; behind
# β=1 always (corrects any bias). CLEAN -> detector quiet -> cross/below β=0 (=β=0 clean
# win, smooth); DEGRADED -> behind bias trips it -> cross/below β->1 (=β=1 degraded win,
# scrub, no overtake). A0 [MEASURED]: clean ~0% engage / degraded-overshoot 37% /
# uncorrected 100%. Uses a CUSUM detector + tick-count ramp + hysteresis;
# TICK-COUNT ramp (no dt/wall-clock/random) -> lockstep bit-deterministic. The per-axis
# design was REFUTED in A0 (cross/below own-signal too small at degraded); gating cross/
# below off the behind detector is the correct coupling.
# A1' SIGNAL = the PERSISTENT bias s=|fused_behind - last_gps_behind| (NOT the innovation;
# A2 NO-GO showed the innovation-CUSUM never fired live — behind β=1 scrubs it small. The
# persistent bias reflects what the fuser corrects, stays large at degraded regardless of
# scrub -> fires at onset + HOLDS). One-sided CUSUM (s is a magnitude >= 0). A0'-calibrated
# (analysis/beta_adapt_a0p_20260620T062550Z): clean s_med ~0.55, degraded s_med ~1.38 ->
# K between them; degraded engage 84-89% HELD, clean ~5%.
BETA_ADAPT_K = 0.9          # CUSUM slack on |bias| (clean s_med 0.55 < K < degraded 1.38)
BETA_ADAPT_H = 3.0          # CUSUM alarm threshold (latch)
BETA_ADAPT_HYST = 0.5       # de-latch when CUSUM < HYST*H (hysteresis vs chatter)
BETA_ADAPT_CAP = 8.0        # CUSUM saturation (bounds de-latch time)
BETA_ADAPT_CLAMP_M = 3.0    # per-sample |bias| clamp [m] (reject huge GPS-glitch spikes)
BETA_ADAPT_RAMP_UP = 5      # tick-count ramp UP (~0.25 s @20 Hz) -- engage cross/below FAST on
                            # detection (A1''': 20->5; cuts the onset low-β window. A0''' offline:
                            # clean-safe (engage 4.2%->4.6%), gate already holds at dwell; the EV
                            # for the s7 dwell-drift is low, A2''' is the empirical test.)
BETA_ADAPT_RAMP_DN = 40     # tick-count ramp DOWN (slower -> hold through transient scrub)
BETA_ADAPT_WARMUP = 20      # ticks held at β_min before the detector arms (cold-start guard)
BETA_ADAPT_CB_MIN = 0.0     # cross/below β floor at clean (=β=0 clean win); behind β=1 fixed

# Cross/below decouple from the behind CUSUM (FD_UWB_BETA_DECOUPLE_CB; default-OFF, additive).
# When ON, _beta_adapt_step holds BELOW β=0 (Schmidt-exclude the range update from the well-
# observed vertical axis) and caps CROSS β at BETA_DECOUPLE_CROSS_CAP * gate * |dy/r| instead of
# ramping both cross+below to 1. Targets the [MEASURED] below-axis range ratchet that false-fires
# the -2.0 m altitude-ceiling guard under degraded GPS (cycle_driver_rca). OFF -> verbatim
# adaptive-β (byte-identical). FM8: cap env-clamped to [0,1] in __init__.
BETA_DECOUPLE_CROSS_CAP = 0.15

# Behind sign-lock guard (FD_UWB_BEHIND_SIGNLOCK_GUARD; DEFAULT-ON since v10 adoption 2026-07-02
# — accept-trade 36-leg net-win `54a21581`: clean-S0 phantom overtakes 2/6->0/12, wind-inert;
# =0 restores the byte-identical no-guard path; additive). The degraded-GPS
# terminal dock-then-overtake is a UWB range-only SIGN-LOCK on the behind axis (RCA
# behind_signlock_rca_20260628 + DESIGN_CHECK_Q1Q2): near dock the range is sign-blind, so a
# wrong-sign fused_behind is reinforced by the range update and overpowers the sign-correct GPS
# ~180-400x. When ON, _behind_signlock_step tapers the behind PUSKF update-fraction 1->floor across
# [R_OUTER, R_INNER] while GPS is sign-confident (|gps_behind| > GPS_MIN) and disagrees with the
# state behind sign, restoring GPS sign authority; behind beta=1 otherwise (UWB still corrects the
# ~0.5 m along-track bias). OFF -> step never called -> verbatim (byte-identical). FM10: radii/floor
# clamped in __init__.
BEHIND_SIGNLOCK_R_INNER_M = 2.5   # full guard (env=1, behind beta->floor) at predicted_range <= this [m]
BEHIND_SIGNLOCK_R_OUTER_M = 4.0   # guard fully relaxed (behind beta=1) at predicted_range >= this [m]
BEHIND_SIGNLOCK_GPS_MIN_M = 0.3   # min |gps_behind| [m] to trust the GPS sign (reject near-zero noise)
BEHIND_SIGNLOCK_BETA_FLOOR = 0.0  # behind update-fraction floor under full guard (0 = full Schmidt-exclude)

# Surge gate-relief (FD_UWB_SURGE_GATE_RELIEF; default-OFF, additive). #352b [MEASURED]: during
# surge the over-confident converged EKF rejects ~70% of range updates that are correctly signed to
# fix the behind under-read, withholding ~1.15 m (~11σ) corrections; rejections are chronic-
# intermittent (max-run 2), NOT a sustained lockout. When ON, a rejection-rate CUSUM (+1 on a
# prior-tick reject, -1 on accept; prior outcome read from the uwb_rejected/update counters -- no
# gate hook) with an innovation-SIGN-consistency guard arms during chronic consistent rejection
# near dock and inflates the BEHIND-position P to a floor so the 3σ gate admits the corrective range
# (raises S -> gate opens; raises K -> fast catch-up). Inflates P, NOT R -> gain PRESERVED (distinct
# from #351's REFUTED R-inflation, which collapsed the gain). Additive diagonal max() -> PSD-safe;
# bounded by the target -> can't grow unbounded; self-de-arms as accepts resume (hysteresis). Pairs
# with FD_UWB_CONTACT_RANGE_EXCLUDE (C confines the admitted range to behind). =0 -> step never
# called -> byte-identical. Count/geometry only -> lockstep bit-deterministic. See F-FUSER-4-CEIL.
SURGE_RELIEF_ARM = 3.0        # CUSUM arm threshold (~3 net rejections; ~0.4 s @ 70% reject rate)
SURGE_RELIEF_HYST = 0.5       # de-arm when CUSUM < HYST*ARM (hysteresis vs chatter)
SURGE_RELIEF_CAP = 8.0        # CUSUM + sign-sum saturation (bounds de-latch/flip time)
SURGE_RELIEF_SIGN = 2.0       # min |sign-sum| to confirm consistent-direction drift (vs symmetric noise)
SURGE_RELIEF_WARMUP = 20      # UWB ticks before the detector arms (cold-start guard)
SURGE_RELIEF_P_POS = 0.25     # behind-position variance floor [m^2] when armed (3σ gate ~1.5 m; K~0.96)

# Behind ring-throttle (FD_UWB_BEHIND_RING_THROTTLE; default-OFF="0", "C"=Form-C CUSUM taper,
# "S"=Form-S stateless innovation-saturation). #405 D2 [MEASURED redirect, D1r 726a4111]: the
# accepted aft-bias re-injection that SUSTAINS the surge rides behind-POSITION beta[0] ~2:1 over
# behind-VELOCITY beta[3] (both signed +aft) -> throttle BOTH behind gains together (beta[3]-alone
# REFUTED as sufficient). A signed CUSUM (Form-C) or stateless saturation (Form-S) on the accepted
# aft innovation near dock computes a CONTINUOUS robust weight w in [FLOOR, 1] applied as
# min(beta[i], w) to behind pos+vel (IDX_X, IDX_VX) ONLY -- NEVER cross/below, NEVER the gate,
# NEVER _predict. Composes down monotone (min) with sign-lock/adaptive-beta. FLOOR>0 (never full
# Schmidt-exclude). Reuses the sign-lock r_outer dock-band gate (no new range const). Deterministic
# (tick/geometry only, RNG/clock-free). See analysis/channeliso_*/D1R_channel_iso.md + the verified
# actuator precedents (Fang/Haile/Wang innovation-saturation; Page CUSUM).
RING_THROTTLE_K = 0.03        # CUSUM slack on the aft innovation [m] (< D1r dock-band aft med 0.086)
RING_THROTTLE_ARM = 1.0       # Form-C CUSUM arm threshold (sustained aft over-read)
RING_THROTTLE_CAP = 10.0      # CUSUM saturation (bounds de-latch time)
RING_THROTTLE_HYST = 0.5      # de-arm when CUSUM < HYST*ARM (hysteresis vs chatter)
RING_THROTTLE_WARMUP = 10     # in-band UWB ticks before the Form-C detector arms (cold-start guard)
RING_THROTTLE_RAMP_UP = 8.0   # Form-C tick-ramp up (gate 0->1 over N armed ticks)
RING_THROTTLE_RAMP_DN = 16.0  # Form-C tick-ramp down
RING_THROTTLE_FLOOR = 0.2     # behind beta floor when fully throttled (>0; keeps some observability)
RING_THROTTLE_SAT_C = 0.05    # Form-S saturation numerator [m]: w = clip(C/|innov|, FLOOR, 1)

# F-DAMP REFUTED [MEASURED 2026-07-13 #404]: fires (dock-band gate on 78-91% of dock ticks) but is
# OVERPOWERED by measurement re-injection — fused-behind surge UNMOVED (std 1.48->1.497), dwell
# unlifted (0/6 both arms). The surge is sustained in the MEASUREMENT-ACCEPTANCE loop (#349/#351),
# NOT the CV velocity extrapolation. Retained default-OFF as the reproducible refutation instrument
# (precedent FD_FUSION_QRECAL_ENABLE). See analysis/surge_fdamp_ab_20260713T035348Z/FDAMP_PILOT_FINDING.md.
# Behind-velocity dock-band damping (F-DAMP; FD_UWB_BEHIND_VDAMP_ENABLE; default-OFF). #403
# [MEASURED]: the goal-1 pass/fail discriminator is the post-arrival aft SURGE (mode-B n_eps>500 =>
# 0/20 dwell), a ~1.5-2 Hz BEHIND-axis limit cycle the CV model (x += vx*dt) extrapolates from a
# ringing behind-velocity; the fuser is the SUSTAINER (#349/#402). tau ~0.4 s damps that cycle
# (period ~0.5-0.7 s) while preserving sub-Hz DC creep; r_band reuses the sign-lock outer envelope
# (the dock band). See #404 PREREG_FDAMP_AB + #403 DISCRIM2_FINDING.
BEHIND_VDAMP_TAU_S_DEFAULT = 0.4    # first-order behind-velocity decay time-constant [s] (a=exp(-dt/tau))
BEHIND_VDAMP_R_BAND_M_DEFAULT = 4.0  # damp only when est. slant range <= this [m] (dock band; = signlock outer)

# Minimum update rate thresholds
MIN_GPS_UPDATE_HZ = 2.0  # Warn if GPS slower than this
MIN_UWB_UPDATE_HZ = 5.0  # Warn if UWB slower than this

# State vector indices
IDX_X = 0  # Position X (behind) in leader NED frame
IDX_Y = 1  # Position Y (cross) in leader NED frame
IDX_Z = 2  # Position Z (below) in leader NED frame
IDX_VX = 3  # Velocity X
IDX_VY = 4  # Velocity Y
IDX_VZ = 5  # Velocity Z


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class GPSMeasurement:
    """GPS-derived relative position measurement."""

    timestamp: float  # Seconds in caller timebase (sim-time)
    behind_m: float  # Along-track (positive = follower behind leader)
    cross_m: float  # Cross-track (positive = follower right of leader track)
    below_m: float  # Vertical (positive = follower below leader)
    vel_behind_m_s: float = 0.0  # Relative velocity along-track
    vel_cross_m_s: float = 0.0  # Relative velocity cross-track
    vel_below_m_s: float = 0.0  # Relative velocity vertical


@dataclass
class UWBMeasurement:
    """UWB range-only measurement."""

    timestamp: float  # Seconds in caller timebase (sim-time)
    range_m: float  # Slant range to leader [m]
    anchor_id: int = 0  # Anchor ID (for multi-anchor systems)


@dataclass
class FusedState:
    """Fused relative pose estimate."""

    timestamp: float  # Seconds in caller timebase (sim-time)
    behind_m: float  # Fused along-track position
    cross_m: float  # Fused cross-track position
    below_m: float  # Fused vertical position
    vel_behind_m_s: float  # Fused along-track velocity
    vel_cross_m_s: float  # Fused cross-track velocity
    vel_below_m_s: float  # Fused vertical velocity
    range_m: float  # Computed slant range
    covariance: List[float] = field(default_factory=lambda: [0.0] * 36)  # 6x6 flattened
    gps_age_s: float = 0.0  # Time since last GPS update
    uwb_age_s: float = 0.0  # Time since last UWB update
    innovation_range: float = 0.0  # Last UWB innovation (for diagnostics)


# ==============================================================================
# RELATIVE POSE FUSION CLASS
# ==============================================================================


class RelativePoseFusion:
    """
    Kalman filter for fusing GPS-derived relative position with UWB range measurements.

    The filter maintains a 6-state vector [x, y, z, vx, vy, vz] representing
    the follower's position and velocity relative to the leader in the leader's
    NED frame.

    GPS provides direct position/velocity updates (3D).
    UWB provides range-only updates (nonlinear, linearized via EKF).

    Usage:
        fusion = RelativePoseFusion()
        fusion.update_gps(gps_measurement)
        fusion.update_uwb(uwb_measurement)
        state = fusion.get_state()
    """

    def __init__(
        self,
        gps_sigma_m: float = GPS_POS_SIGMA_M,
        gps_vel_sigma_m_s: float = GPS_VEL_SIGMA_M_S,
        uwb_sigma_m: Optional[float] = None,
        process_accel_sigma: float = PROCESS_ACCEL_SIGMA_M_S2,
        anchor_positions: Optional[Dict[int, Tuple[float, float, float]]] = None,
        pos_rw_m2_s: Optional[float] = None,
        qrecal_enable: Optional[bool] = None,
        qrecal_accel_sigma: Optional[float] = None,
        puskf_beta_cross: Optional[float] = None,
        puskf_beta_below: Optional[float] = None,
        gps_sigma_axes_m: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Initialize the fusion filter.

        Args:
            gps_sigma_m: GPS position measurement noise std dev [m]
            gps_vel_sigma_m_s: GPS velocity measurement noise std dev [m/s]
            uwb_sigma_m: UWB range measurement noise std dev (assumed-R) [m]. None (default)
                         -> env FD_UWB_RANGE_SIGMA_M else module default UWB_RANGE_SIGMA_M; an
                         explicit non-None value takes precedence (env ignored). Floored at 0.01.
            process_accel_sigma: Process model acceleration noise [m/s^2]
            anchor_positions: Optional dict mapping anchor_id to (behind_m, cross_m, below_m)
                              position offsets in leader frame. Anchor 0 defaults to origin.
        """
        # Measurement noise parameters
        self.gps_sigma_m = gps_sigma_m
        self.gps_vel_sigma_m_s = gps_vel_sigma_m_s
        # UWB assumed range-R env override (FD_UWB_RANGE_SIGMA_M) used when the arg is None (the
        # controller instantiates with no args, baseline_formation_controller.py ~1477). Default
        # str(UWB_RANGE_SIGMA_M) -> byte-identical when unset. The substrate-retune RCA
        # (RETUNE_CHARACTERIZATION_perAxis) showed the assumed-R drop 0.15->0.10 ~tripled the
        # terminal BEHIND fused jitter (higher UWB gain on the β=1 behind axis -> episodic +1..4 m
        # behind excursions -> phantom-gap over-close); this env exposes R for the gated A/B retune
        # sweep WITHOUT touching the R7.1-frozen controller. Explicit arg keeps precedence (env
        # ignored). Floor 0.01 m so R>0 -> S (innovation cov, update_uwb line ~702) never
        # degenerate. SEPARATE lever from POS_RW/QRECAL/PUSKF/adaptive-β (FM9 -- never collapse).
        if uwb_sigma_m is None:
            try:
                uwb_sigma_m = float(os.environ.get("FD_UWB_RANGE_SIGMA_M", str(UWB_RANGE_SIGMA_M)))
            except (TypeError, ValueError):
                uwb_sigma_m = UWB_RANGE_SIGMA_M
        self.uwb_sigma_m = max(0.01, uwb_sigma_m)
        self.process_accel_sigma = process_accel_sigma

        # All-axis constant process-noise recalibration (Q-recal, task #789). Default
        # OFF (0.0) -> no additive term -> byte-identical. Additive ALL-AXIS position
        # random-walk PSD [m^2/s] applied in _predict(): P[i][i] += pos_rw_m2_s*dt on
        # all 3 position axes, sustaining P so K cannot collapse to ~0 under repeated
        # GPS updates (the multi-axis Q-starvation root; fuser_rootcollapse RCA). This
        # is the all-axis SIBLING of qrw_behind (behind-only/legacy) -- a SEPARATE
        # addend, never collapsed into it (FM9). Calibrated delta_q_cal=3.0 -> steady-
        # state K~0.18 (in band [0.1,0.3]) vs R=4.0 across the measured 20->14.66 Hz tick
        # envelope (offline-confirmed tools/qrecal_calibrate.py). This DIRECT position
        # injection (~dt, dt-robust) is the ADOPTED spine-collapse lever; the DWNA sigma-
        # raise alternative is REFUTED -- the GPS update observes velocity so sigma cannot
        # reach position (see QRECAL_ACCEL_SIGMA_DEFAULT). Env override FD_FUSION_POS_RW_M2_S
        # used when the arg is None (the controller instantiates with no args).
        if pos_rw_m2_s is None:
            try:
                pos_rw_m2_s = float(os.environ.get("FD_FUSION_POS_RW_M2_S", "0.0"))
            except (TypeError, ValueError):
                pos_rw_m2_s = 0.0
        self.pos_rw_m2_s = max(0.0, pos_rw_m2_s)

        # DWNA Q-recalibration (#789 PRINCIPLED spine-collapse fix). Default OFF ->
        # byte-identical (the legacy 0.5 q_pos coeff + process_accel_sigma base path in
        # _predict is untouched). When ON, _predict uses the textbook 0.25 coeff and the
        # calibrated accel sigma so the steady-state GPS gain lands K in [0.1,0.3] vs
        # R=4.0 -- repairing the BASE Q MODEL (the Q-starvation root) rather than adding a
        # patch. This is a SEPARATE path from the additive qrw_behind / pos_rw / spine-
        # floor terms (FM9 -- never collapse the levers into one). Env overrides
        # FD_FUSION_QRECAL_ENABLE / FD_FUSION_QRECAL_ACCEL_SIGMA used when the args are None
        # (the controller instantiates no-args). See QRECAL_ACCEL_SIGMA_DEFAULT.
        if qrecal_enable is None:
            qrecal_enable = os.environ.get("FD_FUSION_QRECAL_ENABLE", "0").strip() == "1"
        self._qrecal_enabled = bool(qrecal_enable)
        if qrecal_accel_sigma is None:
            try:
                qrecal_accel_sigma = float(
                    os.environ.get("FD_FUSION_QRECAL_ACCEL_SIGMA", str(QRECAL_ACCEL_SIGMA_DEFAULT))
                )
            except (TypeError, ValueError):
                qrecal_accel_sigma = QRECAL_ACCEL_SIGMA_DEFAULT
        self._qrecal_accel_sigma = max(0.0, qrecal_accel_sigma)

        # C1 HONEST-R (T-358; election C1-HONEST-R-20260809, dose x1.5 =
        # 0.65/0.25/0.05). Per-axis GPS position R in the state's own order
        # (behind, cross, below) -- update_gps builds z = [behind, cross, below,
        # ...], so index i maps 1:1. Default OFF (None + env unset) -> R stays
        # diag(gps_sigma_m**2) and the path is BYTE-IDENTICAL; the sigma=2.0 /
        # R=4.0 default pins in test_uwb_gps_fusion_psd_invariants.py are the
        # detector for that. MEASURED basis: GPS-input sigma is 0.031 (below) /
        # 0.36-0.42 (behind) / 0.167 (cross) against a DECLARED 2.0 -- R
        # overstated 4138x on below, so the filter discards ~99% of a good fix
        # (K_gps ~ 0.01) and rides a UWB stream whose innovations are its own
        # drift (E3C_RCA.json: generator clean at +0.002 m, 6/6 legs). Explicit
        # arg takes precedence (env ignored). SEPARATE lever from
        # POS_RW/QRECAL/PUSKF/uwb_sigma -- never collapse them (FM9).
        # FAIL-CLOSED: any malformed/partial/non-positive value falls back to
        # the scalar default with ONE loud log line. A half-applied noise model
        # is worse than none.
        if gps_sigma_axes_m is None:
            gps_sigma_axes_m = _parse_gps_sigma_axes(
                os.environ.get("FD_FUSION_GPS_SIGMA_AXES", "").strip()
            )
        elif not _valid_sigma_axes(gps_sigma_axes_m):
            logger.error(
                "C1 HONEST-R: gps_sigma_axes_m=%r is not 3 positive floats "
                "(behind, cross, below) -- FAILING CLOSED to scalar "
                "gps_sigma_m=%.3f",
                gps_sigma_axes_m,
                gps_sigma_m,
            )
            gps_sigma_axes_m = None
        self.gps_sigma_axes_m: Optional[Tuple[float, float, float]] = (
            tuple(float(v) for v in gps_sigma_axes_m) if gps_sigma_axes_m else None
        )
        if self.gps_sigma_axes_m is not None:
            logger.info(
                "C1 HONEST-R ACTIVE: per-axis GPS R = (behind %.3f, cross %.3f, "
                "below %.3f) m; scalar gps_sigma_m=%.3f is OVERRIDDEN for "
                "position R only (velocity R untouched)",
                *self.gps_sigma_axes_m,
                gps_sigma_m,
            )

        # Partial-Update Schmidt-Kalman (PUSKF) per-axis UWB-update fraction (clean-margin
        # H2a; default-OFF byte-identical). The all-axis POS_RW re-inflation (above) keeps
        # cross/below covariance non-trivial, so the range-only UWB accept (update_uwb)
        # gets a Kalman-gain foothold on those WELL-observed axes; the sigma-0.1 m range
        # noise then jitters cross/below 8-26x more than raw GPS, fragmenting terminal
        # dwell (H0 RCA clean_margin_rca_20260619). Brink 2017 partial-update (AIAA JGCD
        # 10.2514/1.G002808) applies only a FRACTION beta of the measurement update per
        # state. Here INVERSE orientation: throttle the WELL-observed cross/below axes
        # (beta<1) while keeping the weakly-observed BEHIND axis at FULL update (beta=1)
        # so the ~0.5 m along-track UWB correction (L-A) is preserved. UWB update ONLY --
        # the GPS update (update_gps) is untouched, so cross/below stay GPS-tracked/smooth.
        # Per-axis beta covers position AND that axis's velocity state (else throttled-axis
        # velocity jitter leaks back into position via predict). Static beta (deterministic
        # -> lockstep/RL-safe; online observability-adaptive beta deferred). Env override
        # FD_UWB_PUSKF_BETA_CROSS / _BELOW used when the arg is None (the controller
        # instantiates with no args). Default 1.0 -> full update on every state -> the OFF
        # branch in update_uwb runs the verbatim original path (byte-identical). SEPARATE
        # lever from POS_RW/QRECAL/adaptive-β (FM9 -- never collapse).
        # See H1_RESEARCH_FINDINGS.md.
        if puskf_beta_cross is None:
            try:
                puskf_beta_cross = float(os.environ.get("FD_UWB_PUSKF_BETA_CROSS", "1.0"))
            except (TypeError, ValueError):
                puskf_beta_cross = 1.0
        if puskf_beta_below is None:
            try:
                puskf_beta_below = float(os.environ.get("FD_UWB_PUSKF_BETA_BELOW", "1.0"))
            except (TypeError, ValueError):
                puskf_beta_below = 1.0
        _bc = min(1.0, max(0.0, puskf_beta_cross))
        _bb = min(1.0, max(0.0, puskf_beta_below))
        # Per-state update fractions [behind, cross, below, vbehind, vcross, vbelow]:
        # behind pos+vel ALWAYS full (1.0); cross/below pos+vel throttled by their beta.
        self._puskf_beta = [1.0, _bc, _bb, 1.0, _bc, _bb]
        # Enabled only when a throttle is actually requested; OFF -> verbatim original path.
        self._puskf_enabled = (_bc < 1.0) or (_bb < 1.0)

        # Adaptive-β (A1; default-ON, the adopted single-estimator baseline). When
        # enabled, a behind CUSUM detector drives self._puskf_beta per UWB tick (see
        # _beta_adapt_step + the BETA_ADAPT_* constants). Set FD_UWB_BETA_ADAPT_ENABLE=0
        # to disable -> _puskf_beta stays the static PUSKF value above (default
        # [1,1,1,1,1,1] -> _puskf_enabled False -> verbatim original update). State init
        # unconditionally (cheap; zeroed in reset()). NOTE: when adaptive is ON it
        # OVERRIDES the static PUSKF betas (drives them from the behind disagreement signal).
        self._beta_adapt_enabled = os.environ.get("FD_UWB_BETA_ADAPT_ENABLE", "1").strip() == "1"
        # Cross/below decouple (FD_UWB_BETA_DECOUPLE_CB; default-OFF, additive). ON -> below β=0
        # + cross β capped at cross_cap*gate*|dy/r|; OFF -> verbatim adaptive-β actuator. Config:
        # read ONCE here, never per-tick. See _beta_adapt_step + BETA_DECOUPLE_CROSS_CAP.
        self._beta_decouple_cb_enabled = (
            os.environ.get("FD_UWB_BETA_DECOUPLE_CB", "0").strip() == "1"
        )
        try:
            self._beta_decouple_cross_cap = float(
                os.environ.get("FD_UWB_BETA_DECOUPLE_CROSS_CAP", str(BETA_DECOUPLE_CROSS_CAP))
            )
        except (TypeError, ValueError):
            self._beta_decouple_cross_cap = BETA_DECOUPLE_CROSS_CAP
        # FM8 clamp: β in [0,1] (β>1 over-corrects past full update; β<0 inverts the correction).
        self._beta_decouple_cross_cap = min(1.0, max(0.0, self._beta_decouple_cross_cap))
        self._beta_cusum = 0.0     # one-sided CUSUM on |bias| (A1' persistent-bias signal)
        self._beta_active = False
        self._beta_gate = 0.0      # tick-count ramp 0..1; cross/below β = CB_MIN + (1-CB_MIN)*gate
        self._beta_count = 0       # UWB-tick counter (warmup guard)
        # Behind sign-lock guard (FD_UWB_BEHIND_SIGNLOCK_GUARD; v10 DEFAULT-ON, additive). ON -> when
        # GPS is sign-confident and disagrees with the state behind sign near dock, taper the behind
        # PUSKF update-fraction 1->floor across [r_outer, r_inner] (Schmidt-exclude the UWB range
        # from behind), restoring sign-correct GPS authority; =0 -> _behind_signlock_step never
        # called -> _puskf_beta untouched (byte-identical prior path). Config read ONCE here, never
        # per-tick. See _behind_signlock_step + the BEHIND_SIGNLOCK_* constants. RCA behind_signlock_rca.
        self._behind_signlock_enabled = (
            os.environ.get("FD_UWB_BEHIND_SIGNLOCK_GUARD", "1").strip() == "1"
        )
        try:
            self._signlock_r_inner = float(
                os.environ.get("FD_UWB_SIGNLOCK_R_INNER_M", str(BEHIND_SIGNLOCK_R_INNER_M))
            )
        except (TypeError, ValueError):
            self._signlock_r_inner = BEHIND_SIGNLOCK_R_INNER_M
        try:
            self._signlock_r_outer = float(
                os.environ.get("FD_UWB_SIGNLOCK_R_OUTER_M", str(BEHIND_SIGNLOCK_R_OUTER_M))
            )
        except (TypeError, ValueError):
            self._signlock_r_outer = BEHIND_SIGNLOCK_R_OUTER_M
        try:
            self._signlock_gps_min = float(
                os.environ.get("FD_UWB_SIGNLOCK_GPS_MIN_M", str(BEHIND_SIGNLOCK_GPS_MIN_M))
            )
        except (TypeError, ValueError):
            self._signlock_gps_min = BEHIND_SIGNLOCK_GPS_MIN_M
        try:
            self._signlock_floor = float(
                os.environ.get("FD_UWB_SIGNLOCK_BETA_FLOOR", str(BEHIND_SIGNLOCK_BETA_FLOOR))
            )
        except (TypeError, ValueError):
            self._signlock_floor = BEHIND_SIGNLOCK_BETA_FLOOR
        # FM10 clamps: non-negative radii, strict r_inner < r_outer, beta floor in [0,1].
        self._signlock_r_inner = max(0.0, self._signlock_r_inner)
        self._signlock_r_outer = max(self._signlock_r_inner + 1e-3, self._signlock_r_outer)
        self._signlock_gps_min = max(0.0, self._signlock_gps_min)
        self._signlock_floor = min(1.0, max(0.0, self._signlock_floor))
        self._last_gps_behind: Optional[float] = None  # raw GPS behind for the |bias| signal
        # Contact-range cross/below authority exclude (FD_UWB_CONTACT_RANGE_EXCLUDE; default-OFF,
        # additive). ON -> _contact_range_exclude_step tapers the range PUSKF update-fraction on
        # CROSS+BELOW pos/vel to the sign-lock floor inside [r_inner, r_outer] (Schmidt-exclude the
        # sign-blind range from the axes GPS reads cleanly -- #352 [MEASURED]); behind untouched;
        # =0 -> step never called -> byte-identical. Reuses the sign-lock envelope; config read
        # ONCE here. Smear-absorber for the surge gate-relief lever. See #352 F-FUSER-4-PRE/CEIL.
        self._contact_exclude_enabled = (
            os.environ.get("FD_UWB_CONTACT_RANGE_EXCLUDE", "0").strip() == "1"
        )
        # Surge gate-relief (FD_UWB_SURGE_GATE_RELIEF; default-OFF, additive). ON ->
        # _surge_relief_step arms on chronic consistent-sign range rejection near dock and inflates
        # the behind-position P so the gate admits the withheld corrective range (P not R -> gain
        # preserved, distinct from #351). =0 -> step never called -> byte-identical. Config read
        # ONCE here; per-episode detector state below (zeroed in reset()). See #352b F-FUSER-4-CEIL.
        self._surge_relief_enabled = (
            os.environ.get("FD_UWB_SURGE_GATE_RELIEF", "0").strip() == "1"
        )
        self._relief_cusum = 0.0       # rejection-rate CUSUM (arms on chronic reject)
        self._relief_signsum = 0.0     # signed innovation accumulator (consistent-direction guard)
        self._relief_active = False
        self._relief_count = 0         # UWB-tick counter (warmup guard)
        self._relief_prev_rej = 0      # uwb_rejected_count at last step (prior-tick outcome)
        self._relief_prev_upd = 0      # uwb_update_count at last step
        self._relief_prev_innov = 0.0  # prior-tick innovation (sign of the rejected correction)

        # Behind ring-throttle (#405 D2; FD_UWB_BEHIND_RING_THROTTLE: "0"/off, "C" Form-C CUSUM
        # taper, "S" Form-S stateless saturation). Down-weights BOTH behind gains (beta[0]+beta[3])
        # on sustained accepted aft over-read near dock (D1r 726a4111 redirect). Config read ONCE.
        # OFF (mode="") -> _behind_ring_throttle_step never called -> _puskf_beta untouched
        # (byte-identical). Per-episode detector state below (zeroed in reset()); Form-C only.
        _ring_mode = os.environ.get("FD_UWB_BEHIND_RING_THROTTLE", "0").strip().upper()
        self._ring_mode = _ring_mode if _ring_mode in ("C", "S") else ""
        self._ring_count = 0           # in-band UWB-tick counter (Form-C warmup guard)
        self._ring_cusum = 0.0         # Form-C signed-aft-innovation CUSUM
        self._ring_active = False      # Form-C armed flag (hysteresis)
        self._ring_gate = 0.0          # Form-C tick-ramp gate [0,1]
        self._ring_last_w = 1.0        # last applied robust weight (diagnostics / smoke)

        # F-DAMP REFUTED [MEASURED 2026-07-13 #404] — fires but overpowered by measurement re-injection;
        # retained default-OFF as the reproducible refutation instrument (see constants block + #404 finding).
        # Behind-velocity dock-band damping (F-DAMP; FD_UWB_BEHIND_VDAMP_ENABLE; default-OFF,
        # additive). ON -> _predict applies a first-order decay a=exp(-dt/tau) to the BEHIND
        # velocity state x[IDX_VX] (and its covariance row/col, KF-consistent) inside the dock band
        # (est. slant range <= r_band), attenuating the #403 post-arrival aft surge while GPS/UWB
        # updates re-establish the slow closure. BEHIND-axis ONLY (cross/below velocity untouched).
        # =0 -> the _predict damping branch is not entered -> byte-identical. Stateless (no per-
        # episode accumulator -> nothing added to reset()). Config read ONCE here, never per-tick.
        # tau/r_band env overrides; clamps below. See #404 PREREG_FDAMP_AB.
        self._behind_vdamp_enabled = (
            os.environ.get("FD_UWB_BEHIND_VDAMP_ENABLE", "0").strip() == "1"
        )
        try:
            self._vdamp_tau_s = float(
                os.environ.get("FD_UWB_BEHIND_VDAMP_TAU_S", str(BEHIND_VDAMP_TAU_S_DEFAULT))
            )
        except (TypeError, ValueError):
            self._vdamp_tau_s = BEHIND_VDAMP_TAU_S_DEFAULT
        try:
            self._vdamp_r_band_m = float(
                os.environ.get("FD_UWB_BEHIND_VDAMP_R_BAND_M", str(BEHIND_VDAMP_R_BAND_M_DEFAULT))
            )
        except (TypeError, ValueError):
            self._vdamp_r_band_m = BEHIND_VDAMP_R_BAND_M_DEFAULT
        # Clamps: tau floored > 0 so a=exp(-dt/tau) in (0,1) (never divide-by-0 / total-kill); band >= 0.
        self._vdamp_tau_s = max(1e-3, self._vdamp_tau_s)
        self._vdamp_r_band_m = max(0.0, self._vdamp_r_band_m)

        # Channel-isolation K*innov gauge (FD_UWB_CHANNEL_ISO_DIAG; default-OFF, LOGGING-ONLY).
        # D1 (#405 STEP-2): on each ACCEPTED UWB tick, records the NOMINAL per-state Kalman
        # correction contribution K[i]*innovation for behind-POSITION (IDX_X) and behind-VELOCITY
        # (IDX_VX) -- the channel-isolation read adjudicating whether the accepted aft-bias
        # re-injection lands in beta[0] (position) or beta[3] (velocity). Writes uwb_channel_iso.csv
        # to FD_UWB_CHANNEL_ISO_DIR (fallback FD_UWB_FUSION_DIAG_DIR) so it JOINS uwb_fusion_diag.csv
        # on sim_time (that file is R7.1-frozen + docking-states-only; the join does the dock-scoping).
        # NO behavior change: default-OFF byte-identical; when ON, write-only I/O never read back into
        # the filter -> lockstep-deterministic. Reads PERCEIVED K + innovation ONLY (G4-clean, no truth).
        self._chaniso_enabled = os.environ.get("FD_UWB_CHANNEL_ISO_DIAG", "0").strip() == "1"
        self._chaniso_dir = (
            os.environ.get("FD_UWB_CHANNEL_ISO_DIR", "").strip()
            or os.environ.get("FD_UWB_FUSION_DIAG_DIR", "").strip()
        )
        self._chaniso_file = None        # lazily-opened CSV handle (first accepted tick)
        self._chaniso_writer = None
        self._chaniso_rows_written = 0
        self._chaniso_last = None        # last row tuple (get_diagnostics / unit smoke)

        # T-403 gate sidecar: ONE row per UWB ATTEMPT (accepted AND rejected, EVERY anchor),
        # with H*P*H' logged SEPARATELY from R so the R' counterfactual is computable offline
        # for ANY R' with no re-fly. Reuses FD_UWB_FUSION_DIAG / _DIR -- NO NEW ENV FLAG.
        # Default-OFF -> the call site branch is never entered -> byte-identical.
        self._gatediag = _gate_diag.from_env(INNOVATION_GATE_SIGMA) if _gate_diag else None
        self._gatediag_on = bool(self._gatediag is not None and self._gatediag.enabled)

        # State vector: [x, y, z, vx, vy, vz]
        self.x = [0.0] * 6

        # Covariance matrix (6x6, stored as list of lists)
        # Initialize with large uncertainty
        self.P = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            self.P[i][i] = 100.0  # Position variance [m^2]
            self.P[i + 3][i + 3] = 10.0  # Velocity variance [(m/s)^2]

        # Timestamps
        self.last_predict_time: Optional[float] = None
        self.last_gps_time: Optional[float] = None
        self.last_uwb_time: Optional[float] = None

        # Diagnostics
        self.gps_update_count = 0
        self.uwb_update_count = 0
        self.uwb_rejected_count = 0
        self.last_innovation_range = 0.0

        # History for rate monitoring
        self._gps_times: deque = deque(maxlen=10)
        self._uwb_times: deque = deque(maxlen=10)

        # ADDED 2025-11-27: Range rate tracking for adaptive process noise
        self._prev_range: Optional[float] = None
        self._prev_range_time: Optional[float] = None

        # Multi-anchor support: anchor position registry (behind_m, cross_m, below_m) in leader frame
        # Anchor 0 always defaults to origin (boom tip / primary anchor location)
        if anchor_positions is None:
            self.anchor_positions: Dict[int, Tuple[float, float, float]] = {0: (0.0, 0.0, 0.0)}
        else:
            self.anchor_positions = {0: (0.0, 0.0, 0.0)}
            self.anchor_positions.update(anchor_positions)

        # Per-anchor range-rate tracking for adaptive process noise
        self._prev_range_by_anchor: Dict[int, float] = {}
        self._prev_range_time_by_anchor: Dict[int, float] = {}

        # Per-anchor diagnostics
        self._uwb_updates_by_anchor: Dict[int, int] = {}
        self._uwb_rejected_by_anchor: Dict[int, int] = {}
        self._unknown_anchor_count = 0
        self._last_innovation_by_anchor: Dict[int, float] = {}

        logger.info(
            f"RelativePoseFusion initialized: GPS_sigma={gps_sigma_m:.2f}m, "
            f"UWB_sigma={uwb_sigma_m:.2f}m, anchors={list(self.anchor_positions.keys())}"
        )

    # ==========================================================================
    # ANCHOR MANAGEMENT METHODS
    # ==========================================================================

    def set_anchor_positions(
        self, anchor_positions: Dict[int, Tuple[float, float, float]]
    ) -> None:
        """
        Set positions for multiple anchors at once.

        Args:
            anchor_positions: Dict mapping anchor_id to (behind_m, cross_m, below_m) tuple.
                              Position is in leader frame relative to leader origin.
        """
        self.anchor_positions.update(anchor_positions)

    def set_anchor_position(
        self, anchor_id: int, behind_m: float, cross_m: float, below_m: float
    ) -> None:
        """
        Set position for a single anchor.

        Args:
            anchor_id: Unique identifier for the anchor
            behind_m: Position along-track (positive = anchor behind leader origin)
            cross_m: Position cross-track (positive = anchor right of leader track)
            below_m: Position vertical (positive = anchor below leader origin)
        """
        self.anchor_positions[anchor_id] = (behind_m, cross_m, below_m)

    def get_anchor_position(self, anchor_id: int) -> Tuple[float, float, float]:
        """
        Get anchor position, defaulting to origin if unknown.

        Args:
            anchor_id: Unique identifier for the anchor

        Returns:
            Tuple of (behind_m, cross_m, below_m) position in leader frame
        """
        return self.anchor_positions.get(anchor_id, (0.0, 0.0, 0.0))

    # ==========================================================================
    # PREDICTION AND UPDATE METHODS
    # ==========================================================================

    def _predict(self, dt: float, range_rate_m_s: Optional[float] = None) -> None:
        """
        Propagate state forward by dt seconds using constant-velocity model.

        State transition:
            x(k+1) = F * x(k) + w
        where F is the state transition matrix and w is process noise.

        Args:
            dt: Time step [s]
            range_rate_m_s: Optional range rate for adaptive process noise [m/s]
                            (ADDED 2025-11-27 per intercept_rejoin analysis)
        """
        if dt <= 0:
            return

        # State transition: position += velocity * dt
        self.x[IDX_X] += self.x[IDX_VX] * dt
        self.x[IDX_Y] += self.x[IDX_VY] * dt
        self.x[IDX_Z] += self.x[IDX_VZ] * dt

        # Covariance propagation: P = F*P*F' + Q
        # For constant velocity model with position-velocity correlation:
        # F = [I, dt*I; 0, I]

        # ADAPTIVE PROCESS NOISE (ADDED 2025-11-27)
        # During rapid closure/separation, increase process noise so filter tracks faster
        base_sigma = self.process_accel_sigma
        # DWNA Q-recal (#789, default-OFF): the spine-collapse root is Q-starvation
        # (process_accel_sigma=1.0 too small -> P->0 -> K->0 -> UWB gated out). When ON,
        # use the analytically-calibrated accel sigma (steady-state K in [0.1,0.3] vs R=4.0)
        # as the base instead of 1.0; the transient adaptive closure boost below still
        # multiplies on top (orthogonal). OFF -> base_sigma unchanged -> byte-identical.
        if self._qrecal_enabled:
            base_sigma = self._qrecal_accel_sigma
        if range_rate_m_s is not None and abs(range_rate_m_s) > ADAPTIVE_Q_RANGE_RATE_THRESH:
            base_sigma *= ADAPTIVE_Q_BOOST_FACTOR
            logger.debug(
                f"Adaptive Q boost: |Vc|={abs(range_rate_m_s):.1f} m/s -> sigma={base_sigma:.2f}"
            )

        # Process noise Q (simplified: acceleration affects velocity, which affects position)
        # q_pos coefficient: textbook DWNA is 0.25*sigma^2*dt^4 (Bar-Shalom); the legacy
        # 0.5 carried a 2x error (q_vel/q_pv coefficients were already correct). Corrected
        # ONLY under qrecal so flag-OFF stays byte-identical; negligible at dt^4 either way
        # -- the load-bearing lever is base_sigma above.
        q_pos_coeff = 0.25 if self._qrecal_enabled else 0.5
        q_pos = q_pos_coeff * base_sigma**2 * dt**4  # Position variance from accel
        q_vel = base_sigma**2 * dt**2  # Velocity variance from accel
        q_pv = base_sigma**2 * dt**3 / 2  # Position-velocity covariance

        # Update covariance (simplified version - diagonal dominance assumed)
        for i in range(3):
            # Position variance increases with velocity uncertainty and time
            self.P[i][i] += 2 * dt * self.P[i][i + 3] + dt**2 * self.P[i + 3][i + 3] + q_pos
            # Velocity variance increases with process noise
            self.P[i + 3][i + 3] += q_vel
            # Position-velocity covariance
            self.P[i][i + 3] += dt * self.P[i + 3][i + 3] + q_pv
            self.P[i + 3][i] = self.P[i][i + 3]  # Symmetric

        # All-axis constant process-noise recalibration (Q-recal #789). Adds
        # delta_q*dt to EVERY position diagonal so K stays in [0.1,0.3] on ALL axes
        # (kills the clean-GPS cross divergence). Default 0.0 -> no-op -> byte-identical.
        if self.pos_rw_m2_s > 0.0:
            for _qi in range(3):
                self.P[_qi][_qi] += self.pos_rw_m2_s * dt

        # F-DAMP REFUTED [MEASURED 2026-07-13 #404] — this branch fires but is overpowered by the
        # measurement updates (fused surge UNMOVED); retained default-OFF. See constants block + #404 finding.
        # Behind-velocity dock-band damping (F-DAMP; FD_UWB_BEHIND_VDAMP_ENABLE; #404). OFF ->
        # branch not entered -> byte-identical. First-order decay of the BEHIND velocity state (and
        # its covariance row/col, so P stays consistent with the damped state: P' = A P A' with
        # A = diag(1,1,1,a,1,1)) inside the dock band, attenuating the #403 post-arrival aft surge
        # limit cycle. BEHIND-axis ONLY. dt>0 here (guarded above) -> a=exp(-dt/tau) in (0,1);
        # dt-based -> cadence-robust. r_est from state (cheap; no new inputs).
        if self._behind_vdamp_enabled:
            r_est = math.sqrt(self.x[IDX_X] ** 2 + self.x[IDX_Y] ** 2 + self.x[IDX_Z] ** 2)
            if r_est <= self._vdamp_r_band_m:
                a = math.exp(-dt / self._vdamp_tau_s)
                self.x[IDX_VX] *= a
                for _j in range(6):
                    self.P[IDX_VX][_j] *= a
                    self.P[_j][IDX_VX] *= a

    def update_gps(self, meas: GPSMeasurement) -> bool:
        """
        Update filter with GPS-derived relative position measurement.

        Args:
            meas: GPS measurement containing behind, cross, below position and velocities

        Returns:
            True if update was applied, False if rejected
        """
        now = meas.timestamp

        # Adaptive-β (A1'): record the raw GPS behind so the detector can read the
        # PERSISTENT bias s=|fused_behind - last_gps_behind| each UWB tick. Unconditional
        # + cheap; consumed ONLY in _beta_adapt_step (gated on _beta_adapt_enabled). When
        # adaptive is OFF this value is simply never read (no behavior change).
        self._last_gps_behind = meas.behind_m

        # Predict to measurement time
        if self.last_predict_time is not None:
            dt = now - self.last_predict_time
            if dt > 0:
                self._predict(dt)
        self.last_predict_time = now

        # Measurement vector: [x, y, z, vx, vy, vz]
        z = [
            meas.behind_m,
            meas.cross_m,
            meas.below_m,
            meas.vel_behind_m_s,
            meas.vel_cross_m_s,
            meas.vel_below_m_s,
        ]

        # Measurement matrix H = I (direct observation)
        # Measurement noise R
        # C1 HONEST-R (T-358): per-axis position sigma when elected, else the
        # scalar -- _sa is None on the default path, so these three lines are
        # byte-identical to the pre-C1 expression when the knob is unset.
        # Order is the state's own: index 0 behind, 1 cross, 2 below.
        # Velocity R (indices 3..5) is deliberately UNTOUCHED: the election
        # measured POSITION sigma only, and gps_vel_sigma_m_s has no measured
        # basis here.
        _sa = self.gps_sigma_axes_m
        R = [
            (_sa[0] if _sa else self.gps_sigma_m) ** 2,  # x (behind) variance
            (_sa[1] if _sa else self.gps_sigma_m) ** 2,  # y (cross) variance
            (_sa[2] if _sa else self.gps_sigma_m) ** 2,  # z (below) variance
            self.gps_vel_sigma_m_s**2,  # vx variance
            self.gps_vel_sigma_m_s**2,  # vy variance
            self.gps_vel_sigma_m_s**2,  # vz variance
        ]

        # Kalman update (simplified for H=I)
        for i in range(6):
            # Innovation
            innovation = z[i] - self.x[i]

            # Innovation variance: S = P + R (for H=I)
            S = self.P[i][i] + R[i]

            # Kalman gain: K = P / S (for single measurement per state)
            if S > 1e-6:
                K = self.P[i][i] / S
            else:
                K = 0.0

            # State update
            self.x[i] += K * innovation

            # Covariance update (simplified): P = (I - K*e_i*e_i') * P
            # Scale row i by (1-K), then mirror to column i for symmetry.
            factor = 1 - K
            for j in range(6):
                self.P[i][j] *= factor
            for j in range(6):
                self.P[j][i] = self.P[i][j]

        # Update timestamps and counters
        self.last_gps_time = now
        self.gps_update_count += 1
        self._gps_times.append(now)

        return True

    def _beta_adapt_step(self, cross_cosine: float = 0.0) -> None:
        """Adaptive-β (A1'): a ONE-SIDED CUSUM on the PERSISTENT bias
        s=|fused_behind - last_gps_behind| drives the cross/below PUSKF update-fraction.
        The bias = what the fuser is correcting; it stays large at degraded REGARDLESS of
        scrub (unlike the innovation, which behind β=1 scrubs small -> the A2 NO-GO). So it
        fires at degraded onset and HOLDS -> cross/below ramp to β=1 and STAY there (the A2
        fix). behind β=1 always; cross/below β = CB_MIN + (1-CB_MIN)*gate. Tick-count ramp
        only (no dt/wall-clock/random) -> lockstep bit-deterministic. Called PRE-gate every
        UWB tick when FD_UWB_BETA_ADAPT_ENABLE=1. A0'-calibrated (BETA_ADAPT_* +
        analysis/beta_adapt_a0p_20260620T062550Z).

        cross_cosine = |dy/r| (range Jacobian cross component, from update_uwb). Consumed ONLY
        when FD_UWB_BETA_DECOUPLE_CB=1, where it scales the capped cross β; the decouple branch
        also holds below β=0. When the flag is OFF the arg is ignored and the actuator is the
        verbatim bcb path -> byte-identical to the adopted adaptive-β baseline.
        """
        self._beta_count += 1
        # s = persistent bias magnitude (>=0): how far the fused spine sits from raw GPS =
        # the GPS bias the UWB is correcting. ~0.55 m clean, ~1.38 m at degraded (+1 m).
        if self._last_gps_behind is not None and self._beta_count > BETA_ADAPT_WARMUP:
            s = min(BETA_ADAPT_CLAMP_M, abs(self.x[IDX_X] - self._last_gps_behind))
            self._beta_cusum = min(BETA_ADAPT_CAP, max(0.0, self._beta_cusum + s - BETA_ADAPT_K))
            if self._beta_cusum > BETA_ADAPT_H:
                self._beta_active = True
            elif self._beta_active and self._beta_cusum < BETA_ADAPT_HYST * BETA_ADAPT_H:
                self._beta_active = False
        # Tick-count ramp (integer step; NEVER dt-scaled). Warmup holds the gate at 0.
        if self._beta_count <= BETA_ADAPT_WARMUP:
            self._beta_gate = 0.0
        elif self._beta_active:
            self._beta_gate = min(1.0, self._beta_gate + 1.0 / BETA_ADAPT_RAMP_UP)
        else:
            self._beta_gate = max(0.0, self._beta_gate - 1.0 / BETA_ADAPT_RAMP_DN)
        # Drive the PUSKF actuator: behind pos+vel β=1 (always full); cross/below pos+vel
        # β=bcb. At gate=1 (degraded) bcb=1 -> _puskf_enabled False -> verbatim full update on
        # all axes (= β=1 config); at gate=0 (clean) bcb=CB_MIN=0 -> cross/below no UWB (smooth).
        if self._beta_decouple_cb_enabled:
            # Decouple ON: BELOW β=0 (Schmidt-exclude range from the vertical axis -- kills the
            # below ratchet that false-fires the -2.0 m guard); CROSS β capped at
            # cap*gate*|cross_cosine| (small observability-scaled trickle); BEHIND β=1. Geometry
            # + tick ramp only -> deterministic.
            bx = self._beta_decouple_cross_cap * self._beta_gate * abs(cross_cosine)
            self._puskf_beta = [1.0, bx, 0.0, 1.0, bx, 0.0]
            self._puskf_enabled = True
        else:
            bcb = BETA_ADAPT_CB_MIN + (1.0 - BETA_ADAPT_CB_MIN) * self._beta_gate
            self._puskf_beta = [1.0, bcb, bcb, 1.0, bcb, bcb]
            self._puskf_enabled = bcb < 1.0

    def _behind_signlock_step(self, predicted_range: float) -> None:
        """Sign-conflict-aware UWB-behind gain guard (FD_UWB_BEHIND_SIGNLOCK_GUARD; v10 default-ON,
        additive). The degraded-GPS terminal "dock-then-overtake" is a UWB range-only SIGN-LOCK on
        the behind axis: near dock the range is sign-blind, so once fused_behind holds the wrong
        sign the range update reinforces it and overpowers the sign-correct GPS ~180-400x (see RCA
        behind_signlock_rca_20260628 + DESIGN_CHECK_Q1Q2). When GPS is sign-confident
        (|gps_behind| > gps_min) AND disagrees with the state behind sign AND the dock is near
        (predicted_range < r_outer), taper the behind PUSKF update-fraction 1->floor across
        [r_outer, r_inner] (Schmidt-exclude the range from behind) to hand sign authority back to
        the sign-correct GPS; otherwise behind beta=1 (UWB still corrects the ~0.5 m along-track
        bias). Self-clearing: once the state crosses to the GPS sign the conflict ends and beta
        returns to 1. Geometry + state + last-GPS only -> deterministic (RNG-free), no truth-side
        input. SETS _puskf_beta[behind pos/vel] every call (1.0 when no guard) so it is correct
        with adaptive-beta either ON or OFF; never touches cross/below. Called PRE-gate from
        update_uwb when FD_UWB_BEHIND_SIGNLOCK_GUARD is set.
        """
        beta_behind = 1.0
        gps_b = self._last_gps_behind
        state_b = self.x[IDX_X]
        if (
            gps_b is not None
            and abs(gps_b) > self._signlock_gps_min
            and gps_b * state_b < 0.0
            and predicted_range < self._signlock_r_outer
        ):
            if predicted_range <= self._signlock_r_inner:
                env = 1.0
            else:
                env = (self._signlock_r_outer - predicted_range) / (
                    self._signlock_r_outer - self._signlock_r_inner
                )
            beta_behind = max(self._signlock_floor, 1.0 - env)
        self._puskf_beta[IDX_X] = beta_behind
        self._puskf_beta[IDX_X + 3] = beta_behind
        if beta_behind < 1.0:
            self._puskf_enabled = True

    def _contact_range_exclude_step(self, predicted_range: float) -> None:
        """Contact-range cross/below authority exclude (FD_UWB_CONTACT_RANGE_EXCLUDE; default-OFF,
        additive). At contact the range Jacobian H=[dx/r,dy/r,dz/r] is behind-dominant (dx/r~1);
        the sign-blind range legitimately informs BEHIND but its cross/below components only SMEAR
        axes GPS already reads cleanly (#352 [MEASURED]: contact GPS:fused err ratio cross 0.27,
        fused cross corrupts to 0.5-0.9 m when acceptance rises -- #351 T-legs). Inside
        [r_outer, r_inner] taper the range PUSKF update-fraction 1->floor across the envelope and
        apply it as min(beta[i], taper) to CROSS+BELOW pos/vel (indices 1,2,4,5) -- NEVER raising
        authority adaptive-beta/decouple already reduced (monotone), NEVER touching behind (0,3).
        Confines any accepted range update to behind so it cannot re-smear cross/below -- the
        smear-absorber for the surge gate-relief lever. Reuses the sign-lock envelope + floor; NO
        new constants. Geometry only -> deterministic (RNG-free). Runs AFTER adaptive-beta (which
        reassigns the whole beta vector) so min() sees the final cross/below beta. OFF -> not called
        -> _puskf_beta untouched (byte-identical). See #352 F-FUSER-4-PRE/CEIL.
        """
        if predicted_range >= self._signlock_r_outer:
            return
        if predicted_range <= self._signlock_r_inner:
            env = 1.0
        else:
            env = (self._signlock_r_outer - predicted_range) / (
                self._signlock_r_outer - self._signlock_r_inner
            )
        taper = max(self._signlock_floor, 1.0 - env)
        for i in (IDX_Y, IDX_Z, IDX_Y + 3, IDX_Z + 3):
            self._puskf_beta[i] = min(self._puskf_beta[i], taper)
        if taper < 1.0:
            self._puskf_enabled = True

    def _surge_relief_step(self, predicted_range: float, innovation: float) -> None:
        """Surge gate-relief (FD_UWB_SURGE_GATE_RELIEF; default-OFF, additive). #352b [MEASURED]:
        during surge the converged over-confident EKF rejects ~70% of range updates that are
        correctly signed to fix the behind under-read (~1.15 m, ~11σ withheld), chronic-intermittent
        (max-run 2). A rejection-rate CUSUM (prior-tick outcome read from the uwb_rejected/update
        counters -- NO gate hook) with an innovation-sign-consistency guard arms during chronic
        consistent rejection near dock and inflates the behind-POSITION P to SURGE_RELIEF_P_POS so
        the 3σ gate admits the corrective range (raises S -> gate opens; raises K -> fast catch-up).
        Inflates P NOT R -> gain PRESERVED (opposite of #351's refuted R-inflation). Additive
        diagonal max() -> PSD-safe and bounded. Self-de-arms as accepts resume (hysteresis). Runs
        PRE-gate; count/geometry only -> deterministic (RNG/clock-free). OFF -> not called ->
        byte-identical. Pairs with the contact-range exclude (C confines the admitted range to
        behind). See #352b F-FUSER-4-CEIL.
        """
        self._relief_count += 1
        # Prior-tick outcome from the monotone counters (no hook needed in the gate branches).
        d_rej = self.uwb_rejected_count - self._relief_prev_rej
        d_upd = self.uwb_update_count - self._relief_prev_upd
        self._relief_prev_rej = self.uwb_rejected_count
        self._relief_prev_upd = self.uwb_update_count
        if d_rej > 0:
            self._relief_cusum = min(SURGE_RELIEF_CAP, self._relief_cusum + 1.0)
            ps = 1.0 if self._relief_prev_innov > 0 else (-1.0 if self._relief_prev_innov < 0 else 0.0)
            self._relief_signsum = max(-SURGE_RELIEF_CAP, min(SURGE_RELIEF_CAP, self._relief_signsum + ps))
        elif d_upd > 0:
            self._relief_cusum = max(0.0, self._relief_cusum - 1.0)
            if self._relief_signsum > 0.0:
                self._relief_signsum -= 1.0        # bleed toward 0 on accepts (allow later flip)
            elif self._relief_signsum < 0.0:
                self._relief_signsum += 1.0
        self._relief_prev_innov = innovation
        # Arm / de-arm (hysteresis) past warmup, ONLY on consistent-direction chronic rejection.
        if self._relief_count > SURGE_RELIEF_WARMUP:
            if self._relief_cusum >= SURGE_RELIEF_ARM and abs(self._relief_signsum) >= SURGE_RELIEF_SIGN:
                self._relief_active = True
            elif self._relief_active and self._relief_cusum < SURGE_RELIEF_HYST * SURGE_RELIEF_ARM:
                self._relief_active = False
        # Inflate ONLY when armed, near dock, and THIS innovation continues the established drift.
        cur_sign = 1.0 if innovation > 0 else (-1.0 if innovation < 0 else 0.0)
        if (self._relief_active
                and predicted_range < self._signlock_r_outer
                and cur_sign * self._relief_signsum > 0.0
                and self.P[IDX_X][IDX_X] < SURGE_RELIEF_P_POS):
            self.P[IDX_X][IDX_X] = SURGE_RELIEF_P_POS   # additive diagonal floor -> PSD-safe, bounded

    def _behind_ring_throttle_step(self, predicted_range: float, innovation: float) -> None:
        """Behind ring-throttle (FD_UWB_BEHIND_RING_THROTTLE; #405 D2, default-OFF). Down-weights
        BOTH behind PUSKF update-fractions (beta[IDX_X] pos + beta[IDX_VX] vel) when a SUSTAINED
        accepted aft over-read (innovation>0) near dock indicates the surge re-injection (D1r
        726a4111: beta[0] ~67% / beta[3] ~33% -> throttle both, beta[3]-alone REFUTED). Two forms:
          Form-C ("C"): a one-sided CUSUM on (innov - K) arms (hysteresis) + a tick-ramp gate yield
                        a CONTINUOUS weight w = 1 - (1-FLOOR)*gate in [FLOOR, 1].
          Form-S ("S"): stateless saturation w = clip(SAT_C/innov, FLOOR, 1) on aft over-reads only.
        Applied as min(beta[i], w) to (IDX_X, IDX_VX) ONLY -- NEVER cross/below, NEVER the gate,
        NEVER _predict; composes down monotone with sign-lock/adaptive-beta (min()); FLOOR>0. Runs
        PRE-gate AFTER the sibling steps so min() sees the final behind beta. Range + innovation +
        tick only -> deterministic (RNG/clock-free); PERCEIVED innovation only (G4-clean). OFF
        (mode="") -> not called -> byte-identical. See analysis/channeliso_*/D1R_channel_iso.md.
        """
        if predicted_range >= self._signlock_r_outer:
            # Outside the dock band: bleed the Form-C detector; apply no throttle this tick.
            self._ring_cusum = max(0.0, self._ring_cusum - 1.0)
            self._ring_gate = max(0.0, self._ring_gate - 1.0 / RING_THROTTLE_RAMP_DN)
            self._ring_last_w = 1.0
            return
        if self._ring_mode == "S":
            # Stateless innovation-saturation: down-weight aft (innov>0) over-reads only.
            w = (max(RING_THROTTLE_FLOOR, min(1.0, RING_THROTTLE_SAT_C / innovation))
                 if innovation > 0.0 else 1.0)
        else:
            # Form-C: signed one-sided CUSUM on the aft innovation + tick-ramp gate.
            self._ring_count += 1
            self._ring_cusum = min(RING_THROTTLE_CAP,
                                   max(0.0, self._ring_cusum + (innovation - RING_THROTTLE_K)))
            if self._ring_count > RING_THROTTLE_WARMUP:
                if self._ring_cusum >= RING_THROTTLE_ARM:
                    self._ring_active = True
                elif self._ring_active and self._ring_cusum < RING_THROTTLE_HYST * RING_THROTTLE_ARM:
                    self._ring_active = False
            if self._ring_active:
                self._ring_gate = min(1.0, self._ring_gate + 1.0 / RING_THROTTLE_RAMP_UP)
            else:
                self._ring_gate = max(0.0, self._ring_gate - 1.0 / RING_THROTTLE_RAMP_DN)
            w = 1.0 - (1.0 - RING_THROTTLE_FLOOR) * self._ring_gate
        self._ring_last_w = w
        if w < 1.0:
            self._puskf_beta[IDX_X] = min(self._puskf_beta[IDX_X], w)
            self._puskf_beta[IDX_VX] = min(self._puskf_beta[IDX_VX], w)
            self._puskf_enabled = True

    def _channel_iso_log(self, now: float, anchor_id: int, predicted_range: float,
                         innovation: float, K: List[float]) -> None:
        """D1 (#405 STEP-2) channel-iso gauge: record the NOMINAL per-state Kalman correction
        K[i]*innovation for behind-POSITION (IDX_X) and behind-VELOCITY (IDX_VX) on an ACCEPTED
        UWB tick. LOGGING-ONLY / write-only (never read back -> lockstep-deterministic); G4-clean
        (perceived K + innovation only, no truth). Never raises into the control/estimation path."""
        if not self._chaniso_dir:
            return
        try:
            row = (
                round(now, 6), int(anchor_id), round(predicted_range, 6), round(innovation, 6),
                round(K[IDX_X] * innovation, 9),    # behind-POSITION correction contribution
                round(K[IDX_VX] * innovation, 9),   # behind-VELOCITY correction contribution
                round(K[IDX_X], 9), round(K[IDX_VX], 9),
            )
            if self._chaniso_writer is None:
                import csv as _csv
                os.makedirs(self._chaniso_dir, exist_ok=True)
                self._chaniso_file = open(
                    os.path.join(self._chaniso_dir, "uwb_channel_iso.csv"), "w", newline=""
                )
                self._chaniso_writer = _csv.writer(self._chaniso_file)
                self._chaniso_writer.writerow([
                    "sim_time_s", "anchor_id", "predicted_range_m", "innov_m",
                    "kdx_behind_pos", "kdx_behind_vel", "k_behind_pos", "k_behind_vel",
                ])
            self._chaniso_writer.writerow(row)
            self._chaniso_file.flush()
            self._chaniso_rows_written += 1
            self._chaniso_last = row
        except Exception:
            # FM(e): gauge I/O must NEVER perturb or raise into fusion/control.
            pass

    def update_uwb(self, meas: UWBMeasurement) -> bool:
        """
        Update filter with UWB range measurement.

        This is an Extended Kalman Filter (EKF) update because the range
        measurement is nonlinear: range = sqrt((x-ax)^2 + (y-ay)^2 + (z-az)^2)
        where (ax, ay, az) is the anchor position offset in leader frame.

        Multi-anchor support: Each anchor can have a different position offset
        from the leader origin. Unknown anchors default to origin (0, 0, 0).

        Args:
            meas: UWB measurement containing slant range and anchor_id

        Returns:
            True if update was applied, False if rejected (outlier)
        """
        now = meas.timestamp
        anchor_id = meas.anchor_id

        # Get anchor offset from registry (fallback to origin if unknown)
        if anchor_id not in self.anchor_positions:
            self._unknown_anchor_count += 1
            ax, ay, az = 0.0, 0.0, 0.0
        else:
            ax, ay, az = self.anchor_positions[anchor_id]

        # Per-anchor range rate tracking for adaptive process noise
        range_rate = None
        if anchor_id in self._prev_range_by_anchor:
            prev_range = self._prev_range_by_anchor[anchor_id]
            prev_time = self._prev_range_time_by_anchor[anchor_id]
            dt_r = now - prev_time
            if dt_r > 0.01:  # At least 10ms between measurements
                range_rate = (meas.range_m - prev_range) / dt_r
        self._prev_range_by_anchor[anchor_id] = meas.range_m
        self._prev_range_time_by_anchor[anchor_id] = now

        # Also update legacy single-anchor tracking for backwards compat diagnostics
        self._prev_range = meas.range_m
        self._prev_range_time = now

        # Predict to measurement time (with adaptive Q based on range rate)
        if self.last_predict_time is not None:
            dt = now - self.last_predict_time
            if dt > 0:
                self._predict(dt, range_rate_m_s=range_rate)
        self.last_predict_time = now

        # Compute displacement from anchor to follower position
        # State x,y,z is follower position in leader frame
        # Anchor ax,ay,az is anchor position in leader frame
        # Displacement dx,dy,dz is vector from anchor to follower
        x, y, z = self.x[IDX_X], self.x[IDX_Y], self.x[IDX_Z]
        dx, dy, dz = x - ax, y - ay, z - az

        # Predicted range from anchor to follower
        predicted_range = math.sqrt(dx**2 + dy**2 + dz**2)

        # Avoid division by zero
        if predicted_range < 0.1:
            predicted_range = 0.1

        # Measurement Jacobian H = d(range)/d(state)
        # range = sqrt((x-ax)^2 + (y-ay)^2 + (z-az)^2)
        # d(range)/dx = (x-ax)/range = dx/range
        # H = [dx/r, dy/r, dz/r, 0, 0, 0]
        H = [dx / predicted_range, dy / predicted_range, dz / predicted_range, 0.0, 0.0, 0.0]

        # Innovation (measurement residual)
        innovation = meas.range_m - predicted_range
        self.last_innovation_range = innovation
        self._last_innovation_by_anchor[anchor_id] = innovation

        # Adaptive-β (A1; default-ON). Behind CUSUM detector drives the cross/below PUSKF
        # update-fraction. Placed PRE-gate so it sees every UWB tick (accept + reject).
        # Mutates only self._puskf_beta/_puskf_enabled (consumed by the PUSKF blend in the
        # accept branch below). OFF -> not executed -> _puskf_beta stays the static ctor value.
        if self._beta_adapt_enabled:
            self._beta_adapt_step(abs(H[IDX_Y]))

        # Behind sign-lock guard (FD_UWB_BEHIND_SIGNLOCK_GUARD; v10 default-ON). After the adaptive-beta
        # actuator pins behind beta=1, optionally taper it back down when the UWB range is sign-blind
        # near dock and GPS disagrees with the state sign (see _behind_signlock_step). =0 -> skipped
        # -> behind beta untouched (byte-identical). PRE-gate so a rejected tick just recomputes next.
        if self._behind_signlock_enabled:
            self._behind_signlock_step(predicted_range)

        # Contact-range cross/below exclude (FD_UWB_CONTACT_RANGE_EXCLUDE; default-OFF). AFTER
        # adaptive-beta + sign-lock so min() sees the FINAL cross/below beta; confines any accepted
        # range to behind (the surge gate-relief lever's smear-absorber). OFF -> skipped ->
        # byte-identical. PRE-gate (a rejected tick just recomputes next).
        if self._contact_exclude_enabled:
            self._contact_range_exclude_step(predicted_range)

        # Surge gate-relief (FD_UWB_SURGE_GATE_RELIEF; default-OFF). Chronic consistent-sign
        # rejection near dock -> inflate behind-position P so the gate admits the withheld
        # corrective range (P not R -> gain preserved). PRE-gate (must precede S/gate). OFF ->
        # skipped -> byte-identical. Pairs with the contact-range exclude (its smear-absorber).
        if self._surge_relief_enabled:
            self._surge_relief_step(predicted_range, innovation)

        # Behind ring-throttle (#405 D2; FD_UWB_BEHIND_RING_THROTTLE). AFTER sign-lock/adaptive-beta
        # so min() sees the FINAL behind beta; throttles BOTH behind gains (beta[0]+beta[3]) on
        # sustained accepted aft over-read near dock (D1r redirect). OFF (mode="") -> skipped ->
        # byte-identical. PRE-gate (a rejected tick just recomputes next).
        if self._ring_mode:
            self._behind_ring_throttle_step(predicted_range, innovation)

        # Innovation variance: S = H*P*H' + R
        S = sum(H[i] * sum(self.P[i][j] * H[j] for j in range(6)) for i in range(6))
        S += self.uwb_sigma_m**2

        # Outlier rejection (innovation gate)
        innovation_sigma = abs(innovation) / math.sqrt(S) if S > 1e-6 else float("inf")

        # T-403 gate sidecar (default-OFF, LOGGING-ONLY, G4-clean). Placed AFTER
        # innovation_sigma and BEFORE the gate branch, so BOTH outcomes are captured from ONE
        # call site, and BEFORE any state/covariance mutation, so P is exactly the P the gate
        # used. `not (sigma > thresh)` MIRRORS the branch below VERBATIM -- writing
        # `sigma <= thresh` would disagree with the filter whenever sigma is NaN (nan > t and
        # nan <= t are BOTH False, so the filter accepts while the naive form logs a reject),
        # and the NaN rows are exactly the rows this instrument exists to explain.
        if self._gatediag_on:
            self._gatediag.log(now, anchor_id, meas.range_m, predicted_range, innovation,
                               H, self.P, self.uwb_sigma_m ** 2, S, innovation_sigma,
                               not (innovation_sigma > INNOVATION_GATE_SIGMA))

        if innovation_sigma > INNOVATION_GATE_SIGMA:
            self.uwb_rejected_count += 1
            self._uwb_rejected_by_anchor[anchor_id] = (
                self._uwb_rejected_by_anchor.get(anchor_id, 0) + 1
            )
            logger.debug(
                f"UWB rejected: anchor={anchor_id}, range={meas.range_m:.2f}m, "
                f"predicted={predicted_range:.2f}m, "
                f"innovation={innovation:.2f}m ({innovation_sigma:.1f}σ)"
            )
            return False

        # Kalman gain: K = P*H' / S
        K = [0.0] * 6
        if S > 1e-6:
            for i in range(6):
                K[i] = sum(self.P[i][j] * H[j] for j in range(6)) / S

        # D1 (#405 STEP-2) channel-iso gauge (default-OFF, LOGGING-ONLY, G4-clean): on this ACCEPTED
        # tick, record the NOMINAL per-state K*innov for behind-POS (IDX_X) and behind-VEL (IDX_VX) --
        # the read adjudicating whether the accepted aft-bias re-injection lands in beta[0] (position)
        # or beta[3] (velocity). Placed BEFORE the beta blend so it reflects the raw accepted-range
        # correction direction. Byte-identical when the flag is off (branch not entered).
        if self._chaniso_enabled:
            self._channel_iso_log(now, anchor_id, predicted_range, innovation, K)

        if not self._puskf_enabled:
            # OFF (default, beta=1 every state): verbatim original full UWB update.
            # State update: x = x + K * innovation
            for i in range(6):
                self.x[i] += K[i] * innovation

            # Covariance update: P = (I - K*H) * P
            # Simplified: for each row i, subtract K[i] * (H dot row_i of P)
            new_P = [[0.0] * 6 for _ in range(6)]
            for i in range(6):
                for j in range(6):
                    new_P[i][j] = self.P[i][j] - K[i] * sum(H[k] * self.P[k][j] for k in range(6))
            self.P = new_P
        else:
            # PUSKF partial update (H2a; runs ONLY when a per-axis beta<1 is set). Compute
            # the FULL posterior into temporaries using the SAME arithmetic as the OFF path
            # above, then blend per state by the update fraction beta (Brink 2017 family):
            #   state: x++_i = x-_i + beta_i*(x+_i - x-_i) [verbatim Brink state eq;
            #          beta=1 -> full update, beta=0 -> no update / Schmidt limit]
            #   cov: P++_ij = P+_ij + (1-beta_i)(1-beta_j)*(P-_ij - P+_ij)
            # The covariance is the Hadamard/Schur statement of the partial update:
            # P++ = P+ + (gamma gamma^T) o (P- - P+), gamma = 1-beta. Because P- - P+ =
            # K S K^T is PSD and gamma gamma^T is PSD, their Hadamard product is PSD (Schur
            # product theorem), so P++ = P+ + (PSD) stays SYMMETRIC and PSD (conservative).
            # Limits: all beta=1 -> P+ (full posterior); all beta=0 -> P- (prior). The exact
            # paywalled Brink covariance coefficient was not byte-verified; this provably-
            # consistent PSD form is the adopted statement (see H1_RESEARCH_FINDINGS.md).
            beta = self._puskf_beta
            x_full = [self.x[i] + K[i] * innovation for i in range(6)]
            P_full = [
                [self.P[i][j] - K[i] * sum(H[k] * self.P[k][j] for k in range(6)) for j in range(6)]
                for i in range(6)
            ]
            for i in range(6):
                self.x[i] = self.x[i] + beta[i] * (x_full[i] - self.x[i])
            new_P = [
                [
                    P_full[i][j] + (1.0 - beta[i]) * (1.0 - beta[j]) * (self.P[i][j] - P_full[i][j])
                    for j in range(6)
                ]
                for i in range(6)
            ]
            self.P = new_P

        # Update timestamps and counters
        self.last_uwb_time = now
        self.uwb_update_count += 1
        self._uwb_times.append(now)
        self._uwb_updates_by_anchor[anchor_id] = self._uwb_updates_by_anchor.get(anchor_id, 0) + 1

        return True

    def update_uwb_batch(self, measurements: List[UWBMeasurement]) -> Dict[str, Any]:
        """
        Apply multiple UWB measurements, sorted by timestamp.

        This is useful when measurements from multiple anchors arrive together
        and need to be processed in chronological order.

        Args:
            measurements: List of UWB measurements to process

        Returns:
            Dict with processing summary:
                - total: Number of measurements processed
                - accepted: Number of measurements accepted
                - rejected: Number of measurements rejected
                - by_anchor: Dict mapping anchor_id to {accepted, rejected} counts
        """
        sorted_meas = sorted(measurements, key=lambda m: m.timestamp)
        accepted = 0
        rejected = 0
        by_anchor: Dict[int, Dict[str, int]] = {}

        for meas in sorted_meas:
            aid = meas.anchor_id
            if aid not in by_anchor:
                by_anchor[aid] = {"accepted": 0, "rejected": 0}

            if self.update_uwb(meas):
                accepted += 1
                by_anchor[aid]["accepted"] += 1
            else:
                rejected += 1
                by_anchor[aid]["rejected"] += 1

        return {
            "total": len(measurements),
            "accepted": accepted,
            "rejected": rejected,
            "by_anchor": by_anchor,
        }

    def get_state(self, now_s: Optional[float] = None) -> FusedState:
        """
        Get current fused state estimate.

        Args:
            now_s: Optional current time in caller timebase (sim-time seconds).
                   If provided, predicts state to this time and computes sensor ages.
                   If not provided, returns the last predicted state without further prediction.

        Returns:
            FusedState with position, velocity, range, and diagnostics
        """
        # If now_s provided, predict to that time; otherwise use last_predict_time
        if now_s is not None and self.last_predict_time is not None:
            dt = now_s - self.last_predict_time
            if dt > 0 and dt < 1.0:  # Don't predict more than 1 second
                self._predict(dt)
                self.last_predict_time = now_s
            timestamp = now_s
        else:
            # Return state at last prediction time (no wall-time dependency)
            timestamp = self.last_predict_time if self.last_predict_time is not None else 0.0

        # Output position.
        bx, by, bz = self.x[IDX_X], self.x[IDX_Y], self.x[IDX_Z]

        # Compute range from position
        range_m = math.sqrt(bx ** 2 + by ** 2 + bz ** 2)

        # Flatten covariance
        cov_flat = [self.P[i][j] for i in range(6) for j in range(6)]

        # Compute ages (using same timebase - sim-time)
        gps_age = (
            timestamp - self.last_gps_time if self.last_gps_time is not None else float("inf")
        )
        uwb_age = (
            timestamp - self.last_uwb_time if self.last_uwb_time is not None else float("inf")
        )

        return FusedState(
            timestamp=timestamp,
            behind_m=bx,
            cross_m=by,
            below_m=bz,
            vel_behind_m_s=self.x[IDX_VX],
            vel_cross_m_s=self.x[IDX_VY],
            vel_below_m_s=self.x[IDX_VZ],
            range_m=range_m,
            covariance=cov_flat,
            gps_age_s=gps_age,
            uwb_age_s=uwb_age,
            innovation_range=self.last_innovation_range,
        )

    def get_rates(self) -> Tuple[float, float]:
        """
        Get current GPS and UWB update rates.

        Returns:
            Tuple of (gps_rate_hz, uwb_rate_hz)
        """
        gps_rate = 0.0
        uwb_rate = 0.0

        if len(self._gps_times) >= 2:
            dt = self._gps_times[-1] - self._gps_times[0]
            if dt > 0:
                gps_rate = (len(self._gps_times) - 1) / dt

        if len(self._uwb_times) >= 2:
            dt = self._uwb_times[-1] - self._uwb_times[0]
            if dt > 0:
                uwb_rate = (len(self._uwb_times) - 1) / dt

        return gps_rate, uwb_rate

    def reset(
        self, initial_behind: float = 0.0, initial_cross: float = 0.0, initial_below: float = 0.0
    ) -> None:
        """
        Reset filter state to initial position.

        Args:
            initial_behind: Initial along-track position [m]
            initial_cross: Initial cross-track position [m]
            initial_below: Initial vertical position [m]
        """
        self.x = [initial_behind, initial_cross, initial_below, 0.0, 0.0, 0.0]

        # Reset covariance
        self.P = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            self.P[i][i] = 100.0
            self.P[i + 3][i + 3] = 10.0

        self.last_predict_time = None
        self.last_gps_time = None
        self.last_uwb_time = None
        self.gps_update_count = 0
        self.uwb_update_count = 0
        self.uwb_rejected_count = 0
        self._gps_times.clear()
        self._uwb_times.clear()

        # ADDED 2025-11-27: Reset range rate tracking
        self._prev_range = None
        self._prev_range_time = None

        # Multi-anchor tracking reset
        self._prev_range_by_anchor.clear()
        self._prev_range_time_by_anchor.clear()
        self._uwb_updates_by_anchor.clear()
        self._uwb_rejected_by_anchor.clear()
        self._unknown_anchor_count = 0
        self._last_innovation_by_anchor.clear()

        # Adaptive-β: clear detector/ramp state so a new episode starts un-latched (β at floor).
        self._beta_cusum = 0.0
        self._beta_active = False
        self._beta_gate = 0.0
        self._beta_count = 0
        self._last_gps_behind = None
        # Surge gate-relief: clear detector state so a new episode starts un-armed.
        self._relief_cusum = 0.0
        self._relief_signsum = 0.0
        self._relief_active = False
        self._relief_count = 0
        self._relief_prev_rej = 0
        self._relief_prev_upd = 0
        self._relief_prev_innov = 0.0
        # Behind ring-throttle (#405 D2): clear detector/ramp state so a new episode starts
        # un-armed (behind beta un-throttled). _ring_mode is config (read once) -> NOT reset.
        self._ring_count = 0
        self._ring_cusum = 0.0
        self._ring_active = False
        self._ring_gate = 0.0
        self._ring_last_w = 1.0

        logger.info(
            f"Filter reset to ({initial_behind:.1f}, {initial_cross:.1f}, {initial_below:.1f})"
        )

    def get_diagnostics(self, now_s: Optional[float] = None) -> dict:
        """
        Get diagnostic information for debugging.

        Args:
            now_s: Optional current time in caller timebase (sim-time seconds).
                   Passed to get_state() for age calculations.

        Returns:
            Dictionary with filter statistics
        """
        gps_rate, uwb_rate = self.get_rates()
        state = self.get_state(now_s=now_s)

        diag = {
            # Existing keys (unchanged for backwards compatibility)
            "gps_updates": self.gps_update_count,
            "uwb_updates": self.uwb_update_count,
            "uwb_rejected": self.uwb_rejected_count,
            "gps_rate_hz": gps_rate,
            "uwb_rate_hz": uwb_rate,
            "gps_age_s": state.gps_age_s,
            "uwb_age_s": state.uwb_age_s,
            "position_variance_m2": (self.P[0][0] + self.P[1][1] + self.P[2][2]) / 3,
            "velocity_variance_m2s2": (self.P[3][3] + self.P[4][4] + self.P[5][5]) / 3,
            "last_innovation_range_m": self.last_innovation_range,
            # Multi-anchor diagnostics (new)
            "uwb_updates_by_anchor": dict(self._uwb_updates_by_anchor),
            "uwb_rejected_by_anchor": dict(self._uwb_rejected_by_anchor),
            "unknown_anchor_count": self._unknown_anchor_count,
            "last_innovation_range_by_anchor": dict(self._last_innovation_by_anchor),
            "anchor_count": len(self.anchor_positions),
        }
        # D1 (#405 STEP-2) channel-iso gauge: expose counts ONLY when enabled, so get_diagnostics
        # stays byte-identical when the gauge is off.
        if self._chaniso_enabled:
            diag["channel_iso_rows_written"] = self._chaniso_rows_written
            diag["channel_iso_last"] = self._chaniso_last
        if self._gatediag_on:
            diag["gate_diag_rows_written"] = self._gatediag.rows_written
        if self._ring_mode:
            diag["ring_throttle_mode"] = self._ring_mode      # "C" | "S"
            diag["ring_throttle_active"] = self._ring_active   # Form-C armed (fired?)
            diag["ring_throttle_gate"] = self._ring_gate       # Form-C ramp [0,1]
            diag["ring_throttle_last_w"] = self._ring_last_w   # last robust weight applied
        return diag


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================


def create_gps_measurement(
    timestamp: float,
    behind_m: float,
    cross_m: float,
    below_m: float,
    vel_behind_m_s: float = 0.0,
    vel_cross_m_s: float = 0.0,
    vel_below_m_s: float = 0.0,
) -> GPSMeasurement:
    """Create a GPS measurement from relative geometry values."""
    return GPSMeasurement(
        timestamp=timestamp,
        behind_m=behind_m,
        cross_m=cross_m,
        below_m=below_m,
        vel_behind_m_s=vel_behind_m_s,
        vel_cross_m_s=vel_cross_m_s,
        vel_below_m_s=vel_below_m_s,
    )


def create_uwb_measurement(
    timestamp: float,
    range_m: float,
    anchor_id: int = 0,
) -> UWBMeasurement:
    """Create a UWB measurement from range sensor."""
    return UWBMeasurement(
        timestamp=timestamp,
        range_m=range_m,
        anchor_id=anchor_id,
    )


def compute_range_for_anchor(
    behind_m: float,
    cross_m: float,
    below_m: float,
    anchor_offset_bcb: Tuple[float, float, float],
) -> float:
    """
    Compute the range from a UWB anchor to the follower.

    This helper function uses the SAME convention as update_uwb():
    - behind_m, cross_m, below_m: Follower position in leader body frame
    - anchor_offset_bcb: Anchor position (behind, cross, below) in leader body frame

    Range is computed as the Euclidean distance from anchor to follower.

    Used by both the fusion filter's measurement model and the UWB emitter
    to ensure consistent geometry calculations.

    Args:
        behind_m: Follower along-track position (positive = behind leader)
        cross_m: Follower cross-track position (positive = right of leader)
        below_m: Follower vertical position (positive = below leader)
        anchor_offset_bcb: (anchor_behind_m, anchor_cross_m, anchor_below_m)

    Returns:
        Range in meters from anchor to follower
    """
    ax, ay, az = anchor_offset_bcb
    dx = behind_m - ax
    dy = cross_m - ay
    dz = below_m - az
    return math.sqrt(dx**2 + dy**2 + dz**2)


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    """Self-test: simulate GPS + UWB updates and verify filter convergence."""
    import random

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("UWB + GPS Fusion Module Self-Test")
    print("=" * 60)

    # Create fusion filter
    fusion = RelativePoseFusion()

    # True state (simulated)
    true_behind = 10.0  # 10m behind leader
    true_cross = 5.0  # 5m right of track
    true_below = 8.0  # 8m below leader

    # Simulate 10 seconds of updates
    dt_gps = 0.2  # 5 Hz GPS
    dt_uwb = 0.1  # 10 Hz UWB
    t = 0.0

    for step in range(100):
        t = step * 0.1

        # GPS update every 0.2s
        if step % 2 == 0:
            gps = create_gps_measurement(
                timestamp=t,
                behind_m=true_behind + random.gauss(0, GPS_POS_SIGMA_M),
                cross_m=true_cross + random.gauss(0, GPS_POS_SIGMA_M),
                below_m=true_below + random.gauss(0, GPS_POS_SIGMA_M),
            )
            fusion.update_gps(gps)

        # UWB update every 0.1s
        true_range = math.sqrt(true_behind**2 + true_cross**2 + true_below**2)
        uwb = create_uwb_measurement(
            timestamp=t,
            range_m=true_range + random.gauss(0, UWB_RANGE_SIGMA_M),
        )
        fusion.update_uwb(uwb)

    # Final state (pass sim-time to get_state)
    final_time = t  # Last sim-time from loop (9.9s)
    state = fusion.get_state(now_s=final_time)
    print(
        f"\nTrue state:   behind={true_behind:.1f}m, cross={true_cross:.1f}m, below={true_below:.1f}m"
    )
    print(
        f"Fused state:  behind={state.behind_m:.1f}m, cross={state.cross_m:.1f}m, below={state.below_m:.1f}m"
    )
    print(f"Range: true={true_range:.1f}m, fused={state.range_m:.1f}m")

    # Diagnostics (pass sim-time for accurate age calculation)
    diag = fusion.get_diagnostics(now_s=final_time)
    print(f"\nDiagnostics:")
    print(f"  GPS updates: {diag['gps_updates']}, rate: {diag['gps_rate_hz']:.1f} Hz")
    print(
        f"  UWB updates: {diag['uwb_updates']}, rejected: {diag['uwb_rejected']}, rate: {diag['uwb_rate_hz']:.1f} Hz"
    )
    print(f"  Position variance: {diag['position_variance_m2']:.2f} m^2")

    # Check convergence
    behind_err = abs(state.behind_m - true_behind)
    cross_err = abs(state.cross_m - true_cross)
    below_err = abs(state.below_m - true_below)

    print(f"\nErrors: behind={behind_err:.2f}m, cross={cross_err:.2f}m, below={below_err:.2f}m")

    if behind_err < 2.0 and cross_err < 2.0 and below_err < 2.0:
        print("\n✅ PASS: Single-anchor filter converged within 2m error")
        single_anchor_pass = True
    else:
        print("\n❌ FAIL: Single-anchor filter did not converge")
        single_anchor_pass = False

    print("=" * 60)

    # ==========================================================================
    # Multi-Anchor Test Scenario
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Multi-Anchor Test Scenario")
    print("=" * 60)

    # Define anchor positions on the leader aircraft (boom tip and fuselage)
    # Anchor 0: Boom tip (origin)
    # Anchor 1: 3m behind boom tip (on fuselage)
    # Anchor 2: 2m right of boom tip (wingtip)
    # Anchor 3: 1.5m below boom tip (belly)
    anchor_positions = {
        0: (0.0, 0.0, 0.0),  # Boom tip (origin)
        1: (3.0, 0.0, 0.0),  # 3m behind tip (fuselage)
        2: (0.0, 2.0, 0.0),  # 2m right (wingtip)
        3: (0.0, 0.0, 1.5),  # 1.5m below (belly)
    }

    fusion_multi = RelativePoseFusion(anchor_positions=anchor_positions)

    # True follower state (same as single-anchor test)
    print(
        f"\nTrue follower position: behind={true_behind:.1f}m, cross={true_cross:.1f}m, below={true_below:.1f}m"
    )
    print(f"Anchor positions: {anchor_positions}")

    # Simulate 10 seconds with cycling through anchors
    t = 0.0
    for step in range(100):
        t = step * 0.1

        # GPS update every 0.2s
        if step % 2 == 0:
            gps = create_gps_measurement(
                timestamp=t,
                behind_m=true_behind + random.gauss(0, GPS_POS_SIGMA_M),
                cross_m=true_cross + random.gauss(0, GPS_POS_SIGMA_M),
                below_m=true_below + random.gauss(0, GPS_POS_SIGMA_M),
            )
            fusion_multi.update_gps(gps)

        # UWB update every 0.1s - cycle through anchors
        anchor_id = step % 4
        ax, ay, az = anchor_positions[anchor_id]

        # True range from this anchor to follower
        dx = true_behind - ax
        dy = true_cross - ay
        dz = true_below - az
        true_range_from_anchor = math.sqrt(dx**2 + dy**2 + dz**2)

        uwb = create_uwb_measurement(
            timestamp=t,
            range_m=true_range_from_anchor + random.gauss(0, UWB_RANGE_SIGMA_M),
            anchor_id=anchor_id,
        )
        fusion_multi.update_uwb(uwb)

    # Final state
    final_time = t
    state_multi = fusion_multi.get_state(now_s=final_time)
    print(
        f"\nFused state:  behind={state_multi.behind_m:.1f}m, cross={state_multi.cross_m:.1f}m, below={state_multi.below_m:.1f}m"
    )

    # Diagnostics
    diag_multi = fusion_multi.get_diagnostics(now_s=final_time)
    print(f"\nDiagnostics:")
    print(f"  GPS updates: {diag_multi['gps_updates']}, rate: {diag_multi['gps_rate_hz']:.1f} Hz")
    print(
        f"  UWB updates: {diag_multi['uwb_updates']}, rejected: {diag_multi['uwb_rejected']}, rate: {diag_multi['uwb_rate_hz']:.1f} Hz"
    )
    print(f"  Anchor count: {diag_multi['anchor_count']}")
    print(f"  UWB updates by anchor: {diag_multi['uwb_updates_by_anchor']}")
    print(f"  Position variance: {diag_multi['position_variance_m2']:.2f} m^2")

    # Check convergence
    behind_err_multi = abs(state_multi.behind_m - true_behind)
    cross_err_multi = abs(state_multi.cross_m - true_cross)
    below_err_multi = abs(state_multi.below_m - true_below)

    print(
        f"\nErrors: behind={behind_err_multi:.2f}m, cross={cross_err_multi:.2f}m, below={below_err_multi:.2f}m"
    )

    if behind_err_multi < 2.0 and cross_err_multi < 2.0 and below_err_multi < 2.0:
        print("\n✅ PASS: Multi-anchor filter converged within 2m error")
        multi_anchor_pass = True
    else:
        print("\n❌ FAIL: Multi-anchor filter did not converge")
        multi_anchor_pass = False

    print("=" * 60)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Single-anchor test: {'PASS' if single_anchor_pass else 'FAIL'}")
    print(f"Multi-anchor test:  {'PASS' if multi_anchor_pass else 'FAIL'}")
    print("=" * 60)
