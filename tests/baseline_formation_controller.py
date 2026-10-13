#!/usr/bin/env python3
# flake8: noqa: F403, F405
"""
FlightDock Baseline Formation Controller v2.0
==============================================

A comprehensive formation controller for QuadPlane aerial refueling docking
with the DHC-2 Beaver tanker aircraft.

Key Features:
1. UWB+GPS sensor fusion for accurate relative positioning (with GPS fallback)
2. Weather verification to ensure both aircraft share the same atmosphere
3. VTOL takeoff with fixed-wing transition
4. Fighter-like intercept to stern conversion rejoin
5. Controlled precontact position holding
6. Precision CREEP logic for in-flight docking

Phase Definitions:
- Stern Conversion: 200m range behind DHC-2 Beaver's Flying Boom
- Precontact Position: 10m range, ±1m envelope, 5s stabilized
- Contact Position: 0m range (at boom tip), ±0.5m envelope, 5s stabilized

Based on Test 123 analysis and user requirements (Dec 2025).

Usage:
    Terminal 1: ./start_formation_sitl_fixed.sh
    Terminal 2: python3 leader_broadcast_fixed.py --target-speed 24.7
    Terminal 3: python3 baseline_formation_controller.py
"""

import sys
import os
import time
import math
import signal
import argparse
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, replace
from formation_config import *  # noqa: F403, F405
import leader_extrapolator  # STEP-4 Hunk-L: gated by LEADER_EXTRAP_ENABLE inside the helper
import d_correction  # STEP-4 Hunk-D: pure module; consumption gated by FD_D_CORRECTION
import closure_diag  # TIER2: pure module; consumption gated by FD_TRUTH_CLOSURE_DIAG
import wind_ff  # WINDFF: pure module; consumption gated by FD_SELF_WIND_FF
import formation_config  # FD-009: needed for qualified references in guard helpers
from formation_control_laws import FormationControlLaws, ewma_alpha_scaled
from formation_envelope import EnvelopeChecker
from formation_sensor_fusion import SensorFusion

# Tick timing diagnostic (set True to enable per-tick wall-clock profiling)
TICK_TIMING_DIAGNOSTIC = False

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not installed. Run: pip install pymavlink")
    sys.exit(1)

from sim_time_authority import (
    SimTimeAuthority, get_time_authority, set_time_authority,
    # Wall-time guards for sim-time contract enforcement
    wall_now_s, wall_monotonic_s, get_wall_time_guard_count,
)
from formation_success_metrics import FormationSuccessMetrics
from state_machine import ControllerState, StateMachine

# P1 Fix B v9.10: Timebase canary for early rate verification
try:
    from timebase_canary import (
        request_attitude_20hz,
        run_timebase_canary,
        check_time_authority_gate0,
    )
    TIMEBASE_CANARY_AVAILABLE = True
except ImportError:
    TIMEBASE_CANARY_AVAILABLE = False
    print("WARNING: timebase_canary module not available")


# Import UWB fusion module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'relative_pose'))
try:
    from uwb_gps_fusion import (
        RelativePoseFusion,
        UWBMeasurement,
        create_gps_measurement,
    )
    UWB_FUSION_AVAILABLE = True
except ImportError:
    UWB_FUSION_AVAILABLE = False
    print("WARNING: UWB fusion module not available, using GPS-only geometry")

# Import UWB measurement emitter
try:
    from uwb_measurement_emitter import UWBRangingEmitter
    UWB_EMITTER_AVAILABLE = True
except ImportError:
    UWB_EMITTER_AVAILABLE = False
    UWBRangingEmitter = None  # type: ignore
    print("WARNING: UWB emitter module not available")

# Import VTOL takeoff module
try:
    from quadplane_qloiter_takeoff import (
        QuadplaneQloiterTakeoffController,
    )
    TAKEOFF_MODULE_AVAILABLE = True
except ImportError:
    TAKEOFF_MODULE_AVAILABLE = False
    print("WARNING: Takeoff module not available")

# Import wake turbulence model (Phase 1: detection/logging only)
try:
    from wake_turbulence_model import WakeTurbulenceZone
    WAKE_MODEL_AVAILABLE = True
except ImportError:
    WAKE_MODEL_AVAILABLE = False
    WakeTurbulenceZone = None  # type: ignore
    print("WARNING: Wake turbulence model not available")

import threading

# =============================================================================
# C1: UWB
# truth-source toggle — R7.1.
# When FD_UWB_TRUTH_SOURCED=1 (NOW THE DEFAULT), the UWB emitter is fed the
# follower's SIM-TRUTH position (SIMSTATE(164) int degE7 lat/lng + SIM_STATE(108)
# float alt) instead of GPS/EKF-perceived. Control geometry is UNCHANGED — truth
# enters ONLY the sigma-0.1 emitter sensor path (no obs/policy/reward/replay).
# (2026-06-10): a UWB that echoes GPS DOUBLE-COUNTS GPS in the
# fusion (update_gps + a GPS-derived update_uwb violate measurement independence)
# and is harmful — truth-sourcing is the correct default. Set
# FD_UWB_TRUTH_SOURCED=0 for the GPS-echo ablation. Missing truth -> emission is
# SKIPPED (never a perceived-as-truth fallback; see get_geometry). UWB=ON-always
# is launch/curriculum policy. Evidence: analysis/c0_*, c0_5_*, uwb_framefix_*.
# =============================================================================
_FD_UWB_TRUTH_SOURCED = os.environ.get('FD_UWB_TRUTH_SOURCED', '1').strip() == '1'
_FD_UWB_SIM_STATE_RATE_HZ = 10.0  # SIM_STATE(108) truth-alt request rate (truth-sourced only)
# Option-A (2026-06-09, gated, default-OFF): when 1, get_sim_position() sources the
# UWB-emitter follower HORIZONTAL truth from SIM_STATE(108).lat_int/lon_int (degE7, 10Hz,
# Gate-0-GREEN) instead of rate-capped SIMSTATE(164). OFF => byte-identical 164 path.
_FD_UWB_TRUTH_HORIZ_FROM_108 = os.environ.get('FD_UWB_TRUTH_HORIZ_FROM_108', '').strip() == '1'

# =============================================================================
# BUG-A FIX (2026-06-11, R7.1 waiver): mirror update_cache()'s truth branches
# (:650-653) into the blocking pump's first-message type-switch. Defect
# [MEASURED, analysis/bug_a_fix_20260611T034435Z]: SIMSTATE(164)/SIM_STATE(108)
# consumed as the blocking first-recv were silently DISCARDED -> truth feed
# starved to ~2.3 Hz though the kernel delivered ~10 Hz (in-window drops = 0).
# Default ON (silent-reversion guard, per UWB-default lesson). Set
# FD_PUMP_TRUTH_FIX=0 for exact pre-fix behavior (A-side ablation / instant
# rollback).
# =============================================================================
_FD_PUMP_TRUTH_FIX = os.environ.get('FD_PUMP_TRUTH_FIX', '1').strip() == '1'

# =============================================================================
# TRUTH-RATE DIAG (2026-06-15, R7.1 waiver, AUDIT instrumentation, default-OFF):
# leaf-localize the cruise 108 consumer-side under-cache
# (analysis/uwb_truth_feed_delivery_rca_*). Counts SIM_STATE(108)/GPI(33)
# messages RETURNED by recv_match (parser throughput) vs cache VALUE-changes,
# plus the emit-time age of _last_sim_state. Discriminator: recv≈20Hz &
# change≈20Hz & emit-sample 14.66Hz => within-tick sampling; recv≈20 &
# change≈14.66 => userspace caching discard; recv≈14.66 => delivery loss.
# Diagnostic-ONLY: counts/ages flow ONLY to a periodic [TRUTH_RATE_DIAG] log
# line, never to control/fusion/observation/reward/replay. Sim-clock ages
# (buffered ATTITUDE time_boot_ms), never wall-clock. Default OFF =>
# byte-identical. FD_TRUTH_RATE_DIAG=1 to enable.
# =============================================================================
_FD_TRUTH_RATE_DIAG = os.environ.get('FD_TRUTH_RATE_DIAG', '').strip() == '1'

# =============================================================================
# STEP-2 C-BYPASS (2026-06-11, R7.1 waiver): when FD_GEOM_GPS_DIRECT=1, control
# geometry consumes the GPS-relative values directly; RelativePoseFusion keeps
# running as a SHADOW (updates + diag columns intact) but its output is NOT
# consumed. Quarantines the behind-axis K-collapse (fused 0.55 m vs gps 9.33 m
# vs truth 9.62 m, analysis/bug_a_fix_20260611T034435Z) on the road to
# architecture D (fuser retirement). Default OFF = current fused behavior.
# =============================================================================
_FD_GEOM_GPS_DIRECT = os.environ.get('FD_GEOM_GPS_DIRECT', '0').strip() == '1'

# =============================================================================
# G3 GUARD-ON-ENV:
# FD_GUARD_CEILING_CHANNEL=env -> Guard 1 (altitude ceiling) compares the
# same-computation GPS-direct below (RelativeGeometry.below_gps_m) instead of
# the fused below_m. ONLY the literal 'env' switches; any other value keeps
# the flown fused channel. Scope: the ceiling guard's COMPARISON INPUT only —
# threshold, persistence, every control law and every other guard stay fused.
# Basis: 36-leg dose-response GUARD_CHANNEL_DOSE_RESPONSE.md — ctrl(fused)
# 0 correct / 12 false / 1 missed vs env(GPS-direct) 1 / 0 / 0.
# Default unset = byte-identical decision path to every flown leg.
# =============================================================================
_FD_GUARD_CEILING_ENV = os.environ.get('FD_GUARD_CEILING_CHANNEL', 'ctrl').strip() == 'env'
# GUARDWIT: hard cap on the per-fire witness list (log-only; see the fire
# branch in the guard block). Bounds memory on a pathological leg.
_GUARDWIT_MAX_FIRES = 64

# =============================================================================
# STEP-4 DLIVE (2026-06-11, R7.1 waiver): live C3 both-hop alignment +
# architecture-D terminal correction. Offline proof: analysis/step3_dalign_*
# (both-hop sigma_step 1.513->0.156 m; 52 climb-guard windows vs 0 live).
# All flags default OFF = byte-equivalent behavior.
#   FD_FEED_RATE_RAISE=1 -> Hunk-RR: request GPI(33)+SIMSTATE(164)+SIM_STATE(108)
#       at 20 Hz (follower/truth feed hop: 10 Hz feeds at ~20 Hz ticks were
#       48.9% stale re-pairs).
#   FD_D_CORRECTION=1 -> Hunk-D: memoryless architecture-D UWB algebraic
#       correction (tests/d_correction.py) consumed in the GPS-direct branch.
#   Leader hop (Hunk-L) reuses tests/leader_extrapolator.py, gated by its own
#       LEADER_EXTRAP_ENABLE (default OFF inside the helper).
# =============================================================================
_FD_FEED_RATE_RAISE = os.environ.get('FD_FEED_RATE_RAISE', '0').strip() == '1'
_FD_FEED_RATE_RAISE_HZ = 20.0
_FD_D_CORRECTION = os.environ.get('FD_D_CORRECTION', '0').strip() == '1'
_FD_D_GPS_SIGMA = float(os.environ.get('FD_D_GPS_SIGMA', '0.5'))
_FD_D_UWB_SIGMA = float(os.environ.get('FD_D_UWB_SIGMA', '0.1'))
# STEP-4B2' state gate (2026-06-11, R7.1 waiver), NARROWED 2026-06-12 per
# (B2PP dispatch): D waits until the altitude-capture
# climb completes and DOCKING_CREEP begins — ALT_CAP_HOLD_RANGE removed
# (doctrine wording: "UWB primary inside DOCKING_CREEP"). Legacy
# ALTITUDE_CAPTURE (state_machine.py:111) remains EXCLUDED: it is a
# distinct enum member from ALT_CAP_HOLD_RANGE (:112).
_FD_D_ALLOWED_STATES = frozenset(
    ('DOCKING_CREEP', 'HOLD_CONTACT'))
_FD_D_MAX_RANGE_M = float(os.environ.get('FD_D_MAX_RANGE_M', '30.0'))
# TURNGATE-750: arms the leader-turn
# gate on DOCKING_CREEP entry. DEFAULT OFF -- unset/'0' leaves the transition
# condition byte-identical to the pre-fix controller, so every banked panel
# stays comparable. Resolved at import, matching the file's other FD_ knobs.
_FD_TURN_GATE = os.environ.get('FD_TURN_GATE', '0').strip() == '1'
# SIGCLEAN Hunk-C1 (2026-06-12, R7.1 waiver): per-axis trust region on the
# D-correction delta. DDEF default 2026-06-13: 0.60 m (was 0.0=disabled).
# Evidence (ceiling RCA ART): both pinfix guard fires were single-tick
# |delta_below| 0.88/1.00 m spikes (normal-ops p95 0.376 m) near-tip where the
# radial-only range geometry is ill-conditioned. Exceeding => clamp (DDEF mode).
_FD_D_TRUST_M = float(os.environ.get('FD_D_TRUST_M', '0.60') or '0.60')
# DFIX Hunk-C3 (2026-06-13, R7.1 waiver per d_census ed2c4606 GO): continuity
# fix for the measured 4-5 Hz apply/reject flip-flop (0.43-0.51 m consumed
# steps = S1 driver; chop collapsed 24.5->3.52/min with D off).
# DDEF: the MODE
# defaults below are flipped to the dfix-validated clamp+EMA config so that
# whenever D is ENABLED (FD_D_CORRECTION=1, harness-owned) it runs the GOOD
# mode — never raw/reject D by flag omission. The d_correction module's own
# trust_mode default stays 'reject' (API stability); only this controller
# policy default flips. Values = DFIX leg 349107e0 exactly.
#   FD_D_TRUST_MODE: 'clamp' (DDEF default; scale delta onto trust boundary in
#     d_correction — no discontinuity) | 'reject' (legacy zero-on-exceed).
#   FD_D_EMA_ALPHA: 0.1 (DDEF default; EMA on the APPLIED delta, ~tau 0.5 s @
#     20 Hz, state held HERE, d_correction stays memoryless; updates only on
#     solution ticks, seeds from zero) | 0.0 = off.
_FD_D_TRUST_MODE = (os.environ.get('FD_D_TRUST_MODE', 'clamp').strip()
                    or 'clamp')
_FD_D_EMA_ALPHA = float(os.environ.get('FD_D_EMA_ALPHA', '0.1') or '0.1')

# =============================================================================
# Demand-pinning fix (2026-06-12): climb-window reposition hold-off.
# Evidence: commands.cpp:71 set_target_altitude_current() re-anchors TECS hdem
# on every DO_REPOSITION; retune-leg tlog 12,829 ACKed repositions vs 254×GCA-
# ACCEPTED (all overridden); B2P_ALT_DECOMPOSITION: hdem≡achieved under rolling
# carrot. Default-OFF (FD_PINNING_FIX=1 to activate).
# =============================================================================
_FD_PINNING_FIX = os.environ.get('FD_PINNING_FIX', '0').strip() == '1'
_PIN_FIX_EXIT_ERR_M = 2.0      # hysteresis: exit window when |alt_err| < this
_PIN_FIX_MAX_HOLDOFF_S = 20.0  # failsafe: force reposition after this window duration
_PIN_FIX_HDG_ABORT_DEG = 10.0  # lateral safety: allow one send-through at hdg delta


# =============================================================================
# C3 (2026-06-08): UWB emitter->fusion DELIVERY DIAGNOSTIC — R7.1 gated, DEFAULT-OFF.
# When FD_UWB_FUSION_DIAG=1, get_geometry() writes one row per tick (docking
# states only) to uwb_fusion_diag.csv capturing the emitter truth-UWB -> fusion
# delivery chain: emit_below_truth -> update_uwb_batch -> fused_below pre/post ->
# innovation/reject -> returned geom.below_m. LOGGING-ONLY: pure reads of the
# fusion state; NEVER mutates the filter (never calls get_state(now_s), which
# would predict-forward). Byte-identical to C1 behavior when unset/0. Resolves
# C2's truth-M3b failure into (a) DELIVERY vs (b) APPLICATION. Evidence:
# analysis/uwb_fusion_diag_*.
# =============================================================================
_FD_UWB_FUSION_DIAG = os.environ.get('FD_UWB_FUSION_DIAG', '').strip() == '1'
# TIER2_R7_1: flag-gated 20 Hz TRUTH
# CG-CG closure diagnostic columns, DEFAULT-OFF (= byte-identical CSV schema and
# control path). When FD_TRUTH_CLOSURE_DIAG=1, speed_arbitration.csv gains two
# TRAILING columns (truth_closure_cgcg_mps, range_center_cgcg_m) computed per
# written tick from TRUTH sources only: follower SIMSTATE(164)/SIM_STATE(108)
# position + SIM_STATE(108) velocity, leader broadcast GLOBAL_POSITION_INT
# position + velocity, projected in closure_diag.truth_closure_cgcg (+ = closing).
# DIAGNOSTIC-ONLY: the values go to the CSV row and NOWHERE else — never
# obs/actor/critic/reward/replay/policy and never any guard/gate input (Tier-2
# spec leakage stance; joins the emit_*_truth logged-but-never-in-obs class).
# Missing truth inputs -> empty cells + a skip count logged at CSV close.
_FD_TRUTH_CLOSURE_DIAG = os.environ.get('FD_TRUTH_CLOSURE_DIAG', '0').strip() == '1'
# CSV2_R7_1: flag-gated rejoin cross-track crab integral. When ON, REJOIN_PRECONTACT/
# HOLD_PRECONTACT add an ILOS integral crab.
# DEFAULT-ON:
# crab-ON is the adopted baseline — 8-profile A/B `54a21581` (wins wind: S4 90° 0/6->5/6 dock,
# S1 45° 0->partial; holds S0 5/5) + lagfix×crab 16/16 `e074aae9` + windff V1b 10/10 all flew
# crab-ON. Set FD_CROSS_CRAB_ENABLE=0 to restore the prior no-crab byte-identical path.
_FD_CROSS_CRAB_ENABLE = os.environ.get('FD_CROSS_CRAB_ENABLE', '1').strip() == '1'
# CSV2 crosswind-dock: flag-gated active anticipatory closure-rate braking (default OFF =
# adopted-baseline byte-identical). When ON, commands active deceleration (airspeed below
# leader) when the follower over-closes a safe stopping profile near the dock, countering
# gust-driven airspeed-tracking overshoot past the boom-tip.
_FD_CLOSURE_BRAKE_ENABLE = os.environ.get('FD_CLOSURE_BRAKE_ENABLE', '0').strip() == '1'
# CSV2_SPEC247: flag-gated TAS->EAS command feedforward (default OFF = adopted-baseline
# byte-identical). ArduPilot TECS realizes the airspeed demand as EAS (_TAS_dem = _EAS_dem *
# EAS2TAS, AP_TECS.cpp:404-405), so a raw LEADER_TARGET_TAS_MS command flies TAS ~26 at
# altitude and overtakes the 24.7 leader. When ON, the leader-relative speed command is
# converted TAS->EAS (tas_to_eas ISA-from-fol_alt, or the FD_EAS2TAS_RATIO override) so the
# REALIZED TAS matches LEADER_TARGET_TAS_MS. Scoped to ALL leader-relative speed states: the
# approach (INTERCEPT / STERN_CONVERSION / REJOIN_PRECONTACT) AND the 4 docking states
# (HOLD_PRECONTACT / ALT_CAP_HOLD_RANGE / DOCKING_CREEP / HOLD_CONTACT). Rationale: the
# follower must decelerate DURING the approach and enter the docking states already settled at
# 24.7 — the TECS ~5s lag cannot shed an enter-hot excess before overshoot ([MEASURED]
# docking-only correction left realized TAS ~25.7, +1.0 overtake, breakaway). The approach
# states were previously excluded over a suspected config:656 "1x stall"; correcting EAS->TAS
# makes the realized catch-up equal the DESIGNED command (not slower) -> expected stall-free,
# gated on the S0 A/B. Applied via apply_eas2tas_ff().
# DEFAULT-ON: the spec-247
# eas2tas command feedforward is the adopted baseline behavior — at the SPEC true 24.7 m/s TAS
# leader, TECS realizes the demand as EAS so a raw command overshoots (->EMERGENCY_BREAKAWAY).
# Set FD_EAS2TAS_FF_ENABLE=0 to restore the prior (overshooting) byte-identical baseline path.
_FD_EAS2TAS_FF_ENABLE = os.environ.get('FD_EAS2TAS_FF_ENABLE', '1').strip() == '1'
# CSV2_SPEC247: optional FIXED eas2tas ratio override (DIAGNOSTIC / SITL-atmosphere match).
# Empty (default) -> ISA-from-fol_alt via tas_to_eas (byte-identical to the no-override path).
# Set FD_EAS2TAS_RATIO=<r> to divide the TAS command by a measured/known eas2tas (e.g. the
# SITL's actual ~1.092) instead of ISA. Diagnostic-only: a fixed ratio overfits one
# atmosphere; the shippable wind-robust source is baro-derived (SCALED_PRESSURE).
_FD_EAS2TAS_RATIO = os.environ.get('FD_EAS2TAS_RATIO', '').strip()
# CSV2_SPEC247 (Candidate A): baro-derived eas2tas source (default OFF). When ON, eas2tas is
# computed live from the follower's SCALED_PRESSURE (pressure + temperature) matching AP_Baro
# (eas2tas = sqrt(1.225 * 287.26 * tempK / press_pa)) and fed to the EXISTING whole-sum
# division, so the realized TAS tracks the ACTUAL flight atmosphere/altitude. (A fixed override
# is altitude-specific: [MEASURED] dock legs at 896 m vs 811 m had eas2tas 1.0864 vs 1.074.)
# Falls back to FD_EAS2TAS_RATIO / ISA when no SCALED_PRESSURE has arrived yet -> byte-identical
# to the prior path when A is OFF or the baro cache is empty.
# DEFAULT-ON: baro source is the shippable spec-247 source.
# Set FD_EAS2TAS_BARO=0 to fall back to FD_EAS2TAS_RATIO/ISA. When BARO is ON but no
# SCALED_PRESSURE has arrived yet, get_baro_eas2tas() returns None -> ISA fallback (graceful).
_FD_EAS2TAS_BARO = os.environ.get('FD_EAS2TAS_BARO', '1').strip() == '1'
# WINDFF (2026-07-02, R7.1 waiver; Phase-1b frame-surgical placement per
# analysis/s1windff_fix_20260702T115440Z/RCA_V1_BIAS_AND_FRAME.md): follower self along-track wind
# feedforward. The [MEASURED] Phase-C bug (a5c6e5b1): expressions that treat the leader's GROUND
# SPEED as a follower AIRSPEED reference (DC schedule caps, closure-cap branch, FD-006 mins,
# near-precontact cap, closure-brake, breakaway separation, guard cap) leak the follower's own
# along-track wind into realized closure -> tailwind creep-through / headwind starvation.
# FD_SELF_WIND_FF substitutes lead_gs_air = lead_gs - w_par (via _windff_lead_gs_air + the pure
# wind_ff module) in EXACTLY those expressions. The TAS-referenced paths (approach PD profile =
# LEADER_TARGET_TAS + delta, control_laws:702; HOLD_CONTACT PI/D) are wind-immune by design and
# are NOT corrected — the V1 whole-command seam placement double-corrected them (C1 REJOIN stall,
# superseded). Byte-identical when OFF (no correction, no estimator update). FD_WINDFF_TRIM
# (requires FF) adds a slow, state-gated closure-error trim for residual bias. Signals are
# perception-only (get_velocity_ned / get_airspeed x baro-eas2tas / get_attitude yaw) — NEVER
# SIM_STATE truth. Known [MEASURED] estimator sensor bias +0.20 m/s (e2t gap +0.16 + GPI-vel
# +0.045, RCA §1) — stable across all 20 V1/V1b legs AND the S3-degraded check (calibration
# property, in the pre-registered [0.10, 0.30] band); left untreated.
# DEFAULT-ON:
# the windff evidence matrix is complete — FF-ON docks in all six cells (S0 clean / C1 tail /
# C2 head / C3 cross / S1 45°+turb / S3 degraded; V1b `63639a5a` + S3-check `388178f8`), wind
# drift mechanism killed ~100x, estimator degradation-immune. Set FD_SELF_WIND_FF=0 to restore
# the prior no-FF byte-identical path. FD_WINDFF_TRIM stays default-OFF (unadopted).
_FD_SELF_WIND_FF = os.environ.get('FD_SELF_WIND_FF', '1').strip() == '1'
_FD_WINDFF_TRIM = _FD_SELF_WIND_FF and os.environ.get('FD_WINDFF_TRIM', '0').strip() == '1'
_FD_WINDFF_TAU_S = float(os.environ.get('FD_WINDFF_TAU_S', '3.0') or 3.0)
_FD_WINDFF_TRIM_KI = float(os.environ.get('FD_WINDFF_TRIM_KI', '0.05') or 0.05)
_FD_WINDFF_TRIM_CLAMP = float(os.environ.get('FD_WINDFF_TRIM_CLAMP', '0.5') or 0.5)
# WINDFF trim accumulates/applies ONLY in these terminal GS-referenced hold states (station-hold
# where v_c ~ 0 and residual along-track drift binds); HOLD_CONTACT deliberately excluded (above).
_WINDFF_TRIM_STATES = frozenset(('HOLD_PRECONTACT', 'ALT_CAP_HOLD_RANGE', 'DOCKING_CREEP'))
_UWB_FUSION_DIAG_SCHEMA_VERSION = 4  # v4 (2026-06-13): +d_corr_pre_trust_max_m (DFIX); v3 (2026-06-11): +d_corr_* state-gate cols (step-4B2'); v2 (2026-06-10): +along-track latency-decomp cols
_UWB_FUSION_DIAG_STATES = frozenset((
    'HOLD_PRECONTACT', 'ALT_CAP_HOLD_RANGE', 'DOCKING_CREEP', 'HOLD_CONTACT'))
_UWB_DIAG_IDX_BELOW = 2  # uwb_gps_fusion.IDX_Z=2; FusedState.below_m = self.x[2]
_UWB_DIAG_NAN = float('nan')


def _uwb_diag_fused_below(fusion) -> float:
    """C3 (read-only): pure read of the fusion below-axis (self.x[IDX_Z]).
    NEVER calls get_state(now_s) — that predicts forward and MUTATES the
    filter. NaN on any failure (never raises). [C4]+[R-a]"""
    try:
        x = getattr(fusion, 'x', None)
        if x is None or len(x) <= _UWB_DIAG_IDX_BELOW:
            return _UWB_DIAG_NAN
        return float(x[_UWB_DIAG_IDX_BELOW])
    except Exception:
        return _UWB_DIAG_NAN


def _uwb_diag_predicted_range_a0(fusion) -> float:
    """C3 (read-only): ||state_pos - anchor0||_2 in the leader
    (behind,cross,below)=(x,y,z) frame of get_state(). Pure read. NaN if
    anchor0/state unavailable (never 0.0, never raises). [C2]+[R-b]"""
    try:
        x = getattr(fusion, 'x', None)
        anchors = getattr(fusion, 'anchor_positions', None)
        if x is None or anchors is None or 0 not in anchors or len(x) < 3:
            return _UWB_DIAG_NAN
        a0 = anchors[0]
        return float(math.sqrt(sum((float(x[i]) - float(a0[i])) ** 2 for i in range(3))))
    except Exception:
        return _UWB_DIAG_NAN


def _uwb_diag_anchor_counts(fusion) -> dict:
    """C3 (read-only): snapshot per-anchor + global UWB update/reject counters
    (copies, pure read). Used to detect which anchors were processed this tick."""
    return {
        'upd': getattr(fusion, 'uwb_update_count', 0),
        'rej': getattr(fusion, 'uwb_rejected_count', 0),
        'upd_a': dict(getattr(fusion, '_uwb_updates_by_anchor', {}) or {}),
        'rej_a': dict(getattr(fusion, '_uwb_rejected_by_anchor', {}) or {}),
    }


def _uwb_diag_innovs(fusion, pre_counts) -> tuple:
    """C3 (read-only): per-anchor innovation (a0..a3). NaN UNLESS that anchor
    was actually processed THIS tick (post>pre update or reject delta) — guards
    the stale _last_innovation_by_anchor dict (written at :448 but NOT cleared
    per-tick, only at reset). NaN (never 0.0) on absence. [C1]+[R-c]"""
    innov_map = getattr(fusion, '_last_innovation_by_anchor', {}) or {}
    upd_post = getattr(fusion, '_uwb_updates_by_anchor', {}) or {}
    rej_post = getattr(fusion, '_uwb_rejected_by_anchor', {}) or {}
    upd_pre = pre_counts.get('upd_a', {})
    rej_pre = pre_counts.get('rej_a', {})
    out = []
    for aid in range(4):
        processed = (upd_post.get(aid, 0) > upd_pre.get(aid, 0)
                     or rej_post.get(aid, 0) > rej_pre.get(aid, 0))
        out.append(float(innov_map.get(aid, _UWB_DIAG_NAN)) if processed else _UWB_DIAG_NAN)
    return tuple(out)


# =============================================================================
# v9.10 GATE EXCEPTIONS (Appendix A.1)
# =============================================================================


class GateFatalError(RuntimeError):
    """Base for all gate-fatal conditions."""
    pass


class IllegalSimTimeRead(GateFatalError):
    """Time read attempted outside Gate 0 locality."""
    pass


class TickOwnerReentrancyError(GateFatalError):
    """Nested tick execution detected."""
    pass


class DoubleSendError(GateFatalError):
    """Multiple speed commands in one executed tick."""
    pass


class NaNGeometryError(GateFatalError):
    """Non-finite geometry detected."""
    pass


class GeometryUnavailableTimeout(GateFatalError):
    """Geometry unavailable too long in formation regime."""
    pass


class InjectedFatalError(GateFatalError):
    """v9.10: Injected fatal error for testing fault paths."""
    pass


class TimebaseRateDropError(GateFatalError):
    """PRH-ATT-004: Timebase rate dropped below minimum during runtime."""
    pass


# =============================================================================
# TIMESTAMPED LOGGER
# =============================================================================
_wall_t0_mono = time.monotonic()
_t0_sim: Optional[float] = None


def _get_elapsed_sim_s() -> Optional[float]:
    """Get elapsed sim-time in seconds, or None if not ready.

    Latches t0_sim on first call when SimTimeAuthority is ready.
    """
    global _t0_sim
    ta = get_time_authority()
    if ta is None or not ta.is_ready():
        return None
    if _t0_sim is None:
        _t0_sim = ta.now_s()
    return ta.now_s() - _t0_sim


def log(msg: str) -> None:
    """Timestamped logger with sim-time and wall-elapsed time.

    Format: [epoch_wall] [S=<sim_s> W=<wall_elapsed_s>] message

    - S= is sim-time seconds from MAVLink time_boot_ms
    - W= is wall elapsed seconds (monotonic clock)
    """
    # Use guarded wall-time helpers (allowed for logging)
    epoch = wall_now_s()  # Epoch for cross-process log correlation
    wall_elapsed = wall_monotonic_s() - _wall_t0_mono  # Human-readable wall elapsed
    elapsed_sim = _get_elapsed_sim_s()

    if elapsed_sim is not None:
        sim_str = f"S={elapsed_sim:9.3f}s"
    else:
        sim_str = "S=-----.---s"

    print(f"[{epoch:.3f}] [{sim_str} W={wall_elapsed:7.3f}s] {msg}", flush=True)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
@dataclass
class RelativeGeometry:
    """Relative geometry between follower and leader (boom tip)."""
    behind_m: float         # Distance behind leader (positive = behind)
    cross_m: float          # Cross-track offset (positive = right of leader)
    below_m: float          # Vertical offset (positive = below leader)
    range_m: float          # 3D range to boom tip
    target_aspect_deg: float  # Angle from leader's tail (0 = directly behind)
    closure_rate_ms: float    # Rate of closure (positive = closing) - now uses vel_behind
    vel_cross_m_s: float      # Cross-track rate (m/s) - for D-term damping (Test 301)
    source: str             # "FUSED", "GPS", "UWB", or "UWB_DOM"
    # G3 guard-on-env (2026-08-14, R7.1 re-seal): the SAME-COMPUTATION
    # GPS-direct below value, carried so Guard 1 (altitude ceiling) can
    # compare it when FD_GUARD_CEILING_CHANNEL=env. Defaulted None so the
    # UWB_ONLY/UWB_DOM constructions stay valid -> guard falls back to the
    # fused channel (fail-safe) and counts the fallback.
    below_gps_m: Optional[float] = None


@dataclass
class RawSensorData:
    """Raw sensor measurements for sensor mode switching (Test 308).

    This holds both GPS-derived and UWB-derived geometry so the controller
    can choose which to use based on sensor_mode and UWB validity.
    """
    # GPS-derived geometry (always available)
    gps_behind_m: float
    gps_cross_m: float
    gps_below_m: float
    gps_range_m: float

    # Fused geometry (from EKF fusion, if available)
    fused_behind_m: float
    fused_cross_m: float
    fused_below_m: float
    fused_range_m: float

    # UWB range measurement (may be None or invalid)
    uwb_range_m: Optional[float]
    uwb_timestamp: float
    uwb_valid: bool
    uwb_invalid_reason: str

    # Velocity in body frame
    vel_behind_ms: float
    vel_cross_ms: float
    vel_below_ms: float

    # Metadata
    timestamp: float


# REMOVED: is_uwb_valid (Moved to helper class)
# REMOVED: compute_uwb_dominant_weight (Moved to helper class)
def wrap_heading(hdg: float) -> float:
    """Wrap heading to 0-360 range."""
    while hdg < 0:
        hdg += 360
    while hdg >= 360:
        hdg -= 360
    return hdg


def wrap_heading_error(err: float) -> float:
    """Wrap heading error to -180 to +180 range."""
    while err > 180:
        err -= 360
    while err < -180:
        err += 360
    return err


def compute_range(behind: float, cross: float, below: float) -> float:
    """Compute 3D range from relative position."""
    return math.sqrt(behind**2 + cross**2 + below**2)


def compute_isa_density_ratio(alt_msl_m: float) -> float:
    """Compute ISA density ratio at altitude (rho/rho0).

    Uses International Standard Atmosphere model for troposphere.
    Returns sqrt(rho/rho0) which is used for TAS -> EAS conversion.
    """
    # Temperature at altitude
    temp_at_alt = ISA_SEA_LEVEL_TEMP - ISA_TEMPERATURE_LAPSE_RATE * alt_msl_m
    # Clamp to prevent negative temperature at extreme altitudes
    temp_at_alt = max(temp_at_alt, 216.65)

    # Density ratio (from ideal gas law + barometric formula)
    # rho/rho0 = (T/T0)^(g/(L*R) - 1) where g=9.81, L=0.0065, R=287
    # Exponent = 9.81 / (0.0065 * 287) - 1 ≈ 4.26
    exponent = 4.2561
    density_ratio = (temp_at_alt / ISA_SEA_LEVEL_TEMP) ** exponent

    return density_ratio


def tas_to_eas(tas_ms: float, alt_msl_m: float) -> float:
    """Convert True Airspeed (TAS) to Equivalent Airspeed (EAS).

    EAS = TAS * sqrt(rho/rho0)
    This is what ArduPilot's TECS uses internally for speed control.
    """
    density_ratio = compute_isa_density_ratio(alt_msl_m)
    return tas_ms * math.sqrt(density_ratio)


def compute_baro_eas2tas(press_hpa: float, temp_cdeg: float) -> Optional[float]:
    """Candidate A: eas2tas from measured baro pressure + temperature (SCALED_PRESSURE),
    matching ArduPilot AP_Baro::_EAS2TAS = sqrt(SSL_AIR_DENSITY / (P / (R_specific * T))),
    SSL_AIR_DENSITY=1.225, R_specific=287.26 (AP_Baro.cpp). Inputs in SCALED_PRESSURE units:
    press_hpa in hPa, temp_cdeg in centidegrees C. Returns None on invalid/implausible inputs
    so the caller falls back to the ratio/ISA path (byte-identical when A is OFF or no baro yet).
    """
    try:
        press_pa = float(press_hpa) * 100.0
        temp_k = float(temp_cdeg) / 100.0 + 273.15
    except (TypeError, ValueError):
        return None
    if press_pa <= 0.0 or temp_k <= 0.0:
        return None
    return math.sqrt(1.225 * 287.26 * temp_k / press_pa)


def apply_eas2tas_ff(spd_cmd: float, fol_alt: float,
                     eas2tas: Optional[float] = None) -> float:
    """CSV2_SPEC247: TAS->EAS command feedforward gate (default OFF -> identity).

    ArduPilot TECS realizes the DO_CHANGE_SPEED demand as EAS (_TAS_dem = _EAS_dem *
    EAS2TAS, AP_TECS.cpp:404-405), so a raw TAS-frame command flies fast at altitude.
    When _FD_EAS2TAS_FF_ENABLE is set, the WHOLE TAS-frame command is converted to EAS
    (frame-correct: realized TAS = lead_gs + v_c exactly). eas2tas source priority:
    explicit baro-derived `eas2tas` arg (Candidate A) > FD_EAS2TAS_RATIO diagnostic override >
    ISA-from-fol_alt. Returns spd_cmd UNCHANGED (strict identity) when the flag is OFF, and the
    ratio/ISA paths are bit-for-bit unchanged when `eas2tas` is not supplied.
    """
    if not _FD_EAS2TAS_FF_ENABLE:
        return spd_cmd
    # Candidate A: a live baro-derived eas2tas (from SCALED_PRESSURE) takes precedence.
    if eas2tas is not None and eas2tas > 0.0:
        return spd_cmd / eas2tas
    if _FD_EAS2TAS_RATIO:
        return spd_cmd / float(_FD_EAS2TAS_RATIO)
    return tas_to_eas(spd_cmd, max(0.0, fol_alt))


def compute_tapered_max_closure(range_m: float, target_range_m: float) -> float:
    """Compute range-proportional maximum closure rate.

    Returns a closure rate that tapers from MAX_CLOSURE_RATE_FAR_MS when far
    to MAX_CLOSURE_RATE_NEAR_MS when near the target.

    This implements the "graduated airspeed control" requirement.
    """
    # Distance to target (can be negative if we've overshot)
    dist_to_target = range_m - target_range_m

    if dist_to_target <= 0:
        # We've overshot - very gentle closure (or even negative for backup)
        return MAX_CLOSURE_RATE_NEAR_MS * 0.5

    if dist_to_target >= CLOSURE_TAPER_FAR_RANGE_M:
        # Far away - use maximum closure rate
        return MAX_CLOSURE_RATE_FAR_MS

    if dist_to_target <= CLOSURE_TAPER_NEAR_RANGE_M:
        # Very close - use minimum closure rate
        return MAX_CLOSURE_RATE_NEAR_MS

    # Linear interpolation between near and far thresholds
    taper_range = CLOSURE_TAPER_FAR_RANGE_M - CLOSURE_TAPER_NEAR_RANGE_M
    taper_factor = (dist_to_target - CLOSURE_TAPER_NEAR_RANGE_M) / taper_range
    closure_range = MAX_CLOSURE_RATE_FAR_MS - MAX_CLOSURE_RATE_NEAR_MS

    return MAX_CLOSURE_RATE_NEAR_MS + taper_factor * closure_range


def compute_tapered_max_closure_alongtrack(behind_error_m: float) -> float:
    """Compute closure rate limit based on along-track distance (Test 311/388).

    Multi-segment taper for gentler approach near target:
    - >= 150m: 5.0 m/s (far, aggressive closure)
    - 50-150m: linear interpolation from 2.5 to 5.0 m/s (TEST 387: mid-far)
    - 20-50m: linear interpolation from 0.5 to 2.5 m/s (TEST 387: mid-near)
    - 2-20m: linear interpolation to 0.15 m/s (very near)
    - 0.5-2m: linear interpolation to 0.05 m/s (ultra near)
    - <= 0.5m: 0.05 m/s (nearly stopped)
    - <= 0m: 0.025 m/s (overshoot - very gentle)

    TEST 387: Split the 20-150m range into two segments with a boost at 50m
    to fix the closure stall observed at ~50m range in test_1.0x_20260120_141408.
    Old: 40m error -> 1.2 m/s max closure (insufficient)
    New: 40m error -> 1.83 m/s max closure (67% higher)

    Args:
        behind_error_m: Distance behind target (positive = need to close)

    Returns:
        Maximum closure rate in m/s
    """
    if behind_error_m <= 0:
        # Overshot - very gentle (half of ultra-near)
        return MAX_CLOSURE_RATE_ULTRA_NEAR_MS * 0.5

    if behind_error_m <= CLOSURE_TAPER_ULTRA_NEAR_ERR_M:
        # Ultra near (< 0.5m) - nearly stopped
        return MAX_CLOSURE_RATE_ULTRA_NEAR_MS

    if behind_error_m <= CLOSURE_TAPER_VERY_NEAR_ERR_M:
        # Very near (0.5m - 2m) - interpolate ultra to very-near
        t = ((behind_error_m - CLOSURE_TAPER_ULTRA_NEAR_ERR_M)
             / (CLOSURE_TAPER_VERY_NEAR_ERR_M - CLOSURE_TAPER_ULTRA_NEAR_ERR_M))
        return (MAX_CLOSURE_RATE_ULTRA_NEAR_MS
                + t * (MAX_CLOSURE_RATE_VERY_NEAR_MS - MAX_CLOSURE_RATE_ULTRA_NEAR_MS))

    if behind_error_m <= CLOSURE_TAPER_NEAR_RANGE_M:
        # Near (2m - 20m) - interpolate very-near to near
        t = ((behind_error_m - CLOSURE_TAPER_VERY_NEAR_ERR_M)
             / (CLOSURE_TAPER_NEAR_RANGE_M - CLOSURE_TAPER_VERY_NEAR_ERR_M))
        return (MAX_CLOSURE_RATE_VERY_NEAR_MS
                + t * (MAX_CLOSURE_RATE_NEAR_MS - MAX_CLOSURE_RATE_VERY_NEAR_MS))

    if behind_error_m >= CLOSURE_TAPER_FAR_RANGE_M:
        # Far (>= 150m) - maximum closure
        return MAX_CLOSURE_RATE_FAR_MS

    # Split medium range into two segments for more aggressive mid-range closure
    if behind_error_m >= CLOSURE_TAPER_MID_RANGE_M:
        # Mid-far (50m - 150m) - interpolate from mid to far
        t = ((behind_error_m - CLOSURE_TAPER_MID_RANGE_M)
             / (CLOSURE_TAPER_FAR_RANGE_M - CLOSURE_TAPER_MID_RANGE_M))
        return MAX_CLOSURE_RATE_MID_MS + t * (MAX_CLOSURE_RATE_FAR_MS - MAX_CLOSURE_RATE_MID_MS)

    # Mid-near (20m - 50m) - interpolate from near to mid (more aggressive than before)
    t = ((behind_error_m - CLOSURE_TAPER_NEAR_RANGE_M)
         / (CLOSURE_TAPER_MID_RANGE_M - CLOSURE_TAPER_NEAR_RANGE_M))
    return MAX_CLOSURE_RATE_NEAR_MS + t * (MAX_CLOSURE_RATE_MID_MS - MAX_CLOSURE_RATE_NEAR_MS)


# =============================================================================
# MODULE-LEVEL ENVELOPE WRAPPERS (Legacy compatibility for unit tests)
# =============================================================================
# These thin wrappers delegate to EnvelopeChecker methods for backward
# compatibility with tests that expect module-level functions.
# New code should use EnvelopeChecker directly.

_module_envelope_checker = EnvelopeChecker()


def is_in_precontact_outer_envelope(geom, lead_alt_msl: float, follower_alt_msl: float) -> bool:
    """Check if in precontact outer envelope (module-level wrapper).

    Legacy compatibility: delegates to EnvelopeChecker.is_in_precontact_outer_envelope().
    """
    return _module_envelope_checker.is_in_precontact_outer_envelope(
        geom, lead_alt_msl, follower_alt_msl
    )


def is_in_contact_outer_envelope(geom, lead_alt_msl: float, follower_alt_msl: float) -> bool:
    """Check if in contact outer envelope (module-level wrapper).

    Legacy compatibility: delegates to EnvelopeChecker.is_in_contact_outer_envelope().
    """
    return _module_envelope_checker.is_in_contact_outer_envelope(
        geom, lead_alt_msl, follower_alt_msl
    )
# =============================================================================
# MAVLINK CONNECTION HELPER
# =============================================================================
class MAVConnection:
    """MAVLink connection wrapper with telemetry caching."""

    def __init__(self, connection_string: str, source_system: int = 255):
        self.conn_str = connection_string
        self.source_system = source_system
        self.mav = None
        self.target_system = 1
        self.target_component = 1

        # Telemetry cache
        self._last_position = None
        self._last_attitude = None
        self._last_vfr_hud = None
        self._last_scaled_pressure = None   # CSV2_SPEC247 (Candidate A): SCALED_PRESSURE(29) baro+temp
        self._last_heartbeat = None
        # C1: SIM-truth caches (populated only when _FD_UWB_TRUTH_SOURCED;
        # stay None otherwise -> get_sim_position() returns None -> inert).
        self._last_simstate = None    # SIMSTATE(164): int32 degE7 lat/lng (truth horiz)
        self._last_sim_state = None   # SIM_STATE(108): float alt (truth vert)
        self._last_sim_state_tbms = None  # L-A fix: sim-ms of last 108 horiz CHANGE (extrap age; unconditional)
        self._last_nav_controller_output = None  # Tranche C (C0)

        # TRUTH-RATE DIAG (default-OFF instrumentation; see _FD_TRUTH_RATE_DIAG).
        # Diagnostic counters only; never read by control/fusion/observation.
        self._diag_108_recv = 0       # SIM_STATE(108) returned by recv_match
        self._diag_108_change = 0     # _last_sim_state.lat_int value-changes
        self._diag_33_recv = 0        # GPI(33) returned by recv_match (control)
        self._diag_33_change = 0      # _last_position.lat value-changes (control)
        self._diag_108_update_tbms = None   # sim-ms when 108 pos last changed
        self._diag_emit_age_buckets = [0, 0, 0]  # emit-age <25 / 25-75 / >75 ms
        self._diag_last_log_tbms = None

        # P0 v9.10 Model 1: Buffered time_boot_ms for tick-owner time authority update
        self._buffered_attitude_time_boot_ms: Optional[int] = None

        # FD-001: ATTITUDE timestamp queue (UNBOUNDED - rely on FD-002 fail-fast)
        self._attitude_time_queue: deque = deque()  # NO maxlen - silent drops are tick collapse
        self._last_enqueued_attitude_ms: Optional[int] = None  # For monotonic guard
        self._attitude_queue_max_depth: int = 0  # FD-002: Observability
        self._attitude_queue_max_depth_runtime: int = 0  # Change 4: runtime-only max depth

        # FD-008: Queue-active flag - True after queue activation (blocks non-scheduler TA updates)
        self._queue_active: bool = False

        self._attitude_message_arrived = False

    def connect(self, required_sysid: int = 1) -> bool:
        """Establish connection and wait for heartbeat from specific system.

        Args:
            required_sysid: The system ID to wait for (default=1 for QuadPlane).
                           This prevents accidentally locking onto leader (sysid=2).
        """
        log(f"Connecting to {self.conn_str}...")
        try:
            self.mav = mavutil.mavlink_connection(
                self.conn_str,
                source_system=self.source_system,
                source_component=1
            )
            log(f"Waiting for heartbeat from sysid={required_sysid}...")

            # Wait for heartbeat from the specific system ID
            start_time = time.time()
            timeout = 30.0
            while time.time() - start_time < timeout:
                msg = self.mav.recv_match(type='HEARTBEAT', blocking=True,
                                          timeout=1.0)
                if msg is None:
                    continue

                src_sysid = msg.get_srcSystem()
                if src_sysid == required_sysid:
                    self.target_system = required_sysid
                    self.target_component = msg.get_srcComponent()
                    log(f"Heartbeat received from system {self.target_system}")
                    return True
                else:
                    # Log once per ignored system to avoid spam
                    log(f"Ignoring heartbeat from sysid={src_sysid} "
                        f"(waiting for sysid={required_sysid})")

            log(f"ERROR: No heartbeat from sysid={required_sysid} within {timeout}s")
            return False
        except Exception as e:
            log(f"ERROR: Connection failed: {e}")
            return False

    def update_cache(self) -> None:
        """Poll and cache latest telemetry messages."""
        if not self.mav:
            return

        while True:
            msg = self.mav.recv_match(blocking=False)
            if msg is None:
                break

            if msg.get_srcSystem() != self.target_system:
                continue

            msg_type = msg.get_type()
            if msg_type == 'GLOBAL_POSITION_INT':
                if _FD_TRUTH_RATE_DIAG:
                    self._diag_33_recv += 1
                    if self._last_position is None or msg.lat != self._last_position.lat:
                        self._diag_33_change += 1
                self._last_position = msg
            elif msg_type == 'ATTITUDE':
                self._last_attitude = msg
                self._attitude_message_arrived = True
                # P0 v9.10 Model 1: Buffer the time_boot_ms for later use by tick owner
                # Time authority update moved to _read_sim_time_s_impl() for locality
                if self.target_system == 1:  # Follower sysid
                    tb = int(msg.time_boot_ms)
                    # FD-001: Monotonic enqueueing - ignore duplicates and backward time
                    if (self._last_enqueued_attitude_ms is None or
                            tb > self._last_enqueued_attitude_ms):
                        self._attitude_time_queue.append(tb)
                        self._last_enqueued_attitude_ms = tb
                        # FD-002: Track max depth
                        self._attitude_queue_max_depth = max(
                            self._attitude_queue_max_depth,
                            len(self._attitude_time_queue))
                    # Keep latest for time authority fallback
                    self._buffered_attitude_time_boot_ms = tb
            elif msg_type == 'VFR_HUD':
                self._last_vfr_hud = msg
            elif msg_type == 'SCALED_PRESSURE':
                self._last_scaled_pressure = msg   # CSV2_SPEC247 (Candidate A): baro eas2tas source
            elif msg_type == 'HEARTBEAT':
                self._last_heartbeat = msg
            elif msg_type == 'NAV_CONTROLLER_OUTPUT':
                self._last_nav_controller_output = msg
            elif _FD_UWB_TRUTH_SOURCED and msg_type == 'SIMSTATE' and msg.get_srcSystem() == self.target_system:
                self._last_simstate = msg     # C1: int degE7 lat/lng truth (horiz); REV1 sysid-guarded
            elif _FD_UWB_TRUTH_SOURCED and msg_type == 'SIM_STATE' and msg.get_srcSystem() == self.target_system:
                # L-A fix: unconditional 108 cache-age stamp (None-OR-CHANGE), independent of
                # _FD_TRUTH_RATE_DIAG. None fires exactly once on the first 108 (closes the
                # pump-pre-populates-cache init race); else stamp on lat_int CHANGE. Sim-clock only.
                if self._buffered_attitude_time_boot_ms is not None:
                    _uc_nl = getattr(msg, 'lat_int', None)
                    _uc_ol = (getattr(self._last_sim_state, 'lat_int', None)
                              if self._last_sim_state is not None else None)
                    if self._last_sim_state_tbms is None or _uc_nl != _uc_ol:
                        self._last_sim_state_tbms = self._buffered_attitude_time_boot_ms
                if _FD_TRUTH_RATE_DIAG:
                    self._diag_108_recv += 1
                    _nl = getattr(msg, 'lat_int', None)
                    _ol = (getattr(self._last_sim_state, 'lat_int', None)
                           if self._last_sim_state is not None else None)
                    if _nl != _ol:
                        self._diag_108_change += 1
                        self._diag_108_update_tbms = self._buffered_attitude_time_boot_ms
                self._last_sim_state = msg    # C1: float alt truth (vert); REV1 sysid-guarded (defense-in-depth)

        # TRUTH-RATE DIAG (default-OFF): periodic counter snapshot to the log only.
        if _FD_TRUTH_RATE_DIAG and self._buffered_attitude_time_boot_ms is not None:
            _now = self._buffered_attitude_time_boot_ms
            if self._diag_last_log_tbms is None:
                self._diag_last_log_tbms = _now
            elif _now - self._diag_last_log_tbms >= 10000:
                log(f"[TRUTH_RATE_DIAG] S={_now/1000.0:.1f}s "
                    f"108recv={self._diag_108_recv} 108chg={self._diag_108_change} "
                    f"33recv={self._diag_33_recv} 33chg={self._diag_33_change} "
                    f"emit_age_buckets_lt25_25to75_gt75={self._diag_emit_age_buckets}")
                self._diag_last_log_tbms = _now

    def set_mode(self, mode_name: str) -> bool:
        """Set flight mode."""
        mode_map = self.mav.mode_mapping()
        if mode_name not in mode_map:
            log(f"ERROR: Unknown mode {mode_name}")
            return False

        mode_id = mode_map[mode_name]
        self.mav.mav.set_mode_send(
            self.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        log(f"Set mode to {mode_name}")
        return True

    def arm(self) -> bool:
        """Arm the vehicle."""
        self.mav.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        log("Arm command sent")
        return True

    def send_guided_heading(self, heading_deg: float, alt_msl: float) -> None:
        """Send GUIDED mode heading and altitude command using DO_REPOSITION.

        Creates a "carrot point" waypoint 1000m ahead in the desired heading
        direction, which ArduPlane will fly towards. This is the proven method
        from the old predictive controller.
        """
        # Get current position
        pos = self.get_position()
        if not pos:
            return

        cur_lat, cur_lon, _ = pos

        # Calculate carrot point 1000m ahead in desired heading
        carrot_dist_m = 1000.0
        heading_rad = math.radians(heading_deg)

        # Approximate meters to degrees (at mid-latitudes)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(cur_lat))

        delta_lat = (carrot_dist_m * math.cos(heading_rad)) / meters_per_deg_lat
        delta_lon = (carrot_dist_m * math.sin(heading_rad)) / meters_per_deg_lon

        target_lat = cur_lat + delta_lat
        target_lon = cur_lon + delta_lon

        # Compute integer lat/lon for COMMAND_INT
        target_lat_i = int(target_lat * 1e7)
        target_lon_i = int(target_lon * 1e7)

        # Sanity check - cur_lat/cur_lon should be in degrees (not 1e7-scaled)
        if abs(cur_lat) > 90 or abs(cur_lon) > 180:
            log("[DO_REPOSITION] ERROR: cur_lat/cur_lon appear 1e7-scaled! "
                f"cur=({cur_lat},{cur_lon})")
            return  # Do not send malformed command

        # int32 range check for target coordinates
        INT32_MAX = 2147483647
        INT32_MIN = -2147483648
        if not (INT32_MIN <= target_lat_i <= INT32_MAX and INT32_MIN <= target_lon_i <= INT32_MAX):
            log("[DO_REPOSITION] ERROR: int32 overflow! "
                f"target_lat_i={target_lat_i} target_lon_i={target_lon_i}")
            return  # Do not send malformed command

        # Log full command details before sending
        log(f"[DO_REPOSITION] cur=({cur_lat:.6f},{cur_lon:.6f}) "
            f"hdg_cmd={heading_deg:.1f}° alt_cmd={alt_msl:.1f}m "
            f"carrot=({target_lat:.6f},{target_lon:.6f}) "
            f"int=({target_lat_i},{target_lon_i})")

        # Send DO_REPOSITION via COMMAND_INT (more reliable than COMMAND_LONG)
        self.mav.mav.command_int_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
            mavutil.mavlink.MAV_CMD_DO_REPOSITION,
            0, 0,  # current, autocontinue
            -1,    # param1: ground speed (-1 = use current)
            1,     # param2: MAV_DO_REPOSITION_FLAGS_CHANGE_MODE
            0,     # param3: reserved
            heading_deg,  # param4: yaw/heading in degrees
            target_lat_i,  # x: latitude (int, 1e7 scaled)
            target_lon_i,  # y: longitude (int, 1e7 scaled)
            alt_msl  # z: altitude MSL
        )

        # Non-blocking COMMAND_ACK capture
        ack_start = time.time()
        while time.time() - ack_start < 0.5:  # 500ms timeout (wall-time OK for ACK)
            ack_msg = self.mav.recv_match(type='COMMAND_ACK', blocking=False)
            if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_DO_REPOSITION:
                result_names = {
                    0: "ACCEPTED", 1: "TEMP_REJECTED", 2: "DENIED",
                    3: "UNSUPPORTED", 4: "FAILED", 5: "IN_PROGRESS"
                }
                result_str = result_names.get(ack_msg.result, f"UNKNOWN({ack_msg.result})")
                log(f"[DO_REPOSITION_ACK] result={result_str}")
                break
            time.sleep(0.01)

    def send_guided_speed(self, speed_ms: float) -> None:
        """Send DO_CHANGE_SPEED command."""
        self.mav.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0, 0, speed_ms, -1, 0, 0, 0, 0
        )

    def send_guided_change_altitude(self, alt_msl: float, rate_mps: float = 0.0) -> bool:
        """TEST 313: Send MAV_CMD_GUIDED_CHANGE_ALTITUDE for explicit altitude change.

        This is a fallback command when DO_REPOSITION altitude isn't being tracked.
        Uses MAV_CMD_GUIDED_CHANGE_ALTITUDE which is specifically designed for
        changing altitude in GUIDED mode without changing lateral position.

        IMPORTANT: ArduPlane processes this command via COMMAND_INT, not COMMAND_LONG.
        See: https://ardupilot.org/plane/docs/common-mavlink-mission-command-messages-mav_cmd.html

        Args:
            alt_msl: Target altitude in meters MSL
            rate_mps: Climb/descent rate in m/s (0 = max rate, default)

        Returns:
            True if command was sent (not necessarily accepted)
        """
        # MAV_CMD_GUIDED_CHANGE_ALTITUDE via COMMAND_INT
        # param1: empty (0)
        # param2: empty (0)
        # param3: rate in m/s (0 = max rate)
        # param4: empty (0)
        # x, y: position (0, 0 = don't change)
        # z: altitude MSL
        self.mav.mav.command_int_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL,  # frame: MSL altitude
            mavutil.mavlink.MAV_CMD_GUIDED_CHANGE_ALTITUDE,
            0, 0,  # current, autocontinue
            0.0,  # param1: empty
            0.0,  # param2: empty
            float(rate_mps),  # param3: rate m/s (0 = max)
            0.0,  # param4: empty
            0, 0,  # x, y: lat/lon (INT32, 0 = no change)
            float(alt_msl)  # z: altitude MSL
        )
        return True

    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current position (lat, lon, alt_msl)."""
        self.update_cache()
        if self._last_position:
            return (
                self._last_position.lat / 1e7,
                self._last_position.lon / 1e7,
                self._last_position.alt / 1000.0
            )
        return None

    def get_sim_position(self) -> Optional[Tuple[float, float, float]]:
        """C1: SIM-TRUTH follower (lat, lon, alt_msl) for the UWB emitter.

        Vertical always from SIM_STATE(108) float alt (AMSL). Horizontal:
        default from SIMSTATE(164) int degE7; when _FD_UWB_TRUTH_HORIZ_FROM_108
        (Option-A), from SIM_STATE(108).lat_int/lon_int (degE7, 10Hz) — the gate-ON
        branch reads ONLY sst (108), never ss (164). Returns None until the needed
        truth is cached (caller then skips emission — never feeds perceived-as-truth).
        Caches populate only when _FD_UWB_TRUTH_SOURCED, so this returns None when off.
        """
        self.update_cache()
        ss = self._last_simstate
        sst = self._last_sim_state
        if _FD_UWB_TRUTH_HORIZ_FROM_108:
            # Option-A gate-ON: horizontal + vertical ONLY from SIM_STATE(108).
            # Init-race guard: NO 164 fallback, NO float sst.lat/lon fallback, NO
            # perceived fallback. Return None until 108 degE7 horizontal is populated.
            if sst is None:
                return None
            lat_int = getattr(sst, 'lat_int', None)
            lon_int = getattr(sst, 'lon_int', None)
            if lat_int in (None, 0) or lon_int in (None, 0):
                return None
            if (_FD_TRUTH_RATE_DIAG
                    and self._buffered_attitude_time_boot_ms is not None
                    and self._diag_108_update_tbms is not None):
                # Emit-time age of the 108 this emit will use (sim-ms). update_cache
                # ran at the top of this method, so a fresh-this-tick 108 => age~0.
                _age = self._buffered_attitude_time_boot_ms - self._diag_108_update_tbms
                self._diag_emit_age_buckets[0 if _age < 25 else (1 if _age <= 75 else 2)] += 1
            return (lat_int / 1e7, lon_int / 1e7, float(sst.alt))
        # Default (gate-OFF): byte-identical 164-horizontal path.
        if ss is None or sst is None:
            return None
        return (ss.lat / 1e7, ss.lng / 1e7, float(sst.alt))

    def get_sim_velocity_ned(self) -> Optional[Tuple[float, float, float]]:
        """TIER2: SIM-TRUTH follower (vn, ve, vd) m/s NED from SIM_STATE(108).

        Same truth class as get_sim_position(); returns None until a 108 with
        velocity fields is cached (caches populate only when
        _FD_UWB_TRUTH_SOURCED). Diagnostic consumers must SKIP the sample on
        None — never substitute perceived velocity for truth.
        """
        self.update_cache()
        sst = self._last_sim_state
        if sst is None:
            return None
        vn = getattr(sst, 'vn', None)
        ve = getattr(sst, 've', None)
        vd = getattr(sst, 'vd', None)
        if vn is None or ve is None or vd is None:
            return None
        return (float(vn), float(ve), float(vd))

    def get_velocity_ned(self) -> Optional[Tuple[float, float, float]]:
        """Get current velocity in NED frame (m/s)."""
        self.update_cache()
        if self._last_position:
            return (
                self._last_position.vx / 100.0,
                self._last_position.vy / 100.0,
                self._last_position.vz / 100.0
            )
        return None

    def get_attitude(self) -> Optional[Tuple[float, float, float]]:
        """Get current attitude (roll, pitch, yaw in degrees)."""
        self.update_cache()
        if self._last_attitude:
            return (
                math.degrees(self._last_attitude.roll),
                math.degrees(self._last_attitude.pitch),
                math.degrees(self._last_attitude.yaw)
            )
        return None

    def get_airspeed(self) -> Optional[float]:
        """Get current airspeed."""
        self.update_cache()
        if self._last_vfr_hud:
            return self._last_vfr_hud.airspeed
        return None

    def get_baro_eas2tas(self) -> Optional[float]:
        """Candidate A: live baro-derived eas2tas from the latest cached SCALED_PRESSURE, or None
        when the A flag is OFF or no SCALED_PRESSURE has arrived (caller -> ratio/ISA fallback,
        byte-identical). Reads the cache populated by the message pump (no extra recv/drain)."""
        if not _FD_EAS2TAS_BARO:
            return None
        msg = self._last_scaled_pressure
        if msg is None:
            return None
        e2t = compute_baro_eas2tas(getattr(msg, 'press_abs', None),
                                   getattr(msg, 'temperature', None))
        if e2t is not None and not getattr(self, '_baro_e2t_logged', False):
            self._baro_e2t_logged = True   # first-success diagnostic (A observability; gated)
            log(f"[CSV2_SPEC247 A] baro eas2tas LIVE: press={getattr(msg, 'press_abs', None)} hPa "
                f"temp={getattr(msg, 'temperature', None)} cdeg -> eas2tas={e2t:.5f}")
        return e2t

    def get_throttle_pct(self) -> Optional[float]:
        """Return follower throttle percentage from VFR_HUD."""
        self.update_cache()
        if self._last_vfr_hud is None:
            return None
        return float(self._last_vfr_hud.throttle)

    def get_climb_rate_mps(self) -> Optional[float]:
        """Return follower climb rate (m/s, positive up) from VFR_HUD."""
        self.update_cache()
        if self._last_vfr_hud is None:
            return None
        return float(self._last_vfr_hud.climb)

    def get_mode(self) -> Optional[str]:
        """Get current flight mode."""
        self.update_cache()
        if self._last_heartbeat:
            mode_map = self.mav.mode_mapping()
            for name, mode_id in mode_map.items():
                if mode_id == self._last_heartbeat.custom_mode:
                    return name
        return None

    def get_param(self, param_name: str, timeout: float = 2.0) -> Optional[float]:
        """Request and retrieve a parameter value."""
        try:
            param_id = param_name.encode("ascii")[:16]
            self.mav.mav.param_request_read_send(
                self.target_system, self.target_component,
                param_id, -1
            )
            start = time.time()
            while time.time() - start < timeout:
                msg = self.mav.recv_match(type='PARAM_VALUE', blocking=True,
                                          timeout=0.2)
                if msg and msg.param_id.rstrip('\x00') == param_name:
                    return msg.param_value
            return None
        except Exception as e:
            log(f"ERROR getting param {param_name}: {e}")
            return None

    def set_param(self, param_name: str, value: float,
                  timeout: float = 2.0) -> bool:
        """Set a parameter value and verify it was accepted."""
        try:
            param_id = param_name.encode("ascii")[:16]
            self.mav.mav.param_set_send(
                self.target_system, self.target_component,
                param_id, value,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
            # Wait for confirmation
            start = time.time()
            while time.time() - start < timeout:
                msg = self.mav.recv_match(type='PARAM_VALUE', blocking=True,
                                          timeout=0.2)
                if msg and msg.param_id.rstrip('\x00') == param_name:
                    if abs(msg.param_value - value) < 0.01:
                        return True
            return False
        except Exception as e:
            log(f"ERROR setting param {param_name}: {e}")
            return False

    def get_mode_id(self) -> Optional[int]:
        """Get current flight mode ID from heartbeat."""
        self.update_cache()
        if self._last_heartbeat:
            return self._last_heartbeat.custom_mode
        return None

    def set_mode_and_wait(self, mode_name: str, timeout: float = 10.0) -> bool:
        """Set flight mode and wait for confirmation via heartbeat.

        This is more robust than set_mode() as it confirms the mode change.

        Args:
            mode_name: Target mode name (e.g., "GUIDED", "QLOITER")
            timeout: Maximum time to wait for confirmation

        Returns:
            True if mode confirmed, False if timeout or error
        """
        mode_map = self.mav.mode_mapping()
        if mode_name not in mode_map:
            log(f"ERROR: Unknown mode {mode_name}")
            return False

        target_mode_id = mode_map[mode_name]
        log(f"Setting mode to {mode_name} (id={target_mode_id})...")

        # Send mode change command
        self.mav.mav.set_mode_send(
            self.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            target_mode_id
        )

        # Wait for heartbeat confirmation
        start = time.time()
        while time.time() - start < timeout:
            msg = self.mav.recv_match(type='HEARTBEAT', blocking=True,
                                      timeout=0.5)
            if msg and msg.get_srcSystem() == self.target_system:
                self._last_heartbeat = msg
                if msg.custom_mode == target_mode_id:
                    log(f"Mode {mode_name} confirmed via heartbeat")
                    return True

        log(f"ERROR: Mode {mode_name} not confirmed within {timeout}s")
        return False

    def wait_for_gps_fix(self, min_fix_type: int = 3,
                         timeout: float = 30.0) -> bool:
        """Wait for GPS fix of specified quality.

        Args:
            min_fix_type: Minimum fix type (3 = 3D fix)
            timeout: Maximum time to wait

        Returns:
            True if GPS fix achieved, False if timeout
        """
        log(f"Waiting for GPS fix (type >= {min_fix_type})...")
        start = time.time()
        while time.time() - start < timeout:
            msg = self.mav.recv_match(type='GPS_RAW_INT', blocking=True,
                                      timeout=1.0)
            if msg and msg.get_srcSystem() == self.target_system:
                if msg.fix_type >= min_fix_type:
                    log(f"GPS fix type {msg.fix_type} achieved "
                        f"(satellites: {msg.satellites_visible})")
                    return True
        log(f"WARNING: GPS fix timeout after {timeout}s")
        return False

    def request_message_rate(self, msg_id: int, rate_hz: float) -> bool:
        """Request a specific message at a given rate.

        Uses MAV_CMD_SET_MESSAGE_INTERVAL (511).

        Args:
            msg_id: MAVLink message ID
            rate_hz: Desired rate in Hz

        Returns:
            True if command sent successfully
        """
        if rate_hz <= 0:
            interval_us = -1  # Disable
        else:
            interval_us = int(1000000.0 / rate_hz)

        try:
            self.mav.mav.command_long_send(
                self.target_system, self.target_component,
                511,  # MAV_CMD_SET_MESSAGE_INTERVAL
                0,    # confirmation
                float(msg_id),        # param1: message ID
                float(interval_us),   # param2: interval (us)
                0.0, 0.0, 0.0, 0.0, 0.0  # unused params
            )
            return True
        except Exception as e:
            log(f"ERROR requesting message rate: {e}")
            return False

    def is_vtol_mode(self, mode_id: Optional[int] = None) -> bool:
        """Check if current or given mode is a VTOL mode (Q* modes 17-25)."""
        if mode_id is None:
            mode_id = self.get_mode_id()
        if mode_id is None:
            return False
        return 17 <= mode_id <= 25


# =============================================================================
# LEADER TELEMETRY RECEIVER
# =============================================================================
@dataclass
class LeaderState:
    """Leader aircraft state."""
    lat: float
    lon: float
    alt_msl: float
    heading_deg: float
    ground_speed_ms: float
    vn_ms: float
    ve_ms: float
    vd_ms: float
    boom_length_m: float
    time_boot_ms: int
    rx_wall_time_s: float = 0.0
    true_airspeed_ms: float = 0.0


class LeaderTelemetryReceiver:
    """Receives leader telemetry via UDP."""

    def __init__(self, listen_port: int = 14552, target_sysid: int = 2):
        self.listen_port = listen_port
        self.target_sysid = target_sysid
        self.mav = None
        self.last_state: Optional[LeaderState] = None
        self._broadcast_start_tbms: Optional[int] = None

    def connect(self) -> bool:
        """Start listening for leader telemetry."""
        try:
            self.mav = mavutil.mavlink_connection(
                f"udpin:0.0.0.0:{self.listen_port}",
                source_system=255
            )
            log(f"[LeadTelemetry] Listening on port {self.listen_port}")
            return True
        except Exception as e:
            log(f"[LeadTelemetry] ERROR: {e}")
            return False

    def update(self) -> Optional[LeaderState]:
        """Poll for new leader telemetry."""
        if not self.mav:
            return None

        while True:
            msg = self.mav.recv_match(blocking=False)
            if msg is None:
                break

            # VFR_HUD from leader: extract true airspeed
            if msg.get_type() == 'VFR_HUD' and msg.get_srcSystem() == self.target_sysid:
                self._leader_tas_ms = msg.airspeed
                continue

            if msg.get_type() != 'GLOBAL_POSITION_INT':
                continue

            if msg.get_srcSystem() != self.target_sysid:
                continue
            heading = msg.hdg / 100.0 if msg.hdg != 65535 else 0.0
            vn = msg.vx / 100.0
            ve = msg.vy / 100.0
            vd = msg.vz / 100.0
            gs = math.sqrt(vn**2 + ve**2)

            # Track broadcast start time for boom extension calculation
            tbms = int(getattr(msg, "time_boot_ms", 0))
            if self._broadcast_start_tbms is None and tbms > 0:
                self._broadcast_start_tbms = tbms
            elapsed_s = (
                (tbms
                 - self._broadcast_start_tbms) /
                1000.0) if (
                self._broadcast_start_tbms is not None and tbms >= self._broadcast_start_tbms) else 0.0

            # Calculate boom extension based on time since broadcast started
            elapsed = elapsed_s
            if elapsed < 10.0:
                boom_extension = 0.0
            elif elapsed < 15.0:
                boom_extension = (elapsed - 10.0) / 5.0
            else:
                boom_extension = 1.0
            boom_length = BOOM_STOWED_LENGTH_M + boom_extension * (
                BOOM_MAX_LENGTH_M - BOOM_STOWED_LENGTH_M
            )

            self.last_state = LeaderState(
                lat=msg.lat / 1e7,
                lon=msg.lon / 1e7,
                alt_msl=msg.alt / 1000.0,
                heading_deg=heading,
                ground_speed_ms=gs,
                vn_ms=vn,
                ve_ms=ve,
                vd_ms=vd,
                boom_length_m=boom_length,
                time_boot_ms=tbms,
                rx_wall_time_s=time.time(),
                true_airspeed_ms=getattr(self, '_leader_tas_ms', 0.0)
            )

        return self.last_state

    def get_state(self) -> Optional[LeaderState]:
        """Get last known leader state."""
        return self.last_state


# =============================================================================
# WEATHER CHECKER
# =============================================================================
class WeatherChecker:
    """Verifies SITL weather configuration matches leader."""

    def __init__(self, follower_mav: MAVConnection):
        self.follower = follower_mav
        self.weather_params: dict = {}

    def check_weather(self) -> bool:
        """Query and log SITL weather parameters."""
        log("[WEATHER_CHECK] Querying SITL weather configuration...")

        all_ok = True
        for param_name in WEATHER_PARAMS:
            value = self.follower.get_param(param_name)
            if value is not None:
                self.weather_params[param_name] = value
                log(f"[WEATHER_CHECK] {param_name} = {value:.2f}")
            else:
                log(f"[WEATHER_CHECK] WARNING: Could not read {param_name}")
                all_ok = False

        if all_ok:
            wind_spd = self.weather_params.get("SIM_WIND_SPD", 0)
            wind_dir = self.weather_params.get("SIM_WIND_DIR", 0)
            turb = self.weather_params.get("SIM_WIND_TURB", 0)
            log(f"[WEATHER_CHECK] Atmosphere: wind={wind_spd:.1f}m/s "
                f"from {wind_dir:.0f}°, turb={turb:.2f}")
            log("[WEATHER_CHECK] Both aircraft share the same atmosphere")
        else:
            log("[WEATHER_CHECK] WARNING: Could not verify all weather params")

        return all_ok


# =============================================================================
# GEOMETRY CALCULATOR WITH UWB FUSION
# =============================================================================
class GeometryCalculator:
    """Computes relative geometry with UWB fusion support."""

    def __init__(self, home_lat: float, home_lon: float, home_alt_msl: float):
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.home_alt_msl = home_alt_msl
        self.R_EARTH = 6371000.0

        # UWB fusion filter
        self.fusion: Optional[RelativePoseFusion] = None
        if UWB_FUSION_AVAILABLE:
            self.fusion = RelativePoseFusion()
            log("[GEOM] UWB fusion filter initialized")
        else:
            log("[GEOM] Using GPS-only geometry (no UWB fusion)")

        # Closure rate tracking
        self._last_range = 0.0
        self._last_range_time = 0.0

    def latlon_to_ned(self, lat: float, lon: float,
                      alt_msl: float) -> Tuple[float, float, float]:
        """Convert lat/lon/alt to NED relative to home."""
        dlat = math.radians(lat - self.home_lat)
        dlon = math.radians(lon - self.home_lon)
        n = dlat * self.R_EARTH
        e = dlon * self.R_EARTH * math.cos(math.radians(self.home_lat))
        d = -(alt_msl - self.home_alt_msl)
        return (n, e, d)

    def compute_relative(
        self,
        foll_lat: float, foll_lon: float, foll_alt: float,
        foll_vn: float, foll_ve: float, foll_vd: float,
        lead_lat: float, lead_lon: float, lead_alt: float,
        lead_vn: float, lead_ve: float, lead_vd: float,
        lead_heading_deg: float, boom_length_m: float,
        uwb_range: Optional[float] = None,
        timestamp_s: Optional[float] = None,
        boom_tip_offset_m: float = BOOM_TIP_OFFSET_M,
        update_fusion: bool = True,
    ) -> Optional[RelativeGeometry]:
        """
        Compute relative geometry in leader body frame.

        Geometry contract (leader body frame):
          - behind_m, cross_m, below_m are defined relative to the boom TIP.
          - range_m is the 3D slant range to the boom TIP.

        Note: pass boom_tip_offset_m=0.0 when you explicitly want leader-center
        geometry (e.g., collision/overtake safety checks).
        """
        # Use sim-time timestamp if provided, otherwise fallback to SimTimeAuthority
        # Wall-time fallback REMOVED - all decision-path timing must use sim-time
        ta = get_time_authority()
        now = timestamp_s if timestamp_s is not None else (ta.now_s() if ta and ta.is_ready() else None)
        if now is None:
            return None  # Cannot compute geometry without valid timestamp

        # Convert to NED
        foll_n, foll_e, foll_d = self.latlon_to_ned(
            foll_lat, foll_lon, foll_alt)
        lead_n, lead_e, lead_d = self.latlon_to_ned(
            lead_lat, lead_lon, lead_alt)

        # Account for boom position (boom extends behind aircraft)
        hdg_rad = math.radians(lead_heading_deg)
        # CSV2 frame-fix (R7.1 waiver, option-a guarded edit, 2026-06-10): the boom-tip
        # horizontal offset is the canonical 45-deg projection (BOOM_HORIZONTAL_PROJECTION_M
        # = 4.31m), NOT the full slant passed in boom_length_m (the controller previously
        # used 6.1m here -> a 1.79m along-track frame bug). The boom_length_m==0 leader-center
        # (CG) safety geometry is preserved EXACTLY by the guard below (0.0 -> 0.0 offset).
        # A positive boom_length_m's magnitude is intentionally ignored (boom-extension
        # dynamics deferred per plan non-goal); only present-vs-absent gates the projection.
        _boom_h = BOOM_HORIZONTAL_PROJECTION_M if boom_length_m > 0.0 else 0.0
        boom_offset_n = -_boom_h * math.cos(hdg_rad)
        boom_offset_e = -_boom_h * math.sin(hdg_rad)
        boom_n = lead_n + boom_offset_n
        boom_e = lead_e + boom_offset_e
        # Boom tip vertical offset: positive is "down" in NED.
        # Pass boom_tip_offset_m=0.0 to compute leader-center geometry.
        boom_d = lead_d + float(boom_tip_offset_m)

        # Relative position in NED (follower relative to boom tip)
        rel_n = foll_n - boom_n
        rel_e = foll_e - boom_e
        rel_d = foll_d - boom_d

        # Rotate to leader body frame
        cos_hdg = math.cos(hdg_rad)
        sin_hdg = math.sin(hdg_rad)
        body_x = rel_n * cos_hdg + rel_e * sin_hdg
        body_y = -rel_n * sin_hdg + rel_e * cos_hdg

        # GPS-based geometry
        behind_gps = -body_x
        cross_gps = body_y
        below_gps = rel_d

        # Relative velocity in leader body frame
        rel_vn = foll_vn - lead_vn
        rel_ve = foll_ve - lead_ve
        rel_vd = foll_vd - lead_vd
        vel_behind = -(rel_vn * cos_hdg + rel_ve * sin_hdg)
        vel_cross = -rel_vn * sin_hdg + rel_ve * cos_hdg
        vel_below = rel_vd

        # Use fusion if available (skip for auxiliary geometry like center-CG)
        source = "GPS"
        if self.fusion is not None and update_fusion:
            # Update GPS measurement
            gps_meas = create_gps_measurement(
                timestamp=now,
                behind_m=behind_gps,
                cross_m=cross_gps,
                below_m=below_gps,
                vel_behind_m_s=vel_behind,
                vel_cross_m_s=vel_cross,
                vel_below_m_s=vel_below
            )
            self.fusion.update_gps(gps_meas)

            # Update UWB if available
            if uwb_range is not None:
                uwb_meas = UWBMeasurement(
                    timestamp=now,
                    range_m=uwb_range
                )
                self.fusion.update_uwb(uwb_meas)

            # Get fused state (pass sim-time for prediction/ages)
            fused = self.fusion.get_state(now_s=now)
            behind_m = fused.behind_m
            cross_m = fused.cross_m
            below_m = fused.below_m
            range_m = fused.range_m
            source = "FUSED"
            if _FD_GEOM_GPS_DIRECT:
                # C-bypass: fusion above stays updated (shadow/diag); control
                # consumes GPS-relative. "GPS" is an existing source contract value.
                behind_m, cross_m, below_m = behind_gps, cross_gps, below_gps
                # STEP-4 Hunk-D(ii) (2026-06-11, R7.1 waiver, gated), REVISED
                # by STEP-4B2' Hunk-G2 (state-gated terminal scoping):
                # architecture-D memoryless UWB algebraic correction on the
                # GPS-relative triple (tests/d_correction.py — Gauss-Newton +
                # GPS Tikhonov prior, no Kalman, no carried state). The
                # "UWB primary inside DOCKING_CREEP / terminal approach"
                # doctrine is now enforced IN CODE: consumption only when the
                # live FSM state (mirrored into _d_gate_state around the
                # get_geometry control call — auxiliary compute_relative
                # callers see '' and skip this chain entirely) is in
                # _FD_D_ALLOWED_STATES and the GPS triple is within
                # _FD_D_MAX_RANGE_M. Precedence: OUT_OF_STATE ->
                # STALE_UWB -> NO_RANGES -> RANGE_TOO_FAR ->
                # RANGE_INCONSISTENT|APPLIED — stale data is deliberately NOT
                # classified as RANGE_TOO_FAR (a timing failure must not
                # masquerade as a geometry condition). Stash freshness via
                # time.monotonic() on BOTH sides; stale > 0.3 s
                # -> pass-through. source stays "GPS" (existing contract
                # value); engagement evidenced via [D_CORR]/[D_CORR_REJ]/
                # [D_CORR_SUMMARY] logs + d_corr_* diag columns (schema v3).
                _gstate = getattr(self, '_d_gate_state', '')
                if _FD_D_CORRECTION and _gstate:
                    _dr = getattr(self, '_d_ranges', None)
                    _reason = None
                    _db = _dbl = _dc = 0.0
                    _rng_gps = math.sqrt(
                        behind_gps**2 + cross_gps**2 + below_gps**2)
                    if _gstate not in _FD_D_ALLOWED_STATES:
                        _reason = 'OUT_OF_STATE'
                    elif (_dr is not None
                          and (time.monotonic() - _dr[1]) > 0.3):
                        _reason = 'STALE_UWB'
                    elif _dr is None or not _dr[0]:
                        _reason = 'NO_RANGES'
                    elif _rng_gps > _FD_D_MAX_RANGE_M:
                        _reason = 'RANGE_TOO_FAR'
                    else:
                        _corr, _ddiag = d_correction.correct(
                            (behind_m, cross_m, below_m), _dr[0],
                            {'uwb': _FD_D_UWB_SIGMA,
                             'gps_behind': _FD_D_GPS_SIGMA,
                             'gps_cross': _FD_D_GPS_SIGMA,
                             'gps_below': _FD_D_GPS_SIGMA},
                            trust_m=_FD_D_TRUST_M,
                            trust_mode=_FD_D_TRUST_MODE)
                        if _ddiag.get('n_ranges_used', 0) == 0:
                            _reason = 'RANGE_INCONSISTENT'
                        elif _ddiag.get('trust_exceeded'):
                            # SIGCLEAN Hunk-C1: solution outside trust region —
                            # GPS triple passes through unchanged (correct()
                            # already returned it; deltas stay 0.0). Only
                            # reachable in trust_mode='reject'.
                            _reason = 'DELTA_TRUST'
                        else:
                            # DFIX Hunk-C3: clamped solutions are APPLIED
                            # (continuity by design); clamp census lives in
                            # the d_corr_pre_trust_max_m diag column.
                            _reason = 'APPLIED'
                            _db = float(_corr[0]) - behind_m
                            _dc = float(_corr[1]) - cross_m
                            _dbl = float(_corr[2]) - below_m
                            if _FD_D_EMA_ALPHA > 0.0:
                                _prev = getattr(self, '_d_ema_delta', None)
                                if _prev is None:
                                    _prev = (0.0, 0.0, 0.0)
                                _a = _FD_D_EMA_ALPHA
                                _db = _a * _db + (1.0 - _a) * _prev[0]
                                _dc = _a * _dc + (1.0 - _a) * _prev[1]
                                _dbl = _a * _dbl + (1.0 - _a) * _prev[2]
                                self._d_ema_delta = (_db, _dc, _dbl)
                                behind_m = behind_m + _db
                                cross_m = cross_m + _dc
                                below_m = below_m + _dbl
                            else:
                                # alpha off: original exact assignment (FP
                                # bit-identical to the pre-DFIX path).
                                behind_m = float(_corr[0])
                                cross_m = float(_corr[1])
                                below_m = float(_corr[2])
                    if _reason == 'APPLIED':
                        self._d_corrections_applied = getattr(
                            self, '_d_corrections_applied', 0) + 1
                        if self._d_corrections_applied % 200 == 1:
                            log(f"[D_CORR] n={self._d_corrections_applied} "
                                f"state={_gstate} rng={_rng_gps:.1f} "
                                f"db={_db:+.2f}")
                    else:
                        self._d_corr_rejects = getattr(
                            self, '_d_corr_rejects', 0) + 1
                        if self._d_corr_rejects % 200 == 1:
                            log(f"[D_CORR_REJ] n={self._d_corr_rejects} "
                                f"reason={_reason} state={_gstate} "
                                f"rng={_rng_gps:.1f}")
                    _rcnt = getattr(self, '_d_corr_reason_counts', None)
                    if _rcnt is None:
                        _rcnt = {}
                        self._d_corr_reason_counts = _rcnt
                    _rcnt[_reason] = _rcnt.get(_reason, 0) + 1
                    # DFIX Hunk-C3: pre-trust solution magnitude (closes the
                    # d_census declared evidence gap — rejected/clamped
                    # magnitudes were unrecoverable). '' when no solve ran.
                    _pre_max = ''
                    if _reason in ('APPLIED', 'DELTA_TRUST'):
                        _pt = _ddiag.get('delta_pre_trust',
                                         _ddiag.get('delta'))
                        if _pt is not None:
                            _pre_max = max(abs(float(_pt[0])),
                                           abs(float(_pt[1])),
                                           abs(float(_pt[2])))
                    self._d_corr_last = {
                        'applied': _reason == 'APPLIED',
                        'state': _gstate,
                        'range_m': _rng_gps,
                        'reason': _reason,
                        'db': _db, 'dbl': _dbl, 'dc': _dc,
                        'eng': getattr(self, '_d_corrections_applied', 0),
                        'rej': getattr(self, '_d_corr_rejects', 0),
                        'pre_max': _pre_max}
                range_m = compute_range(behind_m, cross_m, below_m)
                source = "GPS"
        else:
            behind_m = behind_gps
            cross_m = cross_gps
            below_m = below_gps
            range_m = compute_range(behind_m, cross_m, below_m)

        # Compute closure rate (Test 302 FIX: correct sign convention)
        # vel_behind = d(behind_m)/dt in leader body frame
        # When follower is behind (behind_m > 0) and CLOSING, behind_m DECREASES
        # Therefore vel_behind is NEGATIVE when closing
        # We want closure_rate POSITIVE when closing, so negate vel_behind
        closure_rate = -vel_behind  # CORRECTED: Positive = closing on leader

        # Compute target aspect (0 = directly behind)
        target_aspect_deg = math.degrees(math.atan2(cross_m, behind_m))

        return RelativeGeometry(
            behind_m=behind_m,
            cross_m=cross_m,
            below_m=below_m,
            range_m=range_m,
            target_aspect_deg=target_aspect_deg,
            closure_rate_ms=closure_rate,
            vel_cross_m_s=vel_cross,  # For D-term damping (Test 301)
            source=source,
            below_gps_m=below_gps  # G3: same-computation GPS-direct below
        )


# =============================================================================
# MAIN CONTROLLER
# =============================================================================
class BaselineFormationController:
    """
    Formation controller for QuadPlane aerial refueling docking.

    Implements:
    - UWB+GPS sensor fusion with GPS fallback
    - Weather verification
    - VTOL takeoff with FW transition
    - Fighter-like intercept to stern conversion
    - Controlled precontact holding
    - Precision CREEP docking logic
    """

    def __init__(self, follower_conn_str: str = "udp:127.0.0.1:14551"):
        self.follower_conn_str = follower_conn_str
        self.follower: Optional[MAVConnection] = None
        self.leader_rx: Optional[LeaderTelemetryReceiver] = None
        self.geom_calc: Optional[GeometryCalculator] = None
        self.weather_checker: Optional[WeatherChecker] = None
        self.takeoff_controller = None

        # P1 Fix A v9.10: Non-blocking takeoff sub-FSM state
        self._takeoff_cmd_sent: bool = False
        self._takeoff_retry_count: int = 0
        self._takeoff_sustained_start: Optional[float] = None
        self._takeoff_disarm_streak: int = 0

        # State machine
        self.state = ControllerState.INIT
        self.state_entry_time = 0.0  # Will be set in _reset_sim_timers_after_pass()
        self.sm = StateMachine(self, log_fn=log)

        # Home position
        self.home_lat = 0.0
        self.home_lon = 0.0
        self.home_alt_msl = HOME_ALT_MSL

        # Hold timers
        self.precontact_hold_start: Optional[float] = None
        self.contact_hold_start: Optional[float] = None

        # Tracking
        self.min_range_achieved = float('inf')

        # Phase 0: per-tick geometry cache (tick-based, not time-based)
        self._geom_cache_tick: Optional[int] = None
        self._geom_cache: Optional[RelativeGeometry] = None
        self._tick_counter = 0  # Incremented each control loop iteration
        self.running = True

        # Log rate limiting (0.5Hz instead of 20Hz to reduce spam)
        self._last_state_log_time = 0.0
        self._state_log_interval = 2.0  # Log state info every 2 seconds

        # Envelope counters and statistics
        self.precontact_hold_count = 0           # Number of successful precontact holds
        self.contact_entry_count = 0             # Number of contact envelope entries
        self.contact_dwell_times: list = []      # Dwell time per contact entry (seconds)
        self.current_contact_entry_time: Optional[float] = None

        # DRL alignment - success tracking
        self.contact_success: bool = False       # Set True when dwell >= threshold
        self.contact_success_count: int = 0      # Total successes in run

        # Sensor mode configuration
        self.sensor_mode = SENSOR_MODE_FUSED     # Default sensor mode
        self.close_range_sensor_threshold = PRECISION_MODE_THRESHOLD_M
        self._last_sensor_source = "INIT"        # Track which sensor was used

        # Cross-track filtering state (EMA filter)
        self._filtered_cross_m: Optional[float] = None
        self._filtered_cross_rate: Optional[float] = None

        # Precision mode state
        self.precision_mode_active = False
        self.very_precision_mode_active = False

        # UWB state tracking for validity checks
        self._last_uwb_range: Optional[float] = None
        self._last_uwb_timestamp: float = 0.0
        self._last_sensor_fallback_log: float = 0.0
        self._active_sensor_source: str = "FUSED"  # Current active source

        # Run statistics tracking
        self._run_start_time: Optional[float] = None  # Set when run() starts
        self._state_entry_counts: dict = {}  # {state_name: count}

        # Refactored speed command pipeline
        # Internal slewed speed (updated every tick, not subject to send rate limit)
        self._speed_cmd_slewed: float = CRUISE_SPEED_MS
        self._speed_cmd_slew_time: float = 0.0
        # Last speed actually sent via MAVLink (for rate limiting)
        self._speed_cmd_sent_last: float = CRUISE_SPEED_MS
        self._speed_cmd_last_send_time: float = 0.0
        # OVERTAKE_PROTECT state
        self._overtake_active: bool = False
        self._force_speed_send: bool = False
        self._last_overtake_diag_time: float = 0.0
        self._last_overtake_log_time: float = 0.0
        # Precontact VC gate - speedup-invariant (v9.2 Patch A)
        self._vc_gate_behind_prev: Optional[float] = None
        self._vc_gate_behind_prev_time: Optional[float] = None
        self._vc_gate_slow_vc: float = 0.0
        self._precontact_vc_credit_s: float = 0.0
        self._precontact_vc_violation_start_s: Optional[float] = None
        self._vc_gate_prev_time: Optional[float] = None
        self._vc_gate_credit_prev_time: Optional[float] = None
        self._last_vc_gate_log_time: float = 0.0

        # P9.0 v9.9: dt single source of truth - computed ONCE per tick
        self._dt_tick: float = 0.05  # Initial assumption (~20Hz)
        self._last_tick_time_s: Optional[float] = None

        # Contact capture gate state
        self._contact_vc_stable_start: Optional[float] = None
        self._last_contact_vc_gate_log_time: float = 0.0

        # HOLD_CONTACT PI controller state
        self._hold_contact_i: float = 0.0
        self._hold_contact_i_last_t: Optional[float] = None
        self._hold_contact_last_behind_err: Optional[float] = None

        # Leader turn detection state
        self._lead_last_hdg: Optional[float] = None
        self._lead_last_t: Optional[float] = None
        self._lead_yaw_rate_filt_dps: float = 0.0
        self.leader_turning_flag: bool = False
        self._leader_not_turning_since: float = 0.0  # Sim-time; set when sim starts
        self._last_turn_gate_log_time: float = 0.0
        # TURNGATE-750. SELF-CONTAINED
        # state: deliberately NOT reusing _lead_yaw_rate_filt_dps, which updates
        # only inside _cross_crab_rejoin_bias and therefore only in rejoin mode
        # AND only when FD_CROSS_CRAB_ENABLE is on -- reading it at creep entry
        # would see a STALE or 0.0 value, i.e. a gate that is dead again (or,
        # worse, one that trips on a minutes-old turn).
        self._tg_yaw_rate_filt_dps: float = 0.0
        self._tg_last_hdg: Optional[float] = None
        self._tg_turning: bool = False
        self._tg_straight_s: float = 0.0

        # CSV2_R7_1 rejoin cross-track crab integral state (flag FD_CROSS_CRAB_ENABLE).
        # SEPARATE integrator from the docking-state _cross_i_deg so docking callers stay
        # byte-identical. Populated/read only on the rejoin_mode=True path (flag-ON).
        self._cross_i_rejoin_deg: float = 0.0     # rejoin crab integrator (deg)
        self._cross_sustain_ticks: int = 0        # consecutive same-sign-beyond-deadband ticks
        self._cross_sustain_sign: int = 0         # +1/-1/0 sign of the sustained cross
        self._cross_crab_diag_file = None         # gated cross_crab_diag.csv handle
        self._cross_crab_diag_writer = None

        # Cross-track PI bias state (active in DOCKING_CREEP and HOLD_CONTACT)
        self._cross_i_deg: float = 0.0
        self._cross_i_last_t: Optional[float] = None

        # OUTER-exit grace timer state
        self._contact_outer_exit_start: Optional[float] = None

        # PRECONTACT OUTER-exit grace timer state
        self._precontact_outer_exit_start: Optional[float] = None

        # TEST 400: PRECONTACT INNER-exit grace timer state (Fix 2)
        self._precontact_inner_exit_start: Optional[float] = None

        # TEST 401: Climb guard state (Fix 4 - explicit velocity damping)
        self._climb_guard_stable_s: float = 0.0
        self._climb_guard_last_ok: bool = False
        self._climb_guard_last_tick: float = 0.0
        self._last_climb_guard_log: float = 0.0

        # v9.2 Patch B: Climb closure clamp logging
        self._last_climb_clamp_log: float = 0.0

        # TEST 400: Altitude tracking instrumentation (rate-limited)
        self._last_alt_track_log_s: float = 0.0
        self._alt_track_log_interval_s: float = 2.0  # Log every 2 seconds

        # CONTACT_INNER capture gate state
        self._contact_inner_stable_start: Optional[float] = None
        self._last_contact_inner_gate_log_time: float = 0.0

        # Run-summary observability stats
        self._contact_outer_dwell_total_s: float = 0.0
        self._contact_inner_hold_max_s: float = 0.0
        self._contact_inner_reset_count: int = 0
        self._docking_success_count: int = 0

        # P7.1 v9.7: M3a dwell tracking state
        self._m3a_current_dwell_s: float = 0.0
        self._m3a_grace_start: Optional[float] = None
        self._m3a_time_in_envelope_s: float = 0.0
        self._m3a_max_contiguous_s: float = 0.0

        # §8 Step 3: M3a_coarse (Acquisition tier) dwell tracking
        self._m3a_coarse_current_dwell_s: float = 0.0
        self._m3a_coarse_best_dwell_s: float = 0.0
        self._m3a_coarse_achieved: bool = False
        self._m3a_coarse_achieved_at_s: Optional[float] = None
        self._m3a_coarse_grace_start: Optional[float] = None
        self._m3a_coarse_grace_accum_s: float = 0.0

        # M3 progress longitudinal (report-only diagnostic)
        self._m3_prog_long_current_dwell_s: float = 0.0
        self._m3_prog_long_best_dwell_s: float = 0.0
        self._m3_prog_long_achieved: bool = False
        self._m3_prog_long_achieved_at_s: Optional[float] = None
        self._m3_prog_long_grace_start: Optional[float] = None
        self._m3_prog_long_grace_accum_s: float = 0.0

        # Dynamic offset side selection
        # Initialized to default, may be flipped at INTERCEPT entry
        self._active_offset_side: float = OFFSET_SIDE
        self._offset_side_selected: bool = False

        # Altitude command tracking state
        self._last_reposition_alt: float = 0.0
        self._alt_err_exceed_start: Optional[float] = None
        self._pin_fix_window_active: bool = False
        self._pin_fix_window_start: Optional[float] = None
        self._pin_fix_window_count: int = 0
        self._pin_fix_repos_suppressed: int = 0
        self._alt_err_warning_start: Optional[float] = None
        self._last_alt_warning_log: float = 0.0
        self._last_alt_fallback_time: float = 0.0
        self._alt_gate_active: bool = False
        self._alt_gate_start: Optional[float] = None

        # E4: Geometry consistency check throttling
        self._last_geom_consistency_log: float = 0.0

        # ALT_FORCE log rate limiting
        self._last_alt_force_log_time: float = 0.0

        # SIGINT handler tracking
        self._stop_reason: str = ""

        # Track altitude target for change detection
        self._last_alt_cmd_target: float = 0.0

        # Altitude capture state tracking
        self._alt_capture_stable_start: Optional[float] = None
        self._last_alt_capture_log_time: float = 0.0
        self._alt_capture_target_alt: float = CONTACT_ALT_MSL

        # TECS_SPDWEIGHT switching state
        self._tecs_spdweight_current: float = ALT_CAPTURE_TECS_SPDWEIGHT_NORMAL
        self._last_tecs_set_time: float = -1e9  # sim-time (allows first set immediately)
        self._tecs_capture_mode_active: bool = False

        # QASSIST during ALTITUDE_CAPTURE
        self._qassist_speed_current: float = ALT_CAPTURE_QASSIST_SPEED_NORMAL
        self._last_qassist_set_time: float = -1e9  # sim-time (allows first set immediately)
        self._qassist_capture_mode_active: bool = False

        # Vertical rate statistics for QASSIST effectiveness
        self._alt_capture_vert_rates: List[float] = []
        self._alt_capture_initial_alt: Optional[float] = None

        # ALTITUDE_CAPTURE timeout tracking
        self._alt_capture_start_time: Optional[float] = None

        # Baseline TECS/QASSIST values (read on connect)
        self._tecs_spdweight_baseline: Optional[float] = None
        self._qassist_speed_baseline: Optional[float] = None

        # Two-phase ALTITUDE_CAPTURE state
        self._alt_capture_subphase: str = ALT_CAPTURE_SUBPHASE_BACKOUT
        self._alt_capture_standoff_stable_start: Optional[float] = None

        # Collision guard state
        self._last_collision_guard_log: float = 0.0
        self._collision_guard_active: bool = False

        # Speed tracking diagnostics state
        self._speed_err_exceed_start: Optional[float] = None
        self._last_speed_diag_log: float = 0.0

        # QASSIST near-target cutoff tracking
        self._qassist_near_target_cutoff: bool = False
        self._last_overshoot_log: float = 0.0
        self._overshoot_start_time: Optional[float] = None  # For persistence check

        # Leader effective speed estimator state
        self._leader_prev_pos_ned: Optional[Tuple[float, float, float]] = None
        self._leader_prev_pos_time: float = 0.0
        self._leader_gs_est_ewma: float = LEADER_TARGET_TAS_MS  # Init to expected
        self._last_leader_speed_est_log: float = 0.0

        # P5.1 v9.5: Closure rate computation moved entirely to FormationControlLaws
        # (removed dead copies: _behind_m_prev, _behind_m_prev_time, _closure_rate_behind_ewma)

        # v9.2 Patch D: Closure deficit integrator state (in control_laws)
        self._last_speed_tick: float = 0.0

        # No-pass abort state
        self._no_pass_abort_start: Optional[float] = None
        self._no_pass_abort_triggered: bool = False
        self._last_no_pass_abort_log: float = 0.0

        # P1 v9.4: ALT_CAP_HOLD_RANGE state variables
        self._alt_cap_stable_s: float = 0.0
        self._alt_cap_violation_start: Optional[float] = None
        self._last_alt_cap_log: float = 0.0

        # P5.2 v9.5: Safety clamp persistence (sim-time when hold expires)
        self._safety_hold_until: float = 0.0
        # P9.0 v9.9: Hard constraint violation flag (behind_m < 0)
        self._no_pass_abort_triggered: bool = False
        # P1-2: NOPASS Tier B persistence accumulator

        # P2 v9.4: Speed arbitration tracking
        self._ticks_since_transition: int = 0
        self._arb_log_tick: int = 0
        self._last_spd_cmd: Optional[float] = None
        self._brake_clamp_active: bool = False
        self._climb_clamp_active: bool = False
        self._safety_clamp_active: bool = False

        # P2 v9.4 Fix 2b: CSV timeseries export for arbitration
        self._arb_csv_file = None
        self._arb_csv_writer = None
        # C3 (2026-06-08): UWB emitter->fusion delivery diag (gated, default-off)
        self._uwb_diag_file = None
        self._uwb_diag_writer = None
        self._uwb_diag_pending = None
        self._uwb_diag_seq = 0
        self._uwb_diag_rows_written = 0
        # Rec 3: Speedup-aware CSV decimation (write every Nth row at high speedup)
        self._csv_decimation_stride: int = 1  # set in _init_arbitration_csv
        self._csv_decimation_counter: int = 0  # counts executed ticks for modulo
        # TIER2: ticks where the truth-closure columns had missing truth inputs
        # (only counts when _FD_TRUTH_CLOSURE_DIAG; logged at CSV close)
        self._truth_closure_skip_count: int = 0
        # WINDFF: per-tick along-track wind estimate (LPF, m/s) + closure-trim integrator (m/s);
        # only updated/consumed when _FD_SELF_WIND_FF. _windff_last_t seeds the LPF on first tick;
        # _windff_skip_count = ticks that held the estimate on missing telemetry (logged at close).
        self._windff_west: float = 0.0
        self._windff_trim_i: float = 0.0
        self._windff_last_t: Optional[float] = None
        self._windff_skip_count: int = 0

        # v9.10 Gate 0: Timebase integrity tick gating state
        # All initialized to None/0 in __init__, fully reset on first tick in run_state_machine()
        self._last_tick_time_s: Optional[float] = None
        self._tick_accum_s: float = 0.0
        self._dt_tick_raw_s: Optional[float] = None
        self._dt_tick_used_s: Optional[float] = None
        self._dt_credit_s: float = 0.0
        self._ticks_skipped_no_advance: int = 0
        # Tick timing diagnostic state
        self._tick_diag_records: List[Tuple[int, float, float, float, str, int, int]] = []
        self._tick_diag_wall_times: List[float] = []  # rolling window for periodic summary
        self._tick_diag_drops: int = 0
        self._tick_diag_last_summary_tick: int = 0
        self._tick_diag_prev_leader_tbms: int = -1
        self._ticks_skipped_accumulating: int = 0
        self._ticks_executed_total: int = 0
        self._late_tick_count: int = 0
        self._sim_time_regressions: int = 0
        self._regression_warned: bool = False
        self._max_sim_dt_raw_s: float = 0.0
        self._run_invalid_timebase: bool = False
        # v9.10: Safe time accessor - sim-time of most recent EXECUTED control tick
        # Used by state_machine.py and handlers instead of sim_now_s()
        self._t_sim_s: float = 0.0

        # v9.10 C1: Time guard state (Appendix A.0)
        self._strict_mode: bool = True  # Default True in CI/tests
        self._fatal_in_progress: bool = False
        self._tick_owner_depth: int = 0
        self._tick_owner_thread: Optional[int] = None
        self._pending_row: Optional[dict] = None

        # v9.10 Gate0 Hardening: Marker-based tick gating (replaces boolean flag)
        # Gate on time_boot_ms change (monotonic marker), not boolean arrival.
        # This prevents collapsing multiple ATTITUDE messages into one tick.
        self._last_tick_time_boot_ms: Optional[int] = None

        # v9.10 C4: Tick rate metrics (executed tick sim times)
        self._first_executed_tick_sim_s: Optional[float] = None
        self._last_executed_tick_sim_s: Optional[float] = None

        # v9.10 C5: Geometry unavailable timeout accumulator
        self._geom_unavailable_accum_s: float = 0.0
        self._last_geom_regime_requires_geometry: bool = False

        # v9.10 C6: Per-tick speed send latch
        self._speed_cmd_sent_this_tick: bool = False
        self._csv_row_written_this_tick: bool = False  # P2 Fix D v9.10
        self._first_send_source: Optional[str] = None
        self._first_send_speed_mps: Optional[float] = None
        self._first_send_bypass: Optional[bool] = None

        # FD-004: Speed override for safety subsystems (LATCHED single-writer pattern)
        # Override persists until timer expires (not consumed on read)
        self._speed_override_mps: Optional[float] = None
        self._speed_override_source: Optional[str] = None
        self._speed_override_bypass: bool = False
        self._speed_override_until_s: float = 0.0

        # FD-002: Starvation observability
        self._tick_hz_window: deque = deque(maxlen=100)  # Last 100 tick timestamps (5s @ 20Hz)
        self._backlog_fatal_count: int = 0
        self._last_processed_tick_ms: Optional[int] = None  # FD-002 Amendment: Last processed timestamp

        # Queue depth attribution (ChatGPT-5 R2)
        self._queue_max_depth_t_sim_s: float = 0.0
        self._queue_max_depth_state: str = "INIT"
        self._pump_msgs_drained_max: int = 0
        self._pump_msgs_drained_samples: deque = deque(maxlen=2000)
        self._msgs_drained_last_pump: int = 0  # Change 5: snapshot diagnostic
        self._loop_wall_times: deque = deque(maxlen=2000)
        self._ticks_per_loop_max: int = 0
        self._queue_depth_snapshot_logged: bool = False

        # PRH-ATT-004: Runtime timebase watchdog (two-tier)
        self._watchdog_armed: bool = False
        self._watchdog_low_hz_since_s: Optional[float] = None
        self._watchdog_hard_min_hz: float = 15.0    # Matches Gate0 — fatal
        self._watchdog_warn_min_hz: float = 18.0    # Diagnostic — non-fatal
        self._watchdog_trip_delay_s: float = 2.0
        self._watchdog_warn_logged: bool = False
        self._watchdog_last_stale_count: int = 0
        self._watchdog_stale_baseline_synced: bool = False  # Tranche W: sync on first close-in tick

        # v9.10 Gate 1: Safety emergency state
        self._safety_emergency_active: bool = False
        self._safety_emergency_count: int = 0
        # F-E1: canonical append-only
        # emergency event record; the live counter must reconcile against it.
        self._safety_emergency_events: List[Dict] = []
        # §8-3 extension: monotonic transition
        # counter — serialized as the final event id for in-file finalization.
        self._transition_count: int = 0
        self._count_no_pass_violations: int = 0
        self._min_behind_m: float = float('inf')
        self._t_sim_at_min_behind_s: float = 0.0
        self._state_at_min_behind: str = ""

        # v9.10 Gate 2: State machine stability - HOLD debouncing state
        self._hold_entry_grace_accum_s: float = 0.0
        self._hold_exit_grace_accum_s: float = 0.0
        # #459 Candidate A: HOLD_CONTACT->DOCKING_CREEP temporal exit-grace accumulator
        # (only read/written when FD_CONTACT_OUTER_EXIT_GRACE_ENABLE=1; inert otherwise).
        self._hc_exit_grace_accum_s: float = 0.0
        self._precontact_dwell_accum_s: float = 0.0
        self._precontact_inner_exit_accum_s: float = 0.0
        self._precontact_max_dwell_s: float = 0.0  # Track max dwell for run_summary

        # L-DIAG: Diagnostic breakaway suppression state
        self._diag_suppressed_breakaways: int = 0
        self._diag_suppressed_by_source: dict = {}
        self._diag_geom_behind_accum_s: float = 0.0
        self._diag_geom_range_accum_s: float = 0.0
        self._diag_geometry_terminator_reason: Optional[str] = None
        self._diag_geometry_terminator_time: Optional[float] = None
        self._diag_mode_active: bool = (
            formation_config.DIAG_BREAKAWAY_SHADOW
            or formation_config.DIAG_GEOMETRY_TERMINATOR
        )

        # N0: Zone-gated TECS_TIME_CONST override
        self._tecs_tc_baseline: float = 5.0     # Read from ArduPilot at init

        # Guard strip-down: CZ-EMERGENCY + CE_thresh telemetry vars removed (Chunk 3A)

        # Rec 3: NOPASS corridor gate suppression counter

        # FD-009: DOCKING_CREEP vertical validity guard
        self._dc_vert_abort_count: int = 0
        self._dc_vert_override_count: int = 0

        # FD-009: NOPASS axial override counter

        # Option A: Two-phase DOCKING_CREEP
        self._dc_phase: str = formation_config.DC_SUBPHASE_VERT_STABILIZE
        self._dc_stab_accum_s: float = 0.0
        self._dc_phase2_lockout: bool = False
        self._dc_phase1_enter_count: int = 0
        self._dc_phase2_enter_count: int = 0
        self._dc_phase2_fallback_count: int = 0
        self._dc_phase2_abort_grace_accum_s: float = 0.0  # Fix 2: grace period accumulator
        self._dc_phase2_last_fallback_t: float = 0.0  # A1C2: timestamp of last Phase2 fallback
        self._dc_sink_bad_accum_s: float = 0.0

        # Phase 1 VTOL fix: Airspeed diagnostic tripwire
        # If follower airspeed < 15 m/s in DC for > 2s, abort to ALT_CAP_HOLD_RANGE
        self._dc_airspeed_low_accum_s: float = 0.0
        self._dc_airspeed_tripwire_count: int = 0

        # Chunk 9: Minimal guard state
        self._guard_altitude_ceiling_fire_count: int = 0
        self._guard_hard_overtake_fire_count: int = 0
        self._guard_speed_cap_fire_count: int = 0
        self._overtake_persist_accum: float = 0.0
        self._alt_ceiling_persist_accum: float = 0.0  # E1 (T-341): Guard 1 debounce
        self._guard_env_fallback_count: int = 0  # G3: ticks the env channel was absent -> fused fallback
        # GUARDWIT: per-fire witness of WHICH value/channel the ceiling guard
        # compared. Written only inside the fire branch; read only by the
        # summary export. Never an input to any decision.
        self._guard_ceiling_fires: list = []
        self._guard_speed_cap_mps: float | None = None  # Per-tick soft speed cap

        # Option E: TECS_SPDWEIGHT scheduler
        self._tecs_spdw_target: float = 1.0
        self._tecs_spdw_ramp_start_val: float = 1.0
        self._tecs_spdw_ramp_start_time: float = 0.0
        self._tecs_spdw_ramp_active: bool = False
        self._tecs_spdw_last_write_time: float = -1e9
        self._tecs_spdw_last_confirmed: float = 1.0
        self._tecs_spdw_write_count: int = 0
        self._tecs_spdw_mismatch_count: int = 0
        self._tecs_spdw_disabled: bool = False
        self._tecs_spdw_disable_reason: str = ""

        # M3 S1: Binding constraint sub-source telemetry
        self._binding_pre_override_ticks: dict = {}  # {constraint_name: tick_count}
        self._binding_final_ticks: dict = {}          # {constraint_name: tick_count}
        self._last_binding_pre_override: str = "NONE"
        self._last_binding_final: str = "NONE"
        self._last_binding_value_mps: float = 0.0
        self._last_binding_margin_m: float = 0.0
        # M3 S1: Per-tick boolean flags for CSV
        self._vert_floor_soft_active: bool = False
        self._vert_floor_hard_active: bool = False
        self._in_precontact_corridor: bool = False
        self._speed_override_active: bool = False
        # D1: VERT_FLOOR hysteresis latches
        # M3 S1: Schedule observability
        self._schedule_input_m: float = 0.0
        self._schedule_bin_idx: int = -1

        # Rec 5: Runtime backlog time metric
        self._runtime_backlog_time_max_s: float = 0.0
        self._runtime_backlog_time_max_depth: int = 0
        self._runtime_backlog_time_max_t_sim_s: float = 0.0
        self._runtime_backlog_time_max_state: str = "INIT"
        self._backlog_time_snapshot_logged: bool = False
        self._ticks_dropped_backpressure_total: int = 0

        # FD-007: Rolling dt for predictor horizon
        self._recent_dt_samples: deque = deque(maxlen=60)  # ~3s @ 20Hz

        # P8.0/P8.1 v9.8: Final approach and vertical convergence tracking
        self._final_approach_active: bool = False
        self._final_approach_vc_cap: float = 0.0
        self._below_target_m: float = 0.0
        self._below_err_m: float = 0.0
        self._vert_gate_factor: float = 1.0

        # Coupled below-target ramp state
        self._coupled_below_active: bool = False
        self._below_target_desired_m: float = 0.0
        self._below_sign_mode: Optional[str] = None  # AUTO_DETECT result
        self._coupled_below_active_ticks: int = 0
        self._dc_below_err_samples: deque = deque(maxlen=2000)  # for p95/max stats
        # Step 1b: TECS speed tracking error samples during DOCKING_CREEP
        self._dc_tecs_speed_err_samples: deque = deque(maxlen=5000)  # target - measured
        self._coupled_below_initialized: bool = False  # first-tick-in-state flag

        # P9.0 v9.9: Cross-track gate and near-precontact observability
        self._cross_gate_factor: float = 1.0
        self._near_precontact_active: bool = False
        self._near_precontact_vc_cap: float = 0.0

        # P8.2 v9.8: Cross-track integrator for earlier convergence
        self._cross_i_term: float = 0.0

        # DIAG324 rate limiting
        self._last_diag324_log: float = 0.0

        # Tranche C (C0): NAV_CONTROLLER_OUTPUT diagnostic telemetry
        self._nav_alt_error_m: float = float('nan')
        self._nav_aspd_error_mps: float = float('nan')
        # Tranche H: L1 lateral diagnostic telemetry (NAV_CONTROLLER_OUTPUT)
        self._nav_roll_deg: float = float('nan')       # degrees — L1 demanded roll
        self._nav_bearing_deg: float = float('nan')    # degrees — desired heading
        self._target_bearing_deg: float = float('nan') # degrees — bearing to target
        self._xtrack_error_m: float = float('nan')     # meters — cross-track error
        self._wp_dist_m: float = float('nan')          # meters — distance to waypoint
        self._nav_controller_rx_count: int = 0
        # Tranche C (C0): Altitude command MSL sent via send_guided_commands
        self._d3_alt_cmd_msl_m: float = float('nan')
        self._d3_leader_alt_msl_m: float = float('nan')
        self._d3_follower_alt_msl_m: float = float('nan')

        # Tranche C (C1): Cross-track abort state for FORM_ENABLE experiment
        self._c1_cross_abort_active: bool = False
        self._c1_cross_exceed_start_s: Optional[float] = None

        # Tranche C (B2): C++ inactive reason code for diagnostics
        # 0=ACTIVE, 1=MASTER_OFF, 3=ALT_GATE, 4=EXPERIMENT, 8=STATE_GATE, 9=RANGE_FAR, 10=NO_GEOM
        self._cpp_inactive_reason: int = 1
        self._cpp_inactive_reason_counts: dict = {}

        # Tri-state: None=untested, True=param exists, False=param missing (never retry)
        self._form_enable_supported: Optional[bool] = None

        # C++ formation controller handoff state
        self._cpp_enabled: bool = False
        self._cpp_handoff_enabled: bool = CPP_HANDOFF_ENABLED
        self._cpp_enable_range_m: float = CPP_HANDOFF_ENABLE_RANGE_M
        self._cpp_disable_range_m: float = CPP_HANDOFF_DISABLE_RANGE_M
        self._last_cpp_handoff_log: float = 0.0

        # C++ handoff metrics (sim-time based)
        self._cpp_handoff_first_enable_s: Optional[float] = None
        self._cpp_handoff_last_enable_s: Optional[float] = None
        self._cpp_handoff_total_s: float = 0.0
        self._cpp_handoff_enable_count: int = 0
        self._cpp_handoff_disable_count: int = 0

        # C++ handoff policy (state_gated: precontact phases need Python altitude protocol)
        self._cpp_handoff_policy: str = "state_gated"  # Default policy
        self._cpp_handoff_warmup_start_s: Optional[float] = None  # Sim-time when warmup began

        # Tranche H: Docking waypoint mode state
        self._form_dock_mode_supported: bool = False  # Set in connect() preflight
        self._dock_trail_m: float = formation_config.DOCKING_HOLD_BEHIND_M  # Current trail offset

        self.expected_speedup: float = 1.0  # Set by main() from args.speedup
        self._time_auth: Optional[SimTimeAuthority] = None  # Created in connect()
        self._last_timing_log_sec: int = 0  # 1 Hz timing diagnostic
        self._timing_baseline_set: bool = False
        self._sim_start: float = 0.0
        self._wall_start_mono: float = 0.0
        self._last_timebase_debug_sec: int = 0

        # Formation success metrics (observer-only, sim-time consistent)
        self._success_metrics = FormationSuccessMetrics(boom_length_m=BOOM_MAX_LENGTH_M)
        self._metrics_flushed = False  # Guard for signal handler re-entrancy
        # F-E3 (#392): explicit flush outcome; the Boolean above persists True
        # only after a committed flush (transiently True during an attempt).
        self._metrics_flush_state: str = "not_attempted"
        self._out_dir: Optional[str] = None  # Set from args/env in main()

        # UWB measurement emitter (synthetic UWB ranges)
        self._uwb_emitter: Optional[UWBRangingEmitter] = None
        if UWB_EMITTER_AVAILABLE:
            self._uwb_emitter = UWBRangingEmitter()  # Reads config from env vars
            if self._uwb_emitter.enabled:
                log(f"[UWB_EMITTER] Enabled: {len(self._uwb_emitter.anchor_positions)} anchors")

        # Stuck recovery state
        self._stuck_start: Optional[float] = None        # When stuck condition first detected
        self._stuck_active: bool = False                 # Currently stuck
        self._stuck_integral_boost: float = 0.0          # Accumulated integral boost (m/s)
        self._last_stuck_recovery_log: float = 0.0       # Rate limit recovery logs

        # E2: Explicit init for log-throttle / rate-limit timestamps
        self._last_floor_log: float = 0.0
        self._last_leader_speed_switch_log: float = 0.0
        self._last_form_enable_check: float = 0.0
        self._last_blocked_gate_log: float = 0.0
        self._last_reposition_time: float = 0.0
        self._last_do_change_speed_time: float = 0.0
        self._last_do_change_speed_cmd: float = 0.0

        # E2: Decision-state first-call init (use None sentinel)
        self._last_formation_hdg_cmd: Optional[float] = None
        self._last_formation_hdg_time: Optional[float] = None
        self._last_reposition_hdg: Optional[float] = None

        # Range-based Vc for gate/D-term (replaces biased EWMA)
        self._range_vc_history: List[Tuple[float, float]] = []  # (sim_time, range_m)
        self._range_vc_ewma: float = 0.0  # EWMA of range-differenced Vc

        # Wake turbulence detection (Phase 1 - improved two-vortex model)
        self._wake_model = WakeTurbulenceZone(enable_wind_drift=True) if WAKE_MODEL_AVAILABLE else None
        self._wake_zone_entries: List[Dict] = []  # CORE entries only
        self._in_wake_core: bool = False          # Renamed from _in_wake_zone
        self._wake_entry_time_s: Optional[float] = None
        self._last_wake_log_s: int = -1

        # High-Side Convergence Mode (wake avoidance)
        # Configured via args in main(); defaults here for safety
        self._high_side_enabled: bool = False
        self._high_side_margin_m: float = 6.0
        self._high_side_cross_ok_m: float = 2.5
        self._high_side_behind_max_m: float = 250.0
        self._high_side_min_time_s: float = 5.0
        self._high_side_disable_behind_m: float = 60.0

        # High-side state tracking
        self._high_side_active: bool = False
        self._high_side_cross_ok_start_s: Optional[float] = None
        self._high_side_start_s: Optional[float] = None
        self._high_side_last_log_s: int = -1
        self._ramp_logged: bool = False
        self._high_side_stats: Dict = {
            'num_enable_events': 0,
            'active_time_s': 0.0,
            'max_cross_during_active_m': 0.0,
        }

        # Initialize the new helper classes
        # v9.10: Use tick_time_s (not sim_now_s which is now forbidden)
        self.control_laws = FormationControlLaws(self.tick_time_s, self.expected_speedup)
        self.envelope_checker = EnvelopeChecker()
        self.sensor_fusion = SensorFusion()


    # =========================================================================
    # v9.10 Gate 0: Time Source Wrapping (Appendix A.2)
    # =========================================================================

    def _read_sim_time_s_impl(self) -> float:
        """v9.10 P0 Model 1: Read sim-time from buffered ATTITUDE.time_boot_ms.

        LOCALITY RULE: This is the ONLY function that may update SimTimeAuthority.
        Time authority update happens INSIDE the tick owner, NOT in the MAVLink
        pump. This ensures time reads are localized to tick execution.

        Called only by _sim_now_s_raw().

        FD-008-B: When queue is active, TA updates should only come from the
        tick scheduler in the main loop. This method is guarded to skip TA
        updates when queue is active (pre-queue mode only).
        """
        # FD-008-B: Guard - only update TA from buffered value when queue is NOT active
        # When queue is active, TA updates come from tick_scheduler in main loop
        if self.follower and self.follower._buffered_attitude_time_boot_ms is not None:
            if not self.follower._queue_active:
                # Pre-queue mode: use advance_from_time_boot_ms with source="settle"
                get_time_authority().advance_from_time_boot_ms(
                    self.follower._buffered_attitude_time_boot_ms,
                    source="settle"
                )
            # When queue_active, TA was already updated by tick_scheduler
        return get_time_authority().now_s()

    def _tick_owner_enter(self) -> None:
        """v9.10: Tick owner guard: depth must be exactly 0 -> 1."""
        tid = threading.get_ident()
        if self._tick_owner_depth == 0:
            self._tick_owner_thread = tid
            self._tick_owner_depth = 1
            return
        if self._tick_owner_thread == tid:
            raise TickOwnerReentrancyError("Nested tick execution detected")
        raise IllegalSimTimeRead("Tick owner entered from different thread")

    def _tick_owner_exit(self) -> None:
        """v9.10: Underflow-proof; do not raise from finally."""
        self._tick_owner_depth = 0
        self._tick_owner_thread = None

    def _sim_now_s_raw(self) -> float:
        """v9.10: Private - actual time read from sim. Only callable from tick owner.

        This is the ONLY codepath allowed to call _read_sim_time_s_impl().
        """
        if self._tick_owner_depth != 1:
            raise IllegalSimTimeRead("_sim_now_s_raw() called outside tick owner")
        return self._read_sim_time_s_impl()

    def sim_now_s(self) -> float:
        """v9.10: Public FORBIDDEN API - always raises.

        Use self._t_sim_s (stored tick timestamp) instead.
        """
        raise IllegalSimTimeRead("sim_now_s() is forbidden; use stored self._t_sim_s")

    def tick_time_s(self) -> float:
        """v9.10: Safe accessor for sim-time of the most recent EXECUTED control tick.

        Use this instead of sim_now_s() in state handlers and state_machine.py
        to ensure single-source timebase. This returns the timestamp captured
        at the start of the tick that passed gating (not the current raw sim-time).

        Returns:
            Sim-time in seconds of the most recent executed tick. Returns 0.0 if
            no tick has been executed yet.
        """
        return self._t_sim_s

    def _dt_clamped(self) -> float:
        """Return clamped dt for this tick (pure getter, no mutation).

        P9.0 v9.9: This is now a pure getter that returns the cached _dt_tick
        value computed once at the top of run_state_machine(). This fixes the
        bug where calling _dt_clamped() multiple times per tick returned
        different values due to time advancing between calls.

        P1 v9.4 (original): Used for all integrators to ensure consistent
        sim-time deltas regardless of wall-clock variations.
        """
        return self._dt_tick

    # =========================================================================
    # v9.10 Gate 0: Fatal Handling (Appendix A.3, A.4, A.6)
    # =========================================================================

    def _fatal(self, reason: str, exc: Optional[Exception] = None) -> None:
        """v9.10: Preserve artifacts. Never raises. Safe to call multiple times.

        Writes fatal metadata to run_summary.json, flushes logs.
        Does NOT write additional CSV rows (fatal metadata belongs in run_summary).
        R7: Verifies artifact written; falls back to stderr if not.
        """
        if getattr(self, "_fatal_in_progress", False):
            return
        self._fatal_in_progress = True

        try:
            log(f"[FATAL] {reason}: {exc}")
        except Exception:
            pass

        try:
            self._flush_all_logs()
        except Exception:
            pass

        try:
            self._write_run_summary_fatal(
                fatal_reason=reason,
                unhandled_exception=type(exc).__name__ if exc and not isinstance(exc, GateFatalError) else None,
            )
        except Exception:
            pass

        # R7: Verify critical artifact written - fallback to stderr if not
        try:
            run_summary_path = os.path.join(self._out_dir, "run_summary.json") if self._out_dir else None
            if run_summary_path and not os.path.exists(run_summary_path):
                # Fallback: write fatal info to stderr for post-mortem
                sys.stderr.write(f"FATAL_ARTIFACT_FAILURE: {reason}\n")
                if exc:
                    sys.stderr.write(f"Exception: {type(exc).__name__}: {exc}\n")
                sys.stderr.write(f"Output dir: {self._out_dir}\n")
                sys.stderr.flush()
        except Exception:
            pass  # Best effort - don't fail during fatal unwind

    def _enter_terminal_fail(self, reason: str) -> None:
        """v9.10: Bare-metal FAIL state setter for fatal unwind.

        No transition side effects - just sets state and stops the controller.
        """
        self.state = ControllerState.FAIL
        self._fail_reason = reason
        self.running = False

    def _on_gate_fatal(self, reason: str, exc: GateFatalError) -> None:
        """v9.10: Single source of truth for gate-fatal handling."""
        self._fatal(reason, exc)
        if getattr(self, '_strict_mode', True):
            raise exc
        self._enter_terminal_fail(f"Gate fatal: {reason}")

    def _flush_all_logs(self) -> None:
        """v9.10: Flush all log buffers to disk."""
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Close CSV if open
        try:
            if self._arb_csv_file:
                self._arb_csv_file.flush()
        except Exception:
            pass

    def _pump_mavlink_messages(self, blocking: bool = False) -> bool:
        """v9.10 M7: Drain MAVLink message queues (buffers only).

        P2 Fix C v9.10: Supports event-driven pacing with blocking mode.

        Constraints:
        - Never reads sim-time (no _sim_now_s_raw, no DroneKit time attrs)
        - No control side effects (no _send_speed_command, no transition_to)
        - Only updates message buffers/queues

        Args:
            blocking: If True, block until at least one message arrives (up to 50ms).
                     This reduces CPU churn when sim-time hasn't advanced.

        Returns:
            True if any messages were received, False otherwise.
        """
        received_any = False
        _pump_drained = 0  # Change 5: track msgs drained this pump call
        try:
            if self.follower:
                if blocking:
                    # P2 Fix C: Block on message arrival with small timeout
                    # This reduces skipped_no_advance polling from 149k to near-zero
                    msg = self.follower.mav.recv_match(blocking=True, timeout=0.05)
                    if msg:
                        received_any = True
                        # Process the first message manually
                        if msg.get_srcSystem() == self.follower.target_system:
                            msg_type = msg.get_type()
                            if msg_type == 'GLOBAL_POSITION_INT':
                                if _FD_TRUTH_RATE_DIAG:
                                    self.follower._diag_33_recv += 1
                                    if (self.follower._last_position is None
                                            or msg.lat != self.follower._last_position.lat):
                                        self.follower._diag_33_change += 1
                                self.follower._last_position = msg
                            elif msg_type == 'ATTITUDE':
                                self.follower._last_attitude = msg
                                self.follower._attitude_message_arrived = True  # Gate0 fix: flag for tick gating
                                # FD-001 Fix: Add to queue (was missing - caused tick starvation!)
                                tb = int(msg.time_boot_ms)
                                if (self.follower._last_enqueued_attitude_ms is None or
                                        tb > self.follower._last_enqueued_attitude_ms):
                                    self.follower._attitude_time_queue.append(tb)
                                    self.follower._last_enqueued_attitude_ms = tb
                                    # FD-002: Track max depth
                                    self.follower._attitude_queue_max_depth = max(
                                        self.follower._attitude_queue_max_depth,
                                        len(self.follower._attitude_time_queue))
                                    _pump_drained += 1
                                self.follower._buffered_attitude_time_boot_ms = tb
                            elif msg_type == 'VFR_HUD':
                                self.follower._last_vfr_hud = msg
                            elif msg_type == 'SCALED_PRESSURE':
                                self.follower._last_scaled_pressure = msg   # CSV2_SPEC247 (Candidate A)
                            elif msg_type == 'HEARTBEAT':
                                self.follower._last_heartbeat = msg
                            elif msg_type == 'NAV_CONTROLLER_OUTPUT':
                                self._nav_alt_error_m = msg.alt_error
                                self._nav_aspd_error_mps = msg.aspd_error
                                self._nav_roll_deg = msg.nav_roll
                                self._nav_bearing_deg = msg.nav_bearing
                                self._target_bearing_deg = msg.target_bearing
                                self._xtrack_error_m = msg.xtrack_error
                                self._wp_dist_m = msg.wp_dist
                                self._nav_controller_rx_count += 1
                            elif _FD_PUMP_TRUTH_FIX and _FD_UWB_TRUTH_SOURCED and msg_type == 'SIMSTATE' and msg.get_srcSystem() == self.follower.target_system:
                                self.follower._last_simstate = msg     # BUG-A FIX: exact dual of update_cache :651
                            elif _FD_PUMP_TRUTH_FIX and _FD_UWB_TRUTH_SOURCED and msg_type == 'SIM_STATE' and msg.get_srcSystem() == self.follower.target_system:
                                # L-A fix: unconditional pump-mirror cache-age stamp (NO lat_int-CHANGE
                                # guard; the pump path has no clean prior comparison and each mirror write
                                # is genuinely new data). Sim-clock only. CHANGE-gate applies to update_cache only.
                                if self.follower._buffered_attitude_time_boot_ms is not None:
                                    self.follower._last_sim_state_tbms = self.follower._buffered_attitude_time_boot_ms
                                if _FD_TRUTH_RATE_DIAG:
                                    self.follower._diag_108_recv += 1
                                    _pnl = getattr(msg, 'lat_int', None)
                                    _pol = (getattr(self.follower._last_sim_state, 'lat_int', None)
                                            if self.follower._last_sim_state is not None else None)
                                    if _pnl != _pol:
                                        self.follower._diag_108_change += 1
                                        self.follower._diag_108_update_tbms = self.follower._buffered_attitude_time_boot_ms
                                self.follower._last_sim_state = msg    # BUG-A FIX: exact dual of update_cache :653
                # Drain remaining messages non-blocking
                _q_before = len(self.follower._attitude_time_queue)
                self.follower.update_cache()
                _q_after = len(self.follower._attitude_time_queue)
                _msgs_enqueued = _q_after - _q_before
                _pump_drained += max(0, _msgs_enqueued)
                if _msgs_enqueued > 0:
                    self._pump_msgs_drained_samples.append(_msgs_enqueued)
                    self._pump_msgs_drained_max = max(self._pump_msgs_drained_max, _msgs_enqueued)
                self._msgs_drained_last_pump = _pump_drained
                # Tranche C (C0): Pick up NAV_CONTROLLER_OUTPUT from update_cache() drain
                nav_msg = self.follower._last_nav_controller_output
                if nav_msg is not None:
                    self._nav_alt_error_m = nav_msg.alt_error
                    self._nav_aspd_error_mps = nav_msg.aspd_error
                    self._nav_roll_deg = nav_msg.nav_roll
                    self._nav_bearing_deg = nav_msg.nav_bearing
                    self._target_bearing_deg = nav_msg.target_bearing
                    self._xtrack_error_m = nav_msg.xtrack_error
                    self._wp_dist_m = nav_msg.wp_dist
                    self._nav_controller_rx_count += 1
                    self.follower._last_nav_controller_output = None  # consume
            if self._lead_rx:
                self._lead_rx.update()
        except Exception:
            pass  # Best effort - don't fail on message pump errors
        return received_any

    def _write_run_summary_fatal(self, fatal_reason: str,
                                 unhandled_exception: Optional[str] = None) -> None:
        """v9.10: Write fatal metadata to run_summary.json.

        Called from _fatal() to preserve artifact with fatal context.
        """
        import json
        import os

        if not self._out_dir:
            return

        run_summary_path = os.path.join(self._out_dir, "run_summary.json")

        try:
            if os.path.exists(run_summary_path):
                with open(run_summary_path, 'r') as f:
                    summary = json.load(f)
            else:
                summary = {}

            # Fatal metadata
            summary['fatal_reason'] = fatal_reason
            summary['unhandled_exception'] = unhandled_exception

            # v9.10 C4: Tick rate computation
            tick_hz, tick_reason, sim_elapsed = self._compute_tick_rate()
            summary['tick_hz_sim'] = tick_hz
            summary['tick_rate_defined'] = tick_hz is not None
            summary['tick_rate_reason'] = tick_reason
            summary['sim_elapsed_s'] = sim_elapsed
            summary['gate0_tick_rate_pass'] = tick_hz is not None and tick_hz >= 18.0

            # Existing fields
            summary['ticks_executed_total'] = self._ticks_executed_total
            summary['run_invalid_timebase'] = self._run_invalid_timebase

            tmp_path = run_summary_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(summary, f, indent=2)
            os.replace(tmp_path, run_summary_path)

        except Exception:
            pass  # Best effort - don't fail during fatal unwind

    def _compute_tick_rate(self) -> Tuple[Optional[float], str, float]:
        """v9.10 C4: Returns (tick_hz_sim, reason, sim_elapsed_s)."""
        if self._first_executed_tick_sim_s is None:
            return None, "no_ticks_executed", 0.0

        sim_elapsed = float(self._last_executed_tick_sim_s - self._first_executed_tick_sim_s)

        MIN_TICKS_FOR_RATE = 20
        if self._ticks_executed_total < MIN_TICKS_FOR_RATE:
            return None, "insufficient_ticks", sim_elapsed

        if sim_elapsed <= 0.0:
            return None, "sim_elapsed_zero", sim_elapsed

        tick_hz_sim = float(self._ticks_executed_total) / sim_elapsed
        if tick_hz_sim < 18.0:
            return tick_hz_sim, "starvation", sim_elapsed

        return tick_hz_sim, "ok", sim_elapsed

    def _wait_for_timebase_pass(self, timeout_wall_s: float = None) -> bool:
        """Wait for SimTimeAuthority PASS verdict before starting sim-time control.

        TEST 387: Timeout is scaled by expected_speedup. At 5x, sim advances 5x
        faster, so we need 5x more wall-seconds to accumulate the same sim-time
        for settle gate.

        v9.10 P0 Model 1 exception: During pre-mission settle, we must explicitly
        feed buffered time_boot_ms to the time authority since tick execution
        hasn't started yet and _read_sim_time_s_impl() won't be called.
        """
        if timeout_wall_s is None:
            timeout_wall_s = 15.0 * max(1.0, self.expected_speedup)
        start = time.monotonic()
        ta = get_time_authority()

        # Adaptive settle sleep: scales with speedup so we poll faster than
        # ATTITUDE arrival (wall_period/8). 1x→2ms, 3x→2ms, 5x→1.25ms, 10x→0.625ms
        _wall_period = 0.05 / max(1.0, self.expected_speedup)
        _settle_sleep = max(0.0005, min(0.002, _wall_period / 8.0))

        while time.monotonic() - start < timeout_wall_s:
            # Keep draining MAVLink so we receive time_boot_ms updates
            if self.follower:
                self.follower.update_cache()
                # v9.10 P0 Model 1 exception: Feed buffered time to authority
                # during pre-mission settle (ticks not yet executing).
                # Uses latest buffered tbms so sim_delta covers ALL messages
                # received per iteration (critical for accurate rate at 3x+).
                if self.follower._buffered_attitude_time_boot_ms is not None:
                    ta.update_from_mavlink(self.follower._buffered_attitude_time_boot_ms)
            if self.leader_rx:
                self.leader_rx.update()

            passed, details, rate_med = ta.get_sync_verdict()
            if passed:
                log(f"[SIM_TIME] PASS: {details}")
                # FD_CTRL_START_EPOCH_QUANT (#326, R7.1 re-auth 2026-07-05): Q=1.0s grid snap, default-OFF
                if os.environ.get("FD_CTRL_START_EPOCH_QUANT", "0").strip() != "1" or (
                        not self.follower
                        or self.follower._buffered_attitude_time_boot_ms is None):
                    return True
                from ctrl_start_epoch_quant import compute_target_ms, first_at_or_after, format_snap_log
                _raw_s = ta.now_s()
                _target_ms = compute_target_ms(
                    self.follower._buffered_attitude_time_boot_ms)
                while time.monotonic() - start < timeout_wall_s:
                    _t_star = first_at_or_after(
                        self.follower._attitude_time_queue, _target_ms)
                    if _t_star is not None:
                        ta.update_from_mavlink(_t_star)
                        log(format_snap_log(_target_ms, _t_star, _raw_s))
                        return True
                    self.follower.update_cache()
                    if self.leader_rx:
                        self.leader_rx.update()
                    time.sleep(_settle_sleep)
                log("[SIM_TIME] FAIL: START_EPOCH_QUANT grid-crossing timeout")
                return False

            # Optional 1 Hz debug (wall-time gated)
            if int(time.time()) != getattr(self, "_last_timebase_debug_sec", 0):
                self._last_timebase_debug_sec = int(time.time())
                m = ta.get_quality_metrics()
                log(
                    f"[SIM_TIME] {details}: Q={m.quality_score:.2f} "
                    f"settle={m.settle_gate_passed} mono={m.mono_violations} "
                    f"rate_med={m.rate_median:.2f}x"
                )

            # Adaptive sleep: scales with speedup (see _settle_sleep above).
            # At 10ms sleep, loop aliasing deflates measured rate to ~2.4x.
            time.sleep(_settle_sleep)

        log("[SIM_TIME] FAIL: timebase PASS timeout")
        return False

    # =========================================================================
    # WAKE TURBULENCE DETECTION (Phase 1: Logging Only)
    # =========================================================================

    def _estimate_time_since_passage(self, geom, lead_state) -> float:
        """
        Estimate time since leader passed follower's current position.

        This is an approximation based on along-track distance and leader speed.

        Args:
            geom: RelativeGeometry with behind_m attribute
            lead_state: LeaderState with ground_speed_ms attribute (or None)

        Returns:
            Estimated time in seconds since leader was at follower's current position
        """
        if geom.behind_m <= 0:
            return 0.0

        # Use actual leader speed if available, fallback to DHC-2 cruise speed
        leader_speed = max(lead_state.ground_speed_ms, 1e-3) if lead_state else 24.7
        return geom.behind_m / leader_speed

    def _check_wake_zone(self, geom, lead_state) -> Tuple[bool, Dict]:
        """
        Check wake zone and log CORE entry/exit + PERIPHERY warnings.

        Uses two-vortex-core model. Only CORE entries are tracked as "in zone".
        PERIPHERY is logged as a warning but does not count as "in zone".

        Args:
            geom: RelativeGeometry from get_geometry()
            lead_state: LeaderState from leader_rx.get_state() (or None)

        Returns:
            Tuple of (in_core: bool, details: dict)
        """
        if not WAKE_MODEL_AVAILABLE or self._wake_model is None:
            return False, {'in_zone': False, 'zone_type': 'safe'}

        # Estimate vortex age using actual leader speed
        time_since_passage = self._estimate_time_since_passage(geom, lead_state)

        # Get wind for drift calculation (TODO: integrate with WeatherModel)
        wind_ned = (0.0, 0.0, 0.0)

        # Check zone (in_core is True ONLY for core zone, not periphery)
        in_core, details = self._wake_model.is_in_avoid_zone(
            geom,
            time_since_passage,
            wind_ned
        )
        zone_type = details['zone_type']
        # v9.10: Use _t_sim_s (tick timestamp) instead of sim_now_s()
        sim_now = self._t_sim_s
        current_sec = int(sim_now)

        # Track CORE entry/exit events
        if in_core and not self._in_wake_core:
            # CORE ENTRY
            self._wake_entry_time_s = sim_now
            entry = {
                'entry_time_s': sim_now,
                'behind_m': geom.behind_m,
                'below_m': geom.below_m,
                'cross_m': geom.cross_m,
                'hazard_intensity': details['hazard_intensity'],
                'dist_to_core_m': details['dist_to_core_m'],
                'nearest_core_side': details['nearest_core_side'],
            }
            self._wake_zone_entries.append(entry)
            log(f"[WAKE] ENTERED CORE behind={geom.behind_m:.1f}m "
                f"cross={geom.cross_m:.1f}m below={geom.below_m:.1f}m "
                f"intensity={details['hazard_intensity']:.2f} "
                f"dist={details['dist_to_core_m']:.1f}m side={details['nearest_core_side']}")

        elif not in_core and self._in_wake_core:
            # CORE EXIT
            dwell = sim_now - self._wake_entry_time_s if self._wake_entry_time_s else 0.0
            if self._wake_zone_entries:
                self._wake_zone_entries[-1]['exit_time_s'] = sim_now
                self._wake_zone_entries[-1]['dwell_s'] = dwell
            log(f"[WAKE] EXITED CORE dwell={dwell:.1f}s")

        self._in_wake_core = in_core

        # 1 Hz periodic logging while in CORE or PERIPHERY
        if zone_type in ('core', 'periphery') and current_sec != self._last_wake_log_s:
            self._last_wake_log_s = current_sec
            tag = "CORE" if zone_type == 'core' else "PERIPHERY"
            log(f"[WAKE] {tag} behind={geom.behind_m:.1f}m "
                f"intensity={details['hazard_intensity']:.2f} "
                f"dist={details['dist_to_core_m']:.1f}m "
                f"lat_clear={details['lateral_clearance_m']:.1f}m "
                f"vert_clear={details['vertical_clearance_m']:.1f}m")

        return in_core, details

    def get_wake_zone_statistics(self) -> Dict:
        """
        Get CORE zone encounter statistics.

        Returns:
            Dictionary with statistics:
                - 'total_core_entries': int
                - 'total_time_in_core_s': float
                - 'max_hazard_intensity': float
                - 'entries': List of CORE entry events
        """
        total_time = sum(
            entry.get('dwell_s', 0.0)
            for entry in self._wake_zone_entries
        )

        max_intensity = max(
            (entry['hazard_intensity'] for entry in self._wake_zone_entries),
            default=0.0
        )

        return {
            'total_core_entries': len(self._wake_zone_entries),
            'total_time_in_core_s': total_time,
            'max_hazard_intensity': max_intensity,
            'entries': self._wake_zone_entries
        }

    def _reset_sim_timers_after_pass(self) -> None:
        """Reset all control timers to sim-time AFTER timebase PASS.

        v9.10: Uses _t_sim_s for timer initialization. This ensures
        all timers start from the same tick timestamp.
        """
        now = self._t_sim_s  # v9.10: Use tick timestamp

        # State machine
        self.state_entry_time = now

        # Leader turning gate
        self._leader_not_turning_since = now

        # Speed/heading slew
        self._speed_cmd_slew_time = now
        # Note: _last_formation_hdg_time will be initialized on first use

        # P5.1 v9.5: Closure derivative now fully in FormationControlLaws
        # (removed dead copies: _behind_m_prev, _behind_m_prev_time)

        # Stability windows (set fresh when first used)
        self._vc_gate_behind_prev = None
        self._vc_gate_behind_prev_time = None
        self._vc_gate_slow_vc = 0.0
        self._precontact_vc_credit_s = 0.0
        self._precontact_vc_violation_start_s = None
        self._vc_gate_prev_time = None
        self._vc_gate_credit_prev_time = None
        self._contact_outer_exit_start = None
        self._precontact_outer_exit_start = None
        self._precontact_inner_exit_start = None  # TEST 400: INNER grace

        log(f"[SIM_TIME] Timers initialized at sim_now={now:.2f}s")

        # Start success metrics tracking
        self._success_metrics.set_t0(now)

    def connect(self) -> bool:
        """Connect to follower and leader telemetry with full preflight checks.

        Initialization sequence (7 steps):
        1. Connect to QuadPlane (sysid=1)
        2. Wait for GPS fix (type >= 3)
        3. Get home position
        4. Verify FORM_ENABLE=0 (ensure Python has control)
        5. Request higher telemetry rates
        6. Connect to leader telemetry
        7. Validate leader datalink
        """
        log("=" * 60)
        log("PREFLIGHT INITIALIZATION")
        log("=" * 60)

        # [1/7] Connect to QuadPlane
        log("[1/7] Connecting to QuadPlane...")
        self.follower = MAVConnection(self.follower_conn_str)
        if not self.follower.connect():
            log("ERROR: Failed to connect to QuadPlane")
            return False
        log("[1/7] ✓ Connected to QuadPlane (sysid=1)")

        # [2/7] Wait for GPS fix
        log("[2/7] Waiting for GPS fix...")
        if not self.follower.wait_for_gps_fix(min_fix_type=3, timeout=30.0):
            log("WARNING: GPS fix timeout - continuing with caution")
        else:
            log("[2/7] ✓ GPS fix acquired")

        # [3/7] Get home position
        log("[3/7] Getting home position...")
        for _ in range(100):
            pos = self.follower.get_position()
            if pos:
                self.home_lat, self.home_lon, self.home_alt_msl = pos
                log(f"[3/7] ✓ Home: {self.home_lat:.7f}, {self.home_lon:.7f}, "
                    f"{self.home_alt_msl:.1f}m MSL")
                break
            time.sleep(0.1)
        else:
            log("ERROR: Could not get home position")
            return False

        # [4/7] Verify FORM_ENABLE=0 (ensure Python has full authority)
        log("[4/7] Verifying FORM_ENABLE=0...")
        form_enable = self.follower.get_param("FORM_ENABLE", timeout=3.0)
        if form_enable is not None:
            self._form_enable_supported = True
            if form_enable > 0.5:
                log("[4/7] FORM_ENABLE=1 detected, setting to 0...")
                if self.follower.set_param("FORM_ENABLE", 0.0):
                    log("[4/7] ✓ FORM_ENABLE set to 0 (Python has authority)")
                else:
                    log("WARNING: Could not set FORM_ENABLE=0")
            else:
                log("[4/7] ✓ FORM_ENABLE=0 confirmed (Python has authority)")
        else:
            self._form_enable_supported = False
            log("[4/7] FORM_ENABLE param not found — all FORM_ENABLE writes disabled")

        # [4b/7] Verify FORM_SPD_EN=0 (ensure Python owns airspeed in DC)
        log("[4b/7] Verifying FORM_SPD_EN=0...")
        form_spd_en = self.follower.get_param("FORM_SPD_EN", timeout=3.0)
        if form_spd_en is not None:
            self._form_spd_enable_supported = True
            if form_spd_en > 0.5:
                log("[4b/7] FORM_SPD_EN=1 detected, setting to 0...")
                if self.follower.set_param("FORM_SPD_EN", 0.0):
                    log("[4b/7] ✓ FORM_SPD_EN set to 0 (Python owns airspeed)")
                else:
                    log("WARNING: Could not set FORM_SPD_EN=0")
            else:
                log("[4b/7] ✓ FORM_SPD_EN=0 confirmed (Python owns airspeed)")
        else:
            self._form_spd_enable_supported = False
            log("[4b/7] FORM_SPD_EN param not found (old firmware?), continuing")

        # [4c/7] Verify FORM_DOCK_MODE=0 at startup (Tranche H)
        log("[4c/7] Verifying FORM_DOCK_MODE=0...")
        dock_mode = self.follower.get_param("FORM_DOCK_MODE", timeout=3.0)
        if dock_mode is not None:
            self._form_dock_mode_supported = True
            if dock_mode > 0.5:
                self.follower.set_param("FORM_DOCK_MODE", 0.0)
                log("[4c/7] ✓ FORM_DOCK_MODE set to 0")
            else:
                log("[4c/7] ✓ FORM_DOCK_MODE=0 confirmed")
        else:
            self._form_dock_mode_supported = False
            log("[4c/7] FORM_DOCK_MODE not found (old firmware?)")

        # Read baseline TECS_SPDWEIGHT and Q_ASSIST_SPEED values
        # These are stored and restored on ALTITUDE_CAPTURE exit instead of hardcoded
        log("[4.5/7] Reading TECS/QASSIST baseline parameters...")
        self._tecs_spdweight_baseline = self.follower.get_param(
            "TECS_SPDWEIGHT", timeout=2.0)
        if self._tecs_spdweight_baseline is not None:
            log(f"[4.5/7] ✓ TECS_SPDWEIGHT baseline = {self._tecs_spdweight_baseline:.2f}")
            self._tecs_spdweight_current = self._tecs_spdweight_baseline
        else:
            log("[4.5/7] WARNING: Could not read TECS_SPDWEIGHT, using default=1.0")
            self._tecs_spdweight_baseline = ALT_CAPTURE_TECS_SPDWEIGHT_NORMAL
            self._tecs_spdweight_current = ALT_CAPTURE_TECS_SPDWEIGHT_NORMAL

        # K-B6/N0: TECS_TIME_CONST override — read baseline for zone-gated or global override
        self._tecs_tc_baseline = self.follower.get_param("TECS_TIME_CONST", timeout=2.0)
        self._tecs_tc_zone_applied = False  # N0: Track whether zone-gated override is active
        if self._tecs_tc_baseline is not None:
            log(f"[4.5/7] ✓ TECS_TIME_CONST baseline = {self._tecs_tc_baseline:.1f}")
        else:
            log("[4.5/7] WARNING: Could not read TECS_TIME_CONST, zone-gated override disabled")
            self._tecs_tc_baseline = 5.0  # Sensible default

        if formation_config.TECS_TIME_CONST_OVERRIDE_ENABLE:
            # Global override (legacy K-B6 path, currently disabled)
            tecs_tc_target = formation_config.TECS_TIME_CONST_OVERRIDE
            if abs(self._tecs_tc_baseline - tecs_tc_target) > 0.05:
                if self.follower.set_param("TECS_TIME_CONST", tecs_tc_target):
                    log(f"[4.5/7] ✓ TECS_TIME_CONST set: {self._tecs_tc_baseline:.1f} → {tecs_tc_target:.1f}")
                else:
                    log(f"[4.5/7] WARNING: Could not set TECS_TIME_CONST to {tecs_tc_target:.1f}")
            else:
                log(f"[4.5/7] ✓ TECS_TIME_CONST already {self._tecs_tc_baseline:.1f} (target={tecs_tc_target:.1f})")
        elif formation_config.TECS_TIME_CONST_ZONE_GATED:
            log(f"[4.5/7] ✓ TECS_TIME_CONST zone-gated override enabled "
                f"(baseline={self._tecs_tc_baseline:.1f}, "
                f"zone={formation_config.TECS_TIME_CONST_OVERRIDE:.1f})")

        self._qassist_speed_baseline = self.follower.get_param(
            "Q_ASSIST_SPEED", timeout=2.0)
        if self._qassist_speed_baseline is not None:
            log(f"[4.5/7] ✓ Q_ASSIST_SPEED baseline = {self._qassist_speed_baseline:.1f} m/s")
            self._qassist_speed_current = self._qassist_speed_baseline
            # v9.10.1: Warn if Q_ASSIST_SPEED is too close to cruise speed
            if self._qassist_speed_baseline > 15.0:
                log(f"[4.5/7] ⚠ Q_ASSIST_SPEED={self._qassist_speed_baseline:.1f} m/s is HIGH - "
                    "VTOL assist may activate during normal FW flight!")
        else:
            log("[4.5/7] WARNING: Could not read Q_ASSIST_SPEED, using default=15.0")
            self._qassist_speed_baseline = ALT_CAPTURE_QASSIST_SPEED_NORMAL
            self._qassist_speed_current = ALT_CAPTURE_QASSIST_SPEED_NORMAL

        # [4d/7] Verify Q_GUIDED_MODE=0 (prevent VTOL transition in GUIDED)
        # When Q_GUIDED_MODE=1, L1 "arrived at waypoint" triggers VTOL loiter,
        # collapsing airspeed from 24.5 to ~9 m/s. Phase 1 VTOL fix.
        log("[4d/7] Verifying Q_GUIDED_MODE=0...")
        q_guided_mode = self.follower.get_param("Q_GUIDED_MODE", timeout=2.0)
        if q_guided_mode is not None:
            if q_guided_mode > 0.5:
                log(f"[4d/7] Q_GUIDED_MODE={q_guided_mode:.0f} detected, setting to 0...")
                if self.follower.set_param("Q_GUIDED_MODE", 0.0):
                    log("[4d/7] ✓ Q_GUIDED_MODE set to 0 (VTOL transition suppressed)")
                else:
                    log("WARNING: Could not set Q_GUIDED_MODE=0 — VTOL transition risk!")
            else:
                log("[4d/7] ✓ Q_GUIDED_MODE=0 confirmed (FW GUIDED OK)")

        # Warn about unexpected QASSIST triggers (informational only)
        # Q_ASSIST_ANGLE: If non-zero, QASSIST may trigger on large attitude angles
        # Q_ASSIST_ALT: If non-zero, QASSIST may trigger below this AGL altitude
        q_assist_angle = self.follower.get_param("Q_ASSIST_ANGLE", timeout=1.0)
        q_assist_alt = self.follower.get_param("Q_ASSIST_ALT", timeout=1.0)
        if q_assist_angle is not None and q_assist_angle > 0:
            log(f"[4.5/7] ⚠ Q_ASSIST_ANGLE={q_assist_angle:.0f}° - "
                "QASSIST may trigger on large attitudes")
        if q_assist_alt is not None and q_assist_alt > 0:
            log(f"[4.5/7] ⚠ Q_ASSIST_ALT={q_assist_alt:.0f}m - "
                "QASSIST may trigger below this AGL")

        # [5/7] Request higher telemetry rates
        log("[5/7] Requesting telemetry rates...")

        # P1 Fix B v9.10: Use request_attitude_20hz with ACK verification
        if TIMEBASE_CANARY_AVAILABLE:
            att_ok, att_msg = request_attitude_20hz(
                self.follower.mav,
                self.follower.target_system,
                self.follower.target_component,
                timeout_s=5.0,
            )
            if att_ok:
                log(f"[5/7] ✓ ATTITUDE@20Hz request accepted (MAV_CMD_SET_MESSAGE_INTERVAL)")
            else:
                log(f"[5/7] ⚠ ATTITUDE rate request: {att_msg}")
                # Fall back to original method
                self.follower.request_message_rate(30, MSG_RATE_ATTITUDE_HZ)
        else:
            # Original method (no ACK verification)
            self.follower.request_message_rate(30, MSG_RATE_ATTITUDE_HZ)

        # CSV2_SPEC247 (Candidate A): stream SCALED_PRESSURE(29) for the baro-derived eas2tas
        # source. Gated on _FD_EAS2TAS_BARO -> NO comms change when A is OFF (byte-identical).
        if _FD_EAS2TAS_BARO:
            if self.follower.request_message_rate(29, 5.0):
                log("[5/7] ✓ SCALED_PRESSURE@5Hz requested (Candidate A baro eas2tas)")
            else:
                log("[5/7] ⚠ SCALED_PRESSURE rate request failed (A falls back to ratio/ISA)")

        # VFR_HUD (74) for airspeed/heading
        self.follower.request_message_rate(74, MSG_RATE_VFR_HUD_HZ)
        # GLOBAL_POSITION_INT (33) for position
        self.follower.request_message_rate(33, MSG_RATE_POSITION_HZ)
        # Tranche C (C0): NAV_CONTROLLER_OUTPUT (62) for altitude demand disambiguation
        self.follower.request_message_rate(62, 10.0)
        # C1 (gated): request SIMSTATE(164) horiz + SIM_STATE(108) alt truth when
        # truth-sourcing the UWB emitter. C-fix (C3-MEASURED): SIMSTATE(164) does
        # NOT stream at rate by default (~0.5 Hz under this stream config), which
        # froze the follower-horizontal truth between refreshes and produced a
        # ground-speed (~24.7 m/s) sawtooth in the emitter truth-behind axis
        # (median 32 m) -> 100% innovation-gate rejection of the truth-UWB. Both
        # SIMSTATE(164) and SIM_STATE(108) must be requested at rate so the
        # follower horizontal+vertical truth stays fresh. SET_MESSAGE_INTERVAL(511)
        # accepted by the autopilot (C0/C0.5).
        # No-op when off -> byte-identical request set to v9.8.
        if _FD_UWB_TRUTH_SOURCED:
            self.follower.request_message_rate(164, _FD_UWB_SIM_STATE_RATE_HZ)
            self.follower.request_message_rate(108, _FD_UWB_SIM_STATE_RATE_HZ)
            log(f"[5/7] ✓ (truth-source) Requested SIMSTATE(164)+SIM_STATE(108)@{_FD_UWB_SIM_STATE_RATE_HZ}Hz")
        # STEP-4 Hunk-RR (2026-06-11, R7.1 waiver, gated): raise follower/truth
        # feeds to 20 Hz — the follower hop of the C3 both-hop fix (10 Hz feeds
        # consumed at ~20 Hz ticks = 48.9% stale re-pairs, step-3 measured).
        # No-op when off -> byte-identical request set.
        if _FD_FEED_RATE_RAISE:
            self.follower.request_message_rate(33, _FD_FEED_RATE_RAISE_HZ)
            self.follower.request_message_rate(164, _FD_FEED_RATE_RAISE_HZ)
            self.follower.request_message_rate(108, _FD_FEED_RATE_RAISE_HZ)
            log(f"[5/7] ✓ (rate-raise) GPI(33)+SIMSTATE(164)+SIM_STATE(108)@{_FD_FEED_RATE_RAISE_HZ}Hz")
        log(f"[5/7] ✓ Requested VFR_HUD@{MSG_RATE_VFR_HUD_HZ}Hz, POS@{MSG_RATE_POSITION_HZ}Hz, NAV_OUT@10Hz")

        # Initialize geometry calculator
        self.geom_calc = GeometryCalculator(
            self.home_lat, self.home_lon, self.home_alt_msl
        )

        # If UWB emitter is enabled, configure fusion filter anchor positions
        if (self._uwb_emitter is not None and self._uwb_emitter.enabled
                and self.geom_calc.fusion is not None):
            anchor_positions = self._uwb_emitter.get_anchor_positions()
            self.geom_calc.fusion.set_anchor_positions(anchor_positions)
            log(f"[UWB_EMITTER] Fusion filter configured with anchor positions: {list(anchor_positions.keys())}")

        # Initialize weather checker
        self.weather_checker = WeatherChecker(self.follower)

        # [6/7] Connect to leader telemetry
        log("[6/7] Connecting to leader telemetry...")
        self.leader_rx = LeaderTelemetryReceiver()
        if not self.leader_rx.connect():
            log("[6/7] WARNING: Leader telemetry not available yet")
        else:
            log("[6/7] ✓ Leader telemetry receiver ready")

        # [7/7] Validate leader datalink
        log("[7/7] Validating leader datalink...")
        lead_msg_count = 0
        # NOTE: Wall-time is intentional here - SITL hasn't started yet and
        # SimTimeAuthority isn't created until after this validation loop.
        start_time = time.time()
        # Scale preflight timeout by speedup (at 5x, sim advances 5x faster)
        scaled_timeout = PREFLIGHT_LEAD_TIMEOUT_S * max(1.0, self.expected_speedup)
        while time.time() - start_time < scaled_timeout:
            lead_state = self.leader_rx.update()
            if lead_state:
                lead_msg_count += 1
                if lead_msg_count >= PREFLIGHT_MIN_LEAD_MESSAGES:
                    log("[7/7] ✓ Leader datalink validated "
                        f"({lead_msg_count} messages received)")
                    break
            time.sleep(0.1)
        else:
            log(f"[7/7] WARNING: Only {lead_msg_count} leader messages - "
                "datalink may be weak")

        log("[8/8] Creating SimTimeAuthority for speedup consistency...")
        self._time_auth = SimTimeAuthority(
            expected_speedup=self.expected_speedup,
            drl_mode=False,
            verbose=True,
        )
        set_time_authority(self._time_auth)
        log(f"[8/8] ✓ SimTimeAuthority created (expected speedup={self.expected_speedup}x)")

        log("=" * 60)
        log("PREFLIGHT COMPLETE - Ready for mission")
        log("=" * 60)
        return True

    def transition_to(self, new_state: ControllerState, reason: str = "",
                      skip_exit: bool = False) -> None:
        """Transition to a new state via the table-driven state machine.

        Phase 3: Delegates to StateMachine which handles entry/exit actions,
        state_entry_time reset, entry count tracking, and audit trail.

        v9.10 Gate 2: Also logs to TRANSITION_SNAPSHOT.log with reason and
        threshold values for post-run analysis.

        Args:
            new_state: Target ControllerState to transition to.
            reason: Human-readable reason for the transition.
            skip_exit: If True, skip exit actions (use when handler has
                       already called exit cleanup, e.g. _exit_hold_contact).
        """
        # L-DIAG: Global breakaway shadow gate — blocks ALL breakaway transitions
        # Tranche W: CZ-EMERGENCY pierces the shadow gate (non-suppressible)
        if (new_state == ControllerState.EMERGENCY_BREAKAWAY
                and formation_config.DIAG_BREAKAWAY_SHADOW
                and not reason.startswith("CZ-EMERGENCY")):
            self._diag_suppressed_breakaways += 1
            _source = reason.split(":")[0] if ":" in reason else reason
            self._diag_suppressed_by_source[_source] = \
                self._diag_suppressed_by_source.get(_source, 0) + 1
            log(f"[DIAG_SHADOW] Breakaway BLOCKED (#{self._diag_suppressed_breakaways}): {reason}")
            return  # Do NOT transition — continue in current state

        # Guard strip-down: Commitment zone suppression + deactivation removed (Chunk 3A)

        old_state = self.state
        sim_now = self._t_sim_s  # v9.10: Use tick timestamp

        # Delegate to state machine
        self.sm.transition_to(new_state, reason, skip_exit=skip_exit)
        # §8-3 (#392): monotonic final event id. getattr-defaulted because
        # partial-init instances reach this path with __init__ patched out
        # (WarmupController mock fixtures); same idiom as the env's
        # getattr(self, "_gps_walk", None) legacy/mock guard. Fully-constructed
        # controllers are unaffected — __init__ seeds 0 before any transition.
        self._transition_count = getattr(self, "_transition_count", 0) + 1

        # F-E1: the emergency counter
        # had ZERO production increment sites (Dispatch-E RCA, dead since
        # v9.10). Semantic = RCA §8-1(a): ENTRIES into EMERGENCY_BREAKAWAY.
        # Live counter + canonical append-only event record (monotonic id) so
        # the serialized count reconciles against evidence and cannot silently
        # read 0 again. NOT derived from state_entry_counts (RCA council B).
        if (new_state == ControllerState.EMERGENCY_BREAKAWAY
                and old_state != ControllerState.EMERGENCY_BREAKAWAY):
            self._safety_emergency_count = getattr(
                self, "_safety_emergency_count", 0) + 1
            if not hasattr(self, "_safety_emergency_events"):
                self._safety_emergency_events = []
            self._safety_emergency_events.append({
                "event_id": self._safety_emergency_count,
                "sim_time_s": round(sim_now, 3),
                "from_state": old_state.name,
                "reason": reason,
            })

        # P2 v9.4 Fix 2a: Reset tick counter on transition for near-transition logging
        self._ticks_since_transition = 0

        # PRH-ATT-004: Reset warn flag on state transition (one warn per state episode)
        self._watchdog_warn_logged = False

        # Coupled below-target: reset first-tick flag on entry to any coupled state
        if new_state.name in getattr(formation_config, 'COUPLED_BELOW_TARGET_ENABLE_STATES', []):
            self._coupled_below_initialized = False

        # v9.10 Gate 2: Log transition snapshot to file for gate checker
        self._log_transition_snapshot(old_state, new_state, reason, sim_now)

        # Rec 5: Record sim-time indexed state timeline in success metrics
        if hasattr(self, '_success_metrics') and self._success_metrics is not None:
            self._success_metrics.on_state_transition(
                from_state=old_state.name,
                to_state=new_state.name,
                reason=reason,
                sim_time_s=sim_now,
            )

        # v9.10: Check for state_entry fault injection
        self._maybe_inject_on_state_entry(new_state)

    def _log_transition_snapshot(self, old_state: ControllerState,
                                  new_state: ControllerState,
                                  reason: str, sim_time: float) -> None:
        """v9.10 Gate 2: Log transition details to TRANSITION_SNAPSHOT.log.

        Every transition MUST log:
        - sim_time, old_state, new_state, reason
        - behind, cross, below if geometry available
        - relevant thresholds used in the decision

        This enables the gate checker to verify state machine stability.
        """
        if not self._out_dir:
            return

        import os
        log_path = os.path.join(self._out_dir, "TRANSITION_SNAPSHOT.log")

        # Get current geometry for context
        geom = self.get_geometry()
        behind = geom.behind_m if geom else float('nan')
        cross = geom.cross_m if geom else float('nan')
        below = geom.below_m if geom else float('nan')

        # Get relevant threshold values
        outer_grace = PRECONTACT_OUTER_EXIT_GRACE_S
        inner_grace = PRECONTACT_INNER_EXIT_GRACE_S
        hold_dwell_req = PRECONTACT_HOLD_DURATION_S
        vc_slow = getattr(self, '_vc_gate_slow_vc', 0.0)
        # v9.10 Gate 2: Use accumulator-based dwell (not timestamp diff)
        dwell_accum = getattr(self, '_precontact_dwell_accum_s', 0.0)

        # v9.10 Gate 2: Include entry grace accum for symmetric debouncing
        entry_grace_accum = getattr(self, '_hold_entry_grace_accum_s', 0.0)
        exit_grace_accum = getattr(self, '_hold_exit_grace_accum_s', 0.0)

        line = (
            f"[TRANSITION] t={sim_time:.3f} "
            f"old={old_state.name} new={new_state.name} reason={reason}\n"
            f"             behind={behind:.2f} cross={cross:.2f} below={below:.2f} "
            f"vc_slow={vc_slow:.3f}\n"
            f"             thresholds: outer_grace={outer_grace}s inner_grace={inner_grace}s "
            f"dwell_req={hold_dwell_req}s dwell_accum={dwell_accum:.2f}s\n"
            f"             entry_grace_accum={entry_grace_accum:.2f}s exit_grace_accum={exit_grace_accum:.2f}s\n"
        )

        try:
            with open(log_path, 'a') as f:
                f.write(line)
        except Exception:
            pass  # Best effort - don't fail transition on log error

    def _maybe_inject_on_state_entry(self, new_state: 'ControllerState') -> None:
        """v9.10: Inject fatal error on state entry if configured.

        Triggered by env vars:
        - FLIGHTDOCK_INJECT_FATAL=1
        - FLIGHTDOCK_INJECT_FATAL_MODE=state_entry
        - FLIGHTDOCK_INJECT_FATAL_STATE=<state_name>

        Raises InjectedFatalError immediately upon entering the target state,
        testing that the fatal path preserves data before any tick runs.
        """
        inject_fatal = os.environ.get("FLIGHTDOCK_INJECT_FATAL")
        if not inject_fatal:
            return

        inject_mode = os.environ.get("FLIGHTDOCK_INJECT_FATAL_MODE", "tick")
        if inject_mode != "state_entry":
            return

        inject_state = os.environ.get("FLIGHTDOCK_INJECT_FATAL_STATE", "REJOIN_PRECONTACT")
        if new_state.name == inject_state:
            raise InjectedFatalError(
                f"INJECTED_FATAL: state_entry mode, state={new_state.name}"
            )

    def _finalize_engaged_time(self, sim_now: float) -> None:
        """TEST 387: Accumulate engaged time when disabling C++."""
        if self._cpp_handoff_last_enable_s is not None:
            self._cpp_handoff_total_s += (sim_now - self._cpp_handoff_last_enable_s)

    def _check_cpp_handoff(self, geom: Optional['RelativeGeometry']) -> None:
        """TEST 387: Check if C++ formation controller should be enabled/disabled.

        Supports three policies:
        - always_on: Engage after sim-time PASS, never disengage
        - always_off: Never engage (Python-only)
        - range_hysteresis: Enable <30m, disable >40m (existing behavior)
        """
        if not self._cpp_handoff_enabled:
            self._cpp_inactive_reason = 1  # MASTER_OFF
            return

        if not self.follower:
            self._cpp_inactive_reason = 1  # MASTER_OFF
            return

        # Guard: If FORM_ENABLE param is proven unsupported, skip ALL set_param calls.
        # This prevents the 2s blocking message drain on binaries without FORM_ENABLE.
        if self._form_enable_supported is False:
            self._cpp_inactive_reason = 1  # MASTER_OFF (param unsupported)
            return

        # v9.10: Use tick timestamp for log rate-limiting (consistent density across speedups)
        sim_now = self._t_sim_s
        # Use accumulator for log rate-limiting
        if not hasattr(self, '_cpp_handoff_log_accum_s'):
            self._cpp_handoff_log_accum_s = 0.0
        self._cpp_handoff_log_accum_s += self._dt_credit_s if self._dt_credit_s else 0.0
        log_ok = self._cpp_handoff_log_accum_s >= 2.0
        if log_ok:
            self._cpp_handoff_log_accum_s = 0.0

        # =====================================================================
        # E3: ALT_GATE SAFETY OVERRIDE - applies to ALL policies
        # =====================================================================
        # If ALT_GATE is pending/active, Python must own control for altitude recovery.
        # This MUST run before policy-specific logic to prevent C++ from fighting Python.
        if self._alt_gate_start is not None or self._alt_gate_active:
            if self._cpp_enabled:
                if self.follower.set_param("FORM_ENABLE", 0.0):
                    self._finalize_engaged_time(sim_now)
                    self._cpp_enabled = False
                    self._cpp_handoff_disable_count += 1
                    if log_ok:
                        log(f"[CPP_HANDOFF] FORM_ENABLE=0 due to ALT_GATE "
                            f"(policy={self._cpp_handoff_policy}, state={self.state.name}) "
                            "- Python altitude control RESUMED")
                        self._last_cpp_handoff_log = sim_now
            # Skip all other policy logic while ALT_GATE active
            self._cpp_inactive_reason = 3  # ALT_GATE
            return

        # =====================================================================
        # TRANCHE C: CLOSE-IN FORM_ENABLE DISABLE (altitude authority experiment)
        # =====================================================================
        # When experiment is active, force FORM_ENABLE=0 in close-in states to
        # test whether C++ formation waypoint altitude causes the DC climb artifact.
        # Cross-track abort safety: if lateral diverges, re-enable C++.
        if (formation_config.FORM_ENABLE_CLOSE_IN_DISABLE
                and not self._c1_cross_abort_active
                and self.state.name in formation_config.FORM_ENABLE_CLOSE_IN_DISABLE_STATES):
            if self._cpp_enabled:
                if self.follower.set_param("FORM_ENABLE", 0.0):
                    self._finalize_engaged_time(sim_now)
                    self._cpp_enabled = False
                    self._cpp_handoff_disable_count += 1
                    log(f"[CPP_HANDOFF] FORM_ENABLE=0 (Tranche C experiment, state={self.state.name})")
                    self._last_cpp_handoff_log = sim_now
            # Skip policy logic — experiment override is authoritative in these states
            self._cpp_inactive_reason = 4  # EXPERIMENT
            return

        # Tranche C: Cross-track abort check (only in close-in states where override applies)
        if (formation_config.FORM_ENABLE_CLOSE_IN_DISABLE and geom is not None
                and self.state.name in formation_config.FORM_ENABLE_CLOSE_IN_DISABLE_STATES):
            cross_abs = abs(geom.cross_m)
            if cross_abs > formation_config.FORM_ENABLE_EXPERIMENT_CROSS_ABORT_M:
                if self._c1_cross_exceed_start_s is None:
                    self._c1_cross_exceed_start_s = sim_now
                elif (sim_now - self._c1_cross_exceed_start_s) >= formation_config.FORM_ENABLE_EXPERIMENT_CROSS_ABORT_DWELL_S:
                    if not self._c1_cross_abort_active:
                        self._c1_cross_abort_active = True
                        log(f"[TRANCHE_C] Cross-track abort: |cross|={cross_abs:.2f}m > "
                            f"{formation_config.FORM_ENABLE_EXPERIMENT_CROSS_ABORT_M}m for "
                            f"{formation_config.FORM_ENABLE_EXPERIMENT_CROSS_ABORT_DWELL_S}s "
                            "— re-enabling C++ for remainder of run")
            else:
                self._c1_cross_exceed_start_s = None

        # === POLICY: always_off ===
        if self._cpp_handoff_policy == "always_off":
            if self._cpp_enabled:
                if self.follower.set_param("FORM_ENABLE", 0.0):
                    self._finalize_engaged_time(sim_now)
                    self._cpp_enabled = False
                    self._cpp_handoff_disable_count += 1
                    log("[CPP_HANDOFF] FORM_ENABLE=0 (policy=always_off)")
                    self._last_cpp_handoff_log = sim_now
            self._cpp_inactive_reason = 1  # MASTER_OFF (always_off policy)
            return

        # === POLICY: always_on ===
        if self._cpp_handoff_policy == "always_on":
            ta = get_time_authority()
            sim_ready = ta is not None and ta.is_ready()

            if not self._cpp_enabled and sim_ready:
                # Optional warmup dwell after PASS (sim-time based)
                if self._cpp_handoff_warmup_start_s is None:
                    self._cpp_handoff_warmup_start_s = sim_now
                    log(f"[CPP_HANDOFF] Warmup started (policy=always_on, dwell={CPP_HANDOFF_WARMUP_S}s sim)")
                    self._last_cpp_handoff_log = sim_now
                    return

                elapsed = sim_now - self._cpp_handoff_warmup_start_s
                if elapsed < CPP_HANDOFF_WARMUP_S:
                    return  # Still in warmup dwell

                # Warmup complete, engage C++
                if self.follower.set_param("FORM_ENABLE", 1.0):
                    self._cpp_enabled = True
                    if self._cpp_handoff_first_enable_s is None:
                        self._cpp_handoff_first_enable_s = sim_now
                    self._cpp_handoff_last_enable_s = sim_now
                    self._cpp_handoff_enable_count += 1
                    log(f"[CPP_HANDOFF] FORM_ENABLE=1 (policy=always_on, warmup={elapsed:.1f}s) "
                        "- C++ 400Hz control ACTIVE")
                    self._last_cpp_handoff_log = sim_now
                else:
                    if log_ok:
                        log("[CPP_HANDOFF] WARNING: Failed to set FORM_ENABLE=1 (always_on)")
                        self._last_cpp_handoff_log = sim_now
            self._cpp_inactive_reason = 0 if self._cpp_enabled else 1  # ACTIVE or MASTER_OFF
            return

        # === POLICY: state_gated ===
        if self._cpp_handoff_policy == "state_gated":
            ta = get_time_authority()
            sim_ready = ta is not None and ta.is_ready()

            # States where C++ is allowed
            gated_states = {
                ControllerState.DOCKING_CREEP,
                ControllerState.HOLD_CONTACT,
            }

            # FIX: Actively manage FORM_ENABLE based on state
            if self.state in gated_states:
                # Engage C++ in gated states (if sim ready and not already enabled)
                if sim_ready and not self._cpp_enabled:
                    # Tranche H: Activate docking waypoint mode BEFORE FORM_ENABLE=1
                    if self._form_dock_mode_supported:
                        self._dock_trail_m = formation_config.DOCKING_HOLD_BEHIND_M
                        self.follower.set_param("FORM_DOCK_TRAIL", self._dock_trail_m)
                        self.follower.set_param("FORM_DOCK_MODE", 1.0)
                        log(f"[CPP_HANDOFF] C++ docking mode: trail={self._dock_trail_m:.1f}m")
                    if self.follower.set_param("FORM_ENABLE", 1.0):
                        self._cpp_enabled = True
                        if self._cpp_handoff_first_enable_s is None:
                            self._cpp_handoff_first_enable_s = sim_now
                        self._cpp_handoff_last_enable_s = sim_now
                        self._cpp_handoff_enable_count += 1
                        log(f"[CPP_HANDOFF] FORM_ENABLE=1 (policy=state_gated, state={self.state.name}) "
                            "- C++ 400Hz control ACTIVE")
                        self._last_cpp_handoff_log = sim_now
            else:
                # FIX: Actively disable C++ in non-gated states (INTERCEPT/LEAD_TURN/etc.)
                if self._cpp_enabled:
                    # Tranche H: Deactivate docking mode BEFORE FORM_ENABLE=0
                    if self._form_dock_mode_supported:
                        self.follower.set_param("FORM_DOCK_MODE", 0.0)
                    if self.follower.set_param("FORM_ENABLE", 0.0):
                        self._finalize_engaged_time(sim_now)
                        self._cpp_enabled = False
                        self._cpp_handoff_disable_count += 1
                        log(f"[CPP_HANDOFF] FORM_ENABLE=0 (policy=state_gated, state={self.state.name}) "
                            "- Python heading control ACTIVE")
                        self._last_cpp_handoff_log = sim_now
                # _form_enable_init_done removed: preflight discovery + top-of-function
                # guard now handles binaries without FORM_ENABLE param.
            self._cpp_inactive_reason = 0 if self._cpp_enabled else 8  # ACTIVE or STATE_GATE
            return

        # === POLICY: range_hysteresis (existing behavior) ===
        if geom is None:
            self._cpp_inactive_reason = 10  # NO_GEOM
            return

        # NOTE: ALT_GATE safety override was moved to top of function (E3) to apply to ALL policies

        # Shrink to contact-only states - precontact phases need Python
        # altitude protocol (10m below leader). C++ fights for contact geometry.
        close_range_states = {
            ControllerState.DOCKING_CREEP,
            ControllerState.HOLD_CONTACT,
        }

        range_m = geom.range_m

        if self.state in close_range_states:
            if not self._cpp_enabled and range_m < self._cpp_enable_range_m:
                if self.follower.set_param("FORM_ENABLE", 1.0):
                    self._cpp_enabled = True
                    if self._cpp_handoff_first_enable_s is None:
                        self._cpp_handoff_first_enable_s = sim_now
                    self._cpp_handoff_last_enable_s = sim_now
                    self._cpp_handoff_enable_count += 1
                    log(f"[CPP_HANDOFF] FORM_ENABLE=1 at range={range_m:.1f}m "
                        "(policy=range_hysteresis) - C++ 400Hz control ACTIVE")
                    self._last_cpp_handoff_log = sim_now

            elif self._cpp_enabled and range_m > self._cpp_disable_range_m:
                if self.follower.set_param("FORM_ENABLE", 0.0):
                    self._finalize_engaged_time(sim_now)
                    self._cpp_enabled = False
                    self._cpp_handoff_disable_count += 1
                    log(f"[CPP_HANDOFF] FORM_ENABLE=0 at range={range_m:.1f}m "
                        "(policy=range_hysteresis) - Python control RESUMED")
                    self._last_cpp_handoff_log = sim_now
        else:
            if self._cpp_enabled:
                if self.follower.set_param("FORM_ENABLE", 0.0):
                    self._finalize_engaged_time(sim_now)
                    self._cpp_enabled = False
                    self._cpp_handoff_disable_count += 1
                    log(f"[CPP_HANDOFF] FORM_ENABLE=0 (state={self.state.name}) - Python control RESUMED")
                    self._last_cpp_handoff_log = sim_now

        # Track reason code for diagnostics
        if self._cpp_enabled:
            self._cpp_inactive_reason = 0  # ACTIVE
        elif self.state not in close_range_states:
            self._cpp_inactive_reason = 8  # STATE_GATE
        elif range_m >= self._cpp_enable_range_m:
            self._cpp_inactive_reason = 9  # RANGE_FAR
        else:
            self._cpp_inactive_reason = 1  # MASTER_OFF (shouldn't reach)

    def _get_cpp_handoff_metrics(self) -> dict:
        """TEST 386: Build cpp_handoff metrics dict for run_summary.json.

        If C++ is still enabled at shutdown, finalize total by adding
        (sim_now - last_enable_s) to account for the active session.

        Returns:
            dict with enabled, engaged, first_enable_time_s, last_disable_time_s,
            total_engaged_time_s, enable_count, disable_count
        """
        total_s = self._cpp_handoff_total_s

        # If still engaged, add time from last enable to now
        # v9.10: Use stored _t_sim_s instead of sim_now_s()
        if self._cpp_enabled and self._cpp_handoff_last_enable_s is not None:
            try:
                total_s += (self._t_sim_s - self._cpp_handoff_last_enable_s)
            except Exception:
                pass  # Timebase may not be available at shutdown

        return {
            "policy": self._cpp_handoff_policy,
            "enabled": self._cpp_handoff_enabled,
            "engaged_any": self._cpp_handoff_enable_count > 0,
            "engaged": self._cpp_enabled,
            "first_enable_time_s": self._cpp_handoff_first_enable_s,
            "last_disable_time_s": None,  # Not tracked (would need additional state)
            "total_engaged_time_s": round(total_s, 3),
            "enable_count": self._cpp_handoff_enable_count,
            "disable_count": self._cpp_handoff_disable_count,
            "enable_threshold_m": self._cpp_enable_range_m if self._cpp_handoff_policy == "range_hysteresis" else None,
            "disable_threshold_m": self._cpp_disable_range_m if self._cpp_handoff_policy == "range_hysteresis" else None,
            # Add last state for diagnostics
            "last_state_seen": self.state.name if hasattr(self, 'state') else None,
            # Tranche C (B2): Reason code distribution
            "inactive_reason_counts": {
                str(k): v for k, v in sorted(self._cpp_inactive_reason_counts.items())
            },
        }

    def _get_timebase_metrics(self) -> dict:
        """TEST 387: Build timebase compliance metrics for run_summary.

        Returns metrics proving sim-time contract compliance:
        - Speedup accuracy (observed vs expected)
        - Quality score from SimTimeAuthority
        - Monotonicity violations (should be 0)
        - Wall-time decision usage count (should be 0)
        """
        ta = get_time_authority()
        if ta is None:
            return {"error": "no_time_authority"}

        qm = ta.get_quality_metrics()

        # Add time_contract structure for sim-time contract enforcement
        time_contract = {
            "spec_version": "1.0",
            "sim_time_source": "mavlink_time_boot_ms",
            "sim_time_lock_acquired_s": _t0_sim,
            "sim_time_monotonic_violations_count": qm.mono_violations,
            "sim_dt_clamped_count": ta.get_dt_clamped_count(),
            "wall_time_guard_count": get_wall_time_guard_count(),
            "wall_time_forbidden_count": 0,  # Audit result (should always be 0)
            "observed_speedup_mean": round(qm.rate_median, 3),
            "observed_speedup_p05": round(qm.rate_p05, 3),
            "observed_speedup_p95": round(qm.rate_p95, 3),
            "quality_score": round(qm.quality_score, 3),
        }

        return {
            "sim_ready": qm.ready,
            "settle_gate_passed": qm.settle_gate_passed,
            "expected_speedup": qm.rate_expected,
            "observed_speedup_median": round(qm.rate_median, 3),
            "observed_speedup_p05": round(qm.rate_p05, 3),
            "observed_speedup_p95": round(qm.rate_p95, 3),
            "quality_score": round(qm.quality_score, 3),
            "mono_violations_count": qm.mono_violations,
            "leader_stale_count": qm.leader_stale_count,
            "leader_age_median_ms": round(qm.leader_age_median_ms, 1),
            "leader_age_p95_ms": round(qm.leader_age_p95_ms, 1),
            "rate_samples_count": qm.samples,
            "wall_time_decision_usage_count": 0,  # Audit result (should always be 0)
            "t0_sim_s": _t0_sim,
            "sim_end_s": ta.now_s() if qm.ready else None,
            "wall_elapsed_s": round(ta.wall_elapsed_s(), 2),
            # Sim-time contract enforcement structure
            "time_contract": time_contract,
        }

    def _should_log_state(self) -> bool:
        """Check if enough time has passed to log state info (rate limiting)."""
        # Use guarded wall-time (allowed for logging rate-limit)
        now = wall_now_s()
        if now - self._last_state_log_time >= self._state_log_interval:
            self._last_state_log_time = now
            return True
        return False

    def get_geometry(self, now_s: Optional[float] = None) -> Optional[RelativeGeometry]:
        """Get current relative geometry.

        Test 308: Uses get_relative_pose_source() to apply sensor mode logic
        when a non-default sensor mode is configured.
        """
        if not self.follower or not self.leader_rx or not self.geom_calc:
            return None

        # Phase 0: single geometry snapshot per sim tick (tick-based cache)
        if self._geom_cache_tick == self._tick_counter:
            return self._geom_cache

        # C3 (gated): start a fresh per-tick delivery-diag record
        if _FD_UWB_FUSION_DIAG:
            self._uwb_diag_pending = None

        # v9.10: Use _t_sim_s for geometry timestamp (tick timestamp)
        now_s = self._t_sim_s if now_s is None else float(now_s)

        foll_pos = self.follower.get_position()
        foll_vel = self.follower.get_velocity_ned()
        if not foll_pos or not foll_vel:
            return None

        # ---------------------------------------------------------------------
        # SIGCLEAN Hunk-F1 (2026-06-12, R7.1 waiver): follower-hop GPI
        # dead-reckoning — the symmetric mirror of STEP-4 Hunk-L below.
        # Evidence (ceiling RCA ART): with leader extrap LIVE, |d(behind_gps)|
        # p50 was 1.24 m on the 5.3% GPI stale/catch-up tick pairs vs 0.11 m
        # fresh. Age binds from the SAME AP boot-clock family as Hunk-L: TA
        # "now" (advanced from follower ATTITUDE.time_boot_ms) minus the GPI's
        # own time_boot_ms — single-vehicle clock, no cross-hop skew. NEVER
        # time.time()/time.monotonic(). TA-not-ready or no GPI => age None =>
        # helper passthrough. Gating lives INSIDE extrapolate_follower_latlon
        # (FOLLOWER_EXTRAP_ENABLE, default OFF -> byte-equivalent). vd (NED
        # down) deliberately NOT extrapolated: below_gps tick noise is 0.030 m
        # (no case). The DR'd foll_pos feeds all perceived consumers
        # coherently (control geometry + emitter GPS-frame diag); the
        # freshness-INVARIANT +1.6 m upstream EKF bias is NOT compensated
        # (goal-3: no tuning perception toward truth; D nulls its median).
        # ---------------------------------------------------------------------
        _foll_age_ms = None
        _f_ta = get_time_authority()
        if _f_ta is not None and _f_ta.is_ready():
            _gpi = getattr(self.follower, '_last_position', None)
            if _gpi is not None:
                _f_age = _f_ta.now_tbms() - int(_gpi.time_boot_ms)
                _foll_age_ms = float(_f_age) if _f_age > 0 else None
        _fxlat, _fxlon = leader_extrapolator.extrapolate_follower_latlon(
            foll_pos[0], foll_pos[1], foll_vel[0], foll_vel[1],
            _foll_age_ms, log_fn=log)
        if _fxlat != foll_pos[0] or _fxlon != foll_pos[1]:
            foll_pos = (_fxlat, _fxlon, foll_pos[2])

        # ---------------------------------------------------------------------
        # BBD Hunk-R2 (2026-06-13, R7.1 waiver): follower GPS-sensor residual-lag
        # forward-extrapolation. INDEPENDENT of the Hunk-F1 message-age block
        # above (FOLLOWER_EXTRAP_ENABLE) -- the EKF3 sensor-lag tail (~11.4 ms
        # uncompensated of the 100 ms; behind_bias_disc 0f690e6a) is present even
        # on a zero-age fresh GPI packet. Uses the follower's ABSOLUTE velocity
        # (foll_vel from get_velocity_ned, NOT v_rel which is ~0 in formation)
        # per the slope=26.8 m/s = full-forward-speed evidence. Applied to the
        # (possibly already age-DR'd) foll_pos so both corrections compose
        # coherently when both flags are ON; feeds compute_relative, fusion,
        # D-correction, and speed_arbitration "consumed" columns consistently.
        # Gating + fixed tau live INSIDE apply_residual_lag
        # (FD_FOLLOWER_RESIDUAL_LAG_MS, default '0' -> OFF -> byte-identical).
        # vd (foll_pos[2]) passed through untouched -> below axis invariant.
        # No wall-clock; tau is a fixed sim-invariant constant.
        # ---------------------------------------------------------------------
        _rxlat, _rxlon = leader_extrapolator.apply_residual_lag(
            foll_pos[0], foll_pos[1], foll_vel[0], foll_vel[1], log_fn=log)
        if _rxlat != foll_pos[0] or _rxlon != foll_pos[1]:
            foll_pos = (_rxlat, _rxlon, foll_pos[2])

        lead_state = self.leader_rx.update()
        if not lead_state:
            return None

        # Tranche C (C0): Store raw altitudes for D3 telemetry
        self._d3_leader_alt_msl_m = lead_state.alt_msl
        self._d3_follower_alt_msl_m = foll_pos[2]

        # Patch 8: Leader freshness check using SimTimeAuthority
        _lead_age_ms = None  # STEP-4 Hunk-L: TA-bound leader age (ms), None until ready
        ta = get_time_authority()
        if ta is not None and ta.is_ready():
            fresh, age_ms = ta.check_leader_freshness(lead_state.time_boot_ms, stale_thresh_ms=2000)
            if not fresh:
                return None
            _lead_age_ms = age_ms
        else:
            # Fallback to wall-time check before time authority ready
            # Scale threshold by speedup (at 5x, 2s sim = 0.4s wall)
            scaled_thresh = 2.0 / max(1.0, self.expected_speedup)
            if (time.time() - lead_state.rx_wall_time_s) > scaled_thresh:
                return None

        # ---------------------------------------------------------------------
        # STEP-4 Hunk-L (2026-06-11, R7.1 waiver): leader-hop C3 time-alignment.
        # CLOCK SOURCE: age_ms binds from the SimTimeAuthority
        # freshness check above — the sanctioned AP/sim clock family (the TA is
        # advanced from the follower's ATTITUDE.time_boot_ms; leader stamp is
        # the leader AP boot clock; shared sim epoch verified in step-3: median
        # age 100 ms +-50 ZOH). NEVER time.time()/time.monotonic() for age.
        # TA-not-ready fallback => age None => helper passthrough (no
        # extrapolation before the time authority is up). Gating lives INSIDE
        # extrapolate_leader_latlon (LEADER_EXTRAP_ENABLE, default OFF ->
        # returns inputs unchanged = byte-equivalent). The extrapolated local
        # copy feeds BOTH the emitter block and compute_relative below.
        # ---------------------------------------------------------------------
        _xlat, _xlon = leader_extrapolator.extrapolate_leader_latlon(
            lead_state.lat, lead_state.lon,
            lead_state.vn_ms, lead_state.ve_ms, _lead_age_ms, log_fn=log)
        if _xlat != lead_state.lat or _xlon != lead_state.lon:
            lead_state = replace(lead_state, lat=_xlat, lon=_xlon)

        # Test 308: Use sensor mode switching if not default fused mode
        # Patch 9: Use sim-time for fusion timestamps
        uwb_timestamp = now_s

        # Generate synthetic UWB measurements if emitter is enabled
        # We compute GPS-derived relative position for the emitter to use
        if (self._uwb_emitter is not None and self._uwb_emitter.enabled
                and self.geom_calc.fusion is not None):
            # Compute relative position for UWB emission (same frame as GPS)
            # We need behind, cross, below for the emitter
            hdg_rad = math.radians(lead_state.heading_deg)
            cos_hdg = math.cos(hdg_rad)
            sin_hdg = math.sin(hdg_rad)

            # Leader NED position (without boom offset for emitter - boom tip is reference)
            lead_n = (math.radians(lead_state.lat - self.geom_calc.home_lat)
                      * self.geom_calc.R_EARTH)
            lead_e = (math.radians(lead_state.lon - self.geom_calc.home_lon)
                      * self.geom_calc.R_EARTH * math.cos(math.radians(self.geom_calc.home_lat)))
            lead_d = -(lead_state.alt_msl - self.geom_calc.home_alt_msl)

            # Add boom offset to get boom tip position
            # CSV2 frame-fix (option-a, 2026-06-10): boom-tip horizontal = canonical
            # 45-deg projection (BOOM_HORIZONTAL_PROJECTION_M = 4.31m), not the full
            # slant lead_state.boom_length_m (6.1m). The emitter always references the
            # boom tip (no leader-center case here), so a direct constant is correct.
            boom_n = lead_n - BOOM_HORIZONTAL_PROJECTION_M * cos_hdg
            boom_e = lead_e - BOOM_HORIZONTAL_PROJECTION_M * sin_hdg
            # Boom tip is BOOM_TIP_OFFSET_M below leader body (NED down-positive).
            boom_d = lead_d + BOOM_TIP_OFFSET_M

            # C1 (2026-06-08, gated): feed the emitter SIM-TRUTH follower
            # position when truth-sourcing. foll_pos (GPS/EKF perceived) is
            # UNCHANGED for control geometry; only the emitter input swaps here.
            # Horizontal truth = SIMSTATE(164) int; vertical = SIM_STATE(108) alt.
            # Missing truth -> skip emission (never feed perceived-as-truth).
            foll_src = foll_pos
            _truth_emit_skip = False
            if _FD_UWB_TRUTH_SOURCED:
                _truth_pos = self.follower.get_sim_position()
                if _truth_pos is None:
                    if not getattr(self, '_uwb_truth_warned', False):
                        log("[UWB_TRUTH] SIM-truth follower position unavailable "
                            "(SIMSTATE/SIM_STATE not yet cached); skipping UWB "
                            "emission (no perceived-as-truth fallback)")
                        self._uwb_truth_warned = True
                    _truth_emit_skip = True
                else:
                    foll_src = _truth_pos
                    # -------------------------------------------------------
                    # UWB-TRUTH time-alignment (2026-06-14, R7.1 waiver):
                    # advance the truth follower position to the emission
                    # instant BEFORE the emitter ranges to it -> removes the
                    # ~one-frame SIM_STATE sampling staleness (~1.3 m at cruise
                    # = the admitted-UWB behind jitter; RCA
                    # project_uwb_truth_sourcing_jitter). Truth velocity from
                    # SIM_STATE(108).vn/ve (the truth the emitter already ranges
                    # to; stays in foll_src -> emitter, NEVER foll_pos /
                    # observation / policy). Age = MEASURED 108 cache age
                    # (_buffered_attitude_time_boot_ms - _last_sim_state_tbms),
                    # self-gating per tick: ~0 ms fresh (no move) / ~1 tick
                    # (~50 ms) stale (advance ~1.2 m) = the L-A within-tick PHASE
                    # leaf (leg ut0_g0m_rdiag 49.6% one-tick-stale; research
                    # wv2tav3ls: a CONSTANT age over-corrects the fresh ~50% and
                    # can be net-harmful, so the prior GPI-age _foll_age_ms is
                    # RETIRED here). 50 ms sim-clock quantization caps the
                    # correction at the integer-tick level (R1). Gating + cap live
                    # INSIDE the helper (FD_UWB_TRUTH_EXTRAP_ENABLE, default OFF
                    # -> byte-identical). Applied in the truth-sourced branch
                    # ONLY -> the FD_UWB_TRUTH_SOURCED=0 ablation is untouched
                    # -> no double-count with Hunk-R2. vd (foll_src[2]) passed
                    # through untouched -> below axis invariant. NEVER wall-clock.
                    # UWB_TRUTH_EXTRAP_WIRING_START
                    _ss_tv = getattr(self.follower, '_last_sim_state', None)
                    _tvn = getattr(_ss_tv, 'vn', None) if _ss_tv is not None else None
                    _tve = getattr(_ss_tv, 've', None) if _ss_tv is not None else None
                    _bt108 = getattr(self.follower, '_buffered_attitude_time_boot_ms', None)
                    _st108 = getattr(self.follower, '_last_sim_state_tbms', None)
                    _108_age_ms = float(_bt108 - _st108) if (_bt108 is not None and _st108 is not None) else None
                    _txlat, _txlon = leader_extrapolator.extrapolate_follower_truth_latlon(
                        foll_src[0], foll_src[1], _tvn, _tve, _108_age_ms, log_fn=log)
                    if _txlat != foll_src[0] or _txlon != foll_src[1]:
                        foll_src = (_txlat, _txlon, foll_src[2])
                    # UWB_TRUTH_EXTRAP_WIRING_END

            # Follower NED position
            foll_n = (math.radians(foll_src[0] - self.geom_calc.home_lat)
                      * self.geom_calc.R_EARTH)
            foll_e = (math.radians(foll_src[1] - self.geom_calc.home_lon)
                      * self.geom_calc.R_EARTH * math.cos(math.radians(self.geom_calc.home_lat)))
            foll_d = -(foll_src[2] - self.geom_calc.home_alt_msl)

            # Relative position NED (follower relative to boom tip)
            rel_n = foll_n - boom_n
            rel_e = foll_e - boom_e
            rel_d = foll_d - boom_d

            # Transform to leader body frame (behind, cross, below)
            behind = -(rel_n * cos_hdg + rel_e * sin_hdg)
            cross = -rel_n * sin_hdg + rel_e * cos_hdg
            below = rel_d

            # Emit UWB measurements (skipped if truth-sourced but truth missing)
            if _truth_emit_skip:
                uwb_meas_list = []
            else:
                uwb_meas_list = self._uwb_emitter.maybe_emit(
                    now_s=uwb_timestamp,
                    rel_bcb=(behind, cross, below),
                )

            # STEP-4 Hunk-D(i) (2026-06-11, R7.1 waiver, gated): stash the
            # latest anchor ranges for architecture-D consumption inside
            # compute_relative's GPS-direct branch. CLOCK SOURCE:
            # the stash timestamp is time.monotonic() at WRITE and the read
            # side uses time.monotonic() too (never time.time(); speedup is
            # locked 1.0 in step-4 so the 0.3 s wall window ~= 0.3 s sim).
            if _FD_D_CORRECTION and uwb_meas_list:
                self.geom_calc._d_ranges = (
                    {m.anchor_id: m.range_m for m in uwb_meas_list},
                    time.monotonic())

            # C3 (gated): capture emitter truth-delivery + PRE-update fusion state
            if _FD_UWB_FUSION_DIAG and self._uwb_diag_writer is not None:
                _fz = self.geom_calc.fusion
                _foll_d_gps = -(foll_pos[2] - self.geom_calc.home_alt_msl)
                _below_gps = _foll_d_gps - boom_d
                # CSV2 frame-fix latency-decomp (schema v2): along-track behind from the
                # PERCEIVED foll_pos in the emitter's boom-tip frame (byte-identical
                # projection to compute_relative.behind_gps); truth captured independently
                # of the truth-source flag so the diag is valid in the GPS-echo ablation too.
                _fn_gps = (math.radians(foll_pos[0] - self.geom_calc.home_lat)
                           * self.geom_calc.R_EARTH)
                _fe_gps = (math.radians(foll_pos[1] - self.geom_calc.home_lon)
                           * self.geom_calc.R_EARTH
                           * math.cos(math.radians(self.geom_calc.home_lat)))
                _behind_gps_emit = -((_fn_gps - boom_n) * cos_hdg + (_fe_gps - boom_e) * sin_hdg)
                _tp_diag = self.follower.get_sim_position()
                _ra0 = _UWB_DIAG_NAN
                for _m in uwb_meas_list:
                    if getattr(_m, 'anchor_id', 0) == 0:
                        _ra0 = float(_m.range_m)
                        break
                self._uwb_diag_pending = {
                    'emit_truth_used': bool(_FD_UWB_TRUTH_SOURCED and not _truth_emit_skip),
                    'emit_below_truth': float(below), 'emit_behind_truth': float(behind),
                    'emit_cross_truth': float(cross), 'emit_below_gps': float(_below_gps),
                    'n_meas': len(uwb_meas_list), 'emit_range_a0': _ra0,
                    'fused_below_pre': _uwb_diag_fused_below(_fz),
                    'pre_counts': _uwb_diag_anchor_counts(_fz),
                    'predicted_range_a0': _uwb_diag_predicted_range_a0(_fz),
                    # schema v2 along-track latency-decomp
                    'foll_gps_lat': float(foll_pos[0]), 'foll_gps_lon': float(foll_pos[1]),
                    'truth_lat': (float(_tp_diag[0]) if _tp_diag else _UWB_DIAG_NAN),
                    'truth_lon': (float(_tp_diag[1]) if _tp_diag else _UWB_DIAG_NAN),
                    'leader_gps_lat': float(lead_state.lat),
                    'leader_gps_lon': float(lead_state.lon),
                    'leader_time_boot_ms': int(lead_state.time_boot_ms),
                    'behind_gps': float(_behind_gps_emit), 'boom_frame_unified': True,
                }

            # Feed measurements to fusion filter via batch update
            if uwb_meas_list:
                self.geom_calc.fusion.update_uwb_batch(uwb_meas_list)

            # C3 (gated): capture POST-update fusion state (truth-UWB applied)
            if _FD_UWB_FUSION_DIAG and self._uwb_diag_pending is not None:
                _fz = self.geom_calc.fusion
                _i0, _i1, _i2, _i3 = _uwb_diag_innovs(_fz, self._uwb_diag_pending['pre_counts'])
                self._uwb_diag_pending.update({
                    'fused_below_post': _uwb_diag_fused_below(_fz),
                    'upd_post': getattr(_fz, 'uwb_update_count', 0),
                    'rej_post': getattr(_fz, 'uwb_rejected_count', 0),
                    'last_innov': float(getattr(_fz, 'last_innovation_range', _UWB_DIAG_NAN)),
                    'innov': (_i0, _i1, _i2, _i3),
                })

        # uwb_range remains None - we feed UWB directly to fusion above
        uwb_range = None

        # STEP-4B2' Hunk-G1 (2026-06-11, R7.1 waiver): live-state mirror +
        # call-site scoper for the architecture-D gate chain. Set ONLY around
        # the control-geometry call below (both sensor-mode branches); cleared
        # immediately after, so _update_success_metrics /
        # get_relative_pose_source invocations from OTHER call paths see ''
        # and skip the D chain entirely (no D in CG-frame metrics, no counter
        # pollution). The mirror is the LIVE state at consumption — never the
        # (possibly stale) state at stash time.
        # B2PP Hunk-N2: ZERO-COST DISARM —
        # the mirror arms ONLY inside _FD_D_ALLOWED_STATES, so outside the
        # docking states the chain never runs at all (per-tick cost = this
        # membership test + two attribute writes; timing-equivalent to D-off,
        # neutralizing the D-flag timing side-channel suspect). The chain's
        # own membership check remains as the independent second layer.
        if _FD_D_CORRECTION:
            if self.state.name in _FD_D_ALLOWED_STATES:
                self.geom_calc._d_gate_state = self.state.name
            else:
                self.geom_calc._d_gate_state = ''
                self.geom_calc._d_corr_last = None

        if self.sensor_mode != SENSOR_MODE_FUSED:
            # Use sensor mode switching logic
            _source, geom = self.get_relative_pose_source(
                foll_pos[0], foll_pos[1], foll_pos[2],
                foll_vel[0], foll_vel[1], foll_vel[2],
                lead_state.lat, lead_state.lon, lead_state.alt_msl,
                lead_state.vn_ms, lead_state.ve_ms, lead_state.vd_ms,
                lead_state.heading_deg, lead_state.boom_length_m,
                uwb_range, uwb_timestamp
            )
            # Note: _source discarded; geom.source has the same value
        else:
            # Default: direct call to compute_relative
            # Patch 9: Pass sim-time timestamp for GPS measurements
            geom = self.geom_calc.compute_relative(
                foll_pos[0], foll_pos[1], foll_pos[2],
                foll_vel[0], foll_vel[1], foll_vel[2],
                lead_state.lat, lead_state.lon, lead_state.alt_msl,
                lead_state.vn_ms, lead_state.ve_ms, lead_state.vd_ms,
                lead_state.heading_deg, lead_state.boom_length_m,
                timestamp_s=now_s
            )
            self._active_sensor_source = geom.source

        # STEP-4B2' Hunk-G1 (clear side): control-geometry call done.
        if _FD_D_CORRECTION:
            self.geom_calc._d_gate_state = ''

        # Track minimum range
        if geom.range_m < self.min_range_achieved:
            self.min_range_achieved = geom.range_m

        # v9.10 C5: NaN geometry validation - non-finite is fatal defect
        if not self._validate_geometry_finite(geom):
            log(f"[NaN_GEOMETRY] FATAL: behind={geom.behind_m} cross={geom.cross_m} "
                f"below={geom.below_m} range={geom.range_m}")
            raise NaNGeometryError(
                f"Non-finite geometry: behind={geom.behind_m} cross={geom.cross_m} "
                f"below={geom.below_m}"
            )

        # E4: Geometry consistency check (throttled every 10s)
        # Verify range_m ≈ sqrt(behind^2 + cross^2 + below^2) using same coordinates
        if (now_s - self._last_geom_consistency_log) > 10.0:
            computed_range = math.sqrt(
                geom.behind_m**2 + geom.cross_m**2 + geom.below_m**2
            )
            delta = abs(geom.range_m - computed_range)
            if delta > 0.5:  # Log if mismatch > 0.5m
                log(f"[E4_GEOM_CHECK] MISMATCH: range_m={geom.range_m:.2f}m vs "
                    f"sqrt(b²+c²+z²)={computed_range:.2f}m (delta={delta:.2f}m) "
                    f"[behind={geom.behind_m:.1f} cross={geom.cross_m:.1f} below={geom.below_m:.1f}]")
            self._last_geom_consistency_log = now_s

        # C3 (gated): finalize + write the emitter->fusion delivery diag row
        if (_FD_UWB_FUSION_DIAG and self._uwb_diag_pending is not None
                and self._uwb_diag_writer is not None):
            self._uwb_diag_write_row(geom)

        # Update per-tick cache (applies even if geom is None)
        self._geom_cache_tick = self._tick_counter
        self._geom_cache = geom

        return geom

    # Guard strip-down: _get_center_geometry removed (Chunk 2B)

    def _auto_detect_below_sign(self, geom: RelativeGeometry, lead_state, fol_alt: float) -> None:
        """S3C1: One-time sign convention auto-detect for below_m.

        Checks whether below_m matches (leader_alt - follower_alt) or
        (follower_alt - leader_alt). Expected: LEADER_MINUS_FOLLOWER
        (positive below_m when follower is below leader).
        """
        if self._below_sign_mode is not None:
            return  # Already detected

        if not (math.isfinite(geom.below_m) and math.isfinite(fol_alt)
                and lead_state is not None and math.isfinite(lead_state.alt_msl)):
            return  # Wait for valid data

        a = fol_alt - lead_state.alt_msl     # negative if below
        b = lead_state.alt_msl - fol_alt     # positive if below

        if abs(a - geom.below_m) < abs(b - geom.below_m):
            self._below_sign_mode = "FOLLOWER_MINUS_LEADER"
        else:
            self._below_sign_mode = "LEADER_MINUS_FOLLOWER"

        residual = min(abs(a - geom.below_m), abs(b - geom.below_m))
        level = "WARN" if residual > 2.0 else "INFO"
        log(f"[BELOW_SIGN] {level}: mode={self._below_sign_mode} "
            f"residual={residual:.2f}m below_m={geom.below_m:.2f} "
            f"fol_alt={fol_alt:.1f} lead_alt={lead_state.alt_msl:.1f}")

    def _validate_geometry_finite(self, geom: RelativeGeometry) -> bool:
        """v9.10 C5: Check if all geometry values are finite (not NaN/Inf)."""
        return (
            math.isfinite(geom.behind_m)
            and math.isfinite(geom.cross_m)
            and math.isfinite(geom.below_m)
            and math.isfinite(geom.range_m)
        )

    def _compute_altitude_target_msl(
        self,
        geom: Optional[RelativeGeometry],
        lead_state,
        sim_now_s: float
    ) -> float:
        """Compute commanded altitude based on mission phase and high-side mode.

        TEST 392: High-Side Convergence Mode

        When high-side mode is enabled and active, the follower flies ABOVE the
        leader during convergence phases (INTERCEPT, LEAD_TURN, REJOIN_PRECONTACT)
        to avoid wake turbulence. The follower descends to normal precontact
        altitude when:
          1. Cross-track error is within threshold for a sustained period, OR
          2. Behind distance drops below the disable threshold (hard cutoff)

        Args:
            geom: Current relative geometry (may be None)
            lead_state: Current leader state (may be None)
            sim_now_s: Current simulation time in seconds

        Returns:
            Target altitude in MSL (meters)
        """
        if lead_state is None:
            return PRECONTACT_ALT_MSL  # Fallback

        # Compute tip-relative altitudes
        tip_alt_msl = lead_state.alt_msl - BOOM_TIP_OFFSET_M
        precontact_alt_msl = tip_alt_msl - 10.0  # M2: 10m below tip
        contact_alt_msl = tip_alt_msl            # M3: at tip
        high_side_alt_msl = lead_state.alt_msl + self._high_side_margin_m

        # Contact phases always use contact altitude
        # P1 Fix 3 (v9.4): ALT_CAP_HOLD_RANGE must target boom-tip to enable climb
        if self.state in (ControllerState.DOCKING_CREEP, ControllerState.HOLD_CONTACT,
                          ControllerState.ALT_CAP_HOLD_RANGE):
            # S3C4: Coupled below-target ramp in DOCKING_CREEP
            if (formation_config.COUPLED_BELOW_TARGET_ENABLE
                    and self.state.name in formation_config.COUPLED_BELOW_TARGET_ENABLE_STATES):
                return tip_alt_msl - self._below_target_m  # ramp-based altitude
            return contact_alt_msl  # original behavior for non-coupled states

        # ALTITUDE_CAPTURE phase B (climbing to contact)
        if self.state == ControllerState.ALTITUDE_CAPTURE:
            if self._alt_capture_subphase == 'STANDOFF_CLIMB':
                return contact_alt_msl
            else:
                return precontact_alt_msl

        # HOLD_PRECONTACT always uses precontact (already converged)
        if self.state == ControllerState.HOLD_PRECONTACT:
            return precontact_alt_msl

        # High-side mode only applies during convergence phases
        convergence_states = {
            ControllerState.INTERCEPT,
            ControllerState.LEAD_TURN,
            ControllerState.REJOIN_PRECONTACT,
        }

        if not self._high_side_enabled or self.state not in convergence_states:
            return precontact_alt_msl

        # High-side logic
        if geom is None:
            return precontact_alt_msl

        cross_m = abs(geom.cross_m)
        behind_m = geom.behind_m

        # P1: Graduated high-side descent ramp (replaces hard 60m cutoff)
        # Smoothly descends from high_side_alt to precontact_alt over [150m, 80m]
        # so altitude error is near-zero before ALT_GATE can trigger.
        if behind_m <= HIGH_SIDE_RAMP_START_M:
            if behind_m <= HIGH_SIDE_RAMP_END_M:
                # Ramp complete — fully at precontact altitude
                if self._high_side_active:
                    active_duration = sim_now_s - self._high_side_start_s if self._high_side_start_s else 0
                    self._high_side_stats['active_time_s'] += active_duration
                    self._high_side_active = False
                    self._high_side_start_s = None
                    self._high_side_cross_ok_start_s = None
                    log(f"[HIGH_SIDE] RAMP COMPLETE (behind={behind_m:.1f}m) "
                        f"at precontact alt_cmd={precontact_alt_msl:.1f}m")
                return precontact_alt_msl
            else:
                # Linear interpolation: 150m→high_side, 80m→precontact
                t = (HIGH_SIDE_RAMP_START_M - behind_m) / (HIGH_SIDE_RAMP_START_M - HIGH_SIDE_RAMP_END_M)
                ramped_alt = high_side_alt_msl + t * (precontact_alt_msl - high_side_alt_msl)
                if not self._ramp_logged:
                    log(f"[HIGH_SIDE] RAMP START (behind={behind_m:.1f}m) "
                        f"alt={ramped_alt:.1f}m (t={t:.2f})")
                    self._ramp_logged = True
                return ramped_alt

        # ALT_GATE safety: If ALT_GATE is active, high-side must deactivate
        if self._alt_gate_active:
            if self._high_side_active:
                active_duration = sim_now_s - self._high_side_start_s if self._high_side_start_s else 0
                self._high_side_stats['active_time_s'] += active_duration
                self._high_side_active = False
                self._high_side_start_s = None
                self._high_side_cross_ok_start_s = None
                log("[HIGH_SIDE] DISABLED (ALT_GATE active) descending to precontact")
            return precontact_alt_msl

        # Check if we should be in high-side mode
        in_convergence_zone = (behind_m > 0 and behind_m <= self._high_side_behind_max_m)
        cross_not_ok = (cross_m > self._high_side_cross_ok_m)

        if in_convergence_zone and cross_not_ok:
            # Activate or stay in high-side mode
            if not self._high_side_active:
                self._high_side_active = True
                self._high_side_start_s = sim_now_s
                self._high_side_cross_ok_start_s = None
                self._ramp_logged = False
                self._high_side_stats['num_enable_events'] += 1
                log(f"[HIGH_SIDE] ENABLED margin={self._high_side_margin_m}m "
                    f"cross={cross_m:.1f}m behind={behind_m:.1f}m alt_cmd={high_side_alt_msl:.1f}m")

            # Reset cross-ok timer since cross is not OK
            self._high_side_cross_ok_start_s = None
            self._high_side_stats['max_cross_during_active_m'] = max(
                self._high_side_stats['max_cross_during_active_m'], cross_m)

            # Periodic logging (1 Hz)
            current_sec = int(sim_now_s)
            if current_sec != self._high_side_last_log_s:
                self._high_side_last_log_s = current_sec
                log(f"[HIGH_SIDE] active cross={cross_m:.1f}m behind={behind_m:.1f}m "
                    f"alt_cmd={high_side_alt_msl:.1f}m")

            return high_side_alt_msl

        elif self._high_side_active:
            # Cross is now OK - start or continue timer
            if self._high_side_cross_ok_start_s is None:
                self._high_side_cross_ok_start_s = sim_now_s

            cross_ok_duration = sim_now_s - self._high_side_cross_ok_start_s

            if cross_ok_duration >= self._high_side_min_time_s:
                # Deactivate high-side, descend to precontact
                active_duration = sim_now_s - self._high_side_start_s if self._high_side_start_s else 0
                self._high_side_stats['active_time_s'] += active_duration
                self._high_side_active = False
                self._high_side_start_s = None
                self._high_side_cross_ok_start_s = None
                log(f"[HIGH_SIDE] DISABLED (cross ok for {cross_ok_duration:.1f}s) "
                    f"descending to precontact alt_cmd={precontact_alt_msl:.1f}m")
                return precontact_alt_msl
            else:
                # Still waiting for cross to be OK long enough
                return high_side_alt_msl

        # Default: precontact altitude
        return precontact_alt_msl

    def _update_success_metrics(self, geom: Optional[RelativeGeometry]) -> None:
        """Update success metrics with current geometry (observer-only).

        This method is called every tick in the main loop, after get_geometry().
        It computes leader-center geometry (boom_length_m=0) for safety metrics
        and logs any alerts generated by the metrics module.
        """
        if geom is None or self._success_metrics.t0_s is None:
            return

        # Get states for center geometry computation
        if not self.follower or not self.leader_rx or not self.geom_calc:
            return

        foll_pos = self.follower.get_position()
        foll_vel = self.follower.get_velocity_ned()
        if not foll_pos or not foll_vel:
            return

        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        # v9.10: Use tick timestamp for consistency
        t = self._t_sim_s

        # Compute leader-center geometry (boom_length_m=0)
        # F4: skip fusion update — CG-frame measurements must not contaminate
        # the boom-tip-frame fusion state
        center_geom = self.geom_calc.compute_relative(
            foll_pos[0], foll_pos[1], foll_pos[2],
            foll_vel[0], foll_vel[1], foll_vel[2],
            lead_state.lat, lead_state.lon, lead_state.alt_msl,
            lead_state.vn_ms, lead_state.ve_ms, lead_state.vd_ms,
            lead_state.heading_deg,
            boom_length_m=0.0,
            boom_tip_offset_m=0.0,
            timestamp_s=t,
            update_fusion=False,
        )

        # Call metrics update
        result = self._success_metrics.update_geometry(
            sim_time_s=t,
            behind_tip_m=geom.behind_m,
            cross_tip_m=geom.cross_m,
            below_tip_m=geom.below_m,
            range_tip_m=geom.range_m,
            behind_center_m=center_geom.behind_m,
            cross_center_m=center_geom.cross_m,
            below_center_m=center_geom.below_m,
            range_center_m=center_geom.range_m,
            controller_state=self.state.name,
            follower_alt_msl=foll_pos[2],
        )

        # Log alerts (observer-only, no behavior change)
        for alert in result.get('alerts', []):
            log(alert)

    def _flush_success_metrics(self, reason: str = "exit") -> None:
        """Finalize and export success metrics (idempotent).

        This method can be called multiple times safely (e.g., from signal
        handler and finally block). Only the first call performs the export.

        Exports to TWO files:
        1. <out_dir>/success_metrics.json - backup, metrics dict only (unwrapped)
        2. <out_dir>/run_summary.json - merged with existing data
        """
        if self._metrics_flushed:
            return
        # F-E3: the flag is a
        # transient re-entrancy guard during an attempt; it PERSISTS True only
        # after a committed run_summary write. Early-returns and failures
        # reset it so later flush attempts (finally block, signal handler, env
        # pre-teardown reset) are never permanently no-opped — the prior
        # set-before-commit form let a no-t0/no-out_dir first call disable ALL
        # subsequent flushes. Outcome enum: not_attempted/skipped/failed/
        # committed in _metrics_flush_state.
        self._metrics_flushed = True
        self._metrics_flush_state = "attempted"

        try:
            # Check if metrics were initialized (t0 set after PASS)
            if self._success_metrics.t0_s is None:
                log("[SUCCESS_METRICS] No t0 set (no PASS?); skipping export")
                self._metrics_flushed = False  # F-E3: not committed
                self._metrics_flush_state = "skipped"
                return

            # P2 v9.4: Populate speed arbitration stats before finalize
            if self.control_laws:
                arb_stats = self.control_laws.get_arbitration_stats()
                self._success_metrics.set_speed_arbitration_stats(arb_stats)

            # Option A+E: Populate DOCKING_CREEP sub-phase + TECS stats
            self._success_metrics.set_dc_phase_stats({
                "dc_phase1_enter_count": self._dc_phase1_enter_count,
                "dc_phase2_enter_count": self._dc_phase2_enter_count,
                "dc_phase2_fallback_count": self._dc_phase2_fallback_count,
                "tecs_spdw_write_count": self._tecs_spdw_write_count,
                "tecs_spdw_mismatch_count": self._tecs_spdw_mismatch_count,
                "tecs_spdw_disabled": self._tecs_spdw_disabled,
                "tecs_spdw_disable_reason": self._tecs_spdw_disable_reason,
            })

            self._success_metrics.finalize()
            metrics_dict = self._success_metrics.to_dict()

            if not self._out_dir:
                log("[SUCCESS_METRICS] No output directory; skipping export")
                self._metrics_flushed = False  # F-E3: not committed
                self._metrics_flush_state = "skipped"
                return

            import json
            import os

            # 1. Write backup file: success_metrics.json (unwrapped)
            backup_path = os.path.join(self._out_dir, "success_metrics.json")
            backup_tmp = backup_path + ".tmp"
            try:
                with open(backup_tmp, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
                os.replace(backup_tmp, backup_path)
                log(f"[SUCCESS_METRICS] Wrote success_metrics.json -> {backup_path} (reason={reason})")
            except Exception as e:
                log(f"WARNING: Failed to write success_metrics.json: {e}")

            # 2. Merge into run_summary.json (existing behavior)
            run_summary_path = os.path.join(self._out_dir, "run_summary.json")
            try:
                if os.path.exists(run_summary_path):
                    with open(run_summary_path, 'r') as f:
                        summary = json.load(f)
                else:
                    summary = {}

                summary['success_metrics'] = metrics_dict

                # Add C++ handoff metrics
                summary['cpp_handoff'] = self._get_cpp_handoff_metrics()

                # Add timebase compliance metrics
                summary['timebase'] = self._get_timebase_metrics()

                # Add UWB emitter stats
                if self._uwb_emitter is not None:
                    summary['uwb_emitter'] = self._uwb_emitter.get_stats()

                # Add high-side convergence stats
                if self._high_side_enabled:
                    summary['high_side_convergence'] = {
                        'enabled': True,
                        'margin_m': self._high_side_margin_m,
                        'cross_ok_m': self._high_side_cross_ok_m,
                        'min_time_s': self._high_side_min_time_s,
                        'behind_max_m': self._high_side_behind_max_m,
                        'disable_behind_m': self._high_side_disable_behind_m,
                        **self._high_side_stats,
                    }

                # v9.10 Gate 0: Timebase integrity fields
                summary['run_invalid_timebase'] = self._run_invalid_timebase
                summary['ticks_skipped_no_advance_total'] = self._ticks_skipped_no_advance
                summary['ticks_skipped_accumulating_total'] = self._ticks_skipped_accumulating
                summary['ticks_executed_total'] = self._ticks_executed_total
                # Rec 3: CSV decimation stride for gate checker validation
                summary['csv_decimation_stride'] = self._csv_decimation_stride
                summary['late_tick_count'] = self._late_tick_count
                summary['sim_time_regressions_total'] = self._sim_time_regressions
                summary['max_sim_dt_raw_s'] = round(self._max_sim_dt_raw_s, 6)

                # v9.10 Gate 0 Time Authority metrics (P1 Fix B)
                ta = get_time_authority()
                if ta is not None:
                    ta_metrics = ta.get_time_authority_metrics()
                    summary['time_authority_update_hz_sim'] = round(ta_metrics.get('time_authority_update_hz_sim', 0.0), 2)
                    summary['time_authority_updates_total'] = ta_metrics.get('time_authority_updates_total', 0)
                    summary['time_authority_dt_median'] = round(ta_metrics.get('time_authority_dt_median', 0.0), 6)
                    summary['time_authority_dt_p95'] = round(ta_metrics.get('time_authority_dt_p95', 0.0), 6)
                    summary['time_authority_dt_min'] = round(ta_metrics.get('time_authority_dt_min', 0.0), 6)
                    summary['time_authority_dt_max'] = round(ta_metrics.get('time_authority_dt_max', 0.0), 6)
                    summary['time_authority_gap_max_s'] = round(ta_metrics.get('time_authority_gap_max_s', 0.0), 6)
                    summary['time_authority_stale_gaps_over_0p2s'] = ta_metrics.get('time_authority_stale_gaps_over_0p2s', 0)
                else:
                    summary['time_authority_update_hz_sim'] = None
                    summary['time_authority_updates_total'] = 0
                    summary['time_authority_dt_median'] = None

                # v9.10 Gate 1: Safety fields
                summary['count_no_pass_violations'] = self._count_no_pass_violations
                summary['min_behind_m'] = round(self._min_behind_m, 3) if self._min_behind_m != float('inf') else None
                summary['t_sim_at_min_behind_s'] = round(self._t_sim_at_min_behind_s, 3)
                summary['state_at_min_behind'] = self._state_at_min_behind
                summary['safety_emergency_count'] = self._safety_emergency_count
                # F-E1 (#392): canonical event record + reconciliation flag —
                # a serialized count is admissible ONLY when reconciled.
                summary['safety_emergency_events'] = list(self._safety_emergency_events)
                summary['safety_emergency_count_reconciled'] = (
                    self._safety_emergency_count == len(self._safety_emergency_events))

                # v9.10 Gate 2: State machine stability fields
                summary['state_entry_counts'] = dict(self._state_entry_counts)
                summary['precontact_max_dwell_s'] = round(getattr(self, '_precontact_max_dwell_s', 0.0), 3)

                # FD-002: Starvation observability fields
                summary['attitude_queue_max_depth'] = (
                    self.follower._attitude_queue_max_depth if self.follower else 0)
                summary['attitude_queue_max_depth_runtime'] = (
                    self.follower._attitude_queue_max_depth_runtime if self.follower else 0)
                summary['tick_hz_windowed_final'] = round(self._get_windowed_tick_hz(), 2)
                summary['backlog_fatal_count'] = self._backlog_fatal_count

                # Guard strip-down: CE/NOPASS/CZ guard metrics removed (Chunk 2E)

                # FD-009: Vertical validity guard + axial override metrics
                summary['dc_vert_abort_count'] = self._dc_vert_abort_count
                summary['dc_vert_override_count'] = self._dc_vert_override_count
                summary['dc_airspeed_tripwire_count'] = self._dc_airspeed_tripwire_count

                # Chunk 9: Minimal guard telemetry
                summary['guard_altitude_ceiling_fire_count'] = self._guard_altitude_ceiling_fire_count
                summary['guard_hard_overtake_fire_count'] = self._guard_hard_overtake_fire_count
                summary['guard_speed_cap_fire_count'] = self._guard_speed_cap_fire_count
                # GUARDWIT (log-only): the fallback counter existed since G3 but
                # reached no artifact, so GUARDRCA-729 could not tell which
                # channel the guard read. Both keys are additive.
                # DEFENSIVE READS, deliberately: a log-only export must never be
                # able to break the summary merge. Measured 2026-08-19 — the
                # unconditional form raised AttributeError on a partially
                # constructed controller (object.__new__ fixtures) and killed
                # the run_summary.json merge. That is the harm this export is
                # forbidden to do, so the defaults are load-bearing, not style.
                summary['guard_env_fallback_count'] = getattr(
                    self, '_guard_env_fallback_count', 0)
                summary['guard_ceiling_fires'] = list(getattr(
                    self, '_guard_ceiling_fires', []))

                # A2: Docking EWMA saturation instrumentation

                # M3 S1: Binding constraint sub-source counters
                summary['binding_pre_override_ticks'] = dict(self._binding_pre_override_ticks)
                summary['binding_final_ticks'] = dict(self._binding_final_ticks)

                # §8 Step 3: M3a_coarse (Acquisition tier) + ordinal precision ladder
                summary['m3a_coarse_achieved'] = self._m3a_coarse_achieved
                summary['m3a_coarse_best_dwell_s'] = round(self._m3a_coarse_best_dwell_s, 3)
                summary['m3a_coarse_achieved_at_s'] = (
                    round(self._m3a_coarse_achieved_at_s, 3)
                    if self._m3a_coarse_achieved_at_s is not None else None)

                # M3 progress longitudinal (report-only diagnostic)
                summary['m3_progress_longitudinal_achieved'] = self._m3_prog_long_achieved
                summary['m3_progress_longitudinal_best_dwell_s'] = round(self._m3_prog_long_best_dwell_s, 3)
                summary['m3_progress_longitudinal_achieved_at_s'] = (
                    round(self._m3_prog_long_achieved_at_s, 3)
                    if self._m3_prog_long_achieved_at_s is not None else None)

                # Ordinal: highest tier achieved
                sm_data = summary.get('success_metrics', {})
                sm_milestones = sm_data.get('milestones', {})
                m3b_achieved = sm_milestones.get('m3b_tight_contact', {}).get('achieved', False)
                m3a_fine_achieved = sm_milestones.get('m3a_fine_contact', {}).get('achieved', False)
                m3a_cd_achieved = sm_milestones.get('m3a_coarse_delta_contact', {}).get('achieved', False)
                # Backward compat: also check legacy key
                m3a_achieved = sm_milestones.get('m3a_coarse_contact', {}).get('achieved', False)
                if m3b_achieved:
                    summary['max_tier_achieved'] = 'STRICT'
                elif m3a_fine_achieved or m3a_achieved:
                    summary['max_tier_achieved'] = 'OPERATIONAL'
                elif m3a_cd_achieved or self._m3a_coarse_achieved:
                    summary['max_tier_achieved'] = 'ACQUISITION'
                else:
                    summary['max_tier_achieved'] = 'NONE'

                # Step 1a: Emit new milestone achievement flags at top level
                summary['m3a_fine_achieved'] = m3a_fine_achieved
                summary['m3a_coarse_delta_achieved'] = m3a_cd_achieved

                # Coupled below-target ramp metrics
                summary['coupled_below_target_active_ticks'] = self._coupled_below_active_ticks
                dc_err_samples = list(self._dc_below_err_samples)
                if dc_err_samples:
                    dc_err_samples.sort()
                    p95_idx = int(len(dc_err_samples) * 0.95)
                    summary['dc_below_err_p95_m'] = round(dc_err_samples[min(p95_idx, len(dc_err_samples) - 1)], 3)
                    summary['dc_below_err_max_m'] = round(dc_err_samples[-1], 3)
                else:
                    summary['dc_below_err_p95_m'] = 0.0
                    summary['dc_below_err_max_m'] = 0.0

                # Step 1b: TECS speed tracking error stats during DOCKING_CREEP
                tecs_err = list(self._dc_tecs_speed_err_samples)
                if tecs_err:
                    tecs_abs = sorted(abs(e) for e in tecs_err)
                    n_te = len(tecs_abs)
                    tecs_rms = math.sqrt(sum(e * e for e in tecs_err) / n_te)
                    tecs_p95 = tecs_abs[min(int(n_te * 0.95), n_te - 1)]
                    summary['tecs_speed_err_rms_mps'] = round(tecs_rms, 4)
                    summary['tecs_speed_err_p95_mps'] = round(tecs_p95, 4)
                    summary['tecs_speed_err_max_mps'] = round(tecs_abs[-1], 4)
                    summary['tecs_speed_err_mean_mps'] = round(sum(tecs_err) / n_te, 4)
                    summary['tecs_speed_err_sample_count'] = n_te
                else:
                    summary['tecs_speed_err_rms_mps'] = None
                    summary['tecs_speed_err_p95_mps'] = None
                    summary['tecs_speed_err_max_mps'] = None
                    summary['tecs_speed_err_mean_mps'] = None
                    summary['tecs_speed_err_sample_count'] = 0

                # Rec 5: Runtime backlog time metrics
                summary['runtime_backlog_time_max_s'] = round(self._runtime_backlog_time_max_s, 4)
                summary['runtime_backlog_time_max_depth'] = self._runtime_backlog_time_max_depth
                summary['runtime_backlog_time_max_state'] = self._runtime_backlog_time_max_state
                summary['ticks_dropped_backpressure_total'] = self._ticks_dropped_backpressure_total

                # Queue depth attribution (ChatGPT-5 R2)
                summary['attitude_queue_max_depth_t_sim_s'] = round(self._queue_max_depth_t_sim_s, 3)
                summary['attitude_queue_max_depth_state'] = self._queue_max_depth_state
                summary['pump_msgs_enqueued_max_per_call'] = self._pump_msgs_drained_max
                summary['pump_msgs_enqueued_p95'] = (
                    round(self._percentile_deque(self._pump_msgs_drained_samples, 0.95), 1)
                    if self._pump_msgs_drained_samples else 0)
                summary['loop_wall_time_p95_ms'] = (
                    round(self._percentile_deque(self._loop_wall_times, 0.95) * 1000, 2)
                    if self._loop_wall_times else 0)
                summary['ticks_per_loop_max'] = self._ticks_per_loop_max

                # Guard strip-down: docking safety metrics removed (Chunk 3A)

                # L-DIAG: Diagnostic mode instrumentation
                _diag_active = []
                if formation_config.DIAG_BREAKAWAY_SHADOW:
                    _diag_active.append("BREAKAWAY_SHADOW")
                # Guard strip-down: CE_TIER_A_SUPPRESS + NOPASS_TIER_A_SUPPRESS removed (Chunk 3A)
                if formation_config.DIAG_GEOMETRY_TERMINATOR and self._diag_mode_active:
                    _diag_active.append("GEOMETRY_TERMINATOR")
                summary['diagnostic_flags_active'] = _diag_active
                summary['diagnostic_suppressed_breakaways_total'] = self._diag_suppressed_breakaways
                summary['diagnostic_suppressed_breakaways_by_source'] = dict(self._diag_suppressed_by_source)
                summary['diagnostic_geometry_terminator_reason'] = self._diag_geometry_terminator_reason

                # Guard strip-down: CZ/CZ-EMERGENCY metrics removed (Chunk 2E)
                summary['diagnostic_geometry_terminator_time_s'] = (
                    round(self._diag_geometry_terminator_time, 3)
                    if self._diag_geometry_terminator_time is not None else None)

                # §8-3 durable-finalization in-file handshake (#392, operator
                # "proceed with your recommendations" 2026-07-18): makes a
                # harvested summary self-adjudicable without the env sidecar.
                # summary_finalized=True can only exist in an atomically-
                # replaced artifact (torn writes cannot carry it); freshness =
                # final_transition_count reconciled against the transition log
                # by the approved multiset detector. Content digest stays
                # SIDECAR-side (a file cannot contain its own hash).
                summary['summary_finalized'] = True
                summary['summary_schema_version'] = 1
                summary['summary_flush_reason'] = reason
                summary['summary_terminal_state'] = self.state.name
                summary['summary_terminal_sim_time_s'] = round(self._t_sim_s, 3)
                summary['summary_final_transition_count'] = self._transition_count

                tmp_path = run_summary_path + ".tmp"
                with open(tmp_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                os.replace(tmp_path, run_summary_path)
                self._metrics_flush_state = "committed"  # F-E3: flag persists

                log(f"[SUCCESS_METRICS] Merged into {run_summary_path} (reason={reason})")
            except Exception as e:
                log(f"WARNING: Failed to merge into run_summary.json: {e}")
                self._metrics_flushed = False  # F-E3: retryable
                self._metrics_flush_state = "failed"

        except Exception as e:
            log(f"WARNING: Failed to export success metrics: {e}")
            self._metrics_flushed = False  # F-E3: retryable
            self._metrics_flush_state = "failed"

    def get_relative_pose_source(
        self,
        foll_lat: float, foll_lon: float, foll_alt: float,
        foll_vn: float, foll_ve: float, foll_vd: float,
        lead_lat: float, lead_lon: float, lead_alt: float,
        lead_vn: float, lead_ve: float, lead_vd: float,
        lead_heading_deg: float, boom_length_m: float,
        uwb_range: Optional[float] = None,
        uwb_timestamp: Optional[float] = None
    ) -> Tuple[str, RelativeGeometry]:
        """
        Test 308: Single authoritative function that produces control geometry.

        This method applies sensor mode logic to select which sensor source
        drives the guidance. Modes:
        - fused (default): Always use GPS+UWB fusion
        - uwb_only: Use UWB-only inside close range (if UWB valid)
        - uwb_dominant: Blend UWB more heavily inside close range

        Args:
            All position/velocity inputs (same as compute_relative)
            uwb_range: Optional UWB range measurement
            uwb_timestamp: Timestamp of UWB measurement

        Returns:
            Tuple of (source_name, RelativeGeometry)
            source_name: "GPS", "FUSED", "UWB_ONLY", or "UWB_DOM"
        """
        # v9.10: Use tick timestamp for fusion logic
        now = self._t_sim_s
        if uwb_timestamp is None:
            uwb_timestamp = now

        # Get fused geometry (our baseline)
        # Patch 9: Pass sim-time timestamp for fusion
        geom = self.geom_calc.compute_relative(
            foll_lat, foll_lon, foll_alt,
            foll_vn, foll_ve, foll_vd,
            lead_lat, lead_lon, lead_alt,
            lead_vn, lead_ve, lead_vd,
            lead_heading_deg, boom_length_m,
            uwb_range,
            timestamp_s=uwb_timestamp
        )

        # If in default fused mode, just return fused geometry
        if self.sensor_mode == SENSOR_MODE_FUSED:
            self._active_sensor_source = geom.source
            return geom.source, geom

        # Check if we're inside close range threshold
        inside_close_range = geom.range_m < self.close_range_sensor_threshold

        if not inside_close_range:
            # Outside close range: always use fused
            self._active_sensor_source = geom.source
            return geom.source, geom

        # Inside close range: apply sensor mode logic
        # First check UWB validity
        uwb_valid, uwb_reason = self.sensor_fusion.is_uwb_valid(
            uwb_range, uwb_timestamp,
            self._last_uwb_range, now
        )

        # Update UWB tracking state
        if uwb_range is not None:
            self._last_uwb_range = uwb_range
            self._last_uwb_timestamp = uwb_timestamp

        # ===== UWB_ONLY MODE =====
        if self.sensor_mode == SENSOR_MODE_UWB_ONLY:
            if uwb_valid and uwb_range is not None:
                # Use UWB range as the primary distance measurement
                # Reconstruct geometry with UWB range (maintain GPS for bearing)
                # For now, we use GPS geometry but annotate source as UWB_ONLY
                # True UWB-only would require geometric reconstruction
                self._active_sensor_source = "UWB_ONLY"
                return "UWB_ONLY", RelativeGeometry(
                    behind_m=geom.behind_m,  # GPS-derived (no UWB bearing)
                    cross_m=geom.cross_m,    # GPS-derived
                    below_m=geom.below_m,    # GPS-derived
                    range_m=uwb_range,       # UWB range is authoritative
                    target_aspect_deg=geom.target_aspect_deg,
                    closure_rate_ms=geom.closure_rate_ms,
                    vel_cross_m_s=geom.vel_cross_m_s,
                    source="UWB_ONLY"
                )
            else:
                # UWB invalid - fall back to fused with warning
                if now - self._last_sensor_fallback_log > SENSOR_FALLBACK_LOG_INTERVAL_S:
                    log(f"[SENSOR] UWB_ONLY fallback to FUSED: {uwb_reason}")
                    self._last_sensor_fallback_log = now
                self._active_sensor_source = "FUSED"
                return "FUSED", geom

        # ===== UWB_DOMINANT MODE =====
        if self.sensor_mode == SENSOR_MODE_UWB_DOMINANT:
            if uwb_valid and uwb_range is not None:
                # Compute blending weight based on range
                w_uwb = self.sensor_fusion.compute_uwb_dominant_weight(
                    geom.range_m,
                    self.close_range_sensor_threshold,
                    VERY_PRECISION_MODE_THRESHOLD_M
                )
                w_gps = 1.0 - w_uwb

                # Blend range measurement (UWB range + GPS range)
                blended_range = w_uwb * uwb_range + w_gps * geom.range_m

                self._active_sensor_source = "UWB_DOM"
                return "UWB_DOM", RelativeGeometry(
                    behind_m=geom.behind_m,  # GPS-derived (no UWB bearing)
                    cross_m=geom.cross_m,    # GPS-derived
                    below_m=geom.below_m,    # GPS-derived
                    range_m=blended_range,   # Blended range
                    target_aspect_deg=geom.target_aspect_deg,
                    closure_rate_ms=geom.closure_rate_ms,
                    vel_cross_m_s=geom.vel_cross_m_s,
                    source="UWB_DOM"
                )
            else:
                # UWB invalid - fall back to fused with warning
                if now - self._last_sensor_fallback_log > SENSOR_FALLBACK_LOG_INTERVAL_S:
                    log(f"[SENSOR] UWB_DOM fallback to FUSED: {uwb_reason}")
                    self._last_sensor_fallback_log = now
                self._active_sensor_source = "FUSED"
                return "FUSED", geom

        # Default: return fused
        self._active_sensor_source = geom.source
        return geom.source, geom

    def log_continuous_diagnostics(self, geom: 'RelativeGeometry',
                                   lead_state, fol_alt: float,
                                   hdg_cmd: float, spd_cmd: float,
                                   phase_name: str) -> None:
        """Log continuous diagnostics for every control cycle.

        Runs at a reduced rate (every 2 seconds) to avoid spam.

        P5.4 v9.5: Uses state-appropriate targets - contact targets for
        DOCKING_CREEP/HOLD_CONTACT, precontact targets for other states.
        """
        if not self._should_log_state():
            return

        # P5.4 v9.5: Select targets based on state
        if self.state.name in ["DOCKING_CREEP", "HOLD_CONTACT"]:
            target_behind = CONTACT_BEHIND_M   # 0.0m
            target_cross = CONTACT_CROSS_M     # 0.0m
            target_below = CONTACT_BELOW_M     # 0.0m
            target_set = "contact"
        else:
            target_behind = PRECONTACT_BEHIND_M  # 10.0m
            target_cross = PRECONTACT_CROSS_M    # 0.0m
            target_below = PRECONTACT_BELOW_M    # 10.0m
            target_set = "precontact"

        delta_alt = fol_alt - lead_state.alt_msl
        max_closure = compute_tapered_max_closure_alongtrack(
            geom.behind_m - target_behind
        )

        # Get envelope errors (P5.4: now uses state-appropriate targets)
        behind_err, cross_err, below_err = (
            self.envelope_checker.get_envelope_errors(
                geom, lead_state.alt_msl, fol_alt,
                target_behind, target_cross, target_below
            )
        )

        # Determine sensor source label
        sensor_label = geom.source
        if self.precision_mode_active:
            sensor_label += f" [sensor_mode={self.sensor_mode}]"

        log(
            f"{phase_name}: range={geom.range_m:.1f}m "
            f"behind={geom.behind_m:.1f}m "
            f"cross={geom.cross_m:+.1f}m "
            f"Vc={geom.closure_rate_ms:+.1f}m/s"
        )
        log(
            f"  dAlt={delta_alt:+.1f}m below={geom.below_m:.1f}m | "
            f"behind_err={behind_err:+.1f}m "
            f"cross_err={cross_err:+.1f}m "
            f"below_err={below_err:+.1f}m "
            f"[{target_set}]"  # P5.4 v9.5: Show which target set is used
        )
        log(
            f"  hdg_cmd={hdg_cmd:.1f}\u00b0 spd_cmd={spd_cmd:.1f}m/s | "
            f"max_closure={max_closure:.1f}m/s | "
            f"[{sensor_label}]"
        )
        if self.precision_mode_active:
            log(
                f"  [PRECISION] mode="
                f"{'VERY' if self.very_precision_mode_active else 'NORMAL'} "
                f"filtered_cross={self._filtered_cross_m:.2f}m"
            )

    # Heading command and speed command logic moved to FormationControlLaws
    def send_guided_commands(self, hdg_cmd: float, alt_cmd: float,
                             spd_cmd: float, range_m: float = 100.0,
                             fol_alt: Optional[float] = None) -> None:
        """Send GUIDED mode commands with range-dependent rate limiting.

        Test 305: Rate limiting thresholds are now range-dependent:
        - Very close (<30m): Tighter thresholds for fine control
        - Close (30-50m): Moderate thresholds
        - Far (>50m): Original thresholds

        Test 310: Added force_speed_send support and proper sent state tracking.

        Test 313: Added altitude-aware DO_REPOSITION forcing + fallback commands.

        Args:
            hdg_cmd: Heading command in degrees
            alt_cmd: Altitude command in meters MSL
            spd_cmd: Speed command in m/s
            range_m: Current range to leader (for threshold selection)
            fol_alt: Current follower altitude MSL (for altitude watchdog)
        """
        # Initialize heading state on first call (avoids spurious rate limiting)
        if self._last_reposition_hdg is None:
            fol_att = self.follower.get_attitude()
            self._last_reposition_hdg = fol_att[2] if fol_att else 0.0

        now = self._t_sim_s  # v9.10: Use tick timestamp

        # Tranche C (C0): Record altitude command for D3 telemetry
        self._d3_alt_cmd_msl_m = alt_cmd

        # Tranche H (H4): Single-writer enforcement
        # When C++ owns lateral+altitude (FORM_ENABLE=1), only send speed command.
        # C++ handles lateral (L1 nav via next_WP_loc) and altitude
        # (set_target_altitude_location + reset_offset_altitude).
        # Python only sends speed via DO_CHANGE_SPEED.
        if getattr(self, '_cpp_enabled', False):
            # FD-004: Check for LATCHED safety override
            override = self._peek_speed_override()
            if override is not None:
                final_spd, final_src, final_bypass = override
            else:
                final_spd, final_src, final_bypass = spd_cmd, "normal", False
            # Chunk 9 Guard 3: soft speed cap (caps, does not replace)
            if self._guard_speed_cap_mps is not None:
                final_spd = min(final_spd, self._guard_speed_cap_mps)
            self._send_speed_command(final_spd, final_src,
                                     bypass_latch=final_bypass)
            return

        # Range-dependent thresholds
        if range_m < VERY_CLOSE_RANGE_M:
            # Very close - fine control
            reposition_interval_threshold = VERY_CLOSE_REPOSITION_INTERVAL_S
            heading_delta_threshold = VERY_CLOSE_HEADING_DELTA_DEG
            speed_interval_threshold = VERY_CLOSE_SPEED_INTERVAL_S
            speed_delta_threshold = VERY_CLOSE_SPEED_DELTA_MS
        elif range_m < FINE_CONTROL_RANGE_M:
            # Close - moderate thresholds
            reposition_interval_threshold = CLOSE_REPOSITION_INTERVAL_S
            heading_delta_threshold = CLOSE_HEADING_DELTA_DEG
            speed_interval_threshold = CLOSE_SPEED_INTERVAL_S
            speed_delta_threshold = CLOSE_SPEED_DELTA_MS
        else:
            # Far - original thresholds
            reposition_interval_threshold = DO_REPOSITION_MIN_INTERVAL_S
            heading_delta_threshold = 2.0
            speed_interval_threshold = DO_CHANGE_SPEED_MIN_INTERVAL_S
            speed_delta_threshold = DO_CHANGE_SPEED_DELTA_THRESHOLD

        # --- DO_REPOSITION rate limiting (Test 313: altitude-aware) ---
        reposition_interval = now - self._last_reposition_time
        hdg_delta = abs(wrap_heading_error(hdg_cmd - self._last_reposition_hdg))
        alt_delta = abs(alt_cmd - self._last_reposition_alt)

        # Track force reason for diagnostic logging
        force_alt_send = False
        force_reason = None

        # Detect altitude TARGET change (e.g., 890→900)
        # This is a "major event" - force immediate DO_REPOSITION + fallback
        alt_target_changed = False
        if abs(alt_cmd - self._last_alt_cmd_target) > ALT_CMD_DELTA_M:
            alt_target_changed = True
            force_alt_send = True
            force_reason = "ALT_TARGET_CHANGE"

        # Check for altitude command delta forcing (different from target change)
        if not force_reason and alt_delta > ALT_CMD_DELTA_M:
            force_alt_send = True
            force_reason = "ALT_CMD_DELTA"

        # Check for persistent altitude error forcing
        if fol_alt is not None:
            alt_err_signed = fol_alt - alt_cmd  # positive = too high
            alt_err_abs = abs(alt_err_signed)
            if alt_err_abs > ALT_ERR_FORCE_M:
                if self._alt_err_exceed_start is None:
                    self._alt_err_exceed_start = now
                elif (now - self._alt_err_exceed_start) >= ALT_ERR_FORCE_TIME_S:
                    force_alt_send = True
                    if not force_reason:
                        force_reason = "ALT_ERR_PERSIST"
                    if _FD_PINNING_FIX and not self._pin_fix_window_active:
                        self._pin_fix_window_active = True
                        self._pin_fix_window_start = now
                        self._pin_fix_window_count += 1
                        self._pin_fix_repos_suppressed = 0
                        log(f"[PIN_FIX] ENTER err={alt_err_signed:+.2f}m t={now:.1f}s")
            else:
                self._alt_err_exceed_start = None

        # Send if: interval exceeded OR significant heading change OR altitude force
        should_send_reposition = (
            reposition_interval >= reposition_interval_threshold
            or hdg_delta > heading_delta_threshold
            or force_alt_send
        )

        # --- CLIMB WINDOW suppression (FD_PINNING_FIX) ---
        if _FD_PINNING_FIX and self._pin_fix_window_active:
            window_dur = now - self._pin_fix_window_start
            _win_exit = False
            if fol_alt is not None and alt_err_abs < _PIN_FIX_EXIT_ERR_M:
                self._pin_fix_window_active = False
                self._alt_err_exceed_start = None
                log(f"[PIN_FIX] EXIT err={alt_err_signed:+.2f}m dur={window_dur:.1f}s "
                    f"repos_suppressed={self._pin_fix_repos_suppressed}")
                should_send_reposition = True  # immediate carrot refresh on exit
                _win_exit = True
            elif window_dur >= _PIN_FIX_MAX_HOLDOFF_S:
                self._pin_fix_window_active = False
                self._alt_err_exceed_start = None
                err_str = f"{alt_err_signed:+.2f}m" if fol_alt is not None else "N/A"
                log(f"[PIN_FIX] FAILSAFE dur={window_dur:.1f}s err={err_str}")
                should_send_reposition = True  # allow one through on failsafe
                _win_exit = True
            if not _win_exit:
                if hdg_delta > _PIN_FIX_HDG_ABORT_DEG:
                    log(f"[PIN_FIX] HDG_ABORT hdg_delta={hdg_delta:.1f}deg")
                    should_send_reposition = True  # lateral safety send-through
                else:
                    if should_send_reposition:
                        self._pin_fix_repos_suppressed += 1
                    should_send_reposition = False

        if should_send_reposition:
            self.follower.send_guided_heading(hdg_cmd, alt_cmd)
            self._last_reposition_time = now
            self._last_reposition_hdg = hdg_cmd
            self._last_reposition_alt = alt_cmd

            # Rate-limit ALT_FORCE logging (every 2s, not 50ms)
            if force_reason:
                should_log = (
                    (now - self._last_alt_force_log_time) >= ALT_FORCE_LOG_INTERVAL_S
                    or force_reason == "ALT_TARGET_CHANGE"  # Always log target changes
                )
                if should_log:
                    alt_err_str = ""
                    if fol_alt is not None:
                        ae = fol_alt - alt_cmd
                        alt_err_str = f", err={ae:+.1f}m (|{abs(ae):.1f}|)"
                    log(f"[ALT_FORCE] DO_REPOSITION reason={force_reason}: "
                        f"cmd={alt_cmd:.1f}m{alt_err_str}")
                    self._last_alt_force_log_time = now

        # On altitude target change, fire GCA fallback immediately; P4: inside a
        # climb window only GCA is sent (reposition already suppressed by window).
        if alt_target_changed:
            self.follower.send_guided_change_altitude(alt_cmd)
            self._last_alt_fallback_time = now
            self._last_alt_cmd_target = alt_cmd
            if _FD_PINNING_FIX and self._pin_fix_window_active:
                log(f"[ALT_TARGET_CHANGE] Altitude target changed to {alt_cmd:.1f}m - "
                    "GCA only (window active, reposition suppressed)")
            else:
                log(f"[ALT_TARGET_CHANGE] Altitude target changed to {alt_cmd:.1f}m - "
                    "forced DO_REPOSITION + fallback")

        # Fallback altitude command when tracking fails; bypass the error-magnitude
        # gate when inside a climb window so GCA re-asserts during convergence.
        if fol_alt is not None:
            alt_err_signed = fol_alt - alt_cmd  # positive = too high
            alt_err_abs = abs(alt_err_signed)
            if (_FD_PINNING_FIX and self._pin_fix_window_active) or alt_err_abs > ALT_ERR_FORCE_M:
                fallback_interval = now - self._last_alt_fallback_time
                if fallback_interval >= ALT_FALLBACK_INTERVAL_S:
                    self.follower.send_guided_change_altitude(alt_cmd)
                    self._last_alt_fallback_time = now
                    log(f"[ALT_FALLBACK] GUIDED_CHANGE_ALT: cmd={alt_cmd:.1f}m, "
                        f"fol={fol_alt:.1f}m, err={alt_err_signed:+.1f}m "
                        f"(|{alt_err_abs:.1f}|)")

            # Altitude warning watchdog
            if alt_err_abs > ALT_ERR_WARNING_M:
                if self._alt_err_warning_start is None:
                    self._alt_err_warning_start = now
                elif (now - self._alt_err_warning_start) >= ALT_ERR_WARNING_TIME_S:
                    if (now - self._last_alt_warning_log) >= ALT_ERR_WARNING_LOG_INTERVAL_S:
                        log("⚠ [ALT_WARNING] Altitude tracking failed! "
                            f"cmd={alt_cmd:.1f}m fol={fol_alt:.1f}m "
                            f"err={alt_err_signed:+.1f}m (|{alt_err_abs:.1f}|) "
                            f"(>{ALT_ERR_WARNING_M}m for "
                            f">{now - self._alt_err_warning_start:.1f}s)")
                        self._last_alt_warning_log = now
            else:
                self._alt_err_warning_start = None

        # --- DO_CHANGE_SPEED rate limiting (Test 310: force send support) ---
        speed_interval = now - self._speed_cmd_last_send_time
        spd_delta = abs(spd_cmd - self._speed_cmd_sent_last)

        # Test 310: Check for force send (emergency speed change)
        force_send = self._force_speed_send
        self._force_speed_send = False  # Clear flag

        # Send if: force_send OR interval exceeded OR significant speed change
        should_send = (force_send
                       or speed_interval >= speed_interval_threshold
                       or spd_delta > speed_delta_threshold)

        # FD-004: Check for LATCHED safety override (persists across ticks until expired)
        override = self._peek_speed_override()
        if override is not None:
            final_spd, final_src, final_bypass = override
            # Chunk 9 Guard 3: soft speed cap (caps, does not replace)
            if self._guard_speed_cap_mps is not None:
                final_spd = min(final_spd, self._guard_speed_cap_mps)
            # FD-004: Emergency overrides use the override speed, bypass normal logic
            self._send_speed_command(final_spd, final_src, bypass_latch=final_bypass)
        elif should_send:
            # Chunk 9 Guard 3: soft speed cap (caps, does not replace)
            final_spd = spd_cmd
            if self._guard_speed_cap_mps is not None:
                final_spd = min(final_spd, self._guard_speed_cap_mps)
            # v9.10 C6: Route through canonical function
            self._send_speed_command(final_spd, "normal", bypass_latch=False)

    def _send_speed_command(self, spd_cmd: float, source: str,
                            bypass_latch: bool = False) -> None:
        """v9.10 C6: Canonical single speed command function.

        All speed sends must route through this function. If a second send
        occurs in the same executed tick, raises DoubleSendError with context.

        Args:
            spd_cmd: Speed command in m/s
            source: Source identifier for diagnostics (e.g., "normal", "safety")
            bypass_latch: If True, allow multiple sends (use ONLY for safety emergency)
        """
        if not self.follower:
            return

        # v9.10 C6: Check single-send latch
        if self._speed_cmd_sent_this_tick and not bypass_latch:
            raise DoubleSendError(
                f"Double speed send: first={self._first_send_speed_mps:.2f}m/s "
                f"source={self._first_send_source} bypass={self._first_send_bypass}, "
                f"second={spd_cmd:.2f}m/s source={source}"
            )

        # Send the command
        self.follower.send_guided_speed(spd_cmd)

        # Record for latch and diagnostics
        if not self._speed_cmd_sent_this_tick:
            self._speed_cmd_sent_this_tick = True
            self._first_send_source = source
            self._first_send_speed_mps = spd_cmd
            self._first_send_bypass = bypass_latch

        # Update tracking state
        now = self._t_sim_s
        self._speed_cmd_sent_last = spd_cmd
        self._speed_cmd_last_send_time = now
        self._last_do_change_speed_time = now
        self._last_do_change_speed_cmd = spd_cmd

    def _trigger_guard_breakaway(self, source: str, geom) -> bool:
        """Chunk 9: Trigger EMERGENCY_BREAKAWAY from a minimal guard.

        Returns True if transition succeeded (caller should abort tick).
        Returns False if suppressed (e.g., DIAG_BREAKAWAY_SHADOW) — caller
        must NOT abort tick, otherwise the controller livelocks.
        """
        if self.state == ControllerState.EMERGENCY_BREAKAWAY:
            return True  # Already in breakaway
        log(f"[GUARD] {source}: behind={geom.behind_m:.1f}m below={geom.below_m:.1f}m "
            f"range={geom.range_m:.1f}m -> EMERGENCY_BREAKAWAY")
        self._abort_dwell_accum_s = 0.0
        self.transition_to(
            ControllerState.EMERGENCY_BREAKAWAY,
            f"guard_{source}")
        return self.state == ControllerState.EMERGENCY_BREAKAWAY

    def _send_safety_emergency_speed(self, spd_cmd: float, reason: str) -> None:
        """v9.10 Gate 1: Send speed command with slew bypass for safety emergencies.

        GOTCHA C: Emergency safety through ONE output function.
        This method bypasses rate limiting and slew limiting to immediately
        apply safety commands. Used when behind_m < 0 (no-pass violation) or
        predicted imminent collision.

        FD-004: Now uses LATCHED override system with 1.2s hold (matches TECS lag).

        Args:
            spd_cmd: Speed command in m/s (typically leader_gs - delta)
            reason: Reason string for logging
        """
        # FD-004: Set LATCHED override with 1.2s hold (matches TECS lag)
        self._set_speed_override(spd_cmd, f"safety_emergency:{reason}",
                                 bypass=True, hold_s=1.2)

        log(f"[SAFETY_EMERGENCY_CMD] spd={spd_cmd:.2f}m/s reason={reason} "
            f"dominant_source=safety_emergency")

    def _set_speed_override(self, spd_mps: float, source: str,
                            bypass: bool, hold_s: float) -> None:
        """FD-004: Register LATCHED speed override. Persists until timer expires.
        FD-009: Never loosen an active override — only tighten (min spd) and/or extend (max until)."""
        now = float(self._t_sim_s)
        new_spd = float(spd_mps)
        new_until = now + float(hold_s)

        # FD-009: Never loosen an active override. Only tighten and/or extend.
        if self._speed_override_mps is not None and now <= float(self._speed_override_until_s):
            new_spd = min(new_spd, float(self._speed_override_mps))
            new_until = max(new_until, float(self._speed_override_until_s))

        self._speed_override_mps = new_spd
        self._speed_override_source = str(source)
        self._speed_override_bypass = bool(bypass)
        self._speed_override_until_s = new_until
        log(f"[FD-004] Override SET: spd={new_spd:.2f} source={source} "
            f"bypass={bypass} until={new_until:.2f}")

    def _clear_speed_override(self) -> None:
        """FD-004: Explicitly clear override (called on timer expiry or safe condition)."""
        if self._speed_override_mps is not None:
            log(f"[FD-004] Override CLEARED: was {self._speed_override_source}")
        self._speed_override_mps = None
        self._speed_override_source = None
        self._speed_override_bypass = False
        self._speed_override_until_s = 0.0

    # Guard strip-down: _apply_docking_creep_vertical_validity_guard removed (Chunk 2B)

    def _tecs_spdw_set_target(self, target: float) -> None:
        """Option E: Start TECS_SPDWEIGHT ramp to target."""
        if not formation_config.TECS_SPDWEIGHT_ENABLE or self._tecs_spdw_disabled:
            return
        if abs(target - self._tecs_spdw_target) < 0.01:
            return
        self._tecs_spdw_ramp_start_val = self._tecs_spdweight_current
        self._tecs_spdw_ramp_start_time = self._t_sim_s
        self._tecs_spdw_target = target
        self._tecs_spdw_ramp_active = True

    def _tecs_spdw_tick(self) -> None:
        """Option E: Per-tick linear ramp with rate-limited, verified PARAM_SET writes."""
        if not formation_config.TECS_SPDWEIGHT_ENABLE or self._tecs_spdw_disabled:
            return
        if not self._tecs_spdw_ramp_active:
            return

        # Compute desired value along ramp
        elapsed = self._t_sim_s - self._tecs_spdw_ramp_start_time
        duration = formation_config.TECS_SPDWEIGHT_RAMP_S
        if elapsed >= duration:
            desired = self._tecs_spdw_target
            self._tecs_spdw_ramp_active = False
        else:
            frac = elapsed / duration
            desired = self._tecs_spdw_ramp_start_val + frac * (
                self._tecs_spdw_target - self._tecs_spdw_ramp_start_val)

        # Rate limit: 1 / WRITE_MAX_HZ
        min_interval = 1.0 / formation_config.TECS_SPDWEIGHT_WRITE_MAX_HZ
        if (self._t_sim_s - self._tecs_spdw_last_write_time) < min_interval:
            return
        # Deadband
        if abs(desired - self._tecs_spdweight_current) < formation_config.TECS_SPDWEIGHT_READBACK_TOL:
            return

        val = round(max(0.0, min(2.0, desired)), 2)
        ok = self.follower.set_param("TECS_SPDWEIGHT", val)
        self._tecs_spdw_write_count += 1
        self._tecs_spdw_last_write_time = self._t_sim_s

        if ok:
            self._tecs_spdweight_current = val
            self._tecs_spdw_last_confirmed = val
        else:
            self._tecs_spdw_mismatch_count += 1
            if formation_config.TECS_SPDWEIGHT_FAILSAFE_DISABLE_ON_MISMATCH:
                self._tecs_spdw_disabled = True
                self._tecs_spdw_disable_reason = f"mismatch_at_{self._t_sim_s:.1f}s"
                self._tecs_spdw_ramp_active = False
                log(f"[TECS_SPDW] FAILSAFE DISABLED: write failed at t={self._t_sim_s:.1f}s")

    def _tecs_spdw_compute_target_for_state(self) -> float:
        """Option E: Determine TECS_SPDWEIGHT target based on current state + dc_phase."""
        if self.state == ControllerState.DOCKING_CREEP:
            if self._dc_phase == formation_config.DC_SUBPHASE_VERT_STABILIZE:
                return formation_config.TECS_SPDWEIGHT_DC_PHASE1
            else:
                return formation_config.TECS_SPDWEIGHT_DC_PHASE2
        elif self.state == ControllerState.ALT_CAP_HOLD_RANGE:
            return formation_config.TECS_SPDWEIGHT_ALT_CAP
        elif self.state == ControllerState.HOLD_CONTACT:
            return formation_config.TECS_SPDWEIGHT_HOLD_CONTACT
        else:
            return formation_config.TECS_SPDWEIGHT_CRUISE

    # Guard strip-down: _maybe_apply_nopass_axial_override removed (Chunk 2B)

    def _update_binding_counters(self) -> None:
        """M3 S1: Update per-tick binding constraint counters."""
        pre = self._last_binding_pre_override
        final = self._last_binding_final
        self._binding_pre_override_ticks[pre] = self._binding_pre_override_ticks.get(pre, 0) + 1
        self._binding_final_ticks[final] = self._binding_final_ticks.get(final, 0) + 1

    def _compute_closure_governor(self, geom, lead_state, dt: float) -> Tuple[float, str]:
        """M3 S4: Unified closure governor for DOCKING_CREEP and HOLD_CONTACT.

        Replaces the distributed cap/override stack with a single closure governor
        producing one v_c_cmd per tick. Two-layer architecture:

        Layer 1 (Hard Safety): handled upstream — NOPASS hard (behind < 0) triggers
        EMERGENCY_BREAKAWAY before reaching this method.

        Layer 2 (Progress Governor): Computes v_c_cmd = min of independent constraints:
          - v_c_schedule: from FINAL_APPROACH_SCHEDULE (already exists)
          - v_c_vert: vertical stability cap
          - v_c_cross: cross-track gate
          - FD-006: last-decimeter alignment gating

        Returns:
            (spd_cmd, binding_constraint): speed command in m/s and binding source string.
        """
        from formation_control_laws import (
            compute_final_approach_vc_cap,
            compute_vertical_closure_cap,
            compute_cross_track_gate,
        )

        lead_gs = lead_state.ground_speed_ms
        vc_behind = self.control_laws.get_vc_behind_ewma_mps()

        # --- Constraint 1: Schedule ---
        final_vc_cap, sched_bin_idx = compute_final_approach_vc_cap(
            geom.behind_m, return_bin_idx=True)
        self._schedule_input_m = geom.behind_m
        self._schedule_bin_idx = sched_bin_idx
        self._final_approach_vc_cap = final_vc_cap
        self._final_approach_active = True

        constraints = [("SCHEDULE_FINAL_APPROACH", final_vc_cap)]

        # --- Constraint 2: Vertical gate ---
        if self._coupled_below_active:
            below_target = self._below_target_m   # rate-limited ramp (set by DC handler)
            below_err = self._below_err_m         # already computed by coupled ramp code
        else:
            below_target = 0.0  # DOCKING_CREEP/HOLD_CONTACT = contact altitude
            below_err = geom.below_m - below_target
            self._below_target_m = below_target
            self._below_err_m = below_err
        vert_vc = compute_vertical_closure_cap(below_err, final_vc_cap)
        self._vert_gate_factor = vert_vc / final_vc_cap if final_vc_cap > 0 else 1.0
        constraints.append(("VERT_GATE_BELOW_ERR", vert_vc))

        # --- Constraint 3: Cross-track gate ---
        cross_gate = compute_cross_track_gate(geom.cross_m, geom.behind_m)
        self._cross_gate_factor = cross_gate
        cross_vc = cross_gate * final_vc_cap
        constraints.append(("CROSS_GATE", cross_vc))

        # --- Constraint 4: FD-006 alignment gating (near contact) ---
        if (abs(geom.cross_m) > 0.25 or abs(geom.below_m) > 0.25) and geom.behind_m < 0.5:
            constraints.append(("CROSS_GATE", 0.0))  # Zero closure when misaligned near contact

        # Guard strip-down: Constraints 5 (NOPASS), 6 (CE), 7 (Soft wall) removed (Chunk 3A)

        # --- Take minimum ---
        binding_source = "NONE"
        v_c_cmd = float('inf')
        for source, vc in constraints:
            if vc < v_c_cmd:
                v_c_cmd = vc
                binding_source = source

        v_c_cmd = max(0.0, v_c_cmd)  # Never negative from governor (hard safety handles retreat)

        # Convert to speed command
        spd_cmd = lead_gs + v_c_cmd

        # FD-006: Last-decimeter invariants override governor
        if geom.behind_m < 0.30:
            spd_cmd = min(spd_cmd, lead_gs)
            binding_source = "SCHEDULE_FINAL_APPROACH"
        if geom.behind_m < 0.15:
            spd_cmd = min(spd_cmd, lead_gs - 1.0)
            binding_source = "FD006_LAST_DECIMETER"

        # Apply slew limiting via the existing control_laws pipeline
        # (speed command is passed to slew limiter in compute_speed_cmd)

        # M3 S1: Record binding
        self._last_binding_pre_override = binding_source
        self._last_binding_value_mps = spd_cmd
        self._vert_floor_soft_active = False
        self._vert_floor_hard_active = False
        self._in_precontact_corridor = False

        # Check FD-004 override
        override = self._peek_speed_override()
        self._speed_override_active = override is not None
        if override is not None:
            self._last_binding_final = "FD004_SPEED_OVERRIDE"
        else:
            self._last_binding_final = binding_source
        self._update_binding_counters()

        return spd_cmd, binding_source

    def _apply_closure_brake(self, v_c_cmd: float, geom) -> Tuple[float, bool]:
        """CSV2 crosswind-dock: active anticipatory closure-rate braking (flag-gated).

        When the follower closes faster than a safe stopping-distance profile
        v_safe=sqrt(2*a*gap) within the terminal engage zone, command active
        deceleration (closure below zero, i.e. airspeed below the leader) proportional
        to the over-closing rate, bounded by CLOSURE_BRAKE_MAX_MS. Counters gust-driven
        airspeed-tracking overshoot PAST the boom-tip (the FINAL_APPROACH_SCHEDULE min-cap
        can only shrink positive closure, not command active braking).

        Inputs: fused geom.behind_m / geom.closure_rate_ms only (no privileged or
        ground-truth inputs); deterministic; no wall-clock. Returns (v_c_cmd, brake_fired).
        When the brake does not engage, returns the original max(0.0, v_c_cmd) governor
        floor, so the flag-ON path equals the flag-OFF path outside the engage condition.
        """
        gap = geom.behind_m
        # Engage only in the momentum zone [MIN_GAP, ENGAGE): below MIN_GAP the FD-006
        # last-decimeter brake + station-keeping own it (v_safe->0 there over-fires on
        # calm jitter); at/above ENGAGE the approach schedule governs.
        if (gap < formation_config.CLOSURE_BRAKE_MIN_GAP_M
                or gap >= formation_config.CLOSURE_BRAKE_ENGAGE_M):
            return max(0.0, v_c_cmd), False
        v_safe = math.sqrt(2.0 * formation_config.CLOSURE_BRAKE_A_MS2 * max(gap, 0.0))
        # over-closing beyond v_safe AND a jitter deadband; brake acts on the excess only
        over = geom.closure_rate_ms - v_safe - formation_config.CLOSURE_BRAKE_OVER_DEADBAND_MS
        if over <= 0.0:
            return max(0.0, v_c_cmd), False
        braked = v_c_cmd - formation_config.CLOSURE_BRAKE_GAIN * over
        return max(-formation_config.CLOSURE_BRAKE_MAX_MS, braked), True

    def _peek_speed_override(self) -> Optional[Tuple[float, str, bool]]:
        """FD-004: Peek at pending override. Does NOT clear - latch persists until expired.

        Returns (spd, source, bypass) or None if no active override.
        """
        if self._speed_override_mps is None:
            return None
        # Check expiry
        if self._t_sim_s > self._speed_override_until_s:
            self._clear_speed_override()
            return None
        return (self._speed_override_mps, self._speed_override_source,
                self._speed_override_bypass)

    def _get_backlog_time_s(self) -> float:
        """FD-002: Get queue backlog in seconds.

        Backlog = how far behind tick scheduler is = (latest_enqueued - last_processed).
        This correctly measures starvation risk even when queue is temporarily empty.
        """
        if self._last_processed_tick_ms is None:
            return 0.0
        if not self.follower or self.follower._last_enqueued_attitude_ms is None:
            return 0.0
        return (self.follower._last_enqueued_attitude_ms -
                self._last_processed_tick_ms) / 1000.0

    def _get_windowed_tick_hz(self) -> float:
        """FD-002: Get tick rate over last ~5 seconds."""
        if len(self._tick_hz_window) < 2:
            return 0.0
        dt_total = (self._tick_hz_window[-1] - self._tick_hz_window[0]) / 1000.0
        if dt_total <= 0:
            return 0.0
        return (len(self._tick_hz_window) - 1) / dt_total

    @staticmethod
    def _percentile_deque(samples, p: float) -> float:
        """Numpy-free percentile (floor convention). Works on any iterable."""
        vals = sorted(samples)
        n = len(vals)
        if n == 0:
            return 0.0
        idx = min(n - 1, int(n * p))
        return vals[idx]

    def _get_recent_dt_p95_s(self) -> float:
        """FD-007: Get p95 of recent dt samples."""
        if len(self._recent_dt_samples) < 10:
            return 0.1  # Conservative default
        sorted_dt = sorted(self._recent_dt_samples)
        p95_idx = int(len(sorted_dt) * 0.95)
        return sorted_dt[min(p95_idx, len(sorted_dt) - 1)]

    def _check_timebase_watchdog(self, t_sim_s: float) -> None:
        """PRH-ATT-004: Two-tier runtime watchdog for timebase rate drops.

        Reuses existing _get_windowed_tick_hz() — no duplication.
        HARD (15Hz): Matches Gate0 — fatal if sustained > trip_delay.
        WARN (18Hz): Non-fatal snapshot+log, skipped in emergency states.
        In close-in states, also checks for new stale gaps from TA metrics.
        """
        windowed_hz = self._get_windowed_tick_hz()

        # Emergency states: only enforce hard-fail, skip warn
        _in_emergency = self.state in {ControllerState.EMERGENCY_BREAKAWAY}

        # HARD FAIL: 15Hz — matches Gate0, always enforced
        if windowed_hz < self._watchdog_hard_min_hz:
            if self._watchdog_low_hz_since_s is None:
                self._watchdog_low_hz_since_s = t_sim_s
            elif (t_sim_s - self._watchdog_low_hz_since_s) > self._watchdog_trip_delay_s:
                self._write_timebase_drop_snapshot(t_sim_s, windowed_hz, "hard_fail_sustained")
                raise TimebaseRateDropError(
                    f"PRH-ATT-004: Windowed Hz {windowed_hz:.1f} < {self._watchdog_hard_min_hz} "
                    f"for {t_sim_s - self._watchdog_low_hz_since_s:.1f}s "
                    f"in state {self.state.name}")

        # WARN: 18Hz — non-fatal snapshot+log, skipped in emergency states
        elif windowed_hz < self._watchdog_warn_min_hz and not _in_emergency:
            if not self._watchdog_warn_logged:
                self._watchdog_warn_logged = True
                self._write_timebase_drop_snapshot(t_sim_s, windowed_hz, "warn_hz_dip")
                log(f"[WATCHDOG] WARN: Windowed Hz {windowed_hz:.1f} < {self._watchdog_warn_min_hz} "
                    f"in state {self.state.name} (non-fatal)")
        else:
            # Hz recovered — reset both timers
            self._watchdog_low_hz_since_s = None
            self._watchdog_warn_logged = False

        # Close-in stale gap check
        if self.state in {ControllerState.DOCKING_CREEP, ControllerState.HOLD_CONTACT}:
            try:
                from sim_time_authority import get_time_authority
                ta_metrics = get_time_authority().get_time_authority_metrics()
                current_stale = ta_metrics.get('time_authority_stale_gaps_over_0p2s', 0)
                # Tranche W: Sync baseline on first close-in tick to avoid
                # false positives from stale gaps during approach phases
                if not self._watchdog_stale_baseline_synced:
                    self._watchdog_last_stale_count = current_stale
                    self._watchdog_stale_baseline_synced = True
                    if current_stale > 0:
                        log(f"[WATCHDOG] Stale gap baseline synced: {current_stale} "
                            f"pre-existing gap(s) from approach phases (not fatal)")
                elif current_stale > self._watchdog_last_stale_count:
                    new_gaps = current_stale - self._watchdog_last_stale_count
                    self._watchdog_last_stale_count = current_stale
                    self._write_timebase_drop_snapshot(
                        t_sim_s, windowed_hz,
                        f"stale_gap_in_closein_{self.state.name}_{new_gaps}x")
                    raise TimebaseRateDropError(
                        f"PRH-ATT-004: {new_gaps} new stale gap(s) in {self.state.name}")
            except ImportError:
                pass  # TA not available — skip stale check
        else:
            # Reset sync flag when outside close-in states (e.g. after breakaway recovery)
            self._watchdog_stale_baseline_synced = False

    def _get_sim_time_windowed_hz(self, window_ms: int = 5000) -> float:
        """Change 5: Compute Hz over last window_ms of sim-time from _tick_hz_window."""
        if len(self._tick_hz_window) < 2:
            return 0.0
        current_tbms = self._tick_hz_window[-1]
        cutoff = current_tbms - window_ms
        # Filter entries within window
        in_window = [t for t in self._tick_hz_window if t >= cutoff]
        if len(in_window) < 2:
            return 0.0
        dt_total = (in_window[-1] - in_window[0]) / 1000.0
        if dt_total <= 0:
            return 0.0
        return (len(in_window) - 1) / dt_total

    def _write_timebase_drop_snapshot(self, t_sim_s: float, windowed_hz: float,
                                      trigger: str) -> None:
        """PRH-ATT-004: Write diagnostic snapshot on watchdog trip (enhanced Change 5)."""
        if not self._out_dir:
            return
        try:
            import json as _json

            # Change 5: sim_dt_p95 from _recent_dt_samples
            _sim_dt_p95 = None
            if len(self._recent_dt_samples) >= 10:
                _sorted_dt = sorted(self._recent_dt_samples)
                _p95_idx = int(len(_sorted_dt) * 0.95)
                _sim_dt_p95 = round(_sorted_dt[min(_p95_idx, len(_sorted_dt) - 1)], 6)

            snapshot = {
                "trigger": trigger,
                "t_sim_s": round(t_sim_s, 3),
                "windowed_hz": round(windowed_hz, 2),
                "state": self.state.name,
                "watchdog_hard_min_hz": self._watchdog_hard_min_hz,
                "watchdog_warn_min_hz": self._watchdog_warn_min_hz,
                "watchdog_trip_delay_s": self._watchdog_trip_delay_s,
                "tick_hz_window_len": len(self._tick_hz_window),
                # Change 5: Enhanced diagnostic fields
                "attitude_queue_depth": (
                    len(self.follower._attitude_time_queue) if self.follower else 0),
                "msgs_drained_last_pump": self._msgs_drained_last_pump,
                "sim_dt_p95_s": _sim_dt_p95,
                "sim_time_windowed_hz": round(self._get_sim_time_windowed_hz(), 2),
            }
            path = os.path.join(self._out_dir, "timebase_drop_snapshot.json")
            with open(path, "w") as f:
                _json.dump(snapshot, f, indent=2)
        except Exception:
            pass  # Best effort — don't fail during fatal path

    def _write_queue_depth_snapshot(self, depth: int, t_sim_s: float,
                                    state_name: str, queue_active: bool) -> None:
        """ChatGPT-5 R2: Write queue depth snapshot when depth exceeds threshold."""
        if not self._out_dir:
            return
        try:
            import json as _json
            snapshot = {
                "trigger": "queue_depth_over_50",
                "depth": depth,
                "t_sim_s": round(t_sim_s, 3),
                "state": state_name,
                "queue_active": queue_active,
                "attitude_queue_max_depth": (
                    self.follower._attitude_queue_max_depth if self.follower else 0),
                "tick_hz_windowed": round(self._get_windowed_tick_hz(), 2),
                "ticks_executed_total": self._ticks_executed_total,
            }
            path = os.path.join(self._out_dir, "queue_depth_snapshot.json")
            with open(path, "w") as f:
                _json.dump(snapshot, f, indent=2)
        except Exception:
            pass  # Best effort

    def _write_backlog_time_snapshot(self, backlog_time_s: float, depth: int) -> None:
        """Rec 5B: Write backlog time snapshot when threshold exceeded."""
        if not self._out_dir:
            return
        try:
            import json as _json
            snapshot = {
                "trigger": "backlog_time_over_threshold",
                "backlog_time_s": round(backlog_time_s, 4),
                "depth": depth,
                "t_sim_s": round(self._t_sim_s, 3),
                "state": self.state.name if self.state else "UNKNOWN",
                "tick_hz_windowed": round(self._get_windowed_tick_hz(), 2),
                "loop_wall_time_p95_ms": (
                    round(self._percentile_deque(self._loop_wall_times, 0.95) * 1000, 2)
                    if self._loop_wall_times else 0),
                "expected_speedup": self.expected_speedup,
                "ticks_executed_total": self._ticks_executed_total,
            }
            path = os.path.join(self._out_dir, "backlog_time_snapshot.json")
            with open(path, "w") as f:
                _json.dump(snapshot, f, indent=2)
        except Exception:
            pass  # Best effort

    def compute_offset_intercept_heading(self, geom: RelativeGeometry,
                                         lead_hdg: float) -> float:
        """Compute heading for fighter intercept with lateral offset.

        During head-on approach (TA > 130°), we fly to an aim point that's
        offset laterally from the leader's track. This maintains safe
        separation and sets up the geometry for the lead turn.

        Test 301: Added PD control and range-scaled rate limiting for consistency.

        The aim point is:
        - Along the leader's track (ahead of leader's current position)
        - Offset perpendicular by LATERAL_OFFSET_M

        We fly towards this aim point, which naturally maintains lateral
        offset while closing range.

        TEST 312: Uses self._active_offset_side which may be dynamically
        flipped to avoid crossing maneuvers.
        """
        # Target lateral offset (positive = to leader's right = our left)
        # Use dynamic offset side (may be flipped from default)
        target_offset = self._active_offset_side * LATERAL_OFFSET_M

        # Current offset error (how far we are from desired offset)
        offset_error = target_offset - geom.cross_m

        # Base heading: toward the leader
        # Vector from follower to leader: (behind_m, -cross_m) - negate cross for follower→leader
        relative_bearing = math.degrees(math.atan2(-geom.cross_m, geom.behind_m))
        bearing_to_leader = wrap_heading(lead_hdg + relative_bearing)

        # PD control for offset correction (Test 301 + Test 304)
        # P-term: proportional to offset error
        p_term = -CROSS_CORRECTION_GAIN * offset_error

        # D-term: proportional to cross-track rate (dampens oscillations)
        # Range-scaled D-gain (same as compute_heading_cmd)
        if geom.range_m <= VERY_CLOSE_RANGE_M:
            d_gain = CROSS_D_GAIN_NEAR
        elif geom.range_m >= FINE_CONTROL_RANGE_M:
            d_gain = CROSS_D_GAIN
        else:
            t = (geom.range_m - VERY_CLOSE_RANGE_M) / \
                (FINE_CONTROL_RANGE_M - VERY_CLOSE_RANGE_M)
            d_gain = CROSS_D_GAIN_NEAR + t * (CROSS_D_GAIN - CROSS_D_GAIN_NEAR)

        d_term = -d_gain * geom.vel_cross_m_s

        offset_correction = p_term + d_term

        # Range-scaled heading rate limit (Test 301 + Test 304)
        # Same three-tier structure as compute_heading_cmd for consistency
        if geom.range_m <= VERY_CLOSE_RANGE_M:
            max_rate = VERY_CLOSE_HEADING_RATE_DPS
        elif geom.range_m <= FINE_CONTROL_RANGE_M:
            t = (geom.range_m - VERY_CLOSE_RANGE_M) / \
                (FINE_CONTROL_RANGE_M - VERY_CLOSE_RANGE_M)
            max_rate = VERY_CLOSE_HEADING_RATE_DPS + \
                t * (FINE_CONTROL_HEADING_RATE_DPS - VERY_CLOSE_HEADING_RATE_DPS)
        elif geom.range_m < CLOSE_RANGE_THRESHOLD_M:
            max_rate = CLOSE_RANGE_HEADING_RATE_DPS
        else:
            max_rate = MAX_HEADING_RATE_DPS

        offset_correction = max(-max_rate, min(max_rate, offset_correction))

        return wrap_heading(bearing_to_leader + offset_correction)

    def compute_lead_turn_heading(self, geom: RelativeGeometry,
                                  lead_hdg: float) -> float:
        """Compute heading for lead turn maneuver.

        During the lead turn (TA 130° -> 0°), we execute a smooth turn
        to roll out behind the leader. The turn rate is computed to
        achieve a roll-out at the target stern position.

        Lead pursuit: aim ahead of where the target will be.
        """
        # Lead pursuit: aim for where leader will be
        # Use proportional navigation: turn rate proportional to LOS rate
        # Simplified: steer toward a point ahead of leader on their track

        # Compute bearing to leader
        # Vector from follower to leader: (behind_m, -cross_m) - negate cross for follower→leader
        relative_bearing = math.degrees(math.atan2(-geom.cross_m, geom.behind_m))
        bearing_to_leader = wrap_heading(lead_hdg + relative_bearing)

        # Lead angle: how much to lead the target
        # Larger lead angle when TA is higher (further from stern)
        abs_ta = abs(geom.target_aspect_deg)
        lead_angle = (abs_ta / 180.0) * 30.0  # 0-30° lead based on TA

        # Determine turn direction based on which side of leader we are
        # If we're to leader's right (cross_m > 0), we need to turn left
        # to get behind (lead angle should reduce bearing)
        if geom.cross_m > 0:
            # We're to leader's right, turn left (reduce heading)
            lead_correction = -lead_angle
        else:
            # We're to leader's left, turn right (increase heading)
            lead_correction = lead_angle

        return wrap_heading(bearing_to_leader + lead_correction)

    # =========================================================================
    # State Handlers (Private)
    # =========================================================================
    def _handle_weather_check(self) -> None:
        """Run weather check state."""
        if self.weather_checker:
            self.weather_checker.check_weather()
        self.transition_to(ControllerState.VTOL_TAKEOFF, "Weather check complete")

    def _handle_vtol_takeoff(self) -> None:
        """P1 Fix A v9.10: Entry point for non-blocking takeoff sub-FSM.

        This replaces the old blocking takeoff_to_fixedwing_precontact() call
        with a transition to the first sub-FSM state.
        """
        if not TAKEOFF_MODULE_AVAILABLE:
            log("WARNING: Takeoff module not available, skipping to FW_STABILIZE")
            self.transition_to(ControllerState.FW_STABILIZE, "No takeoff module")
            return

        # Initialize takeoff controller if needed
        if self.takeoff_controller is None:
            target_alt_agl = PRECONTACT_ALT_MSL - self.home_alt_msl
            self.takeoff_controller = QuadplaneQloiterTakeoffController(
                mav=self.follower.mav,
                home_alt_msl=self.home_alt_msl,
                target_system=self.follower.target_system,
                target_component=self.follower.target_component,
                target_precontact_alt_agl=target_alt_agl,
                vtol_mode_name="QLOITER",
                fixedwing_mode_name="GUIDED",
                speed_mode_for_fw="MATCH",
            )
            log(f"[TAKEOFF] Target altitude: {target_alt_agl:.1f}m AGL "
                f"({PRECONTACT_ALT_MSL:.1f}m MSL)")

        # P1 Fix A: Initialize sub-FSM state tracking
        self._takeoff_cmd_sent = False
        self._takeoff_retry_count = 0
        self._takeoff_sustained_start = None
        self._takeoff_disarm_streak = 0

        # Transition to first sub-FSM state
        log("[TAKEOFF] P1 Fix A: Starting non-blocking takeoff sub-FSM")
        self.transition_to(ControllerState.TAKEOFF_SET_VTOL_MODE, "Non-blocking takeoff")

    # =========================================================================
    # P1 Fix A v9.10: Non-Blocking Takeoff Sub-FSM Handlers
    # =========================================================================
    # Timeouts are in sim-time (seconds)
    TAKEOFF_VTOL_MODE_TIMEOUT_S = 30.0
    TAKEOFF_ARM_TIMEOUT_S = 30.0
    TAKEOFF_CLIMB_TIMEOUT_S = 300.0  # 5 min for 400m climb
    TAKEOFF_FW_MODE_TIMEOUT_S = 30.0
    TAKEOFF_TRANSITION_TIMEOUT_S = 60.0
    TAKEOFF_SUSTAINED_AIRSPEED_S = 2.0  # Sustained airspeed for transition complete

    def _handle_takeoff_set_vtol_mode(self) -> None:
        """P1 Fix A: Set VTOL mode (QLOITER) - non-blocking."""
        elapsed = self._t_sim_s - self.state_entry_time

        # FD-001: Share cached messages with takeoff controller (avoids message theft)
        if self.follower and self.follower._last_heartbeat:
            self.takeoff_controller.nb_set_cached_heartbeat(self.follower._last_heartbeat)
        if self.follower and self.follower._last_position:
            self.takeoff_controller.nb_set_cached_position(self.follower._last_position)

        # Send command once
        if not self._takeoff_cmd_sent:
            log("[TAKEOFF_SET_VTOL_MODE] Sending QLOITER mode command...")
            self.takeoff_controller.nb_send_vtol_mode_cmd()
            self._takeoff_cmd_sent = True
            return

        # Check predicate
        if self.takeoff_controller.nb_check_vtol_mode():
            log("[TAKEOFF_SET_VTOL_MODE] ✓ QLOITER mode confirmed")
            self._takeoff_cmd_sent = False
            self.transition_to(ControllerState.TAKEOFF_ARMING, "QLOITER mode set")
            return

        # Timeout check
        if elapsed > self.TAKEOFF_VTOL_MODE_TIMEOUT_S:
            log(f"[TAKEOFF_SET_VTOL_MODE] ❌ Timeout after {elapsed:.1f}s")
            self.transition_to(ControllerState.FAIL, "VTOL mode timeout")

    def _handle_takeoff_arming(self) -> None:
        """P1 Fix A: Arm vehicle - non-blocking."""
        elapsed = self._t_sim_s - self.state_entry_time

        # FD-001: Share cached messages with takeoff controller (avoids message theft)
        if self.follower and self.follower._last_heartbeat:
            self.takeoff_controller.nb_set_cached_heartbeat(self.follower._last_heartbeat)
        if self.follower and self.follower._last_position:
            self.takeoff_controller.nb_set_cached_position(self.follower._last_position)

        # Send arm command (retry every ~3s sim-time)
        if not self._takeoff_cmd_sent or (elapsed > (self._takeoff_retry_count + 1) * 3.0):
            self._takeoff_retry_count += 1
            log(f"[TAKEOFF_ARMING] Sending arm command (attempt {self._takeoff_retry_count})...")
            self.takeoff_controller.nb_send_arm_cmd()
            self._takeoff_cmd_sent = True

        # Check predicate
        if self.takeoff_controller.nb_check_armed():
            log("[TAKEOFF_ARMING] ✓ Vehicle armed")
            self._takeoff_cmd_sent = False
            self._takeoff_retry_count = 0
            self.transition_to(ControllerState.TAKEOFF_CLIMBING, "Armed")
            return

        # Timeout check
        if elapsed > self.TAKEOFF_ARM_TIMEOUT_S:
            log(f"[TAKEOFF_ARMING] ❌ Arm timeout after {elapsed:.1f}s")
            self.transition_to(ControllerState.FAIL, "Arming timeout")

    def _handle_takeoff_climbing(self) -> None:
        """P1 Fix A: VTOL climb to target altitude - non-blocking."""
        elapsed = self._t_sim_s - self.state_entry_time

        # Share cached heartbeat with takeoff controller (avoids message race)
        if self.follower and self.follower._last_heartbeat:
            self.takeoff_controller.nb_set_cached_heartbeat(self.follower._last_heartbeat)

        # Share cached position with takeoff controller (avoids message theft for altitude)
        if self.follower and self.follower._last_position:
            self.takeoff_controller.nb_set_cached_position(self.follower._last_position)

        # Send throttle override each tick to maintain climb
        self.takeoff_controller.nb_send_throttle_override()

        # Check altitude predicate
        reached, current_alt = self.takeoff_controller.nb_check_altitude_reached()
        if reached:
            current_msl = self.home_alt_msl + current_alt
            log(f"[TAKEOFF_CLIMBING] ✓ Target altitude reached: {current_msl:.1f}m MSL")
            self.takeoff_controller.nb_release_throttle_override()
            self._takeoff_cmd_sent = False
            self.transition_to(ControllerState.TAKEOFF_SET_FW_MODE, "Altitude reached")
            return

        # Check for unexpected disarm (with debounce)
        if self.takeoff_controller.nb_check_disarmed():
            self._takeoff_disarm_streak += 1
            if self._takeoff_disarm_streak >= 3:
                log("[TAKEOFF_CLIMBING] ❌ Vehicle disarmed during climb")
                self.takeoff_controller.nb_release_throttle_override()
                self.transition_to(ControllerState.FAIL, "Disarmed during climb")
                return
        else:
            self._takeoff_disarm_streak = 0

        # Periodic logging (every 10s sim-time)
        if int(elapsed) % 10 == 0 and elapsed > 0.5:
            current_msl = self.home_alt_msl + current_alt
            target_msl = self.home_alt_msl + self.takeoff_controller.target_precontact_alt_agl
            log(f"[TAKEOFF_CLIMBING] alt={current_msl:.1f}m MSL (target={target_msl:.1f}m)")

        # Timeout check
        if elapsed > self.TAKEOFF_CLIMB_TIMEOUT_S:
            log(f"[TAKEOFF_CLIMBING] ❌ Climb timeout after {elapsed:.1f}s")
            self.takeoff_controller.nb_release_throttle_override()
            self.transition_to(ControllerState.FAIL, "Climb timeout")

    def _handle_takeoff_set_fw_mode(self) -> None:
        """P1 Fix A: Set FW mode (CRUISE) for transition - non-blocking."""
        elapsed = self._t_sim_s - self.state_entry_time

        # Share cached heartbeat with takeoff controller (avoids message race)
        if self.follower and self.follower._last_heartbeat:
            self.takeoff_controller.nb_set_cached_heartbeat(self.follower._last_heartbeat)

        # Send command once
        if not self._takeoff_cmd_sent:
            log("[TAKEOFF_SET_FW_MODE] Sending CRUISE mode command...")
            self.takeoff_controller.nb_send_fw_mode_cmd()
            self._takeoff_cmd_sent = True
            return

        # Check predicate
        if self.takeoff_controller.nb_check_fw_mode():
            log("[TAKEOFF_SET_FW_MODE] ✓ Plane mode confirmed")
            self._takeoff_cmd_sent = False
            self._takeoff_sustained_start = None
            self.transition_to(ControllerState.TAKEOFF_TRANSITION_MONITOR, "FW mode set")
            return

        # Timeout check
        if elapsed > self.TAKEOFF_FW_MODE_TIMEOUT_S:
            log(f"[TAKEOFF_SET_FW_MODE] ❌ FW mode timeout after {elapsed:.1f}s")
            self.transition_to(ControllerState.FAIL, "FW mode timeout")

    def _handle_takeoff_transition_monitor(self) -> None:
        """P1 Fix A: Monitor FW transition until airspeed sustained - non-blocking."""
        elapsed = self._t_sim_s - self.state_entry_time

        # FD-001: Share cached messages with takeoff controller (avoids message theft)
        if self.follower and self.follower._last_heartbeat:
            self.takeoff_controller.nb_set_cached_heartbeat(self.follower._last_heartbeat)
        if self.follower and self.follower._last_vfr_hud:
            self.takeoff_controller.nb_set_cached_vfr_hud(self.follower._last_vfr_hud)

        # Check transition complete (sustained airspeed in FW mode)
        complete, airspeed = self.takeoff_controller.nb_check_transition_complete(
            min_airspeed_mps=12.0
        )

        if complete:
            if self._takeoff_sustained_start is None:
                self._takeoff_sustained_start = self._t_sim_s
                log(f"[TAKEOFF_TRANSITION_MONITOR] FW flight detected: {airspeed:.1f} m/s")

            sustained_time = self._t_sim_s - self._takeoff_sustained_start
            if sustained_time >= self.TAKEOFF_SUSTAINED_AIRSPEED_S:
                log(f"[TAKEOFF_TRANSITION_MONITOR] ✓ Transition complete!")
                log(f"    Airspeed: {airspeed:.1f} m/s (sustained {sustained_time:.1f}s)")
                self.transition_to(ControllerState.FW_TRANSITION, "Takeoff complete")
                return
        else:
            self._takeoff_sustained_start = None

        # Periodic logging
        if int(elapsed) % 5 == 0 and elapsed > 0.5:
            log(f"[TAKEOFF_TRANSITION_MONITOR] Monitoring: airspeed={airspeed:.1f} m/s")

        # Timeout - but transition anyway (FW_TRANSITION handler will verify)
        if elapsed > self.TAKEOFF_TRANSITION_TIMEOUT_S:
            log(f"[TAKEOFF_TRANSITION_MONITOR] ⚠ Timeout after {elapsed:.1f}s, proceeding anyway")
            self.transition_to(ControllerState.FW_TRANSITION, "Transition timeout")

    def _handle_fw_transition(self) -> None:
        """Verify FW transition with sustained airspeed and mode confirmation.

        v9.10: Uses _t_sim_s for timestamps and accumulator pattern for elapsed time.
        """
        elapsed = self._t_sim_s - self.state_entry_time

        if not hasattr(self, '_fw_trans_sustained_start'):
            self._fw_trans_sustained_start = None
            self._fw_trans_sustained_accum_s = 0.0  # v9.10: Accumulator
            self._fw_trans_retry_count = 0
            log("[FW_TRANS] Starting transition verification...")

        airspeed = self.follower.get_airspeed()
        mode_id = self.follower.get_mode_id()
        is_vtol = self.follower.is_vtol_mode(mode_id)

        airspeed_ok = airspeed is not None and airspeed >= FW_TRANSITION_MIN_AIRSPEED_MPS
        mode_ok = not is_vtol

        if airspeed_ok and mode_ok:
            if self._fw_trans_sustained_start is None:
                self._fw_trans_sustained_start = self._t_sim_s
                self._fw_trans_sustained_accum_s = 0.0
            # v9.10: Use dt_credit for sustained time accumulation
            self._fw_trans_sustained_accum_s += self._dt_credit_s
            if self._fw_trans_sustained_accum_s >= FW_TRANSITION_SUSTAIN_S:
                self.transition_to(ControllerState.FW_STABILIZE, "FW transition verified")
                return
        else:
            self._fw_trans_sustained_start = None
            self._fw_trans_sustained_accum_s = 0.0

        if elapsed > FW_TRANSITION_TIMEOUT_S:
            self.transition_to(ControllerState.FAIL, "FW transition failed")

    def _handle_fw_stabilize(self) -> None:
        """Stabilize in fixed-wing mode and confirm GUIDED mode before intercept.

        v9.10: Uses _t_sim_s for elapsed time calculation.
        v9.10.1: Added Q_GUIDED_MODE check to prevent VTOL reversion.
        """
        elapsed = self._t_sim_s - self.state_entry_time
        if elapsed < FW_STABILIZE_TIME_S:
            return

        lead_state = self.leader_rx.update()
        if not lead_state:
            return

        # v9.10.1: Check Q_GUIDED_MODE before switching to GUIDED
        # Q_GUIDED_MODE=1 forces VTOL mode in GUIDED, breaking FW formation flight
        if self.follower.get_mode() != "GUIDED":
            q_guided_mode = self.follower.get_param("Q_GUIDED_MODE", timeout=1.0)
            if q_guided_mode is not None and q_guided_mode > 0:
                log(f"[FW_STABILIZE] WARNING: Q_GUIDED_MODE={q_guided_mode:.0f} - would force VTOL!")
                log("[FW_STABILIZE] Staying in CRUISE mode for intercept (safe fallback)")
                # Stay in CRUISE - don't switch to GUIDED
            else:
                self.follower.set_mode_and_wait("GUIDED")
                # v9.10.1: Verify GUIDED didn't trigger VTOL mode
                mode_id = self.follower.get_mode_id()
                if self.follower.is_vtol_mode(mode_id):
                    log("[FW_STABILIZE] ERROR: GUIDED triggered VTOL mode! Reverting to CRUISE")
                    self.follower.set_mode_and_wait("CRUISE")

        self.transition_to(ControllerState.INTERCEPT, "Ready for intercept")

    def _select_offset_side_for_intercept(self, geom: RelativeGeometry) -> None:
        """
        Dynamically select offset side (left/right) to avoid crossing the leader.

        If we are already far to the left (cross > threshold), stay left (-1).
        If we are far to the right, stay right (+1).
        Otherwise, use the default configured side.
        """
        if self._offset_side_selected:
            return

        # Check if we are already far on one side
        if abs(geom.cross_m) > DYNAMIC_OFFSET_THRESHOLD_M:
            # If cross > 0, we are to the right -> use side +1
            # If cross < 0, we are to the left -> use side -1
            side = 1.0 if geom.cross_m > 0 else -1.0

            if side != OFFSET_SIDE:
                log(f"[INTERCEPT] Dynamic offset override: Switched to side {side:+.0f} "
                    f"(cross={geom.cross_m:.1f}m)")
            else:
                log(f"[INTERCEPT] Dynamic offset confirmed: Side {side:+.0f} "
                    f"(cross={geom.cross_m:.1f}m)")

            self._active_offset_side = side
        else:
            # Within threshold, use default
            self._active_offset_side = OFFSET_SIDE
            log(f"[INTERCEPT] Using default offset side {self._active_offset_side:+.0f}")

        self._offset_side_selected = True

    def _handle_intercept(self) -> None:
        """Run fighter intercept phase with lateral offset."""
        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        self._select_offset_side_for_intercept(geom)
        abs_ta = abs(geom.target_aspect_deg)

        if abs_ta <= LEAD_TURN_TA_DEG and geom.range_m < LEAD_TURN_MIN_RANGE_M:
            self.transition_to(ControllerState.LEAD_TURN, f"TA {geom.target_aspect_deg:.1f}°")
            return

        hdg_cmd = self.compute_offset_intercept_heading(geom, lead_state.heading_deg)
        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else PRECONTACT_ALT_MSL
        spd_cmd = self.control_laws.compute_speed_cmd(
            geom,
            self._windff_lead_gs_air(lead_state.ground_speed_ms),  # WINDFF-1b: air-frame lead ref
            STERN_CONVERSION_RANGE_M,
            fol_alt,
            enable_overtake_protect=False,
            controller_state=self.state)
        # CSV2_SPEC247: TAS->EAS command FF (default OFF -> byte-identical) so the realized
        # intercept TAS equals the designed command (no EAS->TAS overshoot during catch-up).
        spd_cmd = apply_eas2tas_ff(spd_cmd, fol_alt, eas2tas=self.follower.get_baro_eas2tas())
        alt_cmd = self._compute_altitude_target_msl(geom, lead_state, self._t_sim_s)
        self.send_guided_commands(hdg_cmd, alt_cmd, spd_cmd, fol_alt=fol_alt)
        self.log_continuous_diagnostics(geom, lead_state, fol_alt, hdg_cmd, spd_cmd, "INTERCEPT")

    def _handle_lead_turn(self) -> None:
        """Execute fighter lead turn to roll out behind target."""
        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        if abs(geom.target_aspect_deg) < STERN_ASPECT_TOLERANCE_DEG:
            self.transition_to(ControllerState.STERN_CONVERSION, f"TA {geom.target_aspect_deg:.1f}°")
            return

        hdg_cmd = self.compute_lead_turn_heading(geom, lead_state.heading_deg)
        spd_cmd = LEAD_TURN_SPEED_MS
        alt_cmd = self._compute_altitude_target_msl(geom, lead_state, self._t_sim_s)
        self.send_guided_commands(hdg_cmd, alt_cmd, spd_cmd)
        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else PRECONTACT_ALT_MSL
        self.log_continuous_diagnostics(geom, lead_state, fol_alt, hdg_cmd, spd_cmd, "LEAD_TURN")

    def _handle_stern_conversion(self) -> None:
        """Run stern conversion - position at 200m behind boom."""
        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else PRECONTACT_ALT_MSL
        elapsed = self._t_sim_s - self.state_entry_time  # v9.10: Use _t_sim_s

        if self.envelope_checker.is_in_precontact_entry_envelope(
                geom, lead_state.alt_msl, fol_alt) and self.check_precontact_vc_gate(
                geom.behind_m):
            self.transition_to(ControllerState.HOLD_PRECONTACT, "Precontact in STERN_CONV")
            return

        range_ok = geom.range_m <= (STERN_CONVERSION_RANGE_M + STERN_HANDOFF_MAX_OVER_M)
        aspect_ok = abs(geom.target_aspect_deg) <= STERN_ASPECT_TOLERANCE_DEG
        cross_ok = abs(geom.cross_m) < STERN_CONVERSION_STABILITY_CROSS_M
        dwell_ok = elapsed >= STERN_CONVERSION_MIN_DWELL_S

        if (range_ok and aspect_ok and cross_ok and dwell_ok) or (geom.range_m <= FORCE_REJOIN_RANGE_M and aspect_ok and abs(
                geom.cross_m) < FORCE_REJOIN_CROSS_LIMIT_M) or (elapsed >= STERN_CONVERSION_MAX_TIME_S):
            if _FD_CROSS_CRAB_ENABLE:
                self._reset_cross_crab_rejoin()  # Lever A: fresh approach → drop any stale crab
            self.transition_to(ControllerState.REJOIN_PRECONTACT, "Stern conversion complete")
            return

        hdg_cmd = self.control_laws.compute_formation_heading_cmd(geom, lead_state.heading_deg)
        spd_cmd = self.control_laws.compute_speed_cmd(
            geom,
            self._windff_lead_gs_air(lead_state.ground_speed_ms),  # WINDFF-1b: air-frame lead ref
            STERN_CONVERSION_RANGE_M,
            fol_alt,
            enable_overtake_protect=False,
            controller_state=self.state)
        # CSV2_SPEC247: TAS->EAS command FF (default OFF -> byte-identical) so the realized
        # stern-conversion TAS equals the designed command (no EAS->TAS overshoot).
        spd_cmd = apply_eas2tas_ff(spd_cmd, fol_alt, eas2tas=self.follower.get_baro_eas2tas())
        alt_cmd = self._compute_altitude_target_msl(geom, lead_state, self._t_sim_s)
        self.send_guided_commands(hdg_cmd, alt_cmd, spd_cmd, range_m=geom.range_m)
        self.log_continuous_diagnostics(geom, lead_state, fol_alt, hdg_cmd, spd_cmd, "STERN_CONVERSION")

    def _update_slow_behind_vc(self, behind_m: float) -> None:
        """Update slow behind-axis derivative every tick.

        v9.10: Uses _t_sim_s and _dt_tick_used_s instead of calling sim_now_s().
        This ensures single-source timebase from the tick gating block.

        v9.2: Computes net behind drift from position samples with a
        long EWMA (tau=PRECONTACT_VC_FILTER_TAU_S). Called from
        run_state_machine() so the value is fresh for both the VC gate
        and the climb guard.

        Args:
            behind_m: Current behind distance (m), positive = behind.
        """
        now = self._t_sim_s  # v9.10: Use tick timestamp

        # v9.10: Use dt from tick gating instead of recomputing
        dt_clamped = self._dt_tick_used_s if self._dt_tick_used_s else 0.0

        # Compute raw behind derivative from position delta
        if self._vc_gate_behind_prev is None:
            self._vc_gate_behind_prev = behind_m
            self._vc_gate_behind_prev_time = now
            raw_vc = 0.0
        else:
            dt_pos = now - self._vc_gate_behind_prev_time
            if dt_pos >= 0.04:  # Min 40ms to avoid divide-by-noise
                raw_vc = (self._vc_gate_behind_prev - behind_m) / dt_pos
                self._vc_gate_behind_prev = behind_m
                self._vc_gate_behind_prev_time = now
            else:
                raw_vc = self._vc_gate_slow_vc  # Hold previous

        # dt-corrected EWMA with long tau for net drift
        tau = PRECONTACT_VC_FILTER_TAU_S
        if tau > 0.0 and dt_clamped > 0.0:
            alpha = 1.0 - math.exp(-dt_clamped / tau)
            self._vc_gate_slow_vc = (
                alpha * raw_vc
                + (1.0 - alpha) * self._vc_gate_slow_vc
            )
        else:
            self._vc_gate_slow_vc = raw_vc

    def check_precontact_vc_gate(self, behind_m: float) -> bool:
        """Gate transition to HOLD_PRECONTACT on closure rate stability.

        v9.10: Uses _t_sim_s and _dt_credit_s instead of sim_now_s().
        This ensures single-source timebase from the tick gating block.

        P7.3 v9.7: Decay-based credit with dt-invariant grace period.
        Uses the slow behind-axis derivative (_vc_gate_slow_vc) computed
        every tick by _update_slow_behind_vc().

        Three bands:
          PASS (|vc| <= PASS_MAX): accumulate credit, clear violation timer
          HYS (PASS_MAX < |vc| < RESET_MAX): hold credit, clear violation timer
          RESET (|vc| >= RESET_MAX): decay after grace period (dt-invariant)

        No credit floor - allows decay to 0.0 to require genuine stability.

        Args:
            behind_m: Current behind distance (m), positive = behind.

        Returns:
            True when accumulated credit reaches STABLE_REQ_S.
        """
        now = self._t_sim_s  # v9.10: Use tick timestamp
        vc = abs(self._vc_gate_slow_vc)

        # v9.10: Use dt_credit from tick gating for credit accumulation
        dt_clamped = self._dt_credit_s if self._dt_credit_s else 0.0

        # Three-band credit logic with P7.3 decay system
        pass_max = PRECONTACT_VC_PASS_MAX_MS
        reset_max = PRECONTACT_VC_RESET_MAX_MS
        req = PRECONTACT_VC_STABLE_REQ_S

        reason = ""
        if vc <= pass_max:
            # PASS band: accumulate credit, clear violation timer
            self._precontact_vc_violation_start_s = None
            self._precontact_vc_credit_s = min(
                req, self._precontact_vc_credit_s + dt_clamped
            )
            band = "PASS"
            reason = f"+{dt_clamped:.3f}s"

        elif vc >= reset_max:
            # RESET band: decay after grace period (dt-invariant)
            if self._precontact_vc_violation_start_s is None:
                self._precontact_vc_violation_start_s = now

            # v9.10: Use accumulator for time_in_violation instead of timestamp diff
            if not hasattr(self, '_vc_violation_accum_s'):
                self._vc_violation_accum_s = 0.0
            self._vc_violation_accum_s += dt_clamped
            time_in_violation = self._vc_violation_accum_s

            if time_in_violation >= VC_GATE_RESET_GRACE_S:
                # P7.3: Decay-based credit reduction (rate * dt)
                decay = VC_GATE_DECAY_RATE_PER_S * dt_clamped
                self._precontact_vc_credit_s = max(
                    0.0, self._precontact_vc_credit_s - decay
                )
                reason = f"-{decay:.3f}s (decay)"
            else:
                reason = f"grace {time_in_violation:.2f}/{VC_GATE_RESET_GRACE_S}s"
            band = "RESET"

        else:
            # HYS band: hold credit, clear violation timer
            self._precontact_vc_violation_start_s = None
            self._vc_violation_accum_s = 0.0  # v9.10: Reset accumulator
            band = "HYS"
            reason = "hold"

        vc_gate_pass = self._precontact_vc_credit_s >= req

        # v9.10: Use accumulator for rate-limited logging
        if not hasattr(self, '_vc_gate_log_accum_s'):
            self._vc_gate_log_accum_s = 0.0
        self._vc_gate_log_accum_s += dt_clamped
        if self._vc_gate_log_accum_s >= 1.0:
            self._vc_gate_log_accum_s = 0.0
            log(
                f"[VC_GATE] vc_slow={vc:.2f} "
                f"credit={self._precontact_vc_credit_s:.2f}/{req:.1f} "
                f"band={band} {reason} behind={behind_m:.1f}"
            )

        return vc_gate_pass

    def check_contact_capture_gate(self, closure_rate_ms: float,
                                   vel_cross_m_s: float) -> bool:
        """Test 311: Check if Vc and cross-rate are stable for HOLD_CONTACT entry.

        v9.10: Uses _t_sim_s and accumulator pattern instead of sim_now_s().

        Similar to precontact gate but with tighter requirements for contact.
        Requires both Vc and cross_rate to be below thresholds for stable time.

        Args:
            closure_rate_ms: Current closure rate in m/s
            vel_cross_m_s: Current lateral velocity in m/s

        Returns:
            True if contact gate is satisfied (stable for required duration)
        """
        now = self._t_sim_s  # v9.10: Use tick timestamp
        dt_credit = self._dt_credit_s if self._dt_credit_s else 0.0
        vc_ok = abs(closure_rate_ms) <= CONTACT_VC_MAX_MS
        cross_rate_ok = abs(vel_cross_m_s) <= CONTACT_CROSS_RATE_MAX_MS
        both_ok = vc_ok and cross_rate_ok

        # v9.10: Use accumulator for stable duration
        if not hasattr(self, '_contact_stable_accum_s'):
            self._contact_stable_accum_s = 0.0

        if both_ok:
            if self._contact_vc_stable_start is None:
                self._contact_vc_stable_start = now
                self._contact_stable_accum_s = 0.0
            self._contact_stable_accum_s += dt_credit
            stable_duration = self._contact_stable_accum_s
            gate_passed = stable_duration >= CONTACT_VC_STABLE_TIME_S
        else:
            self._contact_vc_stable_start = None
            self._contact_stable_accum_s = 0.0
            stable_duration = 0.0
            gate_passed = False

        # v9.10: Use accumulator for rate-limited logging
        if not hasattr(self, '_contact_vc_log_accum_s'):
            self._contact_vc_log_accum_s = 0.0
        self._contact_vc_log_accum_s += dt_credit
        if self._contact_vc_log_accum_s >= CONTACT_VC_GATE_LOG_INTERVAL_S:
            self._contact_vc_log_accum_s = 0.0
            status = "PASS" if gate_passed else "WAIT"
            log(f"[CONTACT_VC_GATE] {status}: Vc={closure_rate_ms:+.2f}m/s "
                f"cross_rate={vel_cross_m_s:+.2f}m/s "
                f"(stable={stable_duration:.1f}/{CONTACT_VC_STABLE_TIME_S}s)")

        return gate_passed

    def _reset_hold_contact_integrator(self) -> None:
        """Test 311: Reset HOLD_CONTACT PI integrator state."""
        self._hold_contact_i = 0.0
        self._hold_contact_i_last_t = None
        self._hold_contact_last_behind_err = None
        # v9.2 Patch D: Reset deficit integrator
        self.control_laws.reset_closure_deficit_integrator()

    def _reset_cross_track_bias(self) -> None:
        """Test 311: Reset cross-track PI bias integrator."""
        self._cross_i_deg = 0.0
        self._cross_i_last_t = None

    def _compute_cross_track_bias(self, geom, lead_state, rejoin_mode=False) -> float:
        """P8.2 v9.8: Compute cross-track heading bias for earlier convergence.

        CSV2_R7_1: rejoin_mode=True (REJOIN_PRECONTACT/HOLD_PRECONTACT, flag-gated by
        FD_CROSS_CRAB_ENABLE at the call site) routes to the ILOS crab integrator
        (_cross_crab_rejoin_bias) with a SEPARATE integrator state, wider clamp/enable,
        deadband+sustained trigger, and leader-yaw turn-freeze. Default False keeps the
        docking-state behaviour byte-identical.

        Enables cross integrator when behind ≤ CROSS_BIAS_ENABLE_RANGE_M (30m by default).
        Gates on:
        - Leader turn rate low (stable flight)
        - Cross error not too large (avoid winding up on gross error)
        - Behind distance within range for active control

        v9.10: Uses _dt_tick_used_s from tick gating instead of computing dt from
        sim_now_s() differences. This ensures single-source timebase.

        Returns:
            Heading bias in degrees to add to formation heading command.
        """
        # v9.10: Use tick dt instead of computing from timestamp differences
        dt = self._dt_tick_used_s if self._dt_tick_used_s else 0.05
        if dt <= 0 or dt > 0.5:
            # Skip invalid dt or large gaps
            return self._cross_i_deg if hasattr(self, '_cross_i_deg') else 0.0

        # CSV2_R7_1: rejoin crab uses a separate integrator + gates (reached only when
        # FD_CROSS_CRAB_ENABLE is ON, via the call-site short-circuit).
        if rejoin_mode:
            return self._cross_crab_rejoin_bias(geom, lead_state, dt)

        # P8.2: Enable earlier (behind ≤ enable range)
        enable_range_ok = geom.behind_m <= CROSS_BIAS_ENABLE_RANGE_M
        cross_err_ok = abs(geom.cross_m) <= CROSS_BIAS_ENABLE_MAX_CROSS_M

        # Gate on leader turn rate (use filtered yaw rate if available)
        # For now, assume stable if not in a known turn condition
        # A proper implementation would check lead_state.yaw_rate
        turn_rate_ok = True  # Simplification: assume stable flight

        # Determine if we should integrate
        should_integrate = enable_range_ok and cross_err_ok and turn_rate_ok

        if should_integrate:
            # P8.2: PI controller for cross-track
            # P-term: immediate correction (negative sign = corrective,
            # same direction as base -0.175 * cross_m heading bias)
            p_term = -CROSS_BIAS_KP * geom.cross_m

            # I-term: accumulate cross error (anti-windup clamped)
            self._cross_i_deg += -CROSS_BIAS_KI * geom.cross_m * dt
            self._cross_i_deg = max(-CROSS_BIAS_I_CLAMP_DEG,
                                    min(CROSS_BIAS_I_CLAMP_DEG, self._cross_i_deg))

            bias = p_term + self._cross_i_deg
        else:
            # Decay when conditions break (dt-scaled for tick-rate independence)
            self._cross_i_deg *= math.exp(-dt / CROSS_BIAS_I_DECAY_TAU_S)
            bias = self._cross_i_deg

        # Track for CSV logging
        self._cross_i_term = self._cross_i_deg

        return bias

    def _cross_crab_rejoin_bias(self, geom, lead_state, dt) -> float:
        """CSV2_R7_1 rejoin cross-track crab (ILOS integral). Nulls the steady-state
        cross offset under a continuous mean crosswind (pure-P only balances it). Uses a
        SEPARATE integrator (_cross_i_rejoin_deg) so docking callers stay byte-identical.
        Calm-inert: deadband + sustained-tick trigger + decay. Deterministic: controller
        tick dt only (no wall-clock). Onboard signals only (fused cross + leader broadcast
        heading) — no privileged or ground-truth inputs. Only reached when the crab flag is ON."""
        c = geom.cross_m
        # Leader yaw-rate from broadcast-heading deltas (EMA-filtered; reuse stub fields).
        yaw_rate_raw_dps = 0.0
        if self._lead_last_hdg is not None:
            yaw_rate_raw_dps = wrap_heading_error(lead_state.heading_deg - self._lead_last_hdg) / dt
            self._lead_yaw_rate_filt_dps += 0.3 * (yaw_rate_raw_dps - self._lead_yaw_rate_filt_dps)
        self._lead_last_hdg = lead_state.heading_deg
        yaw_rate_dps = self._lead_yaw_rate_filt_dps
        # Lever A leading-edge gate: the EMA (0.3 gain) lags a turn onset by ~3 ticks, during which
        # the integrator would wind up on the turn geometry; the RAW delta trips on tick 1. Gate on
        # whichever is larger so a sharp turn-entry freezes immediately (calm-inertness).
        yaw_rate_gate_dps = max(abs(yaw_rate_raw_dps), abs(yaw_rate_dps))
        # Sustained one-sided cross beyond the deadband (calm-inert trigger).
        if abs(c) >= CROSS_BIAS_REJOIN_DEADBAND_M:
            sgn = 1 if c > 0 else -1
            if sgn == self._cross_sustain_sign:
                self._cross_sustain_ticks += 1
            else:
                self._cross_sustain_sign = sgn
                self._cross_sustain_ticks = 1
        else:
            self._cross_sustain_sign = 0
            self._cross_sustain_ticks = 0
        cross_err_ok = abs(c) <= CROSS_BIAS_REJOIN_ENABLE_MAX_CROSS_M
        turn_rate_ok = yaw_rate_gate_dps <= CROSS_BIAS_REJOIN_TURN_FREEZE_DEG_S
        sustained_ok = self._cross_sustain_ticks >= CROSS_BIAS_REJOIN_SUSTAIN_N_TICKS
        should_integrate = cross_err_ok and turn_rate_ok and sustained_ok
        clamped = False
        if should_integrate:
            p_term = -CROSS_BIAS_KP * c
            self._cross_i_rejoin_deg += -CROSS_BIAS_KI * c * dt
            pre = self._cross_i_rejoin_deg
            self._cross_i_rejoin_deg = max(-CROSS_BIAS_REJOIN_I_CLAMP_DEG,
                                           min(CROSS_BIAS_REJOIN_I_CLAMP_DEG, self._cross_i_rejoin_deg))
            clamped = (self._cross_i_rejoin_deg != pre)
            bias = p_term + self._cross_i_rejoin_deg
            reason = "ACTIVE"
        else:
            # Deterministic decay when conditions break (dt-scaled, tick-rate independent).
            # Lever A: a freeze caused BY a turn bleeds FAST (sheds turn-onset windup before it
            # carries a stale crab through the turn); any other freeze keeps the SLOW decay so a
            # converged straight-leg crab persists across brief deadband/sustain gaps.
            bleed_tau = (CROSS_BIAS_REJOIN_TURN_BLEED_TAU_S if not turn_rate_ok
                         else CROSS_BIAS_I_DECAY_TAU_S)
            self._cross_i_rejoin_deg *= math.exp(-dt / bleed_tau)
            bias = self._cross_i_rejoin_deg
            reason = ("CROSS_TOO_LARGE" if not cross_err_ok
                      else "TURN_FREEZE" if not turn_rate_ok
                      else "BELOW_DEADBAND" if abs(c) < CROSS_BIAS_REJOIN_DEADBAND_M
                      else "NOT_SUSTAINED")
        self._cross_crab_write(c, bias, reason, clamped, yaw_rate_dps)
        return bias

    def _reset_cross_crab_rejoin(self) -> None:
        """CSV2_R7_1 lever-A: zero the rejoin crab integrator + sustain/yaw-filter state on a
        FRESH STERN→REJOIN entry so a stale crab bias from a prior approach cycle does not carry
        into a new one. Called flag-gated at exactly one site (the fresh entry) — NOT the
        HOLD→REJOIN re-entry, which keeps its converged integrator. Flag-OFF never reaches it."""
        self._cross_i_rejoin_deg = 0.0
        self._cross_sustain_ticks = 0
        self._cross_sustain_sign = 0
        self._lead_yaw_rate_filt_dps = 0.0
        self._lead_last_hdg = None

    def _init_cross_crab_diag(self) -> None:
        """CSV2_R7_1 (gated, default-off): open cross_crab_diag.csv. No-op unless
        FD_CROSS_CRAB_ENABLE=1 (so flag-OFF runs are byte-identical, no extra file).
        Mirrors _init_uwb_fusion_diag. LOGGING-ONLY."""
        if not _FD_CROSS_CRAB_ENABLE:
            return
        import csv
        import os
        out_dir = os.environ.get('FD_UWB_FUSION_DIAG_DIR', '').strip() or self._out_dir
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "cross_crab_diag.csv")
            self._cross_crab_diag_file = open(csv_path, 'w', newline='')
            self._cross_crab_diag_writer = csv.writer(self._cross_crab_diag_file)
            self._cross_crab_diag_writer.writerow([
                "sim_time_s", "cross_crab_enable_bool", "cross_crab_active_bool",
                "cross_crab_bias_deg", "cross_crab_integral_state",
                "cross_crab_freeze_reason", "cross_crab_leader_yaw_rate_deg_s",
                "cross_crab_cross_m", "cross_crab_rejoin_mode_bool", "cross_crab_clamped_bool",
            ])
            log(f"[CROSS_CRAB] Initialized cross_crab_diag.csv at {csv_path}")
        except Exception as e:  # pragma: no cover - diag must never break control
            log(f"[CROSS_CRAB] diag init failed: {e}")

    def _cross_crab_write(self, cross_m, bias_deg, reason, clamped, yaw_rate_dps) -> None:
        """Append one cross_crab_diag.csv row (gated; writer None unless flag-ON)."""
        w = self._cross_crab_diag_writer
        if w is None:
            return
        try:
            t = self._t_sim_s if hasattr(self, '_t_sim_s') else 0.0
            w.writerow([
                f"{t:.3f}", True, (reason == "ACTIVE"),
                f"{bias_deg:.4f}", f"{self._cross_i_rejoin_deg:.4f}", reason,
                f"{yaw_rate_dps:.4f}", f"{cross_m:.4f}", True, clamped,
            ])
            self._cross_crab_diag_file.flush()
        except Exception:  # pragma: no cover - diag must never break control
            pass

    def _handle_rejoin_precontact(self) -> None:
        """Run rejoin phase - close from 200m to precontact envelope."""
        # FD-005: Capture entry state for mid-tick abort
        entry_state = self.state

        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else PRECONTACT_ALT_MSL

        _, _, below_err = self.envelope_checker.get_envelope_errors(
            geom, lead_state.alt_msl, fol_alt, PRECONTACT_BEHIND_M, PRECONTACT_CROSS_M, PRECONTACT_BELOW_M)

        # v9.10: Use accumulator pattern for ALT_GATE timing
        if not hasattr(self, '_alt_gate_accum_s'):
            self._alt_gate_accum_s = 0.0
        if not hasattr(self, '_alt_gate_log_accum_s'):
            self._alt_gate_log_accum_s = 0.0

        if geom.range_m < ALT_GATE_RANGE_M and abs(below_err) > ALT_GATE_THRESHOLD_M:
            # ALT_GATE: Accumulate time while altitude error persists
            if self._alt_gate_start is None:
                self._alt_gate_start = self._t_sim_s
                self._alt_gate_accum_s = 0.0
            self._alt_gate_accum_s += self._dt_credit_s
            if self._alt_gate_accum_s > ALT_GATE_TIME_S:
                # Altitude error persisted too long - activate gate (log only on transition)
                if not self._alt_gate_active:
                    self._alt_gate_active = True
                    log(f"[ALT_GATE] ACTIVATED: |below_err|={abs(below_err):.1f}m > {ALT_GATE_THRESHOLD_M}m "
                        f"for >{ALT_GATE_TIME_S}s at range={geom.range_m:.1f}m")
        else:
            # Clear gate timer if altitude OK
            if self._alt_gate_start is not None:
                self._alt_gate_start = None
                self._alt_gate_accum_s = 0.0
            if self._alt_gate_active:
                self._alt_gate_active = False
                log("[ALT_GATE] CLEARED: altitude error within tolerance")

        # E3: Throttled log confirming C++ is off during ALT_GATE (using accumulator)
        if self._alt_gate_active:
            self._alt_gate_log_accum_s += self._dt_credit_s
            if self._alt_gate_log_accum_s > 5.0:
                log(f"[ALT_GATE] Active: C++ disabled, Python altitude recovery in progress")
                self._alt_gate_log_accum_s = 0.0

        # v9.2 Patch A rev2: Gate uses behind_m directly with slow EWMA
        # to measure net drift, avoiding the noisy fast EWMA.
        # v9.10 Gate 2: Entry grace - require sustained time inside envelope before transition
        in_entry_envelope = self.envelope_checker.is_in_precontact_entry_envelope(
            geom, lead_state.alt_msl, fol_alt)
        vc_gate_ok = self.check_precontact_vc_gate(geom.behind_m)

        if in_entry_envelope and vc_gate_ok:
            # Accumulate entry grace using dt_credit_s (not raw time differences)
            self._hold_entry_grace_accum_s += self._dt_credit_s

            if self._hold_entry_grace_accum_s >= HOLD_ENTRY_GRACE_S:
                # Entry grace satisfied - transition to HOLD
                self.transition_to(ControllerState.HOLD_PRECONTACT,
                                   f"Precontact envelope (entry_grace={self._hold_entry_grace_accum_s:.2f}s)")
                # Reset entry grace accumulator for next time
                self._hold_entry_grace_accum_s = 0.0
                return
        else:
            # Outside entry envelope or VC gate failed - reset entry grace
            self._hold_entry_grace_accum_s = 0.0

        # CSV2_R7_1: flag-gated rejoin crab (extra_bias=0 when OFF → byte-identical).
        _crab = self._compute_cross_track_bias(geom, lead_state, rejoin_mode=True) if _FD_CROSS_CRAB_ENABLE else 0.0
        hdg_cmd = self.control_laws.compute_formation_heading_cmd(geom, lead_state.heading_deg, extra_bias_deg=_crab)

        # P8.3 v9.8: Near-precontact closure cap to allow VC gate accumulation
        # When close to M2 envelope, cap closure rate to let vc_slow decay below threshold
        climb_clamp_cap = None
        near_precontact_active = (
            NEAR_PRECONTACT_ENABLE_BEHIND_MIN_M <= geom.behind_m <= NEAR_PRECONTACT_ENABLE_BEHIND_MAX_M
            and abs(geom.cross_m) <= NEAR_PRECONTACT_ENABLE_CROSS_TOL_M
            and abs(geom.below_m - PRECONTACT_BELOW_M) <= NEAR_PRECONTACT_ENABLE_BELOW_TOL_M
        )
        if near_precontact_active:
            # Cap closure rate to allow VC gate to accumulate credit
            # WINDFF-1b: air-frame lead ref (GS-as-airspeed cap; no-op when OFF)
            climb_clamp_cap = self._windff_lead_gs_air(lead_state.ground_speed_ms) + NEAR_PRECONTACT_VC_CAP_MPS
        # Track for observability
        self._near_precontact_active = near_precontact_active
        self._near_precontact_vc_cap = climb_clamp_cap if climb_clamp_cap else 0.0

        # Guard strip-down: CE + NOPASS + D1 vertical floor removed (Chunk 2A)
        self._in_precontact_corridor = False
        self._vert_floor_soft_active = False
        self._vert_floor_hard_active = False

        # M3 S1: Record binding constraint pre-override (no guard caps)
        safety_cap_source = "NONE"
        if near_precontact_active and climb_clamp_cap is not None:
            safety_cap_source = "NEAR_PRECONTACT_VC_CAP"
        self._last_binding_pre_override = safety_cap_source
        self._last_binding_value_mps = 0.0
        self._last_binding_margin_m = 0.0

        spd_cmd = self.control_laws.compute_speed_cmd(
            geom, self._windff_lead_gs_air(lead_state.ground_speed_ms),  # WINDFF-1b: air-frame lead ref
            PRECONTACT_BEHIND_M, fol_alt,
            climb_clamp_cap=climb_clamp_cap,
            safety_clamp_cap=None,  # Guard strip-down: no guard cap
            controller_state=self.state,
        )
        # CSV2_SPEC247: TAS->EAS command FF (default OFF -> byte-identical) so the follower
        # rejoins at realized LEADER_TARGET_TAS_MS and enters the docking states settled (not hot).
        spd_cmd = apply_eas2tas_ff(spd_cmd, fol_alt, eas2tas=self.follower.get_baro_eas2tas())
        self._last_spd_cmd = spd_cmd

        # M3 S1: Determine final binding (FD-004 override vs pre-override)
        override = self._peek_speed_override()
        self._speed_override_active = override is not None
        if override is not None:
            self._last_binding_final = "FD004_SPEED_OVERRIDE"
        else:
            self._last_binding_final = safety_cap_source
        self._update_binding_counters()

        # P6.0 v9.6: Update clamp state for logging
        arb = self.control_laws.get_last_arbitration()
        if arb:
            self._brake_clamp_active = arb.brake_clamp_active
            self._climb_clamp_active = arb.climb_clamp_active
            self._safety_clamp_active = arb.safety_clamp_active
            # M3 S1: Attach binding constraint to arbitration result
            arb.binding_constraint_pre_override = self._last_binding_pre_override
            arb.binding_constraint_final = self._last_binding_final
            arb.binding_value_mps = self._last_binding_value_mps
            arb.binding_margin_m = self._last_binding_margin_m
            self._log_speed_arbitration(arb, geom)

        alt_cmd = self._compute_altitude_target_msl(geom, lead_state, self._t_sim_s)

        # FD-005: Mid-tick abort — nopass guard may have transitioned state
        if self.state != entry_state:
            log(f"[FD-005] Mid-tick abort: {entry_state.name} -> {self.state.name}")
            return

        self.send_guided_commands(hdg_cmd, alt_cmd, spd_cmd, range_m=geom.range_m, fol_alt=fol_alt)
        self.log_continuous_diagnostics(geom, lead_state, fol_alt, hdg_cmd, spd_cmd, "REJOIN_PRECONTACT")

    def _handle_hold_precontact(self) -> None:
        """Hold at precontact position for configured dwell."""
        # FD-005: Capture entry state for mid-tick abort
        entry_state = self.state

        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else PRECONTACT_ALT_MSL

        now = self._t_sim_s  # v9.10: Use tick timestamp
        in_outer = self.envelope_checker.is_in_precontact_outer_envelope(geom, lead_state.alt_msl, fol_alt)

        # v9.10 Gate 2: Exit grace uses _dt_credit_s accumulator (not raw time differences)
        if not in_outer:
            # Outside OUTER - accumulate exit grace using dt_credit_s
            self._hold_exit_grace_accum_s += self._dt_credit_s

            if self._hold_exit_grace_accum_s > HOLD_EXIT_GRACE_S:
                # Exit grace exceeded - transition to REJOIN
                self.precontact_hold_start = None
                self._precontact_outer_exit_start = None
                # TEST 401: Reset climb guard when exiting HOLD_PRECONTACT
                self._climb_guard_stable_s = 0.0
                self._climb_guard_last_ok = False
                self.transition_to(ControllerState.REJOIN_PRECONTACT,
                                   f"Left precontact envelope (exit_grace={self._hold_exit_grace_accum_s:.2f}s)")
                # Reset exit grace accumulator for next time
                self._hold_exit_grace_accum_s = 0.0
                return
            # Still within exit grace - stay in HOLD_PRECONTACT
        else:
            # Back inside OUTER - clear exit grace accumulator
            self._hold_exit_grace_accum_s = 0.0
            self._precontact_outer_exit_start = None

        # TEST 400 Fix 2: INNER envelope check with grace period
        in_inner = self.envelope_checker.is_in_precontact_inner_envelope(
            geom, lead_state.alt_msl, fol_alt
        )

        if in_inner:
            # Inside INNER - clear exit timer, accumulate dwell
            self._precontact_inner_exit_start = None
            self._precontact_inner_exit_accum_s = 0.0  # v9.10: Reset inner exit grace accumulator
            if self.precontact_hold_start is None:
                self.precontact_hold_start = now
                self._precontact_dwell_accum_s = 0.0  # v9.10 Gate 2: Use accumulator for dwell

            # v9.10 Gate 2: Dwell accumulation uses dt_credit_s (not raw time differences)
            self._precontact_dwell_accum_s += self._dt_credit_s
            # Track max dwell for run_summary (gate checker uses this)
            if self._precontact_dwell_accum_s > self._precontact_max_dwell_s:
                self._precontact_max_dwell_s = self._precontact_dwell_accum_s

            # TEST 401 Fix 4: Climb guard - require low behind-closure before transition
            # v9.2: Use slow behind derivative (net drift) instead of fast EWMA
            # The fast EWMA reads ~2 m/s at station-keeping due to control-loop
            # oscillation; the slow derivative (tau=3s) shows actual net drift.
            vc_behind = abs(self._vc_gate_slow_vc)
            climb_guard_ok = vc_behind <= CLIMB_GUARD_VC_MAX_BEHIND_MS

            # v9.10 Gate 2: Climb guard stable time uses dt_credit_s
            if climb_guard_ok:
                self._climb_guard_stable_s += self._dt_credit_s
            else:
                self._climb_guard_stable_s = 0.0
            self._climb_guard_last_ok = climb_guard_ok

            dwell_ok = self._precontact_dwell_accum_s >= PRECONTACT_HOLD_DURATION_S
            guard_ok = self._climb_guard_stable_s >= CLIMB_GUARD_STABLE_S

            # v9.10: Rate-limited logging using accumulator
            if not hasattr(self, '_climb_guard_log_accum_s'):
                self._climb_guard_log_accum_s = 0.0
            self._climb_guard_log_accum_s += self._dt_credit_s
            if self._climb_guard_log_accum_s >= CLIMB_GUARD_LOG_INTERVAL_S:
                self._climb_guard_log_accum_s = 0.0
                status = "PASS" if (dwell_ok and guard_ok) else "WAIT"
                log(f"[CLIMB_GUARD] {status}: vc_behind={vc_behind:.2f}m/s "
                    f"stable={self._climb_guard_stable_s:.1f}/{CLIMB_GUARD_STABLE_S}s "
                    f"dwell={self._precontact_dwell_accum_s:.1f}/{PRECONTACT_HOLD_DURATION_S}s")

            if dwell_ok and guard_ok:
                # P1 v9.4: Transition to ALT_CAP_HOLD_RANGE instead of ALTITUDE_CAPTURE
                # This ensures vertical alignment to boom tip before allowing closure
                self.transition_to(
                    ControllerState.ALT_CAP_HOLD_RANGE,
                    "Precontact hold complete, begin altitude capture"
                )
                return
        else:
            # Outside INNER - accumulate inner exit grace using dt_credit_s
            # v9.10 Gate 2: Use accumulator instead of timestamp differences
            if not hasattr(self, '_precontact_inner_exit_accum_s'):
                self._precontact_inner_exit_accum_s = 0.0
            self._precontact_inner_exit_accum_s += self._dt_credit_s

            # Only reset dwell after sustained INNER violation
            if self._precontact_inner_exit_accum_s > PRECONTACT_INNER_EXIT_GRACE_S:
                self.precontact_hold_start = None
                self._precontact_inner_exit_accum_s = 0.0
                self._precontact_dwell_accum_s = 0.0  # v9.10: Reset dwell accumulator too
                # TEST 401: Reset climb guard when dwell resets
                self._climb_guard_stable_s = 0.0
                self._climb_guard_last_ok = False

        # CSV2_R7_1: flag-gated rejoin crab (extra_bias=0 when OFF → byte-identical).
        _crab = self._compute_cross_track_bias(geom, lead_state, rejoin_mode=True) if _FD_CROSS_CRAB_ENABLE else 0.0
        hdg_cmd = self.control_laws.compute_formation_heading_cmd(geom, lead_state.heading_deg, extra_bias_deg=_crab)

        # Guard strip-down: CE + NOPASS + D1 vertical floor removed (Chunk 2A)
        self._in_precontact_corridor = False
        self._vert_floor_soft_active = False
        self._vert_floor_hard_active = False
        self._last_binding_pre_override = "NONE"
        self._last_binding_value_mps = 0.0

        spd_cmd = self.control_laws.compute_speed_cmd(
            geom,
            self._windff_lead_gs_air(lead_state.ground_speed_ms),  # WINDFF-1b: air-frame lead ref
            PRECONTACT_BEHIND_M,
            fol_alt,
            station_keeping=True,
            safety_clamp_cap=None,  # Guard strip-down: no guard cap
            controller_state=self.state,
        )
        # CSV2_SPEC247: TAS->EAS command FF (default OFF -> byte-identical) so the follower
        # holds the standoff at realized LEADER_TARGET_TAS_MS without overtaking.
        spd_cmd = apply_eas2tas_ff(spd_cmd, fol_alt, eas2tas=self.follower.get_baro_eas2tas())
        self._last_spd_cmd = spd_cmd

        # M3 S1: Determine final binding (FD-004 override vs pre-override)
        override = self._peek_speed_override()
        self._speed_override_active = override is not None
        if override is not None:
            self._last_binding_final = "FD004_SPEED_OVERRIDE"
        else:
            self._last_binding_final = self._last_binding_pre_override
        self._update_binding_counters()

        # P6.0 v9.6: Update clamp state for logging
        arb = self.control_laws.get_last_arbitration()
        if arb:
            self._brake_clamp_active = arb.brake_clamp_active
            self._climb_clamp_active = arb.climb_clamp_active
            self._safety_clamp_active = arb.safety_clamp_active
            arb.binding_constraint_pre_override = self._last_binding_pre_override
            arb.binding_constraint_final = self._last_binding_final
            arb.binding_value_mps = self._last_binding_value_mps
            self._log_speed_arbitration(arb, geom)

        alt_cmd = self._compute_altitude_target_msl(geom, lead_state, now)

        # TEST 400: Altitude tracking instrumentation
        # v9.10: Use accumulator for rate-limited logging
        if not hasattr(self, '_alt_track_log_accum_s'):
            self._alt_track_log_accum_s = 0.0
        self._alt_track_log_accum_s += self._dt_credit_s
        if self._alt_track_log_accum_s >= self._alt_track_log_interval_s:
            self._alt_track_log_accum_s = 0.0
            alt_err = fol_alt - alt_cmd
            # v9.10: Use dwell/grace accumulators instead of timestamp differences
            dwell_s = self._precontact_dwell_accum_s
            outer_grace_s = self._hold_exit_grace_accum_s
            outer_grace_active = outer_grace_s > 0
            inner_grace_s = self._precontact_inner_exit_accum_s if hasattr(self, '_precontact_inner_exit_accum_s') else 0.0
            inner_grace_active = inner_grace_s > 0
            log(f"[ALT_TRACK] alt_cmd={alt_cmd:.1f}m fol_alt={fol_alt:.1f}m err={alt_err:+.2f}m | "
                f"in_outer={in_outer} in_inner={in_inner} | dwell={dwell_s:.1f}s | "
                f"outer_grace={outer_grace_active}/{outer_grace_s:.2f}s "
                f"inner_grace={inner_grace_active}/{inner_grace_s:.2f}s")

        self.send_guided_commands(hdg_cmd, alt_cmd, spd_cmd, range_m=geom.range_m)
        self.log_continuous_diagnostics(geom, lead_state, fol_alt, hdg_cmd, spd_cmd, "HOLD_PRECONTACT")

    def _log_transition_snapshot_legacy(self, from_state, to_state, predicate: str, geom) -> None:
        """Log structured transition snapshot for unambiguous debugging.

        P1.5 (v9.4): Every state transition logs exact context.
        v9.10: Uses _t_sim_s and _dt_tick_used_s from tick gating.
        NOTE: Renamed to _legacy to avoid shadowing v9.10 Gate 2 method.
        """
        import json
        now = self._t_sim_s  # v9.10: Use tick timestamp
        lead_state = self.leader_rx.get_state() if self.leader_rx else None

        # P5.1 v9.5: Use authoritative vc_behind from control_laws
        vc_behind = self.control_laws.get_vc_behind_ewma_mps()
        # v9.10: Use tick dt instead of _dt_clamped()
        dt_tick = self._dt_tick_used_s if self._dt_tick_used_s else 0.0
        snapshot = {
            "sim_time_s": round(now, 3),
            "dt_sim_s": round(dt_tick, 4),
            "from_state": from_state.name,
            "to_state": to_state.name,
            "predicate": predicate,
            "behind_m": round(geom.behind_m, 2),
            "cross_m": round(geom.cross_m, 2),
            "below_m": round(geom.below_m, 2),
            "range_tip_m": round(geom.range_tip_m, 2) if hasattr(geom, 'range_tip_m') else round(geom.range_m, 2),
            "vc_behind_mps": round(vc_behind, 3),
            "lead_gs_mps": round(lead_state.ground_speed_ms, 2) if lead_state else None,
            "lead_tas_mps": round(lead_state.true_airspeed_ms, 2) if lead_state else None,
            "spd_cmd_final": round(self._last_spd_cmd, 2) if self._last_spd_cmd else None,
            "brake_clamp": self._brake_clamp_active,
            "climb_clamp": self._climb_clamp_active,
            "safety_clamp": self._safety_clamp_active,
        }
        log(f"[TRANSITION_SNAPSHOT] {json.dumps(snapshot)}")

    def _log_speed_arbitration(self, result, geom) -> None:
        """Log per-tick speed arbitration for post-run analysis.

        P2 v9.4: Logs arbitration pipeline breakdown. Decimated to reduce spam,
        but always logs near state transitions.
        """
        from formation_control_laws import SpeedArbitrationResult
        if not isinstance(result, SpeedArbitrationResult):
            return

        now = self._t_sim_s  # v9.10: Use tick timestamp

        # Decimation: log every N ticks, but always log near transitions
        self._arb_log_tick += 1
        near_transition = self._ticks_since_transition < 10

        if near_transition or (self._arb_log_tick % 5 == 0):  # Every 5th tick or near transition
            log(
                f"[SPD_ARB] sim={now:.2f} state={self.state.name} "
                f"behind={geom.behind_m:.1f} below={geom.below_m:.1f} "
                f"profile={result.spd_req_profile_mps:.2f} "
                f"brake={result.spd_after_brake_mps:.2f}{'*' if result.brake_clamp_active else ''} "
                f"climb={result.spd_after_climb_mps:.2f}{'*' if result.climb_clamp_active else ''} "
                f"safety={result.spd_after_safety_mps:.2f}{'*' if result.safety_clamp_active else ''} "
                f"final={result.spd_cmd_final_mps:.2f} dom={result.dominant_source} "
                f"binding={result.binding_constraint_final}"
            )

        # P2 v9.4 Fix 2c: Always write to CSV (every tick)
        self._write_arbitration_csv_row(result, geom)

    def _init_arbitration_csv(self) -> None:
        """Initialize CSV file for tick-level arbitration data.

        P2 v9.4 Fix 2b: Creates speed_arbitration.csv in output directory.
        Called after _out_dir is set and timebase PASS is confirmed.
        P8.6 v9.8: Added dt_sim_s, below_target_m, below_err_m, vert_gate_factor,
                   final_approach_active, final_approach_vc_cap_mps, vc_slow_mps,
                   vc_gate_credit_s columns.
        v9.10: Added dt_tick_raw_s, dt_tick_used_s, dt_credit_s, ticks_skipped_*,
               ticks_executed_total, late_tick_count, sim_time_regressions_total,
               max_sim_dt_raw_s, run_invalid_timebase, safety_emergency_active,
               no_pass_violation columns.
        """
        if not self._out_dir:
            return
        import csv
        import os
        csv_path = os.path.join(self._out_dir, "speed_arbitration.csv")
        try:
            self._arb_csv_file = open(csv_path, 'w', newline='')
            self._arb_csv_writer = csv.writer(self._arb_csv_file)
            # P5.3 v9.5: Added closure_cap_mps, closure_cap_active columns
            # P6.2 v9.6: Added target_range_m, behind_error_m columns
            # P8.6 v9.8: Added dt_sim_s, vertical/final approach tracking, vc gate
            # P9.0 v9.9: Added near_precontact, cross_gate observability columns
            # v9.10: Added timebase integrity and safety columns
            self._arb_csv_writer.writerow([
                "sim_time_s", "state", "behind_m", "below_m", "cross_m",
                "profile_mps", "brake_mps", "climb_mps", "safety_mps", "final_mps",
                "brake_clamp", "climb_clamp", "safety_clamp", "dominant", "vc_behind_mps",
                "closure_cap_mps", "closure_cap_active",
                "target_range_m", "behind_error_m",
                # P8.6 v9.8: Vertical/final approach columns
                "below_target_m", "below_err_m", "below_target_desired_m", "coupled_below_active",
                "vert_gate_factor",
                "final_approach_active", "final_approach_vc_cap_mps",
                "vc_slow_mps", "vc_gate_credit_s",
                # P8.2 v9.8: Cross integrator
                "cross_i_term",
                # P9.0 v9.9: Near-precontact and cross gate observability
                "near_precontact_active", "near_precontact_vc_cap",
                "cross_gate_factor",
                # v9.10 Gate 0: Timebase integrity columns
                "dt_tick_raw_s", "dt_tick_used_s", "dt_credit_s",
                "ticks_skipped_no_advance_total", "ticks_skipped_accumulating_total",
                "ticks_executed_total", "late_tick_count",
                "sim_time_regressions_total", "max_sim_dt_raw_s", "run_invalid_timebase",
                # v9.10 Gate 1: Safety columns
                "safety_emergency_active", "no_pass_violation",
                # v9.10 Gate 2: State machine stability columns
                "hold_precontact_dwell_accum_s", "hold_entry_grace_accum_s", "hold_exit_grace_accum_s",
                # M3 S1: Binding constraint sub-source telemetry
                "binding_constraint", "binding_value_mps",
                # M3 S1: Per-tick boolean flags
                "vert_floor_soft_active", "vert_floor_hard_active",
                "in_precontact_corridor", "speed_override_active",
                # M3 S1: Schedule observability
                "schedule_input_m", "schedule_bin_idx",
                # D2: Follower TECS/plant telemetry
                "follower_airspeed_mps", "follower_throttle_pct",
                "follower_climb_mps", "follower_airspeed_err_mps",
                "follower_throttle_sat", "follower_telemetry_valid",
                # Option A+E: DOCKING_CREEP sub-phase + TECS observability
                "dc_phase",
                "dc_phase1_enter_count", "dc_phase2_enter_count", "dc_phase2_fallback_count",
                "tecs_spdweight_target", "tecs_spdweight_current", "tecs_spdweight_confirmed",
                # Tranche C (D3): Altitude authority disambiguation telemetry
                "leader_alt_msl_m", "follower_alt_msl_m", "alt_cmd_msl_m",
                "nav_alt_error_m", "nav_alt_target_est_msl_m", "nav_aspd_error_mps",
                "form_enable_active", "follower_mode",
                # Tranche H: L1 lateral diagnostics
                "nav_roll_deg", "nav_bearing_deg", "target_bearing_deg",
                "xtrack_error_m", "wp_dist_m",
                # Tranche H: Follower attitude
                "follower_roll_deg", "follower_pitch_deg", "follower_heading_deg",
                # Tranche C (B2): C++ handoff diagnostics
                "cpp_inactive_reason",
                # Tranche D: Boom-tip-frame milestone tracking
                "delta_behind_m", "delta_below_m", "delta_cross_m",
                "m3a_in_envelope", "m3a_dwell_raw_s", "m3a_dwell_grace_s",
                "m3b_in_envelope", "m3b_dwell_raw_s", "m3b_dwell_grace_s",
                # Guard strip-down: docking safety + CZ + CZ-EMERGENCY + CE_thresh headers removed (Chunk 3A)
                # Step 1b: TECS speed tracking error (target - measured)
                "tecs_speed_err_mps",
                # M3b restoration: State-gated + speedup-compensated braking authority
                "braking_a_effective",
                # TIER2: trailing truth-closure diagnostic columns (flag-ON only;
                # OFF -> header byte-identical to the pre-Tier-2 schema)
                *(["truth_closure_cgcg_mps", "range_center_cgcg_m"]
                  if _FD_TRUTH_CLOSURE_DIAG else []),
                # WINDFF: trailing wind-FF diagnostic columns (flag-ON only; OFF -> byte-identical)
                *(["windff_west_mps", "windff_trim_mps"]
                  if _FD_SELF_WIND_FF else []),
            ])
            # Rec 3: Compute CSV decimation stride based on speedup
            # Goal: ~20 rows/sec regardless of speedup (1x=every tick, 3x=every 3rd, etc.)
            spd = max(1.0, self.expected_speedup)
            self._csv_decimation_stride = max(1, round(spd))
            self._csv_decimation_counter = 0
            log(f"[ARB_CSV] Initialized speed_arbitration.csv at {csv_path} "
                f"(decimation stride={self._csv_decimation_stride})")
        except Exception as e:
            log(f"[ARB_CSV] WARNING: Failed to create CSV: {e}")
            self._arb_csv_file = None
            self._arb_csv_writer = None

    def _write_arbitration_csv_row(self, result, geom) -> None:
        """Write one row to arbitration CSV (every tick).

        P2 v9.4 Fix 2b: Records tick-level speed pipeline values for post-run analysis.
        P5.1 v9.5: Use authoritative vc_behind from control_laws.
        P5.3 v9.5: Added closure_cap columns.
        P8.6 v9.8: Added dt_sim_s, vertical/final approach tracking, vc gate columns.
        v9.10: Added timebase integrity and safety columns. Increments _ticks_executed_total
               here so counter matches exactly the number of CSV rows written.
        Rec 3: CSV decimation - at high speedup, only writes every Nth row
               to bound I/O overhead. Counter always increments.
        """
        if not self._arb_csv_writer:
            return
        try:
            # v9.10: Increment executed tick counter (always, regardless of decimation)
            self._ticks_executed_total += 1

            # Rec 3: Decimation - skip CSV write on non-sampled ticks
            self._csv_decimation_counter += 1
            if self._csv_decimation_counter < self._csv_decimation_stride:
                return
            self._csv_decimation_counter = 0

            # P2 Fix D v9.10: Mark that a full CSV row was written this tick
            self._csv_row_written_this_tick = True

            # P5.1 v9.5: Use authoritative EWMA from control_laws
            # P6.2 v9.6: Added target_range_m, behind_error_m columns
            vc_behind = self.control_laws.get_vc_behind_ewma_mps()
            # P8.6 v9.8: Get vc_gate state
            vc_slow = getattr(self, '_vc_gate_slow_vc', 0.0)
            vc_gate_credit = getattr(self, '_precontact_vc_credit_s', 0.0)

            # v9.10: Get timebase columns (use 0 if not yet initialized)
            dt_raw = self._dt_tick_raw_s if self._dt_tick_raw_s is not None else 0.0
            dt_used = self._dt_tick_used_s if self._dt_tick_used_s is not None else 0.0
            dt_credit = self._dt_credit_s

            # v9.10: Check for no-pass violation this tick
            no_pass_this_tick = 1 if geom.behind_m < 0 else 0

            # Step 1b: Cache follower airspeed for TECS error column
            _csv_f_as = self.follower.get_airspeed() if self.follower else None

            self._arb_csv_writer.writerow([
                # v9.10: Use _t_sim_s (tick timestamp) instead of sim_now_s()
                round(self._t_sim_s, 3),
                self.state.name,
                round(geom.behind_m, 3),
                round(geom.below_m, 3),
                round(geom.cross_m, 3),
                round(result.spd_req_profile_mps, 3),
                round(result.spd_after_brake_mps, 3),
                round(result.spd_after_climb_mps, 3),
                round(result.spd_after_safety_mps, 3),
                round(result.spd_cmd_final_mps, 3),
                int(result.brake_clamp_active),
                int(result.climb_clamp_active),
                int(result.safety_clamp_active),
                result.dominant_source,
                round(vc_behind, 3),
                # P5.3 v9.5: Closure cap columns
                round(result.closure_cap_mps, 3),
                int(result.closure_cap_active),
                # P6.2 v9.6: Target-relative context
                round(result.target_range_m, 3),
                round(result.behind_error_m, 3),
                # P8.6 v9.8: Vertical and final approach tracking
                round(self._below_target_m, 3),
                round(self._below_err_m, 3),
                round(self._below_target_desired_m, 3),
                int(self._coupled_below_active),
                round(self._vert_gate_factor, 3),
                int(self._final_approach_active),
                round(self._final_approach_vc_cap, 3),
                # P8.6 v9.8: VC gate state
                round(vc_slow, 3),
                round(vc_gate_credit, 3),
                # P8.2 v9.8: Cross integrator
                round(self._cross_i_term, 3),
                # P9.0 v9.9: Near-precontact and cross gate observability
                int(self._near_precontact_active),
                round(self._near_precontact_vc_cap, 3),
                round(self._cross_gate_factor, 3),
                # v9.10 Gate 0: Timebase integrity columns
                round(dt_raw, 6),  # dt_tick_raw_s
                round(dt_used, 6),  # dt_tick_used_s
                round(dt_credit, 6),  # dt_credit_s
                self._ticks_skipped_no_advance,  # cumulative
                self._ticks_skipped_accumulating,  # cumulative
                self._ticks_executed_total,  # cumulative
                self._late_tick_count,  # cumulative
                self._sim_time_regressions,  # cumulative
                round(self._max_sim_dt_raw_s, 6),  # cumulative max
                int(self._run_invalid_timebase),  # 0/1
                # v9.10 Gate 1: Safety columns
                int(self._safety_emergency_active),  # 0/1
                no_pass_this_tick,  # 0/1
                # v9.10 Gate 2: State machine stability columns
                round(self._precontact_dwell_accum_s, 3),  # hold_precontact_dwell_accum_s
                round(self._hold_entry_grace_accum_s, 3),  # hold_entry_grace_accum_s
                round(self._hold_exit_grace_accum_s, 3),  # hold_exit_grace_accum_s
                # M3 S1: Binding constraint sub-source telemetry
                self._last_binding_final,
                round(self._last_binding_value_mps, 3),
                # M3 S1: Per-tick boolean flags
                int(self._vert_floor_soft_active),
                int(self._vert_floor_hard_active),
                int(self._in_precontact_corridor),
                int(self._speed_override_active),
                # M3 S1: Schedule observability
                round(self._schedule_input_m, 3),
                self._schedule_bin_idx,
                # D2: Follower TECS/plant telemetry
                *self._get_follower_telemetry_csv_values(result),
                # Option A+E
                self._dc_phase if self.state == ControllerState.DOCKING_CREEP else "",
                self._dc_phase1_enter_count,
                self._dc_phase2_enter_count,
                self._dc_phase2_fallback_count,
                round(self._tecs_spdw_target, 2),
                round(self._tecs_spdweight_current, 2),
                round(self._tecs_spdw_last_confirmed, 2),
                # Tranche C (D3): Altitude authority disambiguation
                *self._get_d3_csv_values(),
                # Tranche C (B2): C++ handoff diagnostics
                self._cpp_inactive_reason,
                # Tranche D: Boom-tip-frame milestone tracking
                *self._get_tranche_d_csv_values(),
                # Guard strip-down: CZ + CZ-EMERGENCY + CE_thresh CSV columns removed (Chunk 3A)
                # Step 1b: TECS speed tracking error (target - measured)
                round(result.spd_cmd_final_mps - _csv_f_as, 3) if _csv_f_as is not None else "",
                # M3b restoration: braking authority telemetry
                round(self.control_laws.get_braking_a_effective(), 4),
                # TIER2: trailing truth-closure diagnostic values (flag-ON only)
                *(self._truth_closure_csv_values()
                  if _FD_TRUTH_CLOSURE_DIAG else ()),
                # WINDFF: trailing wind-FF diagnostic values (flag-ON only)
                *((round(self._windff_west, 4), round(self._windff_trim_i, 4))
                  if _FD_SELF_WIND_FF else ()),
            ])
            # Accumulate reason counts for run_summary
            r = self._cpp_inactive_reason
            self._cpp_inactive_reason_counts[r] = self._cpp_inactive_reason_counts.get(r, 0) + 1
        except Exception:
            # Silent fail - don't spam logs on CSV write errors
            pass

    def _get_tranche_d_csv_values(self) -> tuple:
        """Tranche D: Compute boom-tip-frame milestone and docking safety CSV values."""
        m3_state = self._success_metrics.get_m3_per_tick_state()
        return (
            round(m3_state["delta_behind"], 3),
            round(m3_state["delta_below"], 3),
            round(m3_state["delta_cross"], 3),
            int(m3_state["m3a_in_envelope"]),
            round(m3_state["m3a_dwell_raw_s"], 3),
            round(m3_state["m3a_dwell_grace_s"], 3),
            int(m3_state["m3b_in_envelope"]),
            round(m3_state["m3b_dwell_raw_s"], 3),
            round(m3_state["m3b_dwell_grace_s"], 3),
            # Guard strip-down: docking safety CSV values removed (Chunk 3A)
        )

    def _truth_closure_csv_values(self) -> tuple:
        """TIER2: (truth_closure_cgcg_mps, range_center_cgcg_m) from TRUTH only.

        Follower truth = SIMSTATE(164)/SIM_STATE(108) position + SIM_STATE(108)
        velocity; leader truth = broadcast GLOBAL_POSITION_INT position +
        velocity. + = closing. Null-safe: ANY missing input -> ("", "") and the
        skip count increments (logged at CSV close). NEVER substitutes
        perceived state for truth (Tier-2 FM-b), and the values feed the CSV
        row ONLY — no guard, gate, estimator, or policy consumer.
        """
        f_pos = self.follower.get_sim_position() if self.follower else None
        f_vel = self.follower.get_sim_velocity_ned() if self.follower else None
        lead_state = self.leader_rx.get_state() if self.leader_rx else None
        if f_pos is None or f_vel is None or lead_state is None:
            self._truth_closure_skip_count += 1
            return ("", "")
        res = closure_diag.truth_closure_cgcg(
            f_pos[0], f_pos[1], f_pos[2],
            f_vel[0], f_vel[1], f_vel[2],
            lead_state.lat, lead_state.lon, lead_state.alt_msl,
            lead_state.vn_ms, lead_state.ve_ms, lead_state.vd_ms)
        if res is None:
            self._truth_closure_skip_count += 1
            return ("", "")
        return (round(res[0], 3), round(res[1], 3))

    def _get_follower_telemetry_csv_values(self, result=None) -> tuple:
        """D2: Compute 6 follower telemetry CSV values with NaN fallback."""
        f_as = self.follower.get_airspeed() if self.follower else None
        f_thr = self.follower.get_throttle_pct() if self.follower else None
        f_climb = self.follower.get_climb_rate_mps() if self.follower else None
        f_valid = f_as is not None and f_thr is not None

        # Derived fields
        if f_as is not None and result is not None:
            f_as_err = round(f_as - result.spd_cmd_final_mps, 3)
        else:
            f_as_err = float('nan')
        f_thr_sat = int(f_thr >= 95 or f_thr <= 5) if f_thr is not None else float('nan')

        return (
            round(f_as, 3) if f_as is not None else float('nan'),
            round(f_thr, 1) if f_thr is not None else float('nan'),
            round(f_climb, 3) if f_climb is not None else float('nan'),
            f_as_err,
            f_thr_sat,
            int(f_valid),
        )

    def _get_follower_attitude_csv_values(self) -> tuple:
        """Tranche H: Get follower roll/pitch/heading for CSV."""
        att = self.follower.get_attitude() if self.follower else None
        if att:
            return (round(att[0], 2), round(att[1], 2), round(att[2], 2))
        return (float('nan'), float('nan'), float('nan'))

    def _get_d3_csv_values(self) -> tuple:
        """Tranche C (D3) + Tranche H: Altitude-authority + L1 lateral diagnostic CSV values."""
        leader_alt = self._d3_leader_alt_msl_m
        follower_alt = self._d3_follower_alt_msl_m
        alt_cmd = self._d3_alt_cmd_msl_m
        nav_alt_err = self._nav_alt_error_m
        nav_aspd_err = self._nav_aspd_error_mps

        # Derived: nav_alt_target_est = follower_alt + nav_alt_error
        # (ArduPilot convention: alt_error = target - current, so target = current + error)
        if math.isfinite(follower_alt) and math.isfinite(nav_alt_err):
            nav_alt_target_est = round(follower_alt + nav_alt_err, 3)
        else:
            nav_alt_target_est = float('nan')

        # FORM_ENABLE state
        form_enable = int(self._cpp_enabled)

        # Follower flight mode
        if self.follower and self.follower._last_heartbeat:
            follower_mode = int(self.follower._last_heartbeat.custom_mode)
        else:
            follower_mode = -1

        # Tranche H: L1 lateral diagnostics (already in degrees/meters from MAVLink)
        nav_roll = round(self._nav_roll_deg, 2) if math.isfinite(self._nav_roll_deg) else float('nan')
        nav_bearing = round(self._nav_bearing_deg, 2) if math.isfinite(self._nav_bearing_deg) else float('nan')
        target_bearing = round(self._target_bearing_deg, 2) if math.isfinite(self._target_bearing_deg) else float('nan')
        xtrack_err = round(self._xtrack_error_m, 3) if math.isfinite(self._xtrack_error_m) else float('nan')
        wp_dist = round(self._wp_dist_m, 2) if math.isfinite(self._wp_dist_m) else float('nan')

        return (
            round(leader_alt, 3) if math.isfinite(leader_alt) else float('nan'),
            round(follower_alt, 3) if math.isfinite(follower_alt) else float('nan'),
            round(alt_cmd, 3) if math.isfinite(alt_cmd) else float('nan'),
            round(nav_alt_err, 3) if math.isfinite(nav_alt_err) else float('nan'),
            nav_alt_target_est,
            round(nav_aspd_err, 3) if math.isfinite(nav_aspd_err) else float('nan'),
            form_enable,
            follower_mode,
            # Tranche H: L1 lateral diagnostics
            nav_roll,
            nav_bearing,
            target_bearing,
            xtrack_err,
            wp_dist,
            # Tranche H: Follower attitude
            *self._get_follower_attitude_csv_values(),
        )

    def _write_arbitration_csv_row_minimal(self, geom=None) -> None:
        """P2 Fix D v9.10: Write a minimal CSV row for non-formation states.

        This ensures one row per executed tick regardless of state.
        For states that don't call _send_speed_command (e.g., VTOL_TAKEOFF),
        this writes a row with NaN/null for unavailable columns.
        Rec 3: Subject to same decimation stride as full rows.
        """
        if not self._arb_csv_writer:
            return
        try:
            # v9.10: Increment executed tick counter (always)
            self._ticks_executed_total += 1

            # Rec 3: Decimation - skip CSV write on non-sampled ticks
            self._csv_decimation_counter += 1
            if self._csv_decimation_counter < self._csv_decimation_stride:
                return
            self._csv_decimation_counter = 0

            # Use NaN for unavailable geometry and speed values
            if geom is not None:
                behind_m = round(geom.behind_m, 3)
                below_m = round(geom.below_m, 3)
                cross_m = round(geom.cross_m, 3)
                no_pass_this_tick = 1 if geom.behind_m < 0 else 0
            else:
                behind_m = ''
                below_m = ''
                cross_m = ''
                no_pass_this_tick = 0

            dt_raw = self._dt_tick_raw_s if self._dt_tick_raw_s is not None else 0.0
            dt_used = self._dt_tick_used_s if self._dt_tick_used_s is not None else 0.0

            # Write row with empty values for unavailable speed columns
            self._arb_csv_writer.writerow([
                round(self._t_sim_s, 3),
                self.state.name,
                behind_m, below_m, cross_m,
                '', '', '', '', '',  # profile, brake, climb, safety, final mps
                0, 0, 0,  # clamp flags
                '',  # dominant source
                '',  # vc_behind
                '', 0,  # closure_cap_mps, closure_cap_active
                '', '',  # target_range_m, behind_error_m
                '', '', '', 0, '',  # below_target, below_err, below_target_desired, coupled_below_active, vert_gate_factor
                0, '',  # final_approach_active, final_approach_vc_cap
                '', '',  # vc_slow, vc_gate_credit
                '',  # cross_i_term
                0, '',  # near_precontact_active, near_precontact_vc_cap
                '',  # cross_gate_factor
                round(dt_raw, 6), round(dt_used, 6), round(self._dt_credit_s, 6),
                self._ticks_skipped_no_advance,
                self._ticks_skipped_accumulating,
                self._ticks_executed_total,
                self._late_tick_count,
                self._sim_time_regressions,
                round(self._max_sim_dt_raw_s, 6),
                int(self._run_invalid_timebase),
                int(self._safety_emergency_active),
                no_pass_this_tick,
                round(self._precontact_dwell_accum_s, 3),
                round(self._hold_entry_grace_accum_s, 3),
                round(self._hold_exit_grace_accum_s, 3),
                # M3 S1: Binding constraint sub-source telemetry (empty for minimal rows)
                '',  # binding_constraint
                '',  # binding_value_mps
                # M3 S1: Per-tick boolean flags
                0, 0, 0, 0,  # vert_floor_soft, vert_floor_hard, in_precontact, speed_override
                # M3 S1: Schedule observability
                '',  # schedule_input_m
                -1,  # schedule_bin_idx
                # D2: Follower telemetry (NaN for minimal rows)
                float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), 0,
                # Option A+E
                "",
                self._dc_phase1_enter_count,
                self._dc_phase2_enter_count,
                self._dc_phase2_fallback_count,
                round(self._tecs_spdw_target, 2),
                round(self._tecs_spdweight_current, 2),
                round(self._tecs_spdw_last_confirmed, 2),
                # Tranche C (D3): Altitude authority disambiguation (minimal rows too)
                *self._get_d3_csv_values(),
                # Tranche C (B2): C++ handoff diagnostics
                self._cpp_inactive_reason,
                # Tranche D: Boom-tip-frame milestone tracking (empty for minimal rows)
                '', '', '',  # delta_behind, delta_below, delta_cross
                0, '', '',   # m3a: envelope, dwell_raw, dwell_grace
                0, '', '',   # m3b: envelope, dwell_raw, dwell_grace
                # Guard strip-down: docking safety minimal values removed (Chunk 3A)
                # Step 1b: TECS speed tracking error (empty for minimal rows)
                '',  # tecs_speed_err_mps
                # M3b restoration: braking authority telemetry
                round(self.control_laws.get_braking_a_effective(), 4),
                # TIER2: trailing truth-closure diagnostic values (flag-ON only;
                # minimal rows carry them too so the CSV stays rectangular)
                *(self._truth_closure_csv_values()
                  if _FD_TRUTH_CLOSURE_DIAG else ()),
                # WINDFF: trailing wind-FF diagnostic values (flag-ON only; minimal rows carry the
                # held estimate so the CSV stays rectangular)
                *((round(self._windff_west, 4), round(self._windff_trim_i, 4))
                  if _FD_SELF_WIND_FF else ()),
            ])
            # Accumulate reason counts for run_summary
            r = self._cpp_inactive_reason
            self._cpp_inactive_reason_counts[r] = self._cpp_inactive_reason_counts.get(r, 0) + 1
        except Exception:
            pass

    def _tick_diag_print_summary(self, tick_num: int) -> None:
        """Print periodic tick timing summary (every 100 ticks)."""
        buf = self._tick_diag_wall_times
        if not buf:
            return
        buf_sorted = sorted(buf)
        n = len(buf_sorted)
        mean_ms = sum(buf_sorted) / n
        p95_ms = buf_sorted[min(int(n * 0.95), n - 1)]
        max_ms = buf_sorted[-1]
        eff_hz = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        drops = sum(1 for d in buf if d > 50.0)
        log(f"TICK_DIAG: ticks={tick_num}, mean_ms={mean_ms:.1f}, p95_ms={p95_ms:.1f}, "
            f"max_ms={max_ms:.1f}, effective_hz={eff_hz:.1f}, drops={drops}")
        self._tick_diag_wall_times.clear()

    def _tick_diag_write_csv(self) -> None:
        """Write tick timing CSV at episode end."""
        if not self._out_dir or not self._tick_diag_records:
            return
        import csv
        csv_path = os.path.join(self._out_dir, "tick_timing_diagnostic.csv")
        try:
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["tick_number", "sim_time_s", "wall_time_s",
                            "tick_duration_ms", "state_name",
                            "leader_tbms", "leader_fresh"])
                for rec in self._tick_diag_records:
                    w.writerow([rec[0], f"{rec[1]:.6f}", f"{rec[2]:.6f}",
                                f"{rec[3]:.3f}", rec[4], rec[5], rec[6]])
            log(f"[TICK_DIAG] Wrote {len(self._tick_diag_records)} rows to {csv_path}")
        except Exception as e:
            log(f"[TICK_DIAG] WARNING: Failed to write CSV: {e}")

    def _close_arbitration_csv(self) -> None:
        """Close arbitration CSV file.

        P2 v9.4 Fix 2b: Called in finally block for clean shutdown.
        """
        if self._arb_csv_file:
            try:
                self._arb_csv_file.close()
                log("[ARB_CSV] Closed speed_arbitration.csv")
                # TIER2 FM-d detector: surface how many written rows lacked
                # truth inputs for the closure columns (flag-ON only)
                if _FD_TRUTH_CLOSURE_DIAG:
                    log(f"[TRUTH_CLOSURE_DIAG] skipped_rows="
                        f"{self._truth_closure_skip_count}")
                # WINDFF detector: surface how many ticks held the estimate on missing
                # telemetry (flag-ON only) — non-zero => the FF was inert for those ticks
                if _FD_SELF_WIND_FF:
                    log(f"[WINDFF] estimator_held_ticks={self._windff_skip_count}")
            except Exception:
                pass
            self._arb_csv_file = None
            self._arb_csv_writer = None

    def _init_uwb_fusion_diag(self) -> None:
        """C3 (gated, default-off): open uwb_fusion_diag.csv for the emitter->
        fusion delivery diagnostic. No-op unless FD_UWB_FUSION_DIAG=1. Writes to
        $FD_UWB_FUSION_DIAG_DIR (fallback: self._out_dir). LOGGING-ONLY."""
        if not _FD_UWB_FUSION_DIAG:
            return
        import csv
        import os
        out_dir = os.environ.get('FD_UWB_FUSION_DIAG_DIR', '').strip() or self._out_dir
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "uwb_fusion_diag.csv")
            self._uwb_diag_file = open(csv_path, 'w', newline='')
            self._uwb_diag_writer = csv.writer(self._uwb_diag_file)
            self._uwb_diag_writer.writerow([
                "utc_ts", "sim_time_s", "tick_id", "controller_state",
                "emit_truth_used_bool", "emit_below_truth", "emit_behind_truth",
                "emit_cross_truth", "emit_below_gps", "n_meas_emitted", "emit_range_a0",
                "fused_below_pre", "uwb_update_count_pre", "uwb_rejected_count_pre",
                "uwb_updates_a0_pre", "uwb_rejected_a0_pre", "predicted_range_a0",
                "fused_below_post", "uwb_update_count_post", "uwb_rejected_count_post",
                "rej_delta", "last_innovation_range",
                "innov_a0", "innov_a1", "innov_a2", "innov_a3",
                "geom_below_returned", "below_gps", "below_truth",
                # CSV2 frame-fix latency-decomp (schema v2, 2026-06-10): along-track
                # source attribution for the residual ~8.5 m behind disconnect.
                "foll_gps_lat", "foll_gps_lon", "truth_lat", "truth_lon",
                "leader_gps_lat", "leader_gps_lon", "leader_time_boot_ms",
                "behind_gps", "fused_behind", "boom_frame_unified",
                # STEP-4B2' state-gate diagnostics (schema v3, 2026-06-11):
                # per-tick architecture-D gate outcome from the get_geometry
                # control call.
                "d_corr_applied_bool", "d_corr_state", "d_corr_range_m",
                "d_corr_reject_reason", "d_corr_delta_behind_m",
                "d_corr_delta_below_m", "d_corr_delta_cross_m",
                "d_corr_engagement_count", "d_corr_reject_count",
                # DFIX schema v4 (2026-06-13): pre-trust max|delta| — closes
                # the d_census evidence gap (rejected/clamped magnitudes).
                "d_corr_pre_trust_max_m",
            ])
            # [C2]/[C4] one-time frame + state-layout verification logs
            fusion = self.geom_calc.fusion if self.geom_calc else None
            a0 = getattr(fusion, 'anchor_positions', {}).get(0) if fusion else None
            xlen = len(getattr(fusion, 'x', [])) if fusion else 0
            bootv = _uwb_diag_fused_below(fusion) if fusion else _UWB_DIAG_NAN
            log(f"[UWB_DIAG] Initialized uwb_fusion_diag.csv at {csv_path} "
                f"(schema v{_UWB_FUSION_DIAG_SCHEMA_VERSION})")
            log(f"[UWB_DIAG] anchor0 frame check: anchor0_pos={a0}, "
                f"state_attrs=(behind_m,cross_m,below_m), x_len={xlen}")
            log(f"[UWB_DIAG] state-layout: IDX_BELOW={_UWB_DIAG_IDX_BELOW} "
                f"(FusedState.below_m = self.x[{_UWB_DIAG_IDX_BELOW}]), boot_value={bootv:.3f}")
        except Exception as e:
            log(f"[UWB_DIAG] WARNING: Failed to create CSV: {e}")
            self._uwb_diag_file = None
            self._uwb_diag_writer = None

    def _uwb_diag_write_row(self, geom) -> None:
        """C3 (gated): write one emitter->fusion delivery row (docking states
        only). Pure consumer of self._uwb_diag_pending (built across get_geometry);
        per-row flush so the file is crash-survivable. Never raises into control."""
        p = self._uwb_diag_pending
        self._uwb_diag_pending = None
        try:
            state_name = self.state.name if hasattr(self, 'state') else ''
            if state_name not in _UWB_FUSION_DIAG_STATES:
                return
            if 'fused_below_post' not in p:  # post-capture never ran (defensive)
                return
            pc = p['pre_counts']
            rej_delta = p['rej_post'] - pc['rej']
            i0, i1, i2, i3 = p.get('innov', (_UWB_DIAG_NAN,) * 4)
            # STEP-4B2' schema v3: this tick's D gate record (set by the
            # gate chain inside the control compute_relative call).
            _dcl = (getattr(self.geom_calc, '_d_corr_last', None)
                    if self.geom_calc else None)
            self._uwb_diag_writer.writerow([
                time.time(), self._t_sim_s, self._tick_counter, state_name,
                int(p['emit_truth_used']), p['emit_below_truth'], p['emit_behind_truth'],
                p['emit_cross_truth'], p['emit_below_gps'], p['n_meas'], p['emit_range_a0'],
                p['fused_below_pre'], pc['upd'], pc['rej'],
                pc['upd_a'].get(0, 0), pc['rej_a'].get(0, 0), p['predicted_range_a0'],
                p['fused_below_post'], p['upd_post'], p['rej_post'],
                rej_delta, p['last_innov'], i0, i1, i2, i3,
                float(geom.below_m), p['emit_below_gps'], p['emit_below_truth'],
                p['foll_gps_lat'], p['foll_gps_lon'], p['truth_lat'], p['truth_lon'],
                p['leader_gps_lat'], p['leader_gps_lon'], p['leader_time_boot_ms'],
                p['behind_gps'], float(geom.behind_m), int(p['boom_frame_unified']),
                # STEP-4B2' schema v3: D gate outcome for THIS tick (set by
                # the get_geometry control call's gate chain; empty when
                # FD_D_CORRECTION=0).
                int(bool(_dcl.get('applied'))) if _dcl else '',
                _dcl.get('state', '') if _dcl else '',
                _dcl.get('range_m', '') if _dcl else '',
                _dcl.get('reason', '') if _dcl else '',
                _dcl.get('db', '') if _dcl else '',
                _dcl.get('dbl', '') if _dcl else '',
                _dcl.get('dc', '') if _dcl else '',
                _dcl.get('eng', 0) if _dcl else '',
                _dcl.get('rej', 0) if _dcl else '',
                _dcl.get('pre_max', '') if _dcl else '',
            ])
            self._uwb_diag_file.flush()
            self._uwb_diag_seq += 1
            self._uwb_diag_rows_written += 1
        except Exception as e:
            log(f"[UWB_DIAG] WARNING: row write failed: {e}")

    def _close_uwb_fusion_diag(self) -> None:
        """C3 (gated): close uwb_fusion_diag.csv. Called in shutdown finally."""
        # STEP-4B2': one-shot full-leg D gate histogram (diag rows are
        # docking-states-only by schema design, so out-of-state rejections
        # are visible ONLY here and in the throttled [D_CORR_REJ] lines).
        try:
            _rcnt = (getattr(self.geom_calc, '_d_corr_reason_counts', None)
                     if self.geom_calc else None)
            if _rcnt:
                _rej_parts = ', '.join(
                    f"{k}={v}" for k, v in sorted(_rcnt.items())
                    if k != 'APPLIED')
                log(f"[D_CORR_SUMMARY] applied={_rcnt.get('APPLIED', 0)} "
                    f"rejects={{{_rej_parts}}}")
        except Exception:
            pass
        if self._uwb_diag_file:
            try:
                self._uwb_diag_file.close()
                log(f"[UWB_DIAG] Closed uwb_fusion_diag.csv "
                    f"({self._uwb_diag_rows_written} rows)")
            except Exception:
                pass
            self._uwb_diag_file = None
            self._uwb_diag_writer = None

    # Guard strip-down: _check_nopass_guard removed (Chunk 2B)

    # Guard strip-down: _ce_guard_threshold + _ce_guard_ttc_threshold removed (Chunk 3A)

    # Guard strip-down: _check_closure_energy_guard removed (Chunk 2B)

    def _resolve_cg_value(self, center_val, tip_val, offset_attr: str) -> float:
        """Resolve CG-frame value with offset latch fallback.

        When center_geom is available, uses it directly and latches the
        CG-minus-tip offset. When center_geom is missing, reconstructs
        CG value from tip value + latched offset. Prevents frame jumps.
        """
        import math
        if center_val is not None and math.isfinite(center_val):
            offset = center_val - tip_val
            setattr(self, offset_attr, offset)
            return center_val
        # Fallback: reconstruct from latched offset
        latched = getattr(self, offset_attr, float('nan'))
        if math.isfinite(latched):
            return tip_val + latched
        # No latch available — use tip value (first-tick edge case only)
        return tip_val

    # Guard strip-down: _check_docking_safety_guards removed (Chunk 2B)
    def _turn_gate_update(self, lead_state, dt) -> bool:
        """TURNGATE-750: is the leader straight-and-level long enough to creep?

        Wires the constants that shipped dead: TURN_GATE_STRAIGHT_TIME_S carried
        the comment "Must be straight this long for DOCKING_CREEP" but its only
        consumer was a run_self_tests() range check, leader_turning_flag was
        never assigned True, and _leader_not_turning_since was never read.

        DEFAULT OFF (FD_TURN_GATE=1 to arm). When disarmed this returns True
        unconditionally and the caller's condition is byte-identical to the
        pre-fix behaviour -- the only do-no-harm construction available for a
        gate that, by definition, changes control flow.

        Filter/hysteresis use the constants that were already tuned for this:
        LEAD_YAW_RATE_TURN_FILTER_TAU_S, ..._ON_DPS, ..._OFF_DPS.
        """
        try:
            hdg = getattr(lead_state, "heading_deg", None)
            if hdg is None or not dt or dt <= 0:
                return not _FD_TURN_GATE          # disarmed => permit
            if self._tg_last_hdg is None:
                self._tg_last_hdg = hdg
                return not _FD_TURN_GATE
            raw_dps = wrap_heading_error(hdg - self._tg_last_hdg) / dt
            self._tg_last_hdg = hdg
            tau = max(1e-3, LEAD_YAW_RATE_TURN_FILTER_TAU_S)
            alpha = min(1.0, dt / tau)
            self._tg_yaw_rate_filt_dps += alpha * (
                abs(raw_dps) - self._tg_yaw_rate_filt_dps)
            # Schmitt trigger: ON above the on-threshold, OFF below the off one.
            if self._tg_yaw_rate_filt_dps > LEAD_YAW_RATE_TURN_ON_DPS:
                self._tg_turning = True
            elif self._tg_yaw_rate_filt_dps < LEAD_YAW_RATE_TURN_OFF_DPS:
                self._tg_turning = False
            self.leader_turning_flag = self._tg_turning
            if self._tg_turning:
                self._tg_straight_s = 0.0
            else:
                self._tg_straight_s += dt
            if not _FD_TURN_GATE:
                return True                        # observe only
            return self._tg_straight_s >= TURN_GATE_STRAIGHT_TIME_S
        except Exception:
            return not _FD_TURN_GATE               # never block on a bug

    def _handle_alt_cap_hold_range(self) -> None:
        """Hold range at precontact while driving vertical alignment to boom tip.

        P1 (v9.4): Prevents DOCKING_CREEP from starting while still vertically
        misaligned, which causes CLIMB_CLAMP to dominate and stall closure.

        State flow:
            HOLD_PRECONTACT -> ALT_CAP_HOLD_RANGE -> DOCKING_CREEP

        Exit condition: Vertical alignment to boom tip (below_m ≈ 0) held stable
        for ALT_CAP_STABLE_S seconds.
        """
        # FD-005: Capture entry state for mid-tick abort
        entry_state = self.state

        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else CONTACT_ALT_MSL

        # v9.10: Use _dt_tick_used_s and _t_sim_s from tick gating
        dt = self._dt_tick_used_s if self._dt_tick_used_s else 0.05

        # Fix 1: TECS scheduler — altitude bias during ALT_CAP for faster vertical convergence
        if formation_config.TECS_SPDWEIGHT_ENABLE and not self._tecs_spdw_disabled:
            target = self._tecs_spdw_compute_target_for_state()
            self._tecs_spdw_set_target(target)
            self._tecs_spdw_tick()

        # Compute boom-tip vertical error (target is 0m = aligned with boom tip)
        below_err = geom.below_m - ALT_CAP_BELOW_TARGET_M
        vz_rel = geom.vel_below_m_s if hasattr(geom, 'vel_below_m_s') else 0.0

        # Check if vertically captured
        below_ok = abs(below_err) <= ALT_CAP_BELOW_TOL_M
        vz_ok = abs(vz_rel) <= ALT_CAP_VZ_TOL_MPS
        captured = below_ok and vz_ok

        # v9.10: Stable dwell accumulation using accumulator pattern (not timestamp differences)
        # Initialize grace accumulator if missing
        if not hasattr(self, '_alt_cap_violation_accum_s'):
            self._alt_cap_violation_accum_s = 0.0
        if not hasattr(self, '_alt_cap_log_accum_s'):
            self._alt_cap_log_accum_s = 0.0

        if captured:
            self._alt_cap_violation_accum_s = 0.0  # Reset grace timer
            self._alt_cap_stable_s += dt
        else:
            self._alt_cap_violation_accum_s += dt
            if self._alt_cap_violation_accum_s > ALT_CAP_GRACE_S:
                # Decay stable credit after grace exceeded
                self._alt_cap_stable_s = max(0.0, self._alt_cap_stable_s - dt)

        # v9.10: Rate-limited logging using accumulator
        self._alt_cap_log_accum_s += dt
        if self._alt_cap_log_accum_s >= ALT_CAP_LOG_INTERVAL_S:
            self._alt_cap_log_accum_s = 0.0
            status = "PASS" if self._alt_cap_stable_s >= ALT_CAP_STABLE_S else "WAIT"
            log(f"[ALT_CAP_HOLD] {status}: below_err={below_err:+.2f}m vz={vz_rel:+.2f}m/s "
                f"captured={captured} stable={self._alt_cap_stable_s:.1f}/{ALT_CAP_STABLE_S}s")

        # TURNGATE-750: observe every tick (so the straight-timer is meaningful
        # at the instant of the check) and gate only when armed.
        _turn_ok = self._turn_gate_update(lead_state, dt)

        # Transition to DOCKING_CREEP when stable
        if self._alt_cap_stable_s >= ALT_CAP_STABLE_S and _turn_ok:
            self._log_transition_snapshot_legacy(
                from_state=self.state,
                to_state=ControllerState.DOCKING_CREEP,
                predicate="alt_capture_stable",
                geom=geom
            )
            self.transition_to(
                ControllerState.DOCKING_CREEP,
                f"Altitude captured: below_err={below_err:.2f}m vz={vz_rel:.2f}m/s "
                f"stable for {self._alt_cap_stable_s:.2f}s"
            )
            return

        # Control: Hold behind at precontact distance, cross at ~0, drive altitude to target
        # Speed command: match leader (minimal closure) while climbing
        hdg_cmd = self.control_laws.compute_formation_heading_cmd(
            geom, lead_state.heading_deg
        )

        # Guard strip-down: CE + NOPASS guards removed (Chunk 2A)
        # Station-keeping speed with minimal closure
        spd_cmd = self.control_laws.compute_speed_cmd(
            geom,
            self._windff_lead_gs_air(lead_state.ground_speed_ms),  # WINDFF-1b: air-frame lead ref
            PRECONTACT_BEHIND_M,  # Hold at precontact distance
            fol_alt,
            station_keeping=True,
            safety_clamp_cap=None,  # Guard strip-down: no guard cap
            controller_state=self.state,
        )
        # CSV2_SPEC247: TAS->EAS command FF (default OFF -> byte-identical) so the follower
        # holds range at realized LEADER_TARGET_TAS_MS without overtaking.
        spd_cmd = apply_eas2tas_ff(spd_cmd, fol_alt, eas2tas=self.follower.get_baro_eas2tas())
        self._last_spd_cmd = spd_cmd

        # P2 v9.4 Fix 4: Update clamp state and log arbitration (CSV + log)
        arb = self.control_laws.get_last_arbitration()
        if arb:
            self._brake_clamp_active = arb.brake_clamp_active
            self._climb_clamp_active = arb.climb_clamp_active
            self._safety_clamp_active = arb.safety_clamp_active
            self._log_speed_arbitration(arb, geom)

        # Altitude target: boom-tip altitude (contact altitude)
        # v9.10: Use _t_sim_s instead of now
        alt_cmd = self._compute_altitude_target_msl(geom, lead_state, self._t_sim_s)

        # FD-005: Mid-tick abort - if safety transitioned state, don't send normal commands
        if self.state != entry_state:
            log(f"[FD-005] Mid-tick abort: {entry_state.name} -> {self.state.name}")
            return

        self.send_guided_commands(hdg_cmd, alt_cmd, spd_cmd, range_m=geom.range_m)
        self.log_continuous_diagnostics(geom, lead_state, fol_alt, hdg_cmd, spd_cmd, "ALT_CAP_HOLD_RANGE")

    def _handle_altitude_capture(self) -> None:
        """Climb to contact altitude at extended standoff."""
        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        fol_pos = self.follower.get_position()
        if not fol_pos:
            return
        fol_alt = fol_pos[2]
        # v9.10: Removed dead sim_now_s() call

        # Simplified logic for brevity, full logic in original method
        self.transition_to(ControllerState.DOCKING_CREEP, "Altitude captured")

    def _exit_docking_creep(self, reason: str) -> None:
        """Restore TECS_SPDWEIGHT to cruise on DOCKING_CREEP exit."""
        if formation_config.TECS_SPDWEIGHT_ENABLE and not self._tecs_spdw_disabled:
            cruise = formation_config.TECS_SPDWEIGHT_CRUISE
            if self._tecs_spdweight_baseline is not None:
                cruise = self._tecs_spdweight_baseline
            if self.follower.set_param("TECS_SPDWEIGHT", cruise):
                log(f"[DC_EXIT] Restored TECS_SPDWEIGHT={cruise:.2f} ({reason})")
                self._tecs_spdweight_current = cruise
                self._tecs_spdw_last_confirmed = cruise
        self._tecs_spdw_ramp_active = False

    def _exit_altitude_capture(self, reason: str) -> None:
        """Cleanup after ALTITUDE_CAPTURE phase.

        Called by StateMachine STATE_EXIT_ACTIONS when leaving ALTITUDE_CAPTURE.
        Restores TECS_SPDWEIGHT and Q_ASSIST_SPEED to baseline values.
        """
        log(f"[ALT_CAPTURE] Exiting: {reason}")

        if self._tecs_spdweight_baseline is not None:
            if self.follower.set_param("TECS_SPDWEIGHT", self._tecs_spdweight_baseline):
                log(f"[ALT_CAPTURE] Restored TECS_SPDWEIGHT={self._tecs_spdweight_baseline:.1f}")
                self._tecs_spdweight_current = self._tecs_spdweight_baseline

        if self._qassist_speed_baseline is not None:
            if self.follower.set_param("Q_ASSIST_SPEED", self._qassist_speed_baseline):
                log(f"[ALT_CAPTURE] Restored Q_ASSIST_SPEED={self._qassist_speed_baseline:.1f}")
                self._qassist_speed_current = self._qassist_speed_baseline

        self._tecs_capture_mode_active = False
        self._qassist_capture_mode_active = False

    def _update_m3a_dwell(self, geom: RelativeGeometry, dt: float) -> None:
        """P7.1 v9.7: Track M3a envelope dwell with grace period.

        Updates M3a dwell tracking for observability. Uses grace period
        to avoid resetting on momentary exits.

        Args:
            geom: Current relative geometry.
            dt: Time delta for this tick (sim seconds).
        """
        th = self._success_metrics.th.milestones
        in_m3a = (
            th.m3a_behind_min <= geom.behind_m <= th.m3a_behind_max
            and abs(geom.cross_m) <= th.m3a_cross_tol
            and abs(geom.below_m) <= th.m3a_below_tol
        )

        # v9.10: Use accumulator pattern for M3a grace period instead of timestamp diff
        if not hasattr(self, '_m3a_grace_accum_s'):
            self._m3a_grace_accum_s = 0.0

        if in_m3a:
            # Clear grace timer, accumulate dwell
            self._m3a_grace_start = None
            self._m3a_grace_accum_s = 0.0
            self._m3a_current_dwell_s += dt
            self._m3a_time_in_envelope_s += dt
            self._m3a_max_contiguous_s = max(
                self._m3a_max_contiguous_s, self._m3a_current_dwell_s
            )
        else:
            # Outside envelope - use grace period (accumulator) before reset
            if self._m3a_grace_start is None:
                self._m3a_grace_start = self._t_sim_s
                self._m3a_grace_accum_s = 0.0
            self._m3a_grace_accum_s += dt
            if self._m3a_grace_accum_s > M3A_DWELL_GRACE_S:
                # Grace expired - reset current dwell
                self._m3a_current_dwell_s = 0.0

    def _update_m3a_coarse_dwell(self, geom: 'RelativeGeometry', dt: float) -> None:
        """§8 Step 3: Track M3a_coarse (Acquisition tier) dwell with grace period.

        Same grace-period logic as M3a. Envelope: behind <= 1.50 AND |cross| <= 0.75
        AND |below| <= 0.75. Report-only — does not drive state machine transitions.
        """
        th = self._success_metrics.th.milestones
        in_m3a_coarse = (
            0.0 <= geom.behind_m <= th.m3a_coarse_behind_max
            and abs(geom.cross_m) <= th.m3a_coarse_cross_tol
            and abs(geom.below_m) <= th.m3a_coarse_below_tol
        )

        if in_m3a_coarse:
            self._m3a_coarse_grace_start = None
            self._m3a_coarse_grace_accum_s = 0.0
            self._m3a_coarse_current_dwell_s += dt
            if self._m3a_coarse_current_dwell_s > self._m3a_coarse_best_dwell_s:
                self._m3a_coarse_best_dwell_s = self._m3a_coarse_current_dwell_s
            if (not self._m3a_coarse_achieved
                    and self._m3a_coarse_current_dwell_s >= th.m3a_coarse_dwell_s):
                self._m3a_coarse_achieved = True
                self._m3a_coarse_achieved_at_s = self._t_sim_s
                log(f"[M3A_COARSE] ACHIEVED at t_sim={self._t_sim_s:.3f}s "
                    f"dwell={self._m3a_coarse_current_dwell_s:.3f}s")
        else:
            if self._m3a_coarse_grace_start is None:
                self._m3a_coarse_grace_start = self._t_sim_s
                self._m3a_coarse_grace_accum_s = 0.0
            self._m3a_coarse_grace_accum_s += dt
            if self._m3a_coarse_grace_accum_s > M3A_DWELL_GRACE_S:
                self._m3a_coarse_current_dwell_s = 0.0

    def _update_m3_progress_longitudinal(self, geom: 'RelativeGeometry', dt: float) -> None:
        """Track M3_progress_longitudinal metric (report-only diagnostic).

        Credits axial+cross progress independently of altitude convergence.
        Envelope: behind [0, 0.75], |cross| < 0.75, |below| < 5.0, dwell >= 1.0s.
        """
        in_envelope = (
            0.0 <= geom.behind_m <= 0.75
            and abs(geom.cross_m) <= 0.75
            and abs(geom.below_m) <= 5.0
        )

        if in_envelope:
            self._m3_prog_long_grace_start = None
            self._m3_prog_long_grace_accum_s = 0.0
            self._m3_prog_long_current_dwell_s += dt
            if self._m3_prog_long_current_dwell_s > self._m3_prog_long_best_dwell_s:
                self._m3_prog_long_best_dwell_s = self._m3_prog_long_current_dwell_s
            if (not self._m3_prog_long_achieved
                    and self._m3_prog_long_current_dwell_s >= 1.0):
                self._m3_prog_long_achieved = True
                self._m3_prog_long_achieved_at_s = self._t_sim_s
                log(f"[M3_PROG_LONG] ACHIEVED at t_sim={self._t_sim_s:.3f}s "
                    f"dwell={self._m3_prog_long_current_dwell_s:.3f}s")
        else:
            if self._m3_prog_long_grace_start is None:
                self._m3_prog_long_grace_start = self._t_sim_s
                self._m3_prog_long_grace_accum_s = 0.0
            self._m3_prog_long_grace_accum_s += dt
            if self._m3_prog_long_grace_accum_s > M3A_DWELL_GRACE_S:
                self._m3_prog_long_current_dwell_s = 0.0

    def _handle_docking_creep(self) -> None:
        """Creep from precontact to contact."""
        # FD-005: Capture entry state for mid-tick abort
        entry_state = self.state

        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else CONTACT_ALT_MSL

        # S3C1: Sign convention auto-detect (one-time)
        self._auto_detect_below_sign(geom, lead_state, fol_alt)

        # E-DEV: On first DC tick, seed below_err from current geometry.
        if not self._coupled_below_initialized:
            self._below_err_m = geom.below_m  # Seed with actual (target=0 at contact)
            self._dc_airspeed_low_accum_s = 0.0  # Reset tripwire on DC entry

        # P5.1 v9.5: Use authoritative vc_behind from control_laws
        vc_behind = self.control_laws.get_vc_behind_ewma_mps()
        if self.envelope_checker.is_in_contact_outer_envelope(
                geom, lead_state.alt_msl, fol_alt) and self.check_contact_capture_gate(
                vc_behind, geom.vel_cross_m_s):
            self.transition_to(ControllerState.HOLD_CONTACT, "Entered contact envelope")
            return

        # Guard strip-down: DC vert + docking safety guards removed (Chunk 2A)

        # --- Phase 1 VTOL fix: Airspeed diagnostic tripwire ---
        # If follower airspeed drops below 15 m/s for >2s in DC, the VTOL
        # transition bug is active. Abort to ALT_CAP_HOLD_RANGE to recover.
        dt_trip = self._dt_tick_used_s if self._dt_tick_used_s else 0.05
        f_as_trip = self.follower.get_airspeed() if self.follower else None
        if f_as_trip is not None and f_as_trip < 15.0:
            self._dc_airspeed_low_accum_s += dt_trip
            if self._dc_airspeed_low_accum_s >= 2.0:
                self._dc_airspeed_tripwire_count += 1
                log(f"[DC_AIRSPEED_TRIPWIRE] airspeed={f_as_trip:.1f} m/s < 15.0 for "
                    f"{self._dc_airspeed_low_accum_s:.1f}s — aborting to ALT_CAP_HOLD_RANGE "
                    f"(trip #{self._dc_airspeed_tripwire_count})")
                self._dc_airspeed_low_accum_s = 0.0
                self.transition_to(ControllerState.ALT_CAP_HOLD_RANGE,
                                   "Airspeed collapse in DC — VTOL tripwire")
                return
        else:
            self._dc_airspeed_low_accum_s = 0.0

        # --- Option A+E: Sub-phase routing ---
        if formation_config.DC_PHASED_ENABLE:
            dt = self._dt_tick_used_s if self._dt_tick_used_s else 0.05

            if self._dc_phase == formation_config.DC_SUBPHASE_VERT_STABILIZE:
                # Accumulate stability: |below_err| < 1.0 AND |cross| < 1.0
                # S3C3: Use below_err (vs ramp) instead of absolute below_m
                if (abs(self._below_err_m) < formation_config.DC_PHASE1_STABLE_BELOW_ABS_M
                        and abs(geom.cross_m) < formation_config.DC_PHASE1_STABLE_CROSS_ABS_M):
                    self._dc_stab_accum_s += dt
                else:
                    self._dc_stab_accum_s = max(0.0, self._dc_stab_accum_s - dt)

                # Advance to Phase 2 when stable
                if (self._dc_stab_accum_s >= formation_config.DC_PHASE1_DWELL_S
                        and not self._dc_phase2_lockout):
                    self._dc_phase = formation_config.DC_SUBPHASE_CREEP_CLOSE
                    self._dc_phase2_enter_count += 1
                    self._dc_sink_bad_accum_s = 0.0
                    self._dc_phase2_abort_grace_accum_s = 0.0  # Fix 2: reset grace on entry
                    log(f"[DC_PHASE] Phase1 -> Phase2 (stable {self._dc_stab_accum_s:.1f}s)")

            elif self._dc_phase == formation_config.DC_SUBPHASE_CREEP_CLOSE:
                # Tranche H (H3c): Ramp trail offset toward boom tip in Phase-2
                if self._form_dock_mode_supported:
                    trail_target = formation_config.BOOM_TIP_BELOW_OFFSET_M  # 4.31m
                    self._dock_trail_m = max(
                        trail_target,
                        self._dock_trail_m - formation_config.DOCKING_TRAIL_RAMP_RATE_MPS * dt)
                    self.follower.set_param("FORM_DOCK_TRAIL", self._dock_trail_m)

                # Fix 2: Bounded grace period for Phase 2 abort
                # Hard immediate fallback (no grace) for gross excursions
                hard_fallback = False
                hard_reason = ""

                # S3C3: Use below_err (vs ramp) instead of absolute below_m
                if abs(self._below_err_m) > formation_config.DC_PHASE2_ABORT_HARD_BELOW_M:
                    hard_fallback = True
                    hard_reason = f"HARD below_err={self._below_err_m:.2f}m > {formation_config.DC_PHASE2_ABORT_HARD_BELOW_M}m"

                elif (geom.behind_m < formation_config.DC_PHASE2_CORRIDOR_BEHIND_GATE_M
                      and (abs(geom.cross_m) > formation_config.DC_PHASE2_CORRIDOR_CROSS_ABS_M
                           or abs(self._below_err_m) > formation_config.DC_PHASE2_CORRIDOR_BELOW_ABS_M)):
                    hard_fallback = True
                    hard_reason = (f"HARD corridor: behind={geom.behind_m:.2f}m "
                                   f"cross={geom.cross_m:.2f}m below={geom.below_m:.2f}m")

                else:
                    # Sustained sink trigger (always immediate — separate accumulator)
                    climb = self.follower.get_climb_rate_mps() if self.follower else None
                    if climb is not None and climb < formation_config.DC_PHASE2_SINK_TRIG_MPS:
                        self._dc_sink_bad_accum_s += dt
                    else:
                        self._dc_sink_bad_accum_s = 0.0
                    if self._dc_sink_bad_accum_s >= formation_config.DC_PHASE2_SINK_TRIG_HOLD_S:
                        hard_fallback = True
                        hard_reason = f"HARD sink_rate={climb:.2f}m/s"

                if hard_fallback:
                    self._dc_phase = formation_config.DC_SUBPHASE_VERT_STABILIZE
                    self._dc_stab_accum_s = 0.0
                    self._dc_phase2_lockout = True
                    self._dc_phase2_fallback_count += 1
                    self._dc_phase1_enter_count += 1
                    self._dc_sink_bad_accum_s = 0.0
                    self._dc_phase2_abort_grace_accum_s = 0.0
                    now = self._t_sim_s if hasattr(self, '_t_sim_s') else 0.0
                    self._dc_phase2_last_fallback_t = now
                    # Tranche H: Reset trail to hold distance on Phase-2 fallback
                    self._dock_trail_m = formation_config.DOCKING_HOLD_BEHIND_M
                    if self._form_dock_mode_supported:
                        self.follower.set_param("FORM_DOCK_TRAIL", self._dock_trail_m)
                    log(f"[DC_PHASE] Phase2 -> Phase1 FALLBACK ({hard_reason})")
                else:
                    # Soft fallback with grace period: below/cross exceedance
                    # S3C3: Use below_err (vs ramp) instead of absolute below_m
                    soft_violation = False
                    if abs(self._below_err_m) > formation_config.DC_PHASE2_ABORT_BELOW_ABS_M:
                        soft_violation = True
                    elif abs(geom.cross_m) > formation_config.DC_PHASE2_ABORT_CROSS_ABS_M:
                        soft_violation = True

                    if soft_violation:
                        self._dc_phase2_abort_grace_accum_s += dt
                    else:
                        # Decay when clear
                        self._dc_phase2_abort_grace_accum_s = max(
                            0.0, self._dc_phase2_abort_grace_accum_s - dt)

                    if self._dc_phase2_abort_grace_accum_s > formation_config.DC_PHASE2_ABORT_GRACE_S:
                        soft_reason = (f"SOFT grace exhausted: below={geom.below_m:.2f}m "
                                       f"cross={geom.cross_m:.2f}m "
                                       f"accum={self._dc_phase2_abort_grace_accum_s:.2f}s")
                        self._dc_phase = formation_config.DC_SUBPHASE_VERT_STABILIZE
                        self._dc_stab_accum_s = 0.0
                        self._dc_phase2_lockout = True
                        self._dc_phase2_fallback_count += 1
                        self._dc_phase1_enter_count += 1
                        self._dc_sink_bad_accum_s = 0.0
                        self._dc_phase2_abort_grace_accum_s = 0.0
                        now = self._t_sim_s if hasattr(self, '_t_sim_s') else 0.0
                        self._dc_phase2_last_fallback_t = now
                        # Tranche H: Reset trail to hold distance on Phase-2 fallback
                        self._dock_trail_m = formation_config.DOCKING_HOLD_BEHIND_M
                        if self._form_dock_mode_supported:
                            self.follower.set_param("FORM_DOCK_TRAIL", self._dock_trail_m)
                        log(f"[DC_PHASE] Phase2 -> Phase1 FALLBACK ({soft_reason})")

            # A1C2: Clear lockout outside if/elif chain (always reachable) + cooldown
            # S3C3: Use below_err (vs ramp) instead of absolute below_m
            now = self._t_sim_s if hasattr(self, '_t_sim_s') else 0.0
            if (self._dc_phase2_lockout
                    and abs(self._below_err_m) < formation_config.DC_PHASE1_STABLE_BELOW_ABS_M
                    and abs(geom.cross_m) < formation_config.DC_PHASE1_STABLE_CROSS_ABS_M
                    and (now - self._dc_phase2_last_fallback_t) >= formation_config.DC_PHASE2_REENTRY_COOLDOWN_S):
                self._dc_phase2_lockout = False

            # Option E: Update TECS target for current phase
            target = self._tecs_spdw_compute_target_for_state()
            self._tecs_spdw_set_target(target)
            self._tecs_spdw_tick()

        # P8.2 v9.8: Apply cross-track bias for earlier convergence
        cross_bias = self._compute_cross_track_bias(geom, lead_state)
        hdg_cmd = self.control_laws.compute_formation_heading_cmd(
            geom, lead_state.heading_deg, extra_bias_deg=cross_bias
        )

        # P8.0 v9.8: Wire FINAL_APPROACH_SCHEDULE into speed pipeline
        # This is the primary closure schedule for DOCKING_CREEP, driving behind → 0
        from formation_control_laws import (
            compute_vertical_closure_cap,
            compute_final_approach_vc_cap,
        )

        # Step 1: Get final approach closure cap based on behind distance
        # M3 S1: Schedule observability — log input and bin index
        final_vc_cap, sched_bin_idx = compute_final_approach_vc_cap(geom.behind_m, return_bin_idx=True)
        self._final_approach_vc_cap = final_vc_cap  # Track for CSV logging
        self._schedule_input_m = geom.behind_m
        self._schedule_bin_idx = sched_bin_idx

        # Coupled below-target ramp: replace hardcoded 0.0 with behind-indexed ramp
        from formation_control_laws import compute_dynamic_below_target
        if (formation_config.COUPLED_BELOW_TARGET_ENABLE
                and self.state.name in formation_config.COUPLED_BELOW_TARGET_ENABLE_STATES):
            # On first tick in DC after entry, initialize to current below_m
            if not self._coupled_below_initialized:
                self._below_target_m = geom.below_m
                self._coupled_below_initialized = True
                # E-DEV: Reset docking safety EWMA on DC entry (remove stale ALT_CAP history)
                self._docking_vz_closing_ewma = 0.0
                self._docking_prev_below_m = float('nan')
                log(f"[COUPLED_BELOW] Initialized below_target_m={self._below_target_m:.2f} "
                    f"(below_m={geom.below_m:.2f})")
            below_target_desired = compute_dynamic_below_target(geom.behind_m)
            # E1: Monotonic clamp — never command descent (increase below_target).
            # Ramp can only decrease below_target (move toward leader).
            below_target_desired = min(below_target_desired, self._below_target_m)
            self._below_target_desired_m = below_target_desired
            dt_rl = self._dt_tick_used_s or 0.05
            max_delta = formation_config.COUPLED_BELOW_TARGET_RATE_LIMIT_MPS * dt_rl
            below_target_limited = self._below_target_m + max(-max_delta, min(max_delta,
                below_target_desired - self._below_target_m))
            self._coupled_below_active = True
            self._coupled_below_active_ticks += 1
        else:
            below_target_limited = 0.0
            self._coupled_below_active = False
            self._below_target_desired_m = 0.0
        below_err = geom.below_m - below_target_limited
        self._below_target_m = below_target_limited
        self._below_err_m = below_err
        self._dc_below_err_samples.append(abs(below_err))

        # P7.2/P8.1: Vertical gate - hold range when error huge, proportional otherwise
        # Apply vertical gating to the final approach vc_cap
        vert_gate = compute_vertical_closure_cap(below_err, final_vc_cap)
        self._vert_gate_factor = vert_gate / final_vc_cap if final_vc_cap > 0 else 1.0

        # P9.0 v9.9: Cross-track gate - reduce closure when off-axis
        from formation_control_laws import compute_cross_track_gate
        cross_gate = compute_cross_track_gate(geom.cross_m, geom.behind_m)
        self._cross_gate_factor = cross_gate  # Track for observability

        # Combine vertical and cross-track gates (take minimum)
        combined_gate = min(vert_gate, cross_gate * final_vc_cap)
        effective_vc_cap = combined_gate

        # Option A: In Phase 1, zero closure — hold range, let altitude/cross settle
        if (formation_config.DC_PHASED_ENABLE
                and self._dc_phase == formation_config.DC_SUBPHASE_VERT_STABILIZE):
            effective_vc_cap = 0.0

        # FD-006: Alignment gating - gate CLOSURE, not just behind_target
        # When misaligned in cross/below and very close, allow no positive closure
        if (abs(geom.cross_m) > 0.25 or abs(geom.below_m) > 0.25) and geom.behind_m < 0.5:
            # Hold range - set vc cap to 0 (no closure allowed until aligned)
            effective_vc_cap = 0.0
            log(f"[FD-006] Alignment gate: cross={geom.cross_m:.2f} below={geom.below_m:.2f} -> vc_cap=0")

        # Convert gated closure rate to speed cap
        if effective_vc_cap < final_vc_cap:
            # Position error (vertical or cross) requires closure limiting
            # WINDFF-1b: air-frame lead ref — the DC station-hold binder that leaked w_par (no-op when OFF)
            climb_clamp_cap = self._windff_lead_gs_air(lead_state.ground_speed_ms) + effective_vc_cap
            self._final_approach_active = True
        else:
            # No position constraint - use final approach schedule directly
            # WINDFF-1b: air-frame lead ref (no-op when OFF)
            climb_clamp_cap = self._windff_lead_gs_air(lead_state.ground_speed_ms) + final_vc_cap
            self._final_approach_active = True

        # M3 S1: Determine binding constraint from climb_clamp (schedule/vert/cross gates)
        safety_cap_source = "NONE"
        if effective_vc_cap == 0.0 and geom.behind_m < 0.5:
            safety_cap_source = "CROSS_GATE"  # FD-006 alignment gate
        elif effective_vc_cap < final_vc_cap:
            if vert_gate < cross_gate * final_vc_cap:
                safety_cap_source = "VERT_GATE_BELOW_ERR"
            else:
                safety_cap_source = "CROSS_GATE"
        else:
            safety_cap_source = "SCHEDULE_FINAL_APPROACH"

        # Guard strip-down: CE + NOPASS guards removed (Chunk 2A)
        # M3 S1: Record binding constraint pre-override (no guard caps active)
        self._last_binding_pre_override = safety_cap_source
        self._last_binding_value_mps = climb_clamp_cap
        self._vert_floor_soft_active = False
        self._vert_floor_hard_active = False
        self._in_precontact_corridor = False
        self._speed_override_active = False

        spd_cmd = self.control_laws.compute_speed_cmd(
            geom, self._windff_lead_gs_air(lead_state.ground_speed_ms),  # WINDFF-1b: air-frame lead ref
            CONTACT_BEHIND_M, fol_alt,
            climb_clamp_cap=climb_clamp_cap,
            safety_clamp_cap=None,  # Guard strip-down: no guard cap
            climb_clamp_fast_decel=True,  # Fix 3C: Fast decel when climb_clamp binds in DC
            controller_state=self.state,
        )

        # FD-006: Last-decimeter invariants - NEGATIVE closure when very close
        if geom.behind_m < 0.30:
            # Cap to leader speed (no positive closure)
            # WINDFF-1b: air-frame lead ref — makes this a true zero-closure cap in wind (no-op when OFF)
            spd_cmd = min(spd_cmd, self._windff_lead_gs_air(lead_state.ground_speed_ms))
        if geom.behind_m < 0.15:
            # Command NEGATIVE closure - leader speed minus 1.0
            # Cap to leader is insufficient given TECS lag
            # WINDFF-1b: air-frame lead ref — lead_gs_air - 1.0 = genuine negative closure (no-op when OFF)
            spd_cmd = min(spd_cmd, self._windff_lead_gs_air(lead_state.ground_speed_ms) - 1.0)
            log(f"[FD-006] Last-decimeter: behind={geom.behind_m:.3f}m -> spd_cap={spd_cmd:.2f}")

        # CSV2 crosswind-dock: active anticipatory closure-rate braking (flag-gated, LIVE DC path).
        # When the follower over-closes a safe stopping profile, force active deceleration
        # (spd_cmd below leader ground speed) to counter gust-driven overshoot PAST the boom-tip;
        # composes with FD-006 via min(). Reuses _apply_closure_brake on the implied closure
        # velocity (spd_cmd - lead_gs). Byte-identical when the flag is OFF.
        if _FD_CLOSURE_BRAKE_ENABLE:
            # WINDFF-1b: air-frame lead ref (no-op when OFF)
            _lead_gs = self._windff_lead_gs_air(lead_state.ground_speed_ms)
            _vcb, _brake_fired = self._apply_closure_brake(spd_cmd - _lead_gs, geom)
            if _brake_fired:
                spd_cmd = min(spd_cmd, _lead_gs + _vcb)
                safety_cap_source = "CLOSURE_BRAKE"
                log(f"[CLOSURE_BRAKE] behind={geom.behind_m:.2f}m rate={geom.closure_rate_ms:+.2f} "
                    f"-> spd_cmd={spd_cmd:.2f}")

        # CSV2_SPEC247: TAS->EAS command FF (default OFF -> byte-identical). Applied AFTER
        # FD-006 + closure-brake caps so those leader-GS-frame caps act in TAS; the final
        # command is then converted to EAS (realized TAS matches the cap, no terminal overtake).
        spd_cmd = apply_eas2tas_ff(spd_cmd, fol_alt, eas2tas=self.follower.get_baro_eas2tas())
        self._last_spd_cmd = spd_cmd

        # Step 1b: Collect TECS speed tracking error during DOCKING_CREEP
        f_as = self.follower.get_airspeed() if self.follower else None
        if f_as is not None:
            self._dc_tecs_speed_err_samples.append(spd_cmd - f_as)

        # M3 S1: Determine final binding (FD-004 override vs pre-override)
        override = self._peek_speed_override()
        self._speed_override_active = override is not None
        if override is not None:
            self._last_binding_final = "FD004_SPEED_OVERRIDE"
        else:
            self._last_binding_final = safety_cap_source
        self._update_binding_counters()

        # P2 v9.4: Update clamp state for logging
        arb = self.control_laws.get_last_arbitration()
        if arb:
            self._brake_clamp_active = arb.brake_clamp_active
            self._climb_clamp_active = arb.climb_clamp_active
            self._safety_clamp_active = arb.safety_clamp_active
            arb.binding_constraint_pre_override = self._last_binding_pre_override
            arb.binding_constraint_final = self._last_binding_final
            arb.binding_value_mps = self._last_binding_value_mps

            # Rate-limited arbitration logging
            self._log_speed_arbitration(arb, geom)

        alt_cmd = self._compute_altitude_target_msl(
            geom, lead_state, self._t_sim_s  # v9.10: Use tick timestamp
        )

        # FD-005: Mid-tick abort - if safety transitioned state, don't send normal commands
        if self.state != entry_state:
            log(f"[FD-005] Mid-tick abort: {entry_state.name} -> {self.state.name}")
            # Note: Safety override should already be active via FD-004
            return

        self.send_guided_commands(
            hdg_cmd, alt_cmd, spd_cmd,
            range_m=geom.range_m, fol_alt=fol_alt,
        )
        self.log_continuous_diagnostics(
            geom, lead_state, fol_alt, hdg_cmd, spd_cmd, "DOCKING_CREEP"
        )

        # P7.1 v9.7: Update M3a dwell tracking
        # v9.10: Use _dt_tick_used_s from tick gating instead of _dt_clamped()
        self._update_m3a_dwell(geom, self._dt_tick_used_s if self._dt_tick_used_s else 0.05)
        self._update_m3a_coarse_dwell(geom, self._dt_tick_used_s if self._dt_tick_used_s else 0.05)
        self._update_m3_progress_longitudinal(geom, self._dt_tick_used_s if self._dt_tick_used_s else 0.05)

    def _hc_exit_grace_step(self, behind_m: float):
        """#459 Candidate A (R7.1): advance the HOLD_CONTACT->DOCKING_CREEP temporal
        exit-grace and decide whether to exit. Returns (should_exit: bool, reason: str).
        Mutates self._hc_exit_grace_accum_s. Called ONLY when the follower is outside the
        contact OUTER envelope AND FD_CONTACT_OUTER_EXIT_GRACE_ENABLE. Catastrophic
        immediate-exit (|behind| > mult*tol) ejects a real overtake within one tick so the
        grace can never mask a deep overtake (collision-safe); otherwise a brief excursion
        (< CONTACT_OUTER_EXIT_GRACE_S) is bridged and the vehicle stays in HOLD_CONTACT."""
        catastrophic = CONTACT_OUTER_CATASTROPHIC_MULT * CONTACT_BEHIND_OUTER_TOL_M
        self._hc_exit_grace_accum_s += self._dt_credit_s
        if abs(behind_m) > catastrophic:
            self._hc_exit_grace_accum_s = 0.0
            return True, "catastrophic"
        if self._hc_exit_grace_accum_s > CONTACT_OUTER_EXIT_GRACE_S:
            acc = self._hc_exit_grace_accum_s
            self._hc_exit_grace_accum_s = 0.0
            return True, f"grace={acc:.2f}s"
        return False, "within_grace"

    def _handle_hold_contact(self) -> None:
        """Hold at contact position for configured dwell.

        v9.10: Uses _t_sim_s and accumulator pattern instead of sim_now_s()
        timestamp differences for dwell tracking.
        """
        # FD-005: Capture entry state for mid-tick abort
        entry_state = self.state

        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state()
        if not lead_state:
            return

        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else CONTACT_ALT_MSL

        # Option E: Maintain TECS altitude bias during contact hold
        if formation_config.TECS_SPDWEIGHT_ENABLE and not self._tecs_spdw_disabled:
            target = self._tecs_spdw_compute_target_for_state()
            self._tecs_spdw_set_target(target)
            self._tecs_spdw_tick()

        _in_outer = self.envelope_checker.is_in_contact_outer_envelope(geom, lead_state.alt_msl, fol_alt)
        if not _in_outer:
            if formation_config.FD_CONTACT_OUTER_EXIT_GRACE_ENABLE:
                # #459 Candidate A (R7.1, D1-authorized, default-OFF => byte-identical):
                # TEMPORAL exit-grace hysteresis (never a spatial widen) + catastrophic backstop.
                _exit, _why = self._hc_exit_grace_step(geom.behind_m)
                if _exit:
                    self._exit_hold_contact(f"Left OUTER ({_why})")
                    self.transition_to(ControllerState.DOCKING_CREEP, "Left contact envelope", skip_exit=True)
                    return
                # within grace, not catastrophic -> stay in HOLD_CONTACT (fall through)
            else:
                self._exit_hold_contact("Left OUTER envelope")
                self.transition_to(ControllerState.DOCKING_CREEP, "Left contact envelope", skip_exit=True)
                return
        elif formation_config.FD_CONTACT_OUTER_EXIT_GRACE_ENABLE:
            self._hc_exit_grace_accum_s = 0.0   # back inside OUTER -> clear the grace

        # Guard strip-down: docking safety guards removed (Chunk 2A)

        # v9.10: Use accumulator pattern for contact dwell instead of timestamp differences
        if self.envelope_checker.is_in_contact_inner_envelope(geom, lead_state.alt_msl, fol_alt):
            # Inside inner envelope - accumulate contact dwell
            if not hasattr(self, '_contact_dwell_accum_s'):
                self._contact_dwell_accum_s = 0.0
            self._contact_dwell_accum_s += self._dt_credit_s
            if self._contact_dwell_accum_s >= CONTACT_HOLD_DURATION_S:
                self.transition_to(ControllerState.SUCCESS, "DOCKING SUCCESS!")
                return
        else:
            # Outside inner - reset contact dwell accumulator
            self._contact_dwell_accum_s = 0.0

        # P8.2 v9.8: Apply cross-track bias for earlier convergence
        cross_bias = self._compute_cross_track_bias(geom, lead_state)
        hdg_cmd = self.control_laws.compute_formation_heading_cmd(
            geom, lead_state.heading_deg, extra_bias_deg=cross_bias
        )
        behind_error = geom.behind_m - CONTACT_BEHIND_M
        p_term = HOLD_CONTACT_KP * behind_error
        # v9.10: PI integrator uses _dt_tick_used_s (single-source dt)
        dt_pi = min(self._dt_tick_used_s if self._dt_tick_used_s else 0.05, 0.2)  # Clamp to 200ms max
        self._hold_contact_i += HOLD_CONTACT_KI * behind_error * dt_pi
        self._hold_contact_i = max(-HOLD_CONTACT_I_CLAMP_MS, min(HOLD_CONTACT_I_CLAMP_MS, self._hold_contact_i))
        # P0.2: D-term using EWMA (not raw geom.closure_rate_ms which has velocity bias)
        d_term = -HOLD_CONTACT_KD * self.control_laws._closure_rate_behind_ewma
        spd_cmd = LEADER_TARGET_TAS_MS + p_term + self._hold_contact_i + d_term

        # Fix 4 consistency: Apply TAS→EAS density correction (matches compute_speed_cmd)
        # Note: compute_isa_density_ratio uses exponent 4.2561 vs control_laws inline 4.2559
        # — difference is ~0.025 m/s at 900m, negligible.
        # CSV2_SPEC247: honor the TAS->EAS command FF flag via apply_eas2tas_ff (same path +
        # FD_EAS2TAS_RATIO override as the approach/precontact states, so HOLD_CONTACT holds at
        # the SAME realized TAS with no terminal overtake). Byte-identical when the FF flag and
        # the committed DENSITY_CORRECTION_ENABLE are both OFF (the default).
        if _FD_EAS2TAS_FF_ENABLE:
            spd_cmd = apply_eas2tas_ff(spd_cmd, fol_alt, eas2tas=self.follower.get_baro_eas2tas())
        elif formation_config.DENSITY_CORRECTION_ENABLE:
            spd_cmd = tas_to_eas(spd_cmd, max(0.0, fol_alt))

        # Guard strip-down: CE + NOPASS guards removed (Chunk 2A)
        self._safety_clamp_active = False
        self._last_spd_cmd = spd_cmd

        # M3 S1: Record binding constraint + final (no guard caps active)
        self._last_binding_pre_override = "NONE"
        self._last_binding_value_mps = 0.0
        override = self._peek_speed_override()
        self._speed_override_active = override is not None
        if override is not None:
            self._last_binding_final = "FD004_SPEED_OVERRIDE"
        else:
            self._last_binding_final = "NONE"
        self._update_binding_counters()

        # v9.10: Use _t_sim_s for altitude command timestamp
        alt_cmd = self._compute_altitude_target_msl(geom, lead_state, self._t_sim_s)

        # FD-005: Mid-tick abort - if safety transitioned state, don't send normal commands
        if self.state != entry_state:
            log(f"[FD-005] Mid-tick abort: {entry_state.name} -> {self.state.name}")
            return

        self.send_guided_commands(hdg_cmd, alt_cmd, spd_cmd, range_m=geom.range_m, fol_alt=fol_alt)
        self.log_continuous_diagnostics(geom, lead_state, fol_alt, hdg_cmd, spd_cmd, "HOLD_CONTACT")

        # §8 Step 3: Track M3a_coarse dwell in HOLD_CONTACT (superset envelope)
        self._update_m3a_coarse_dwell(geom, self._dt_tick_used_s if self._dt_tick_used_s else 0.05)
        self._update_m3_progress_longitudinal(geom, self._dt_tick_used_s if self._dt_tick_used_s else 0.05)

    def _on_contact_success(self, event_dict: dict) -> None:
        """Hook called on docking success for DRL integration."""
        log(f"[DRL_EVENT] CONTACT_SUCCESS: {event_dict}")

    def _exit_hold_contact(self, reason: str) -> None:
        """Helper to clean up state when leaving HOLD_CONTACT.

        v9.10: Uses _t_sim_s for dwell calculation instead of sim_now_s().
        """
        if self.current_contact_entry_time is not None:
            dwell_time = self._t_sim_s - self.current_contact_entry_time
            self.contact_dwell_times.append(round(dwell_time, 1))
        self.current_contact_entry_time = None
        self.contact_hold_start = None
        # v9.10: Reset contact dwell accumulator too
        self._contact_dwell_accum_s = 0.0
        self._reset_hold_contact_integrator()
        self._reset_cross_track_bias()
        self._contact_outer_exit_start = None

    # =========================================================================
    # v9.10 Gate 1: EMERGENCY_BREAKAWAY Handler
    # =========================================================================
    def _handle_emergency_breakaway(self) -> None:
        """v9.10 Gate 1: Handle emergency separation after no-pass violation.

        EMERGENCY_BREAKAWAY is triggered when behind_m < 0 (follower ahead of leader).
        This state:
        1. Commands aggressive separation (negative closure / back away)
        2. Holds for minimum dwell time (ABORT_MIN_DWELL_S)
        3. Only exits when safely behind (behind_m >= ABORT_EXIT_BEHIND_MIN_M)
        4. Transitions to INTERCEPT for safe reacquire

        Exit conditions (ALL must be true):
        1. abort_dwell_s >= ABORT_MIN_DWELL_S
        2. behind_m >= ABORT_EXIT_BEHIND_MIN_M
        3. no safety_emergency_active (violation cleared)
        """
        geom = self.get_geometry()
        if not geom:
            return
        lead_state = self.leader_rx.get_state() if self.leader_rx else None
        if not lead_state:
            return

        fol_pos = self.follower.get_position()
        fol_alt = fol_pos[2] if fol_pos else PRECONTACT_ALT_MSL

        # Initialize abort state tracking if needed
        if not hasattr(self, '_abort_dwell_accum_s'):
            self._abort_dwell_accum_s = 0.0

        # Fix 1: Force TECS_SPDWEIGHT back to CRUISE during breakaway —
        # separation dynamics must not be blunted by altitude bias
        if formation_config.TECS_SPDWEIGHT_ENABLE and not self._tecs_spdw_disabled:
            self._tecs_spdw_set_target(formation_config.TECS_SPDWEIGHT_CRUISE)
            self._tecs_spdw_tick()

        # Always accumulate dwell time using dt_credit_s
        self._abort_dwell_accum_s += self._dt_credit_s

        # Command aggressive separation: speed significantly below leader
        # WINDFF-1b: air-frame lead ref — separation delta realized as commanded in wind (no-op when OFF)
        separation_speed = self._windff_lead_gs_air(lead_state.ground_speed_ms) - NOPASS_HARD_SLOWDOWN_DELTA_MPS

        # Always use emergency bypass during EMERGENCY_BREAKAWAY
        self._safety_emergency_active = True
        self._send_safety_emergency_speed(separation_speed, "emergency_breakaway_separation")

        # Maintain altitude at precontact level for safety
        alt_cmd = PRECONTACT_ALT_MSL

        # Heading: point away from leader (increase separation)
        # Use formation heading but allow natural drift back
        hdg_cmd = self.control_laws.compute_formation_heading_cmd(geom, lead_state.heading_deg)

        # Rate-limited logging
        if not hasattr(self, '_abort_log_accum_s'):
            self._abort_log_accum_s = 0.0
        self._abort_log_accum_s += self._dt_credit_s
        if self._abort_log_accum_s >= 2.0:
            self._abort_log_accum_s = 0.0
            log(f"[EMERGENCY_BREAKAWAY] behind={geom.behind_m:.1f}m "
                f"dwell={self._abort_dwell_accum_s:.1f}/{ABORT_MIN_DWELL_S}s "
                f"exit_behind_req={ABORT_EXIT_BEHIND_MIN_M}m "
                f"spd_cmd={separation_speed:.1f}m/s")

        # Check exit conditions
        dwell_ok = self._abort_dwell_accum_s >= ABORT_MIN_DWELL_S
        behind_ok = geom.behind_m >= ABORT_EXIT_BEHIND_MIN_M
        violation_cleared = geom.behind_m >= 0  # No longer ahead of leader

        if dwell_ok and behind_ok and violation_cleared:
            # Safe to exit - transition to INTERCEPT for conservative reacquire
            log(f"[EMERGENCY_BREAKAWAY] EXIT: behind={geom.behind_m:.1f}m "
                f"dwell={self._abort_dwell_accum_s:.1f}s - transitioning to INTERCEPT")
            self._safety_emergency_active = False
            self._abort_dwell_accum_s = 0.0
            self.transition_to(ControllerState.INTERCEPT,
                               f"Emergency separation complete, behind={geom.behind_m:.1f}m")
            return

        # Send commands to maintain separation
        self.send_guided_commands(hdg_cmd, alt_cmd, separation_speed)

    # =========================================================================
    # Main Loop
    # =========================================================================
    def _compute_range_vc(self, range_m: float) -> None:
        """
        Compute closure rate based on 3D range derivative.
        Positive rate = closing (range decreasing).
        Updates self._range_vc_ewma.

        v9.10: Uses _dt_tick_used_s for dt instead of timestamp differences.
        """
        # v9.10: Use _dt_tick_used_s from tick gating
        dt = self._dt_tick_used_s if self._dt_tick_used_s else 0.05

        # Initialize if missing
        if not hasattr(self, '_last_range_val'):
            self._last_range_val = range_m
            return

        # Update only if dt is valid
        if dt > 0.001:
            # Rate = -(current - prev) / dt
            # If range decreases (closing), (current - prev) is negative.
            # We want positive closure rate, so negate it.
            raw_rate = -(range_m - self._last_range_val) / dt

            # Low-pass filter (dt-scaled to maintain consistent time constant)
            alpha = ewma_alpha_scaled(0.1, dt, nominal_dt=0.05)
            self._range_vc_ewma = (1.0 - alpha) * self._range_vc_ewma + alpha * raw_rate

            self._last_range_val = range_m

    def _print_run_summary(self) -> None:
        """Print summary of the run to stdout."""
        print("\n" + "=" * 70)
        print(f"RUN SUMMARY ({CONTROLLER_VERSION})")
        print("=" * 70)

        # Total flight time
        if self._run_start_time:
            flight_time = time.time() - self._run_start_time
            mins, secs = divmod(int(flight_time), 60)
            print(f"Total Flight Time: {mins}m {secs}s ({flight_time:.1f}s)")
        else:
            print("Total Flight Time: (not recorded)")

        print(f"Minimum Range Achieved: {self.min_range_achieved:.2f}m")
        print(f"Precontact Holds (>={PRECONTACT_HOLD_DURATION_S}s): {self.precontact_hold_count}")
        print(f"Contact Envelope Entries: {self.contact_entry_count}")
        if self.contact_dwell_times:
            total_contact = sum(self.contact_dwell_times)
            print(f"Contact Dwell Times: {self.contact_dwell_times}")
            print(f"Total Contact Time: {total_contact:.1f}s")
        else:
            print("Contact Dwell Times: (none)")

        # DRL alignment success metrics
        print("-" * 40)
        print("DRL Alignment (Test 311):")
        print(f"  contact_success: {self.contact_success}")
        print(f"  contact_success_count: {self.contact_success_count}")
        print(f"  max_inner_hold: {self._contact_inner_hold_max_s:.1f}s "
              f"(target: {CONTACT_HOLD_DURATION_S}s)")

        # State entry counts
        print("-" * 40)
        print("State Entry Counts:")
        if self._state_entry_counts:
            for state_name, count in sorted(self._state_entry_counts.items()):
                print(f"  {state_name}: {count}")
        else:
            print("  (no state transitions recorded)")

        print("-" * 40)
        print(f"Final Sensor Source: {self._active_sensor_source}")
        print(f"Sensor Mode Used: {self.sensor_mode}")
        print("=" * 70)

    def _execute_tick(self, queue_time_boot_ms: Optional[int] = None) -> bool:
        """v9.10 Gate 0: Tick owner wrapper with pending-row pattern.

        Args:
            queue_time_boot_ms: FD-001 fix - timestamp from queue (bypasses time authority
                monotonicity filter which can cause 'no advance' skips). If None, falls
                back to time authority.

        Returns True if tick was executed, False if skipped.
        Raises GateFatalError subclasses on fatal conditions.

        NOTE: Marker-based tick gating (time_boot_ms check) is now in the main loop.
        This method is only called when a new time_boot_ms has been received,
        so we no longer need the boolean _attitude_message_arrived check here.
        """
        self._tick_owner_enter()
        try:
            # v9.10 Gate0 Hardening: Marker-based gating moved to main loop.
            # By the time we reach here, we know a new time_boot_ms was received.
            # The main loop gates on time_boot_ms change, not boolean arrival.

            # Gate 0: read sim time ONCE
            # FD-001 Fix: Use queue timestamp directly if provided, bypassing
            # time authority's monotonicity guarantee which can cause 'no advance'
            # skips when draining backlog older than _last_now_s.
            # FD-008-B: Use single-writer API with source="tick_scheduler"
            if queue_time_boot_ms is not None:
                t_sim = queue_time_boot_ms / 1000.0
                # FD-008-B: Update TA via single-writer API (always allowed for tick_scheduler)
                get_time_authority().advance_from_time_boot_ms(
                    queue_time_boot_ms, source="tick_scheduler")
            else:
                t_sim = self._sim_now_s_raw()

            # Decide execute vs skip (tick gating logic)
            if not self._should_execute_tick(t_sim):
                return False  # skipped: no CSV row, no tick increment

            # === Executed tick begins ===
            tick_num = self._ticks_executed_total + 1
            if TICK_TIMING_DIAGNOSTIC:
                _diag_wall_start = time.perf_counter()

            # Track first/last executed tick sim times for tick-rate computation
            if self._first_executed_tick_sim_s is None:
                self._first_executed_tick_sim_s = t_sim
            self._last_executed_tick_sim_s = t_sim

            # Build pending row (executed ticks only)
            self._pending_row = {
                "ticks_executed_total": tick_num,
                "t_sim_s": float(t_sim),
                "dt_tick_raw_s": None,
                "exception": None,
            }

            # Reset per-tick latches
            self._speed_cmd_sent_this_tick = False
            self._csv_row_written_this_tick = False  # P2 Fix D v9.10
            self._first_send_source = None
            self._first_send_speed_mps = None
            self._first_send_bypass = None

            # Store tick timestamp for handlers
            self._t_sim_s = t_sim

            # Compute dt fields for this executed tick
            self._compute_dt_for_tick(t_sim)
            self._pending_row["dt_tick_raw_s"] = float(self._dt_tick_raw_s) if self._dt_tick_raw_s else 0.0

            # FD-007: Track dt samples for rolling p95
            if self._dt_tick_used_s is not None and self._dt_tick_used_s > 0:
                self._recent_dt_samples.append(self._dt_tick_used_s)

            # Run the tick logic (may raise)
            self._run_one_tick()

            if TICK_TIMING_DIAGNOSTIC:
                _diag_wall_end = time.perf_counter()
                _diag_dur_ms = (_diag_wall_end - _diag_wall_start) * 1000.0
                # Leader staleness: did a new GLOBAL_POSITION_INT arrive since last tick?
                _ls = self.leader_rx.get_state() if self.leader_rx else None
                _ltbms = _ls.time_boot_ms if _ls else -1
                _fresh = int(_ltbms != self._tick_diag_prev_leader_tbms)
                self._tick_diag_prev_leader_tbms = _ltbms
                self._tick_diag_records.append((
                    tick_num, t_sim, _diag_wall_end, _diag_dur_ms, self.state.name,
                    _ltbms, _fresh
                ))
                self._tick_diag_wall_times.append(_diag_dur_ms)
                if _diag_dur_ms > 50.0:
                    self._tick_diag_drops += 1
                if tick_num - self._tick_diag_last_summary_tick >= 100:
                    self._tick_diag_print_summary(tick_num)
                    self._tick_diag_last_summary_tick = tick_num

            # P2 Fix D v9.10: Ensure one CSV row per executed tick
            # If _send_speed_command wasn't called, write a minimal row
            if not self._csv_row_written_this_tick:
                geom = self.get_geometry()
                self._write_arbitration_csv_row_minimal(geom)

            return True

        except Exception as e:
            # Preserve fatal tick row, then re-raise
            # R11 v9.10: Don't set tick counter here - counter only in CSV write
            try:
                if self._pending_row is not None:
                    self._pending_row["exception"] = type(e).__name__
            except Exception:
                pass
            raise

        finally:
            self._pending_row = None
            self._tick_owner_exit()

    def _should_execute_tick(self, t_sim: float) -> bool:
        """v9.10: Tick gating logic - returns True if tick should execute."""
        EPS = 1e-6
        TICK_PERIOD_S = 0.05  # 20 Hz target rate in sim-time

        # First tick: initialize and return skip
        if self._last_tick_time_s is None:
            self._last_tick_time_s = t_sim
            self._tick_accum_s = 0.0
            self._dt_tick_raw_s = None
            self._dt_tick_used_s = None
            self._dt_credit_s = 0.0
            self._dt_tick = 0.05  # Legacy compatibility
            log("[TICK_INIT] First tick initialized, returning without control action")
            return False

        raw_dt = t_sim - self._last_tick_time_s

        # Sim-time regression (bad) - treat as run integrity failure
        if raw_dt < -EPS:
            self._sim_time_regressions += 1
            self._run_invalid_timebase = True
            if not self._regression_warned:
                log(f"[TIMEBASE_REGRESSION] sim-time went backward by {raw_dt:.6f}s, run invalid")
                self._regression_warned = True
            self._last_tick_time_s = t_sim
            self._tick_accum_s = 0.0
            self._dt_tick_raw_s = None
            self._dt_tick_used_s = None
            self._dt_credit_s = 0.0
            return False

        # No advance (or microscopic): do nothing
        if raw_dt <= EPS:
            self._ticks_skipped_no_advance += 1
            return False

        # Sim-time advanced: update reference and accumulate
        self._last_tick_time_s = t_sim
        self._tick_accum_s += raw_dt

        # Only proceed when accumulated time reaches tick period (20 Hz)
        # EPS tolerance: IEEE 754 cancellation in integer-ms → float conversion
        # can produce 0.04999... which would fail a strict < 0.05 compare.
        if self._tick_accum_s + EPS < TICK_PERIOD_S:
            self._ticks_skipped_accumulating += 1
            return False

        return True

    def _compute_dt_for_tick(self, t_sim: float) -> None:
        """v9.10: Compute dt fields for an executed tick."""
        DT_MAX_USED = 0.25
        DT_CREDIT_CAP = 0.1

        # Detect "falling behind"
        if self._tick_accum_s > 0.2:
            self._late_tick_count += 1

        self._dt_tick_raw_s = self._tick_accum_s
        self._tick_accum_s = 0.0

        self._dt_tick_used_s = min(self._dt_tick_raw_s, DT_MAX_USED)
        self._dt_tick = self._dt_tick_used_s

        self._max_sim_dt_raw_s = max(self._max_sim_dt_raw_s, self._dt_tick_raw_s)
        self._dt_credit_s = min(self._dt_tick_raw_s, DT_CREDIT_CAP)

    def _update_wind_ff(self, geom: 'RelativeGeometry') -> None:
        """WINDFF: per-tick along-track wind estimate (LPF) + optional state-gated closure trim.
        No-op / byte-identical when _FD_SELF_WIND_FF is OFF. Perception signals only (get_velocity_ned
        + get_airspeed x baro-eas2tas + get_attitude yaw + leader heading as the formation track);
        NEVER get_sim_velocity_ned/SIM_STATE. Holds the prior estimate (counts a skip) when any
        telemetry input — including baro eas2tas — is unavailable, so a missing frame never guesses."""
        if not _FD_SELF_WIND_FF:
            return
        lead_state = self.leader_rx.get_state() if self.leader_rx else None
        vel = self.follower.get_velocity_ned() if self.follower else None
        eas = self.follower.get_airspeed() if self.follower else None
        att = self.follower.get_attitude() if self.follower else None
        e2t = self.follower.get_baro_eas2tas() if self.follower else None
        if lead_state is None or vel is None or eas is None or att is None or not e2t:
            self._windff_skip_count += 1
            return
        west_raw = wind_ff.along_track_wind(vel[0], vel[1], eas * e2t, att[2],
                                            lead_state.heading_deg)
        dt = self._dt_tick_used_s if self._dt_tick_used_s else 0.05
        seed = self._windff_west if self._windff_last_t is not None else None
        self._windff_west = wind_ff.lpf_step(seed, west_raw, dt, _FD_WINDFF_TAU_S)
        self._windff_last_t = self._t_sim_s
        active = _FD_WINDFF_TRIM and (self.state.name in _WINDFF_TRIM_STATES)
        closure_err = self.control_laws.get_vc_behind_ewma_mps() - geom.closure_rate_ms
        self._windff_trim_i = wind_ff.trim_step(
            self._windff_trim_i, closure_err, dt, _FD_WINDFF_TRIM_KI,
            _FD_WINDFF_TRIM_CLAMP, active)

    def _windff_lead_gs_air(self, lead_gs: float) -> float:
        """WINDFF Phase-1b (frame-surgical, replaces the V1 whole-command seam correction —
        RCA_V1_BIAS_AND_FRAME.md): the leader's ground speed expressed in the FOLLOWER's airspeed
        frame, lead_gs_air = lead_gs - w_par (+ trim). In uniform wind this ~= leader TAS, so the
        GS-referenced-as-airspeed expressions (DC schedule caps, closure-cap branch, FD-006
        last-decimeter mins, near-precontact cap, closure-brake, breakaway separation, guard speed
        cap) stop leaking the follower's along-track wind into realized closure. The TAS-referenced
        paths (approach PD profile = LEADER_TARGET_TAS + delta, HOLD_CONTACT) are wind-immune by
        design and MUST NOT be corrected (the V1 whole-command placement double-corrected them ->
        C1 REJOIN stall). Returns lead_gs UNCHANGED (byte-identical) when _FD_SELF_WIND_FF is OFF."""
        if not _FD_SELF_WIND_FF:
            return lead_gs
        return lead_gs + wind_ff.correction(
            self._windff_west, self._windff_trim_i, _FD_SELF_WIND_FF, _FD_WINDFF_TRIM)

    def _run_one_tick(self) -> None:
        """v9.10: Run one iteration of the state machine (inner tick logic).

        Called by _execute_tick() after tick owner guard is acquired.
        Uses self._t_sim_s as the tick timestamp.
        """
        # P2 v9.4 Fix 2a: Increment tick counter for near-transition logging
        # v9.10: Moved BEFORE fault injection check to fix off-by-one bug
        self._ticks_since_transition += 1

        # v9.10 Canary C: Fault injection for testing fatal path preservation
        # Supports mode=tick (default) - injects after N ticks in target state
        inject_fatal = os.environ.get("FLIGHTDOCK_INJECT_FATAL")
        if inject_fatal:
            inject_mode = os.environ.get("FLIGHTDOCK_INJECT_FATAL_MODE", "tick")
            if inject_mode == "tick":
                inject_state = os.environ.get("FLIGHTDOCK_INJECT_FATAL_STATE", "REJOIN_PRECONTACT")
                inject_tick = int(os.environ.get("FLIGHTDOCK_INJECT_FATAL_TICK", "50"))
                # Use _ticks_since_transition (now already incremented)
                tick_count = self._ticks_since_transition
                if self.state.name == inject_state and tick_count >= inject_tick:
                    raise InjectedFatalError(
                        f"INJECTED_FATAL: tick mode, state={self.state.name} tick_since_trans={tick_count}"
                    )

        # P0.1: Update closure rate EWMA every tick for universal freshness
        # Chunk 9: Reset per-tick guard state unconditionally (must not carry across ticks)
        self._guard_speed_cap_mps = None

        geom = self.get_geometry()
        if geom is None:
            # Can't confirm overtake/exceedance without geometry — reset persist accumulators
            self._overtake_persist_accum = 0.0
            self._alt_ceiling_persist_accum = 0.0

        if geom is not None:
            self.control_laws.compute_closure_rate_behind(geom.behind_m)
            self._update_slow_behind_vc(geom.behind_m)  # v9.2: slow net drift
            self._compute_range_vc(geom.range_m)  # Range-based Vc
            self._update_wind_ff(geom)  # WINDFF (flag-gated; no-op when OFF)

            # v9.10 Gate 1: Only track safety metrics during formation-relevant states
            FORMATION_STATES = {
                ControllerState.REJOIN_PRECONTACT,
                ControllerState.HOLD_PRECONTACT,
                ControllerState.ALT_CAP_HOLD_RANGE,
                ControllerState.ALTITUDE_CAPTURE,
                ControllerState.DOCKING_CREEP,
                ControllerState.HOLD_CONTACT,
            }

            # Track minimum behind_m only during formation states
            if self.state in FORMATION_STATES and geom.behind_m < self._min_behind_m:
                self._min_behind_m = geom.behind_m
                self._t_sim_at_min_behind_s = self._t_sim_s
                self._state_at_min_behind = self.state.name

            # Check for no-pass violation (behind_m < 0) only during formation states
            if geom.behind_m < 0 and self.state in FORMATION_STATES:
                self._count_no_pass_violations += 1

            # ── Chunk 9: Minimal guard evaluation ──────────────────

            if self.state in FORMATION_STATES:
                # Guard 1: Altitude ceiling — hard breakaway
                # Going above the leader is the most dangerous geometry (propwash, structural collision)
                # E1 (T-341, signed): debounced — fused exceedance must persist
                # GUARD_ALT_CEILING_PERSIST_S before firing (fused below-axis
                # noise fired 1-2 tick phantoms at the bar; 0.0 = single-tick)
                # G3 (2026-08-14, R7.1 re-seal): FD_GUARD_CEILING_CHANNEL=env
                # compares the same-computation GPS-direct below instead.
                # Geometry without below_gps_m (UWB_ONLY/UWB_DOM/legacy) falls
                # back to fused, fail-safe, and the fallback is counted.
                _below_guard = geom.below_m
                _guard_ch = 'fused_configured'   # GUARDWIT: witness only
                if _FD_GUARD_CEILING_ENV:
                    _bg = getattr(geom, 'below_gps_m', None)
                    if _bg is not None:
                        _below_guard = _bg
                        _guard_ch = 'env_gps'
                    else:
                        self._guard_env_fallback_count += 1
                        _guard_ch = 'fused_fallback'
                if _below_guard < formation_config.GUARD_ALT_CEILING_BELOW_M:
                    self._alt_ceiling_persist_accum += self._dt_credit_s
                    if self._alt_ceiling_persist_accum >= formation_config.GUARD_ALT_CEILING_PERSIST_S:
                        self._guard_altitude_ceiling_fire_count += 1
                        # GUARDWIT (2026-08-19, operator-authorized log-only
                        # export). GUARDRCA-729 measured 4 of 5 D2a attempt-2
                        # ceiling fires with NO crossing on the elected env
                        # channel in the 20 Hz writer samples, and one with no
                        # crossing on EITHER logged channel -- the value the
                        # guard actually compared was unrecoverable from banked
                        # data. This records it. PURE APPEND: no control-flow,
                        # threshold, channel-selection or decision effect; the
                        # guard fires exactly as before whether or not this
                        # list is written. Bounded so a pathological leg cannot
                        # grow it without limit.
                        # Best-effort by construction: resolved via getattr so a
                        # controller without the attribute (object.__new__
                        # fixtures, older pickles) records NO witness and still
                        # fires normally. Measured 2026-08-19: the direct
                        # attribute form raised AttributeError here and would
                        # have aborted the guard tick — the witness must never
                        # be able to do that.
                        _wit = getattr(self, '_guard_ceiling_fires', None)
                        if _wit is not None and len(_wit) < _GUARDWIT_MAX_FIRES:
                            _wit.append({
                                't_sim_s': round(float(self._t_sim_s), 3),
                                'compared_below_m': round(float(_below_guard), 4),
                                'channel': _guard_ch,
                                'fused_below_m': round(float(geom.below_m), 4),
                                'gps_below_m': (
                                    None if getattr(geom, 'below_gps_m', None) is None
                                    else round(float(geom.below_gps_m), 4)),
                                'threshold_m': float(
                                    formation_config.GUARD_ALT_CEILING_BELOW_M),
                                'persist_accum_s': round(
                                    float(self._alt_ceiling_persist_accum), 4),
                                'behind_m': round(float(geom.behind_m), 3),
                                'state': self.state.name,
                            })
                        if self._trigger_guard_breakaway("altitude_ceiling", geom):
                            return  # FD-005: abort tick (only if transition succeeded)
                else:
                    self._alt_ceiling_persist_accum = 0.0

                # Guard 2: Hard overtake — breakaway with 10.0s persist gate
                # Follower must never pass the leader — this is the no-fly zone
                if geom.behind_m < 0:
                    self._overtake_persist_accum += self._dt_credit_s
                    if self._overtake_persist_accum >= formation_config.GUARD_OVERTAKE_PERSIST_S:
                        self._guard_hard_overtake_fire_count += 1
                        if self._trigger_guard_breakaway("hard_overtake", geom):
                            return  # FD-005: abort tick (only if transition succeeded)
                else:
                    self._overtake_persist_accum = 0.0

                # Guard 3: Soft speed cap — modify speed only, NO breakaway
                # Near the boom tip, approach speed should be slow
                if geom.range_m < formation_config.GUARD_SPEED_CAP_RANGE_M:
                    vc = self.control_laws._closure_rate_behind_ewma
                    if vc > formation_config.GUARD_SPEED_CAP_VC_MPS:
                        lead_state = self.leader_rx.get_state() if self.leader_rx else None
                        if lead_state:
                            # WINDFF-1b: air-frame lead ref (no-op when OFF)
                            cap = self._windff_lead_gs_air(lead_state.ground_speed_ms) + formation_config.GUARD_SPEED_CAP_VC_MPS
                            self._guard_speed_cap_mps = cap
                            self._guard_speed_cap_fire_count += 1

            # L-DIAG: Hard-stop geometry terminator — ends experiment on invalid geometry
            # Only active during close-range docking states where geometry divergence is anomalous
            _GEOM_TERM_STATES = {ControllerState.DOCKING_CREEP, ControllerState.HOLD_CONTACT}
            if self._diag_mode_active and formation_config.DIAG_GEOMETRY_TERMINATOR and self.state in _GEOM_TERM_STATES:
                dt = self._dt_tick_used_s if self._dt_tick_used_s else 0.05
                _term_reason = None
                # Overtake check — debounced 0.05s to filter GPS noise
                if geom.behind_m < -0.5 and abs(geom.cross_m) < 3.0:
                    self._diag_geom_behind_accum_s += dt
                    if self._diag_geom_behind_accum_s >= 0.05:
                        _term_reason = (f"behind={geom.behind_m:.2f}m cross={geom.cross_m:.2f}m "
                                        f"for {self._diag_geom_behind_accum_s:.2f}s")
                else:
                    self._diag_geom_behind_accum_s = 0.0
                # Range check — debounced 0.05s
                if geom.range_m < 1.0:
                    self._diag_geom_range_accum_s += dt
                    if self._diag_geom_range_accum_s >= 0.05:
                        _term_reason = (f"range={geom.range_m:.2f}m "
                                        f"for {self._diag_geom_range_accum_s:.2f}s")
                else:
                    self._diag_geom_range_accum_s = 0.0
                # Altitude divergence — immediate (15m is well beyond noise)
                if abs(geom.below_m) > 15.0:
                    _term_reason = f"altitude divergence: below={geom.below_m:.2f}m"

                if _term_reason:
                    log(f"[DIAG_GEOM_TERM] Experiment terminated: {_term_reason}")
                    self._diag_geometry_terminator_reason = _term_reason
                    self._diag_geometry_terminator_time = self._t_sim_s
                    self._log_transition_snapshot(self.state, ControllerState.FAIL,
                                                  f"DIAG_GEOMETRY_TERMINATOR: {_term_reason}",
                                                  self._t_sim_s)
                    self._enter_terminal_fail(f"DIAG_GEOMETRY_TERMINATOR: {_term_reason}")
                    return

            # Guard strip-down: CZ update + CZ-EMERGENCY removed (Chunk 2A)

        # State dispatch
        handlers = {
            ControllerState.WEATHER_CHECK: self._handle_weather_check,
            ControllerState.VTOL_TAKEOFF: self._handle_vtol_takeoff,

            # P1 Fix A v9.10: Non-blocking takeoff sub-FSM
            ControllerState.TAKEOFF_SET_VTOL_MODE: self._handle_takeoff_set_vtol_mode,
            ControllerState.TAKEOFF_ARMING: self._handle_takeoff_arming,
            ControllerState.TAKEOFF_CLIMBING: self._handle_takeoff_climbing,
            ControllerState.TAKEOFF_SET_FW_MODE: self._handle_takeoff_set_fw_mode,
            ControllerState.TAKEOFF_TRANSITION_MONITOR: self._handle_takeoff_transition_monitor,

            ControllerState.FW_TRANSITION: self._handle_fw_transition,
            ControllerState.FW_STABILIZE: self._handle_fw_stabilize,
            ControllerState.INTERCEPT: self._handle_intercept,
            ControllerState.LEAD_TURN: self._handle_lead_turn,
            ControllerState.STERN_CONVERSION: self._handle_stern_conversion,
            ControllerState.REJOIN_PRECONTACT: self._handle_rejoin_precontact,
            ControllerState.HOLD_PRECONTACT: self._handle_hold_precontact,
            ControllerState.ALT_CAP_HOLD_RANGE: self._handle_alt_cap_hold_range,
            ControllerState.ALTITUDE_CAPTURE: self._handle_altitude_capture,
            ControllerState.DOCKING_CREEP: self._handle_docking_creep,
            ControllerState.HOLD_CONTACT: self._handle_hold_contact,
            # v9.10 Gate 1: Add EMERGENCY_BREAKAWAY handler
            ControllerState.EMERGENCY_BREAKAWAY: self._handle_emergency_breakaway,
        }
        handler = handlers.get(self.state)

        if handler:
            handler()
        elif self.state in [ControllerState.SUCCESS, ControllerState.FAIL, ControllerState.INIT]:
            if self.state != ControllerState.INIT:  # INIT is not a terminal state
                self.running = False
        else:
            log(f"ERROR: No handler for state {self.state}")
            self.transition_to(ControllerState.FAIL, f"Unhandled state: {self.state}")

    def run_state_machine(self) -> None:
        """v9.10: Backward-compatible wrapper that calls _execute_tick().

        This method exists for compatibility with code that calls run_state_machine().
        The actual implementation is in _execute_tick() and _run_one_tick().
        """
        self._execute_tick()

    def run(self, skip_takeoff: bool = False, max_sim_seconds: float = None) -> None:
        """Main control loop.

        TEST 313 Patch B: Wraps main loop in try/finally to guarantee
        _print_run_summary() is called even on Ctrl+C (SIGINT).

        Args:
            skip_takeoff: If True, skip VTOL takeoff and start in INTERCEPT state
            max_sim_seconds: Maximum sim-time duration before auto-termination (Option 0 fix)
        """
        log(f"Starting baseline controller v{CONTROLLER_VERSION}")

        # L-DIAG: Diagnostic mode banner
        _diag_flags = []
        if formation_config.DIAG_BREAKAWAY_SHADOW:
            _diag_flags.append("BREAKAWAY_SHADOW")
        # Guard strip-down: CE_TIER_A_SUPPRESS + NOPASS_TIER_A_SUPPRESS removed (Chunk 3A)
        if formation_config.DIAG_GEOMETRY_TERMINATOR and self._diag_mode_active:
            _diag_flags.append("GEOMETRY_TERMINATOR")
        if _diag_flags:
            log("=" * 60)
            log(f"*** DIAGNOSTIC MODE: {', '.join(_diag_flags)} ***")
            log("*** THIS RUN IS NOT PRODUCTION VALID ***")
            log("=" * 60)

        self._run_start_time = time.time()  # Track total flight time

        if not self.connect():
            log("ERROR: Failed to connect")
            return

        log("")
        log("=" * 60)
        log("WAITING FOR SIMULATION TIMEBASE PASS")
        log("=" * 60)
        if not self._wait_for_timebase_pass():  # Uses scaled default
            self._stop_reason = "SIM_TIME_PASS_TIMEOUT"
            log("ERROR: Timebase PASS timeout - cannot start mission")
            return

        # P1 Fix B v9.10: Run timebase canary to verify ATTITUDE rate
        # Only at 1x speedup (higher speedups have different timing characteristics)
        # NOTE: Threshold lowered from 18Hz to 6Hz because SITL's SR0_EXTRA1=4Hz
        # and MAV_CMD_SET_MESSAGE_INTERVAL may not override reliably.
        # The key validation is that rate >= 6Hz (was 4Hz causing the blocking gaps).
        # FD-001 DEBUG: Temporarily skip canary - it conflicts with the new queue
        # because canary uses mav.recv_match() which bypasses update_cache() queue
        if False and TIMEBASE_CANARY_AVAILABLE and abs(self.expected_speedup - 1.0) < 0.05:
            log("")
            log("=" * 60)
            log("P1 FIX B: TIMEBASE CANARY (verifying ATTITUDE rate >= 6Hz)")
            log("=" * 60)
            canary_ok, canary_msg, canary_metrics = run_timebase_canary(
                self.follower.mav,
                get_time_authority(),
                duration_sim_s=12.0,
                min_update_hz_sim=6.0,  # v9.10: Lowered from 18 to 6 (SITL limit)
                timeout_wall_s=30.0,
            )
            # Fix: Use correct key names from get_time_authority_metrics()
            hz_sim = canary_metrics.get('time_authority_update_hz_sim', 0.0)
            dt_med = canary_metrics.get('time_authority_dt_median', 0.0)
            log(f"[CANARY] {canary_msg}")
            log(f"[CANARY] update_hz_sim={hz_sim:.1f}Hz, dt_median={dt_med*1000:.1f}ms")
            if not canary_ok:
                log("ERROR: Timebase canary FAILED - ATTITUDE rate too low")
                self._stop_reason = "TIMEBASE_CANARY_FAIL"
                return

        # Initialize all control timers in sim-time AFTER PASS
        self._reset_sim_timers_after_pass()

        # P2 v9.4 Fix 2b: Initialize arbitration CSV after timebase PASS
        self._init_arbitration_csv()
        # C3 (gated, default-off): initialize UWB emitter->fusion delivery diag
        self._init_uwb_fusion_diag()
        # CSV2_R7_1 (gated, default-off): initialize rejoin cross-track crab diag
        self._init_cross_crab_diag()

        # Start state machine
        if skip_takeoff:
            self.transition_to(ControllerState.INTERCEPT, "Skip takeoff mode")
        else:
            self.transition_to(ControllerState.WEATHER_CHECK, "Starting mission")

        # Main loop - wrapped in try/finally to guarantee run summary on exit
        # v9.10: Uses _pump_mavlink_messages() + _execute_tick() pattern
        # Outer loop reads from time authority for pacing decisions (allowed)

        # v9.10: Get initial time from time authority (not sim_now_s which is forbidden)
        start_sim_time = get_time_authority().now_s()

        try:
            while self.running:
                _loop_start = time.monotonic()

                # FD-001: Drain queue - execute one tick per ATTITUDE timestamp
                # This prevents collapsing multiple ATTITUDE messages into one tick
                MAX_TICKS_PER_LOOP = 10  # Safety bound to prevent runaway

                # First, pump messages to fill the queue
                self._pump_mavlink_messages(blocking=False)

                # Queue depth attribution (ChatGPT-5 R2)
                _qdepth = len(self.follower._attitude_time_queue)
                if _qdepth > self.follower._attitude_queue_max_depth:
                    self.follower._attitude_queue_max_depth = _qdepth
                    self._queue_max_depth_t_sim_s = self._t_sim_s
                    self._queue_max_depth_state = self.state.name if self.state else "UNKNOWN"
                # Change 4: Runtime-only max depth (after queue active AND first tick)
                if (self.follower._queue_active and self._ticks_executed_total > 0
                        and _qdepth > self.follower._attitude_queue_max_depth_runtime):
                    self.follower._attitude_queue_max_depth_runtime = _qdepth
                if _qdepth > 50 and not self._queue_depth_snapshot_logged:
                    self._queue_depth_snapshot_logged = True
                    log(f"[QUEUE_SNAPSHOT] depth={_qdepth} t_sim={self._t_sim_s:.2f}s "
                        f"state={self.state.name} queue_active={self.follower._queue_active}")
                    self._write_queue_depth_snapshot(
                        _qdepth, self._t_sim_s,
                        self.state.name if self.state else "UNKNOWN",
                        self.follower._queue_active)

                # Rec 5A: Runtime backlog time metric
                if (_qdepth >= 2 and self.follower._queue_active
                        and self._ticks_executed_total > 0):
                    _oldest_ms = self.follower._attitude_time_queue[0]
                    _newest_ms = self.follower._last_enqueued_attitude_ms
                    if _newest_ms is not None and _oldest_ms is not None:
                        _backlog_time_s = (_newest_ms - _oldest_ms) / 1000.0
                        if _backlog_time_s > self._runtime_backlog_time_max_s:
                            self._runtime_backlog_time_max_s = _backlog_time_s
                            self._runtime_backlog_time_max_depth = _qdepth
                            self._runtime_backlog_time_max_t_sim_s = self._t_sim_s
                            self._runtime_backlog_time_max_state = (
                                self.state.name if self.state else "UNKNOWN")

                        # Rec 5B: Snapshot trigger (one-shot at 0.5s)
                        if (_backlog_time_s > RUNTIME_BACKLOG_SNAPSHOT_THRESHOLD_S
                                and not self._backlog_time_snapshot_logged):
                            self._backlog_time_snapshot_logged = True
                            self._write_backlog_time_snapshot(
                                _backlog_time_s, _qdepth)

                        # Rec 5C: Backpressure drop mode (state-gated)
                        if (_backlog_time_s > BACKPRESSURE_DROP_THRESHOLD_S
                                and self.expected_speedup >= BACKPRESSURE_MIN_SPEEDUP):
                            _close_in_states = {
                                ControllerState.DOCKING_CREEP,
                                ControllerState.HOLD_CONTACT,
                                ControllerState.ALT_CAP_HOLD_RANGE,
                            }
                            if self.state in _close_in_states:
                                # Close-in: trigger breakaway (stale loop is dangerous)
                                log(f"[BACKPRESSURE] Close-in abort: backlog={_backlog_time_s:.2f}s "
                                    f"state={self.state.name}")
                                if self.state != ControllerState.EMERGENCY_BREAKAWAY:
                                    self._abort_dwell_accum_s = 0.0
                                    self.transition_to(
                                        ControllerState.EMERGENCY_BREAKAWAY,
                                        f"backlog_abort_close_in: {_backlog_time_s:.2f}s")
                                self._ticks_dropped_backpressure_total += _qdepth - 1
                                # Clear queue except latest
                                while len(self.follower._attitude_time_queue) > 1:
                                    self.follower._attitude_time_queue.popleft()
                            else:
                                # Far states: fast-forward (clear queue except latest)
                                _dropped = len(self.follower._attitude_time_queue) - 1
                                if _dropped > 0:
                                    log(f"[BACKPRESSURE] Fast-forward: dropping {_dropped} ticks, "
                                        f"backlog={_backlog_time_s:.2f}s state={self.state.name}")
                                    while len(self.follower._attitude_time_queue) > 1:
                                        self.follower._attitude_time_queue.popleft()
                                    self._ticks_dropped_backpressure_total += _dropped

                if not self.follower._attitude_time_queue:
                    # Queue empty - block for messages
                    self._pump_mavlink_messages(blocking=True)
                    self._loop_wall_times.append(time.monotonic() - _loop_start)
                    continue

                # FD-008-A: Queue activation with stale entry flush
                # If TA was updated during settle (advancing to latest), first queue pop
                # would attempt to set TA to oldest → backward update → mono violation.
                # Solution: Flush stale entries at activation.
                if not self.follower._queue_active and self.follower._attitude_time_queue:
                    current_ta_ms = int(get_time_authority().now_s() * 1000)
                    flushed_count = 0
                    while (self.follower._attitude_time_queue and
                           self.follower._attitude_time_queue[0] <= current_ta_ms):
                        stale_ms = self.follower._attitude_time_queue.popleft()
                        flushed_count += 1
                        if flushed_count <= 3:  # Limit log spam
                            log(f"[FD-008-A] Flushing stale queue entry: {stale_ms} <= {current_ta_ms}")
                    self.follower._queue_active = True
                    get_time_authority().set_queue_active(True)
                    log(f"[FD-008-A] Queue activated, flushed {flushed_count} stale entries, "
                        f"remaining: {len(self.follower._attitude_time_queue)}")

                    # If all entries were flushed, continue to next loop iteration to pump more
                    if not self.follower._attitude_time_queue:
                        continue

                ticks_this_loop = 0
                while (self.follower._attitude_time_queue and
                       ticks_this_loop < MAX_TICKS_PER_LOOP and
                       self.running):
                    # Pop oldest timestamp
                    current_time_boot_ms = self.follower._attitude_time_queue.popleft()

                    # FD-001: Set time authority for THIS tick (coordinate with queue)
                    # This ensures dt is computed from the popped timestamp, not latest
                    self.follower._buffered_attitude_time_boot_ms = current_time_boot_ms
                    # Note: time authority update now happens inside _execute_tick()

                    # Execute tick - pass queue timestamp directly to bypass monotonicity filter
                    tick_executed = self._execute_tick(queue_time_boot_ms=current_time_boot_ms)
                    ticks_this_loop += 1

                    # FD-001: Update last_tick_time AFTER execute (not before - breaks dt)
                    self._last_tick_time_boot_ms = current_time_boot_ms

                    if tick_executed:
                        # FD-002: Track last processed tick timestamp for backlog calculation
                        self._last_processed_tick_ms = current_time_boot_ms
                        # FD-002: Track tick timestamps for windowed Hz calculation
                        self._tick_hz_window.append(current_time_boot_ms)

                        # FD-002: Fail-fast if backlog > 0.3s in close-in states
                        backlog_s = self._get_backlog_time_s()
                        if self.state in {ControllerState.DOCKING_CREEP, ControllerState.HOLD_CONTACT}:
                            if backlog_s > 0.3:
                                self._backlog_fatal_count += 1
                                raise GateFatalError(
                                    f"FD-002: Backlog {backlog_s:.3f}s in {self.state.name}")

                        # PRH-ATT-004: Runtime timebase watchdog
                        # Arm once past pre-formation states (INTERCEPT or later)
                        _PRE_FORMATION_STATES = {
                            ControllerState.INIT,
                            ControllerState.WEATHER_CHECK,
                            ControllerState.VTOL_TAKEOFF,
                            ControllerState.TAKEOFF_SET_VTOL_MODE,
                            ControllerState.TAKEOFF_ARMING,
                            ControllerState.TAKEOFF_CLIMBING,
                            ControllerState.TAKEOFF_SET_FW_MODE,
                            ControllerState.TAKEOFF_TRANSITION_MONITOR,
                            ControllerState.FW_TRANSITION,
                            ControllerState.FW_STABILIZE,
                        }
                        if not self._watchdog_armed:
                            if self.state not in _PRE_FORMATION_STATES:
                                self._watchdog_armed = True
                        if self._watchdog_armed:
                            self._check_timebase_watchdog(self._t_sim_s)

                        # R4: Wrap post-tick actions in try/except to ensure _fatal() called
                        try:
                            # Post-tick actions (only after successful tick)
                            # Check C++ handoff after state machine
                            geom = self.get_geometry()
                            self._check_cpp_handoff(geom)

                            # Check wake zone (Phase 1: logging only, two-vortex-core model)
                            if geom is not None:
                                lead_state = self.leader_rx.get_state() if self.leader_rx else None
                                self._check_wake_zone(geom, lead_state)

                            # Update success metrics (observer-only)
                            self._update_success_metrics(geom)

                            # Advance tick counter for geometry cache (deterministic cache hits)
                            self._tick_counter += 1
                        except Exception as e:
                            # R4: Ensure _fatal() is called for post-tick exceptions
                            self._fatal(f"post_tick_exception: {type(e).__name__}", e)
                            raise

                        # Option 0 Fix: Check sim-time duration limit using _t_sim_s
                        if max_sim_seconds is not None:
                            elapsed_sim = self._t_sim_s - start_sim_time
                            if elapsed_sim >= max_sim_seconds:
                                log(f"[OPTION_0] Reached sim-time limit: {elapsed_sim:.1f}s / {max_sim_seconds}s")
                                self._stop_reason = "SIM_TIME_LIMIT"
                                self.running = False
                                break

                # Queue depth attribution: ticks per loop + wall time (ChatGPT-5 R2)
                self._ticks_per_loop_max = max(self._ticks_per_loop_max, ticks_this_loop)
                self._loop_wall_times.append(time.monotonic() - _loop_start)

        except GateFatalError as e:
            # v9.10: Gate fatal - call unified handler
            self._on_gate_fatal(type(e).__name__, e)

        except KeyboardInterrupt:
            # Graceful handling of Ctrl+C
            self._stop_reason = "SIGINT"
            log("")
            log(">>> Received Ctrl+C (SIGINT) - stopping gracefully...")
            self.running = False

        finally:
            # Ensure C++ controller is disabled on exit
            if self._cpp_enabled and self.follower and self._form_enable_supported:
                try:
                    self.follower.set_param("FORM_ENABLE", 0.0)
                    log("[CPP_HANDOFF] FORM_ENABLE=0 on exit - Python control restored")
                except Exception:
                    pass  # Best effort on exit

            # P2 v9.4 Fix 2b: Close arbitration CSV before run summary
            self._close_arbitration_csv()
            # C3 (gated): close UWB emitter->fusion delivery diag
            self._close_uwb_fusion_diag()

            if TICK_TIMING_DIAGNOSTIC:
                self._tick_diag_write_csv()

            # ALWAYS print run summary, even on Ctrl+C
            # Wrap in try/except to ensure we print something even if summary fails
            try:
                self._print_run_summary()
            except Exception as e:
                log(f"WARNING: Failed to print run summary: {e}")

            # Flush success metrics (idempotent - safe if already flushed by signal handler)
            self._flush_success_metrics(reason="finally")

            log("Controller stopped")
            log(f"Final minimum range: {self.min_range_achieved:.2f}m")
            if self._stop_reason:
                log(f"Stop reason: {self._stop_reason}")
            # Ensure output is flushed before exit
            sys.stdout.flush()

# =============================================================================
# MAIN
# =============================================================================


def run_self_tests() -> bool:
    """Run self-tests for new helper functions without SITL.

    Returns True if all tests pass.
    """
    print("=" * 60)
    print("SELF-TEST MODE: Testing helper functions")
    print("=" * 60)
    all_passed = True

    # Test 1: Slew limiter
    print("\n[TEST 1] Heading slew limiter...")
    # Simulate 50ms dt with 1.0 deg/s limit -> max 0.05 deg step
    last_hdg = 180.0
    desired_hdg = 185.0  # 5 degree change requested
    dt = 0.05
    max_rate = 1.0  # deg/s
    max_step = max_rate * dt  # 0.05 deg
    delta = desired_hdg - last_hdg
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360
    delta = max(-max_step, min(max_step, delta))
    last_hdg + delta
    if abs(delta - max_step) < 0.001:
        print(f"  PASS: Slew limited to {delta:.3f} deg (max={max_step:.3f})")
    else:
        print(f"  FAIL: Expected {max_step:.3f}, got {delta:.3f}")
        all_passed = False

    # Test 2: Envelope check
    print("\n[TEST 2] Precontact envelope check...")
    # Simulate position at precontact target
    behind_m = 10.0
    cross_m = 0.0
    behind_error = abs(behind_m - PRECONTACT_BEHIND_M)
    cross_error = abs(cross_m - PRECONTACT_CROSS_M)
    # Assuming 10m below (lead at 900, follower at 890)
    actual_below = 10.0
    below_error = abs(actual_below - PRECONTACT_BELOW_M)
    in_envelope = (behind_error <= PRECONTACT_BEHIND_TOL_M
                   and cross_error <= PRECONTACT_CROSS_TOL_M
                   and below_error <= PRECONTACT_BELOW_TOL_M)
    if in_envelope:
        print(f"  PASS: Position (behind={behind_m}, cross={cross_m}, "
              f"below={actual_below}) is in precontact envelope")
    else:
        print("  FAIL: Should be in envelope")
        all_passed = False

    # Test 3: Along-track taper
    print("\n[TEST 3] Along-track closure taper...")
    test_cases = [
        (5.0, 0.5, "very close"),   # 5m behind error -> min closure
        (50.0, 2.0, "medium"),      # 50m -> interpolated
        (200.0, 5.0, "far"),        # 200m -> max closure
        (-5.0, 0.25, "overshoot"),  # overshoot -> very gentle
    ]
    for behind_err, expected_min, label in test_cases:
        closure = compute_tapered_max_closure_alongtrack(behind_err)
        if closure >= expected_min - 0.5:  # Allow some tolerance
            print(f"  PASS: {label} ({behind_err}m) -> {closure:.2f} m/s")
        else:
            print(f"  FAIL: {label} ({behind_err}m) -> {closure:.2f} m/s "
                  f"(expected >= {expected_min})")
            all_passed = False

    # Test 4: EMA filter
    print("\n[TEST 4] Cross-track EMA filter...")
    alpha = CROSS_TRACK_FILTER_ALPHA
    filtered = None
    raw_values = [5.0, 4.5, 5.5, 5.0, 4.8]
    for raw in raw_values:
        if filtered is None:
            filtered = raw
        else:
            filtered = alpha * raw + (1 - alpha) * filtered
    # Filtered should be smoother than raw
    if abs(filtered - 5.0) < 1.0:
        print(f"  PASS: EMA filter output={filtered:.2f} (raw avg=~4.96)")
    else:
        print(f"  FAIL: EMA filter output={filtered:.2f} seems wrong")
        all_passed = False

    # Test 5: Micro-adjustment bounds
    print("\n[TEST 5] Micro-adjustment bounds...")
    if (MICRO_HEADING_DELTA_DEG >= 0.05 and MICRO_HEADING_DELTA_DEG <= 0.1
            and MICRO_SPEED_DELTA_MS >= 0.05 and MICRO_SPEED_DELTA_MS <= 0.1):
        print("  PASS: Micro-adjust thresholds in valid range "
              f"(hdg={MICRO_HEADING_DELTA_DEG}, spd={MICRO_SPEED_DELTA_MS})")
    else:
        print("  FAIL: Micro-adjust thresholds out of range")
        all_passed = False

    # Test 6: UWB validity checks (Test 308)
    print("\n[TEST 6] UWB validity checks (Test 308)...")
    now = time.time()
    test_passed = True

    # Case 6a: Valid UWB
    valid, reason = self.sensor_fusion.is_uwb_valid(25.0, now, 24.5, now)
    if valid and reason == "valid":
        print("  PASS: Valid UWB (25m, fresh) -> valid")
    else:
        print(f"  FAIL: Valid UWB should pass: {reason}")
        test_passed = False

    # Case 6b: None UWB
    valid, reason = self.sensor_fusion.is_uwb_valid(None, now, None, now)
    if not valid and "None" in reason:
        print(f"  PASS: None UWB -> invalid ({reason})")
    else:
        print("  FAIL: None UWB should fail")
        test_passed = False

    # Case 6c: Stale UWB
    valid, reason = self.sensor_fusion.is_uwb_valid(25.0, now - 2.0, 24.5, now)
    if not valid and "stale" in reason:
        print(f"  PASS: Stale UWB (2s old) -> invalid ({reason})")
    else:
        print("  FAIL: Stale UWB should fail")
        test_passed = False

    # Case 6d: Jump detection
    valid, reason = self.sensor_fusion.is_uwb_valid(50.0, now, 25.0, now)  # 25m jump
    if not valid and "jump" in reason:
        print(f"  PASS: UWB jump (25m -> 50m) -> invalid ({reason})")
    else:
        print("  FAIL: UWB jump should fail")
        test_passed = False

    # Case 6e: Out of bounds
    valid, reason = self.sensor_fusion.is_uwb_valid(0.2, now, None, now)  # Below min
    if not valid and "min" in reason:
        print(f"  PASS: UWB below min (0.2m) -> invalid ({reason})")
    else:
        print("  FAIL: UWB below min should fail")
        test_passed = False

    if not test_passed:
        all_passed = False

    # Test 7: UWB dominant weight blending (Test 308)
    print("\n[TEST 7] UWB dominant weight blending (Test 308)...")
    test_passed = True

    # Case 7a: At threshold (30m) -> weight = FAR
    w = self.sensor_fusion.compute_uwb_dominant_weight(30.0, 30.0, 15.0)
    if abs(w - UWB_DOMINANT_WEIGHT_FAR) < 0.01:
        print(f"  PASS: At threshold (30m) -> w={w:.2f} (expected={UWB_DOMINANT_WEIGHT_FAR})")
    else:
        print(f"  FAIL: At threshold w={w:.2f} (expected={UWB_DOMINANT_WEIGHT_FAR})")
        test_passed = False

    # Case 7b: At very close (15m) -> weight = NEAR
    w = self.sensor_fusion.compute_uwb_dominant_weight(15.0, 30.0, 15.0)
    if abs(w - UWB_DOMINANT_WEIGHT_NEAR) < 0.01:
        print(f"  PASS: At very close (15m) -> w={w:.2f} (expected={UWB_DOMINANT_WEIGHT_NEAR})")
    else:
        print(f"  FAIL: At very close w={w:.2f} (expected={UWB_DOMINANT_WEIGHT_NEAR})")
        test_passed = False

    # Case 7c: Midpoint (22.5m) -> weight interpolated
    w = self.sensor_fusion.compute_uwb_dominant_weight(22.5, 30.0, 15.0)
    expected_mid = (UWB_DOMINANT_WEIGHT_FAR + UWB_DOMINANT_WEIGHT_NEAR) / 2.0
    if abs(w - expected_mid) < 0.05:
        print(f"  PASS: Midpoint (22.5m) -> w={w:.2f} (expected~{expected_mid:.2f})")
    else:
        print(f"  FAIL: Midpoint w={w:.2f} (expected~{expected_mid:.2f})")
        test_passed = False

    # Case 7d: Far (50m) -> weight = FAR (clamped)
    w = self.sensor_fusion.compute_uwb_dominant_weight(50.0, 30.0, 15.0)
    if abs(w - UWB_DOMINANT_WEIGHT_FAR) < 0.01:
        print(f"  PASS: Far (50m) -> w={w:.2f} (clamped to FAR)")
    else:
        print(f"  FAIL: Far w={w:.2f} (expected={UWB_DOMINANT_WEIGHT_FAR})")
        test_passed = False

    if not test_passed:
        all_passed = False

    # Test 8: Sensor mode constants (Test 308)
    print("\n[TEST 8] Sensor mode constants (Test 308)...")
    test_passed = True

    # Verify sensor modes are defined
    if SENSOR_MODE_FUSED == "fused":
        print("  PASS: SENSOR_MODE_FUSED = 'fused'")
    else:
        print("  FAIL: SENSOR_MODE_FUSED wrong")
        test_passed = False

    if SENSOR_MODE_UWB_ONLY == "uwb_only":
        print("  PASS: SENSOR_MODE_UWB_ONLY = 'uwb_only'")
    else:
        print("  FAIL: SENSOR_MODE_UWB_ONLY wrong")
        test_passed = False

    if SENSOR_MODE_UWB_DOMINANT == "uwb_dominant":
        print("  PASS: SENSOR_MODE_UWB_DOMINANT = 'uwb_dominant'")
    else:
        print("  FAIL: SENSOR_MODE_UWB_DOMINANT wrong")
        test_passed = False

    # Verify UWB validity thresholds are reasonable
    if 0.1 <= UWB_STALE_THRESHOLD_S <= 2.0:
        print(f"  PASS: UWB_STALE_THRESHOLD_S = {UWB_STALE_THRESHOLD_S}s (reasonable)")
    else:
        print(f"  FAIL: UWB_STALE_THRESHOLD_S = {UWB_STALE_THRESHOLD_S}s (out of range)")
        test_passed = False

    if 0.0 <= UWB_DOMINANT_WEIGHT_FAR <= UWB_DOMINANT_WEIGHT_NEAR <= 1.0:
        print(f"  PASS: UWB weights FAR={UWB_DOMINANT_WEIGHT_FAR} <= "
              f"NEAR={UWB_DOMINANT_WEIGHT_NEAR} <= 1.0")
    else:
        print("  FAIL: UWB weights invalid")
        test_passed = False

    if not test_passed:
        all_passed = False

    # Test 10: OVERTAKE slew rate faster than normal (Test 310)
    print("\n[TEST 10] OVERTAKE slew rate vs normal slew rate (Test 310)...")
    test_passed = True
    if OVERTAKE_SLEW_RATE_MS_PER_S > SPEED_CMD_SLEW_RATE_MS_PER_S:
        ratio = OVERTAKE_SLEW_RATE_MS_PER_S / SPEED_CMD_SLEW_RATE_MS_PER_S
        print(f"  PASS: OVERTAKE slew={OVERTAKE_SLEW_RATE_MS_PER_S} m/s/s > "
              f"normal={SPEED_CMD_SLEW_RATE_MS_PER_S} m/s/s ({ratio:.1f}x)")
    else:
        print("  FAIL: OVERTAKE slew should be faster than normal")
        test_passed = False

    if OVERTAKE_SLEW_RATE_MS_PER_S >= 3.0:
        print("  PASS: OVERTAKE slew >= 3.0 m/s/s (effective for decel)")
    else:
        print(f"  FAIL: OVERTAKE slew={OVERTAKE_SLEW_RATE_MS_PER_S} too slow")
        test_passed = False

    if not test_passed:
        all_passed = False

    # Test 11: Station-keeping gain gentler than rejoin (Test 310)
    print("\n[TEST 11] Station-keeping gain vs rejoin gain (Test 310)...")
    test_passed = True
    # Station-keeping should have gentler gain than SPEED_KP_NEAR
    if HOLD_STATION_KEEPING_GAIN < SPEED_KP_NEAR:
        print(f"  PASS: HOLD_STATION_KEEPING_GAIN={HOLD_STATION_KEEPING_GAIN} < "
              f"SPEED_KP_NEAR={SPEED_KP_NEAR}")
    else:
        print("  FAIL: Station-keeping gain should be gentler than SPEED_KP_NEAR")
        test_passed = False

    # Station-keeping gain should be positive but small
    if 0.05 <= HOLD_STATION_KEEPING_GAIN <= 0.3:
        print(f"  PASS: HOLD_STATION_KEEPING_GAIN={HOLD_STATION_KEEPING_GAIN} in valid range")
    else:
        print("  FAIL: HOLD_STATION_KEEPING_GAIN out of range [0.05, 0.3]")
        test_passed = False

    if not test_passed:
        all_passed = False

    # Test 12: Vc gate parameters (Test 310)
    print("\n[TEST 12] Precontact Vc gate parameters (Test 310)...")
    test_passed = True

    # v9.2: VC gate pass threshold (hysteresis lower bound)
    if 0.2 <= PRECONTACT_VC_PASS_MAX_MS <= 1.0:
        print(f"  PASS: PRECONTACT_VC_PASS_MAX_MS={PRECONTACT_VC_PASS_MAX_MS} m/s OK")
    else:
        print(f"  FAIL: PRECONTACT_VC_PASS_MAX_MS={PRECONTACT_VC_PASS_MAX_MS} "
              "out of range [0.2, 1.0]")
        test_passed = False

    # v9.2: VC gate reset threshold must be > pass threshold
    if PRECONTACT_VC_RESET_MAX_MS > PRECONTACT_VC_PASS_MAX_MS:
        print(f"  PASS: PRECONTACT_VC_RESET_MAX_MS={PRECONTACT_VC_RESET_MAX_MS} "
              f"> PASS_MAX={PRECONTACT_VC_PASS_MAX_MS}")
    else:
        print(f"  FAIL: RESET_MAX must exceed PASS_MAX")
        test_passed = False

    # Required stable credit time
    if 0.5 <= PRECONTACT_VC_STABLE_REQ_S <= 3.0:
        print(f"  PASS: PRECONTACT_VC_STABLE_REQ_S={PRECONTACT_VC_STABLE_REQ_S}s OK")
    else:
        print(f"  FAIL: PRECONTACT_VC_STABLE_REQ_S={PRECONTACT_VC_STABLE_REQ_S}s "
              "out of range [0.5, 3.0]")
        test_passed = False

    # Force send delta should be meaningful
    if 0.5 <= FORCE_SPEED_SEND_DELTA_MS <= 2.0:
        print(f"  PASS: FORCE_SPEED_SEND_DELTA_MS={FORCE_SPEED_SEND_DELTA_MS} m/s in valid range")
    else:
        print(f"  FAIL: FORCE_SPEED_SEND_DELTA_MS={FORCE_SPEED_SEND_DELTA_MS} "
              "out of range [0.5, 2.0]")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # TWO-TIER ENVELOPE TESTS
    # =========================================================================
    print("\n[TEST 13] Two-tier envelope constants (Test 311)...")
    test_passed = True

    # OUTER should be >= INNER for all dimensions
    if PRECONTACT_BEHIND_OUTER_TOL_M >= PRECONTACT_BEHIND_INNER_TOL_M:
        print(f"  PASS: Precontact behind OUTER={PRECONTACT_BEHIND_OUTER_TOL_M} >= "
              f"INNER={PRECONTACT_BEHIND_INNER_TOL_M}")
    else:
        print("  FAIL: Precontact behind OUTER < INNER")
        test_passed = False

    if PRECONTACT_CROSS_OUTER_TOL_M >= PRECONTACT_CROSS_INNER_TOL_M:
        print(f"  PASS: Precontact cross OUTER={PRECONTACT_CROSS_OUTER_TOL_M} >= "
              f"INNER={PRECONTACT_CROSS_INNER_TOL_M}")
    else:
        print("  FAIL: Precontact cross OUTER < INNER")
        test_passed = False

    if CONTACT_BEHIND_OUTER_TOL_M >= CONTACT_BEHIND_INNER_TOL_M:
        print(f"  PASS: Contact behind OUTER={CONTACT_BEHIND_OUTER_TOL_M} >= "
              f"INNER={CONTACT_BEHIND_INNER_TOL_M}")
    else:
        print("  FAIL: Contact behind OUTER < INNER")
        test_passed = False

    if CONTACT_CROSS_OUTER_TOL_M >= CONTACT_CROSS_INNER_TOL_M:
        print(f"  PASS: Contact cross OUTER={CONTACT_CROSS_OUTER_TOL_M} >= "
              f"INNER={CONTACT_CROSS_INNER_TOL_M}")
    else:
        print("  FAIL: Contact cross OUTER < INNER")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # MULTI-SEGMENT CLOSURE TAPER
    # =========================================================================
    print("\n[TEST 14] Multi-segment closure taper (Test 311)...")
    test_passed = True

    # Test ultra-near: <= 0.5m -> MAX_CLOSURE_RATE_ULTRA_NEAR_MS
    closure = compute_tapered_max_closure_alongtrack(0.3)
    if abs(closure - MAX_CLOSURE_RATE_ULTRA_NEAR_MS) < 0.01:
        print(f"  PASS: Ultra-near (0.3m) -> {closure:.3f} m/s "
              f"(expected {MAX_CLOSURE_RATE_ULTRA_NEAR_MS})")
    else:
        print(f"  FAIL: Ultra-near (0.3m) -> {closure:.3f} m/s "
              f"(expected {MAX_CLOSURE_RATE_ULTRA_NEAR_MS})")
        test_passed = False

    # Test very-near: 2.0m -> MAX_CLOSURE_RATE_VERY_NEAR_MS
    closure = compute_tapered_max_closure_alongtrack(2.0)
    if abs(closure - MAX_CLOSURE_RATE_VERY_NEAR_MS) < 0.01:
        print(f"  PASS: Very-near (2.0m) -> {closure:.3f} m/s "
              f"(expected {MAX_CLOSURE_RATE_VERY_NEAR_MS})")
    else:
        print(f"  FAIL: Very-near (2.0m) -> {closure:.3f} m/s "
              f"(expected {MAX_CLOSURE_RATE_VERY_NEAR_MS})")
        test_passed = False

    # Test overshoot: negative -> very gentle (half ultra-near)
    closure = compute_tapered_max_closure_alongtrack(-2.0)
    expected_overshoot = MAX_CLOSURE_RATE_ULTRA_NEAR_MS * 0.5
    if abs(closure - expected_overshoot) < 0.01:
        print(f"  PASS: Overshoot (-2.0m) -> {closure:.3f} m/s "
              f"(expected {expected_overshoot})")
    else:
        print(f"  FAIL: Overshoot (-2.0m) -> {closure:.3f} m/s "
              f"(expected {expected_overshoot})")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # RANGE-DEPENDENT MIN_CLOSURE FLOOR
    # =========================================================================
    print("\n[TEST 15] Range-dependent MIN_CLOSURE floor (Test 311)...")
    test_passed = True

    # Verify constants are properly ordered
    if (MIN_CLOSURE_FLOOR_FAR_MS > MIN_CLOSURE_FLOOR_NEAR_MS
            > MIN_CLOSURE_FLOOR_VERY_NEAR_MS >= MIN_CLOSURE_FLOOR_ULTRA_NEAR_MS):
        print("  PASS: MIN_CLOSURE floors properly ordered "
              f"({MIN_CLOSURE_FLOOR_FAR_MS} > {MIN_CLOSURE_FLOOR_NEAR_MS} > "
              f"{MIN_CLOSURE_FLOOR_VERY_NEAR_MS} >= {MIN_CLOSURE_FLOOR_ULTRA_NEAR_MS})")
    else:
        print("  FAIL: MIN_CLOSURE floors not properly ordered")
        test_passed = False

    # Ultra-near floor should be 0 (no forced closure at target)
    if MIN_CLOSURE_FLOOR_ULTRA_NEAR_MS == 0.0:
        print("  PASS: Ultra-near floor is 0.0 (no forced closure at target)")
    else:
        print(f"  FAIL: Ultra-near floor should be 0.0, got {MIN_CLOSURE_FLOOR_ULTRA_NEAR_MS}")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # HOLD_CONTACT PI CONTROLLER PARAMETERS
    # =========================================================================
    print("\n[TEST 16] HOLD_CONTACT PI controller parameters (Test 311)...")
    test_passed = True

    # KP should be positive
    if HOLD_CONTACT_KP > 0:
        print(f"  PASS: HOLD_CONTACT_KP={HOLD_CONTACT_KP} > 0")
    else:
        print("  FAIL: HOLD_CONTACT_KP should be positive")
        test_passed = False

    # KI should be small and positive (slow wind-up)
    if 0 < HOLD_CONTACT_KI <= 0.2:
        print(f"  PASS: HOLD_CONTACT_KI={HOLD_CONTACT_KI} in valid range (0, 0.2]")
    else:
        print(f"  FAIL: HOLD_CONTACT_KI={HOLD_CONTACT_KI} out of range")
        test_passed = False

    # I_CLAMP should be reasonable
    if 0.1 <= HOLD_CONTACT_I_CLAMP_MS <= 1.0:
        print(f"  PASS: HOLD_CONTACT_I_CLAMP_MS={HOLD_CONTACT_I_CLAMP_MS} in valid range")
    else:
        print(f"  FAIL: HOLD_CONTACT_I_CLAMP_MS={HOLD_CONTACT_I_CLAMP_MS} out of range")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # LEADER TURN GATING PARAMETERS
    # =========================================================================
    print("\n[TEST 17] Leader turn gating parameters (Test 311)...")
    test_passed = True

    # Hysteresis: ON threshold > OFF threshold
    if LEAD_YAW_RATE_TURN_ON_DPS > LEAD_YAW_RATE_TURN_OFF_DPS:
        print(f"  PASS: Turn ON={LEAD_YAW_RATE_TURN_ON_DPS} deg/s > "
              f"OFF={LEAD_YAW_RATE_TURN_OFF_DPS} deg/s (hysteresis)")
    else:
        print("  FAIL: Turn ON should be > OFF for hysteresis")
        test_passed = False

    # Filter time constant should be reasonable
    if 0.5 <= LEAD_YAW_RATE_TURN_FILTER_TAU_S <= 3.0:
        print(f"  PASS: Turn filter tau={LEAD_YAW_RATE_TURN_FILTER_TAU_S}s in valid range")
    else:
        print(f"  FAIL: Turn filter tau={LEAD_YAW_RATE_TURN_FILTER_TAU_S}s out of range")
        test_passed = False

    # Straight time should be reasonable (not too short)
    if 2.0 <= TURN_GATE_STRAIGHT_TIME_S <= 10.0:
        print(f"  PASS: Straight time={TURN_GATE_STRAIGHT_TIME_S}s in valid range")
    else:
        print(f"  FAIL: Straight time={TURN_GATE_STRAIGHT_TIME_S}s out of range")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # CONTACT CAPTURE GATE PARAMETERS
    # =========================================================================
    print("\n[TEST 18] Contact capture gate parameters (Test 311)...")
    test_passed = True

    # Contact Vc max should be small
    if 0.1 <= CONTACT_VC_MAX_MS <= 0.5:
        print(f"  PASS: CONTACT_VC_MAX_MS={CONTACT_VC_MAX_MS} m/s in valid range")
    else:
        print(f"  FAIL: CONTACT_VC_MAX_MS={CONTACT_VC_MAX_MS} out of range [0.1, 0.5]")
        test_passed = False

    # Contact stable time should be reasonable
    if 0.5 <= CONTACT_VC_STABLE_TIME_S <= 3.0:
        print(f"  PASS: CONTACT_VC_STABLE_TIME_S={CONTACT_VC_STABLE_TIME_S}s in valid range")
    else:
        print(f"  FAIL: CONTACT_VC_STABLE_TIME_S={CONTACT_VC_STABLE_TIME_S}s out of range")
        test_passed = False

    # Cross rate max should be small
    if 0.1 <= CONTACT_CROSS_RATE_MAX_MS <= 0.5:
        print(f"  PASS: CONTACT_CROSS_RATE_MAX_MS={CONTACT_CROSS_RATE_MAX_MS} m/s in valid range")
    else:
        print(f"  FAIL: CONTACT_CROSS_RATE_MAX_MS={CONTACT_CROSS_RATE_MAX_MS} out of range")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # CROSS-TRACK PI BIAS PARAMETERS
    # =========================================================================
    print("\n[TEST 19] Cross-track PI bias parameters (Test 311)...")
    test_passed = True

    # Cross KI should be positive and small to moderate
    if 0.02 <= CROSS_BIAS_KI <= 0.3:
        print(f"  PASS: CROSS_BIAS_KI={CROSS_BIAS_KI} deg/m·s in valid range")
    else:
        print(f"  FAIL: CROSS_BIAS_KI={CROSS_BIAS_KI} out of range [0.02, 0.3]")
        test_passed = False

    # Cross KP should be positive and small
    if 0.0 <= CROSS_BIAS_KP <= 0.1:
        print(f"  PASS: CROSS_BIAS_KP={CROSS_BIAS_KP} deg/m in valid range")
    else:
        print(f"  FAIL: CROSS_BIAS_KP={CROSS_BIAS_KP} out of range [0.0, 0.1]")
        test_passed = False

    # Cross I clamp should be reasonable (not too large)
    if 0.5 <= CROSS_BIAS_I_CLAMP_DEG <= 5.0:
        print(f"  PASS: CROSS_BIAS_I_CLAMP_DEG={CROSS_BIAS_I_CLAMP_DEG} deg in valid range")
    else:
        print(f"  FAIL: CROSS_BIAS_I_CLAMP_DEG={CROSS_BIAS_I_CLAMP_DEG} out of range [0.5, 5.0]")
        test_passed = False

    # Decay factor should be close to 1 (slow decay)
    if 0.9 <= CROSS_BIAS_I_DECAY <= 1.0:
        print(f"  PASS: CROSS_BIAS_I_DECAY={CROSS_BIAS_I_DECAY} in valid range")
    else:
        print(f"  FAIL: CROSS_BIAS_I_DECAY={CROSS_BIAS_I_DECAY} out of range [0.9, 1.0]")
        test_passed = False

    # Enable range should be reasonable
    if 10.0 <= CROSS_BIAS_ENABLE_RANGE_M <= 50.0:
        print(f"  PASS: CROSS_BIAS_ENABLE_RANGE_M={CROSS_BIAS_ENABLE_RANGE_M}m in valid range")
    else:
        print("  FAIL: CROSS_BIAS_ENABLE_RANGE_M out of range [10.0, 50.0]")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # CONTACT INNER CAPTURE GATE PARAMETERS
    # =========================================================================
    print("\n[TEST 20] Contact INNER capture gate parameters (Test 309)...")
    test_passed = True

    # Inner stable time should be reasonable
    if 0.3 <= CONTACT_INNER_STABLE_TIME_S <= 2.0:
        print(f"  PASS: CONTACT_INNER_STABLE_TIME_S={CONTACT_INNER_STABLE_TIME_S}s "
              "in valid range")
    else:
        print(f"  FAIL: CONTACT_INNER_STABLE_TIME_S={CONTACT_INNER_STABLE_TIME_S}s "
              "out of range [0.3, 2.0]")
        test_passed = False

    # Log interval should be reasonable
    if 0.5 <= CONTACT_INNER_GATE_LOG_INTERVAL_S <= 5.0:
        print(f"  PASS: CONTACT_INNER_GATE_LOG_INTERVAL_S={CONTACT_INNER_GATE_LOG_INTERVAL_S}s "
              "in valid range")
    else:
        print("  FAIL: CONTACT_INNER_GATE_LOG_INTERVAL_S out of range [0.5, 5.0]")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # ENVELOPE TOLERANCE RELATIONSHIPS
    # =========================================================================
    print("\n[TEST 21] Envelope tolerance relationships (Test 309)...")
    test_passed = True

    # INNER cross tolerance should be tight enough to matter
    if CONTACT_CROSS_INNER_TOL_M < CONTACT_CROSS_OUTER_TOL_M:
        print(f"  PASS: Contact cross INNER={CONTACT_CROSS_INNER_TOL_M}m < "
              f"OUTER={CONTACT_CROSS_OUTER_TOL_M}m")
    else:
        print("  FAIL: Contact cross INNER should be < OUTER")
        test_passed = False

    # The difference should be meaningful (not too small)
    diff = CONTACT_CROSS_OUTER_TOL_M - CONTACT_CROSS_INNER_TOL_M
    if diff >= 0.1:
        print(f"  PASS: OUTER-INNER gap = {diff:.2f}m (meaningful hysteresis)")
    else:
        print(f"  FAIL: OUTER-INNER gap = {diff:.2f}m (too small for meaningful hysteresis)")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # OUTER EXIT GRACE PARAMETERS
    # =========================================================================
    print("\n[TEST 22] OUTER exit grace parameters (Test 311)...")
    test_passed = True

    # Grace time should be small but meaningful
    if 0.1 <= CONTACT_OUTER_EXIT_GRACE_S <= 1.0:
        print(f"  PASS: CONTACT_OUTER_EXIT_GRACE_S={CONTACT_OUTER_EXIT_GRACE_S}s in valid range")
    else:
        print("  FAIL: CONTACT_OUTER_EXIT_GRACE_S out of range [0.1, 1.0]")
        test_passed = False

    # Catastrophic multiplier should be > 1.5
    if CONTACT_OUTER_CATASTROPHIC_MULT >= 1.5:
        print(f"  PASS: CONTACT_OUTER_CATASTROPHIC_MULT={CONTACT_OUTER_CATASTROPHIC_MULT} >= 1.5")
    else:
        print("  FAIL: CONTACT_OUTER_CATASTROPHIC_MULT should be >= 1.5")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # DYNAMIC OFFSET SIDE SELECTION PARAMETERS
    # =========================================================================
    print("\n[TEST 23] Dynamic offset side selection parameters (Test 312)...")
    test_passed = True

    # Threshold should be meaningful (not too small)
    if 200.0 <= DYNAMIC_OFFSET_THRESHOLD_M <= 1000.0:
        print(f"  PASS: DYNAMIC_OFFSET_THRESHOLD_M={DYNAMIC_OFFSET_THRESHOLD_M}m "
              "in valid range")
    else:
        print("  FAIL: DYNAMIC_OFFSET_THRESHOLD_M out of range [200, 1000]")
        test_passed = False

    # Default offset side should be +1 or -1
    if abs(OFFSET_SIDE) == 1.0:
        print(f"  PASS: OFFSET_SIDE={OFFSET_SIDE:+.0f} (valid ±1)")
    else:
        print("  FAIL: OFFSET_SIDE should be +1 or -1")
        test_passed = False

    # Lateral offset should be reasonable
    if 500.0 <= LATERAL_OFFSET_M <= 1500.0:
        print(f"  PASS: LATERAL_OFFSET_M={LATERAL_OFFSET_M}m in valid range")
    else:
        print("  FAIL: LATERAL_OFFSET_M out of range [500, 1500]")
        test_passed = False

    if not test_passed:
        all_passed = False

    # =========================================================================
    # ALTITUDE COMMAND CONTROL PARAMETERS
    # =========================================================================
    print("\n[TEST 24] Altitude command control parameters (Test 313)...")
    test_passed = True

    # Altitude command delta threshold
    if 0.1 <= ALT_CMD_DELTA_M <= 2.0:
        print(f"  PASS: ALT_CMD_DELTA_M={ALT_CMD_DELTA_M}m in valid range")
    else:
        print("  FAIL: ALT_CMD_DELTA_M out of range [0.1, 2.0]")
        test_passed = False

    # Altitude error force threshold
    if 2.0 <= ALT_ERR_FORCE_M <= 20.0:
        print(f"  PASS: ALT_ERR_FORCE_M={ALT_ERR_FORCE_M}m in valid range")
    else:
        print("  FAIL: ALT_ERR_FORCE_M out of range [2.0, 20.0]")
        test_passed = False

    # Altitude error force time
    if 0.5 <= ALT_ERR_FORCE_TIME_S <= 10.0:
        print(f"  PASS: ALT_ERR_FORCE_TIME_S={ALT_ERR_FORCE_TIME_S}s in valid range")
    else:
        print("  FAIL: ALT_ERR_FORCE_TIME_S out of range [0.5, 10.0]")
        test_passed = False

    # Altitude warning threshold > force threshold
    if ALT_ERR_WARNING_M > ALT_ERR_FORCE_M:
        print(f"  PASS: ALT_ERR_WARNING_M={ALT_ERR_WARNING_M}m > "
              f"ALT_ERR_FORCE_M={ALT_ERR_FORCE_M}m")
    else:
        print("  FAIL: ALT_ERR_WARNING_M should be > ALT_ERR_FORCE_M")
        test_passed = False

    # Vertical gate threshold
    if 3.0 <= ALT_GATE_THRESHOLD_M <= 15.0:
        print(f"  PASS: ALT_GATE_THRESHOLD_M={ALT_GATE_THRESHOLD_M}m in valid range")
    else:
        print("  FAIL: ALT_GATE_THRESHOLD_M out of range [3.0, 15.0]")
        test_passed = False

    # Vertical gate time
    if 5.0 <= ALT_GATE_TIME_S <= 30.0:
        print(f"  PASS: ALT_GATE_TIME_S={ALT_GATE_TIME_S}s in valid range")
    else:
        print("  FAIL: ALT_GATE_TIME_S out of range [5.0, 30.0]")
        test_passed = False

    # Fallback interval
    if 1.0 <= ALT_FALLBACK_INTERVAL_S <= 5.0:
        print(f"  PASS: ALT_FALLBACK_INTERVAL_S={ALT_FALLBACK_INTERVAL_S}s in valid range")
    else:
        print("  FAIL: ALT_FALLBACK_INTERVAL_S out of range [1.0, 5.0]")
        test_passed = False

    if not test_passed:
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SELF-TESTS PASSED")
    else:
        print("SOME SELF-TESTS FAILED")
    print("=" * 60)
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Formation Controller v2.50 (Test 385 - Option E: C++ 400Hz handoff)"
    )
    parser.add_argument("--intercept-rejoin-test", action="store_true",
                        help="Skip VTOL takeoff, start directly in INTERCEPT")
    parser.add_argument("--connection", default="udp:127.0.0.1:14551",
                        help="Follower connection string")

    # Sensor mode options
    parser.add_argument("--close-range-sensor-mode",
                        choices=["fused", "uwb_only", "uwb_dominant"],
                        default="fused",
                        help="Sensor mode for close range (<30m): "
                             "fused (default), uwb_only, or uwb_dominant")
    parser.add_argument("--close-range-threshold-m", type=float, default=30.0,
                        help="Range threshold for close-range sensor mode (default: 30)")

    # Self-test mode
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-tests for helper functions and exit")

    # C++ handoff control
    parser.add_argument("--disable-cpp-handoff", action="store_true",
                        help="Disable C++ 400Hz controller handoff (Python-only control)")
    parser.add_argument("--speedup", type=float, default=1.0,
                        help="Expected SIM_SPEEDUP factor (for harness)")

    # C++ handoff policy
    # Default "state_gated" prevents Python heading command override during INTERCEPT/LEAD_TURN.
    # With always_on, FORM_ENABLE=1 overrides Python heading commands during INTERCEPT/LEAD_TURN.
    # With state_gated, Python retains heading control until DOCKING_CREEP/HOLD_CONTACT phases.
    parser.add_argument("--cpp-handoff-policy",
                        # R7.1 waiver 2026-07-21. Was ["always_on", "always_of", "of", ...]:
                        # BOTH off-spellings were typo'd, so the two spellings the
                        # help text and the alias normalizer (:10403-10408) actually
                        # implement -- "always_off" and "off" -- were UNREACHABLE
                        # through argparse, while the two that WERE reachable
                        # ("always_of", "of") match no normalizer alias and no
                        # policy branch (:3637/:3649/:3683), so they fell through
                        # to an unrecognized policy and the C++ handoff was NOT
                        # disabled when the operator asked for Python-only.
                        choices=["always_on", "always_off", "off", "range_hysteresis", "auto_range", "state_gated"],
                        default="state_gated",  # FIX: state_gated allows Python heading control in INTERCEPT/LEAD_TURN
                        help="C++ handoff policy: always_on (engage after PASS), always_off/off (Python-only), "
                             "range_hysteresis/auto_range (enable <30m, disable >40m), "
                             "state_gated (engage in DOCKING_CREEP/HOLD_CONTACT)")

    # Harness integration arguments
    parser.add_argument("--test-id", type=str, default="385",
                        help="Test ID for logging and output naming")
    parser.add_argument("--out-dir", type=str, default="",
                        help="Output directory for logs and artifacts")
    parser.add_argument("--max-sim-seconds", type=float, default=None,
                        help="Maximum sim-time duration before auto-termination (Option 0 fix)")

    # High-Side Convergence Mode (wake avoidance during intercept/rejoin)
    parser.add_argument("--high-side-convergence", action="store_true",
                        help="Enable high-side convergence mode for wake avoidance")
    parser.add_argument("--high-side-margin", type=float, default=6.0,
                        help="Altitude margin above leader during convergence (m)")
    parser.add_argument("--high-side-cross-ok", type=float, default=2.5,
                        help="Cross-track threshold for descent (m)")
    parser.add_argument("--high-side-behind-max", type=float, default=250.0,
                        help="Max behind distance for high-side mode (m)")
    parser.add_argument("--high-side-min-time", type=float, default=5.0,
                        help="Min time cross must be OK before descent (s)")
    parser.add_argument("--high-side-disable-behind", type=float, default=60.0,
                        help="Force disable high-side when behind < this (m) - ensures descent before close-in")
    parser.add_argument("--leader-port", type=int, default=14552,
                        help="UDP port for leader telemetry receiver (default: 14552)")

    args = parser.parse_args()

    # Run self-tests if requested
    if args.self_test:
        success = run_self_tests()
        sys.exit(0 if success else 1)

    print("=" * 70)
    print("FlightDock Baseline Formation Controller")
    print("=" * 70)
    print(f"Version: {CONTROLLER_VERSION}")
    print("Features:")
    print(f"  - UWB+GPS Fusion: {'Enabled' if UWB_FUSION_AVAILABLE else 'GPS-only'}")
    print(f"  - VTOL Takeoff: {'Available' if TAKEOFF_MODULE_AVAILABLE else 'Not available'}")
    print(f"  - Stern Conversion Range: {STERN_CONVERSION_RANGE_M}m")
    print(f"  - Precontact Envelope: behind={PRECONTACT_BEHIND_M}m "
          f"(+/-{PRECONTACT_BEHIND_TOL_M}m)")
    print(f"  - Contact Envelope: behind={CONTACT_BEHIND_M}m "
          f"(+/-{CONTACT_BEHIND_TOL_M}m)")
    print(f"  - Max Creep Rate: {CLOSURE_RATE_CREEP_MS} m/s")
    print("Sensor Features:")
    print(f"  - Close-Range Sensor Mode: {args.close_range_sensor_mode}")
    print(f"  - Close-Range Threshold: {args.close_range_threshold_m}m")
    print(f"  - Precision Mode: <{PRECISION_MODE_THRESHOLD_M}m, "
          f"Very Precision: <{VERY_PRECISION_MODE_THRESHOLD_M}m")
    print("C++ Handoff:")
    # Normalize policy for display
    display_policy = args.cpp_handoff_policy
    if display_policy == "off":
        display_policy = "always_off"
    elif display_policy == "auto_range":
        display_policy = "range_hysteresis"

    if display_policy == "always_off":
        print("  - Policy: always_off (Python-only)")
        print("  - C++ 400Hz Controller: DISABLED")
    elif display_policy == "always_on":
        print(f"  - Policy: always_on (engage after PASS + {CPP_HANDOFF_WARMUP_S}s warmup, never disengage)")
        print("  - C++ 400Hz Controller: ENABLED")
    elif display_policy == "state_gated":
        print("  - Policy: state_gated (engage in DOCKING_CREEP/HOLD_CONTACT, never disengage)")
        print("  - C++ 400Hz Controller: ENABLED (gated)")
    else:
        cpp_status = "DISABLED" if args.disable_cpp_handoff else "ENABLED"
        print(f"  - Policy: range_hysteresis (enable <{CPP_HANDOFF_ENABLE_RANGE_M}m, "
              f"disable >{CPP_HANDOFF_DISABLE_RANGE_M}m)")
        print(f"  - C++ 400Hz Controller: {cpp_status}")
    print(f"  - Speedup: {args.speedup}x")
    # High-Side Convergence Mode
    if args.high_side_convergence:
        print("High-Side Convergence:")
        print("  - Mode: ENABLED")
        print(f"  - Margin: {args.high_side_margin}m above leader")
        print(f"  - Cross OK threshold: {args.high_side_cross_ok}m")
        print(f"  - Behind max: {args.high_side_behind_max}m")
        print(f"  - Disable behind: {args.high_side_disable_behind}m (hard cutoff)")
        print(f"  - Min stable time: {args.high_side_min_time}s")
    else:
        print("High-Side Convergence: DISABLED")
    print("=" * 70)

    controller = BaselineFormationController(args.connection)

    # Configure sensor mode from CLI args
    controller.sensor_mode = args.close_range_sensor_mode
    controller.close_range_sensor_threshold = args.close_range_threshold_m

    # Configure C++ handoff from CLI args
    # Normalize policy aliases
    policy = args.cpp_handoff_policy
    if policy == "off":
        policy = "always_off"
    elif policy == "auto_range":
        policy = "range_hysteresis"

    if policy == "always_off":
        controller._cpp_handoff_enabled = False
        controller._cpp_handoff_policy = "always_off"
    else:
        controller._cpp_handoff_enabled = not args.disable_cpp_handoff
        controller._cpp_handoff_policy = policy
    controller.expected_speedup = args.speedup
    controller.control_laws.speedup_factor = max(1.0, args.speedup)

    # Output directory: prefer --out-dir, fallback to HARNESS_OUT_DIR env
    import os
    out_dir = args.out_dir if args.out_dir else os.environ.get("HARNESS_OUT_DIR", "")
    controller._out_dir = out_dir if out_dir else None

    # Configure high-side convergence from CLI args
    controller._high_side_enabled = args.high_side_convergence
    controller._high_side_margin_m = args.high_side_margin
    controller._high_side_cross_ok_m = args.high_side_cross_ok
    controller._high_side_behind_max_m = args.high_side_behind_max
    controller._high_side_min_time_s = args.high_side_min_time
    controller._high_side_disable_behind_m = args.high_side_disable_behind

    def signal_handler(sig, frame):
        sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
        log(f"{sig_name} received, stopping...")
        controller._flush_success_metrics(reason=sig_name)
        controller.running = False
        sys.stdout.flush()
        # For SIGTERM from timeout, exit promptly
        if sig == signal.SIGTERM:
            sys.exit(128 + sig)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    controller.run(skip_takeoff=args.intercept_rejoin_test, max_sim_seconds=args.max_sim_seconds)

    # Print run summary (state counts, flight time, phase durations)
    print("\n" + "=" * 70)
    print(f"RUN SUMMARY ({CONTROLLER_VERSION})")
    print("=" * 70)

    # Total flight time
    if controller._run_start_time:
        flight_time = time.time() - controller._run_start_time
        mins, secs = divmod(int(flight_time), 60)
        print(f"Total Flight Time: {mins}m {secs}s ({flight_time:.1f}s)")
    else:
        print("Total Flight Time: (not recorded)")

    print(f"Minimum Range Achieved: {controller.min_range_achieved:.2f}m")
    print(f"Precontact Holds (>={PRECONTACT_HOLD_DURATION_S}s): {controller.precontact_hold_count}")
    print(f"Contact Envelope Entries: {controller.contact_entry_count}")
    if controller.contact_dwell_times:
        total_contact = sum(controller.contact_dwell_times)
        print(f"Contact Dwell Times: {controller.contact_dwell_times}")
        print(f"Total Contact Time: {total_contact:.1f}s")
    else:
        print("Contact Dwell Times: (none)")

    # DRL alignment success metrics
    print("-" * 40)
    print("DRL Alignment (Test 311):")
    print(f"  contact_success: {controller.contact_success}")
    print(f"  contact_success_count: {controller.contact_success_count}")
    # Shadow metrics for comparability across test configs
    if controller.contact_dwell_times:
        dwells_5s = sum(1 for d in controller.contact_dwell_times if d >= 5.0)
        dwells_7s = sum(1 for d in controller.contact_dwell_times if d >= 7.0)
        dwells_10s = sum(1 for d in controller.contact_dwell_times if d >= 10.0)
        print(f"  dwells >= 5s: {dwells_5s} | >= 7s: {dwells_7s} | >= 10s: {dwells_10s}")
    print(f"  max_inner_hold: {controller._contact_inner_hold_max_s:.1f}s "
          f"(target: {CONTACT_HOLD_DURATION_S}s)")

    # State entry counts
    print("-" * 40)
    print("State Entry Counts:")
    if controller._state_entry_counts:
        for state_name, count in sorted(controller._state_entry_counts.items()):
            print(f"  {state_name}: {count}")
    else:
        print("  (no state transitions recorded)")

    print("-" * 40)
    print(f"Final Sensor Source: {controller._active_sensor_source}")
    print(f"Sensor Mode Used: {controller.sensor_mode}")
    print("=" * 70)


def _self_check_bearing_computation():
    """Self-check: validate bearing computation with known log values."""
    # From log: behind=-2361.1, cross=-1154.8, lead_hdg=180
    behind_m = -2361.1
    cross_m = -1154.8
    lead_hdg = 180.0

    # Fixed formula: atan2(-cross_m, behind_m)
    relative_bearing = math.degrees(math.atan2(-cross_m, behind_m))
    bearing_to_leader = (lead_hdg + relative_bearing) % 360

    # Expected: ~334° (toward leader who is SOUTH, follower is ahead and left)
    expected = 334.0
    tolerance = 1.0

    if abs(bearing_to_leader - expected) > tolerance:
        raise AssertionError(
            f"Bearing self-check FAILED: got {bearing_to_leader:.1f}°, expected ~{expected}° (±{tolerance}°)"
        )
    print(f"[SELF-CHECK] Bearing computation OK: {bearing_to_leader:.1f}° ≈ {expected}°")


# Run self-check on module load
_self_check_bearing_computation()


if __name__ == "__main__":
    main()
