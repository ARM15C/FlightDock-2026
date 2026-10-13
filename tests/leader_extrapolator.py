#!/usr/bin/env python3
"""
R7M leader time-alignment helper (CSV2_R7M).
============================================

Responsibility
    Pure leader lat/lon dead-reckoning by the *measured* leader-packet age, so
    the controller's GPS geometry (and the UWB synthetic ranges derived from the
    same leader position) reference the leader at the follower's measurement
    instant instead of a ~0.4 s-stale position. Corrects the GPS_LEADER_STALE
    behind-axis under-read confirmed by the R7M RCA.

Public API
    extrapolate_leader_latlon(lat, lon, vn_ms, ve_ms, age_ms, *, log_fn=None)
        -> (lat, lon)
    apply_residual_lag(lat, lon, vn_ms, ve_ms, *, log_fn=None)
        -> (lat, lon) # follower GPS-sensor residual-lag (fixed tau, BBD Hunk-R1)
    extrapolate_follower_truth_latlon(lat, lon, vn_ms, ve_ms, age_ms, *, log_fn=None)
        -> (lat, lon) # follower TRUTH -> UWB-emitter time-align (UWB-TRUTH, 2026-06-14)

Side effects
    None except optional log_fn(...) calls:
      - per-call ``[LEADER_EXTRAP_VERBOSE]`` line when LEADER_EXTRAP_VERBOSE=1
        (FAIL-branch diagnosis; fires before any early return so suppressed
        calls are visible too),
      - ``[LEADER_EXTRAP] WARNING`` when age exceeds the cap (raw passthrough),
      - throttled ``[LEADER_EXTRAP]`` diagnostic every 100th *applied* call.
    The module-level throttle counter ``_APPLIED_N`` is the only mutable state.

Env gates (ALL default OFF -> byte-identical to pre-R7M behavior)
    LEADER_EXTRAP_ENABLE=1 enable the leader-hop extrapolation
    LEADER_EXTRAP_VERBOSE=1 per-call age_ms/dt_s diagnostic (FAIL-branch only)
    FOLLOWER_EXTRAP_ENABLE=1 enable the follower-hop extrapolation (SIGCLEAN
                               Hunk-X1, 2026-06-12: symmetric mirror of Hunk-L
                               for the follower GPI feed; tighter 0.5 s cap
                               because the raised GPI feed runs at 20 Hz)
    FD_FOLLOWER_RESIDUAL_LAG_MS=<ms> enable the follower GPS-sensor residual-lag
                               forward-extrapolation (BBD Hunk-R1, 2026-06-13:
                               fixed-tau, message-age-INDEPENDENT; '0'/unset/
                               out-of-range -> OFF -> byte-identical)
    FD_UWB_TRUTH_EXTRAP_ENABLE=1 enable the follower-TRUTH -> UWB-emitter time-
                               alignment (UWB-TRUTH, 2026-06-14: forward-extrapolate
                               the truth fed to the emitter to the tick instant by its
                               measured age; default unset -> OFF -> byte-identical)

Caps: file hard 240 LoC (raised 195->240 for UWB-TRUTH, declared); per-function <= 25 LoC.
"""
import os
import math

# Cap guards pathological link-loss / packet-loss tails, NOT normal pipeline lag
# (10 Hz interval + ~0.4 s mean lag has a tail that exceeds 0.8 s on valid packets).
MAX_LEADER_EXTRAP_DT_S = 1.5
# Must match tests/geometry.py R_EARTH and diagnostic_frame_log._ned.
R_EARTH_M = 6371000.0
_RAD2DEG = 180.0 / math.pi

_APPLIED_N = 0  # throttle counter for the [LEADER_EXTRAP] applied-call diagnostic


def _enabled() -> bool:
    return os.environ.get("LEADER_EXTRAP_ENABLE") == "1"


def _verbose() -> bool:
    return os.environ.get("LEADER_EXTRAP_VERBOSE") == "1"


def extrapolate_leader_latlon(lat, lon, vn_ms, ve_ms, age_ms, *, log_fn=None):
    """Advance leader (lat, lon) by velocity * clip(age_ms/1000, 0, MAX).

    Returns the inputs unchanged when the gate is OFF, when age_ms is None or
    non-positive (negative-age / clock-skew guard), or when age_ms exceeds the
    MAX cap (pathological staleness -> raw passthrough + WARNING).
    """
    global _APPLIED_N
    if _verbose() and log_fn is not None:
        _dt = (age_ms / 1000.0) if age_ms is not None else float("nan")
        log_fn(f"[LEADER_EXTRAP_VERBOSE] age_ms={age_ms} dt_s={_dt:.4f}")
    if not _enabled():
        return lat, lon
    if age_ms is None or age_ms <= 0:
        return lat, lon
    if age_ms >= MAX_LEADER_EXTRAP_DT_S * 1000.0:
        if log_fn is not None:
            log_fn(f"[LEADER_EXTRAP] WARNING age_ms={age_ms:.1f} exceeds cap "
                   f"{MAX_LEADER_EXTRAP_DT_S * 1000.0:.0f}ms; raw passthrough")
        return lat, lon
    dt_s = age_ms / 1000.0
    dlat = (vn_ms * dt_s) / R_EARTH_M * _RAD2DEG
    dlon = (ve_ms * dt_s) / (R_EARTH_M * math.cos(math.radians(lat))) * _RAD2DEG
    _APPLIED_N += 1
    if log_fn is not None and (_APPLIED_N % 100 == 1):
        log_fn(f"[LEADER_EXTRAP] age_ms={age_ms:.1f} dt_s={dt_s:.4f} "
               f"north_m={vn_ms * dt_s:.3f} east_m={ve_ms * dt_s:.3f}")
    return lat + dlat, lon + dlon


# ---------------------------------------------------------------------------
# SIGCLEAN Hunk-X1 (2026-06-12): follower-hop mirror of Hunk-L.
# Evidence: with leader extrap LIVE, |d(behind_gps)| p50 = 1.24 m on the 5.3%
# GPI stale/catch-up tick pairs vs 0.11 m fresh (pinfix leg diag). Dead-reckon
# the follower GPI lat/lon by its measured time-boot age using the GPI's OWN
# vx/vy (perception-honest; no truth fields). The freshness-INVARIANT +1.6 m
# upstream EKF bias is deliberately NOT compensated here (goal-3: no tuning
# perception toward truth); D-correction nulls its median downstream.
# ---------------------------------------------------------------------------
MAX_FOLLOWER_EXTRAP_DT_S = 0.5
_FOLLOWER_APPLIED_N = 0


def _follower_enabled() -> bool:
    return os.environ.get("FOLLOWER_EXTRAP_ENABLE") == "1"


def extrapolate_follower_latlon(lat, lon, vn_ms, ve_ms, age_ms, *, log_fn=None):
    """Advance follower (lat, lon) by velocity * clip(age_ms/1000, 0, MAX).

    Same contract as extrapolate_leader_latlon: returns inputs unchanged when
    the FOLLOWER_EXTRAP_ENABLE gate is OFF, when age_ms is None/non-positive,
    or when age_ms exceeds MAX_FOLLOWER_EXTRAP_DT_S (passthrough + WARNING).
    """
    global _FOLLOWER_APPLIED_N
    if not _follower_enabled():
        return lat, lon
    if age_ms is None or age_ms <= 0:
        return lat, lon
    if age_ms >= MAX_FOLLOWER_EXTRAP_DT_S * 1000.0:
        if log_fn is not None:
            log_fn(f"[FOLLOWER_EXTRAP] WARNING age_ms={age_ms:.1f} exceeds cap "
                   f"{MAX_FOLLOWER_EXTRAP_DT_S * 1000.0:.0f}ms; raw passthrough")
        return lat, lon
    dt_s = age_ms / 1000.0
    dlat = (vn_ms * dt_s) / R_EARTH_M * _RAD2DEG
    dlon = (ve_ms * dt_s) / (R_EARTH_M * math.cos(math.radians(lat))) * _RAD2DEG
    _FOLLOWER_APPLIED_N += 1
    if log_fn is not None and (_FOLLOWER_APPLIED_N % 100 == 1):
        log_fn(f"[FOLLOWER_EXTRAP] age_ms={age_ms:.1f} dt_s={dt_s:.4f} "
               f"north_m={vn_ms * dt_s:.3f} east_m={ve_ms * dt_s:.3f}")
    return lat + dlat, lon + dlon


# ---------------------------------------------------------------------------
# BBD Hunk-R1 (2026-06-13): follower GPS-sensor residual-lag forward-extrapolation.
# Evidence: behind_bias_disc SITL A/B (SIM_GPS1_LAG_MS in {0,100,300}ms x N=2,
# committed 0f690e6a) proved the standing +0.3 m along-track "behind" disparity
# is TEMPORAL: behind = v_forward * (actual_lag - 88.6ms). EKF3 compensates
# ~88.6ms; the uncompensated tail tau_residual ~= 11.4ms remains in the follower
# EKF OUTPUT position even on a ZERO-AGE fresh GPI packet -- a SEPARATE concern
# from the Hunk-X1 message-age dead-reckoning above. Canonical fix (patent
# US11867803B2 + ArduPilot EKF3 source, deep-research wdx4i58nv): forward-
# extrapolate position by velocity * tau_residual, using the TAIL (~11.4ms) NOT
# the full 100ms (full-lag double-compensates and flips sign). Slope evidence
# (26.8 m/s = FULL forward speed, not v_rel~=0 in formation) => follower-ABSOLUTE
# velocity. Vertical (vd) deliberately NOT extrapolated -> below axis exactly
# invariant. INDEPENDENTLY gated by FD_FOLLOWER_RESIDUAL_LAG_MS (v10 DEFAULT
# '11.4' = the fitted EKF3 tail, adopted 2026-07-02 — lagfix×crab 16/16
# `e074aae9` S1 safety recovery + every leg since flew 11.4; '0'/blank restores
# the byte-identical passthrough); never inspects FOLLOWER_EXTRAP_ENABLE.
# ---------------------------------------------------------------------------
MAX_RESIDUAL_LAG_MS = 50.0  # sanity cap; fitted tail is 11.4ms. >cap or <=0 -> OFF
_RESIDUAL_APPLIED_N = 0


def _residual_lag_ms() -> float:
    """Parsed FD_FOLLOWER_RESIDUAL_LAG_MS; 11.4 (fitted tail) when UNSET (v10
    default); 0.0 (OFF) on '0'/blank/parse-fail or out-of-range
    (>MAX_RESIDUAL_LAG_MS or <=0) -- a fat-fingered '100' fails SAFE to OFF,
    never to a sign-flipping over-correction."""
    raw = os.environ.get("FD_FOLLOWER_RESIDUAL_LAG_MS", "11.4")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return v if 0.0 < v <= MAX_RESIDUAL_LAG_MS else 0.0


def apply_residual_lag(lat, lon, vn_ms, ve_ms, *, log_fn=None):
    """Advance follower (lat, lon) by follower-ABSOLUTE velocity * tau_residual.

    tau_residual (ms) comes from FD_FOLLOWER_RESIDUAL_LAG_MS. Returns inputs
    UNCHANGED when the flag is OFF/0/blank/out-of-range (byte-identical), or when
    vn_ms/ve_ms is None (velocity-source guard -> passthrough). Independent of
    FOLLOWER_EXTRAP_ENABLE and of message age: the EKF3 sensor-lag tail is present
    on a fresh packet. Pure except the throttle counter. vd untouched by design.
    """
    global _RESIDUAL_APPLIED_N
    tau_ms = _residual_lag_ms()
    if tau_ms <= 0.0:
        return lat, lon
    if vn_ms is None or ve_ms is None:
        return lat, lon
    dt_s = tau_ms / 1000.0
    dlat = (vn_ms * dt_s) / R_EARTH_M * _RAD2DEG
    dlon = (ve_ms * dt_s) / (R_EARTH_M * math.cos(math.radians(lat))) * _RAD2DEG
    _RESIDUAL_APPLIED_N += 1
    if log_fn is not None and (_RESIDUAL_APPLIED_N % 100 == 1):
        log_fn(f"[FOLLOWER_RESIDUAL_LAG] tau_ms={tau_ms:.1f} dt_s={dt_s:.4f} "
               f"north_m={vn_ms * dt_s:.4f} east_m={ve_ms * dt_s:.4f}")
    return lat + dlat, lon + dlon


# ---------------------------------------------------------------------------
# UWB-TRUTH time-alignment (2026-06-14): forward-extrapolate the follower TRUTH fed
# to the UWB emitter to the tick instant, removing the ~one-frame SIM_STATE staleness
# (~1.3 m at cruise = admitted-UWB behind jitter; RCA project_uwb_truth_sourcing_jitter).
# Truth vel (SIM_STATE.vn/ve) stays in the emitter feed, NEVER estimator/policy; vd
# untouched. Gated FD_UWB_TRUTH_EXTRAP_ENABLE (default OFF -> byte-identical).
# ---------------------------------------------------------------------------
_UWB_TRUTH_APPLIED_N = 0


def _uwb_truth_extrap_enabled() -> bool:
    return os.environ.get("FD_UWB_TRUTH_EXTRAP_ENABLE") == "1"


def extrapolate_follower_truth_latlon(lat, lon, vn_ms, ve_ms, age_ms, *, log_fn=None):
    """Advance follower TRUTH (lat, lon) by truth velocity * clip(age_ms/1000, 0, MAX).

    For the UWB emitter's truth input ONLY; mirrors extrapolate_follower_latlon's
    flat-earth formula EXACTLY. Returns inputs unchanged when FD_UWB_TRUTH_EXTRAP_ENABLE
    is OFF, vn_ms/ve_ms is None, age_ms is None/<=0, or age_ms exceeds the cap (+
    WARNING). vd untouched by design (below axis invariant); pure but for the counter.
    """
    global _UWB_TRUTH_APPLIED_N
    if not _uwb_truth_extrap_enabled():
        return lat, lon
    if vn_ms is None or ve_ms is None or age_ms is None or age_ms <= 0:
        return lat, lon
    if age_ms >= MAX_FOLLOWER_EXTRAP_DT_S * 1000.0:
        if log_fn is not None:
            log_fn(f"[UWB_TRUTH_EXTRAP] WARNING age_ms={age_ms:.1f} exceeds cap "
                   f"{MAX_FOLLOWER_EXTRAP_DT_S * 1000.0:.0f}ms; raw passthrough")
        return lat, lon
    dt_s = age_ms / 1000.0
    dlat = (vn_ms * dt_s) / R_EARTH_M * _RAD2DEG
    dlon = (ve_ms * dt_s) / (R_EARTH_M * math.cos(math.radians(lat))) * _RAD2DEG
    _UWB_TRUTH_APPLIED_N += 1
    if log_fn is not None and (_UWB_TRUTH_APPLIED_N % 100 == 1):
        log_fn(f"[UWB_TRUTH_EXTRAP] age_ms={age_ms:.1f} dt_s={dt_s:.4f} "
               f"north_m={vn_ms * dt_s:.3f} east_m={ve_ms * dt_s:.3f}")
    return lat + dlat, lon + dlon
