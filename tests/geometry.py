#!/usr/bin/env python3
"""
FlightDock Geometry Module (Phase 1 Refactor)
==============================================

Provides all geometry calculations for the formation controller:
- Relative position computation (behind, cross, below)
- Envelope checks (precontact/contact, inner/outer)
- Closure rate computation with multi-segment taper
- UWB/GPS fusion support
- Coordinate conversions

GEOMETRY CONTRACT (Phase 0):
- behind_m: distance behind boom TIP (not leader body)
- cross_m: lateral offset from boom TIP
- below_m: vertical distance below boom TIP (not leader body)
- range_m: slant range to boom TIP

All envelope checks, altitude commands, and metrics use this definition.
Boom tip is 4.31m below leader CG (BOOM_TIP_BELOW_OFFSET_M).
(45-deg boom: 6.1 * sin(45deg) = 4.31m vertical component.)

(C) 2025-2026 FlightDock Research Project
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# Import time interface for timestamps
from time_interface import get_time_authority, is_sim_time_ready, sim_now_s

# Import UWB fusion module (optional)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "relative_pose")
)
try:
    from uwb_gps_fusion import (
        FusedState,
        GPSMeasurement,
        RelativePoseFusion,
        UWBMeasurement,
        create_gps_measurement,
    )

    UWB_FUSION_AVAILABLE = True
except ImportError:
    UWB_FUSION_AVAILABLE = False
    RelativePoseFusion = None  # type: ignore
    create_gps_measurement = None  # type: ignore
    UWBMeasurement = None  # type: ignore


__all__ = [
    # Dataclasses
    "RelativeGeometry",
    "RawSensorData",
    # GeometryCalculator class
    "GeometryCalculator",
    # Envelope functions
    "is_in_body_envelope",
    "is_in_precontact_outer_envelope",
    "is_in_precontact_inner_envelope",
    "is_in_contact_outer_envelope",
    "is_in_contact_inner_envelope",
    "is_in_precontact_envelope",
    "is_in_contact_envelope",
    "get_envelope_errors",
    # Closure rate functions
    "compute_tapered_max_closure",
    "compute_tapered_max_closure_alongtrack",
    # Utility functions
    "compute_range",
    "wrap_heading",
    "wrap_heading_error",
    "compute_isa_density_ratio",
    "tas_to_eas",
    # UWB validation
    "is_uwb_valid",
    "compute_uwb_dominant_weight",
    # Constants
    "BOOM_TIP_BELOW_OFFSET_M",
    "BOOM_TIP_OFFSET_M",
    "BOOM_STOWED_LENGTH_M",
    "BOOM_MAX_LENGTH_M",
    "PRECONTACT_BEHIND_M",
    "PRECONTACT_CROSS_M",
    "PRECONTACT_BELOW_M",
    "CONTACT_BEHIND_M",
    "CONTACT_CROSS_M",
    "CONTACT_BELOW_M",
    # Outer tolerances
    "PRECONTACT_BEHIND_OUTER_TOL_M",
    "PRECONTACT_CROSS_OUTER_TOL_M",
    "PRECONTACT_BELOW_OUTER_TOL_M",
    "CONTACT_BEHIND_OUTER_TOL_M",
    "CONTACT_CROSS_OUTER_TOL_M",
    "CONTACT_BELOW_OUTER_TOL_M",
    # Inner tolerances
    "PRECONTACT_BEHIND_INNER_TOL_M",
    "PRECONTACT_CROSS_INNER_TOL_M",
    "PRECONTACT_BELOW_INNER_TOL_M",
    "CONTACT_BEHIND_INNER_TOL_M",
    "CONTACT_CROSS_INNER_TOL_M",
    "CONTACT_BELOW_INNER_TOL_M",
    # Legacy tolerances
    "PRECONTACT_BEHIND_TOL_M",
    "PRECONTACT_CROSS_TOL_M",
    "PRECONTACT_BELOW_TOL_M",
    "CONTACT_BEHIND_TOL_M",
    # Closure taper constants
    "CLOSURE_TAPER_FAR_RANGE_M",
    "CLOSURE_TAPER_MID_RANGE_M",
    "CLOSURE_TAPER_NEAR_RANGE_M",
    "CLOSURE_TAPER_VERY_NEAR_ERR_M",
    "CLOSURE_TAPER_ULTRA_NEAR_ERR_M",
    "MAX_CLOSURE_RATE_FAR_MS",
    "MAX_CLOSURE_RATE_MID_MS",
    "MAX_CLOSURE_RATE_NEAR_MS",
    "MAX_CLOSURE_RATE_VERY_NEAR_MS",
    "MAX_CLOSURE_RATE_ULTRA_NEAR_MS",
    # UWB constants
    "UWB_STALE_THRESHOLD_S",
    "UWB_RANGE_MIN_M",
    "UWB_RANGE_MAX_M",
    "UWB_JUMP_THRESHOLD_M",
    "UWB_NAN_IS_INVALID",
    "UWB_DOMINANT_WEIGHT_FAR",
    "UWB_DOMINANT_WEIGHT_NEAR",
    # ISA constants
    "ISA_SEA_LEVEL_TEMP",
    "ISA_TEMPERATURE_LAPSE_RATE",
    # Feature flag
    "UWB_FUSION_AVAILABLE",
]


# =============================================================================
# GEOMETRY CONSTANTS
# =============================================================================

# Flying Boom parameters (from DHC-2 Beaver)
# Tranche F: Aligned with 45-deg boom model. 6.1 * sin(45deg) = 4.31m below leader CG.
BOOM_TIP_BELOW_OFFSET_M = 4.31  # Vertical component of boom tip position below leader CG
BOOM_TIP_OFFSET_M = BOOM_TIP_BELOW_OFFSET_M  # Legacy alias
BOOM_STOWED_LENGTH_M = 2.4  # Stowed boom length
BOOM_MAX_LENGTH_M = 6.1  # Fully extended boom length

# Precontact position targets (in leader body frame, relative to boom TIP)
PRECONTACT_BEHIND_M = 10.0  # Target distance behind boom tip
PRECONTACT_CROSS_M = 0.0  # Target lateral offset (centered)
PRECONTACT_BELOW_M = 10.0  # Target altitude below boom tip

# Contact position targets (at boom tip)
CONTACT_BEHIND_M = 0.0  # At boom tip (0m behind)
CONTACT_CROSS_M = 0.0  # Centered laterally
CONTACT_BELOW_M = 0.0  # Altitude matched (boom extends down)

# Envelope tolerances: SINGLE-SOURCED from formation_config (T-386).
# These were duplicated literals here and had silently diverged from the
# authoritative values that the live path (EnvelopeChecker -> controller,
# DRL env) actually uses: BELOW_OUTER 3.0 vs 4.0 (TEST 400), BEHIND_INNER
# 1.0 vs 2.0 and CROSS_INNER 1.5 vs 2.0 (TEST 400, M2-milestone alignment).
# formation_config is a leaf module (stdlib-only imports), so this cannot
# cycle. Re-exported under the same names, including the legacy aliases
# below, so every existing importer of this module is unaffected.
# Pinned by tests/unit/test_tolerance_single_source.py.
import formation_config as _fc  # noqa: E402

# Precontact OUTER tolerances (for capture/stay logic)
PRECONTACT_BEHIND_OUTER_TOL_M = _fc.PRECONTACT_BEHIND_OUTER_TOL_M
PRECONTACT_CROSS_OUTER_TOL_M = _fc.PRECONTACT_CROSS_OUTER_TOL_M
PRECONTACT_BELOW_OUTER_TOL_M = _fc.PRECONTACT_BELOW_OUTER_TOL_M

# Precontact INNER tolerances (for dwell accumulation)
PRECONTACT_BEHIND_INNER_TOL_M = _fc.PRECONTACT_BEHIND_INNER_TOL_M
PRECONTACT_CROSS_INNER_TOL_M = _fc.PRECONTACT_CROSS_INNER_TOL_M
PRECONTACT_BELOW_INNER_TOL_M = _fc.PRECONTACT_BELOW_INNER_TOL_M

# Contact OUTER tolerances (for capture/stay logic)
CONTACT_BEHIND_OUTER_TOL_M = _fc.CONTACT_BEHIND_OUTER_TOL_M
CONTACT_CROSS_OUTER_TOL_M = _fc.CONTACT_CROSS_OUTER_TOL_M
CONTACT_BELOW_OUTER_TOL_M = _fc.CONTACT_BELOW_OUTER_TOL_M

# Contact INNER tolerances (Stage 1 precision - corrected to ±0.20m all axes)
CONTACT_BEHIND_INNER_TOL_M = _fc.CONTACT_BEHIND_INNER_TOL_M
CONTACT_CROSS_INNER_TOL_M = _fc.CONTACT_CROSS_INNER_TOL_M
CONTACT_BELOW_INNER_TOL_M = _fc.CONTACT_BELOW_INNER_TOL_M

# Legacy tolerance aliases (compatibility)
PRECONTACT_BEHIND_TOL_M = PRECONTACT_BEHIND_OUTER_TOL_M
PRECONTACT_CROSS_TOL_M = PRECONTACT_CROSS_OUTER_TOL_M
PRECONTACT_BELOW_TOL_M = PRECONTACT_BELOW_OUTER_TOL_M
CONTACT_BEHIND_TOL_M = CONTACT_BEHIND_OUTER_TOL_M


# =============================================================================
# CLOSURE TAPER CONSTANTS
# =============================================================================

# Range thresholds for multi-segment taper
CLOSURE_TAPER_FAR_RANGE_M = 150.0  # Range beyond which max closure applies
CLOSURE_TAPER_MID_RANGE_M = 50.0  # Mid-range threshold
CLOSURE_TAPER_NEAR_RANGE_M = 20.0  # Range at which closure should be minimal
CLOSURE_TAPER_VERY_NEAR_ERR_M = 2.0  # Below this, use very-near closure limit
CLOSURE_TAPER_ULTRA_NEAR_ERR_M = 0.5  # Below this, use ultra-near (nearly stopped)

# Closure rate limits by range segment
MAX_CLOSURE_RATE_FAR_MS = 5.0  # Max closure when far (m/s)
MAX_CLOSURE_RATE_MID_MS = 3.0  # Max closure at mid-range (50m)
MAX_CLOSURE_RATE_NEAR_MS = 1.0  # Max closure when near (m/s)
MAX_CLOSURE_RATE_VERY_NEAR_MS = 0.15  # Max closure at 2m error
MAX_CLOSURE_RATE_ULTRA_NEAR_MS = 0.05  # Max closure at 0.5m error


# =============================================================================
# UWB VALIDITY CONSTANTS
# =============================================================================

UWB_STALE_THRESHOLD_S = 0.5  # Max age for UWB data before stale
UWB_RANGE_MIN_M = 0.5  # Minimum valid UWB range
UWB_RANGE_MAX_M = 150.0  # Maximum valid UWB range
UWB_JUMP_THRESHOLD_M = 10.0  # Max range jump between consecutive samples
UWB_NAN_IS_INVALID = True  # Treat NaN UWB values as invalid

# UWB-dominant blending weights
UWB_DOMINANT_WEIGHT_FAR = 0.5  # UWB weight at close_range_threshold
UWB_DOMINANT_WEIGHT_NEAR = 0.9  # UWB weight at very_precision threshold


# =============================================================================
# ISA ATMOSPHERE CONSTANTS
# =============================================================================

ISA_SEA_LEVEL_TEMP = 288.15  # K (15°C)
ISA_TEMPERATURE_LAPSE_RATE = 0.0065  # K/m


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class RelativeGeometry:
    """Relative geometry between follower and leader (boom tip).

    All distances are in meters, referenced to the boom TIP position
    (not leader body center).
    """

    behind_m: float  # Distance behind leader (positive = behind)
    cross_m: float  # Cross-track offset (positive = right of leader)
    below_m: float  # Vertical offset (positive = below leader)
    range_m: float  # 3D range to boom tip
    target_aspect_deg: float  # Angle from leader's tail (0 = directly behind)
    closure_rate_ms: float  # Rate of closure (positive = closing)
    vel_cross_m_s: float  # Cross-track rate (m/s) - for D-term damping
    vel_below_m_s: float  # Vertical rate (m/s) - for altitude D-term damping
    source: str  # "FUSED", "GPS", "UWB", or "UWB_DOM"


@dataclass
class RawSensorData:
    """Raw sensor measurements for sensor mode switching.

    Holds both GPS-derived and UWB-derived geometry so the controller
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


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


# =============================================================================
# UWB VALIDATION
# =============================================================================


def is_uwb_valid(
    uwb_range: Optional[float],
    uwb_timestamp: float,
    last_uwb_range: Optional[float],
    current_time: Optional[float] = None,
) -> Tuple[bool, str]:
    """Check if UWB measurement is valid for use.

    Args:
        uwb_range: Current UWB range measurement (may be None)
        uwb_timestamp: Timestamp of UWB measurement
        last_uwb_range: Previous UWB range (for jump detection)
        current_time: Current time (default: sim_now_s())

    Returns:
        Tuple of (is_valid, reason_string)
    """
    if current_time is None:
        if is_sim_time_ready():
            current_time = sim_now_s()
        else:
            import time

            current_time = time.time()

    # Check for None
    if uwb_range is None:
        return False, "uwb_range is None"

    # Check for NaN
    if UWB_NAN_IS_INVALID and math.isnan(uwb_range):
        return False, "uwb_range is NaN"

    # Check for out-of-bounds
    if uwb_range < UWB_RANGE_MIN_M:
        return False, f"uwb_range {uwb_range:.1f}m < min {UWB_RANGE_MIN_M}m"

    if uwb_range > UWB_RANGE_MAX_M:
        return False, f"uwb_range {uwb_range:.1f}m > max {UWB_RANGE_MAX_M}m"

    # Check for stale data
    age = current_time - uwb_timestamp
    if age > UWB_STALE_THRESHOLD_S:
        return False, f"uwb_age {age:.2f}s > stale_threshold {UWB_STALE_THRESHOLD_S}s"

    # Check for jumps (if we have previous data)
    if last_uwb_range is not None:
        jump = abs(uwb_range - last_uwb_range)
        if jump > UWB_JUMP_THRESHOLD_M:
            return False, f"uwb_jump {jump:.1f}m > threshold {UWB_JUMP_THRESHOLD_M}m"

    return True, "valid"


def compute_uwb_dominant_weight(range_m: float, threshold_m: float, very_close_m: float) -> float:
    """Compute UWB weight for uwb_dominant mode.

    Weight ramps from UWB_DOMINANT_WEIGHT_FAR at threshold
    to UWB_DOMINANT_WEIGHT_NEAR at very_close.

    Args:
        range_m: Current range
        threshold_m: Close range threshold (weight = FAR here)
        very_close_m: Very close threshold (weight = NEAR here)

    Returns:
        UWB weight between 0.0 and 1.0
    """
    if range_m >= threshold_m:
        return UWB_DOMINANT_WEIGHT_FAR

    if range_m <= very_close_m:
        return UWB_DOMINANT_WEIGHT_NEAR

    # Linear interpolation
    t = (range_m - very_close_m) / (threshold_m - very_close_m)
    return UWB_DOMINANT_WEIGHT_NEAR + t * (UWB_DOMINANT_WEIGHT_FAR - UWB_DOMINANT_WEIGHT_NEAR)


# =============================================================================
# CLOSURE RATE FUNCTIONS
# =============================================================================


def compute_tapered_max_closure(range_m: float, target_range_m: float) -> float:
    """Compute range-proportional maximum closure rate.

    Returns a closure rate that tapers from MAX_CLOSURE_RATE_FAR_MS when far
    to MAX_CLOSURE_RATE_NEAR_MS when near the target.
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
    """Compute closure rate limit based on along-track distance.

    Multi-segment taper for gentler approach near target:
    - >= 150m: 5.0 m/s (far, aggressive closure)
    - 50-150m: linear interpolation from 3.0 to 5.0 m/s (mid-far)
    - 20-50m: linear interpolation from 1.0 to 3.0 m/s (mid-near)
    - 2-20m: linear interpolation to 0.15 m/s (very near)
    - 0.5-2m: linear interpolation to 0.05 m/s (ultra near)
    - <= 0.5m: 0.05 m/s (nearly stopped)
    - <= 0m: 0.025 m/s (overshoot - very gentle)

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
        t = (behind_error_m - CLOSURE_TAPER_ULTRA_NEAR_ERR_M) / (
            CLOSURE_TAPER_VERY_NEAR_ERR_M - CLOSURE_TAPER_ULTRA_NEAR_ERR_M
        )
        return MAX_CLOSURE_RATE_ULTRA_NEAR_MS + t * (
            MAX_CLOSURE_RATE_VERY_NEAR_MS - MAX_CLOSURE_RATE_ULTRA_NEAR_MS
        )

    if behind_error_m <= CLOSURE_TAPER_NEAR_RANGE_M:
        # Near (2m - 20m) - interpolate very-near to near
        t = (behind_error_m - CLOSURE_TAPER_VERY_NEAR_ERR_M) / (
            CLOSURE_TAPER_NEAR_RANGE_M - CLOSURE_TAPER_VERY_NEAR_ERR_M
        )
        return MAX_CLOSURE_RATE_VERY_NEAR_MS + t * (
            MAX_CLOSURE_RATE_NEAR_MS - MAX_CLOSURE_RATE_VERY_NEAR_MS
        )

    if behind_error_m >= CLOSURE_TAPER_FAR_RANGE_M:
        # Far (>= 150m) - maximum closure
        return MAX_CLOSURE_RATE_FAR_MS

    # Split medium range into two segments for more aggressive mid-range closure
    if behind_error_m >= CLOSURE_TAPER_MID_RANGE_M:
        # Mid-far (50m - 150m) - interpolate from mid to far
        t = (behind_error_m - CLOSURE_TAPER_MID_RANGE_M) / (
            CLOSURE_TAPER_FAR_RANGE_M - CLOSURE_TAPER_MID_RANGE_M
        )
        return MAX_CLOSURE_RATE_MID_MS + t * (MAX_CLOSURE_RATE_FAR_MS - MAX_CLOSURE_RATE_MID_MS)

    # Mid-near (20m - 50m) - interpolate from near to mid
    t = (behind_error_m - CLOSURE_TAPER_NEAR_RANGE_M) / (
        CLOSURE_TAPER_MID_RANGE_M - CLOSURE_TAPER_NEAR_RANGE_M
    )
    return MAX_CLOSURE_RATE_NEAR_MS + t * (MAX_CLOSURE_RATE_MID_MS - MAX_CLOSURE_RATE_NEAR_MS)


# =============================================================================
# ENVELOPE FUNCTIONS
# =============================================================================


def is_in_body_envelope(
    geom: RelativeGeometry,
    lead_alt_msl: float,
    follower_alt_msl: float,
    target_behind: float,
    target_cross: float,
    target_below: float,
    tol_behind: float,
    tol_cross: float,
    tol_below: float,
) -> bool:
    """Generic body-frame envelope check.

    Args:
        geom: RelativeGeometry with behind_m, cross_m fields
        lead_alt_msl: Leader altitude MSL (m)
        follower_alt_msl: Follower altitude MSL (m)
        target_behind: Target behind distance (m)
        target_cross: Target cross offset (m)
        target_below: Target below offset (m)
        tol_behind: Tolerance for behind (m)
        tol_cross: Tolerance for cross (m)
        tol_below: Tolerance for below (m)

    Returns:
        True if within all three envelope tolerances
    """
    behind_error = abs(geom.behind_m - target_behind)
    cross_error = abs(geom.cross_m - target_cross)
    # Phase 0 geometry contract: below is referenced to the boom TIP.
    actual_below = (lead_alt_msl - BOOM_TIP_OFFSET_M) - follower_alt_msl
    below_error = abs(actual_below - target_below)

    return behind_error <= tol_behind and cross_error <= tol_cross and below_error <= tol_below


def is_in_precontact_outer_envelope(
    geom: RelativeGeometry, lead_alt_msl: float, follower_alt_msl: float
) -> bool:
    """Check if in precontact OUTER envelope (for capture/stay logic)."""
    return is_in_body_envelope(
        geom,
        lead_alt_msl,
        follower_alt_msl,
        PRECONTACT_BEHIND_M,
        PRECONTACT_CROSS_M,
        PRECONTACT_BELOW_M,
        PRECONTACT_BEHIND_OUTER_TOL_M,
        PRECONTACT_CROSS_OUTER_TOL_M,
        PRECONTACT_BELOW_OUTER_TOL_M,
    )


def is_in_precontact_inner_envelope(
    geom: RelativeGeometry, lead_alt_msl: float, follower_alt_msl: float
) -> bool:
    """Check if in precontact INNER envelope (for dwell accumulation)."""
    return is_in_body_envelope(
        geom,
        lead_alt_msl,
        follower_alt_msl,
        PRECONTACT_BEHIND_M,
        PRECONTACT_CROSS_M,
        PRECONTACT_BELOW_M,
        PRECONTACT_BEHIND_INNER_TOL_M,
        PRECONTACT_CROSS_INNER_TOL_M,
        PRECONTACT_BELOW_INNER_TOL_M,
    )


def is_in_contact_outer_envelope(
    geom: RelativeGeometry, lead_alt_msl: float, follower_alt_msl: float
) -> bool:
    """Check if in contact OUTER envelope (for capture/stay logic)."""
    return is_in_body_envelope(
        geom,
        lead_alt_msl,
        follower_alt_msl,
        CONTACT_BEHIND_M,
        CONTACT_CROSS_M,
        CONTACT_BELOW_M,
        CONTACT_BEHIND_OUTER_TOL_M,
        CONTACT_CROSS_OUTER_TOL_M,
        CONTACT_BELOW_OUTER_TOL_M,
    )


def is_in_contact_inner_envelope(
    geom: RelativeGeometry, lead_alt_msl: float, follower_alt_msl: float
) -> bool:
    """Check if in contact INNER envelope (for dwell accumulation)."""
    return is_in_body_envelope(
        geom,
        lead_alt_msl,
        follower_alt_msl,
        CONTACT_BEHIND_M,
        CONTACT_CROSS_M,
        CONTACT_BELOW_M,
        CONTACT_BEHIND_INNER_TOL_M,
        CONTACT_CROSS_INNER_TOL_M,
        CONTACT_BELOW_INNER_TOL_M,
    )


def is_in_precontact_envelope(
    geom: RelativeGeometry, lead_alt_msl: float, follower_alt_msl: float
) -> bool:
    """Check if in precontact envelope (legacy: uses OUTER for compatibility)."""
    return is_in_precontact_outer_envelope(geom, lead_alt_msl, follower_alt_msl)


def is_in_contact_envelope(
    geom: RelativeGeometry, lead_alt_msl: float, follower_alt_msl: float
) -> bool:
    """Check if in contact envelope (legacy: uses OUTER for compatibility)."""
    return is_in_contact_outer_envelope(geom, lead_alt_msl, follower_alt_msl)


def get_envelope_errors(
    geom: RelativeGeometry,
    lead_alt_msl: float,
    follower_alt_msl: float,
    target_behind: float,
    target_cross: float,
    target_below: float,
) -> Tuple[float, float, float]:
    """Get envelope errors for logging.

    Returns:
        Tuple of (behind_error, cross_error, below_error)
    """
    behind_error = geom.behind_m - target_behind
    cross_error = geom.cross_m - target_cross
    # Phase 0 geometry contract: below is referenced to the boom TIP.
    actual_below = (lead_alt_msl - BOOM_TIP_OFFSET_M) - follower_alt_msl
    below_error = actual_below - target_below

    return (behind_error, cross_error, below_error)


# =============================================================================
# GEOMETRY CALCULATOR CLASS
# =============================================================================


class GeometryCalculator:
    """Computes relative geometry with UWB fusion support.

    Geometry contract (leader body frame):
      - behind_m, cross_m, below_m are defined relative to the boom TIP.
      - range_m is the 3D slant range to the boom TIP.
    """

    def __init__(self, home_lat: float, home_lon: float, home_alt_msl: float):
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.home_alt_msl = home_alt_msl
        self.R_EARTH = 6371000.0

        # UWB fusion filter
        self.fusion: Optional[RelativePoseFusion] = None
        if UWB_FUSION_AVAILABLE:
            self.fusion = RelativePoseFusion()

        # Closure rate tracking
        self._last_range = 0.0
        self._last_range_time = 0.0

    def latlon_to_ned(self, lat: float, lon: float, alt_msl: float) -> Tuple[float, float, float]:
        """Convert lat/lon/alt to NED relative to home."""
        dlat = math.radians(lat - self.home_lat)
        dlon = math.radians(lon - self.home_lon)
        n = dlat * self.R_EARTH
        e = dlon * self.R_EARTH * math.cos(math.radians(self.home_lat))
        d = -(alt_msl - self.home_alt_msl)
        return (n, e, d)

    def compute_relative(
        self,
        foll_lat: float,
        foll_lon: float,
        foll_alt: float,
        foll_vn: float,
        foll_ve: float,
        foll_vd: float,
        lead_lat: float,
        lead_lon: float,
        lead_alt: float,
        lead_vn: float,
        lead_ve: float,
        lead_vd: float,
        lead_heading_deg: float,
        boom_length_m: float,
        uwb_range: Optional[float] = None,
        timestamp_s: Optional[float] = None,
        boom_tip_offset_m: float = BOOM_TIP_OFFSET_M,
    ) -> Optional[RelativeGeometry]:
        """Compute relative geometry in leader body frame.

        Geometry contract (leader body frame):
          - behind_m, cross_m, below_m are defined relative to the boom TIP.
          - range_m is the 3D slant range to the boom TIP.

        Note: pass boom_tip_offset_m=0.0 when you explicitly want leader-center
        geometry (e.g., collision/overtake safety checks).

        Args:
            foll_lat, foll_lon, foll_alt: Follower position
            foll_vn, foll_ve, foll_vd: Follower velocity NED
            lead_lat, lead_lon, lead_alt: Leader position
            lead_vn, lead_ve, lead_vd: Leader velocity NED
            lead_heading_deg: Leader heading in degrees
            boom_length_m: Boom extension length
            uwb_range: Optional UWB range measurement
            timestamp_s: Optional sim-time timestamp (default: sim_now_s())
            boom_tip_offset_m: Vertical offset for boom tip (default: 4.31m)

        Returns:
            RelativeGeometry or None if timestamp unavailable
        """
        # Use sim-time timestamp if provided, otherwise use SimTimeAuthority
        ta = get_time_authority()
        now = (
            timestamp_s
            if timestamp_s is not None
            else (ta.now_s() if ta and ta.is_ready() else None)
        )
        if now is None:
            return None  # Cannot compute geometry without valid timestamp

        # Convert to NED
        foll_n, foll_e, foll_d = self.latlon_to_ned(foll_lat, foll_lon, foll_alt)
        lead_n, lead_e, lead_d = self.latlon_to_ned(lead_lat, lead_lon, lead_alt)

        # Account for boom position (boom extends behind aircraft)
        hdg_rad = math.radians(lead_heading_deg)
        boom_offset_n = -boom_length_m * math.cos(hdg_rad)
        boom_offset_e = -boom_length_m * math.sin(hdg_rad)
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

        # Use fusion if available
        source = "GPS"
        if self.fusion is not None and create_gps_measurement is not None:
            # Update GPS measurement
            gps_meas = create_gps_measurement(
                timestamp=now,
                behind_m=behind_gps,
                cross_m=cross_gps,
                below_m=below_gps,
                vel_behind_m_s=vel_behind,
                vel_cross_m_s=vel_cross,
                vel_below_m_s=vel_below,
            )
            self.fusion.update_gps(gps_meas)

            # Update UWB if available
            if uwb_range is not None and UWBMeasurement is not None:
                uwb_meas = UWBMeasurement(timestamp=now, range_m=uwb_range)
                self.fusion.update_uwb(uwb_meas)

            # Get fused state (pass sim-time for prediction/ages)
            fused = self.fusion.get_state(now_s=now)
            behind_m = fused.behind_m
            cross_m = fused.cross_m
            below_m = fused.below_m
            range_m = fused.range_m
            source = "FUSED"
        else:
            behind_m = behind_gps
            cross_m = cross_gps
            below_m = below_gps
            range_m = compute_range(behind_m, cross_m, below_m)

        # Compute closure rate
        # vel_behind = d(behind_m)/dt in leader body frame
        # When follower is behind (behind_m > 0) and CLOSING, behind_m DECREASES
        # Therefore vel_behind is NEGATIVE when closing
        # We want closure_rate POSITIVE when closing, so negate vel_behind
        closure_rate = -vel_behind

        # Compute target aspect (0 = directly behind)
        target_aspect_deg = math.degrees(math.atan2(cross_m, behind_m))

        return RelativeGeometry(
            behind_m=behind_m,
            cross_m=cross_m,
            below_m=below_m,
            range_m=range_m,
            target_aspect_deg=target_aspect_deg,
            closure_rate_ms=closure_rate,
            vel_cross_m_s=vel_cross,
            vel_below_m_s=vel_below,
            source=source,
        )


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("geometry module self-test...")

    # Test basic geometry computation
    calc = GeometryCalculator(0.0, 0.0, 0.0)
    calc.fusion = None  # Disable fusion for unit test

    # Test with follower 10m behind leader (heading 0deg = north)
    # For heading 0, "behind" means south (negative N)
    foll_n = -10.0
    foll_e = 0.0
    foll_lat = math.degrees(foll_n / calc.R_EARTH)
    foll_lon = 0.0

    geom = calc.compute_relative(
        foll_lat=foll_lat,
        foll_lon=foll_lon,
        foll_alt=0.0,
        foll_vn=0.0,
        foll_ve=0.0,
        foll_vd=0.0,
        lead_lat=0.0,
        lead_lon=0.0,
        lead_alt=0.0,
        lead_vn=0.0,
        lead_ve=0.0,
        lead_vd=0.0,
        lead_heading_deg=0.0,
        boom_length_m=0.0,
        timestamp_s=1.0,
        boom_tip_offset_m=0.0,  # No vertical offset for this test
    )

    print(f"behind_m: {geom.behind_m:.3f} (expected: 10.0)")
    print(f"cross_m: {geom.cross_m:.3f} (expected: 0.0)")
    print(f"below_m: {geom.below_m:.3f} (expected: 0.0)")
    print(f"range_m: {geom.range_m:.3f} (expected: 10.0)")

    # Test closure rate taper
    print(f"\nClosure rate at 100m: {compute_tapered_max_closure_alongtrack(100.0):.2f} m/s")
    print(f"Closure rate at 50m: {compute_tapered_max_closure_alongtrack(50.0):.2f} m/s")
    print(f"Closure rate at 20m: {compute_tapered_max_closure_alongtrack(20.0):.2f} m/s")
    print(f"Closure rate at 5m: {compute_tapered_max_closure_alongtrack(5.0):.2f} m/s")
    print(f"Closure rate at 1m: {compute_tapered_max_closure_alongtrack(1.0):.2f} m/s")

    print("\nSelf-test complete.")
