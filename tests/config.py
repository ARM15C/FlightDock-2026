#!/usr/bin/env python3
"""
FlightDock Configuration Module (Phase 1 Refactor)
===================================================

Provides all configuration parameters for the formation controller
organized into frozen dataclasses for type safety and immutability.

Configuration hierarchy:
- FormationConfig: Master config aggregating all sub-configs
  - GeometryConfig: Boom geometry parameters
  - PrecontactConfig: Precontact position targets and tolerances
  - ContactConfig: Contact position targets and tolerances
  - SpeedConfig: Speed limits and targets
  - ClosureRateConfig: Closure rate taper parameters
  - AltitudeConfig: Altitude targets and tolerances
  - CommandRateLimits: MAVLink command rate limiting
  - ControlGains: PID and control gains
  - SensorConfig: UWB/GPS sensor parameters
  - CppHandoffConfig: C++ controller handoff parameters

USAGE:
    from config import DEFAULT_CONFIG, FormationConfig

    # Use defaults
    cfg = DEFAULT_CONFIG
    print(cfg.speed.cruise_ms)

    # Create custom config
    custom_cfg = FormationConfig(
        speed=SpeedConfig(cruise_ms=26.0)
    )

(C) 2025-2026 FlightDock Research Project
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict

__all__ = [
    # Config classes
    "GeometryConfig",
    "PrecontactConfig",
    "ContactConfig",
    "SpeedConfig",
    "ClosureRateConfig",
    "AltitudeConfig",
    "CommandRateLimits",
    "ControlGains",
    "SensorConfig",
    "CppHandoffConfig",
    "InterceptConfig",
    "TransitionConfig",
    "FormationConfig",
    # Default instance
    "DEFAULT_CONFIG",
    # Legacy constant aliases
    "BOOM_TIP_OFFSET_M",
    "BOOM_STOWED_LENGTH_M",
    "BOOM_MAX_LENGTH_M",
    "PRECONTACT_ALT_MSL",
    "CONTACT_ALT_MSL",
    "HOME_ALT_MSL",
    "CRUISE_SPEED_MS",
    "SAFE_FLOOR_MS",
    "SAFE_MAX_MS",
    "LEADER_TARGET_TAS_MS",
    "PRECONTACT_HOLD_DURATION_S",
    "CONTACT_HOLD_DURATION_S",
    "STERN_CONVERSION_RANGE_M",
    "LATERAL_TOLERANCE_M",
]


# =============================================================================
# GEOMETRY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class GeometryConfig:
    """Geometry parameters for boom and aircraft."""

    boom_tip_offset_m: float = 4.31  # Boom tip 4.31m below leader CG (45-deg boom)
    boom_stowed_length_m: float = 2.4  # Stowed boom length
    boom_max_length_m: float = 6.1  # Fully extended boom length
    stern_conversion_range_m: float = 200.0  # Stern conversion target range
    lateral_tolerance_m: float = 2.0  # Lateral tolerance for altitude capture


# =============================================================================
# PRECONTACT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PrecontactConfig:
    """Precontact position targets and tolerances."""

    # Position targets (in leader body frame, relative to boom TIP)
    behind_m: float = 10.0  # Target distance behind boom tip
    cross_m: float = 0.0  # Target lateral offset (centered)
    below_m: float = 10.0  # Target altitude below boom tip

    # OUTER tolerances (for capture/stay logic)
    behind_outer_tol_m: float = 1.5  # ±1.5m behind
    cross_outer_tol_m: float = 2.0  # ±2.0m lateral
    below_outer_tol_m: float = 3.0  # ±3.0m altitude

    # INNER tolerances (for dwell accumulation)
    behind_inner_tol_m: float = 1.0  # ±1.0m - actual requirement
    cross_inner_tol_m: float = 1.5  # ±1.5m - realistic for lateral
    below_inner_tol_m: float = 2.5  # ±2.5m - altitude is noisier

    # Hold duration
    hold_duration_s: float = 5.0  # Hold at precontact for 5s

    # Capture gate
    vc_max_ms: float = 0.5  # Max closure rate to enter hold
    vc_stable_time_s: float = 1.5  # Must be stable for this long


# =============================================================================
# CONTACT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ContactConfig:
    """Contact position targets and tolerances."""

    # Position targets (at boom tip)
    behind_m: float = 0.0  # At boom tip (0m behind)
    cross_m: float = 0.0  # Centered laterally
    below_m: float = 0.0  # Altitude matched

    # OUTER tolerances (for capture/stay logic)
    behind_outer_tol_m: float = 0.5  # ±0.5m behind
    cross_outer_tol_m: float = 0.5  # ±0.5m lateral
    below_outer_tol_m: float = 1.0  # ±1.0m altitude

    # INNER tolerances (Stage 1 precision)
    behind_inner_tol_m: float = 0.20  # ±0.20m - tight behind
    cross_inner_tol_m: float = 0.10  # ±0.10m - precision lateral
    below_inner_tol_m: float = 0.40  # ±0.40m - altitude

    # Hold duration
    hold_duration_s: float = 5.0  # Hold at contact for 5s

    # Capture gate
    vc_max_ms: float = 0.20  # Max closure rate to enter contact hold
    vc_stable_time_s: float = 1.0  # Must be stable for this long
    cross_rate_max_ms: float = 0.3  # Max lateral rate for contact capture

    # Outer exit grace
    outer_exit_grace_s: float = 0.25  # Must be outside this long to exit
    outer_catastrophic_mult: float = 2.0  # Immediate exit if error > mult * tolerance


# =============================================================================
# SPEED CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SpeedConfig:
    """Speed limits and targets."""

    cruise_ms: float = 24.7  # Nominal cruise speed (m/s)
    safe_floor_ms: float = 19.0  # Minimum safe airspeed (m/s)
    safe_max_ms: float = 31.0  # Maximum safe airspeed (m/s)
    leader_target_tas_ms: float = 24.7  # Leader's target TAS (m/s)
    creep_closure_ms: float = 1.0  # Creep closure rate (m/s)

    # Slew rate limiting
    cmd_slew_rate_ms_per_s: float = 1.0  # Max speed change per second
    overtake_slew_rate_ms_per_s: float = 5.0  # Emergency slew for overtake

    # Force speed send
    force_send_delta_ms: float = 1.0  # Force send if target drops >1 m/s


# =============================================================================
# CLOSURE RATE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ClosureRateConfig:
    """Closure rate taper parameters."""

    # Range thresholds for multi-segment taper
    far_range_m: float = 150.0  # Range beyond which max closure applies
    mid_range_m: float = 50.0  # Mid-range threshold
    near_range_m: float = 20.0  # Range at which closure should be minimal
    very_near_err_m: float = 2.0  # Below this, use very-near limit
    ultra_near_err_m: float = 0.5  # Below this, use ultra-near (nearly stopped)

    # Closure rate limits by range segment
    far_ms: float = 5.0  # Max closure when far (m/s)
    mid_ms: float = 3.0  # Max closure at mid-range (50m)
    near_ms: float = 1.0  # Max closure when near (m/s)
    very_near_ms: float = 0.15  # Max closure at 2m error
    ultra_near_ms: float = 0.05  # Max closure at 0.5m error

    # Minimum closure floors (prevent stuck at target)
    min_floor_far_ms: float = 0.50  # Min closure when behind_err > 20m
    min_floor_near_ms: float = 0.20  # Min closure when 5m < behind_err < 20m
    min_floor_very_near_ms: float = 0.08  # Min closure when 1m < behind_err < 5m
    min_floor_ultra_near_ms: float = 0.0  # No forced closure when behind_err < 1m

    # EWMA smoothing
    ewma_alpha: float = 0.3  # EWMA smoothing for closure rate


# =============================================================================
# ALTITUDE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AltitudeConfig:
    """Altitude targets and tolerances."""

    precontact_msl: float = 890.0  # Precontact altitude MSL
    contact_msl: float = 900.0  # Contact altitude MSL (match leader for boom)
    home_msl: float = 584.1  # Home altitude MSL

    # Altitude capture phase
    capture_tolerance_m: float = 1.5  # Must be within ±1.5m of target
    capture_stable_time_s: float = 3.0  # Must hold altitude for 3s
    capture_vert_rate_max_ms: float = 0.3  # Vertical rate must be below this
    capture_standoff_m: float = 50.0  # Distance behind during climb
    capture_timeout_s: float = 90.0  # Timeout if altitude capture fails

    # Command forcing
    cmd_delta_m: float = 0.5  # Force resend if alt_cmd changes by > 0.5m
    err_force_m: float = 5.0  # Force resend if |fol_alt - alt_cmd| > 5m
    err_force_time_s: float = 2.0  # For more than 2 seconds
    err_warning_m: float = 10.0  # Warn if |fol_alt - alt_cmd| > 10m
    err_warning_time_s: float = 5.0  # For more than 5 seconds
    fallback_interval_s: float = 2.5  # Send fallback altitude cmd every 2.5s

    # Vertical gating
    gate_threshold_m: float = 6.0  # Gate if |below_err| > 6m
    gate_time_s: float = 10.0  # For more than 10s
    gate_range_m: float = 150.0  # Only gate if range < 150m
    gate_standoff_m: float = 50.0  # Add 50m to behind target when gated


# =============================================================================
# COMMAND RATE LIMITS
# =============================================================================


@dataclass(frozen=True)
class CommandRateLimits:
    """MAVLink command rate limiting parameters."""

    # DO_REPOSITION rate limiting
    reposition_min_interval_s: float = 0.2  # Min interval between reposition commands
    heading_delta_deg: float = 2.0  # Send if heading changes by more than this

    # DO_CHANGE_SPEED rate limiting
    speed_min_interval_s: float = 0.5  # Min interval between speed commands
    speed_delta_threshold: float = 0.3  # Send if speed changes by more than this

    # Very close range (<30m) - fine control
    very_close_reposition_interval_s: float = 0.12  # Fast heading updates
    very_close_heading_delta_deg: float = 0.4  # Small heading change threshold
    very_close_speed_interval_s: float = 0.20  # Fast speed updates
    very_close_speed_delta_ms: float = 0.08  # Tiny speed change threshold

    # Close range (30-50m) - transition
    close_reposition_interval_s: float = 0.20  # Moderate heading updates
    close_heading_delta_deg: float = 0.8  # Moderate heading threshold
    close_speed_interval_s: float = 0.30  # Moderate speed updates
    close_speed_delta_ms: float = 0.15  # Small speed threshold


# =============================================================================
# CONTROL GAINS
# =============================================================================


@dataclass(frozen=True)
class ControlGains:
    """PID and control gains."""

    # Speed control: Range-scaled Kp
    speed_kp_far: float = 0.08  # Proportional gain at far range
    speed_kp_near: float = 0.20  # Proportional gain at close range
    speed_kd: float = 0.3  # Derivative gain (damping)

    # Lateral control
    cross_correction_gain: float = 0.175  # Lateral P-gain
    cross_d_gain: float = 0.5  # Lateral D-gain at far range
    cross_d_gain_near: float = 0.8  # Lateral D-gain at close range

    # HOLD_CONTACT PI controller
    hold_contact_kp: float = 0.30  # Proportional gain on behind_error
    hold_contact_ki: float = 0.05  # Integral gain (slow wind-up)
    hold_contact_kd: float = 0.20  # Derivative gain (uses closure_rate)
    hold_contact_i_clamp_ms: float = 0.4  # Max integrator contribution (m/s)

    # Cross-track PI heading bias
    cross_bias_enable_range_m: float = 30.0  # Start PI below this range
    cross_bias_enable_max_cross_m: float = 2.0  # Only integrate if cross_err below this
    cross_bias_enable_near_contact_m: float = 5.0  # Extra gating: behind_err < this
    cross_bias_kp: float = 0.02  # Proportional gain (deg/m)
    cross_bias_ki: float = 0.08  # Integral gain (deg/m·s)
    cross_bias_i_clamp_deg: float = 2.0  # Max integrator heading bias (±deg)
    cross_bias_i_decay: float = 0.995  # Decay when not integrating

    # Heading slew rates by range
    heading_slew_very_close_dps: float = 1.0  # Max heading rate inside 30m
    heading_slew_close_dps: float = 2.5  # Max heading rate 30-50m
    heading_slew_medium_dps: float = 5.0  # Max heading rate 50-100m
    heading_slew_far_dps: float = 25.0  # Max heading rate >100m

    # Hold station-keeping
    hold_station_keeping_gain: float = 0.12  # Speed correction per m behind error


# =============================================================================
# SENSOR CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SensorConfig:
    """UWB/GPS sensor parameters."""

    # UWB validity criteria
    uwb_stale_threshold_s: float = 0.5  # Max age for UWB data
    uwb_range_min_m: float = 0.5  # Minimum valid UWB range
    uwb_range_max_m: float = 150.0  # Maximum valid UWB range
    uwb_jump_threshold_m: float = 10.0  # Max range jump between samples
    uwb_nan_is_invalid: bool = True  # Treat NaN as invalid

    # UWB-dominant blending weights
    uwb_dominant_weight_far: float = 0.5  # UWB weight at close_range_threshold
    uwb_dominant_weight_near: float = 0.9  # UWB weight at very_precision threshold

    # Cross-track filtering
    cross_track_filter_alpha: float = 0.3  # EMA filter coefficient
    cross_rate_filter_alpha: float = 0.2  # EMA filter for cross rate

    # Precision mode thresholds
    precision_mode_threshold_m: float = 30.0  # Enter precision mode below this
    very_precision_mode_threshold_m: float = 15.0  # Extra-fine control below this

    # Sensor fallback logging
    fallback_log_interval_s: float = 5.0  # Min seconds between fallback warnings


# =============================================================================
# C++ HANDOFF CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CppHandoffConfig:
    """C++ controller handoff parameters."""

    enable_range_m: float = 30.0  # Hand off to C++ below this range
    disable_range_m: float = 40.0  # Return to Python above this (hysteresis)
    enabled: bool = True  # Master switch for C++ handoff feature
    warmup_s: float = 2.0  # Sim-time dwell after PASS before engaging


# =============================================================================
# INTERCEPT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class InterceptConfig:
    """Fighter intercept geometry parameters."""

    lateral_offset_m: float = 800.0  # Lateral offset during head-on approach
    lead_turn_ta_deg: float = 150.0  # Target aspect for lead turn
    lead_turn_speed_ms: float = 30.0  # Airspeed during lead turn
    offset_side: float = 1.0  # +1 = right, -1 = left
    lead_turn_min_range_m: float = 2000.0  # Min range before lead turn
    stern_aspect_tolerance_deg: float = 15.0  # Acceptable aspect angle for stern

    # Dynamic offset
    dynamic_offset_enable: bool = True  # Enable dynamic offset side selection
    dynamic_offset_threshold_m: float = 500.0  # Only flip if |cross_m| > this


# =============================================================================
# TRANSITION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TransitionConfig:
    """Fixed-wing transition parameters."""

    min_airspeed_mps: float = 12.5  # Achievable at high speedup
    sustain_s: float = 3.0  # Sustain airspeed for this long
    timeout_s: float = 60.0  # Timeout per attempt
    retries: int = 3  # Number of retry attempts
    stabilize_time_s: float = 5.0  # Time to stabilize after FW confirmed


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class FormationConfig:
    """Master configuration aggregating all sub-configs."""

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    precontact: PrecontactConfig = field(default_factory=PrecontactConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    closure_rate: ClosureRateConfig = field(default_factory=ClosureRateConfig)
    altitude: AltitudeConfig = field(default_factory=AltitudeConfig)
    command_limits: CommandRateLimits = field(default_factory=CommandRateLimits)
    gains: ControlGains = field(default_factory=ControlGains)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    cpp_handoff: CppHandoffConfig = field(default_factory=CppHandoffConfig)
    intercept: InterceptConfig = field(default_factory=InterceptConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        import dataclasses

        return dataclasses.asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================

DEFAULT_CONFIG = FormationConfig()


# =============================================================================
# LEGACY CONSTANT ALIASES (for backward compatibility)
# =============================================================================

# These aliases allow existing code to continue using UPPERCASE constants
# while the new code can use the config dataclasses

BOOM_TIP_OFFSET_M = DEFAULT_CONFIG.geometry.boom_tip_offset_m
BOOM_STOWED_LENGTH_M = DEFAULT_CONFIG.geometry.boom_stowed_length_m
BOOM_MAX_LENGTH_M = DEFAULT_CONFIG.geometry.boom_max_length_m
PRECONTACT_ALT_MSL = DEFAULT_CONFIG.altitude.precontact_msl
CONTACT_ALT_MSL = DEFAULT_CONFIG.altitude.contact_msl
HOME_ALT_MSL = DEFAULT_CONFIG.altitude.home_msl
CRUISE_SPEED_MS = DEFAULT_CONFIG.speed.cruise_ms
SAFE_FLOOR_MS = DEFAULT_CONFIG.speed.safe_floor_ms
SAFE_MAX_MS = DEFAULT_CONFIG.speed.safe_max_ms
LEADER_TARGET_TAS_MS = DEFAULT_CONFIG.speed.leader_target_tas_ms
PRECONTACT_HOLD_DURATION_S = DEFAULT_CONFIG.precontact.hold_duration_s
CONTACT_HOLD_DURATION_S = DEFAULT_CONFIG.contact.hold_duration_s
STERN_CONVERSION_RANGE_M = DEFAULT_CONFIG.geometry.stern_conversion_range_m
LATERAL_TOLERANCE_M = DEFAULT_CONFIG.geometry.lateral_tolerance_m


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("config module self-test...")

    # Test default config
    cfg = DEFAULT_CONFIG
    print(f"cruise_ms: {cfg.speed.cruise_ms}")
    print(f"boom_tip_offset_m: {cfg.geometry.boom_tip_offset_m}")
    print(f"precontact_behind_m: {cfg.precontact.behind_m}")

    # Test JSON serialization
    json_str = cfg.to_json()
    print(f"\nJSON config (truncated):\n{json_str[:500]}...")

    # Test legacy aliases
    print(f"\nLegacy aliases:")
    print(f"CRUISE_SPEED_MS: {CRUISE_SPEED_MS}")
    print(f"BOOM_TIP_OFFSET_M: {BOOM_TIP_OFFSET_M}")

    # Test frozen (should raise error)
    try:
        cfg.speed.cruise_ms = 30.0  # type: ignore
        print("ERROR: Config should be frozen!")
    except Exception:
        print("Config is correctly frozen (immutable)")

    print("\nSelf-test complete.")
