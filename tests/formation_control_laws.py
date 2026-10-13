import math
import os
from dataclasses import dataclass
from typing import Optional

import formation_config as config
from geometry import RelativeGeometry, wrap_heading, wrap_heading_error
from formation_logging import log
from state_machine import ControllerState


@dataclass
class SpeedArbitrationResult:
    """Per-tick speed arbitration breakdown for debugging.

    P2 v9.4: Explicit pipeline with per-stage values for post-run analysis.
    P5.3 v9.5: Added closure_cap tracking fields.
    P6.1 v9.6: Added target_range_m and behind_error_m for target-relative gating.
    M3 v9.10: Added binding_constraint sub-source telemetry (S1).
    """

    spd_req_profile_mps: float  # Closure schedule output BEFORE any clamps
    spd_after_brake_mps: float  # After braking clamp (P5.3: now closure-cap based)
    spd_after_climb_mps: float  # After climb clamp (ALT_CAP_* states only)
    spd_after_safety_mps: float  # After safety/no-pass clamp
    spd_cmd_final_mps: float  # Final command sent
    brake_clamp_active: bool  # P5.3: Now indicates closure-cap active
    climb_clamp_active: bool
    safety_clamp_active: bool
    dominant_source: str  # "profile" | "brake" | "climb" | "safety"
    deficit_i: float  # Closure deficit integrator state
    # P5.3 v9.5: Closure cap tracking
    closure_cap_mps: float = 0.0  # Current closure cap (0 if not active)
    closure_cap_active: bool = False  # True if closure cap is limiting speed
    # P6.1 v9.6: Target-relative context
    target_range_m: float = 0.0  # Target behind distance for this tick
    behind_error_m: float = 0.0  # behind_m - target_range_m (+ = still behind target)
    # M3 v9.10 S1: Binding constraint sub-source telemetry
    binding_constraint_pre_override: str = (
        "NONE"  # Which cap bound the safety stack (before FD-004)
    )
    binding_constraint_final: str = "NONE"  # What determined the sent speed (FD-004 if active)
    binding_value_mps: float = 0.0  # The cap value that bound final command
    binding_margin_m: float = 0.0  # Context margin (e.g., behind_pred - margin)


def ewma_alpha_scaled(base_alpha: float, dt: float, nominal_dt: float = 0.05) -> float:
    """Scale EWMA alpha to maintain consistent time constant across variable dt.

    Keeps the effective time constant tau fixed regardless of actual dt.
    tau = -nominal_dt / ln(1 - base_alpha)
    alpha_scaled = 1 - exp(-dt / tau)

    Args:
        base_alpha: The EWMA alpha tuned for nominal_dt (0 < alpha < 1)
        dt: Actual time step in seconds
        nominal_dt: The nominal time step base_alpha was tuned for (default 0.05s = 20Hz)

    Returns:
        Scaled alpha that preserves the same effective time constant
    """
    if dt <= 0 or base_alpha <= 0 or base_alpha >= 1:
        return base_alpha
    tau = -nominal_dt / math.log(1 - base_alpha)
    return 1.0 - math.exp(-dt / tau)


def compute_dynamic_below_target(behind_m: float) -> float:
    """P7.2 v9.7: Ramp below_target from PRECONTACT to CONTACT.

    Linear interpolation:
    - behind_m >= VERT_RAMP_START_BEHIND_M: return PRECONTACT_BELOW_M
    - behind_m <= VERT_RAMP_END_BEHIND_M: return CONTACT_BELOW_M
    - Between: linear ramp

    Args:
        behind_m: Current distance behind boom tip (positive = behind).

    Returns:
        Target below_m in meters (0 at contact, 10 at precontact).
    """
    if behind_m >= config.VERT_RAMP_START_BEHIND_M:
        return config.PRECONTACT_BELOW_M
    if behind_m <= config.VERT_RAMP_END_BEHIND_M:
        return config.CONTACT_BELOW_M
    # Linear ramp
    progress = (config.VERT_RAMP_START_BEHIND_M - behind_m) / (
        config.VERT_RAMP_START_BEHIND_M - config.VERT_RAMP_END_BEHIND_M
    )
    return config.PRECONTACT_BELOW_M * (1.0 - progress) + config.CONTACT_BELOW_M * progress


def compute_vertical_closure_cap(below_err_m: float, base_vc_cap: float) -> float:
    """P7.2 v9.7: Hold range when vertical error huge, proportional gate otherwise.

    NOT a floor - when error is huge, closure is ZERO.

    Args:
        below_err_m: Vertical error (below_m - below_target).
        base_vc_cap: Base closure cap (from closure schedule or final approach).

    Returns:
        Gated closure cap in m/s.
    """
    abs_err = abs(below_err_m)

    if abs_err > config.VERT_ERR_HOLD_RANGE_M:
        # Error too large: hold range, focus on vertical
        return 0.0

    # Proportional gate within corridor
    gate = 1.0 - (abs_err / config.VERT_ERR_HOLD_RANGE_M)
    gate = max(0.0, min(1.0, gate))
    return base_vc_cap * gate


def proximity_factor(behind_m: float) -> float:
    """P9.0 v9.9: Compute proximity factor for position-based gating.

    Returns 0.0 at 10m+ behind, 1.0 at 2m or closer.
    Used to tighten cross-track tolerance as we get closer.

    Args:
        behind_m: Current distance behind boom tip (positive = behind).

    Returns:
        Factor from 0.0 (far) to 1.0 (close).
    """
    return max(0.0, min(1.0, (10.0 - behind_m) / 8.0))


def compute_cross_track_gate(cross_m: float, behind_m: float) -> float:
    """P9.0 v9.9: Gate closure based on cross-track alignment.

    When close to contact, require tighter cross-track alignment.
    Returns a gate factor from 0.0 (hold range) to 1.0 (no constraint).

    Args:
        cross_m: Cross-track offset in meters.
        behind_m: Behind distance in meters.

    Returns:
        Gate factor (0.0-1.0) to multiply closure cap.
    """
    # Cross-track tolerance tightens as we get closer
    # At 10m+: allow 2.0m cross
    # At 2m: allow 0.3m cross
    prox = proximity_factor(behind_m)
    cross_allow = 2.0 - prox * 1.7  # Lerp from 2.0 to 0.3

    abs_cross = abs(cross_m)
    if abs_cross <= cross_allow:
        return 1.0  # No constraint

    # Proportional gate: reduce closure as cross error increases beyond threshold
    gate = max(0.0, 1.0 - (abs_cross - cross_allow) / 1.0)
    return gate


def compute_final_approach_vc_cap(behind_m: float, return_bin_idx: bool = False) -> float:
    """P7.1 v9.7: Interpolate closure cap from final approach schedule.

    Uses FINAL_APPROACH_SCHEDULE as a piecewise linear schedule.
    Above the highest breakpoint, returns that breakpoint's vc.
    Below the lowest breakpoint, returns the lowest breakpoint's vc.

    Args:
        behind_m: Current distance behind boom tip (positive = behind).
        return_bin_idx: M3 S1 - if True, return (vc_cap, bin_idx) tuple.

    Returns:
        Maximum closure rate in m/s for this behind distance.
        If return_bin_idx=True, returns (vc_cap, bin_idx) where bin_idx
        is the schedule segment index used for interpolation (-1 = above/below).
    """
    schedule = config.FINAL_APPROACH_SCHEDULE
    if not schedule:
        return (1.0, -1) if return_bin_idx else 1.0

    # Above highest breakpoint
    if behind_m >= schedule[0][0]:
        vc = schedule[0][1]
        return (vc, -1) if return_bin_idx else vc

    # Below lowest breakpoint
    if behind_m <= schedule[-1][0]:
        vc = schedule[-1][1]
        return (vc, len(schedule) - 1) if return_bin_idx else vc

    # Interpolate between breakpoints
    for i in range(len(schedule) - 1):
        d_hi, vc_hi = schedule[i]
        d_lo, vc_lo = schedule[i + 1]
        if d_lo <= behind_m < d_hi:
            t = (behind_m - d_lo) / (d_hi - d_lo)
            vc = vc_lo + t * (vc_hi - vc_lo)
            return (vc, i) if return_bin_idx else vc

    vc = schedule[-1][1]  # Fallback
    return (vc, len(schedule) - 1) if return_bin_idx else vc


class FormationControlLaws:
    """
    Encapsulates the core control laws for formation flight, including heading,
    speed, and altitude command computation.
    """

    # States where docking-authority braking constant applies
    _DOCKING_STATES = frozenset(
        {
            ControllerState.HOLD_PRECONTACT,
            ControllerState.ALT_CAP_HOLD_RANGE,
            ControllerState.DOCKING_CREEP,
            ControllerState.HOLD_CONTACT,
        }
    )

    def __init__(self, sim_now_s_fn, speedup_factor: float = 1.0) -> None:
        """Initialize formation control laws.

        Args:
            sim_now_s_fn: Callable returning current sim-time in seconds.
            speedup_factor: SIM_SPEEDUP factor for predictive ramp scaling.
        """
        self.sim_now_s_fn = sim_now_s_fn
        self.speedup_factor = max(1.0, speedup_factor)
        self._last_formation_hdg_cmd: Optional[float] = None
        self._last_formation_hdg_time: Optional[float] = None
        self._speed_cmd_slewed: float = config.CRUISE_SPEED_MS
        self._speed_cmd_slew_time: float = 0.0
        self._last_speed_tick: float = 0.0
        # v9.2 Patch D: Anti-stall closure deficit integrator
        self._closure_deficit_i: float = 0.0
        self._last_closure_sched_log: float = 0.0
        self._stuck_integral_boost: float = 0.0
        self._stuck_start: Optional[float] = None
        self._stuck_active: bool = False
        self._last_stuck_recovery_log: float = 0.0
        self._last_floor_log: float = 0.0
        self._last_leader_speed_switch_log: float = 0.0

        self._last_overtake_diag_time: float = 0.0
        self._last_speed_diag_log: float = 0.0
        self._behind_m_prev = None

        # State-gated braking ramp (smooth transition, tau=0.5s)
        self._braking_a_current: float = config.CLOSURE_BRAKING_A_MAX_MS2
        self._braking_a_target: float = config.CLOSURE_BRAKING_A_MAX_MS2
        self._braking_a_effective: float = config.CLOSURE_BRAKING_A_MAX_MS2
        self._behind_m_prev_time = 0.0
        self._closure_rate_behind_ewma = 0.0
        # P5.3 v9.5: Closure cap state
        self._closure_cap_active: bool = False
        self._closure_cap_mps: float = 0.0
        # P2 v9.4: Speed arbitration result tracking
        self._last_arbitration: Optional[SpeedArbitrationResult] = None
        self._arb_total_ticks: int = 0
        self._arb_dominant_counts: dict = {"profile": 0, "brake": 0, "climb": 0, "safety": 0}

    def get_vc_behind_ewma_mps(self) -> float:
        """Return current behind-axis closure rate EWMA (m/s).

        P5.1 v9.5: Getter for the authoritative closure rate EWMA.
        Positive = closing (behind_m decreasing).
        """
        return self._closure_rate_behind_ewma

    def compute_heading_cmd(self, geom: RelativeGeometry, lead_hdg: float) -> float:
        """Computes heading for intercept phase."""
        relative_bearing = math.degrees(math.atan2(-geom.cross_m, geom.behind_m))
        bearing_to_leader = wrap_heading(lead_hdg + relative_bearing)

        p_term = -config.CROSS_CORRECTION_GAIN * geom.cross_m

        if geom.range_m <= config.VERY_CLOSE_RANGE_M:
            d_gain = config.CROSS_D_GAIN_NEAR
        elif geom.range_m >= config.FINE_CONTROL_RANGE_M:
            d_gain = config.CROSS_D_GAIN
        else:
            t = (geom.range_m - config.VERY_CLOSE_RANGE_M) / (
                config.FINE_CONTROL_RANGE_M - config.VERY_CLOSE_RANGE_M
            )
            d_gain = config.CROSS_D_GAIN_NEAR + t * (
                config.CROSS_D_GAIN - config.CROSS_D_GAIN_NEAR
            )

        d_term = -d_gain * geom.vel_cross_m_s
        cross_correction = p_term + d_term

        if geom.range_m <= config.VERY_CLOSE_RANGE_M:
            max_rate = config.VERY_CLOSE_HEADING_RATE_DPS
        elif geom.range_m <= config.FINE_CONTROL_RANGE_M:
            t = (geom.range_m - config.VERY_CLOSE_RANGE_M) / (
                config.FINE_CONTROL_RANGE_M - config.VERY_CLOSE_RANGE_M
            )
            max_rate = config.VERY_CLOSE_HEADING_RATE_DPS + t * (
                config.FINE_CONTROL_HEADING_RATE_DPS - config.VERY_CLOSE_HEADING_RATE_DPS
            )
        elif geom.range_m < config.CLOSE_RANGE_THRESHOLD_M:
            max_rate = config.CLOSE_RANGE_HEADING_RATE_DPS
        else:
            max_rate = config.MAX_HEADING_RATE_DPS

        cross_correction = max(-max_rate, min(max_rate, cross_correction))
        return wrap_heading(bearing_to_leader + cross_correction)

    def compute_formation_heading_cmd(
        self, geom: RelativeGeometry, lead_hdg: float, extra_bias_deg: float = 0.0
    ) -> float:
        """Computes heading for close-formation phases."""
        if self._last_formation_hdg_cmd is None:
            self._last_formation_hdg_cmd = lead_hdg
            self._last_formation_hdg_time = self.sim_now_s_fn()

        p_term = -config.CROSS_CORRECTION_GAIN * geom.cross_m

        if geom.range_m <= config.VERY_CLOSE_RANGE_M:
            d_gain = config.CROSS_D_GAIN_NEAR
        elif geom.range_m >= config.FINE_CONTROL_RANGE_M:
            d_gain = config.CROSS_D_GAIN
        else:
            t = (geom.range_m - config.VERY_CLOSE_RANGE_M) / (
                config.FINE_CONTROL_RANGE_M - config.VERY_CLOSE_RANGE_M
            )
            d_gain = config.CROSS_D_GAIN_NEAR + t * (
                config.CROSS_D_GAIN - config.CROSS_D_GAIN_NEAR
            )

        d_term = -d_gain * geom.vel_cross_m_s
        cross_correction_deg = p_term + d_term
        hdg_desired = wrap_heading(lead_hdg + cross_correction_deg + extra_bias_deg)

        if geom.range_m <= config.VERY_CLOSE_RANGE_M:
            max_rate_dps = config.HEADING_SLEW_VERY_CLOSE_DPS
        elif geom.range_m <= config.FINE_CONTROL_RANGE_M:
            t = (geom.range_m - config.VERY_CLOSE_RANGE_M) / (
                config.FINE_CONTROL_RANGE_M - config.VERY_CLOSE_RANGE_M
            )
            max_rate_dps = config.HEADING_SLEW_VERY_CLOSE_DPS + t * (
                config.HEADING_SLEW_CLOSE_DPS - config.HEADING_SLEW_VERY_CLOSE_DPS
            )
        elif geom.range_m < config.CLOSE_RANGE_THRESHOLD_M:
            max_rate_dps = config.HEADING_SLEW_MEDIUM_DPS
        else:
            max_rate_dps = config.HEADING_SLEW_FAR_DPS

        now = self.sim_now_s_fn()
        dt = now - self._last_formation_hdg_time
        if dt <= 0:
            dt = 0.05

        max_step_deg = max_rate_dps * dt
        delta = wrap_heading_error(hdg_desired - self._last_formation_hdg_cmd)
        delta = max(-max_step_deg, min(max_step_deg, delta))
        hdg_cmd = wrap_heading(self._last_formation_hdg_cmd + delta)

        self._last_formation_hdg_cmd = hdg_cmd
        self._last_formation_hdg_time = now
        return hdg_cmd

    def _compute_closure_cap(self, behind_m: float) -> Optional[float]:
        """Compute max allowed closure rate based on distance.

        P5.3 v9.5: Closure-cap approach replaces the old reduction-based braking.
        Instead of commanding below-leader speeds (which causes fallback), we cap
        the closure rate and convert to a speed cap that's still >= leader speed
        while behind the contact threshold.

        Args:
            behind_m: Current distance behind boom tip (positive = behind)

        Returns:
            Max closure rate in m/s if inside closure-cap zone, None otherwise.
        """
        if behind_m >= config.CLOSURE_CAP_ENABLE_RANGE_M:
            return None

        schedule = config.CLOSURE_CAP_SCHEDULE
        # Linear interpolation between breakpoints
        for i in range(len(schedule) - 1):
            d_hi, vc_hi = schedule[i]
            d_lo, vc_lo = schedule[i + 1]
            if d_lo <= behind_m < d_hi:
                t = (behind_m - d_lo) / (d_hi - d_lo)
                return vc_lo + t * (vc_hi - vc_lo)

        # Below lowest breakpoint (at or past contact)
        return schedule[-1][1]

    def _compute_desired_closure(
        self, behind_err_m: float, braking_a: Optional[float] = None
    ) -> float:
        """Compute desired closure rate from piecewise schedule + braking clamp.

        v9.2 Patch C: Range-dependent closure schedule that ramps down as
        behind error decreases, with a physics-based braking clamp to
        prevent overshoot. Only active below CLOSURE_SCHEDULE_ENABLE_RANGE_M.

        Args:
            behind_err_m: Along-track error (positive = behind target).
            braking_a: Effective braking deceleration (m/s²). If None, uses
                       the current ramped/speedup-compensated value.

        Returns:
            Desired closure rate in m/s (non-negative).
        """
        if braking_a is None:
            braking_a = self._braking_a_effective
        bp = config.CLOSURE_SCHEDULE_BREAKPOINTS
        if behind_err_m >= bp[0][0]:
            profile = bp[0][1]
        elif behind_err_m <= bp[-1][0]:
            profile = config.CLOSURE_SCHEDULE_FINAL_MS
        else:
            profile = config.CLOSURE_SCHEDULE_FINAL_MS
            for i in range(len(bp) - 1):
                d_hi, v_hi = bp[i]
                d_lo, v_lo = bp[i + 1]
                if d_lo <= behind_err_m <= d_hi:
                    t = (behind_err_m - d_lo) / (d_hi - d_lo)
                    profile = v_lo + t * (v_hi - v_lo)
                    break

        # Braking clamp: v_safe = sqrt(2 * a_max * d)
        d = max(behind_err_m, 0.001)
        v_safe = math.sqrt(2.0 * braking_a * d)
        return min(profile, v_safe)

    def get_braking_a_effective(self) -> float:
        """Return the current effective braking deceleration for CSV telemetry."""
        return self._braking_a_effective

    def compute_speed_cmd(
        self,
        geom: RelativeGeometry,
        lead_gs: float,
        target_range_m: float,
        follower_alt_msl: float,
        enable_overtake_protect: bool = True,
        station_keeping: bool = False,
        climb_clamp_cap: Optional[float] = None,
        safety_clamp_cap: Optional[float] = None,
        climb_clamp_fast_decel: bool = False,
        controller_state=None,
    ) -> float:
        """Compute speed command with explicit arbitration pipeline.

        P2 v9.4: Pipeline order:
        1. spd_profile ← PD control + closure schedule
        2. spd_after_brake = min(spd_profile, braking_cap) # Physics-based
        3. spd_after_climb = min(spd_after_brake, climb_cap) # If provided
        4. spd_after_safety = min(spd_after_climb, safety_cap) # No-pass guard
        5. spd_final = slew_limit(spd_after_safety)

        Args:
            geom: Current geometry
            lead_gs: Leader ground speed
            target_range_m: Target behind distance
            follower_alt_msl: Follower altitude MSL
            enable_overtake_protect: Enable adaptive braking
            station_keeping: Use station-keeping gains
            climb_clamp_cap: Optional climb clamp speed cap (from controller)
            safety_clamp_cap: Optional safety/no-pass speed cap (from controller)

        Returns:
            Final speed command (m/s)

        Note: Arbitration result stored in self._last_arbitration for logging.
        """
        now = self.sim_now_s_fn()

        # Compute dt early so it's available for all integrators/filters
        dt_tick = now - self._last_speed_tick if self._last_speed_tick > 0 else 0.05
        dt_tick = max(0.0, min(dt_tick, 0.25))
        self._last_speed_tick = now

        # P7.4 v9.7: dt spike detection and integrator freeze
        # Prevents dt spikes from destabilizing integrators at high speedup.
        # Safety down-slew uses full dt_tick (must bind immediately).
        is_dt_spike = dt_tick >= config.DT_SPIKE_THRESHOLD_S
        if is_dt_spike and config.DT_SPIKE_INTEGRATOR_FREEZE:
            dt_integrator = 0.0  # Freeze integrators on spike
        else:
            dt_integrator = dt_tick

        # State-gated braking: select target based on controller state
        if controller_state in self._DOCKING_STATES:
            self._braking_a_target = config.CLOSURE_BRAKING_A_MAX_DOCKING_MS2
        else:
            self._braking_a_target = config.CLOSURE_BRAKING_A_MAX_MS2

        # #336 subtractive cross-gate decouple + behind deadband: active ONLY in
        # DOCKING_CREEP / HOLD_CONTACT (flag-gated, default OFF). D1 (deadband)
        # applies in the PD branch below; D2 (floor) applies at the Stage-3 climb
        # clamp. Byte-identical when OFF (flag False => _spd_cross_decouple = False).
        _spd_cross_decouple = (
            config.SPD_CROSS_DECOUPLE_ENABLE
            and controller_state in (ControllerState.DOCKING_CREEP,
                                     ControllerState.HOLD_CONTACT))
        if not hasattr(self, "_last_spd_deadband_log"):
            self._last_spd_deadband_log = 0.0
            self._last_cross_decouple_log = 0.0
        if not hasattr(self, "_last_park_bias_log"):  # #340 F1
            self._last_park_bias_log = 0.0

        # Exponential ramp (tau=0.5s) to prevent command discontinuity
        if dt_tick > 0.0:
            alpha = 1.0 - math.exp(-dt_tick / 0.5)
            self._braking_a_current += alpha * (self._braking_a_target - self._braking_a_current)

        # Braking authority: TECS in SITL runs at sim-rate, so deceleration
        # authority is speedup-invariant. Speedup compensation is applied only
        # to the predictive ramp range (line 518), not to the braking constant.
        self._braking_a_effective = self._braking_a_current

        # 1. Internal Range Vc Calculation (D-term)
        if not hasattr(self, "_last_range_val"):
            self._last_range_val = geom.range_m
            self._last_range_time = now
            self._range_vc_ewma = 0.0

        dt_range = now - self._last_range_time
        if dt_range > 0.001:
            raw_rate = -(geom.range_m - self._last_range_val) / dt_range
            alpha = ewma_alpha_scaled(0.1, dt_range, nominal_dt=0.05)
            self._range_vc_ewma = (1.0 - alpha) * self._range_vc_ewma + alpha * raw_rate
            self._last_range_val = geom.range_m
            self._last_range_time = now

        actual_vc = self._range_vc_ewma

        # 2. Use target speed as baseline (ignore estimator for simplicity in helper)
        leader_ref_tas = config.LEADER_TARGET_TAS_MS

        # 3. Along-track error
        behind_error = geom.behind_m - target_range_m

        # 4. Closure rate limit (schedule for close range, taper for far)
        # v9.2 Patch C: Use piecewise schedule below threshold
        use_schedule = geom.range_m < config.CLOSURE_SCHEDULE_ENABLE_RANGE_M and behind_error > 0
        if use_schedule:
            desired_closure = self._compute_desired_closure(behind_error)
            max_closure = max(desired_closure, 0.05)
        else:
            max_closure = config.MAX_CLOSURE_RATE_FAR_MS
            if behind_error < 20.0:
                max_closure = config.MAX_CLOSURE_RATE_NEAR_MS
        # RC-4 fix: Predictive deceleration ramp (replaces reactive *0.5)
        # Smoothly tapers max_closure from 100% at ramp_start to 20% at/below 0m.
        # Step 2C: Speedup-scaled ramp range — at higher speedup with fast closure,
        # start braking earlier to compensate for TECS wall-clock lag.
        _closure_for_ramp = (
            abs(self._closure_rate_behind_ewma)
            if hasattr(self, "_closure_rate_behind_ewma")
            else 0.0
        )
        ramp_start_m = max(5.0, _closure_for_ramp * 3.0 * self.speedup_factor)
        if behind_error > 0 and behind_error < ramp_start_m:
            decel_factor = max(0.2, min(1.0, behind_error / ramp_start_m))
            max_closure = max_closure * decel_factor

        # 5. Closure-Cap Braking (P5.3 v9.5)
        # Replaces old reduction-based braking that caused 8-9m stall
        # P6.1 v9.6: Gate closure-cap on target-relative position
        # - Only active when behind_error > 0 (still behind target)
        # - Not active when safety is already clamping (avoid conflicting commands)
        closure_rate_behind = self._closure_rate_behind_ewma
        safety_active = safety_clamp_cap is not None

        # P6.1 v9.6: Three conditions for closure-cap activation:
        # 1. Inside range threshold (behind_m < CLOSURE_CAP_ENABLE_RANGE_M)
        # 2. Still behind target (behind_error > 0)
        # 3. Safety not already active (prevents command conflicts)
        vc_cap_raw = self._compute_closure_cap(geom.behind_m)
        closure_cap_active = vc_cap_raw is not None and behind_error > 0.0 and not safety_active
        vc_cap = vc_cap_raw if closure_cap_active else None

        self._closure_cap_active = closure_cap_active
        self._closure_cap_mps = vc_cap if vc_cap else 0.0
        self._adaptive_brake_active = False  # Legacy flag, kept for compatibility

        raw_speed = None
        if enable_overtake_protect and closure_cap_active:
            # P5.3: Convert closure cap to speed cap
            # Key insight: speed = leader_gs + allowed_closure when behind > threshold
            # This ensures we stay >= leader speed while behind, preventing backward drift
            if geom.behind_m > config.CLOSURE_CAP_BELOW_LEADER_THRESHOLD_M:
                # Still behind: speed cap allows closure up to vc_cap
                speed_cap_from_closure = lead_gs + vc_cap
            else:
                # Very close / at contact: allow below-leader for station keeping
                speed_cap_from_closure = lead_gs + vc_cap * 0.5

            # If current closure exceeds cap, need more aggressive braking
            if closure_rate_behind > vc_cap:
                vc_excess = closure_rate_behind - vc_cap
                # Proportional reduction based on excess closure
                brake_speed = lead_gs + vc_cap - vc_excess * config.CLOSURE_CAP_BRAKE_GAIN
                speed_cap_from_closure = min(speed_cap_from_closure, brake_speed)

            raw_speed = speed_cap_from_closure
            raw_speed = max(config.SAFE_FLOOR_MS, raw_speed)
            self._adaptive_brake_active = True  # Set for slew rate selection

        if raw_speed is None:
            # PD Control
            kp = config.SPEED_KP_NEAR if geom.range_m < 30.0 else config.SPEED_KP_FAR
            if station_keeping:
                kp = config.HOLD_STATION_KEEPING_GAIN

            # #336 D1: subtractive along-track deadband (DC/HC only, flag-gated).
            # Zero the P-term inside +/-DB so the D-term (-Kd*actual_vc) alone
            # nulls the closing rate => converge to exactly leader speed at
            # contact instead of dithering across the target.
            behind_error_p = behind_error
            if _spd_cross_decouple:
                # #340 F1: park-bias shifts the P-path null FORWARD by BIAS m
                # (perceived park at -BIAS) so truth lands in-box under the
                # #338 contact-window truth-aft offset. P-term path ONLY.
                _be = behind_error
                if config.SPD_PARK_BIAS_ENABLE:
                    _be = behind_error + config.SPD_PARK_BIAS_M
                    if (now - self._last_park_bias_log) >= 2.0:
                        self._last_park_bias_log = now
                        log(f"[PARK_BIAS] behind_err={behind_error:.3f} "
                            f"biased={_be:.3f} (+{config.SPD_PARK_BIAS_M:.2f} fwd)")
                _db = config.SPD_BEHIND_DEADBAND_M
                if abs(_be) <= _db:
                    behind_error_p = 0.0
                    if (now - self._last_spd_deadband_log) >= 2.0:
                        self._last_spd_deadband_log = now
                        log(f"[SPD_DEADBAND] behind_err={behind_error:.3f} "
                            f"biased={_be:.3f} within "
                            f"+/-{_db:.2f} -> P-term=0 (D-term parks at leader)")
                else:
                    behind_error_p = math.copysign(abs(_be) - _db, _be)

            p_term = kp * behind_error_p
            d_term = -config.SPEED_KD * actual_vc

            # Stuck Detection (far-range stall recovery)
            is_stuck = (
                behind_error > config.STUCK_BEHIND_ERR_MIN_M
                and abs(actual_vc) < config.STUCK_VC_THRESHOLD_MS
            )

            if is_stuck:
                if self._stuck_start is None:
                    self._stuck_start = now
                if (now - self._stuck_start) > config.STUCK_TIME_THRESHOLD_S:
                    if not self._stuck_active:
                        self._stuck_active = True
                    # P7.4 v9.7: Use dt_integrator (freezes on spike)
                    self._stuck_integral_boost += config.STUCK_INTEGRAL_GAIN * dt_integrator
                    self._stuck_integral_boost = min(
                        self._stuck_integral_boost,
                        config.STUCK_INTEGRAL_MAX_MS,
                    )
            else:
                self._stuck_start = None
                self._stuck_active = False
                if self._stuck_integral_boost > 0:
                    # P7.4 v9.7: Use dt_integrator for decay (freezes on spike)
                    self._stuck_integral_boost *= math.exp(-dt_integrator / 2.5)

            speed_delta = p_term + d_term + self._stuck_integral_boost
            speed_delta = max(-max_closure, min(max_closure, speed_delta))

            # Closure Floor
            if behind_error > 5.0 and speed_delta < config.MIN_CLOSURE_FLOOR_NEAR_MS:
                speed_delta = config.MIN_CLOSURE_FLOOR_NEAR_MS

            # =============================================================
            # v9.2 Patch D: Anti-stall closure deficit integrator
            # P7.4 v9.7: Uses dt_integrator (freezes on spike)
            # =============================================================
            # Integrates when desired closure exceeds measured, only when
            # meaningfully behind target. Resets near target or on S1 risk.
            if use_schedule and behind_error > 0:
                deficit = desired_closure - actual_vc
                s1_risk = behind_error < 0.5 or actual_vc > 1.5

                # I2: Anti-windup — freeze integrator when safety clamp active
                safety_active = safety_clamp_cap is not None
                if (
                    behind_error > config.CLOSURE_DEFICIT_MIN_BEHIND_M
                    and deficit > config.CLOSURE_DEFICIT_MIN_ERROR_MS
                    and not s1_risk
                    and not safety_active
                ):
                    # P7.4: Use dt_integrator (freezes on spike)
                    self._closure_deficit_i += config.CLOSURE_DEFICIT_KI * deficit * dt_integrator
                else:
                    # Decay when inactive OR safety-clamped (I2)
                    self._closure_deficit_i *= math.exp(
                        -dt_integrator / config.CLOSURE_DEFICIT_DECAY_TAU_S
                    )

                # Anti-windup clamp
                self._closure_deficit_i = max(
                    0.0,
                    min(self._closure_deficit_i, config.CLOSURE_DEFICIT_I_MAX),
                )

                # RC-3 fix: Early decay when approaching target (was hard reset at 1.0m)
                # Begin exponential decay at 3.0m so integrator unwinds before arrival
                if behind_error < 3.0:
                    decay_rate = (3.0 - behind_error) / 3.0  # 0 at 3m, 1.0 at 0m
                    self._closure_deficit_i *= math.exp(-decay_rate * 2.0 * dt_integrator)
                if behind_error < 0.5 or s1_risk:
                    self._closure_deficit_i = 0.0

                speed_delta += self._closure_deficit_i

                # Braking clamp: ALWAYS the last check (Patch C)
                # Uses state-gated + speedup-compensated effective braking
                d_safe = max(behind_error, 0.001)
                v_safe = math.sqrt(2.0 * self._braking_a_effective * d_safe)
                speed_delta = min(speed_delta, v_safe)
            else:
                # Outside schedule range: decay deficit integrator
                # P7.4: Uses dt_integrator (freezes on spike)
                self._closure_deficit_i *= math.exp(
                    -dt_integrator / config.CLOSURE_DEFICIT_DECAY_TAU_S
                )

            # Rate-limited closure schedule logging
            if use_schedule and (now - self._last_closure_sched_log) >= 2.0:
                self._last_closure_sched_log = now
                d_safe = max(behind_error, 0.001)
                v_safe = math.sqrt(2.0 * self._braking_a_effective * d_safe)
                log(
                    f"[CLOSURE_SCHED] behind_err={behind_error:.1f} "
                    f"profile={desired_closure:.2f} "
                    f"v_safe={v_safe:.2f} "
                    f"braking_a={self._braking_a_effective:.3f} "
                    f"actual_vc={actual_vc:.2f} "
                    f"deficit_i={self._closure_deficit_i:.3f} "
                    f"speed_delta={speed_delta:.2f}"
                )

            raw_speed = leader_ref_tas + speed_delta

        # RC-4 fix: Convert TAS→EAS before sending to ArduPilot
        # ArduPilot DO_CHANGE_SPEED with speed_type=0 interprets as EAS/CAS.
        # Gated by DENSITY_CORRECTION_ENABLE for quick revert if M3b regresses.
        # ISA density ratio: rho/rho0 = (1 - h/44330.77)^4.2559
        if config.DENSITY_CORRECTION_ENABLE:
            _h_m = max(0.0, follower_alt_msl)
            _density_ratio = max(0.5, (1.0 - _h_m / 44330.77) ** 4.2559)
            raw_speed = raw_speed * math.sqrt(_density_ratio)

        # P2 v9.4: Arbitration pipeline tracking
        # Stage 1: Profile speed (before any clamps)
        spd_profile = raw_speed

        # Stage 2: Braking clamp (already applied via adaptive_brake_active)
        spd_after_brake = raw_speed
        brake_clamp_active = self._adaptive_brake_active

        # Stage 3: Climb clamp (if provided by controller)
        spd_after_climb = spd_after_brake
        climb_clamp_active = False
        climb_clamp_cap_eff = climb_clamp_cap
        # #336 D2: subtractive floor (DC/HC only, flag-gated). The lateral/vertical
        # gate may FREEZE approach (hold leader speed, zero closure) but must NEVER
        # command BELOW the leader's TRUE 24.7 -> removes the perceived-GS-driven
        # recede half of the CROSS_GATE limit-cycle (#335 binder). The overtake
        # branch (raw_speed), FD-006 last-decimeter negative closure, and the
        # Stage-4 safety clamp are untouched (all applied elsewhere).
        if climb_clamp_cap_eff is not None and _spd_cross_decouple:
            _floored = max(climb_clamp_cap_eff, config.LEADER_TARGET_TAS_MS)
            if _floored > climb_clamp_cap_eff and (now - self._last_cross_decouple_log) >= 2.0:
                self._last_cross_decouple_log = now
                log(f"[CROSS_DECOUPLE] climb_clamp {climb_clamp_cap_eff:.2f} -> floor "
                    f"{_floored:.2f} (>= leader {config.LEADER_TARGET_TAS_MS:.1f}); "
                    f"approach-freeze not recede")
            climb_clamp_cap_eff = _floored
        if climb_clamp_cap_eff is not None and spd_after_climb > climb_clamp_cap_eff:
            spd_after_climb = climb_clamp_cap_eff
            climb_clamp_active = True

        # Stage 4: Safety clamp / no-pass guard (if provided by controller)
        spd_after_safety = spd_after_climb
        safety_clamp_active = False
        if safety_clamp_cap is not None and spd_after_safety > safety_clamp_cap:
            spd_after_safety = safety_clamp_cap
            safety_clamp_active = True

        # 6. Floor/ceiling + slew Limiting
        raw_speed = max(config.SAFE_FLOOR_MS, min(config.SAFE_MAX_MS, spd_after_safety))

        dt = now - self._speed_cmd_slew_time

        # P5.2 v9.5: Asymmetric slew - faster down-slew when safety clamp active
        if self._adaptive_brake_active:
            slew_rate_up = config.OVERTAKE_SLEW_RATE_MS_PER_S
            slew_rate_down = config.OVERTAKE_SLEW_RATE_MS_PER_S
        elif safety_clamp_active or (climb_clamp_active and climb_clamp_fast_decel):
            # Safety clamp or DC climb clamp: normal up-slew, fast down-slew
            slew_rate_up = config.SPEED_CMD_SLEW_RATE_MS_PER_S
            slew_rate_down = config.NOPASS_GUARD_SAFETY_DOWN_SLEW
        else:
            slew_rate_up = config.SPEED_CMD_SLEW_RATE_MS_PER_S
            slew_rate_down = config.SPEED_CMD_SLEW_RATE_MS_PER_S

        if dt > 0:
            delta = raw_speed - self._speed_cmd_slewed
            if delta > 0:
                # Speeding up
                max_delta = slew_rate_up * dt
            else:
                # Slowing down
                max_delta = slew_rate_down * dt
            if abs(delta) > max_delta:
                self._speed_cmd_slewed += math.copysign(max_delta, delta)
            else:
                self._speed_cmd_slewed = raw_speed

        self._speed_cmd_slew_time = now

        # P2 v9.4: Determine dominant source
        dominant = "profile"
        if safety_clamp_active:
            dominant = "safety"
        elif climb_clamp_active:
            dominant = "climb"
        elif brake_clamp_active:
            dominant = "brake"

        # Store arbitration result (P5.3 v9.5: added closure cap fields)
        # P6.1 v9.6: Added target_range_m and behind_error_m
        self._last_arbitration = SpeedArbitrationResult(
            spd_req_profile_mps=spd_profile,
            spd_after_brake_mps=spd_after_brake,
            spd_after_climb_mps=spd_after_climb,
            spd_after_safety_mps=spd_after_safety,
            spd_cmd_final_mps=self._speed_cmd_slewed,
            brake_clamp_active=brake_clamp_active,
            climb_clamp_active=climb_clamp_active,
            safety_clamp_active=safety_clamp_active,
            dominant_source=dominant,
            deficit_i=self._closure_deficit_i,
            closure_cap_mps=self._closure_cap_mps,
            closure_cap_active=self._closure_cap_active,
            target_range_m=target_range_m,
            behind_error_m=behind_error,
        )

        # Update dominant counts
        self._arb_total_ticks += 1
        self._arb_dominant_counts[dominant] += 1

        return self._speed_cmd_slewed

    # =========================================================================
    # Closure Rate Computation (tied to behind_m axis)
    # =========================================================================
    def compute_closure_rate_behind(self, behind_m: float) -> float:
        """Test 324: Compute closure rate along the behind axis.

        Adapted for FormationControlLaws class.
        """
        # 1. Use the function pointer for time, not the method call
        now = self.sim_now_s_fn()

        # 2. Initialize variables on first run (Lazy Initialization)
        # P0.3: Safe init for late entry / reset cases
        if not hasattr(self, "_behind_m_prev") or self._behind_m_prev is None:
            self._behind_m_prev = behind_m
            self._behind_m_prev_time = now
            self._closure_rate_behind_ewma = 0.0
            return 0.0

        # P0.3: Reset if EWMA is non-finite (NaN/inf protection)
        if not math.isfinite(self._closure_rate_behind_ewma):
            self._closure_rate_behind_ewma = 0.0

        # Compute dt using sim-time
        dt = now - self._behind_m_prev_time

        # Widen bounds for high speedup (dt>2.0 more frequent at 5x+)
        if dt < 0.01 or dt > 5.0:  # Skip invalid intervals
            self._behind_m_prev = behind_m
            self._behind_m_prev_time = now
            return self._closure_rate_behind_ewma

        # Closure rate = (prev - current) / dt_sim
        # Positive when behind_m is decreasing (getting closer)
        closure_sample = (self._behind_m_prev - behind_m) / dt

        # EWMA smooth (dt-scaled to maintain consistent time constant)
        alpha = ewma_alpha_scaled(config.CLOSURE_RATE_EWMA_ALPHA, dt, nominal_dt=0.05)
        self._closure_rate_behind_ewma = (
            alpha * closure_sample + (1 - alpha) * self._closure_rate_behind_ewma
        )

        # P0 debug: Throttled log to confirm EWMA updates each tick (1 Hz max)
        if os.environ.get("CLOSURE_EWMA_DEBUG") and hasattr(self, "_last_ewma_debug_log"):
            if now - self._last_ewma_debug_log >= 1.0:
                print(
                    f"[EWMA_DEBUG] raw={closure_sample:+.2f} ewma={self._closure_rate_behind_ewma:+.2f} dt={dt:.3f}"
                )
                self._last_ewma_debug_log = now
        elif os.environ.get("CLOSURE_EWMA_DEBUG"):
            self._last_ewma_debug_log = now

        # Update previous values
        self._behind_m_prev = behind_m
        self._behind_m_prev_time = now

        return self._closure_rate_behind_ewma

    def reset_closure_deficit_integrator(self) -> None:
        """v9.2 Patch D: Reset deficit integrator on state transitions."""
        self._closure_deficit_i = 0.0

    def get_last_arbitration(self) -> Optional[SpeedArbitrationResult]:
        """P2 v9.4: Get last speed arbitration result for logging."""
        return self._last_arbitration

    def get_arbitration_stats(self) -> dict:
        """P2 v9.4: Get arbitration statistics for run_summary.

        Returns:
            Dict with total ticks and percentage breakdown by dominant source.
        """
        total = self._arb_total_ticks
        if total == 0:
            return {
                "total_ticks": 0,
                "pct_dominant_profile": 0.0,
                "pct_dominant_brake": 0.0,
                "pct_dominant_climb": 0.0,
                "pct_dominant_safety": 0.0,
            }

        return {
            "total_ticks": total,
            "pct_dominant_profile": 100.0 * self._arb_dominant_counts["profile"] / total,
            "pct_dominant_brake": 100.0 * self._arb_dominant_counts["brake"] / total,
            "pct_dominant_climb": 100.0 * self._arb_dominant_counts["climb"] / total,
            "pct_dominant_safety": 100.0 * self._arb_dominant_counts["safety"] / total,
        }
