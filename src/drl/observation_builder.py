"""Observation vector builder for the DRL pipeline (42-D; config.OBS_DIM).

Constructs a normalised, clipped, NaN-sanitised numpy array from
baseline geometry, follower/leader telemetry, phase context, and
the previous applied residual (POMDP fix).

HISTORICAL: Campaign 3 P2 added 2 zero-padded wind channels; D6.1
Amendment-O (SIGNED 2026-07-18) DELETED them. ``wind_channels`` is INERT.

Includes a first-order LP filter for leader yaw rate (index 20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry import wrap_heading_error

from . import config


@dataclass
class FollowerState:
    """Follower telemetry snapshot."""
    airspeed_ms: float
    alt_msl_m: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    vd_ms: float
    throttle_pct: float = 50.0  # C3o V5: normalized throttle (0-100%)


@dataclass
class LeaderState:
    """Leader telemetry snapshot."""
    airspeed_ms: float
    heading_deg: float
    alt_msl_m: float
    vd_ms: float


@dataclass
class EnvelopeFlags:
    """Binary envelope membership flags."""
    in_inner: bool
    in_outer: bool


@dataclass
class SensorInfo:
    """Sensor quality context."""
    mode_id: int     # 0=GPS, 1=FUSED, 2=UWB_DOM, 3=UWB_ONLY
    uwb_valid: bool


class ObservationBuilder:
    """Builds the normalised observation vector (42-D; see config.OBS_DIM).

    Parameters
    ----------
    speedup_factor: float
        Current SITL speedup (1.0 or 2.0).
    yaw_rate_alpha: float
        LP filter coefficient for leader yaw rate. Higher = more responsive.
    wind_channels: bool
        INERT — retained only for caller compatibility. D6.1 Amendment-O
        (SIGNED 2026-07-18) DELETED the two legacy wind channels along with
        set_wind_noise; the builder now always emits the 42-D layout
        (config.OBS_DIM), whatever this flag says. The pre-Campaign-3 shape
        (config.OBS_DIM_LEGACY) is a historical reference only and is not
        producible on this path.
    """

    def __init__(self, speedup_factor: float, yaw_rate_alpha: float = 0.3,
                 wind_channels: bool = True) -> None:
        self._speedup_factor = speedup_factor
        self._yaw_rate_alpha = yaw_rate_alpha
        self._wind_channels = wind_channels
        # LP filter state for leader yaw rate
        self._prev_leader_heading_deg: Optional[float] = None
        self._leader_yaw_rate_filt_dps: float = 0.0
        # Previous applied residual (set externally between steps)
        self._prev_applied_residual = np.zeros(3, dtype=np.float32)
        # C3i I5: previous closure rate for delta computation
        self._prev_closure_rate_ms: float = 0.0
        # C3n: Previous behind_error for drift rate computation
        self._prev_behind_error: float = 0.0
        # C3o V5: Previous follower airspeed for longitudinal acceleration
        self._prev_follower_airspeed: float = -1.0  # sentinel: return 0 on first tick
        # D6.1 Amendment-O: wind channels + set_wind_noise deleted; the
        # wind_channels ctor arg is retained for caller compatibility but is
        # INERT — the builder always emits the 42-D layout (config.OBS_DIM).

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state (call at episode start)."""
        self._prev_leader_heading_deg = None
        self._leader_yaw_rate_filt_dps = 0.0
        self._prev_applied_residual = np.zeros(3, dtype=np.float32)
        self._prev_closure_rate_ms = 0.0
        self._prev_behind_error = 0.0
        self._prev_follower_airspeed = -1.0

    def set_prev_applied_residual(self, residual: np.ndarray) -> None:
        """Store the applied residual from the current tick for next obs."""
        self._prev_applied_residual = np.asarray(residual, dtype=np.float32)

    def build(
        self,
        geom,                        # RelativeGeometry dataclass
        follower: FollowerState,
        leader: LeaderState,
        phase_id: int,               # 0=INIT, 1=REJOIN, 2=PRECONTACT, 3=CONTACT
        envelope: EnvelopeFlags,
        sensor: SensorInfo,
        dt_s: float,
        continuous_contact_m3b_dwell_s: float,
        milestone_tier: int,         # 0=NONE..4=STRICT
        target_behind: float,
        target_below: float,
        speed_delta_ms: float,       # Baseline speed correction
        max_closure_ms: float,       # Current closure limit
        bearing_to_leader_deg: float = 0.0,
        wake_hazard_intensity: float = 0.0,
    ) -> np.ndarray:
        """Build observation vector — shape (config.OBS_DIM,) float32.

        All values normalised, NaN-sanitised, and clipped to ±OBS_CLIP.
        """
        # --- Leader yaw rate LP filter (index 20) ---
        if self._prev_leader_heading_deg is not None and dt_s > 0:
            raw_rate = wrap_heading_error(
                leader.heading_deg - self._prev_leader_heading_deg
            ) / dt_s
            self._leader_yaw_rate_filt_dps = (
                self._yaw_rate_alpha * raw_rate
                + (1 - self._yaw_rate_alpha) * self._leader_yaw_rate_filt_dps
            )
        self._prev_leader_heading_deg = leader.heading_deg

        # --- Envelope errors ---
        behind_error = geom.behind_m - target_behind
        cross_error = geom.cross_m - 0.0  # target cross is always 0
        below_error = geom.below_m - target_below

        # --- Heading error ---
        heading_error = wrap_heading_error(
            follower.heading_deg - leader.heading_deg
        )

        # --- Raw observation (config.OBS_DIM elements) ---
        raw = np.array([
            # 0-3: Relative position
            geom.behind_m,
            geom.cross_m,
            geom.below_m,
            geom.range_m,
            # 4-6: Relative velocity
            geom.closure_rate_ms,
            geom.vel_cross_m_s,
            geom.vel_below_m_s,
            # 7-9: Envelope errors
            behind_error,
            cross_error,
            below_error,
            # 10-15: Follower state
            follower.airspeed_ms,
            follower.alt_msl_m,
            follower.heading_deg,
            follower.pitch_deg,
            follower.roll_deg,
            follower.vd_ms,
            # 16-20: Leader state
            leader.airspeed_ms,
            leader.heading_deg,
            leader.alt_msl_m,
            leader.vd_ms,
            self._leader_yaw_rate_filt_dps,
            # 21-23: Heading geometry
            heading_error,
            bearing_to_leader_deg,
            geom.target_aspect_deg,
            # 24-25: Speed & closure context
            speed_delta_ms,
            max_closure_ms,
            # 26-28: Phase & envelope flags
            float(phase_id),
            float(envelope.in_inner),
            float(envelope.in_outer),
            # D6.1 Amendment-O (SIGNED 2026-07-18): dropped legacy 29-30
            # (sensor_mode_id, uwb_valid), 31 (dt_s), 34 (speedup_factor) and
            # legacy 39-40 wind channels (structurally disconnected — fork (a));
            # HISTORICAL: obs[38] wake KEPT (now index 34); 48-D -> 42-D.
            # 29: continuous_contact_m3b_dwell_s (C3h: was accumulated)
            continuous_contact_m3b_dwell_s,
            # 30: milestone tier
            float(milestone_tier),
            # 31-33: POMDP fix — action memory
            self._prev_applied_residual[0],
            self._prev_applied_residual[1],
            self._prev_applied_residual[2],
            # 34: Wake turbulence hazard
            wake_hazard_intensity,
        ], dtype=np.float32)

        # C3i I4: TTC to front M3b boundary at current closure rate.
        # Uses margin to front boundary (behind_m + 0.7m), not raw behind_m.
        # Positive = seconds until front wall; negative = already past boundary.
        # Sentinel +5.0 when not closing (no overshoot risk).
        behind_margin_front = geom.behind_m + config.CONTACT_M3B_BEHIND_TOL
        if geom.closure_rate_ms > 0.01:
            ttc_raw = behind_margin_front / geom.closure_rate_ms
        else:
            ttc_raw = 5.0
        ttc_clamped = float(np.clip(ttc_raw, -5.0, 5.0))

        # C3i I5: closure rate delta — acceleration/deceleration signal.
        closure_rate_delta = (
            (geom.closure_rate_ms - self._prev_closure_rate_ms) / max(dt_s, 0.001)
        )
        self._prev_closure_rate_ms = geom.closure_rate_ms
        crd_clamped = float(np.clip(closure_rate_delta, -5.0, 5.0))

        # Append raw clamped values — OBS_NORMS handles normalization.
        raw = np.append(raw, np.array([ttc_clamped, crd_clamped], dtype=np.float32))

        # C3n: Aft-boundary anticipation features (behind-axis specific, SIGNED).
        # These channels are LIVE regardless of the RFL phase-gate state.
        # Before the phase-gate activates offsets, the agent sees non-zero
        # aft-boundary obs with zero applied_residual — this is by design
        # (early warning before authority is granted), not a bug.
        behind_error_now = geom.behind_m - target_behind

        # obs[43]: Signed distance to aft M3b boundary.
        # Positive = inside (safe), negative = past boundary (violated).
        aft_boundary_dist = float(np.clip(
            config.CONTACT_M3B_BEHIND_TOL - behind_error_now, -1.0, 1.0))

        # obs[44]: Signed behind drift rate (positive = toward aft boundary).
        behind_drift_rate = (behind_error_now - self._prev_behind_error) / max(dt_s, 0.001)
        self._prev_behind_error = behind_error_now
        behind_drift_rate = float(np.clip(behind_drift_rate, -5.0, 5.0))

        # obs[45]: TTC to aft boundary (directional-safe).
        # Only compute when drifting TOWARD boundary (drift > 0).
        # Drifting away or stationary → 10.0 sentinel (safe).
        if behind_drift_rate > 0.01:
            aft_ttc = float(np.clip(aft_boundary_dist / behind_drift_rate, 0.0, 10.0))
        else:
            aft_ttc = 10.0

        raw = np.append(raw, np.array([
            aft_boundary_dist, behind_drift_rate, aft_ttc,
        ], dtype=np.float32))

        # C3o V5: Leading indicators for double-integrator anticipatory control
        # obs[46]: Longitudinal acceleration (d(airspeed)/dt) — LEADING indicator.
        # Position (obs[43]) and velocity (obs[44]) are lagging for a double-integrator.
        if self._prev_follower_airspeed < 0:
            long_accel = 0.0  # First tick: sentinel, no valid derivative
        else:
            long_accel = ((follower.airspeed_ms - self._prev_follower_airspeed)
                          / max(dt_s, 0.001))
        self._prev_follower_airspeed = follower.airspeed_ms
        long_accel = float(np.clip(long_accel, -5.0, 5.0))

        # obs[47]: Follower throttle percentage — actuator state of the double-integrator.
        # Enables the critic to predict future velocity change before it manifests in airspeed.
        throttle = float(np.clip(follower.throttle_pct, 0.0, 100.0))

        raw = np.append(raw, np.array([long_accel, throttle], dtype=np.float32))

        # --- Normalise ---
        norms = np.array(config.OBS_NORMS[:len(raw)], dtype=np.float32)
        obs = raw / norms

        # --- NaN/inf sanitisation ---
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Clip ---
        obs = np.clip(obs, -config.OBS_CLIP, config.OBS_CLIP)

        return obs
