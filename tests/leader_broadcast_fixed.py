#!/usr/bin/env python3
"""
FlightDock Phase 7C - DHC-2 Beaver Broadcaster with Physical Weather Model
==========================================================================

This script broadcasts DHC-2 Beaver lead aircraft telemetry that lives in the same
wind/turbulence/weather field as the ArduPilot SITL QuadPlane follower.

CRITICAL CHANGE (2025-12-03): The DHC-2 Beaver now uses AIRSPEED + WIND for motion,
not a fixed ground-speed racetrack. This is required for:
  - Sim2Real transfer (real aircraft experience wind)
  - DRL domain randomization (vary wind conditions during training)
  - Correct relative motion when follower also experiences wind

Architecture:
  - WeatherModel: Single source of truth for wind/turbulence/gusts
  - Motion integration: v_ground = v_air + wind, where v_air is along track at airspeed
  - Telemetry: Reports correct airspeed AND groundspeed (they differ when windy!)
  - SITL Sync: Automatically pushes SIM_WIND_* params to ArduPilot SITL QuadPlane

DUAL-LINK ARCHITECTURE:
  - UDP link: udpout:127.0.0.1:14550 -> MAVProxy ingest (for monitoring/GCS)
  - TCP link: tcp:127.0.0.1:5763 -> Direct to SITL SERIAL2 (for ArduPlane processing)
  - Both links broadcast GLOBAL_POSITION_INT at 10 Hz with sysid=2
  - TCP link is required for ArduPlane FormationController to receive lead telemetry
  - MAVProxy does NOT forward external UDP to SITL by default, hence TCP bypass

SITL Weather Sync (Option B - Single Atmosphere):
  - By default, weather is synced to ArduPilot SITL via PARAM_SET messages
  - Both DHC-2 Beaver and QuadPlane share the same atmosphere
  - This eliminates airspeed vs groundspeed mismatches between aircraft
  - Use --no-sitl-weather-sync to disable (for debugging only)

Usage:
    # Calm conditions (default)
    python3 leader_broadcast_fixed.py

    # 3 m/s wind from north with turbulence (synced to SITL)
    python3 leader_broadcast_fixed.py --wind-speed 3.0 --wind-dir 0 --turbulence 0.3

    # DRL-style randomized weather with seed (synced to SITL)
    python3 leader_broadcast_fixed.py --wind-speed 2.0 --wind-dir 45 --turbulence 0.2 --weather-seed 42

    # Disable SITL sync (leader only experiences wind)
    python3 leader_broadcast_fixed.py --wind-speed 2.0 --no-sitl-weather-sync

    # Custom endpoints for UDP/TCP
    python3 leader_broadcast_fixed.py --udp-out 127.0.0.1:14550 --sitl-tcp 127.0.0.1:5763

Arguments:
    --target-speed SPEED Lead aircraft AIRSPEED in m/s (default: 24.7)
    --udp-out HOST:PORT UDP output endpoint for MAVProxy (default: 127.0.0.1:14550)
    --sitl-tcp HOST:PORT Direct SITL TCP endpoint (default: 127.0.0.1:5763)
    --rate-hz HZ Telemetry broadcast rate in Hz (default: 10)
    --wind-speed SPEED Base wind speed in m/s (default: 0.0)
    --wind-dir DEG Wind direction FROM which wind blows in degrees (default: 0.0, north)
    --turbulence INTENSITY Turbulence intensity 0.0-1.0 (default: 0.0)
    --gust-amplitude SPEED Gust amplitude in m/s (default: 0.0)
    --weather-seed SEED Random seed for weather (default: None = random)
    --no-sitl-weather-sync Disable syncing weather to SITL QuadPlane
    --sitl-sysid SYSID Target system ID for SITL QuadPlane (default: 1)
    --sitl-compid COMPID Target component ID for SITL QuadPlane (default: 1)
"""
import argparse
import math
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
from pymavlink import mavutil
from sim_time_authority import (  # TEST 387: Wall-time guards for sim-time contract enforcement
    SimTimeAuthority,
    wall_monotonic_s,
    wall_now_s,
)
from engage_epoch_quant import resolve_engage_epoch  # Option B (#308 C+B): epoch-quantized engage latch

# =============================================================================
# TAS-Only Speed Reference for FlightDock Research
# =============================================================================
# UPDATED (2025-12-08): DHC-2 Beaver now uses True Airspeed (TAS) directly
# as its primary speed reference. This simplifies the docking control system
# by eliminating EAS/TAS conversion confusion.
#
# Design Decision:
#   - FlightDock research operates strictly with TAS (True Airspeed)
#   - --target-speed argument specifies TAS directly (default: 24.7 m/s)
#   - No EAS→TAS conversion needed - what you specify is what you get
#   - Groundspeed = TAS + wind component (in NED frame)
#
# Rationale:
#   - TAS is the actual speed of the aircraft through the air mass
#   - TAS is directly observable from relative motion between aircraft
#   - Using TAS eliminates altitude-dependent EAS/TAS conversion errors
#   - Simplifies speed matching between leader and follower
#
# Note on ArduPilot QuadPlane:
#   - ArduPilot TECS targets EAS internally, but VFR_HUD reports EAS
#   - The follower controller must account for this when commanding speeds
#   - At 900m MSL: EAS ≈ TAS / 1.047 (eas2tas factor)
#   - For TAS = 24.7 m/s, equivalent EAS ≈ 23.6 m/s
# =============================================================================


class LeaderTxDelayLine:
    """#379 (P0.3): constant leader-datalink transport delay, sender-side.

    Knob: FD_LEADER_TX_DELAY_S (seconds). Unset/0 => inert (the send paths are
    behaviorally identical to pre-#379). Active => every outbound leader frame
    captured at sim-time t is transmitted at t + delay with its ORIGINAL field
    values and time_boot_ms — a delayed link delivers stale data bearing stale
    stamps, so receiver-side leader age IS the commanded delay. Pure FIFO keyed
    on sim-time: delivery preserves capture order, so the receiver's tbms
    monotonic contract holds. Boot gap: the first delivery lands delay seconds
    after the first capture — exactly a real link's first-packet arrival;
    absorbed by warmup (witness detector b). Truth-side channels are NOT
    routed through this line (G1/G4; witness detector a).
    """

    def __init__(self, delay_s: float) -> None:
        from collections import deque
        self.delay_s = max(0.0, float(delay_s))
        self._q = deque()

    @property
    def active(self) -> bool:
        return self.delay_s > 0.0

    def push(self, t_capture: float, kind: str, args: tuple) -> None:
        self._q.append((float(t_capture), kind, args))

    def pop_due(self, sim_now: float) -> list:
        due = []
        while self._q and (float(sim_now) - self._q[0][0]) >= self.delay_s:
            due.append(self._q.popleft())
        return due


_PHASELOC_FH = None          # #512 P-EMIT handle; None => probe inert
_PHASELOC_RESOLVED = False   # env var read once, at first send


def _phaseloc_fh():
    """#512 P-EMIT probe (ledger T-6 / PREREG_512 §2). Returns an append handle
    iff FD_PHASELOC_EMIT_LOG names a writable path, else None.

    Unset => this returns None on every call and no file is ever opened, so the
    send path is behaviourally identical to pre-#512 (PREREG_512 gate G-2,
    asserted by tests/unit/test_phaseloc_emit_probe.py). Localization needs the
    emit instant on the SAME clock domain as the MAVProxy-relay listener and the
    drain shim, which is CLOCK_REALTIME; wall_now_s() is the sanctioned accessor
    for cross-process logging correlation (sim_time_authority.py:696-704).
    """
    global _PHASELOC_FH, _PHASELOC_RESOLVED
    if not _PHASELOC_RESOLVED:
        _PHASELOC_RESOLVED = True
        path = os.environ.get("FD_PHASELOC_EMIT_LOG", "").strip()
        if path:
            try:
                _PHASELOC_FH = open(path, "a", buffering=1)
            except Exception:
                _PHASELOC_FH = None   # probe failure must never kill a leg
    return _PHASELOC_FH


_ESTAMP_SOCK = None          # T-392a side channel; None => disabled/never opened
_ESTAMP_ADDR = None


def _estamp_open(target: str) -> None:
    """T-392a: open the epoch-stamp side channel ('host:port'). Failure is
    swallowed — an observability channel must never kill a leg."""
    global _ESTAMP_SOCK, _ESTAMP_ADDR
    if not target or os.environ.get("FD_EPOCH_STAMP", "1").strip() == "0":
        return
    try:
        import socket as _s
        host, _, port = target.rpartition(":")
        _ESTAMP_ADDR = (host or "127.0.0.1", int(port))
        _ESTAMP_SOCK = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
    except Exception:
        _ESTAMP_SOCK, _ESTAMP_ADDR = None, None


def _estamp_send(wire_tbms: int, raw_latch_tbms: int) -> None:
    """Mirror one position packet's (wire stamp, raw ATTITUDE latch) pair."""
    if _ESTAMP_SOCK is None or _ESTAMP_ADDR is None:
        return
    try:
        _ESTAMP_SOCK.sendto(b"ES %d %d\n" % (int(wire_tbms),
                                             int(raw_latch_tbms)),
                            _ESTAMP_ADDR)
    except Exception:
        pass


def _tx_send_posvfr(links, a) -> None:
    """#379 helper: GLOBAL_POSITION_INT + VFR_HUD on both links. Used by BOTH
    the immediate (delay inactive) and delayed-delivery paths — one payload
    shape, one send routine, no drift between the two paths."""
    fh = _phaseloc_fh()
    t_wall = wall_now_s() if fh is not None else 0.0   # stamp BEFORE the sends
    for link in links:
        if link is not None:
            try:
                link.mav.global_position_int_send(
                    a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
                link.mav.vfr_hud_send(a[9], a[10], a[11], a[12], a[13], a[14])
            except Exception:
                pass  # TCP may be disconnected
    # T-392a: mirror the stamp pair AFTER the wire sends (never before — the
    # wire is the product, this is the witness). a[15] rides the SAME tuple
    # through the delay line, so a delayed packet reports the latch that was
    # live when it was BUILT ("stale stamps ARE the delay"). len() guard keeps
    # legacy 15-tuples working untouched.
    if len(a) > 15:
        _estamp_send(a[0], a[15])
    if fh is not None:
        try:
            fh.write('{"t_wall": %.6f, "t_mono": %.6f, "tbms": %d}\n'
                     % (t_wall, wall_monotonic_s(), a[0]))
        except Exception:
            pass


def _tx_send_att(links, a) -> None:
    """#379 helper: ATTITUDE on both links (see _tx_send_posvfr)."""
    for link in links:
        if link is not None:
            try:
                link.mav.attitude_send(a[0], a[1], a[2], a[3], a[4], a[5], a[6])
            except Exception:
                pass  # TCP may be disconnected


def get_air_density_isa(alt_msl_m: float) -> float:
    """
    Compute air density at altitude using International Standard Atmosphere (ISA).

    ISA model (for troposphere, altitude < 11000m):
    - Temperature: T = T0 - L * h (linear lapse rate)
    - Pressure: P = P0 * (T/T0)^(g/(R*L))
    - Density: rho = P / (R * T)

    Args:
        alt_msl_m: Altitude above mean sea level in meters

    Returns:
        Air density in kg/m³
    """
    T0 = 288.15  # Sea level standard temperature (K)
    P0 = 101325.0  # Sea level standard pressure (Pa)
    L = 0.0065  # Temperature lapse rate (K/m)
    R = 287.05  # Gas constant for air (J/(kg·K))
    g = 9.80665  # Standard gravity (m/s²)

    # Clamp altitude to troposphere
    alt = max(0.0, min(alt_msl_m, 11000.0))

    # Temperature at altitude
    T = T0 - L * alt

    # Pressure at altitude (using barometric formula)
    P = P0 * (T / T0) ** (g / (R * L))

    # Density from ideal gas law
    rho = P / (R * T)

    return rho


def get_eas2tas_factor(alt_msl_m: float) -> float:
    """
    Get the EAS to TAS conversion factor at a given altitude.

    Args:
        alt_msl_m: Altitude above mean sea level in meters

    Returns:
        eas2tas factor (TAS = EAS * factor)
    """
    RHO0 = 1.225  # Sea level standard density (kg/m³)
    rho = get_air_density_isa(alt_msl_m)
    return math.sqrt(RHO0 / rho)


# =============================================================================
# WeatherModel: Shared Wind/Turbulence/Gust Environment
# =============================================================================
# This model provides the wind vector in NED frame for the DHC-2 Beaver.
# In the future, this same model will drive ArduPilot SITL via SIM_WIND_* params
# so both aircraft experience identical weather conditions.
#
# ArduPilot SIM_* Parameter Alignment:
#   - base_wind_speed_m_s -> SIM_WIND_SPD
#   - base_wind_dir_deg -> SIM_WIND_DIR (direction FROM which wind blows)
#   - turbulence_intensity -> SIM_WIND_TURB (0.0-1.0 scale)
#   - turbulence_scale_m -> ~SIM_WIND_T_ALT (altitude scale for turbulence)
#   - gust_amplitude_m_s -> Custom (ArduPilot doesn't have direct equiv)
#
# For DRL Domain Randomization:
#   - Instantiate with seed= for reproducible episodes
#   - Vary parameters across episodes: wind speed 0-5 m/s, direction 0-360°, etc.
# =============================================================================


class WeatherModel:
    """
    Physically consistent wind/turbulence model for aerial docking simulation.

    Provides a time-varying wind vector in NED frame that can be shared between
    the DHC-2 Beaver leader script and ArduPilot SITL (via parameter sync).

    Wind Components:
    1. Base wind: Steady-state wind from configured direction
    2. Turbulence: Low-frequency random variations (Ornstein-Uhlenbeck process)
    3. Gusts: Periodic larger variations (sinusoidal + harmonics)

    All components combine additively: wind_ned = base + turbulence + gust
    """

    def __init__(
        self,
        base_wind_speed_m_s: float = 0.0,
        base_wind_dir_deg: float = 0.0,  # Direction FROM which wind blows (ArduPilot convention)
        base_wind_alt_m: float = 0.0,  # Reference altitude for base wind (for future gradient)
        turbulence_intensity: float = 0.0,  # 0.0-1.0 scale, maps to ~SIM_WIND_TURB
        turbulence_scale_m: float = 100.0,  # Spatial scale of turbulence
        gust_amplitude_m_s: float = 0.0,  # Max gust deviation from base
        gust_period_s: float = 30.0,  # Primary gust period
        seed: Optional[int] = None,  # For reproducible weather in DRL
    ):
        """
        Initialize weather model.

        Args:
            base_wind_speed_m_s: Steady wind speed (m/s)
            base_wind_dir_deg: Direction FROM which wind blows (degrees, 0=N, 90=E)
            base_wind_alt_m: Reference altitude for wind (m AGL)
            turbulence_intensity: 0.0 (calm) to 1.0 (severe), scales turbulence amplitude
            turbulence_scale_m: Spatial correlation scale of turbulence (m)
            gust_amplitude_m_s: Peak gust amplitude above/below base wind (m/s)
            gust_period_s: Period of primary gust oscillation (s)
            seed: Random seed for reproducibility in DRL training
        """
        self.base_wind_speed_m_s = base_wind_speed_m_s
        self.base_wind_dir_deg = base_wind_dir_deg
        self.base_wind_alt_m = base_wind_alt_m
        self.turbulence_intensity = turbulence_intensity
        self.turbulence_scale_m = turbulence_scale_m
        self.gust_amplitude_m_s = gust_amplitude_m_s
        self.gust_period_s = gust_period_s
        self.seed = seed

        # Initialize RNG for turbulence
        self._rng = np.random.default_rng(seed)

        # Ornstein-Uhlenbeck state for turbulence (N, E, D components)
        # Turbulence is modeled as a mean-reverting random process
        self._turb_state_ned = np.zeros(3)
        self._turb_theta = 0.1  # Mean reversion rate (1/s)
        self._turb_sigma = 1.0  # Noise amplitude (scaled by intensity)

        # Time tracking for turbulence integration
        self._last_update_time = 0.0

        # Precompute base wind vector in NED
        # ArduPilot convention: wind_dir is direction FROM which wind blows
        # So if wind_dir=0 (from north), wind velocity is toward SOUTH (negative N)
        # If wind_dir=90 (from east), wind velocity is toward WEST (negative E)
        theta_rad = math.radians(self.base_wind_dir_deg)
        self._base_wind_ned = np.array(
            [
                -self.base_wind_speed_m_s * math.cos(theta_rad),  # N component
                -self.base_wind_speed_m_s * math.sin(theta_rad),  # E component
                0.0,  # D component (no vertical base wind)
            ]
        )

        # Gust direction unit vector (aligned with base wind)
        if self.base_wind_speed_m_s > 0.01:
            self._gust_dir_ned = self._base_wind_ned / np.linalg.norm(
                self._base_wind_ned[:2] + [0.01]
            )
        else:
            # Default gust direction if no base wind (from north)
            self._gust_dir_ned = np.array([-1.0, 0.0, 0.0])

    def update(self, t: float) -> None:
        """
        Advance internal turbulence/gust state to time t.

        Call this once per simulation step before get_wind_ned().

        Args:
            t: Current simulation time in seconds
        """
        if self._last_update_time == 0.0:
            self._last_update_time = t
            return

        dt = t - self._last_update_time
        if dt <= 0:
            return

        self._last_update_time = t

        # Update Ornstein-Uhlenbeck turbulence state
        # dX = -theta * X * dt + sigma * sqrt(dt) * dW
        if self.turbulence_intensity > 0:
            noise = self._rng.standard_normal(3)
            decay = math.exp(-self._turb_theta * dt)
            diffusion = self._turb_sigma * math.sqrt(1 - decay**2)
            self._turb_state_ned = decay * self._turb_state_ned + diffusion * noise

    def get_wind_ned(self, alt_m: float = 0.0) -> np.ndarray:
        """
        Return current wind vector in NED frame at given altitude.

        Args:
            alt_m: Altitude in meters (currently unused, for future wind gradient)

        Returns:
            Wind velocity vector in NED frame (m/s) as numpy array [vn, ve, vd]
        """
        # Start with base wind
        wind_ned = self._base_wind_ned.copy()

        # Add turbulence component (scaled by intensity)
        if self.turbulence_intensity > 0:
            # Scale: intensity 0.0-1.0 maps to roughly 0-3 m/s turbulence amplitude
            turb_amplitude = 3.0 * self.turbulence_intensity
            wind_ned += turb_amplitude * self._turb_state_ned

        # Add gust component (sinusoidal with harmonics for realism)
        if self.gust_amplitude_m_s > 0 and self.gust_period_s > 0:
            t = self._last_update_time
            # Primary gust oscillation
            gust_primary = math.sin(2.0 * math.pi * t / self.gust_period_s)
            # Higher harmonic for texture
            gust_harmonic = 0.3 * math.sin(4.0 * math.pi * t / self.gust_period_s + 0.7)
            gust_magnitude = self.gust_amplitude_m_s * (gust_primary + gust_harmonic)
            wind_ned += gust_magnitude * self._gust_dir_ned

        return wind_ned

    def to_ardupilot_sim_params(self) -> Dict[str, float]:
        """
        Return ArduPilot SIM_* parameter equivalents for this weather model.

        Use this to configure ArduPilot SITL with matching weather conditions.
        In MAVProxy or via parameter file:
            param set SIM_WIND_SPD <value>
            param set SIM_WIND_DIR <value>
            etc.

        Returns:
            Dictionary mapping SIM_* parameter names to values
        """
        return {
            "SIM_WIND_SPD": self.base_wind_speed_m_s,
            "SIM_WIND_DIR": self.base_wind_dir_deg,
            "SIM_WIND_TURB": self.turbulence_intensity,
            # Note: ArduPilot's turbulence model differs from ours;
            # these are approximate mappings
            "SIM_WIND_T_ALT": self.turbulence_scale_m,
            # Gust parameters don't have direct ArduPilot equivalents
            # We document them for manual config if needed
            "_GUST_AMP_M_S": self.gust_amplitude_m_s,  # Custom param
            "_GUST_PERIOD_S": self.gust_period_s,  # Custom param
        }

    def get_description(self) -> str:
        """Return human-readable description of current weather configuration."""
        parts = []
        if self.base_wind_speed_m_s > 0.01:
            parts.append(
                f"Wind: {self.base_wind_speed_m_s:.1f} m/s from {self.base_wind_dir_deg:.0f}°"
            )
        else:
            parts.append("Wind: Calm")

        if self.turbulence_intensity > 0.01:
            turb_desc = {
                0.0: "none",
                0.25: "light",
                0.5: "moderate",
                0.75: "severe",
                1.0: "extreme",
            }
            closest = min(turb_desc.keys(), key=lambda k: abs(k - self.turbulence_intensity))
            parts.append(f"Turbulence: {turb_desc[closest]} ({self.turbulence_intensity:.2f})")

        if self.gust_amplitude_m_s > 0.01:
            parts.append(
                f"Gusts: ±{self.gust_amplitude_m_s:.1f} m/s @ {self.gust_period_s:.0f}s period"
            )

        if self.seed is not None:
            parts.append(f"Seed: {self.seed}")

        return " | ".join(parts)


# =============================================================================
# SITL Weather Sync: Push WeatherModel → ArduPilot SIM_* Parameters
# =============================================================================
# This function ensures the ArduPilot SITL QuadPlane experiences the same
# wind/turbulence as the DHC-2 Beaver leader by setting SIM_* parameters.
#
# Why this matters:
#   - Without sync: QuadPlane may have different wind than the leader
#   - With sync: Both aircraft share the same atmosphere (Option B)
#   - This eliminates airspeed vs groundspeed mismatches between aircraft
#
# For DRL Domain Randomization:
#   - Call this once per episode after weather randomization
#   - Both aircraft will see the same randomized weather
# =============================================================================


def sync_weather_to_sitl(
    mav,
    weather: WeatherModel,
    sitl_sysid: int = 1,
    sitl_compid: int = 1,
    verbose: bool = True,
    connection_string: str = "",
) -> bool:
    """
    Push the WeatherModel configuration into the ArduPilot SITL QuadPlane.

    This ensures the SITL atmosphere (wind + turbulence) matches the leader's
    WeatherModel, so both aircraft share the same environment.

    Added robust verification with PARAM_REQUEST_READ.
    Handle unidirectional connections (udpout:).

    Architecture Note:
        - udpout: connections are OUTPUT-ONLY and cannot receive responses
        - For udpout:, we skip Phase 2 verification (PARAM_SET still works)
        - For bidirectional connections (tcp:, udp:, udpin:), full verification

    Phases:
        - Phase 0: Hard reset all SIM_WIND_* to zero (clear stale weather)
        - Phase 1: Apply actual weather values from WeatherModel
        - Phase 2: Verify each parameter (SKIPPED for udpout: connections)

    Args:
        mav: Existing MAVLink connection (via MAVProxy) used by the leader.
        weather: WeatherModel instance containing the weather configuration.
        sitl_sysid: Target system ID of the QuadPlane SITL (default: 1).
        sitl_compid: Target component ID of the QuadPlane SITL (default: 1).
        verbose: Print each parameter as it's set.
        connection_string: MAVLink connection string (to detect udpout:).

    Returns:
        True if sync completed (verified or assumed for udpout:), False on error.
    """
    # Core SIM_WIND_* parameters to sync (in order of importance)
    CORE_WIND_PARAMS = ["SIM_WIND_SPD", "SIM_WIND_DIR", "SIM_WIND_TURB"]

    sim_params = weather.to_ardupilot_sim_params()
    success = True
    any_mismatch = False

    # =========================================================================
    #Detect unidirectional connections
    # =========================================================================
    # udpout: connections are OUTPUT-ONLY - they cannot receive MAVLink
    # responses. PARAM_SET commands will be sent, but PARAM_VALUE responses
    # cannot be received for verification. In this case, we skip Phase 2.
    is_unidirectional = "udpout:" in connection_string.lower()

    if is_unidirectional and verbose:
        print("  [SITL_WEATHER_SYNC] Note: Using udpout: (output-only connection)")
        print("  [SITL_WEATHER_SYNC] PARAM_SET will be sent; verification skipped")

    def _set_param(name: str, value: float) -> bool:
        """Helper: send PARAM_SET for a single parameter."""
        try:
            param_id = name.encode("ascii")
            if len(param_id) > 16:
                param_id = param_id[:16]
            mav.mav.param_set_send(
                sitl_sysid,
                sitl_compid,
                param_id,
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            return True
        except Exception as e:
            print(f"  [SITL_WEATHER_SYNC] FAILED to set {name}: {e}")
            return False

    def _verify_param(name: str, expected_value: float, tolerance: float = 1e-3) -> bool:
        """
        Helper: request PARAM_VALUE readback and verify against expected.
        Returns True if verified OK, False if mismatch or timeout.
        """
        try:
            # Encode parameter name
            param_id = name.encode("ascii")
            if len(param_id) > 16:
                param_id = param_id[:16]

            # Request parameter value
            mav.mav.param_request_read_send(
                sitl_sysid, sitl_compid, param_id, -1  # param_index=-1 means use param_id string
            )

            # Wait for PARAM_VALUE response (timeout after 1s)
            start_time = time.time()
            while time.time() - start_time < 1.0:
                msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.2)
                if msg is None:
                    continue
                # Check if this is our parameter (name match)
                msg_param_id = msg.param_id.rstrip("\x00") if hasattr(msg, "param_id") else ""
                if msg_param_id == name:
                    returned_value = msg.param_value
                    if abs(returned_value - expected_value) <= tolerance:
                        print(
                            f"  [SITL_WEATHER_VERIFY] {name} OK: sent={expected_value:.2f} got={returned_value:.2f}"
                        )
                        return True
                    else:
                        print(
                            f"  [SITL_WEATHER_VERIFY] {name} MISMATCH: "
                            f"sent={expected_value:.2f} got={returned_value:.2f}"
                        )
                        return False

            # Timeout
            print(f"  [SITL_WEATHER_VERIFY] {name} TIMEOUT: no PARAM_VALUE received")
            return False

        except Exception as e:
            print(f"  [SITL_WEATHER_VERIFY] {name} ERROR: {e}")
            return False

    # =========================================================================
    # Phase 0: Hard reset SIM_WIND_* params to zero before applying new weather
    # =========================================================================
    # This eliminates stale wind from previous runs that could cause mismatches.
    if verbose:
        print("  [SITL_WEATHER_SYNC] Phase 0: Resetting SIM_WIND_* to zero...")

    for name in CORE_WIND_PARAMS:
        if not _set_param(name, 0.0):
            success = False
        time.sleep(0.1)  # Small delay between sets

    # Wait for reset to take effect
    time.sleep(0.3)

    # =========================================================================
    # Phase 1: Apply actual weather values from WeatherModel
    # =========================================================================
    if verbose:
        print("  [SITL_WEATHER_SYNC] Phase 1: Applying weather parameters...")

    for name, value in sim_params.items():
        # Only send real ArduPilot parameters (ignore private/internal keys starting with "_")
        if not name.startswith("SIM_"):
            continue

        if _set_param(name, value):
            if verbose:
                print(f"  [SITL_WEATHER_SYNC] Set {name} = {value:.2f}")
        else:
            success = False

        time.sleep(0.2)  # Delay between sets for SITL to process

    # Wait for all parameters to settle
    #Increased from 0.5s to 1.5s for direct SITL connection
    # SITL needs more time to process PARAM_SET via secondary serial port
    time.sleep(1.5)

    # =========================================================================
    # Phase 2: Verify each core parameter with PARAM_REQUEST_READ readback
    # =========================================================================
    #Skip verification for unidirectional (udpout:) connections
    # since they cannot receive PARAM_VALUE responses.
    if is_unidirectional:
        if verbose:
            print("  [SITL_WEATHER_SYNC] Phase 2: Skipped (udpout: cannot receive)")
            print("  [SITL_WEATHER_SYNC] Params were sent - assuming success")
        # Return success since PARAM_SET was sent (we just can't verify)
        return success

    # Bidirectional connection - perform full verification
    if verbose:
        print("  [SITL_WEATHER_SYNC] Phase 2: Verifying parameters...")

    for name in CORE_WIND_PARAMS:
        expected = sim_params.get(name, 0.0)
        if not _verify_param(name, expected):
            any_mismatch = True

    # High-level warning if any mismatch
    if any_mismatch:
        print("")
        print("  " + "=" * 60)
        print("  [SITL_WEATHER_VERIFY] WARNING: One or more SIM_WIND_* params")
        print("  did not match requested values - SITL atmosphere may differ.")
        print("  " + "=" * 60)
        print("")
        success = False

    return success


def print_sitl_weather_summary(weather: WeatherModel, sitl_sysid: int = 1):
    """
    Print a clear summary of the SITL weather configuration for debugging.

    Args:
        weather: WeatherModel instance.
        sitl_sysid: Target system ID (for display purposes).
    """
    sim_params = weather.to_ardupilot_sim_params()

    print("\n" + "=" * 60)
    print(f"SITL Atmosphere (QuadPlane sysid={sitl_sysid})")
    print("=" * 60)
    print(f"  SIM_WIND_SPD  = {sim_params.get('SIM_WIND_SPD', 0.0):.1f} m/s")
    print(
        f"  SIM_WIND_DIR  = {sim_params.get('SIM_WIND_DIR', 0.0):.0f} deg (direction FROM which wind blows)"
    )
    print(f"  SIM_WIND_TURB = {sim_params.get('SIM_WIND_TURB', 0.0):.2f}")

    # Show internal/custom params if present
    if "_GUST_AMP_M_S" in sim_params and sim_params["_GUST_AMP_M_S"] > 0:
        print(f"  (Leader-only) Gust amplitude = {sim_params['_GUST_AMP_M_S']:.1f} m/s")
        print(f"  (Leader-only) Gust period    = {sim_params.get('_GUST_PERIOD_S', 30.0):.0f} s")
        print("  NOTE: Gusts apply only to DHC-2 leader (no ArduPilot equivalent)")

    # Compute expected effect
    wind_spd = sim_params.get("SIM_WIND_SPD", 0.0)
    wind_dir = sim_params.get("SIM_WIND_DIR", 0.0)
    if wind_spd > 0.01:
        # Effect on north/south legs
        # Wind from north (dir=0): headwind on north leg, tailwind on south leg
        # Wind from east (dir=90): crosswind on both legs
        import math

        theta = math.radians(wind_dir)
        headwind_north = wind_spd * math.cos(theta)  # Positive = headwind on north leg
        crosswind = wind_spd * math.sin(theta)

        print(f"\n  Expected effects on racetrack legs:")
        north_sign = "+" if headwind_north > 0 else ""
        print(
            f"    North leg: {north_sign}{-headwind_north:.1f} m/s GS delta (headwind: {headwind_north:.1f})"
        )
        print(
            f"    South leg: {'+' if headwind_north < 0 else ''}{headwind_north:.1f} m/s groundspeed delta"
        )
        if abs(crosswind) > 0.1:
            print(
                f"    Crosswind component: {abs(crosswind):.1f} m/s from {'east' if crosswind > 0 else 'west'}"
            )
    else:
        print("\n  Calm conditions - no wind effect on groundspeed")

    print("=" * 60 + "\n")


# =============================================================================
# Main Leader Broadcast Script
# =============================================================================


# ---------------------------------------------------------------------------
# #492 FD_LEADER_CONT_CLOCK (2026-07-20): leader-local interpolated sim clock.
#   WHY: the leader's sim clock is ZOH-held from the FOLLOWER's ATTITUDE tbms
#   (update_sim_time -> time_auth.now_s(), no interpolation), so the leader's
#   trajectory integration AND its broadcast stamp advance in ~50 ms lattice
#   jumps — the synthetic leader "teleports" v*50ms (~1.235 m at cruise) per
#   step while raw truth is continuous (#490 A6; grid census #489). This helper
#   interpolates BETWEEN ATTITUDE arrivals using the wall time elapsed since the
#   last accepted update, scaled by the expected speedup, capped at one nominal
#   ATTITUDE interval so a stalled feed fails SAFE back to today's ZOH value.
#   Pure function — unit-testable without MAVLink. Gated by FD_LEADER_CONT_CLOCK
#   ('1' to enable); unset => caller never invokes it => byte-identical.
# ---------------------------------------------------------------------------
CONT_CLOCK_CAP_S = 0.050  # one nominal ATTITUDE interval; stall => ZOH fallback


# ---------------------------------------------------------------------------
# T-186 ARM B-0: broadcaster BOOT-DRAW
# observability. WHY: the T-93 latch is decided per-boot, but the banked A/B
# legs never captured the broadcaster's init records (the spawner does not
# sink this process's stdout — the T-185 declared-sink lesson). This probe
# SELF-SINKS: when FD_LEADER_BOOT_PROBE=1, init/boot-draw records append to
# <FD_LEADER_PROBE_DIR>/leader_boot_<pid>.log (identifying header = argv, so
# the lane mapping is by connection string, never by mtime). Flag unset =>
# no file, no lines, byte-identical behavior. Observability ONLY — no
# behavior change on any path. Budget-capped like the T-165 epoch probe.
# ---------------------------------------------------------------------------
_BOOT_PROBE_BUDGET = 40


def boot_probe_line(tag, tbms_int, extra=""):
    """Pure formatter for one boot-draw record (unit-tested without MAVLink).

    parity50/parity100 = the tick / broadcast-grid phase draws under test
    (grid parity separated 10/12 control legs with 2 counterexamples —
    MEASURED candidate, not an assumed mechanism)."""
    t = int(tbms_int)
    return (f"[BOOT_PROBE] {tag} tbms={t} parity50={t % 50} "
            f"parity100={t % 100}{extra}")


class BootProbeSink:
    """Lazy append-only sink for boot-probe lines; also mirrors to stdout.

    enabled=False (the shipped default) => emit() is a no-op and no file is
    ever created. The first emit writes an identifying header (argv) so the
    probe file maps to its lane by connection string."""

    def __init__(self, enabled, probe_dir, header, budget=_BOOT_PROBE_BUDGET):
        self.enabled = bool(enabled)
        self._dir = probe_dir or "."
        self._header = header
        self._budget = int(budget)
        self._fh = None

    def emit(self, line):
        if not self.enabled or self._budget <= 0:
            return
        self._budget -= 1
        try:
            if self._fh is None:
                path = os.path.join(
                    self._dir, f"leader_boot_{os.getpid()}.log")
                self._fh = open(path, "a")
                self._fh.write(f"[BOOT_PROBE] header {self._header}\n")
            self._fh.write(line + "\n")
            self._fh.flush()
            print(line)
        except OSError:
            pass  # observability must never take down the broadcaster


def interp_sim_time(zoh_sim_s, wall_now_s, wall_last_update_s, speedup,
                    cap_s=CONT_CLOCK_CAP_S):
    """ZOH sim time + capped wall-elapsed extrapolation since the last update.

    Returns ``zoh_sim_s + min(max(wall_now - wall_last, 0) * speedup, cap_s)``.
    Negative wall deltas (clock weirdness) contribute 0. speedup <= 0 is treated
    as 1.0 (fail-safe: never freezes the clock below ZOH). Monotonicity across
    ATTITUDE arrivals holds because the next ZOH value >= previous ZOH + one
    interval >= previous interp (cap == interval); integrate_sim_dt's dt<=0
    guard remains the runtime backstop.
    """
    if speedup is None or speedup <= 0:
        speedup = 1.0
    ahead = max(wall_now_s - wall_last_update_s, 0.0) * speedup
    return zoh_sim_s + min(ahead, cap_s)


def integrate_sim_dt(current_sim_time, last_integrated_sim_time):
    """Position-integration dt on POSITIVE sim-time advance only (RCA a515e577).

    Returns (dt, new_last_integrated). Stale / non-advancing polls
    (current_sim_time <= last_integrated_sim_time) return dt=0.0 and DO NOT
    advance the anchor, so the integrated dt telescopes exactly to
    (sim_end - sim_start) with no phantom distance from a non-positive-dt floor.
    No upper clamp: under JSON-lockstep dt is small, and faithful integration
    must not discard a genuine sim-time advance. dt is only ever a multiplier
    (position / heading / air-distance), never a divisor, so dt=0.0 is safe.
    """
    if current_sim_time <= last_integrated_sim_time:
        return 0.0, last_integrated_sim_time
    return current_sim_time - last_integrated_sim_time, current_sim_time


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="DHC-2 Beaver Lead Aircraft Broadcaster with Physical Weather",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calm conditions (original behavior)
  python3 leader_broadcast_fixed.py

  # Light tailwind from north (wind blows FROM north = toward south)
  python3 leader_broadcast_fixed.py --wind-speed 2.0 --wind-dir 0

  # Crosswind from east with turbulence
  python3 leader_broadcast_fixed.py --wind-speed 3.0 --wind-dir 90 --turbulence 0.3

  # Full weather with reproducible seed for DRL
  python3 leader_broadcast_fixed.py --wind-speed 2.0 --wind-dir 45 \\
      --turbulence 0.2 --gust-amplitude 1.0 --weather-seed 42
        """,
    )

    # Aircraft parameters
    parser.add_argument(
        "--target-speed",
        type=float,
        default=24.7,
        help="Lead aircraft TRUE AIRSPEED (TAS) in m/s (default: 24.7). "
        "This is the actual speed through the air mass. "
        "FlightDock research uses TAS exclusively for simplicity.",
    )
    # Legacy --connection argument (deprecated, use --udp-out instead)
    parser.add_argument(
        "--connection",
        type=str,
        default=None,
        help="[DEPRECATED] Use --udp-out instead. MAVLink UDP output.",
    )
    parser.add_argument(
        "--start-offset",
        type=float,
        default=-1.0,
        help="Initial position offset along pattern in meters. "
        "If negative (default), auto-start halfway down SOUTH_LEG heading south. "
        "If positive, explicit offset override in meters.",
    )

    # =========================================================================
    #Leader Profile Selection (diagnostic modes)
    # =========================================================================
    # --profile controls the leader trajectory pattern:
    #   racetrack: Default obround pattern (straight legs + semicircular turns)
    #   straight: Constant heading, constant speed (no turns) - for baseline testing
    #   circle: Constant-radius turn at fixed yaw rate - for turn-following validation
    # =========================================================================
    parser.add_argument(
        "--profile",
        type=str,
        default="racetrack",
        choices=["racetrack", "straight", "circle"],
        help="Leader trajectory profile: racetrack (default), straight, or circle",
    )
    parser.add_argument(
        "--hdg",
        type=float,
        default=180.0,
        help="Initial/constant heading in degrees (default: 180 = SOUTH, "
        "toward the follower's airfield). Used by straight and circle "
        "profiles. T-478 (2026-08-13): the default was 0.0 = NORTH, which "
        "pointed the leader AWAY from the follower on every caller that did "
        "not pass --hdg explicitly. Two layers already defended against that "
        "(config.JSON_LEADER_HDG_DEG passed explicitly by the env, and the "
        "TEST-331 auto-rule below), but a DEFAULT that encodes the wrong "
        "answer is a defect in its own right: T-383b measured the defect "
        "RECURRING 2026-07-21..08-09 via a path that bypassed the auto-rule "
        "(warmup 501 s vs 246.5 s on straight legs). The safe heading is now "
        "the default, so a new call site inherits it rather than the hazard.",
    )
    parser.add_argument(
        "--epoch-stamp-out",
        type=str,
        default="",
        help="T-392a: host:port for the epoch-stamp side channel (plain UDP, "
        "additive, empty = disabled). Each position packet is mirrored as "
        "'ES <wire_tbms> <raw_attitude_latch_tbms>' so the consumer can align "
        "on the ATTITUDE time_boot_ms this broadcaster ACTUALLY latched "
        "instead of re-deriving it from the wire stamp (which is a ZOH of the "
        "latch plus a monotonic clamp). Sends nothing when empty or when "
        "FD_EPOCH_STAMP=0.",
    )
    parser.add_argument(
        "--omega-deg-s",
        type=float,
        default=1.5,
        help="Yaw rate in deg/s for circle profile (default: 1.5). "
        "Positive = right turn (CW from above), negative = left turn (CCW).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="CCW",
        choices=["CW", "CCW"],
        help="Turn direction for circle profile: CW (right) or CCW (left, default)",
    )

    # Dual-link arguments for UDP + TCP
    parser.add_argument(
        "--udp-out",
        type=str,
        default="127.0.0.1:14550",
        help="UDP output endpoint HOST:PORT for MAVProxy ingest " "(default: 127.0.0.1:14550)",
    )
    parser.add_argument(
        "--sitl-tcp",
        type=str,
        default="127.0.0.1:5763",
        help="Direct SITL TCP endpoint HOST:PORT for ArduPlane "
        "(default: 127.0.0.1:5763). Required for C++ "
        "FormationController to receive lead telemetry.",
    )
    parser.add_argument(
        "--rate-hz", type=float, default=10.0, help="Telemetry broadcast rate in Hz (default: 10)"
    )

    # Weather parameters (align with ArduPilot SIM_WIND_* where possible)
    parser.add_argument(
        "--wind-speed",
        type=float,
        default=0.0,
        help="Base wind speed in m/s (default: 0.0 = calm). Maps to SIM_WIND_SPD.",
    )
    parser.add_argument(
        "--wind-dir",
        type=float,
        default=0.0,
        help="Wind direction FROM which wind blows in degrees (default: 0 = from north). "
        "Maps to SIM_WIND_DIR. 0=N, 90=E, 180=S, 270=W.",
    )
    parser.add_argument(
        "--turbulence",
        type=float,
        default=0.0,
        help="Turbulence intensity 0.0-1.0 (default: 0.0 = none). Maps to SIM_WIND_TURB.",
    )
    parser.add_argument(
        "--gust-amplitude",
        type=float,
        default=0.0,
        help="Gust amplitude in m/s above/below base wind (default: 0.0)",
    )
    parser.add_argument(
        "--gust-period",
        type=float,
        default=30.0,
        help="Gust oscillation period in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--weather-seed",
        type=int,
        default=None,
        help="Random seed for reproducible weather (default: None = random)",
    )

    # SITL weather sync parameters
    parser.add_argument(
        "--no-sitl-weather-sync",
        action="store_true",
        help="Disable syncing WeatherModel to ArduPilot SITL SIM_* wind/turbulence params. "
        "By default, weather is synced to ensure QuadPlane and DHC-2 share the same atmosphere.",
    )
    parser.add_argument(
        "--sitl-sysid",
        type=int,
        default=1,
        help="Target system ID for the QuadPlane SITL (default: 1)",
    )
    parser.add_argument(
        "--sitl-compid",
        type=int,
        default=1,
        help="Target component ID for the QuadPlane SITL (default: 1)",
    )
    #Dedicated SITL port for weather sync (bypasses MAVProxy)
    parser.add_argument(
        "--sitl-weather-port",
        type=int,
        default=5763,
        help="Direct SITL TCP port for weather sync (default: 5763). "
        "This bypasses MAVProxy for reliable PARAM_SET delivery.",
    )

    #Roll sign convention and debug flags
    parser.add_argument(
        "--bank-deg",
        type=float,
        default=None,
        help="Override bank angle in degrees for debug (default: 3.0)",
    )
    parser.add_argument(
        "--roll-sign-left",
        type=float,
        default=-1.0,
        help="Roll sign for LEFT turns: -1.0 = negative roll (default), "
        "+1.0 = positive roll. NED+FRD convention: left bank = negative.",
    )

    # TEST 328/329: Sim-time synchronization with sysid filtering
    parser.add_argument(
        "--time-sync-conn",
        type=str,
        default=None,
        help="MAVLink connection to follower for time sync (e.g., "
        '"udpin:127.0.0.1:14553"). If set, leader trajectory updates '
        "use sim-time from follower time_boot_ms instead of wall time.",
    )
    parser.add_argument(
        "--time-sync-source-sysid",
        type=int,
        default=1,
        help="Only accept time_boot_ms from this sysid (default: 1 = "
        "QuadPlane follower). Prevents locking onto leader sysid=2.",
    )
    parser.add_argument(
        "--time-sync-source-compid",
        type=int,
        default=None,
        help="Optional: Only accept time_boot_ms from this compid. "
        "If not set, accepts any compid from the filtered sysid.",
    )
    parser.add_argument(
        "--sim-speedup",
        type=float,
        default=1.0,
        help="Expected SIM_SPEEDUP value (default: 1.0). Used for timing "
        "validation and rate diagnostics.",
    )
    parser.add_argument(
        "--require-time-sync",
        action="store_true",
        default=True,
        help="If set (default), exit non-zero if time-sync fails within timeout. "
        "Use --no-require-time-sync to allow wall-time fallback.",
    )
    parser.add_argument(
        "--no-require-time-sync",
        dest="require_time_sync",
        action="store_false",
        help="Allow wall-time fallback if time-sync fails.",
    )

    args = parser.parse_args()

    # Configuration - TAS-only speed reference
    # NOTE: args.target_speed IS the True Airspeed (TAS) directly.
    # FlightDock research uses TAS exclusively - no EAS conversion needed.

    # Dual-link connection strings
    # - Legacy --connection overrides --udp-out if specified (backwards compat)
    # - UDP link: for MAVProxy monitoring
    # - TCP link: for direct SITL ArduPlane (required for C++ FormationController)
    if args.connection is not None:
        # Legacy mode: use --connection as UDP output
        UDP_CONNECTION = args.connection
        print("[WARN] --connection is deprecated; use --udp-out instead")
    else:
        UDP_CONNECTION = f"udpout:{args.udp_out}"
    TCP_CONNECTION = f"tcp:{args.sitl_tcp}"
    BROADCAST_RATE_HZ = args.rate_hz
    # LB-2: Use sim-time interval (not wall-scaled). Broadcasts trigger when
    # sim-time advances by BROADCAST_INTERVAL_SIM seconds, naturally adapting
    # if the sim can't sustain the requested speedup.
    BROADCAST_INTERVAL_SIM = 1.0 / BROADCAST_RATE_HZ
    # LB-5: Wall-time flood guard to prevent link saturation at high speedups.
    # P0.4a: scale the wall cap by sim speedup —
    # the guarded quantity is per-SIM-second link load, so a fixed 100 Hz wall
    # cap silently starved the 10 Hz sim feed at S>=10 (min(10,100/S) sim-Hz,
    # zero margin at S=10; skipped sends are LOST, not deferred). At S=1 the
    # product is exactly 100.0 => behavior-identical to the pre-fix constant.
    MAX_SEND_HZ_WALL = 100.0 * args.sim_speedup
    MIN_SEND_DT_WALL = 1.0 / MAX_SEND_HZ_WALL
    # LB-6: Flood guard diagnostics
    flood_skip_pos = 0
    flood_skip_att = 0
    last_flood_report = 0.0
    FLOOD_REPORT_INTERVAL = 5.0  # Report every 5 wall-seconds
    # LB-6: Speedup-aware loop pacing
    BASE_SLEEP = 0.01  # 100Hz at 1x
    MIN_SLEEP = 0.005  # Cap at 200Hz wall-time
    scaled_sleep = max(BASE_SLEEP / args.sim_speedup, MIN_SLEEP)
    print(
        f"[BROADCAST] rate={BROADCAST_RATE_HZ:.0f}Hz sim-time "
        f"(interval={BROADCAST_INTERVAL_SIM*1000:.1f}ms sim, "
        f"max {MAX_SEND_HZ_WALL:.0f}Hz wall)"
    )
    print(
        f"[LB-6] Loop pacing: {1/scaled_sleep:.0f}Hz wall "
        f"(scaled from {1/BASE_SLEEP:.0f}Hz at 1x, cap {1/MIN_SLEEP:.0f}Hz)"
    )

    # Sim-time authority (LB-1: Use SimTimeAuthority for consistent timing)
    time_auth = SimTimeAuthority(expected_speedup=args.sim_speedup, drl_mode=False, verbose=False)

    # T-186 ARM B-0: boot-draw probe sink (flag-gated, default OFF => no-op)
    boot_probe = BootProbeSink(
        enabled=os.environ.get("FD_LEADER_BOOT_PROBE", "0").strip() == "1",
        probe_dir=os.environ.get("FD_LEADER_PROBE_DIR", ""),
        header=" ".join(sys.argv[1:]))

    # Home position (same as QuadPlane)
    HOME_LAT = -35.363261
    HOME_LON = 149.165230
    HOME_ALT_MSL = 584.0
    TARGET_ALT_MSL = 900.0
    TARGET_ALT_AGL = TARGET_ALT_MSL - HOME_ALT_MSL

    # TAS-only configuration (2025-12-08 update)
    # --target-speed specifies TAS directly; EAS is computed for reference only
    LEAD_AIRSPEED_TAS = args.target_speed  # True Airspeed (what we actually fly)
    EAS2TAS_FACTOR = get_eas2tas_factor(TARGET_ALT_MSL)
    LEAD_AIRSPEED_EAS = LEAD_AIRSPEED_TAS / EAS2TAS_FACTOR  # Reference only
    AIR_DENSITY_AT_ALT = get_air_density_isa(TARGET_ALT_MSL)

    # Create weather model
    weather = WeatherModel(
        base_wind_speed_m_s=args.wind_speed,
        base_wind_dir_deg=args.wind_dir,
        turbulence_intensity=args.turbulence,
        gust_amplitude_m_s=args.gust_amplitude,
        gust_period_s=args.gust_period,
        seed=args.weather_seed,
    )

    # =========================================================================
    # DUAL-LINK ARCHITECTURE
    # =========================================================================
    # Create TWO MAVLink connections for lead telemetry broadcast:
    # 1. UDP link → MAVProxy (for monitoring, GCS forwarding)
    # 2. TCP link → Direct to SITL SERIAL2 (for ArduPlane FormationController)
    #
    # Why dual links? MAVProxy does NOT forward GLOBAL_POSITION_INT from
    # external UDP sources to SITL by default. The C++ FormationController
    # in ArduPlane needs to receive lead position directly via SERIAL2.
    # =========================================================================
    print("=" * 70)
    print("FlightDock Phase 7C - DHC-2 Beaver with Physical Weather Model")
    print("=" * 70)
    print("\n[1/6] Establishing DUAL-LINK connections (Fix B Test 230)...")
    print("  System ID: 2 (DHC-2 Beaver lead)")
    print("  Component ID: 1 (Autopilot)")
    print(f"  Broadcast rate: {BROADCAST_RATE_HZ:.0f} Hz")

    # Link 1: UDP output to MAVProxy
    print(f"\n  [Link 1] UDP → MAVProxy: {UDP_CONNECTION}")
    mav_udp = mavutil.mavlink_connection(UDP_CONNECTION, source_system=2, source_component=1)
    print("    ✓ UDP link initialized")

    # T-392a: epoch-stamp side channel (additive; the MAVLink wire above is
    # untouched). It exists because the env's leader consumer lives inside
    # the R7.1-frozen controller and cannot be taught a new message type.
    _estamp_open(getattr(args, "epoch_stamp_out", "") or "")
    if _ESTAMP_SOCK is not None:
        print(f"    ✓ epoch-stamp side channel → {_ESTAMP_ADDR}")

    # Link 2: TCP direct to SITL (with retry logic)
    print(f"  [Link 2] TCP → SITL direct: {TCP_CONNECTION}")
    mav_tcp = None
    tcp_connected = False
    tcp_retry_interval = 5.0  # seconds between reconnect attempts
    tcp_last_warn_time = 0.0

    def try_connect_tcp():
        """Attempt TCP connection with error handling. Returns connection or None."""
        nonlocal mav_tcp, tcp_connected
        try:
            mav_tcp = mavutil.mavlink_connection(
                TCP_CONNECTION, source_system=2, source_component=1
            )
            # Wait for heartbeat from SITL to confirm connection
            mav_tcp.wait_heartbeat(timeout=3.0)
            tcp_connected = True
            return mav_tcp
        except Exception as e:
            tcp_connected = False
            return None

    # Initial TCP connection attempt
    if try_connect_tcp():
        print(f"    ✓ TCP link established (SITL sysid={mav_tcp.target_system})")
    else:
        print("    ⚠ TCP link failed - will retry in background")
        print("      (Make sure SITL is running with SERIAL2 on tcp:5763)")

    # Send initial heartbeats on both links
    print("\n  Sending initial heartbeats...")
    for link_name, link in [("UDP", mav_udp), ("TCP", mav_tcp)]:
        if link is not None:
            try:
                link.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_FIXED_WING,
                    mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    0,
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                )
            except Exception:
                pass
    time.sleep(0.5)
    print("  ✓ Both links active" if tcp_connected else "  ✓ UDP link active")

    # =========================================================================
    # TEST 328/329: TIME SYNC CONNECTION with SYSID FILTER + MONOTONIC GUARD
    # =========================================================================
    # If --time-sync-conn is specified, connect to follower to receive time_boot_ms.
    # This ensures leader trajectory updates use sim-time instead of wall time.
    #
    # CRITICAL FIX: Filter by sysid to prevent locking onto:
    #   - Our own sysid=2 messages looped back by MAVProxy
    #   - Any other sysid that isn't the follower
    #
    # The filter accepts ONLY messages matching --time-sync-source-sysid (default=1).
    # =========================================================================
    mav_time_sync = None
    sim_time_s: float = 0.0
    sim_time_ready: bool = False
    last_time_boot_ms_valid: int = 0  # For monotonic enforcement
    last_time_boot_ms_raw: Optional[int] = None  # Last raw value received
    last_time_update_wall: float = time.time()  # Wall time of last update
    sim_time_stale_warn_time: float = 0.0
    time_sync_health_log_time: float = 0.0
    time_sync_drop_count: int = 0  # Non-monotonic drops
    time_sync_accept_count: int = 0  # Accepted updates
    time_sync_filter_sysid: int = args.time_sync_source_sysid
    time_sync_filter_compid: Optional[int] = args.time_sync_source_compid
    time_sync_mode: str = "WALL"  # "SIM" or "WALL"
    time_sync_wall_start: float = time.time()  # Wall-time anchor for fallback

    # =========================================================================
    # TEST 333: SETTLE GATE for time_boot_ms stability
    # =========================================================================
    # Before broadcasting, ensure time_boot_ms is stable (consecutive monotonic
    # updates). This prevents wall-time-based timestamps at startup.
    # =========================================================================
    TIME_SYNC_SETTLE_MIN_SAMPLES: int = 5  # Min consecutive monotonic samples
    time_sync_settle_count: int = 0  # Consecutive monotonic samples
    time_sync_settled: bool = False  # True once settle gate passes

    def _get_msg_sysid(msg) -> int:
        """Safely get source system ID from MAVLink message."""
        if hasattr(msg, "get_srcSystem"):
            return msg.get_srcSystem()
        elif hasattr(msg, "_header"):
            return msg._header.srcSystem
        return 0

    def _get_msg_compid(msg) -> int:
        """Safely get source component ID from MAVLink message."""
        if hasattr(msg, "get_srcComponent"):
            return msg.get_srcComponent()
        elif hasattr(msg, "_header"):
            return msg._header.srcComponent
        return 0

    def update_sim_time() -> float:
        """Poll for time_boot_ms and update sim_time_s. Returns current sim time.

        SYSID FILTER: Only accepts messages from time_sync_filter_sysid.
        MONOTONIC: Rejects time_boot_ms that would go backwards.
        SETTLE: Tracks consecutive monotonic samples for settle gate.
        """
        nonlocal sim_time_s, sim_time_ready, last_time_boot_ms_valid
        nonlocal last_time_boot_ms_raw, last_time_update_wall
        nonlocal time_sync_drop_count, time_sync_accept_count, time_sync_mode
        nonlocal time_sync_settle_count, time_sync_settled
        nonlocal sim_time_stale_warn_time

        if mav_time_sync is None:
            # No time-sync connection: use wall-time from internal anchor
            return time.time() - time_sync_wall_start

        # Poll for ATTITUDE (non-blocking, drain queue)
        # FIX: Use ATTITUDE instead of GLOBAL_POSITION_INT for time_boot_ms.
        # ATTITUDE is sent at 50Hz+ (vs 4-10Hz for GPI), preventing staleness at 5x speedup.
        try:
            while True:
                msg = mav_time_sync.recv_match(type="ATTITUDE", blocking=False)
                if msg is None:
                    break

                # === SYSID FILTER ===
                src_sysid = _get_msg_sysid(msg)
                if src_sysid != time_sync_filter_sysid:
                    # Ignore wrong sysid (e.g., our own sysid=2 looped back)
                    continue

                # === COMPID FILTER (optional) ===
                if time_sync_filter_compid is not None:
                    src_compid = _get_msg_compid(msg)
                    if src_compid != time_sync_filter_compid:
                        continue

                # === VALID time_boot_ms CHECK ===
                if not hasattr(msg, "time_boot_ms") or msg.time_boot_ms <= 0:
                    continue

                candidate_tbms = msg.time_boot_ms
                last_time_boot_ms_raw = candidate_tbms

                # === MONOTONIC ENFORCEMENT ===
                if candidate_tbms < last_time_boot_ms_valid:
                    time_sync_drop_count += 1
                    time_sync_settle_count = 0  # Reset settle on non-monotonic
                    # Rate-limited warning (max 1 per second)
                    wall_now = time.time()
                    if wall_now - sim_time_stale_warn_time > 1.0:
                        sim_time_stale_warn_time = wall_now
                        print(
                            f"[TIME_SYNC] DROP non-monotonic: "
                            f"new={candidate_tbms}ms < last={last_time_boot_ms_valid}ms "
                            f"(drops={time_sync_drop_count})"
                        )
                    continue

                # === ACCEPT UPDATE ===
                last_time_boot_ms_valid = candidate_tbms
                sim_time_s = candidate_tbms / 1000.0
                last_time_update_wall = time.time()
                time_sync_accept_count += 1
                # LB-1: Feed to SimTimeAuthority
                time_auth.update_from_mavlink(candidate_tbms)

                # === SETTLE GATE TRACKING ===
                time_sync_settle_count += 1
                if not time_sync_settled:
                    if time_sync_settle_count >= TIME_SYNC_SETTLE_MIN_SAMPLES:
                        time_sync_settled = True
                        print(
                            f"[TIME_SYNC] SETTLE GATE PASSED: "
                            f"{time_sync_settle_count} consecutive monotonic "
                            f"tbms={candidate_tbms}ms"
                        )
                        # T-186 ARM B-0: settle draw
                        boot_probe.emit(boot_probe_line(
                            "settled", candidate_tbms,
                            f" count={time_sync_settle_count}"))

                if not sim_time_ready:
                    sim_time_ready = True
                    time_sync_mode = "SIM"
                    print(
                        f"[TIME_SYNC] LOCKED src_sysid={src_sysid} "
                        f"time_boot_ms={candidate_tbms}ms "
                        f"sim_speedup_expected={args.sim_speedup}x"
                    )
                    # T-186 ARM B-0: first-accepted-sample draw
                    boot_probe.emit(boot_probe_line("locked", candidate_tbms))

        except Exception as e:
            print(f"[TIME_SYNC] Exception in update_sim_time: {e}")

        # Return sim-time if ready, else wall-time fallback
        if sim_time_ready:
            return sim_time_s
        else:
            # Wall-time fallback (from internal anchor, not global time.time())
            wall_elapsed = time.time() - time_sync_wall_start
            return wall_elapsed

    def log_time_sync_health():
        """Log time-sync health status (rate-limited to every 5 wall-seconds)."""
        nonlocal time_sync_health_log_time
        wall_now = time.time()
        if wall_now - time_sync_health_log_time < 5.0:
            return
        time_sync_health_log_time = wall_now

        if mav_time_sync is None:
            return  # No time-sync connection

        stale_age = wall_now - last_time_update_wall
        wall_elapsed = wall_now - time_sync_wall_start

        if sim_time_ready and wall_elapsed > 1.0:
            speedup_est = sim_time_s / wall_elapsed
        else:
            speedup_est = 0.0

        print(
            f"[TIME_SYNC_HEALTH] mode={time_sync_mode} "
            f"src_sysid={time_sync_filter_sysid} "
            f"last_tbms={last_time_boot_ms_valid}ms "
            f"sim_t={sim_time_s:.1f}s "
            f"speedup_est={speedup_est:.2f}x "
            f"stale={stale_age:.1f}s "
            f"accept={time_sync_accept_count} drop={time_sync_drop_count}"
        )

    if args.time_sync_conn:
        print(f"\n  [Link 3] Time-sync → Follower: {args.time_sync_conn}")
        print(
            f"    Sysid filter: {time_sync_filter_sysid} "
            f"(compid: {time_sync_filter_compid or 'any'})"
        )
        try:
            mav_time_sync = mavutil.mavlink_connection(
                args.time_sync_conn, source_system=2, source_component=1
            )
            print("    ✓ Time-sync link initialized")
            print(f"    Expected SIM_SPEEDUP: {args.sim_speedup:.1f}x")
            print(
                f"    Waiting for settle gate ({TIME_SYNC_SETTLE_MIN_SAMPLES} "
                f"consecutive monotonic samples)..."
            )

            # Wait up to 15 wall-seconds for settle gate
            wait_start = time.time()
            wait_timeout = 15.0
            while time.time() - wait_start < wait_timeout:
                update_sim_time()  # Still poll MAVLink to feed time_auth
                test_time = time_auth.now_s()  # LB-1: Use authority for time
                # Wait for settle gate (not just first message)
                if time_sync_settled:
                    print(
                        f"    ✓ Time-sync READY: settle gate passed in "
                        f"{time.time() - wait_start:.1f}s "
                        f"(tbms={last_time_boot_ms_valid}ms)"
                    )
                    break
                elif sim_time_ready and not time_sync_settled:
                    # Progress: have first message, waiting for settle
                    pass  # settle gate logging handled in update_sim_time()
                time.sleep(0.05)  # Poll faster for settle
            else:
                # Timeout - check if we at least got first message
                if sim_time_ready:
                    print(
                        f"    ⚠ Settle gate timeout ({wait_timeout}s) but "
                        f"sim_time_ready=True - continuing with partial sync"
                    )
                    print(
                        f"    Settle count: {time_sync_settle_count} / "
                        f"{TIME_SYNC_SETTLE_MIN_SAMPLES}"
                    )
                else:
                    print(
                        f"    ✗ No time_boot_ms from sysid={time_sync_filter_sysid} "
                        f"within {wait_timeout}s"
                    )
                    # LB-4: Hard failure instead of silent wall-time fallback
                    if args.require_time_sync:
                        print("    FATAL: --require-time-sync is True, exiting.")
                        sys.exit(1)
                    else:
                        print("    Mode: WALL-TIME FALLBACK (time-sync NOT active)")
                        time_sync_mode = "WALL"

        except Exception as e:
            print(f"    ✗ Time-sync connection failed: {e}")
            # LB-4: Hard failure instead of silent wall-time fallback
            if args.require_time_sync:
                print("    FATAL: --require-time-sync is True, exiting.")
                sys.exit(1)
            else:
                print("    (Leader will use wall-time for trajectory updates)")
                mav_time_sync = None

    # Obround pattern parameters
    # NOTE: Turn radius uses TAS (actual speed through air)
    NM_TO_M = 1852.0  # Nautical mile to meters conversion
    LEG_LENGTH = 10.0 * NM_TO_M  # meters (10 NM straight legs)

    #Bank angle configurable via CLI for debug
    BANK_ANGLE_DEG_DEFAULT = 3.0  # 7C.16: Reduce to 3° for ~1.2°/s turn rate
    BANK_ANGLE_DEG = args.bank_deg if args.bank_deg is not None else BANK_ANGLE_DEG_DEFAULT

    #Roll sign convention
    # =============================================================================
    # NED + FRD Sign Convention:
    # - NED frame: North-East-Down (X=North, Y=East, Z=Down)
    # - FRD body: Forward-Right-Down (X=Forward, Y=Right, Z=Down)
    # - Roll (phi): positive = right wing down (body Y axis tilts toward NED +Z)
    # - For LEFT turn (counterclockwise from above): aircraft banks LEFT (left wing down)
    #   This means right wing UP, so roll is NEGATIVE in NED+FRD convention.
    # - ROLL_SIGN_LEFT_TURN = -1.0 means left turns produce negative roll
    # =============================================================================
    ROLL_SIGN_LEFT_TURN = args.roll_sign_left

    TURN_RADIUS = (LEAD_AIRSPEED_TAS**2) / (9.81 * math.tan(math.radians(BANK_ANGLE_DEG)))
    TURN_ARC = math.pi * TURN_RADIUS

    # Compute turn point coordinates
    R_EARTH = 6378137.0
    dlat_deg = (LEG_LENGTH / R_EARTH) * (180.0 / math.pi)
    lat_south = HOME_LAT
    lon_south = HOME_LON
    lat_north = HOME_LAT + dlat_deg
    lon_north = HOME_LON

    print("\n[2/5] Aircraft Configuration:")
    print(f"  Home position: {HOME_LAT:.6f}, {HOME_LON:.6f}")
    print(f"  Altitude: {TARGET_ALT_MSL:.1f}m MSL (Contact Position)")
    print(f"  Air density at altitude: {AIR_DENSITY_AT_ALT:.4f} kg/m³ (ISA)")
    print(f"  eas2tas factor: {EAS2TAS_FACTOR:.4f}")
    print(
        f"  *** TAS: {LEAD_AIRSPEED_TAS:.1f} m/s ({LEAD_AIRSPEED_TAS * 1.944:.1f} knots) *** [PRIMARY]"
    )
    print(
        f"      EAS: {LEAD_AIRSPEED_EAS:.1f} m/s ({LEAD_AIRSPEED_EAS * 1.944:.1f} knots) [reference only]"
    )
    print("  (Groundspeed = TAS when no wind, varies with wind)")
    print("\n  Pattern: Obround racetrack (straight legs + semicircular turns)")
    print(f"    - Straight leg length: {LEG_LENGTH:.0f}m ({LEG_LENGTH/1852:.1f} NM)")
    print(f"    - Turn radius: {TURN_RADIUS:.1f}m")
    print(f"    - Turn arc length: {TURN_ARC:.1f}m")
    print(f"    - Bank angle: {BANK_ANGLE_DEG:.1f}°")
    #Show roll sign convention
    roll_sign_desc = (
        "negative (left wing down)" if ROLL_SIGN_LEFT_TURN < 0 else "positive (right wing down)"
    )
    print(f"    - Roll sign for LEFT turn: {ROLL_SIGN_LEFT_TURN:+.1f} ({roll_sign_desc})")

    # Pattern timing (based on TAS, not ground speed)
    leg_time_air = LEG_LENGTH / LEAD_AIRSPEED_TAS
    turn_time_air = TURN_ARC / LEAD_AIRSPEED_TAS
    circuit_time_air = 2 * leg_time_air + 2 * turn_time_air
    circuit_length = 2 * LEG_LENGTH + 2 * TURN_ARC

    print(f"    - Perimeter: {circuit_length:.0f}m")
    print(
        f"    - Circuit time (airspeed-based): {circuit_time_air:.1f}s ({circuit_time_air/60:.1f} min)"
    )
    print("    NOTE: Actual circuit time varies with wind!")

    # =========================================================================
    # Auto-compute start offset if sentinel value (-1.0)
    # Default: halfway down SOUTH_LEG, heading South (180 deg)
    # =========================================================================
    auto_start_used = False
    if args.start_offset < 0.0:
        # SOUTH_LEG boundary: [LEG_LENGTH + TURN_ARC, 2*LEG_LENGTH + TURN_ARC)
        # Halfway down SOUTH_LEG = (LEG_LENGTH + TURN_ARC) + (LEG_LENGTH / 2)
        #                        = 1.5 * LEG_LENGTH + TURN_ARC
        auto_offset = 1.5 * LEG_LENGTH + TURN_ARC
        args.start_offset = auto_offset
        auto_start_used = True
        print(
            f"\n  [START_MODE] Auto: halfway SOUTH_LEG, heading=180deg, "
            f"start_offset={auto_offset:.0f}m"
        )
        # TEST 331 FIX: When using straight profile with auto-start, set heading
        # to 180° (south) to match the SOUTH_LEG position.
        # T-478 (2026-08-13) DELIBERATELY KEPT, having been proposed for
        # deletion once the default became 180.0. The deletion argument was
        # that this rule can now only fire on an EXPLICIT `--hdg 0.0` and so
        # only stomps stated intent. That is true but it is the weaker
        # consideration: this branch is layer 3 of the T-383b three-layer
        # defense, and T-383b exists because the north-heading defect ACTUALLY
        # RECURRED 2026-07-21..08-09 at measurable cost. Removing a proven
        # detector to tidy a hypothetical is the wrong trade. Explicit north on
        # an auto-start straight leg is also exactly the "tail-chase bad-start
        # class" the 2026-08-09 rules out, so being
        # overridden here is the intended outcome, not a surprise.
        # Pinned by tests/unit/drl/test_leader_spawn_default.py FM1-layer3.
        if args.profile == "straight" and args.hdg == 0.0:
            args.hdg = 180.0
            print(f"  [START_MODE] Auto-set hdg=180.0 for straight profile")

    print("\n[3/5] Weather Configuration:")
    print(f"  DHC-2 Beaver: {weather.get_description()}")

    # =========================================================================
    # SITL WEATHER SYNC: Push WeatherModel → ArduPilot SIM_* Parameters
    # =========================================================================
    # This ensures both aircraft (QuadPlane follower + DHC-2 Beaver leader)
    # experience the same atmosphere. Without this sync, the QuadPlane may
    # have different wind than the leader, causing airspeed vs groundspeed
    # mismatches that make docking control unstable.
    #
    # Uses dedicated TCP port (5763) that connects
    # directly to SITL, bypassing MAVProxy. This is necessary because MAVProxy
    # does NOT forward PARAM_SET messages from external links to SITL by default.
    # =========================================================================
    if not args.no_sitl_weather_sync:
        print("\n  Synchronizing weather to ArduPilot SITL QuadPlane...")

        #Create dedicated connection to SITL for weather sync
        # This bypasses MAVProxy, which doesn't forward PARAM_SET to SITL
        sitl_weather_conn_str = f"tcp:127.0.0.1:{args.sitl_weather_port}"
        print(f"  [7C22] Opening direct SITL connection: {sitl_weather_conn_str}")

        try:
            # Create dedicated MAVLink connection to SITL
            # Use system ID 2 (DHC-2) to match the main connection
            weather_mav = mavutil.mavlink_connection(
                sitl_weather_conn_str, source_system=2, source_component=1
            )

            # Wait for heartbeat from SITL to confirm connection
            print("  [7C22] Waiting for SITL heartbeat...")
            weather_mav.wait_heartbeat(timeout=5.0)
            print(f"  [7C22] ✓ Connected to SITL (sysid={weather_mav.target_system})")

            # Sync weather using the dedicated connection
            sync_success = sync_weather_to_sitl(
                weather_mav,
                weather,
                sitl_sysid=args.sitl_sysid,
                sitl_compid=args.sitl_compid,
                verbose=True,
                connection_string=sitl_weather_conn_str,
            )

            # Close the dedicated weather sync connection
            weather_mav.close()
            print("  [7C22] ✓ Closed dedicated SITL connection")

            if sync_success:
                print("  ✓ SITL weather sync complete")
            else:
                print("  ⚠ SITL weather sync had errors (check above)")

        except Exception as e:
            print(f"  [7C22] ❌ Failed to connect to SITL for weather sync: {e}")
            print(
                f"  [7C22]    Make sure SITL is running with --out=tcp:0.0.0.0:{args.sitl_weather_port}"
            )
            print("  ⚠ Weather sync failed - proceeding without sync")

        # Print detailed summary of synced atmosphere
        print_sitl_weather_summary(weather, sitl_sysid=args.sitl_sysid)
    else:
        print("\n  [SITL_WEATHER_SYNC] Skipped (disabled via --no-sitl-weather-sync)")
        print("  ⚠ WARNING: QuadPlane SITL may have different wind than DHC-2 Beaver!")
        print("             This can cause airspeed vs groundspeed mismatches.")

    # Compute expected ground speed for display (DHC-2 Beaver)
    # On north leg (heading 0°): air velocity = (AIRSPEED, 0, 0) in NED
    # Ground velocity = air velocity + wind
    initial_wind = weather.get_wind_ned(TARGET_ALT_AGL)
    gs_north_leg = math.sqrt((LEAD_AIRSPEED_TAS + initial_wind[0]) ** 2 + initial_wind[1] ** 2)
    gs_south_leg = math.sqrt((-LEAD_AIRSPEED_TAS + initial_wind[0]) ** 2 + initial_wind[1] ** 2)
    print(f"  DHC-2 Beaver expected ground speeds:")
    print(f"    North leg: {gs_north_leg:.1f} m/s (TAS={LEAD_AIRSPEED_TAS:.1f} m/s)")
    print(f"    South leg: {gs_south_leg:.1f} m/s (TAS={LEAD_AIRSPEED_TAS:.1f} m/s)")

    # =========================================================================
    # Explicit leader atmosphere logging for cross-check with QuadPlane
    # =========================================================================
    # These [LEADER_ATMOS] lines should be compared against [ATMOS_DEBUG] lines
    # in the QuadPlane controller log to verify both aircraft share the same atmosphere.
    RHO0 = 1.225  # sea level standard density
    print("")
    print(
        f"[LEADER_ATMOS] alt_msl={TARGET_ALT_MSL:.1f}m rho={AIR_DENSITY_AT_ALT:.4f} kg/m^3 rho0={RHO0:.3f} kg/m^3"
    )
    print(
        f"[LEADER_ATMOS] TAS={LEAD_AIRSPEED_TAS:.2f} m/s [PRIMARY] | "
        f"EAS={LEAD_AIRSPEED_EAS:.2f} m/s | eas2tas={EAS2TAS_FACTOR:.4f}"
    )
    print(
        f"[LEADER_ATMOS] wind_spd={args.wind_speed:.1f} m/s "
        f"wind_dir={args.wind_dir:.0f} deg turb={args.turbulence:.2f}"
    )

    print("\n[4/5] Flying Boom:")
    print("  Stowed length: 2.4m")
    print("  Max extension: 6.1m")
    print("  Extension rate: 15%/sec")
    print("  Gimbal range: ±20° pitch/yaw")
    print("  Boom update rate: 5 Hz")

    # =========================================================================
    #Profile Configuration and Startup Log
    # =========================================================================
    profile = args.profile
    profile_hdg_deg = args.hdg
    profile_omega_deg_s = args.omega_deg_s
    if args.dir == "CW":
        profile_omega_deg_s = abs(profile_omega_deg_s)  # CW = positive yaw rate
    else:
        profile_omega_deg_s = -abs(profile_omega_deg_s)  # CCW = negative yaw rate

    # Compute circle radius if needed: R = V / omega
    if profile == "circle" and abs(profile_omega_deg_s) > 0.01:
        omega_rad_s = math.radians(profile_omega_deg_s)
        circle_radius = LEAD_AIRSPEED_TAS / abs(omega_rad_s)
    else:
        circle_radius = 0.0

    # Emit [LEAD_PROFILE] startup log (for grep validation)
    if profile == "straight":
        print(
            f"\n[LEAD_PROFILE] profile=straight hdg={profile_hdg_deg:.0f} "
            f"tas={LEAD_AIRSPEED_TAS:.1f} alt={TARGET_ALT_MSL:.0f}"
        )
    elif profile == "circle":
        dir_str = "CW" if profile_omega_deg_s > 0 else "CCW"
        print(
            f"\n[LEAD_PROFILE] profile=circle omega={abs(profile_omega_deg_s):.1f} "
            f"V={LEAD_AIRSPEED_TAS:.1f} R={circle_radius:.0f} dir={dir_str}"
        )
    else:
        print(
            f"\n[LEAD_PROFILE] profile=racetrack tas={LEAD_AIRSPEED_TAS:.1f} "
            f"alt={TARGET_ALT_MSL:.0f}"
        )

    print("\n[5/5] Broadcasting started - Press Ctrl+C to stop")
    print(f"\n{'='*70}")

    # Pattern state machine
    class PatternState:
        NORTH_LEG = 0
        NORTH_TO_SOUTH_TURN = 1
        SOUTH_LEG = 2
        SOUTH_TO_NORTH_TURN = 3

        @staticmethod
        def get_name(state):
            names = ["NORTH_LEG", "NORTH_TO_SOUTH_TURN", "SOUTH_LEG", "SOUTH_TO_NORTH_TURN"]
            return names[state] if 0 <= state < len(names) else "UNKNOWN"

    # Initialize pattern
    pattern_state = PatternState.NORTH_LEG
    # TEST 328: pattern_start_time uses sim-time (or wall-time if no sync)
    # LB-1: Use SimTimeAuthority for consistent timing
    pattern_start_time = time_auth.now_s()
    state_start_time = pattern_start_time

    # Position tracking in NED frame (meters from HOME)
    # This is the actual position, which will drift with wind
    beaver_pos_ned = np.array([0.0, TURN_RADIUS, 0.0])  # Start at (0, +R, 0) for NORTH_LEG

    # Track distance traveled along the pattern (air-relative, for state transitions)
    air_distance_in_state = 0.0

    boom_extension = 0.0
    last_heartbeat = 0
    # LB-2: Track last-send times in sim seconds (None until first send)
    last_position_sim = None
    last_attitude_sim = None
    # TEST 387: Velocity filter state to mimic EKF dynamics (reduce velocity mismatch)
    filtered_vn = None
    filtered_ve = None
    last_sent_tbms = 0  # LB-3: Monotonic clamp for outgoing time_boot_ms
    last_pos_send_wall = 0.0  # LB-5: Wall-time flood guard
    last_att_send_wall = 0.0  # LB-5: Wall-time flood guard
    last_display = 0
    last_validation = 0
    last_dt = 0.01  # Initial dt estimate

    # Track current attitude for ATTITUDE messages
    current_roll_rad = 0.0
    current_pitch_rad = 0.0
    turn_state = "STRAIGHT"  #Track turn state for status output
    # yaw is computed from heading_rad below

    # Expected state durations based on TAS (for state transitions)
    EXPECTED_STATE_DURATIONS = {
        PatternState.NORTH_LEG: LEG_LENGTH / LEAD_AIRSPEED_TAS,
        PatternState.NORTH_TO_SOUTH_TURN: math.pi * TURN_RADIUS / LEAD_AIRSPEED_TAS,
        PatternState.SOUTH_LEG: LEG_LENGTH / LEAD_AIRSPEED_TAS,
        PatternState.SOUTH_TO_NORTH_TURN: math.pi * TURN_RADIUS / LEAD_AIRSPEED_TAS,
    }

    # Apply start offset if requested
    if args.start_offset > 0.0:
        print(f"\n  Applying start offset: {args.start_offset:.0f}m along pattern")

        state_boundaries = [
            (0.0, LEG_LENGTH),
            (LEG_LENGTH, LEG_LENGTH + TURN_ARC),
            (LEG_LENGTH + TURN_ARC, 2 * LEG_LENGTH + TURN_ARC),
            (2 * LEG_LENGTH + TURN_ARC, circuit_length),
        ]

        offset_distance = args.start_offset % circuit_length
        initial_state = PatternState.NORTH_LEG
        distance_into_state = offset_distance

        for state_num, (start_dist, end_dist) in enumerate(state_boundaries):
            if start_dist <= offset_distance < end_dist:
                initial_state = state_num
                distance_into_state = offset_distance - start_dist
                break

        # Set initial position based on offset
        pattern_state = initial_state
        air_distance_in_state = distance_into_state

        # Compute initial position for each state
        if pattern_state == PatternState.NORTH_LEG:
            beaver_pos_ned = np.array([distance_into_state, TURN_RADIUS, 0.0])
        elif pattern_state == PatternState.NORTH_TO_SOUTH_TURN:
            angle = distance_into_state / TURN_RADIUS
            beaver_pos_ned = np.array(
                [LEG_LENGTH + TURN_RADIUS * math.sin(angle), TURN_RADIUS * math.cos(angle), 0.0]
            )
        elif pattern_state == PatternState.SOUTH_LEG:
            beaver_pos_ned = np.array([LEG_LENGTH - distance_into_state, -TURN_RADIUS, 0.0])
        elif pattern_state == PatternState.SOUTH_TO_NORTH_TURN:
            angle = distance_into_state / TURN_RADIUS
            beaver_pos_ned = np.array(
                [-TURN_RADIUS * math.sin(angle), -TURN_RADIUS * math.cos(angle), 0.0]
            )

        print(f"   Initial state: {PatternState.get_name(initial_state)}")
        print(f"   Distance into state: {distance_into_state:.0f}m")
        print(
            f"   Initial position NED: ({beaver_pos_ned[0]:.0f}, {beaver_pos_ned[1]:.0f}, {beaver_pos_ned[2]:.0f})"
        )

        # Print expected heading for the initial state
        if initial_state == PatternState.NORTH_LEG:
            print("   Expected heading: 0.0 deg (North)")
        elif initial_state == PatternState.SOUTH_LEG:
            print("   Expected heading: 180.0 deg (South)")
        elif initial_state == PatternState.NORTH_TO_SOUTH_TURN:
            print("   Expected heading: varying (in N->S turn)")
        elif initial_state == PatternState.SOUTH_TO_NORTH_TURN:
            print("   Expected heading: varying (in S->N turn)")

    # Telemetry sequence tracking
    position_sequence = 0
    position_sequence_10s_mark = 0
    last_sequence_report = 0

    # LB-1: Track sim-time initialization for pattern timer reset
    sim_time_initialized = False

    # DETERMINISTIC ENGAGEMENT (FD_ENGAGE_SIM_TIME_S, RCA 2026-07-03): optional fixed follower
    # sim-time (seconds) at which the trajectory epoch is captured. When set, the leader holds its
    # initial (stationary) state until the follower sim-clock crosses this threshold, so
    # pattern_start_time lands on the SAME follower frame every run — killing the wall-variable
    # settle-epoch float (0.135 s ≈ 2.7 frames measured on the PC_J1a/J1b determinism pair).
    # Unset/empty => None => OFF => init on first-ready = byte-identical prior behavior. Scoped to
    # the JSON-lockstep determinism workstream (does not affect native or wall-fallback runs).
    _engage_env = os.environ.get("FD_ENGAGE_SIM_TIME_S", "").strip()
    ENGAGE_SIM_TIME_S = float(_engage_env) if _engage_env else None
    if ENGAGE_SIM_TIME_S is not None:
        print(f"[LB-ENGAGE] Deterministic engagement armed: hold trajectory epoch until "
              f"follower sim-time >= {ENGAGE_SIM_TIME_S:.3f}s")

    # OPTION B (#308 C+B dispatch, FD_ENGAGE_EPOCH_QUANT): when the gate is armed,
    # latch the trajectory epoch to the frozen T_ENGAGE constant (already on the
    # 1 ms tbms/FDM frame grid) instead of the wall-drained crossing sample —
    # killing the ±25-50 ms SG-D4 epoch draw at its source (identical-seed runs
    # then latch a bit-identical trajectory epoch). ADOPTED default "1"
    # : SG-D4 PASSed first time ever with
    # quant ON (pair inits 76.00==76.00; epoch spread 0.000 across 8 boots;
    # quant-OFF rollback boot showed the legacy 76.03 draw persists). "0" =
    # legacy wall-latched epoch (rollback arm, log line byte-identical via
    # resolve_engage_epoch passthrough). The quant branch only runs when the
    # engage gate itself is armed (determinism campaigns); unarmed gate
    # (ENGAGE_SIM_TIME_S is None) is unaffected at either default.
    ENGAGE_EPOCH_QUANT = os.environ.get("FD_ENGAGE_EPOCH_QUANT", "1").strip() == "1"
    if ENGAGE_EPOCH_QUANT and ENGAGE_SIM_TIME_S is not None:
        print(f"[LB-ENGAGE] Epoch quantization ON: trajectory epoch snapped to "
              f"{ENGAGE_SIM_TIME_S:.3f}s (raw wall-crossing sample logged as diagnostic)")

    # RCA a515e577: positive-sim-time integration diagnostics (no phantom-dt floor)
    dt_raw = 0.0
    dt_stale_skipped = 0
    dt_integrated_sum = 0.0
    pos0_ned = None

    # Initialize boom state (set in broadcast block, used in display block)
    boom_extension = 0.0  # 0% = stowed
    boom_length = 2.4  # Stowed length in meters

    try:
        # TEST 328: Use sim-time for trajectory updates if available
        # LB-1: Use SimTimeAuthority for consistent timing
        update_sim_time()  # Poll MAVLink to feed time_auth
        last_sim_time = time_auth.now_s()
        # TEST 387: Use guarded wall-time (allowed for I/O rate limiting)
        last_wall_time = wall_now_s()  # Wall time for I/O rate limiting

        # HARDENING: Staleness threshold for tbms extrapolation
        TBMS_STALE_THRESH_S = 0.5  # 500ms staleness threshold

        # #379: sender-side transport delay line (unset/0 = inert; the banner
        # line is the config echo of record for the broadcaster side).
        tx_delay = LeaderTxDelayLine(
            float(os.environ.get("FD_LEADER_TX_DELAY_S", "0") or "0"))
        print(f"[#379] leader TX transport delay: {tx_delay.delay_s:.3f} s "
              f"({'ACTIVE' if tx_delay.active else 'inert'})")
        # #492: continuous-clock gate, read ONCE at loop entry (config echo of
        # record for the broadcaster side; per-leg provenance echoes this env).
        cont_clock = os.environ.get("FD_LEADER_CONT_CLOCK", "0").strip() == "1"
        print(f"[#492] leader continuous clock: "
              f"{'ACTIVE' if cont_clock else 'inert (ZOH)'}")

        while True:
            # Transport fix (RCA 20260627T000303Z): drain the reverse SERIAL2
            # stream so the follower's outbound telemetry cannot fill the TCP
            # recv buffer → window-0 → SERIAL2 RX starvation (root cause of the
            # per-boot lead-feed death at the DC handoff).
            if mav_tcp is not None and os.environ.get("FD_LEADER_TCP_DRAIN", "1") == "1":
                for _ in range(64):
                    if mav_tcp.recv_match(blocking=False) is None:
                        break

            # Update sim-time from follower (if connected)
            # LB-1: Use SimTimeAuthority for consistent timing
            update_sim_time()  # Poll MAVLink to feed time_auth
            current_sim_time = time_auth.now_s()
            # TEST 387: Use guarded wall-time (allowed for I/O rate limiting)
            current_wall_time = wall_now_s()
            # #492: interpolate between ATTITUDE arrivals (flag-gated; unset =>
            # this branch never runs => byte-identical ZOH behavior). Feeds BOTH
            # integrate_sim_dt and the broadcast stamp below, preserving the
            # stamp/position consistency property (#490 A3). sim_time_ready
            # guard: never interpolate the pre-lock wall fallback.
            if cont_clock and sim_time_ready:
                current_sim_time = interp_sim_time(
                    current_sim_time, time.time(), last_time_update_wall,
                    args.sim_speedup)

            # LB-1: Gate leader motion on sim-time readiness
            if not time_auth.is_ready():
                # Still allow heartbeats / logging in wall time
                time.sleep(0.01)
                continue

            # DETERMINISTIC ENGAGEMENT BARRIER (FD_ENGAGE_SIM_TIME_S): hold the leader at its
            # initial (stationary) state until the follower sim-clock crosses the fixed threshold,
            # so the trajectory epoch is captured on the same follower frame every run. Default OFF
            # (ENGAGE_SIM_TIME_S is None) => _engage_hold always False => original path byte-identical.
            _engage_hold = (ENGAGE_SIM_TIME_S is not None
                            and not sim_time_initialized
                            and current_sim_time < ENGAGE_SIM_TIME_S)
            if _engage_hold:
                # Freeze the trajectory clock: last_sim_time=now => dt=0 => beaver_pos_ned stationary;
                # pattern/state refs pinned to now => elapsed≈0 (boom stowed). The broadcast block
                # below STILL runs, so the follower keeps a fixed, deterministic leader target while
                # it settles onto it before engagement.
                pattern_start_time = current_sim_time
                state_start_time = current_sim_time
                last_sim_time = current_sim_time
                air_distance_in_state = 0.0

            # LB-1: Reset pattern timers on first ready (deferred to T_ENGAGE when armed)
            if not sim_time_initialized and not _engage_hold:
                sim_time_initialized = True
                # OPTION B: epoch = frozen T_ENGAGE (quant) or wall-latched crossing
                # (legacy). resolve_engage_epoch returns (epoch, log_suffix); default
                # path is byte-identical (epoch == current_sim_time, suffix == "").
                _epoch, _raw_suffix = resolve_engage_epoch(
                    ENGAGE_EPOCH_QUANT, ENGAGE_SIM_TIME_S, current_sim_time)
                pattern_start_time = _epoch
                state_start_time = _epoch
                last_sim_time = _epoch
                air_distance_in_state = 0.0
                pos0_ned = beaver_pos_ned.copy()  # RCA a515e577: origin for effective-speed diag
                _engage_tag = " (engage)" if ENGAGE_SIM_TIME_S is not None else ""
                print(f"[LB-1] Sim-time initialized{_engage_tag} at {_epoch:.2f}s{_raw_suffix}")

            # dt integrates position on genuine POSITIVE sim-time advance only (RCA a515e577):
            # no dt<=0 floor (was injecting phantom distance during stale polls), no >0.5 cap
            # (was discarding real advance). last_sim_time is the last INTEGRATED sim time.
            dt_raw = current_sim_time - last_sim_time
            dt, last_sim_time = integrate_sim_dt(current_sim_time, last_sim_time)
            if dt > 0.0:
                dt_integrated_sum += dt
            else:
                dt_stale_skipped += 1
            last_wall_time = current_wall_time

            # Elapsed time for logging and state transitions (sim-time)
            elapsed = current_sim_time - pattern_start_time

            # Update weather model
            weather.update(elapsed)
            wind_ned = weather.get_wind_ned(TARGET_ALT_AGL)

            # Heartbeat at 1 Hz (send on BOTH links) - uses wall time for rate limiting
            if current_wall_time - last_heartbeat >= 1.0:
                for link in [mav_udp, mav_tcp]:
                    if link is not None:
                        try:
                            link.mav.heartbeat_send(
                                mavutil.mavlink.MAV_TYPE_FIXED_WING,
                                mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                0,
                                mavutil.mavlink.MAV_STATE_ACTIVE,
                            )
                        except Exception:
                            pass  # TCP may be disconnected
                last_heartbeat = current_wall_time

                #Retry TCP connection if disconnected (every 5s)
                if (
                    not tcp_connected
                    and (current_wall_time - tcp_last_warn_time) >= tcp_retry_interval
                ):
                    tcp_last_warn_time = current_wall_time
                    if try_connect_tcp():
                        print(f"[{elapsed:.1f}s] TCP reconnected to SITL")
                    else:
                        print(f"[{elapsed:.1f}s] TCP still disconnected - retrying...")

            # =========================================================================
            # MOTION MODEL: Airspeed + Wind (Profile-aware)
            # =========================================================================
            # The DHC-2 Beaver flies at constant AIRSPEED.
            # Its ground-relative velocity is: v_ground = v_air + wind
            # Position advances by v_ground * dt
            #
            # Profile modes:
            #   racetrack: State machine with straight legs + semicircular turns
            #   straight: Constant heading, no turns
            #   circle: Constant yaw rate, circular path
            # =========================================================================

            # =========================================================================
            #STRAIGHT and CIRCLE profile handling
            # =========================================================================
            profile_handled = False

            if profile == "straight":
                # Straight-line: constant heading, constant speed, no yaw rate
                hdg_rad = math.radians(profile_hdg_deg)
                track_unit_ned = np.array(
                    [math.cos(hdg_rad), math.sin(hdg_rad), 0.0]  # N component # E component
                )
                current_roll_rad = 0.0  # Wings level
                turn_state = "STRAIGHT"
                profile_handled = True

            elif profile == "circle":
                # Circle: constant yaw rate, integrate heading, constant bank
                # Update heading: hdg += omega * dt
                omega_rad_s = math.radians(profile_omega_deg_s)
                profile_hdg_deg += profile_omega_deg_s * dt  # Integrate heading
                # Wrap heading to 0-360
                profile_hdg_deg = profile_hdg_deg % 360.0

                hdg_rad = math.radians(profile_hdg_deg)
                track_unit_ned = np.array(
                    [math.cos(hdg_rad), math.sin(hdg_rad), 0.0]  # N component # E component
                )

                # Bank angle for coordinated turn: tan(bank) = v * omega / g
                if abs(omega_rad_s) > 0.001:
                    bank_rad = math.atan(LEAD_AIRSPEED_TAS * omega_rad_s / 9.81)
                    current_roll_rad = bank_rad  # Positive omega = right turn = positive roll
                else:
                    current_roll_rad = 0.0

                turn_state = "CIRCLE_CW" if profile_omega_deg_s > 0 else "CIRCLE_CCW"
                profile_handled = True

            # =========================================================================
            # RACETRACK profile (default): Original pattern state machine
            # =========================================================================
            if not profile_handled:
                # Compute track heading based on pattern state
                if pattern_state == PatternState.NORTH_LEG:
                    hdg_rad = 0.0  # North
                    track_unit_ned = np.array([1.0, 0.0, 0.0])

                    # Check for state transition (based on air distance, not position)
                    if air_distance_in_state >= LEG_LENGTH:
                        old_state = PatternState.get_name(pattern_state)
                        pattern_state = PatternState.NORTH_TO_SOUTH_TURN
                        new_state = PatternState.get_name(pattern_state)
                        print(f"[{elapsed:6.1f}s] State transition: {old_state} → {new_state}")
                        air_distance_in_state = 0.0

                elif pattern_state == PatternState.NORTH_TO_SOUTH_TURN:
                    # Arc angle based on air distance traveled
                    arc_angle = air_distance_in_state / TURN_RADIUS  # radians
                    if arc_angle > math.pi:
                        arc_angle = math.pi
                        old_state = PatternState.get_name(pattern_state)
                        pattern_state = PatternState.SOUTH_LEG
                        new_state = PatternState.get_name(pattern_state)
                        print(f"[{elapsed:6.1f}s] State transition: {old_state} → {new_state}")
                        air_distance_in_state = 0.0

                    # Heading during turn: 0° → 270° → 180° (LEFT turn through WEST)
                    hdg_rad = -arc_angle
                    track_unit_ned = np.array([math.cos(arc_angle), -math.sin(arc_angle), 0.0])

                elif pattern_state == PatternState.SOUTH_LEG:
                    hdg_rad = math.pi  # South
                    track_unit_ned = np.array([-1.0, 0.0, 0.0])

                    if air_distance_in_state >= LEG_LENGTH:
                        old_state = PatternState.get_name(pattern_state)
                        pattern_state = PatternState.SOUTH_TO_NORTH_TURN
                        new_state = PatternState.get_name(pattern_state)
                        print(f"[{elapsed:6.1f}s] State transition: {old_state} → {new_state}")
                        air_distance_in_state = 0.0

                elif pattern_state == PatternState.SOUTH_TO_NORTH_TURN:
                    arc_angle = air_distance_in_state / TURN_RADIUS
                    if arc_angle > math.pi:
                        arc_angle = math.pi
                        old_state = PatternState.get_name(pattern_state)
                        pattern_state = PatternState.NORTH_LEG
                        new_state = PatternState.get_name(pattern_state)
                        print(f"[{elapsed:6.1f}s] State transition: {old_state} → {new_state}")
                        print(f"           ✓ Circuit complete!")
                        air_distance_in_state = 0.0

                    # Heading during turn: 180° → 0° (counterclockwise via east)
                    hdg_rad = math.pi - arc_angle
                    track_unit_ned = np.array([-math.cos(arc_angle), math.sin(arc_angle), 0.0])

                # Compute roll angle for racetrack (inside not profile_handled block)
                if pattern_state in [
                    PatternState.NORTH_TO_SOUTH_TURN,
                    PatternState.SOUTH_TO_NORTH_TURN,
                ]:
                    bank_angle_rad = math.radians(BANK_ANGLE_DEG)
                    current_roll_rad = ROLL_SIGN_LEFT_TURN * bank_angle_rad
                    turn_state = "LEFT_TURN"
                else:
                    current_roll_rad = 0.0
                    turn_state = "STRAIGHT"

            # Compute air-relative velocity (aircraft flying along track at TAS)
            v_air_ned = LEAD_AIRSPEED_TAS * track_unit_ned

            # Pitch is 0 (level flight, no climb/descent modeled)
            current_pitch_rad = 0.0

            # Compute ground-relative velocity (air + wind)
            v_ground_ned = v_air_ned + wind_ned

            # Update position based on ground velocity
            beaver_pos_ned += v_ground_ned * dt

            # Update air distance traveled (for state transitions) using TAS
            air_distance_in_state += LEAD_AIRSPEED_TAS * dt

            # Extract position and velocity
            north_m = beaver_pos_ned[0]
            east_m = beaver_pos_ned[1]
            vn = v_ground_ned[0]  # Ground-relative velocity (what GPS would see)
            ve = v_ground_ned[1]
            vd = v_ground_ned[2]

            # =================================================================
            # TEST 387: Apply first-order filter to mimic EKF dynamics
            # =================================================================
            # This reduces velocity mismatch between the Python kinematic model
            # (exact calculation) and the QuadPlane's EKF (filtered estimate).
            # Lower alpha = more smoothing (more lag).
            # =================================================================
            VELOCITY_FILTER_ALPHA = 0.3  # Tunable: lower = more smoothing
            if filtered_vn is None:
                filtered_vn = vn
                filtered_ve = ve
            filtered_vn = VELOCITY_FILTER_ALPHA * vn + (1 - VELOCITY_FILTER_ALPHA) * filtered_vn
            filtered_ve = VELOCITY_FILTER_ALPHA * ve + (1 - VELOCITY_FILTER_ALPHA) * filtered_ve
            vn, ve = filtered_vn, filtered_ve

            # =================================================================
            # Continuous heading from air velocity (0-360°)
            # =================================================================
            # Compute heading from air velocity vector for continuous 0-360°.
            # This ensures the controller can detect actual yaw rate during turns
            # instead of seeing discrete jumps (0°→180°) at turn boundaries.
            # Use air velocity (track heading) not ground velocity to get where
            # the aircraft is POINTING, not where it's DRIFTING due to wind.
            # =================================================================
            heading_rad = math.atan2(v_air_ned[1], v_air_ned[0])  # NED -> radians
            heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0  # 0-360°

            # Compute TAS and groundspeed (FlightDock uses TAS exclusively)
            # TAS is the primary speed reference for this research project
            # EAS is computed for reference only (ArduPilot comparison)
            airspeed_tas = LEAD_AIRSPEED_TAS  # True Airspeed [PRIMARY]
            airspeed_eas = LEAD_AIRSPEED_EAS  # Equivalent Airspeed [reference only]
            groundspeed = math.sqrt(vn**2 + ve**2)  # Horizontal ground speed

            # Convert NED to lat/lon
            lat = HOME_LAT + (north_m / 111111.0)
            lon = HOME_LON + (east_m / (111111.0 * math.cos(math.radians(HOME_LAT))))
            alt_msl = TARGET_ALT_MSL

            # Send GLOBAL_POSITION_INT at configured rate (default 10 Hz)
            # LB-2: Trigger based on sim-time elapsed (adapts to actual speedup)
            if last_position_sim is None:
                last_position_sim = current_sim_time
                # T-186 ARM B-0: the position-grid EPOCH draw (parity latches
                # here for the whole flight — the candidate init race)
                boot_probe.emit(boot_probe_line(
                    "grid_pos_epoch", int(current_sim_time * 1000)))
            if (current_sim_time - last_position_sim) >= BROADCAST_INTERVAL_SIM:
                # Boom control logic
                # Extend boom to 100% after 10s warmup and keep it
                # extended for the entire docking test. The boom must be fully
                # extended when QuadPlane begins the intercept/rejoin phase.
                BOOM_EXTEND_START_S = 10.0  # Start extending after warmup
                BOOM_EXTEND_DURATION_S = 5.0  # Time to fully extend (rate ~20%/s)
                if elapsed < BOOM_EXTEND_START_S:
                    boom_extension = 0.0  # Stowed during warmup
                elif elapsed < BOOM_EXTEND_START_S + BOOM_EXTEND_DURATION_S:
                    # Ramp from 0% to 100% over BOOM_EXTEND_DURATION_S
                    boom_extension = (elapsed - BOOM_EXTEND_START_S) / BOOM_EXTEND_DURATION_S
                else:
                    boom_extension = 1.0  # Fully extended for docking

                boom_length = 2.4 + boom_extension * 3.7

                # LB-3: Derive time_boot_ms from SimTimeAuthority (consistent sim-time)
                time_boot_ms = int(current_sim_time * 1000)
                # Monotonic clamp to avoid duplicates
                time_boot_ms = max(time_boot_ms, last_sent_tbms + 1)
                last_sent_tbms = time_boot_ms
                # T-186 ARM B-0: first sends' realized parity (budget-capped)
                boot_probe.emit(boot_probe_line("send_pos", time_boot_ms))
                position_sequence = (position_sequence + 1) % 65536

                # GLOBAL_POSITION_INT uses ground-relative velocity (vn, ve, vd)
                # This is what a GPS would measure
                # LB-5: Wall-time flood guard - skip send if too fast, but keep state updates
                # TEST 387: Use guarded wall-time (allowed for I/O rate limiting)
                wall_now = wall_monotonic_s()
                if wall_now - last_pos_send_wall >= MIN_SEND_DT_WALL:
                    last_pos_send_wall = wall_now
                    # #379: one payload tuple feeds BOTH the immediate path and
                    # the delay line (field order matches _tx_send_posvfr).
                    _posvfr_args = (
                        time_boot_ms,
                        int(lat * 1e7),
                        int(lon * 1e7),
                        int(alt_msl * 1000),
                        int(TARGET_ALT_AGL * 1000),
                        int(vn * 100),   # Ground velocity north (cm/s)
                        int(ve * 100),   # Ground velocity east (cm/s)
                        int(vd * 100),   # Ground velocity down (cm/s)
                        int(heading_deg * 100),
                        LEAD_AIRSPEED_TAS,   # airspeed: constant TAS (24.7 m/s)
                        groundspeed,         # groundspeed: horizontal GS (m/s)
                        int(heading_deg),    # heading: 0-360 deg
                        50,                  # throttle: arbitrary (unused)
                        alt_msl,             # alt: MSL altitude (m)
                        -vd,                 # climb: up-positive
                        # T-392a [15]: the RAW ATTITUDE tbms this broadcaster
                        # latched (NOT a re-derivation of a[0], which is a ZOH
                        # of this value plus a monotonic clamp). Rides the
                        # tuple so the delay line reports build-time truth.
                        last_time_boot_ms_valid,
                    )
                    if tx_delay.active:
                        tx_delay.push(current_sim_time, "posvfr", _posvfr_args)
                    else:
                        #Send on BOTH links (UDP to MAVProxy, TCP to SITL direct)
                        _tx_send_posvfr([mav_udp, mav_tcp], _posvfr_args)
                else:
                    flood_skip_pos += 1  # LB-6: Track skipped frames

                last_position_sim = current_sim_time  # LB-2: Track in sim-time

            # =================================================================
            # Send ATTITUDE at same rate as position
            # =================================================================
            # This provides roll/pitch/yaw for boom tip inertial targeting.
            # The follower uses this to compute the boom tip position in NED
            # using full Euler rotation instead of yaw-only.
            # =================================================================
            # LB-2: Trigger based on sim-time elapsed (adapts to actual speedup)
            if last_attitude_sim is None:
                last_attitude_sim = current_sim_time
                # T-186 ARM B-0: the attitude-grid EPOCH draw
                boot_probe.emit(boot_probe_line(
                    "grid_att_epoch", int(current_sim_time * 1000)))
            if (current_sim_time - last_attitude_sim) >= BROADCAST_INTERVAL_SIM:
                # LB-3: Derive time_boot_ms from SimTimeAuthority (consistent sim-time)
                time_boot_ms = int(current_sim_time * 1000)
                # Monotonic clamp to avoid duplicates
                time_boot_ms = max(time_boot_ms, last_sent_tbms + 1)
                last_sent_tbms = time_boot_ms

                # Compute angular rates (rad/s)
                # On straight legs: 0 rad/s
                # In turns: yaw rate = v / R (positive for left turn)
                if pattern_state in [
                    PatternState.NORTH_TO_SOUTH_TURN,
                    PatternState.SOUTH_TO_NORTH_TURN,
                ]:
                    yaw_rate = LEAD_AIRSPEED_TAS / TURN_RADIUS  # rad/s, left turn
                else:
                    yaw_rate = 0.0

                # LB-5: Wall-time flood guard - skip send if too fast, but keep state updates
                # TEST 387: Use guarded wall-time (allowed for I/O rate limiting)
                wall_now = wall_monotonic_s()
                if wall_now - last_att_send_wall >= MIN_SEND_DT_WALL:
                    last_att_send_wall = wall_now
                    _att_args = (
                        time_boot_ms,
                        current_roll_rad,   # roll (rad)
                        current_pitch_rad,  # pitch (rad)
                        heading_rad,        # yaw (rad) - air-velocity hdg
                        0.0,                # rollspeed (rad/s) - steady
                        0.0,                # pitchspeed (rad/s) - steady
                        yaw_rate,           # yawspeed (rad/s)
                    )
                    if tx_delay.active:
                        tx_delay.push(current_sim_time, "att", _att_args)
                    else:
                        #Send on BOTH links (UDP to MAVProxy, TCP to SITL direct)
                        _tx_send_att([mav_udp, mav_tcp], _att_args)
                else:
                    flood_skip_att += 1  # LB-6: Track skipped frames

                last_attitude_sim = current_sim_time  # LB-2: Track in sim-time

            # #379: deliver due delayed frames. FIFO order preserves the
            # receiver-side tbms monotonic contract; the capture-side cadence
            # (10 Hz + wall flood guard at capture) bounds delivery rate, so
            # no second flood guard is applied here (disclosed).
            if tx_delay.active:
                for _t_cap, _kind, _args in tx_delay.pop_due(current_sim_time):
                    if _kind == "posvfr":
                        _tx_send_posvfr([mav_udp, mav_tcp], _args)
                    else:
                        _tx_send_att([mav_udp, mav_tcp], _args)

            # Display progress every 5 seconds (wall time)
            if current_wall_time - last_display >= 5.0:
                expected_dur = EXPECTED_STATE_DURATIONS[pattern_state]
                circuit_progress = (air_distance_in_state / expected_dur / LEAD_AIRSPEED_TAS) * 100
                state_name = PatternState.get_name(pattern_state)
                wind_mag = np.linalg.norm(wind_ned[:2])
                wind_dir_from = (math.degrees(math.atan2(-wind_ned[1], -wind_ned[0])) + 360) % 360

                #Show turn_state and full attitude (roll, pitch, yaw)
                roll_deg = math.degrees(current_roll_rad)
                pitch_deg = math.degrees(current_pitch_rad)
                yaw_deg = heading_deg  # Yaw = heading in level flight

                print(f"\n[{elapsed:6.1f}s] {state_name} | Turn: {turn_state}")
                print(f"  Position: {lat:.6f}, {lon:.6f} | NED: N={north_m:.0f}m, E={east_m:.0f}m")
                print(
                    f"  Heading: {heading_deg:.1f}° | TAS: {airspeed_tas:.1f} m/s | "
                    f"GS: {groundspeed:.1f} m/s | EAS: {airspeed_eas:.1f} m/s"
                )
                print(
                    f"  Attitude: roll={roll_deg:+.1f}° pitch={pitch_deg:+.1f}° yaw={yaw_deg:.1f}° (7D.40)"
                )
                gs_delta = groundspeed - airspeed_tas
                print(
                    f"  Wind: {wind_mag:.1f} m/s from {wind_dir_from:.0f}° | ΔGS: {gs_delta:+.1f} m/s"
                )
                print(f"  Boom: {boom_extension*100:.0f}% extended ({boom_length:.1f}m)")
                _span = max(current_sim_time - pattern_start_time, 1e-9)
                _trav = float(np.linalg.norm(beaver_pos_ned - pos0_ned)) if pos0_ned is not None else 0.0
                _eff = _trav / _span
                print(
                    f"  [LEADER_DT_DIAG] sim_span={_span:.2f}s integrated_dt_sum={dt_integrated_sum:.3f} "
                    f"dt_ratio={dt_integrated_sum/_span:.4f} stale_skipped={dt_stale_skipped} "
                    f"dt_raw_last={dt_raw:.4f} eff_speed={_eff:.3f} tas={LEAD_AIRSPEED_TAS:.1f} "
                    f"speed_ratio={_eff/LEAD_AIRSPEED_TAS:.4f}"
                )

                last_display = current_wall_time

            # Telemetry statistics (every 10 seconds, wall time)
            if current_wall_time - last_sequence_report >= 10.0:
                packets_sent_last_10s = position_sequence - position_sequence_10s_mark
                tcp_status = "✓" if tcp_connected else "✗"
                print(
                    f"\n  Telemetry: {packets_sent_last_10s/10:.1f} Hz | "
                    f"Total: {position_sequence} packets | "
                    f"TCP:{tcp_status}"
                )
                position_sequence_10s_mark = position_sequence
                last_sequence_report = current_wall_time

            # TEST 329: Time-sync health logging (every 5 wall-seconds)
            log_time_sync_health()

            # LB-6: Periodic flood guard report
            # TEST 387: Use guarded wall-time (allowed for I/O rate limiting)
            wall_now_report = wall_monotonic_s()
            if wall_now_report - last_flood_report >= FLOOD_REPORT_INTERVAL:
                if flood_skip_pos > 0 or flood_skip_att > 0:
                    print(
                        f"[LB-6] Flood guard: skipped {flood_skip_pos} pos, {flood_skip_att} att in last {FLOOD_REPORT_INTERVAL:.0f}s"
                    )
                flood_skip_pos = 0
                flood_skip_att = 0
                last_flood_report = wall_now_report

            # LB-6: Speedup-aware loop pacing
            # Scale loop rate with speedup, capped at 200Hz to avoid CPU starvation.
            # At 1x: sleep(0.01) = 100Hz, at 3x+: sleep(0.005) = 200Hz (capped)
            time.sleep(scaled_sleep)

    except KeyboardInterrupt:
        print("\n\nBroadcaster stopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        #Close both links
        try:
            mav_udp.close()
            print("UDP connection closed")
        except Exception:
            pass
        try:
            if mav_tcp is not None:
                mav_tcp.close()
                print("TCP connection closed")
        except Exception:
            pass
        print("All connections closed")


if __name__ == "__main__":
    main()
