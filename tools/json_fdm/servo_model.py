#!/usr/bin/env python3
"""servo_model — ServoModel actuator-lag port for the JSON-lockstep FDM (S2/O1).

RESPONSIBILITY
    Faithful port of ArduPilot's SITL ServoModel in the configuration our vehicles
    actually FLY: both live param files set SIM_SERVO_DELAY=0, SIM_SERVO_FILTER=0,
    SIM_SERVO_SPEED=0.14 (quadplane_formation_modern_v1.parm:1312-14,
    _mav_clean.parm:1354-56), which reduces ServoModel::apply_filter to the simple
    path (ServoModel.cpp:53-62): a 1-pole low-pass with cutoff = 1/(2*pi*speed)
    ~= 1.137 Hz, i.e. LowPassFilter alpha = dt/(dt + speed). Native applies it to
    ALL FOUR channels (ail/elev/rud filter_angle SIM_Plane.cpp:400-402; throttle
    filter_range :451); the FDM's PWM->v maps already match filter_angle/filter_range
    exactly — this module adds ONLY the missing filter, applied to the decoded v.
    Deliberate deviations (documented): (1) lazy-init each channel to its first
    sample instead of ArduPilot's 0-init — the jsn vehicle boots AIRBORNE at cruise
    trim, native boots on-ground where 0-init is moot; (2) dt<=0 is a no-op hold
    (frozen in S2_SUBSTRATE_PREREG.md) instead of ArduPilot's alpha=1 snap — at
    dt=0 the plant does not advance, so the actuator must not either. The slew /
    delay / 2-pole-butterworth branches are NOT ported (configured OFF in both
    live param files — porting them would model config native does not fly).

PUBLIC API
    DEFAULT_SERVO_SPEED: float = 0.14 (SIM_SERVO_SPEED, s per 60 deg)
    lowpass_alpha(dt, speed_s) -> float
    ServoLag(speed_s=DEFAULT_SERVO_SPEED)
        .apply(channel_idx, v, dt) -> float # one channel, lazy-init
        .apply4(ail, elev, thr, rud, dt) -> (ail, elev, thr, rud)

SIDE EFFECTS
    None beyond per-channel filter state on the instance. Pure arithmetic; no I/O,
    no RNG, no wall-clock, no env reads, no imports of training/reward/observation/
    replay/env/fuser/controller/truth.

CAPS
    file soft 90 / hard 110 LoC; function LoC cap 20.

COVERAGE
    tests/unit/test_json_fdm_servo_model.py (S1..S6).
"""
from __future__ import annotations

DEFAULT_SERVO_SPEED: float = 0.14  # SIM_SERVO_SPEED default AND both live param files


def lowpass_alpha(dt: float, speed_s: float) -> float:
    """ArduPilot calc_lowpass_alpha_dt at cutoff=1/(2*pi*speed): rc == speed_s, so
    alpha = dt/(dt + speed_s). dt<=0 or speed<=0 -> 0.0 (no-op hold; see header)."""
    if dt <= 0.0 or speed_s <= 0.0:
        return 0.0
    return dt / (dt + speed_s)


class ServoLag:
    """Per-channel 1p actuator low-pass (4 channels: ail=0, elev=1, thr=2, rud=3)."""

    def __init__(self, speed_s: float = DEFAULT_SERVO_SPEED) -> None:
        self.speed_s = float(speed_s)
        self._state = [None, None, None, None]  # lazy-init: None until first sample

    def apply(self, channel_idx: int, v: float, dt: float) -> float:
        """Filter one decoded servo value. First sample initializes the channel."""
        prev = self._state[channel_idx]
        if prev is None:
            self._state[channel_idx] = v
            return v
        out = prev + (v - prev) * lowpass_alpha(dt, self.speed_s)
        self._state[channel_idx] = out
        return out

    def apply4(self, ail: float, elev: float, thr: float, rud: float, dt: float):
        """Filter all four channels (native order: SIM_Plane.cpp:400-402,451)."""
        return (self.apply(0, ail, dt), self.apply(1, elev, dt),
                self.apply(2, thr, dt), self.apply(3, rud, dt))
