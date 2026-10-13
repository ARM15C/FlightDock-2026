# Responsibility: PURE follower self along-track wind feedforward math for the S1 wind fix
#   (WINDFF, controller flag FD_SELF_WIND_FF). Stateless functions only — no I/O, no SITL, no
#   MAVLink, no wall clock, no truth/SIM_STATE. The controller owns the LPF/trim state and passes
#   it in/out each tick. Mechanism ([MEASURED] Phase-C a5c6e5b1): the controller commands
#   `spd_cmd = lead_GS + v_c` as AIRSPEED; TECS realizes it as TAS, so realized closure = v_c + w_par
#   (the follower's own along-track wind). Subtracting an estimated w_par at the TAS-realization seam
#   converts the lead_GS reference to a lead_TAS reference -> realized closure = v_c.
# Public API:
#   along_track_wind(vn, ve, tas_meas, heading_deg, track_deg) -> float # w_par, tailwind-positive
#   lpf_step(prev, sample, dt_s, tau_s) -> float # 1st-order low-pass
#   trim_step(prev_i, closure_err, dt_s, ki, clamp, active) -> float # state-gated PI-I trim
#   correction(west, trim, ff_on, trim_on) -> float # net TAS-frame delta
# Side effects: NONE. Determinism: fixed-dt, wall-clock-free (lockstep-safe). Caps: file soft 90 /
#   hard 120 LoC; per-function <= 35. Coverage: tests/unit/test_wind_ff.py (projection/LPF/sign,
#   cos-crab, anti-windup + clamp + state-gating, OFF byte-identity, leakage statics).
import math


def along_track_wind(vn: float, ve: float, tas_meas: float,
                     heading_deg: float, track_deg: float) -> float:
    """Scalar wind-triangle estimate of the along-track wind (m/s, tailwind-positive):
        w_par = GS_along - TAS * cos(psi_fol - track)
    GS_along projects the perception ground velocity (vn, ve) onto the formation track;
    the TAS term uses the follower airframe heading so a crab angle is exactly removed
    (cos(heading-track); ~0.02 m/s residual at 1 m/s cross over 24.7 TAS). Inputs are all
    perception-side (get_velocity_ned / get_airspeed x eas2tas / get_attitude yaw); NEVER truth."""
    tr = math.radians(track_deg)
    gs_along = vn * math.cos(tr) + ve * math.sin(tr)
    tas_along = tas_meas * math.cos(math.radians(heading_deg - track_deg))
    return gs_along - tas_along


def lpf_step(prev, sample: float, dt_s: float, tau_s: float) -> float:
    """First-order low-pass toward `sample`. Seeds to `sample` when prev is None (first tick)
    or when dt/tau are non-positive (degenerate -> pass through). alpha = 1 - exp(-dt/tau),
    dt clamped to tau so a long gap cannot overshoot. Deterministic (no wall clock)."""
    if prev is None or tau_s <= 0.0 or dt_s <= 0.0:
        return sample
    alpha = 1.0 - math.exp(-min(dt_s, tau_s) / tau_s)
    return prev + alpha * (sample - prev)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def trim_step(prev_i: float, closure_err: float, dt_s: float, ki: float,
              clamp: float, active: bool) -> float:
    """Integrator-dominant closure trim on (v_c_cmd - measured_closure), m/s.
    Accumulates ki*err*dt ONLY when `active` (state-gated to the terminal hold states); otherwise
    the integrator is HELD (frozen), not reset, so it does not wind down between DC episodes.
    Anti-windup: the integrator is clamped to +-clamp every call (held value included), so a run
    at the rail cannot accumulate. dt clamped to 0.2 s. Positive err (closing too slow / falling
    behind) grows the trim -> speeds up; negative err (over-closing) drives it negative -> the
    net command can fall below lead_GS (the negative closure the governor's max(0,.) cannot express)."""
    if not active or dt_s <= 0.0:
        return _clip(prev_i, -clamp, clamp)
    return _clip(prev_i + ki * closure_err * min(dt_s, 0.2), -clamp, clamp)


def correction(west: float, trim: float, ff_on: bool, trim_on: bool) -> float:
    """Net TAS-frame delta to add to a GS-referenced speed command: -west (+trim when enabled).
    Returns exactly 0.0 when ff_on is False so the caller stays byte-identical with the flag OFF."""
    if not ff_on:
        return 0.0
    return -west + (trim if trim_on else 0.0)
