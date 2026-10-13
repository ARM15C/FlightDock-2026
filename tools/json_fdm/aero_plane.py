#!/usr/bin/env python3
"""aero_plane — SIM_Plane-equivalent aerodynamic force/moment model (J4A port).

RESPONSIBILITY
    Pure, deterministic port of ArduPilot's SIM_Plane aerodynamics (a last_letter
    skywalker_2013 model) so a JSON-lockstep FDM can reproduce native-substrate
    cruise dynamics. Computes body-frame aero force, aero moments (consumed by the
    integrator DIRECTLY as angular acceleration — SIM_Plane folds inertia into the
    coefficients; there is NO inertia tensor), and the force->body-accel mapping.
    Source-traced to ardupilot_formation/libraries/SITL/SIM_Plane.{cpp,h}:
      Coefficients SIM_Plane.h:48-92
      lift_coeff SIM_Plane.cpp:235-255 (liftCoeff)
      drag_coeff SIM_Plane.cpp:257-270 (dragCoeff)
      get_force SIM_Plane.cpp:344-396 (getForce)
      get_torque SIM_Plane.cpp:273-341 (getTorque)
      body_accel_* SIM_Plane.cpp:502-505
    NOT aerodynamically accurate (SIM_Plane.cpp:16-17); the objective is equivalence
    to what `--model quadplane` flies, not real-world fidelity. See the dispatch
    source_trace/ docs for the full matrix, omissions, and frame conventions.

PUBLIC API
    GRAVITY_MSS, SSL_AIR_DENSITY: float constants
    Coefficients: frozen dataclass, SIM_Plane.h defaults verbatim
    DEFAULT_COEFFICIENTS: Coefficients()
    lift_coeff(alpha, c) -> float
    drag_coeff(alpha, c) -> float
    get_force(alpha, beta, airspeed, gyro, ail, elev, rud, c, rho) -> (ax,ay,az)
    get_torque(alpha, beta, airspeed, gyro, ail, elev, rud, force, c, rho) -> (la,ma,na)
    air_density_at(alt_msl_m) -> float
    thrust_scale(mass, hover_throttle=0.7) -> float
    body_accel_from_forces(thrust_n, force, mass) -> (ax,ay,az)

SIDE EFFECTS
    None. Pure functions of their arguments. No I/O, no globals mutated, no
    wall-clock, no RNG. No imports of training / reward / observation / replay /
    env / fuser / controller / truth.

CAPS
    file soft 200 / hard 300 LoC; function LoC cap 60.

COVERAGE
    tests/unit/test_json_fdm_aero_plane.py (T1..T6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

GRAVITY_MSS: float = 9.80665          # AP_Math GRAVITY_MSS; NED +z down
SSL_AIR_DENSITY: float = 1.225        # SIM_Aircraft.h:281 air_density default
_IS_ZERO: float = 1.0e-9             # is_zero() guard for division by airspeed

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Coefficients:
    """SIM_Plane.h:48-92 default_coefficients, values verbatim (skywalker_2013)."""
    s: float = 0.45
    b: float = 1.88
    c: float = 0.24
    c_lift_0: float = 0.56
    c_lift_deltae: float = 0.0
    c_lift_a: float = 6.9
    c_lift_q: float = 0.0
    mcoeff: float = 50.0
    oswald: float = 0.9
    alpha_stall: float = 0.4712
    c_drag_q: float = 0.0
    c_drag_deltae: float = 0.0
    c_drag_p: float = 0.1
    c_y_0: float = 0.0
    c_y_b: float = -0.98
    c_y_p: float = 0.0
    c_y_r: float = 0.0
    c_y_deltaa: float = 0.0
    c_y_deltar: float = -0.2
    c_l_0: float = 0.0
    c_l_p: float = -1.0
    c_l_b: float = -0.12
    c_l_r: float = 0.14
    c_l_deltaa: float = 0.25
    c_l_deltar: float = -0.037
    c_m_0: float = 0.045
    c_m_a: float = -0.7
    c_m_q: float = -20.0
    c_m_deltae: float = 1.0
    c_n_0: float = 0.0
    c_n_b: float = 0.25
    c_n_p: float = 0.022
    c_n_r: float = -1.0
    c_n_deltaa: float = 0.0
    c_n_deltar: float = 0.1
    deltaa_max: float = 0.3491
    deltae_max: float = 0.3491
    deltar_max: float = 0.3491
    cg_offset: Vec3 = (-0.15, 0.0, -0.05)   # SIM_Plane.h:91 CGOffset


DEFAULT_COEFFICIENTS = Coefficients()


def lift_coeff(alpha: float, c: Coefficients = DEFAULT_COEFFICIENTS) -> float:
    """Sigmoid stall-blended lift coeff. Port of SIM_Plane.cpp:235-255 liftCoeff."""
    alpha0 = c.alpha_stall
    M = c.mcoeff
    # clamp |alpha-alpha0| to avoid exp() overflow (SIM_Plane.cpp:243-248)
    if alpha - alpha0 > 0.8:
        alpha = alpha0 + 0.8
    elif alpha0 - alpha > 0.8:
        alpha = alpha0 - 0.8
    em = math.exp(-M * (alpha - alpha0))
    ep = math.exp(M * (alpha + alpha0))
    sigmoid = (1 + em + ep) / (1 + em) / (1 + ep)
    linear = (1.0 - sigmoid) * (c.c_lift_0 + c.c_lift_a * alpha)
    flat_plate = sigmoid * (2 * math.copysign(1, alpha) * math.sin(alpha) ** 2 * math.cos(alpha))
    return linear + flat_plate


def drag_coeff(alpha: float, c: Coefficients = DEFAULT_COEFFICIENTS) -> float:
    """Parabolic-polar drag coeff. Port of SIM_Plane.cpp:257-270 dragCoeff."""
    ar = (c.b ** 2) / c.s
    return c.c_drag_p + (c.c_lift_0 + c.c_lift_a * alpha) ** 2 / (math.pi * c.oswald * ar)


def get_force(alpha: float, beta: float, airspeed: float, gyro: Vec3,
              ail: float, elev: float, rud: float,
              c: Coefficients = DEFAULT_COEFFICIENTS, rho: float = SSL_AIR_DENSITY) -> Vec3:
    """Body-frame aero force (N). Port of SIM_Plane.cpp:344-396 getForce."""
    c_lift_a = lift_coeff(alpha, c)
    c_drag_a = drag_coeff(alpha, c)
    ca, sa = math.cos(alpha), math.sin(alpha)
    c_x_a = -c_drag_a * ca + c_lift_a * sa
    c_x_q = -c.c_drag_q * ca + c.c_lift_q * sa
    c_z_a = -c_drag_a * sa - c_lift_a * ca
    c_z_q = -c.c_drag_q * sa - c.c_lift_q * ca
    p, q, r = gyro
    qbar = 0.5 * rho * airspeed ** 2 * c.s
    if abs(airspeed) <= _IS_ZERO:
        return (0.0, 0.0, 0.0)
    twoV = 2 * airspeed
    ax = qbar * (c_x_a + c_x_q * c.c * q / twoV
                 - c.c_drag_deltae * ca * abs(elev) + c.c_lift_deltae * sa * elev)
    ay = qbar * (c.c_y_0 + c.c_y_b * beta + c.c_y_p * c.b * p / twoV
                 + c.c_y_r * c.b * r / twoV + c.c_y_deltaa * ail + c.c_y_deltar * rud)
    az = qbar * (c_z_a + c_z_q * c.c * q / twoV
                 - c.c_drag_deltae * sa * abs(elev) - c.c_lift_deltae * ca * elev)
    return (ax, ay, az)


def get_torque(alpha: float, beta: float, airspeed: float, gyro: Vec3,
               ail: float, elev: float, rud: float, force: Vec3,
               c: Coefficients = DEFAULT_COEFFICIENTS, rho: float = SSL_AIR_DENSITY) -> Vec3:
    """Aero moments (used as angular accel). Port of SIM_Plane.cpp:273-341 getTorque."""
    p, q, r = gyro
    ea = airspeed
    qbar = 0.5 * rho * ea ** 2 * c.s
    if abs(ea) <= _IS_ZERO:
        la = ma = na = 0.0
    else:
        twoV = 2 * ea
        la = qbar * c.b * (c.c_l_0 + c.c_l_b * beta + c.c_l_p * c.b * p / twoV
                           + c.c_l_r * c.b * r / twoV + c.c_l_deltaa * ail + c.c_l_deltar * rud)
        ma = qbar * c.c * (c.c_m_0 + c.c_m_a * alpha + c.c_m_q * c.c * q / twoV + c.c_m_deltae * elev)
        na = qbar * c.b * (c.c_n_0 + c.c_n_b * beta + c.c_n_p * c.b * p / twoV
                           + c.c_n_r * c.b * r / twoV + c.c_n_deltaa * ail + c.c_n_deltar * rud)
    # CG-offset r x F (SIM_Plane.cpp:336-338)
    gx, gy, gz = c.cg_offset
    fx, fy, fz = force
    la += gy * fz - gz * fy
    ma += -gx * fz + gz * fx
    na += -gy * fx + gx * fy
    return (la, ma, na)


def air_density_at(alt_msl_m: float) -> float:
    """ISA-troposphere air density (kg/m^3) at MSL altitude; == SSL_AIR_DENSITY at 0 m.
    Verbatim port of tests/leader_broadcast_fixed.py:105-138; matches the native SIM
    altitude-density path (SIM_Aircraft.cpp:717-718 -> AP_Baro_atmosphere) to ~1e-3 at the
    docking band. Used only by the FDM's opt-in 'isa' air-density model (default stays fixed)."""
    t0, p0, lapse, r_gas, g = 288.15, 101325.0, 0.0065, 287.05, GRAVITY_MSS
    alt = max(0.0, min(alt_msl_m, 11000.0))
    temp = t0 - lapse * alt
    press = p0 * (temp / t0) ** (g / (r_gas * lapse))
    return press / (r_gas * temp)


def thrust_scale(mass: float, hover_throttle: float = 0.7) -> float:
    """N per unit throttle. SIM_Plane.cpp:47 thrust_scale=(mass*GRAVITY_MSS)/hover_throttle."""
    return (mass * GRAVITY_MSS) / hover_throttle


def body_accel_from_forces(thrust_n: float, force: Vec3, mass: float) -> Vec3:
    """(thrust,0,0)+force, /mass. SIM_Plane.cpp:504-505 (gravity-free body accel)."""
    return ((thrust_n + force[0]) / mass, force[1] / mass, force[2] / mass)
