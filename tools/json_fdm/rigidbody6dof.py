#!/usr/bin/env python3
"""rigidbody6dof — 6-DOF rigid-body integrator + specific-force IMU (J4A port).

RESPONSIBILITY
    Pure, deterministic, fixed-step port of ArduPilot's rigid-body update and DCM
    algebra so a JSON-lockstep FDM produces physically-consistent IMU (the J4
    disqualifier for point-mass FDMs). Integrates body rates, attitude (DCM),
    earth-frame velocity and NED position, and emits the accelerometer SPECIFIC
    FORCE (gravity-projected, body FRD). Cruise/airborne only — ground interaction,
    smoothing, and sensor noise are out of scope (see OMISSION_REGISTER O3/O4/O5).
    Source-traced to ardupilot_formation/libraries/SITL/:
      step() SIM_Aircraft.cpp:704-765 (update_dynamics, airborne path)
      specific_force SIM_Aircraft.cpp:748
      dcm_rotate AP_Math Matrix3::rotate (row += row x g)
      dcm_normalize AP_Math Matrix3::normalize (Gram-Schmidt)
      dcm_from_euler AP_Math Matrix3::from_euler (3-2-1)
      dcm_to_euler AP_Math Matrix3::to_euler
    DCM is body->earth (SIM_Aircraft.h:211): earth = dcm@body, body = dcm.T@earth.

PUBLIC API
    dcm_from_euler(roll,pitch,yaw)->3x3; dcm_to_euler(dcm)->(r,p,y)
    dcm_rotate(dcm,g)->3x3; dcm_normalize(dcm)->3x3
    specific_force(dcm, accel_earth)->(3,)
    RigidBody6DOF(dcm, gyro, velocity_ef, position, eas2tas=1.0)
        .step(accel_body_forces, rot_accel, dt, wind_ef=(0,0,0)) -> dict
        .state_dict() -> dict; .euler() -> (r,p,y)

SIDE EFFECTS
    None beyond mutating its own instance state in step(). No I/O, no wall-clock,
    no RNG. No training/reward/observation/replay/env/fuser/controller/truth import.

CAPS
    file soft 200 / hard 300 LoC; function LoC cap 60.

COVERAGE
    tests/unit/test_json_fdm_rigidbody6dof.py (T7..T11).
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

from aero_plane import GRAVITY_MSS

_GYRO_LIMIT = math.radians(2000.0)          # SIM_Aircraft.cpp:723-725
_ACCEL_LIMIT = 64.0 * GRAVITY_MSS           # SIM_Aircraft.cpp:728
_G_NED = np.array([0.0, 0.0, GRAVITY_MSS])  # gravity, NED +z down


def dcm_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->earth DCM from 3-2-1 euler. Port of Matrix3::from_euler."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp],
    ])


def dcm_to_euler(dcm: np.ndarray) -> Tuple[float, float, float]:
    """(roll,pitch,yaw) from body->earth DCM. Port of Matrix3::to_euler."""
    pitch = -math.asin(max(-1.0, min(1.0, dcm[2][0])))
    roll = math.atan2(dcm[2][1], dcm[2][2])
    yaw = math.atan2(dcm[1][0], dcm[0][0])
    return (roll, pitch, yaw)


def dcm_rotate(dcm: np.ndarray, g: np.ndarray) -> np.ndarray:
    """First-order DCM update: each row += row x g. Port of Matrix3::rotate."""
    out = dcm.copy()
    for i in range(3):
        out[i] = dcm[i] + np.cross(dcm[i], g)
    return out


def dcm_normalize(dcm: np.ndarray) -> np.ndarray:
    """Gram-Schmidt re-orthonormalize. Port of Matrix3::normalize."""
    a, b = dcm[0], dcm[1]
    error = float(np.dot(a, b))
    t0 = a - b * (0.5 * error)
    t1 = b - a * (0.5 * error)
    t2 = np.cross(t0, t1)
    return np.array([t0 / np.linalg.norm(t0),
                     t1 / np.linalg.norm(t1),
                     t2 / np.linalg.norm(t2)])


def specific_force(dcm: np.ndarray, accel_earth: np.ndarray) -> np.ndarray:
    """Accelerometer specific force (body FRD). SIM_Aircraft.cpp:748.

    accel_body = dcm^T * (accel_earth - gravity). Level + steady -> (0,0,-g).
    """
    return dcm.T @ (accel_earth - _G_NED)


class RigidBody6DOF:
    """Fixed-step 6-DOF integrator (airborne). Mirrors update_dynamics ordering."""

    def __init__(self, dcm: np.ndarray, gyro: Sequence[float],
                 velocity_ef: Sequence[float], position: Sequence[float],
                 eas2tas: float = 1.0) -> None:
        self.dcm = np.array(dcm, dtype=float)
        self.gyro = np.array(gyro, dtype=float)
        self.velocity_ef = np.array(velocity_ef, dtype=float)
        self.position = np.array(position, dtype=float)
        self.eas2tas = float(eas2tas)
        self.accel_body = np.array([0.0, 0.0, -GRAVITY_MSS])   # default level-at-rest IMU
        self.velocity_air_ef = self.velocity_ef.copy()
        self.velocity_air_bf = self.dcm.T @ self.velocity_air_ef
        self.airspeed = float(np.linalg.norm(self.velocity_air_ef) / self.eas2tas)

    def step(self, accel_body_forces: Sequence[float], rot_accel: Sequence[float],
             dt: float, wind_ef: Sequence[float] = (0.0, 0.0, 0.0)) -> Dict:
        """One physics step (airborne). Port of SIM_Aircraft.cpp:704-765."""
        af = np.array(accel_body_forces, dtype=float)
        ra = np.array(rot_accel, dtype=float)
        wind = np.array(wind_ef, dtype=float)

        self.gyro = np.clip(self.gyro + ra * dt, -_GYRO_LIMIT, _GYRO_LIMIT)
        af = np.clip(af, -_ACCEL_LIMIT, _ACCEL_LIMIT)

        self.dcm = dcm_normalize(dcm_rotate(self.dcm, self.gyro * dt))

        accel_earth = self.dcm @ af + _G_NED                       # SIM_Aircraft.cpp:737-738
        self.accel_body = self.dcm.T @ (accel_earth - _G_NED)      # specific force (748)

        self.velocity_ef = self.velocity_ef + accel_earth * dt     # 751
        self.position = self.position + self.velocity_ef * dt       # 755
        self.velocity_air_ef = self.velocity_ef - wind             # 758
        self.velocity_air_bf = self.dcm.T @ self.velocity_air_ef   # 761
        self.airspeed = float(np.linalg.norm(self.velocity_air_ef) / self.eas2tas)
        return self.state_dict()

    def euler(self) -> Tuple[float, float, float]:
        return dcm_to_euler(self.dcm)

    def state_dict(self) -> Dict:
        r, p, y = self.euler()
        return {
            "gyro": self.gyro.tolist(),
            "accel_body": self.accel_body.tolist(),
            "position": self.position.tolist(),
            "velocity": self.velocity_ef.tolist(),
            "attitude": [r, p, y],
            "airspeed": self.airspeed,
            "velocity_air_bf": self.velocity_air_bf.tolist(),
        }
