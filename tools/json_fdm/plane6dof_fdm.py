#!/usr/bin/env python3
"""plane6dof_fdm — SIM_Plane 6-DOF dynamics FDM over the JSON-lockstep backend.

RESPONSIBILITY
    Orchestrate the JSON-SITL protocol/clock (reused UNCHANGED from minimal_fdm)
    with the source-traced SIM_Plane dynamics: decode the ArduPilot servo packet,
    run one fixed-step physics update (aero_plane + rigidbody6dof + dryden_wind),
    and reply with the mandatory SIM_JSON state carrying PHYSICAL IMU. Lockstep
    stays ON (no_time_sync absent). Cruise-only, airborne IC; no docking, no leader,
    no controller, no reward/observation/training coupling. The objective is
    native-substrate equivalence to `--model quadplane`, NOT real-world fidelity.
    step_physics() is exposed so offline validation drives the dynamics with NO
    socket/SITL. Protocol/clock source: minimal_fdm (J1/J2/J3-proven); dynamics
    source: see the dispatch source_trace/ docs.

PUBLIC API
    DEFAULT_MASS, DEFAULT_HOVER_THROTTLE, CRUISE_AIRSPEED: constants
    decode_servos(pwm) -> (aileron, elevator, throttle, rudder)
    Plane6DOFFDM(bind_host, bind_port, mass, hover_throttle, coeff, air_density,
                 thrust_calibration_mass, air_density_model, home_alt_amsl,
                 sensor_airspeed_frame, init_velocity_ned, init_euler,
                 init_position_ned, wind, ic_hold_sec, max_frames, log_jsonl_path,
                 recv_timeout_s, idempotent_dup, servo_lag, servo_speed)
    apply_env_overrides(args) -> args # FD_FDM_MASS_OVERRIDE (S2 Arm-B diagnostic)
        .step_physics(pwm, dt) -> state dict # one physics step, no socket
        .build_state(timestamp) -> bytes # mandatory SIM_JSON reply line
        .bind() -> int; .run() -> dict # UDP loop (reuses minimal_fdm protocol)
    selftest() -> int; main(argv=None) -> int

SIDE EFFECTS
    run()/bind() create+bind ONE UDP socket (default 0.0.0.0:9002) and reply to the
    servo-packet sender; optional per-frame JSONL log. step_physics()/build_state()
    are pure of I/O. No wall-clock in the physics path (only diagnostic rx-stamp in
    the log, never fed to the clock). No SITL spawn, no source write. No imports of
    training/reward/observation/replay/env/fuser/controller/truth.

CAPS
    file soft 430 / hard 450 LoC; function LoC cap 60; localhost/LAN UDP only.
    (Soft cap 280->300 declared at the sensor-airspeed-frame parity change,
    atmoconsist dispatch 2026-07-03; 300->330 declared at the --ic-hold-sec
    below-fix, P0 dispatch 2026-07-04; 330->345 declared at the --idempotent-dup
    frame-origin lever, #365 2026-07-10; 350->372 declared at the --servo-lag O1
    actuator port, S2 dispatch 2026-07-14; 372->430 declared at the --wind-*
    DrydenWind curriculum-v3 wiring, #465 2026-07-17 — not silent.)

COVERAGE
    tests/unit/test_json_fdm_plane6dof.py (T15..T25) +
    tests/unit/test_fdm_wind_wiring.py (W1..W10, #465) + offline validation (P4).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bare-name house style
from minimal_fdm import parse_servo_packet, next_timestamp, DEFAULT_PORT  # noqa: E402
import aero_plane  # noqa: E402
from aero_plane import DEFAULT_COEFFICIENTS, SSL_AIR_DENSITY  # noqa: E402
from rigidbody6dof import RigidBody6DOF, dcm_from_euler, specific_force  # noqa: E402
from dryden_wind import DrydenWind  # noqa: E402
from servo_model import ServoLag, DEFAULT_SERVO_SPEED  # noqa: E402 (S2/O1 port)

DEFAULT_MASS: float = 2.0               # SIM_Plane.cpp:41 (SIM_QuadPlane differs; see O9)
DEFAULT_HOVER_THROTTLE: float = 0.7     # SIM_Plane.h:44
CRUISE_AIRSPEED: float = 24.7           # follower TAS matched to DHC-2 leader
DEFAULT_ALT_M: float = 100.0


def _angle(pwm: int) -> float:
    """PWM->[-1,1] surface deflection (SIM_Aircraft.h:185). 0 => neutral (no signal)."""
    if pwm == 0:
        return 0.0
    return 2.0 * ((pwm - 1000) / 1000.0 - 0.5)


def _range(pwm: int) -> float:
    """PWM->[0,1] throttle. 0 => off (no signal)."""
    if pwm == 0:
        return 0.0
    return max(0.0, min(1.0, (pwm - 1000) / 1000.0))


def decode_servos(pwm: Sequence[int]):
    """ail=ch0, elev=ch1, thr=ch2, rud=ch3 (SIM_Plane.cpp:400-451)."""
    return _angle(pwm[0]), _angle(pwm[1]), _range(pwm[2]), _angle(pwm[3])


def wind_ramp_scale(t_s: float, ramp_start_s: float = 0.0,
                    ramp_dur_s: float = 0.0) -> float:
    """#465b wind ramp-in gain in [0,1], applied to the wind OUTPUT only.

    Repairs the MEASURED 4 m/s warmup wind-wall (W465_REPORT.md finding 1): FDM
    wind is boot-static, so cells >= 4 m/s fail warmup outright. This lets wind
    stay at zero through warmup and engage for the measured episode.

    0 for t < ramp_start_s; linear 0->1 over [ramp_start_s, ramp_start_s+ramp_dur_s];
    1 thereafter. ramp_dur_s == 0 => step at ramp_start_s. BOTH knobs 0 => DISABLED,
    returns exactly 1.0 before any arithmetic (byte-identical to pre-#465b).
    """
    if ramp_start_s <= 0.0 and ramp_dur_s <= 0.0:
        return 1.0                       # disabled — no float op on the wind vector
    if t_s < ramp_start_s:
        return 0.0
    if ramp_dur_s <= 0.0:
        return 1.0                       # step engagement at ramp_start_s
    frac = (t_s - ramp_start_s) / ramp_dur_s
    return 1.0 if frac >= 1.0 else frac


class Plane6DOFFDM:
    """JSON-lockstep FDM: minimal_fdm protocol + SIM_Plane dynamics."""

    def __init__(self, bind_host: str = "0.0.0.0", bind_port: int = DEFAULT_PORT,
                 mass: float = DEFAULT_MASS, hover_throttle: float = DEFAULT_HOVER_THROTTLE,
                 coeff=DEFAULT_COEFFICIENTS, air_density: float = SSL_AIR_DENSITY,
                 thrust_calibration_mass: Optional[float] = None,
                 air_density_model: str = "fixed", home_alt_amsl: float = 0.0,
                 sensor_airspeed_frame: str = "tas",
                 init_velocity_ned: Sequence[float] = (CRUISE_AIRSPEED, 0.0, 0.0),
                 init_euler: Sequence[float] = (0.0, 0.0, 0.0),
                 init_position_ned: Sequence[float] = (0.0, 0.0, -DEFAULT_ALT_M),
                 wind: Optional[DrydenWind] = None,
                 wind_ramp_start_s: float = 0.0, wind_ramp_dur_s: float = 0.0,
                 ic_hold_sec: float = 0.0,
                 max_frames: Optional[int] = None, log_jsonl_path: Optional[str] = None,
                 recv_timeout_s: Optional[float] = None,
                 idempotent_dup: bool = False, servo_lag: bool = False,
                 servo_speed: float = DEFAULT_SERVO_SPEED) -> None:
        self.bind_host = bind_host
        self.bind_port = int(bind_port)
        self.coeff = coeff
        self.mass = float(mass)
        # thrust_scale uses CALIBRATION mass (native keeps 2.0 kg), decoupled from gross; None => gross (legacy).
        self.thrust_calibration_mass = (float(thrust_calibration_mass)
                                        if thrust_calibration_mass is not None else self.mass)
        self.thrust_scale = aero_plane.thrust_scale(self.thrust_calibration_mass, hover_throttle)
        self.air_density = float(air_density)
        self.air_density_model = str(air_density_model)   # "fixed" (default, byte-identical) | "isa"
        self.home_alt_amsl = float(home_alt_amsl)
        self.sensor_airspeed_frame = str(sensor_airspeed_frame)  # "tas" (default) | "eas" (native parity)
        self.rb = RigidBody6DOF(dcm_from_euler(*init_euler), (0.0, 0.0, 0.0),
                                init_velocity_ned, init_position_ned)
        self.wind = wind if wind is not None else DrydenWind()   # default-OFF
        self._last_wind_ef = (0.0, 0.0, 0.0)  # #465 jsonl witness sample (logged only when wind.enabled)
        # #465b ramp-in: engage wind at a commanded sim-time so warmup flies at
        # wind=0 (the 4 m/s warmup wall). Both 0 => disabled, byte-identical.
        # The clock is Sigma dt inside step_physics — the SAME dt the physics
        # integrates on, so it cannot drift from the state it gates.
        self._wind_ramp_start_s = float(wind_ramp_start_s)
        self._wind_ramp_dur_s = float(wind_ramp_dur_s)
        self._wind_t_s = 0.0
        # P0 below-fix: sim-seconds to hold the vehicle
        # STATIONARY at the ICs before releasing dynamics, so boot datum captures (baro
        # cal / EKF origin / vertical datum) sample a motionless state = native on-ground
        # boot by construction. 0.0 (default) => branch never taken => byte-identical.
        # (wind/dynamics NOT stepped during hold; release = velocity step to IC cruise.)
        self.ic_hold_sec = float(ic_hold_sec)
        self.fdm_timestamp_s = 0.0
        self.accepted = 0
        self.rejected = 0
        self.last_frame_count: Optional[int] = None
        self.max_frames = max_frames
        self.log_jsonl_path = log_jsonl_path
        self.recv_timeout_s = recv_timeout_s
        self.sock: Optional[socket.socket] = None
        # #365 frame-origin lever: repeated servo frame_count => byte-identical
        # re-reply (no clock advance, no physics step), collapsing the two boot
        # candidate origins. Default False => branch dead, byte-identical.
        self.idempotent_dup = bool(idempotent_dup)
        self.dup_replies = 0
        self._last_reply: Optional[bytes] = None
        # S2/O1 port: native ServoModel 1p LPF, 4 channels. False => branch dead, byte-identical.
        self.servo_lag = ServoLag(servo_speed) if servo_lag else None

    def _air_density_now(self) -> float:
        """Aero density this step: fixed self.air_density (default, byte-identical) or ISA at AMSL (opt-in)."""
        if self.air_density_model == "isa":
            return aero_plane.air_density_at(self.home_alt_amsl - float(self.rb.position[2]))
        return self.air_density

    def _sensor_eas2tas_now(self) -> float:
        """ISA eas2tas at current AMSL for the SENSOR frame only (wire-output). ALWAYS the ISA
        model — deliberately decoupled from --air-density-model (aero-forces knob, J4-exonerated)."""
        alt_amsl = self.home_alt_amsl - float(self.rb.position[2])
        return float(np.sqrt(SSL_AIR_DENSITY / aero_plane.air_density_at(alt_amsl)))

    def step_physics(self, pwm: Sequence[int], dt: float) -> Dict:
        """One physics step (no socket). alpha/beta from prev-step-end air state."""
        vab = self.rb.velocity_air_bf
        alpha = float(np.arctan2(vab[2], vab[0]))
        beta = float(np.arctan2(vab[1], vab[0]))
        ail, elev, thr, rud = decode_servos(pwm)
        if self.servo_lag is not None:  # S2/O1: native actuator lag on the decoded values
            ail, elev, thr, rud = self.servo_lag.apply4(ail, elev, thr, rud, dt)
        gyro = tuple(self.rb.gyro)
        v = self.rb.airspeed
        rho = self._air_density_now()
        force = aero_plane.get_force(alpha, beta, v, gyro, ail, elev, rud, self.coeff, rho)
        rot_accel = aero_plane.get_torque(alpha, beta, v, gyro, ail, elev, rud, force,
                                          self.coeff, rho)
        accel_body_forces = aero_plane.body_accel_from_forces(thr * self.thrust_scale, force, self.mass)
        wind_ef = self.wind.update(dt)
        # #465b ramp: scale the OUTPUT only. self.wind.update() has already
        # advanced the Dryden state/seed, so the gust realization — and with it
        # twin determinism and the #465 sigma/L/lag-1 calibration — is untouched.
        self._wind_t_s += dt
        _s = wind_ramp_scale(self._wind_t_s, self._wind_ramp_start_s, self._wind_ramp_dur_s)
        if _s != 1.0:
            wind_ef = tuple(float(c) * _s for c in wind_ef)
        self._last_wind_ef = wind_ef  # #465 witness capture — AFTER scaling (#465b FM-4)
        return self.rb.step(accel_body_forces, rot_accel, dt, wind_ef)

    def build_state(self, timestamp: float) -> bytes:
        """Mandatory SIM_JSON reply + optional airspeed (no_time_sync ABSENT => lockstep ON)."""
        s = self.rb.state_dict()
        # Optional SIM_JSON AIRSPEED field (SIM_JSON.cpp:398-402: used directly as
        # airspeed_pitot). Frame selectable (atmoconsist parity fix, 2026-07-03):
        #   "tas" (default, byte-identical legacy): HAL divides by ISA once
        #     (sitl_airspeed.cpp:34) => realized frame ISA(alt) ~1.04, INCONSISTENT with
        #     the frozen controller's warm-baro eas2tas belief ~1.08 => B2.2 REJOIN stall.
        #   "eas": emit TAS/isa_eas2tas(alt_AMSL), replicating the NATIVE backend's double
        #     ISA division (SIM_Aircraft.cpp:717 + update_eas_airspeed :1364-1371, then HAL)
        #     => realized ~ISA^2 ~1.08 == native operating point == belief.
        # SENSOR frame only — never feeds back into the integrator/aero (s["airspeed"] is
        # TAS, rb.eas2tas=1.0); decoupled from --air-density-model. (Airspeed zero-offset
        # auto-cal artifact from the airborne-boot IC remains; see sentinel/SENTINEL_VERDICT.md.)
        wire_airspeed = (s["airspeed"] / self._sensor_eas2tas_now()
                         if self.sensor_airspeed_frame == "eas" else s["airspeed"])
        state = {
            "timestamp": float(timestamp),
            "imu": {"gyro": s["gyro"], "accel_body": s["accel_body"]},
            "position": s["position"],
            "attitude": s["attitude"],
            "velocity": s["velocity"],
            "airspeed": wire_airspeed,
        }
        return (json.dumps(state, separators=(",", ":")) + "\n").encode("ascii")

    def _held_state_bytes(self, timestamp: float) -> bytes:
        """IC-hold reply: stationary vehicle at the ICs — frozen pose, zero rates/velocity/
        airspeed, rest specific-force IMU (mandatory SIM_JSON fields intact, lockstep ON)."""
        s = self.rb.state_dict()
        sf = specific_force(dcm_from_euler(*s["attitude"]), np.zeros(3))
        state = {"timestamp": float(timestamp),
                 "imu": {"gyro": [0.0, 0.0, 0.0], "accel_body": [float(x) for x in sf]},
                 "position": s["position"], "attitude": s["attitude"],
                 "velocity": [0.0, 0.0, 0.0], "airspeed": 0.0}
        return (json.dumps(state, separators=(",", ":")) + "\n").encode("ascii")

    def bind(self) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.bind_host, self.bind_port))
        if self.recv_timeout_s:
            s.settimeout(self.recv_timeout_s)
        self.sock = s
        self.bind_port = s.getsockname()[1]
        return self.bind_port

    def _log(self, fh, pkt: Dict, dt: float, st: Dict, dup: int = 0) -> None:
        if fh is None:
            return
        import time  # diagnostic rx-stamp ONLY; never feeds the physics clock
        rec = {"frame_count": pkt["frame_count"], "frame_rate": pkt["frame_rate"], "dt": dt,
               "fdm_timestamp": self.fdm_timestamp_s, "airspeed": st["airspeed"],
               "alt_m": -st["position"][2], "pos": st["position"], "att": st["attitude"],
               "gyro": st["gyro"], "accel_body": st["accel_body"], "wall_rx_time": time.time()}
        if self.wind.enabled:
            # #465 wind witness — absent at default (byte-identical log)
            rec["wind_ned"] = [float(x) for x in self._last_wind_ef]
        if dup:
            rec["dup"] = 1  # #365 lever only — absent at default (byte-identical log)
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        fh.flush()

    def run(self) -> Dict:
        """Blocking recv/step/reply loop (reuses minimal_fdm parse + clock)."""
        fh = open(self.log_jsonl_path, "w") if self.log_jsonl_path else None
        if self.sock is None:
            self.bind()
        sl = self.servo_lag  # S2 knob-honored gate: FDM-side confirmation in the startup log
        print("plane6dof_fdm: listening udp://%s:%d max_frames=%s servo_lag=%d servo_speed=%.3f wind_enabled=%d" %
              (self.bind_host, self.bind_port, self.max_frames,
               1 if sl is not None else 0, sl.speed_s if sl is not None else 0.0,
               1 if self.wind.enabled else 0), flush=True)
        if self.wind.enabled:
            # #465 config echo (detector: witness legs assert commanded == realized config)
            print("plane6dof_fdm: wind mean_ned=%s sigma=%s length_scale=%s airspeed=%.3f seed=%d"
                  " ramp_start_s=%.3f ramp_dur_s=%.3f" %
                  (np.round(self.wind.mean, 6).tolist(), np.round(self.wind.sigma, 6).tolist(),
                   np.round(self.wind.length_scale, 6).tolist(), self.wind.airspeed,
                   self.wind._seed, self._wind_ramp_start_s, self._wind_ramp_dur_s), flush=True)
        try:
            while self.max_frames is None or self.accepted < self.max_frames:
                try:
                    data, addr = self.sock.recvfrom(2048)
                except socket.timeout:
                    break
                pkt = parse_servo_packet(data)
                if pkt is None:
                    self.rejected += 1
                    continue
                if (self.idempotent_dup and self._last_reply is not None
                        and pkt["frame_count"] == self.last_frame_count):
                    # #365 lever: SITL re-sent a frame_count it already sent (boot
                    # double-send / starvation resend). Re-deliver the SAME bytes.
                    self.sock.sendto(self._last_reply, addr)
                    self.dup_replies += 1
                    self._log(fh, pkt, 0.0, self.rb.state_dict(), dup=1)
                    continue
                dt = (1.0 / pkt["frame_rate"]) if pkt["frame_rate"] else 0.0
                if self.fdm_timestamp_s < self.ic_hold_sec:  # IC-hold window (default 0.0 => never)
                    self.fdm_timestamp_s = next_timestamp(self.fdm_timestamp_s, pkt["frame_rate"])
                    self._last_reply = self._held_state_bytes(self.fdm_timestamp_s)
                    self.sock.sendto(self._last_reply, addr)
                    self.accepted += 1
                    self.last_frame_count = pkt["frame_count"]
                    self._log(fh, pkt, 0.0, self.rb.state_dict())
                    continue
                st = self.step_physics(pkt["pwm"], dt)
                self.fdm_timestamp_s = next_timestamp(self.fdm_timestamp_s, pkt["frame_rate"])
                self._last_reply = self.build_state(self.fdm_timestamp_s)
                self.sock.sendto(self._last_reply, addr)
                self.accepted += 1
                self.last_frame_count = pkt["frame_count"]
                self._log(fh, pkt, dt, st)
        finally:
            if self.sock is not None:
                self.sock.close()
            if fh is not None:
                fh.close()
        summary = {"accepted": self.accepted, "rejected": self.rejected,
                   "fdm_timestamp_s": self.fdm_timestamp_s, "last_frame_count": self.last_frame_count}
        if self.idempotent_dup:
            summary["dup_replies"] = self.dup_replies  # absent at default (byte-identical)
        if self.wind.enabled:
            # #465 — absent at default (byte-identical)
            summary["wind"] = {"mean_ned": self.wind.mean.tolist(),
                               "sigma": self.wind.sigma.tolist(),
                               "length_scale": self.wind.length_scale.tolist(),
                               "seed": self.wind._seed,
                               "ramp_start_s": self._wind_ramp_start_s,
                               "ramp_dur_s": self._wind_ramp_dur_s}
        print("plane6dof_fdm: done %s" % json.dumps(summary, separators=(",", ":")), flush=True)
        return summary


def selftest() -> int:
    """Standalone offline sanity (no SITL): level IMU, trim boundedness, determinism."""
    ok = True
    # level specific force = (0,0,-g)
    sf = specific_force(dcm_from_euler(0, 0, 0), np.zeros(3))
    c1 = abs(sf[0]) < 1e-9 and abs(sf[1]) < 1e-9 and abs(sf[2] + aero_plane.GRAVITY_MSS) < 1e-9
    print("selftest level-IMU(0,0,-g): %s (%s)" % ("PASS" if c1 else "FAIL", np.round(sf, 4).tolist()))
    ok &= c1
    # 600-step trim cruise: bounded, finite
    fdm = Plane6DOFFDM(bind_port=0)
    pwm = [1500, 1500, 1500, 1500] + [1500] * 12
    for _ in range(600):
        fdm.step_physics(pwm, 1.0 / 400.0)
    st = fdm.rb.state_dict()
    finite = all(np.isfinite(st["position"])) and np.isfinite(st["airspeed"])
    bounded = st["airspeed"] < 200.0 and max(abs(g) for g in st["gyro"]) < 40.0
    print("selftest trim-bounded: %s (V=%.2f alt=%.1f)" %
          ("PASS" if (finite and bounded) else "FAIL", st["airspeed"], -st["position"][2]))
    ok &= finite and bounded
    # determinism: two identical runs match
    def _run():
        f = Plane6DOFFDM(bind_port=0)
        for _ in range(200):
            f.step_physics(pwm, 1.0 / 400.0)
        return f.rb.state_dict()["position"]
    det = np.allclose(_run(), _run(), atol=0.0, rtol=0.0)
    print("selftest determinism: %s" % ("PASS" if det else "FAIL"))
    ok &= det
    print("SELFTEST: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def build_wind(speed_ms, from_deg, sigma_csv, length_csv, seed, airspeed) -> Optional[DrydenWind]:
    """#465: construct the DrydenWind for the CLI knobs, or None at all-defaults.

    FROM-bearing convention (curriculum v3): wind blows FROM compass from_deg
    TOWARD from_deg+180, so mean_ned = (-W*cos(th), -W*sin(th), 0.0); FROM-180
    => (+W, 0, 0) EXACTLY (sub-1e-9 trig residue snapped to 0.0 so both the
    contract identity and the wind-off exact-zero guarantee hold bitwise).
    speed==0 with any sigma>0 is the W0 turbulence-bridge stage (mean zero,
    gusts live). Returns None when speed==0 AND all sigma==0 => the ctor's
    default-OFF DrydenWind => byte-identical no-wind path.
    """
    sigma = tuple(float(x) for x in str(sigma_csv).split(","))
    length = tuple(float(x) for x in str(length_csv).split(","))
    if len(sigma) != 3 or len(length) != 3:
        raise ValueError("wind sigma/length must be N,E,D comma triples")
    if float(speed_ms) == 0.0 and not any(s > 0.0 for s in sigma):
        return None
    th = float(np.radians(float(from_deg)))
    mean = [-float(speed_ms) * float(np.cos(th)), -float(speed_ms) * float(np.sin(th)), 0.0]
    mean = [0.0 if abs(m) < 1e-9 else m for m in mean]
    return DrydenWind(mean_ned=mean, sigma=sigma, length_scale=length,
                      airspeed=float(airspeed), seed=int(seed))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SIM_Plane 6-DOF JSON-lockstep FDM (J4A).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bind-host", default="0.0.0.0")
    ap.add_argument("--bind-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--log-jsonl", default=None)
    ap.add_argument("--mass", type=float, default=DEFAULT_MASS)
    ap.add_argument("--thrust-calibration-mass", type=float, default=None)  # native 2.0; None=gross
    ap.add_argument("--air-density-model", choices=("fixed", "isa"), default="fixed")
    ap.add_argument("--sensor-airspeed-frame", choices=("tas", "eas"), default="tas")
    ap.add_argument("--home-alt-amsl", type=float, default=0.0)
    ap.add_argument("--ic-hold-sec", type=float, default=0.0)  # P0 below-fix; 0 = OFF
    ap.add_argument("--alt", type=float, default=DEFAULT_ALT_M)
    ap.add_argument("--airspeed", type=float, default=CRUISE_AIRSPEED)
    ap.add_argument("--recv-timeout-s", type=float, default=None)
    ap.add_argument("--idempotent-dup", type=int, choices=(0, 1), default=0)  # #365 lever; 0 = OFF
    # S2/O1 lever; env-var defaults => A/B toggles with zero harness-argv edits (config-time only).
    ap.add_argument("--servo-lag", type=int, choices=(0, 1),
                    default=int(os.environ.get("FD_FDM_SERVO_LAG", "0")))
    ap.add_argument("--servo-speed", type=float,
                    default=float(os.environ.get("FD_FDM_SERVO_SPEED", str(DEFAULT_SERVO_SPEED))))
    # #465 curriculum-v3 wind (P0.2): env-var defaults => drivers/curriculum set
    # FD_FDM_WIND_* with zero harness-argv edits (servo-lag precedent). ALL
    # defaults => build_wind() returns None => ctor default-OFF => byte-identical.
    ap.add_argument("--wind-speed-ms", type=float,
                    default=float(os.environ.get("FD_FDM_WIND_SPEED_MS", "0") or "0"))
    ap.add_argument("--wind-from-deg", type=float,
                    default=float(os.environ.get("FD_FDM_WIND_FROM_DEG", "180") or "180"))
    ap.add_argument("--wind-sigma-ms",
                    default=os.environ.get("FD_FDM_WIND_SIGMA_MS", "0,0,0") or "0,0,0")
    ap.add_argument("--wind-length-m",
                    default=os.environ.get("FD_FDM_WIND_LENGTH_M", "300,300,300") or "300,300,300")
    ap.add_argument("--wind-seed", type=int,
                    default=int(os.environ.get("FD_FDM_WIND_SEED", "20260622") or "20260622"))
    # #465b ramp-in (repairs the 4 m/s warmup wind-wall). Both 0 => disabled.
    ap.add_argument("--wind-ramp-start-s", type=float,
                    default=float(os.environ.get("FD_FDM_WIND_RAMP_START_S", "0") or "0"))
    ap.add_argument("--wind-ramp-dur-s", type=float,
                    default=float(os.environ.get("FD_FDM_WIND_RAMP_DUR_S", "0") or "0"))
    args = ap.parse_args(argv)
    apply_env_overrides(args)
    if args.selftest:
        return selftest()
    Plane6DOFFDM(bind_host=args.bind_host, bind_port=args.bind_port, mass=args.mass,
                 thrust_calibration_mass=args.thrust_calibration_mass,
                 air_density_model=args.air_density_model, home_alt_amsl=args.home_alt_amsl,
                 sensor_airspeed_frame=args.sensor_airspeed_frame,
                 wind=build_wind(args.wind_speed_ms, args.wind_from_deg, args.wind_sigma_ms,
                                 args.wind_length_m, args.wind_seed, args.airspeed),
                 wind_ramp_start_s=args.wind_ramp_start_s,
                 wind_ramp_dur_s=args.wind_ramp_dur_s,
                 init_velocity_ned=(args.airspeed, 0.0, 0.0),
                 init_position_ned=(0.0, 0.0, -args.alt), ic_hold_sec=args.ic_hold_sec,
                 max_frames=args.max_frames, log_jsonl_path=args.log_jsonl,
                 recv_timeout_s=args.recv_timeout_s,
                 idempotent_dup=bool(args.idempotent_dup),
                 servo_lag=bool(args.servo_lag), servo_speed=args.servo_speed).run()
    return 0


def apply_env_overrides(args) -> None:
    """S2 Arm-B diagnostic: FD_FDM_MASS_OVERRIDE beats harness argv --mass; unset => no-op."""
    mo = os.environ.get("FD_FDM_MASS_OVERRIDE")
    if mo:
        args.mass = float(mo)


if __name__ == "__main__":
    sys.exit(main())
