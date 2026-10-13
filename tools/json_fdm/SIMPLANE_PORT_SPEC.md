# SIMPLANE_PORT_SPEC — JSON-lockstep 6-DOF FDM (J4A)

In-tree spec for the SIM_Plane-derived Python FDM modules in `tools/json_fdm/`. Companion to the dispatch ART `analysis/jsonlockstep_j4a_20260622T172558Z/` (full source-trace, omission register, frame conventions, validation evidence).

## Purpose
A deterministic, lockstep-safe 6-DOF fixed-wing FDM that feeds ArduPilot SITL over the JSON backend (`--model JSON`, UDP :9002) with **physically-consistent IMU**. Objective: **native-substrate equivalence** — reproduce what `--model quadplane` flies (SIM_QuadPlane's cruise aero is unchanged SIM_Plane) — **NOT real-world aero fidelity**. SIM_Plane is, by ArduPilot's own statement, "not aerodynamically accurate." The structural win over the native backend is lockstep determinism + sim-time/wall-clock decoupling for speedup and off-policy TD3.

## Modules
- `aero_plane.py` — SIM_Plane force/moment model (pure): `Coefficients` (SIM_Plane.h:48-92 verbatim), `lift_coeff`, `drag_coeff`, `get_force`, `get_torque`, `body_accel_from_forces`, `thrust_scale`.
- `rigidbody6dof.py` — `update_dynamics` port (pure, numpy): DCM algebra (from/to_euler, rotate, normalize), `RigidBody6DOF.step`, and the **specific-force IMU** (`specific_force`).
- `dryden_wind.py` — seeded Dryden forming-filter wind (default-OFF; deterministic, wall-clock-free).
- `plane6dof_fdm.py` — orchestrator: reuses `minimal_fdm` protocol/clock (imported, unedited), runs the UDP loop, exposes `step_physics()` for offline use, CLI + `--selftest`.

## Source-trace summary
Every load-bearing element maps to `ardupilot_formation/libraries/SITL/SIM_{Plane,Aircraft,JSON}.{cpp,h}`. Highlights:
- Force/moment: `SIM_Plane.cpp` getForce(344-396)/getTorque(273-341)/liftCoeff(235-255)/dragCoeff(257-270).
- Integrator + IMU: `SIM_Aircraft.cpp` update_dynamics(704-765); **specific force = dcmᵀ·(accel_earth − g)** (748).
- Lockstep clock: `SIM_JSON.cpp` 461-469; `no_time_sync` absent ⇒ EKF3 (378-388).
- **Critical simplification:** `getTorque` returns moments used **directly as angular accel** (`gyro += rot_accel·dt`, SIM_Aircraft.cpp:721) — SIM_Plane folds inertia into the coefficients; the port replicates this (NO inertia tensor) for equivalence.
Full matrix: ART `source_trace/SIMPLANE_SOURCE_TRACE.md`.

## Conventions
NED earth frame (position/velocity m, m/s); FRD body frame (gyro rad/s, accel_body specific force m/s²); DCM body→earth. Position emitted as **NED meters from home** (NOT LLA). Level steady IMU = (0,0,−9.80665). Servos: ail=ch0, elev=ch1, thr=ch2, rud=ch3; PWM 1000–2000 → angle −1..1 / range 0..1. Full detail: ART `source_trace/FRAME_CONVENTIONS.md`.

## Omissions (J4A)
ServoModel actuator lag (instantaneous servos), native IIR turbulence (replaced by Dryden, default-off), sensor noise (default-off for determinism), ground/takeoff/variants, eas2tas/altitude air-density, pitot AoA-limit, QuadPlane quad-lift + frame-mass×1.5 (mass default = SIM_Plane 2.0 kg). None affects wind-off cruise equivalence; the mass delta (O9) is the headline caveat. Full register: ART `source_trace/OMISSION_REGISTER.md`.

## Non-claims
No real-world aero fidelity. No docking, no M3b, no native-equivalence (yet), no TD3 readiness, no EKF-convergence claim, no aircraft-fidelity validation beyond offline regression. SysID is a port-faithfulness cross-check, not independent validation.

## J4A / J4B gate definitions
- **J4A (this dispatch, offline only):** build the source-traced modules; prove via unit tests (coefficient exactness + control-derivative signs), offline S1–S7 (boundedness, faithful signs, determinism, frame-rate convergence), and a sysid cross-check; State-Leakage audit. **No SITL.**
- **J4B (separate operator authorization required):** SITL boundary proof on the flying FDM — EKF3 convergence + live IMU-consistency, dynamic position-tracking, the J3 GPS-glitch re-confirmation under dynamics, and the controlled native-vs-JSON A/B that the offline sysid could not provide (resolves the O9 mass question + crosswind-crab reproduction). Carries an RR-1 reward-channel gate before any TD3.
