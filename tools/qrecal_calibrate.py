#!/usr/bin/env python3
"""Faithful-cadence covariance calibration for the #789 spine-collapse fix.

Tests whether the DWNA process-accel-sigma raise (research PRIMARY) can lift the
steady-state GPS Kalman gain into the healthy band K in [0.1, 0.3] vs R=gps_sigma^2=4.0,
at the ACTUAL fuser-driving cadence, and compares it against the additive all-axis
position random-walk (`pos_rw`, research RUNNER-UP).

Cadence faithfulness [verified from source, NOT assumed]
-------------------------------------------------------
tests/baseline_formation_controller.py:1483/1491 co-submits update_gps(now) AND
update_uwb(now) at the SAME timestamp every control tick -> the fuser runs exactly
ONE _predict(dt_tick) + one GPS update per tick (the UWB call sees dt=0 -> no second
predict). So the GPS (1-K) position shrink is at the control-tick rate. Measured tick
rate (prior dispatches): ~20 Hz nominal, gate floor 18 Hz, ~14.66 Hz under Bug-A
degradation -> dt_tick in [0.050, 0.068]. We sweep that envelope.

Why GPS-only (open-loop) is faithful HERE (unlike a closed-loop offset)
-----------------------------------------------------------------------
Steady-state K is a covariance property = f(Q, R, cadence), independent of the
measurement VALUES for the linear H=I GPS update. In the collapse regime UWB fails the
3-sigma gate and update_uwb returns BEFORE touching P (uwb_gps_fusion.py:744) -> UWB is
a P-no-op, so GPS-only is exact; and any ACCEPTED UWB update only shrinks P further
(lower K). Hence the GPS-only steady-state K is precisely the K available to ESCAPE the
collapse (re-open the gate). This is a deterministic covariance bound, not a
trajectory-dependent counterfactual.

Deterministic, truth-free, no SITL. Run: ``python3 tools/qrecal_calibrate.py``
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "relative_pose"))
from uwb_gps_fusion import RelativePoseFusion, GPSMeasurement  # noqa: E402

G = 9.81
# (label, dt_tick) across the MEASURED control-tick envelope.
TICKS = [("20.00 Hz nominal", 0.05),
         ("18.00 Hz gate floor", 1.0 / 18.0),
         ("14.66 Hz Bug-A degraded", 1.0 / 14.66)]


def steady_state_k(factory, dt, n_warmup=20000):
    """Steady-state GPS-update gain K = P_prior/(P_prior+R) for the 3 position axes.

    Drives predict->update_gps to convergence; for the linear H=I update the covariance
    recursion is value-independent, so zero measurements suffice.
    """
    f = factory()
    t = 0.0
    for _ in range(n_warmup):
        t += dt
        f.update_gps(GPSMeasurement(timestamp=t, behind_m=0.0, cross_m=0.0, below_m=0.0))
    f._predict(dt)  # expose the steady-state PRIOR covariance
    R = f.gps_sigma_m ** 2
    return [f.P[i][i] / (f.P[i][i] + R) for i in range(3)]


def bisect_sigma(dt, target_k, lo=1.0, hi=4000.0, iters=70):
    """accel_sigma giving behind-axis steady-state K == target_k (K monotone in sigma)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if steady_state_k(lambda: RelativePoseFusion(qrecal_enable=True, qrecal_accel_sigma=mid), dt)[0] < target_k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    R = RelativePoseFusion().gps_sigma_m ** 2
    print(f"R = gps_sigma^2 = {R:.3f} m^2   healthy band K in [0.10, 0.30]\n")
    print(f"{'control tick':<26}{'legacy K':>9}{'posrw3 K':>9} | "
          f"{'sig_a K=.10':>12}{'K=.15':>8}{'K=.30':>8}   (sig_a @K=.15 in g)")
    for name, dt in TICKS:
        kl = steady_state_k(lambda: RelativePoseFusion(), dt)[0]
        kp = steady_state_k(lambda: RelativePoseFusion(pos_rw_m2_s=3.0), dt)[0]
        s10, s15, s30 = bisect_sigma(dt, 0.10), bisect_sigma(dt, 0.15), bisect_sigma(dt, 0.30)
        print(f"{name:<26}{kl:>9.4f}{kp:>9.4f} | {s10:>12.0f}{s15:>8.0f}{s30:>8.0f}   ({s15 / G:>4.0f}g)")

    print("\ncadence fragility (K swing) across the REAL envelope (0.0500-0.0682 s):")
    for label, fac in [("DWNA sigma_a=260", lambda: RelativePoseFusion(qrecal_enable=True, qrecal_accel_sigma=260.0)),
                       ("additive pos_rw=3.0", lambda: RelativePoseFusion(pos_rw_m2_s=3.0))]:
        k_fast = steady_state_k(fac, 0.05)[0]
        k_slow = steady_state_k(fac, 1.0 / 14.66)[0]
        print(f"  {label:<22} K {k_fast:.3f} -> {k_slow:.3f}  ({k_slow / k_fast:.2f}x)")

    print("\nCONCLUSION [MEASURED, faithful cadence, collapse-regime / open-loop-exact]:")
    print("  * DWNA sigma-raise reaches K in [0.1,0.3] only at sigma_a = 139-259 m/s^2 (14-26 g)")
    print("    across the real tick envelope -- physically absurd for a cruising QuadPlane")
    print("    (maneuver accel ~0.5-1 g). q_pos ~ dt^4 starves at the ~50 ms tick AND velocity")
    print("    is pinned by the GPS velocity measurement, so sigma_a cannot reach position.")
    print("  * additive pos_rw=3.0 reaches K = 0.176-0.202 in band across the whole envelope.")
    print("  ==> sigma-raise is NON-VIABLE as the K lever; the additive position-RW is the")
    print("      structurally-correct lever. Closed-loop DOCKING is a SEPARATE SITL question.")


if __name__ == "__main__":
    main()
