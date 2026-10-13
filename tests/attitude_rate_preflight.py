#!/usr/bin/env python3
"""
FlightDock PRH-ATT-001: ATTITUDE Rate Preflight CLI
=====================================================

Standalone CLI tool that proves ATTITUDE reception is >= 18 Hz and stable
before any validation-grade run proceeds. Used by:
  - Harness preflight gate (PRH-ATT-002)
  - Canary delegation (PRH-ATT-003)
  - Manual operator verification

USAGE:
    python3 tests/attitude_rate_preflight.py \\
        --conn udp:127.0.0.1:14551 --speedup 1.0 \\
        --settle-sim-s 5 --measure-sim-s 30 \\
        --min-hz 18 --max-dt-p95-s 0.08 --stale-gap-thresh-s 0.2 \\
        --window-sim-s 5 --mode validation \\
        --report /path/to/report.json

EXIT CODES:
    0 = PASS (or FAIL in debug mode with live link)
    1 = FAIL in validation mode
    2 = Link dead (NO_HEARTBEAT / NO_ATTITUDE) in any mode

(C) 2025-2026 FlightDock Research Project
"""

import argparse
import json
import os
import sys
import time

# Add script directory to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mavlink_cadence_stats import CadencePassCriteria, MavlinkCadenceStats  # type: ignore

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not installed. Run: pip install pymavlink")
    sys.exit(2)


# ============================================================================
# Fail reason enum
# ============================================================================
class FailReason:
    NO_HEARTBEAT = "NO_HEARTBEAT"
    NO_ATTITUDE = "NO_ATTITUDE"
    SRX_READBACK_TIMEOUT = "SRX_READBACK_TIMEOUT"
    SRX_READBACK_MISMATCH = "SRX_READBACK_MISMATCH"
    HZ_LOW = "HZ_LOW"
    DT_P95_HIGH = "DT_P95_HIGH"
    STALE_GAPS_PRESENT = "STALE_GAPS_PRESENT"
    WINDOW_HZ_LOW = "WINDOW_HZ_LOW"
    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"


# ============================================================================
# Step 1: Configure ATTITUDE rate
# ============================================================================


def configure_attitude_rate(mav, target_sys, target_comp, param_conn_str=None):
    """Send MAV_CMD_SET_MESSAGE_INTERVAL to request ATTITUDE at 20 Hz.

    NOTE: SR*_EXTRA1 are MAVProxy-only parameters (not ArduPilot params).
    They cannot be set or read through any direct SITL connection (TCP 5763)
    or through split MAVProxy paths (udpout→14560 / udp←14551).
    MAV_CMD_SET_MESSAGE_INTERVAL is the reliable mechanism — it goes through
    MAVProxy to ArduPilot and directly controls the ATTITUDE stream rate.
    The actual rate is validated by measurement in step 2.

    Returns:
        dict with 'mav_cmd_set_interval_us' and optional 'fail_reason'.
    """
    actions = {
        "mav_cmd_set_interval_us": 50000,
    }

    # Send MAV_CMD_SET_MESSAGE_INTERVAL for ATTITUDE at 50,000 us (20 Hz)
    MSG_ID_ATTITUDE = mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE
    mav.mav.command_long_send(
        target_sys,
        target_comp,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        float(MSG_ID_ATTITUDE),
        50000.0,  # 50ms = 20 Hz
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    print("[PREFLIGHT] Requested ATTITUDE at 20 Hz via MAV_CMD_SET_MESSAGE_INTERVAL")

    return actions, None


# ============================================================================
# Step 2: Measure ATTITUDE rate
# ============================================================================
def measure_attitude_rate(mav, args):
    """Collect ATTITUDE timestamps for measure_sim_s after settle.

    Returns:
        (cadence, msg_count, fail_reason)
    """
    settle_sim_s = args.settle_sim_s
    measure_sim_s = args.measure_sim_s
    speedup = args.speedup

    wall_timeout = max(120, measure_sim_s / max(speedup, 0.1) * 1.5 + 60)

    cadence = MavlinkCadenceStats(
        stale_gap_thresh_s=args.stale_gap_thresh_s,
        window_s=args.window_sim_s,
        reset_on_backward=False,
    )

    settled = False
    settle_start_tbms = None
    msg_count = 0

    print(
        f"[PREFLIGHT] Collecting for {measure_sim_s}s sim-time "
        f"(settle {settle_sim_s}s, wall timeout {wall_timeout:.0f}s)..."
    )

    wall_start = time.monotonic()

    while time.monotonic() - wall_start < wall_timeout:
        msg = mav.recv_match(type="ATTITUDE", blocking=True, timeout=1.0)
        if msg is None:
            continue
        if not hasattr(msg, "time_boot_ms"):
            continue

        tbms = msg.time_boot_ms
        msg_count += 1

        # First message
        if settle_start_tbms is None:
            settle_start_tbms = tbms
            print(f"[PREFLIGHT] First ATTITUDE at tbms={tbms}ms")
            continue

        # Settle gate
        if not settled:
            sim_elapsed = (tbms - settle_start_tbms) / 1000.0
            if sim_elapsed >= settle_sim_s:
                settled = True
                print(f"[PREFLIGHT] Settle complete at tbms={tbms}ms, starting measurement")
            continue

        # Feed into cadence tracker
        cadence.observe_time_boot_ms(tbms)

        # Check if we've collected enough
        if cadence._count >= 2:
            span_ms = (cadence._last_ms or 0) - (cadence._first_ms or 0)
            span_s = span_ms / 1000.0
            if span_s >= measure_sim_s:
                break

    if settle_start_tbms is None:
        return None, msg_count, FailReason.NO_ATTITUDE

    if not settled:
        return None, msg_count, FailReason.NO_ATTITUDE

    return cadence, msg_count, None


# ============================================================================
# Step 3: Evaluate pass/fail and produce report
# ============================================================================
def run_preflight(args):
    """Main preflight entry point. Returns (report_dict, exit_code)."""

    report = {
        "verdict": "FAIL",
        "mode": args.mode,
        "conn": args.conn,
        "speedup": args.speedup,
        "actions": None,
        "measurement": None,
        "pass_criteria": {
            "min_hz": args.min_hz,
            "max_dt_p95_s": args.max_dt_p95_s,
            "max_stale_gaps": 0,
            "min_window_hz": args.min_hz,
            "min_samples": 100,
        },
        "pass_results": None,
        "fail_reason": None,
    }

    # Connect
    print(f"[PREFLIGHT] Connecting to {args.conn}...")
    try:
        mav = mavutil.mavlink_connection(args.conn, source_system=254)
        mav.wait_heartbeat(timeout=10)
    except Exception as e:
        print(f"[PREFLIGHT] No heartbeat: {e}")
        report["fail_reason"] = FailReason.NO_HEARTBEAT
        return report, 2

    target_sys = mav.target_system
    target_comp = mav.target_component
    print(f"[PREFLIGHT] Heartbeat from system {target_sys}")

    # Configure ATTITUDE rate via MAV_CMD_SET_MESSAGE_INTERVAL
    actions, config_fail = configure_attitude_rate(mav, target_sys, target_comp)
    report["actions"] = actions

    if config_fail and args.mode == "validation":
        report["fail_reason"] = config_fail
        return report, 1

    # Measure
    cadence, msg_count, measure_fail = measure_attitude_rate(mav, args)

    if measure_fail:
        report["fail_reason"] = measure_fail
        exit_code = 2 if measure_fail in (FailReason.NO_HEARTBEAT, FailReason.NO_ATTITUDE) else 1
        if args.mode == "debug" and exit_code != 2:
            exit_code = 0
        return report, exit_code

    # Extract metrics
    m = cadence.metrics()
    dt_min, dt_med, dt_p95, dt_max = cadence.dt_stats()
    hz = cadence.hz_sim_est()
    min_window_hz = cadence.min_window_hz()

    report["measurement"] = {
        "count_total": m["count_total"],
        "hz_sim_est": round(hz, 2),
        "min_window_hz": round(min_window_hz, 2),
        "dt_median_s": round(dt_med, 6),
        "dt_p95_s": round(dt_p95, 6),
        "gap_max_s": round(m["gap_max_s"], 6),
        "stale_gaps": m["stale_gaps"],
    }

    # Evaluate pass criteria
    pass_hz = hz >= args.min_hz
    pass_window = min_window_hz >= args.min_hz
    pass_p95 = dt_p95 <= args.max_dt_p95_s
    pass_stale = m["stale_gaps"] == 0
    pass_samples = m["count_total"] >= 100

    report["pass_results"] = {
        "hz": pass_hz,
        "min_window_hz": pass_window,
        "dt_p95": pass_p95,
        "stale_gaps": pass_stale,
    }

    overall = pass_hz and pass_window and pass_p95 and pass_stale and pass_samples

    if not pass_samples:
        report["fail_reason"] = FailReason.INSUFFICIENT_SAMPLES
    elif not pass_hz:
        report["fail_reason"] = FailReason.HZ_LOW
    elif not pass_window:
        report["fail_reason"] = FailReason.WINDOW_HZ_LOW
    elif not pass_p95:
        report["fail_reason"] = FailReason.DT_P95_HIGH
    elif not pass_stale:
        report["fail_reason"] = FailReason.STALE_GAPS_PRESENT

    report["verdict"] = "PASS" if overall else "FAIL"

    # Console output
    print("")
    print("=============================================================")
    print("  ATTITUDE RATE PREFLIGHT RESULTS")
    print("=============================================================")
    print(f"  Mode:                        {args.mode}")
    print(f"  Total samples:               {m['count_total']}")
    print(f"  Hz (sim-time):               {hz:.2f}")
    print(f"  Min window Hz ({args.window_sim_s}s):       {min_window_hz:.2f}")
    print(f"  dt median:                   {dt_med:.4f}s")
    print(f"  dt p95:                      {dt_p95:.4f}s")
    print(f"  gap max:                     {m['gap_max_s']:.4f}s")
    print(f"  stale gaps (>{args.stale_gap_thresh_s}s):     {m['stale_gaps']}")
    print(f"  ---")
    print(f"  Hz >= {args.min_hz}:              {'PASS' if pass_hz else 'FAIL'}")
    print(f"  Window Hz >= {args.min_hz}:       {'PASS' if pass_window else 'FAIL'}")
    print(f"  dt_p95 <= {args.max_dt_p95_s}s:       {'PASS' if pass_p95 else 'FAIL'}")
    print(f"  stale_gaps == 0:             {'PASS' if pass_stale else 'FAIL'}")
    print(f"  samples >= 100:              {'PASS' if pass_samples else 'FAIL'}")
    print(f"  ---")
    print(f"  VERDICT: {report['verdict']}")
    print("=============================================================")

    # Exit code
    if overall:
        return report, 0
    elif args.mode == "debug":
        return report, 0
    else:
        return report, 1


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlightDock PRH-ATT-001: ATTITUDE Rate Preflight")
    parser.add_argument(
        "--conn", default="udp:127.0.0.1:14551", help="MAVLink connection for telemetry (ATTITUDE)"
    )
    parser.add_argument(
        "--param-conn",
        default=None,
        dest="param_conn",
        help="(deprecated, ignored) Previously used for param ops.",
    )
    parser.add_argument("--speedup", type=float, default=1.0, help="Expected SIM_SPEEDUP")
    parser.add_argument(
        "--settle-sim-s", type=float, default=5.0, help="Settle period in sim-seconds"
    )
    parser.add_argument(
        "--measure-sim-s", type=float, default=30.0, help="Measurement duration in sim-seconds"
    )
    parser.add_argument("--min-hz", type=float, default=18.0, help="Minimum acceptable Hz")
    parser.add_argument(
        "--max-dt-p95-s", type=float, default=0.08, help="Maximum acceptable dt p95 in seconds"
    )
    parser.add_argument(
        "--stale-gap-thresh-s", type=float, default=0.2, help="Gap threshold for stale detection"
    )
    parser.add_argument(
        "--window-sim-s", type=float, default=5.0, help="Sliding window size for min-Hz tracking"
    )
    parser.add_argument(
        "--mode",
        choices=["validation", "debug"],
        default="validation",
        help="validation=exit 1 on fail, debug=exit 0 unless link dead",
    )
    parser.add_argument("--report", default=None, help="Path to write JSON report")

    args = parser.parse_args()

    report, exit_code = run_preflight(args)

    # Write JSON report
    report_path = args.report
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[PREFLIGHT] Report written to {report_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
