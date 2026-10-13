#!/usr/bin/env python3
"""
FlightDock Timebase Smoketest Tool
==================================

Validates SIM_SPEEDUP-compatible timing before running full tests.

USAGE:
    # Basic test (10s, expects 1x speedup)
    python tools/timebase_smoketest.py --conn udp:127.0.0.1:14550

    # Test at 5x speedup for 30 wall-seconds
    python tools/timebase_smoketest.py --conn udp:127.0.0.1:14550 \
        --expected-speedup 5.0 --duration 30

    # Include leader freshness check
    python tools/timebase_smoketest.py --conn udp:127.0.0.1:14550 \
        --leader-conn udp:127.0.0.1:14555 --expected-speedup 5.0

OUTPUT:
    Prints per-second metrics:
    - sim_t: current sim-time (from follower time_boot_ms)
    - wall_t: wall-time elapsed
    - rate: instantaneous sim/wall ratio
    - rate_med: rolling median rate
    - settle: PENDING or PASSED
    - quality: 0.0-1.0 score

    Final summary with PASS/FAIL verdict.

(C) 2025-2026 FlightDock Research Project
"""

import argparse

# Add tests directory to path for sim_time_authority import
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

try:
    from sim_time_authority import SimTimeAuthority, TimeQualityMetrics
except ImportError as e:
    print(f"ERROR: Could not import sim_time_authority: {e}")
    print("Ensure tests/sim_time_authority.py exists")
    sys.exit(1)

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not installed")
    sys.exit(1)


def run_smoketest(
    conn_str: str,
    expected_speedup: float = 1.0,
    duration_wall_s: float = 10.0,
    sysid_filter: int = 1,
    leader_conn_str: Optional[str] = None,
    verbose: bool = True,
) -> tuple:
    """Run timebase smoketest.

    Args:
        conn_str: MAVLink connection string for follower
        expected_speedup: Expected SIM_SPEEDUP factor
        duration_wall_s: Test duration in wall-seconds
        sysid_filter: Only accept messages from this sysid
        leader_conn_str: Optional MAVLink connection for leader messages
        verbose: Print per-second status

    Returns:
        (passed: bool, metrics: TimeQualityMetrics, reason: str)
    """
    print("=" * 70)
    print("TIMEBASE SMOKETEST")
    print("=" * 70)
    print(f"Follower connection: {conn_str}")
    print(f"Expected speedup: {expected_speedup}x")
    print(f"Duration: {duration_wall_s}s (wall)")
    print(f"Sysid filter: {sysid_filter}")
    if leader_conn_str:
        print(f"Leader connection: {leader_conn_str}")
    print("-" * 70)

    # Initialize time authority
    authority = SimTimeAuthority(
        expected_speedup=expected_speedup,
        drl_mode=False,  # We want strict checking for smoketest
        verbose=True,
    )

    # Connect to follower
    try:
        mav_fol = mavutil.mavlink_connection(
            conn_str,
            source_system=254,  # Test tool sysid
            source_component=1,
        )
        print("[CONN] Follower link initialized")
    except Exception as e:
        return False, TimeQualityMetrics(), f"Connection failed: {e}"

    # Optional leader connection
    mav_lead = None
    if leader_conn_str:
        try:
            mav_lead = mavutil.mavlink_connection(
                leader_conn_str,
                source_system=254,
                source_component=2,
            )
            print("[CONN] Leader link initialized")
        except Exception as e:
            print(f"[WARN] Leader connection failed: {e}")

    # Wait for first message
    print("[WAIT] Waiting for first time_boot_ms...")
    wait_start = time.monotonic()
    wait_timeout = 10.0
    first_tbms = None

    while time.monotonic() - wait_start < wait_timeout:
        msg = mav_fol.recv_match(
            type=["GLOBAL_POSITION_INT", "SYSTEM_TIME", "HEARTBEAT"],
            blocking=False,
        )
        if msg is None:
            time.sleep(0.01)
            continue

        # Check sysid
        src_sysid = getattr(msg, "_header", None)
        if src_sysid:
            src_sysid = src_sysid.srcSystem
        else:
            src_sysid = getattr(msg, "get_srcSystem", lambda: 0)()
        if src_sysid != sysid_filter:
            continue

        # Get time_boot_ms
        tbms = getattr(msg, "time_boot_ms", None)
        if tbms and tbms > 0:
            first_tbms = tbms
            authority.update_from_mavlink(tbms)
            break
    else:
        return False, TimeQualityMetrics(), "Timeout waiting for first time_boot_ms"

    print(f"[READY] First tbms={first_tbms}ms (sysid={sysid_filter})")
    print("-" * 70)

    # Main test loop
    test_start_wall = time.monotonic()
    last_print_wall = test_start_wall
    print_interval = 1.0

    msg_count = 0

    while True:
        wall_now = time.monotonic()
        wall_elapsed = wall_now - test_start_wall

        if wall_elapsed >= duration_wall_s:
            break

        # Poll follower
        msg = mav_fol.recv_match(
            type=["GLOBAL_POSITION_INT", "SYSTEM_TIME"],
            blocking=False,
        )
        if msg:
            src_sysid = getattr(msg, "_header", None)
            if src_sysid:
                src_sysid = src_sysid.srcSystem
            else:
                src_sysid = getattr(msg, "get_srcSystem", lambda: 0)()

            if src_sysid == sysid_filter:
                tbms = getattr(msg, "time_boot_ms", None)
                if tbms and tbms > 0:
                    authority.update_from_mavlink(tbms)
                    msg_count += 1

        # Poll leader (if connected)
        if mav_lead:
            msg = mav_lead.recv_match(
                type=["GLOBAL_POSITION_INT"],
                blocking=False,
            )
            if msg:
                tbms = getattr(msg, "time_boot_ms", None)
                if tbms and tbms > 0:
                    authority.check_leader_freshness(tbms)

        # Periodic status print
        if verbose and wall_now - last_print_wall >= print_interval:
            last_print_wall = wall_now
            metrics = authority.get_quality_metrics()

            settle_str = "PASSED" if metrics.settle_gate_passed else "PENDING"
            print(
                f"[{wall_elapsed:5.1f}s] "
                f"sim_t={authority.now_s():7.1f}s "
                f"rate_med={metrics.rate_median:5.2f}x "
                f"settle={settle_str:7s} "
                f"quality={metrics.quality_score:.2f} "
                f"msgs={msg_count}"
            )

        time.sleep(0.01)

    # Final verdict
    print("-" * 70)
    metrics = authority.get_quality_metrics()
    passed, reason, rate_med = authority.get_sync_verdict()

    print("FINAL METRICS:")
    print(f"  Samples: {metrics.samples}")
    print(f"  Settle gate: {'PASSED' if metrics.settle_gate_passed else 'FAILED'}")
    print(f"  Rate median: {metrics.rate_median:.2f}x (expected {expected_speedup:.2f}x)")
    print(f"  Rate P05-P95: {metrics.rate_p05:.2f}x - {metrics.rate_p95:.2f}x")
    print(f"  Monotonicity violations: {metrics.mono_violations}")
    if mav_lead:
        print(f"  Leader age median: {metrics.leader_age_median_ms:.0f}ms")
        print(f"  Leader age P95: {metrics.leader_age_p95_ms:.0f}ms")
        print(f"  Leader stale count: {metrics.leader_stale_count}")
    print(f"  Quality score: {metrics.quality_score:.2f}")
    print("-" * 70)

    if passed:
        print(f"VERDICT: PASS ({reason})")
    else:
        # OPTION A: Display clear abort banner for all failures
        print("")
        print("=" * 70)
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║ ABORT: TIMEBASE GATE FAILED                                       ║")
        print("╠═══════════════════════════════════════════════════════════════════╣")
        print(f"║ Reason: {reason:<58s} ║")
        print("║                                                                   ║")

        # Provide specific guidance based on failure reason
        if reason == "MONOTONICITY":
            print("║ Monotonicity violations detected (backwards time jumps).          ║")
            print("║ This indicates unstable MAVLink timing or message reordering.    ║")
            print("║                                                                   ║")
            print("║ Recommended actions:                                              ║")
            print("║   1. Reduce SIM_SPEEDUP                                           ║")
            print("║   2. Check for CPU throttling or overload                         ║")
            print("║   3. Verify SITL is not being starved of CPU time                 ║")
        elif reason.startswith("QUALITY_LOW"):
            print("║ Overall timing quality is below threshold (< 0.95).               ║")
            print("║ This indicates rate drift, leader staleness, or monotonicity     ║")
            print("║ issues that would compromise test validity.                      ║")
            print("║                                                                   ║")
            print("║ Recommended actions:                                              ║")
            print("║   1. Reduce SIM_SPEEDUP to improve stability                      ║")
            print("║   2. Check system load and available CPU resources                ║")
            print("║   3. Verify MAVLink connection quality                            ║")
        elif reason.startswith("RATE_DRIFT"):
            print("║ Measured speedup rate differs from expected by >25%.              ║")
            print("║ System cannot sustain the requested SIM_SPEEDUP.                  ║")
            print("║                                                                   ║")
            print("║ Recommended actions:                                              ║")
            print("║   1. Reduce SIM_SPEEDUP                                           ║")
            print("║   2. Run speedup sweep to calibrate system capability             ║")
        else:
            print("║ Timing infrastructure not ready or unstable.                      ║")
            print("║                                                                   ║")
            print("║ Recommended actions:                                              ║")
            print("║   1. Check SITL is running and responding                         ║")
            print("║   2. Verify MAVLink connection                                    ║")

        print("╚═══════════════════════════════════════════════════════════════════╝")
        print("=" * 70)
        print("")
        print(f"VERDICT: FAIL ({reason})")

    return passed, metrics, reason


def main():
    parser = argparse.ArgumentParser(
        description="FlightDock Timebase Smoketest Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--conn",
        "-c",
        required=True,
        help="MAVLink connection string for follower " "(e.g., udp:127.0.0.1:14550)",
    )
    parser.add_argument(
        "--expected-speedup",
        "-s",
        type=float,
        default=1.0,
        help="Expected SIM_SPEEDUP factor (default: 1.0)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="Test duration in wall-seconds (default: 10)",
    )
    parser.add_argument(
        "--sysid",
        "-i",
        type=int,
        default=1,
        help="Only accept messages from this sysid (default: 1)",
    )
    parser.add_argument(
        "--leader-conn",
        "-l",
        default=None,
        help="Optional MAVLink connection for leader freshness check",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-second status output",
    )

    args = parser.parse_args()

    passed, metrics, reason = run_smoketest(
        conn_str=args.conn,
        expected_speedup=args.expected_speedup,
        duration_wall_s=args.duration,
        sysid_filter=args.sysid,
        leader_conn_str=args.leader_conn,
        verbose=not args.quiet,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
