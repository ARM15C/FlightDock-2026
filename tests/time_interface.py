#!/usr/bin/env python3
"""
FlightDock Time Interface Module (Phase 1 Refactor)
====================================================

Thin wrapper around SimTimeAuthority providing a clean interface for
sim-time access in the formation controller.

DESIGN GOALS:
1. Single point of access for all timing operations in decision paths
2. Wall-time is guarded and instrumented for audit trail
3. dt computation with sanity clamping (1ms-500ms)
4. Clear separation: sim-time for decisions, wall-time for logging/I/O only

USAGE:
    from time_interface import (
        sim_now_s, sim_dt, is_sim_time_ready, wait_for_timebase_pass,
        wall_now_s, wall_monotonic_s, get_wall_time_guard_count,
    )

    # In decision paths (ONLY after PASS gate)
    now = sim_now_s()
    dt = sim_dt(prev_time_s)

    # For logging/I/O (guarded)
    epoch = wall_now_s() # Increments audit counter

(C) 2025-2026 FlightDock Research Project
"""
from dataclasses import dataclass
from typing import Optional

from sim_time_authority import (
    SimTimeAuthority,
    TimeQualityMetrics,
    get_time_authority,
    get_wall_time_guard_count,
    reset_wall_time_guard_count,
    set_time_authority,
    wall_monotonic_s,
    wall_now_s,
)

__all__ = [
    # Core sim-time access (decision paths ONLY)
    "sim_now_s",
    "sim_dt",
    # Gating and synchronization
    "wait_for_timebase_pass",
    "is_sim_time_ready",
    # Quality metrics (for logging, not decisions)
    "get_quality_metrics",
    "get_sync_verdict",
    # Wall-time access (logging/I/O ONLY, guarded)
    "wall_now_s",
    "wall_monotonic_s",
    "get_wall_time_guard_count",
    "reset_wall_time_guard_count",
    # Initialization
    "set_expected_speedup",
    "initialize_from_authority",
    "get_time_authority",
    "set_time_authority",
    # Re-export types
    "SimTimeAuthority",
    "TimeQualityMetrics",
]


# =============================================================================
# CORE SIM-TIME ACCESS (DECISION PATHS ONLY)
# =============================================================================


def sim_now_s() -> float:
    """Get current sim-time in seconds.

    CRITICAL: Only valid after PASS gate. Use is_sim_time_ready() first.

    Returns:
        Current simulation time in seconds from SimTimeAuthority.
        Falls back to wall-time if authority not ready (pre-PASS).
    """
    authority = get_time_authority()
    return authority.now_s()


def sim_dt(prev_time_s: float) -> float:
    """Compute dt since prev_time_s with sanity clamping (1ms-500ms).

    Args:
        prev_time_s: Previous timestamp in sim-time seconds

    Returns:
        Clamped dt in seconds (min=0.001, max=0.5)
    """
    authority = get_time_authority()
    return authority.dt_clamped(prev_time_s)


# =============================================================================
# GATING AND SYNCHRONIZATION
# =============================================================================


def wait_for_timebase_pass(timeout_wall_s: Optional[float] = None) -> bool:
    """Wait for sim-time authority to pass all quality gates.

    This should be called during initialization to ensure the sim-time
    authority has settled before making timing-critical decisions.

    Args:
        timeout_wall_s: Maximum wall-time to wait (optional, no timeout if None)

    Returns:
        True if PASS gate passed, False if timeout or failed.
    """
    import time

    authority = get_time_authority()
    start = time.monotonic()

    while True:
        passed, reason, rate = authority.get_sync_verdict()
        if passed:
            return True

        if timeout_wall_s is not None:
            elapsed = time.monotonic() - start
            if elapsed >= timeout_wall_s:
                print(f"[TIME_INTERFACE] Timeout waiting for PASS gate: {reason}")
                return False

        # Brief sleep to avoid busy-wait
        time.sleep(0.05)


def is_sim_time_ready() -> bool:
    """Check if sim-time is ready (has passed settle gate).

    Returns:
        True if sim-time authority has received time_boot_ms and is ready.
    """
    authority = get_time_authority()
    return authority.is_ready()


def is_settle_passed() -> bool:
    """Check if settle gate has passed (rate is stable).

    Returns:
        True if sim-time authority has passed settle gate.
    """
    authority = get_time_authority()
    return authority.is_settle_passed()


# =============================================================================
# QUALITY METRICS (FOR LOGGING, NOT DECISIONS)
# =============================================================================


def get_quality_metrics() -> TimeQualityMetrics:
    """Get sim-time health metrics for audit trail.

    Returns:
        TimeQualityMetrics with rate, freshness, and quality info.
    """
    authority = get_time_authority()
    return authority.get_quality_metrics()


def get_sync_verdict() -> tuple:
    """Get SYNC_VERDICT (PASS/FAIL) and reason.

    Returns:
        Tuple of (passed: bool, reason: str, rate_median: float)
    """
    authority = get_time_authority()
    return authority.get_sync_verdict()


# =============================================================================
# INITIALIZATION
# =============================================================================


def set_expected_speedup(speedup: float) -> None:
    """Set expected speedup factor for the time authority.

    Should be called before the authority starts receiving MAVLink updates.

    Args:
        speedup: Expected SIM_SPEEDUP factor (e.g., 1.0, 3.0, 5.0)
    """
    authority = get_time_authority()
    authority.expected_speedup = speedup


def initialize_from_authority(authority: SimTimeAuthority) -> None:
    """Initialize the time interface with a specific authority instance.

    Args:
        authority: SimTimeAuthority instance to use globally.
    """
    set_time_authority(authority)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_dt_clamped_count() -> int:
    """Get count of dt_clamped() calls for instrumentation.

    Returns:
        Number of times sim_dt() or dt_clamped() was called.
    """
    authority = get_time_authority()
    return authority.get_dt_clamped_count()


def update_from_mavlink(time_boot_ms: int) -> bool:
    """Update time authority from MAVLink time_boot_ms.

    Args:
        time_boot_ms: time_boot_ms field from follower MAVLink message

    Returns:
        True if update accepted, False if rejected (monotonicity violation)
    """
    authority = get_time_authority()
    return authority.update_from_mavlink(time_boot_ms)


def check_leader_freshness(leader_msg_tbms: int, stale_thresh_ms: Optional[int] = None) -> tuple:
    """Check leader message freshness relative to follower time.

    Args:
        leader_msg_tbms: time_boot_ms from leader's MAVLink message
        stale_thresh_ms: Override for staleness threshold

    Returns:
        (is_fresh: bool, age_ms: int)
    """
    authority = get_time_authority()
    return authority.check_leader_freshness(leader_msg_tbms, stale_thresh_ms)


def now_tbms() -> int:
    """Get current follower time_boot_ms (for leader timestamping).

    Returns:
        Current time_boot_ms from the time authority.
    """
    authority = get_time_authority()
    return authority.now_tbms()


def wall_elapsed_s() -> float:
    """Get wall-time elapsed since authority creation.

    Returns:
        Elapsed wall-time in seconds (for display/logging only).
    """
    authority = get_time_authority()
    return authority.wall_elapsed_s()


def log_diagnostic(interval_s: float = 10.0) -> None:
    """Log diagnostic info (rate-limited).

    Args:
        interval_s: Minimum interval between log messages.
    """
    authority = get_time_authority()
    authority.log_diagnostic(interval_s)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("time_interface module self-test...")

    # Create and set authority
    auth = SimTimeAuthority(expected_speedup=1.0, drl_mode=True, verbose=True)
    initialize_from_authority(auth)

    print(f"sim_time_ready: {is_sim_time_ready()}")
    print(f"settle_passed: {is_settle_passed()}")

    # Simulate some updates
    import time

    base_tbms = 1000
    for i in range(20):
        update_from_mavlink(base_tbms + i * 100)
        time.sleep(0.05)

    print(f"sim_time_ready: {is_sim_time_ready()}")
    print(f"settle_passed: {is_settle_passed()}")
    print(f"sim_now_s: {sim_now_s():.3f}")

    metrics = get_quality_metrics()
    print(f"quality_score: {metrics.quality_score:.3f}")

    passed, reason, rate = get_sync_verdict()
    print(f"sync_verdict: {'PASS' if passed else 'FAIL'} ({reason})")

    print("Self-test complete.")
