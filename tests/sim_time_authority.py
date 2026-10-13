#!/usr/bin/env python3
"""
FlightDock Sim-Time Authority Module
====================================

Single master clock for SIM_SPEEDUP-compatible timing.

SIM-TIME CONTRACT:
    This module enforces a strict contract for time usage in formation control:

    1. AUTHORITATIVE SOURCE: Follower SITL time_boot_ms is THE ONLY source of
       sim-time truth. All dwell timers, slew rate limits, and gating logic
       MUST use SimTimeAuthority.now_s() or dt_clamped().

    2. WALL-TIME IS FORBIDDEN in decision paths. Wall-time may only be used for:
       - Logging timestamps (cross-process correlation)
       - I/O rate limiting (flood guards, throttles)
       - Preflight timeouts (before sim-time becomes available)
       Use wall_now_s() or wall_monotonic_s() for these - they increment a
       guard counter for contract auditing.

    3. LEADER FRESHNESS, NOT DRIFT: Leader messages are checked for freshness
       relative to follower time (age_ms < threshold), NOT for cross-process
       drift which is undefined under SIM_SPEEDUP.

    4. QUALITY GATING: After a settle period, the system enforces hard gates on
       timing quality (rate accuracy, monotonicity) before test results are
       considered valid.

DESIGN GOALS:
1. Follower SITL time_boot_ms is THE authoritative time source
2. All dwell timers, slew limits, gates use this single clock
3. No cross-process drift comparisons - only freshness checks
4. DRL training mode: no aborts, only quality metrics

USAGE:
    from sim_time_authority import SimTimeAuthority, get_time_authority

    # Initialize once at startup
    authority = SimTimeAuthority(expected_speedup=5.0, drl_mode=True)
    set_time_authority(authority)

    # Update from MAVLink messages
    authority.update_from_mavlink(msg.time_boot_ms)

    # Get current sim-time
    now = authority.now_s()

    # Check leader message freshness
    fresh, age_ms = authority.check_leader_freshness(leader_msg_tbms)

    # Get time quality metrics for DRL
    metrics = authority.get_quality_metrics()

(C) 2025-2026 FlightDock Research Project
"""
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mavlink_cadence_stats import MavlinkCadenceStats  # type: ignore

__all__ = [
    "SimTimeAuthority",
    "TimeQualityMetrics",
    "get_time_authority",
    "set_time_authority",
    # TEST 387: Wall-time guard helpers for sim-time contract enforcement
    "wall_now_s",
    "wall_monotonic_s",
    "get_wall_time_guard_count",
    "reset_wall_time_guard_count",
]


@dataclass
class TimeQualityMetrics:
    """Time quality metrics for DRL rollout validation."""

    # Basic info
    ready: bool = False
    samples: int = 0
    settle_gate_passed: bool = False

    # Rate measurements (d_sim / d_wall)
    rate_median: float = 0.0
    rate_p05: float = 0.0
    rate_p95: float = 0.0
    rate_expected: float = 1.0

    # Leader freshness (ms)
    leader_age_median_ms: float = 0.0
    leader_age_p95_ms: float = 0.0
    leader_stale_count: int = 0

    # Monotonicity
    mono_violations: int = 0

    # Quality score (0.0 = bad, 1.0 = perfect)
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "ready": self.ready,
            "samples": self.samples,
            "settle_gate_passed": self.settle_gate_passed,
            "rate_median": round(self.rate_median, 3),
            "rate_p05": round(self.rate_p05, 3),
            "rate_p95": round(self.rate_p95, 3),
            "rate_expected": round(self.rate_expected, 3),
            "leader_age_median_ms": round(self.leader_age_median_ms, 1),
            "leader_age_p95_ms": round(self.leader_age_p95_ms, 1),
            "leader_stale_count": self.leader_stale_count,
            "mono_violations": self.mono_violations,
            "quality_score": round(self.quality_score, 3),
        }


class SimTimeAuthority:
    """Single master clock for SIM_SPEEDUP-compatible timing.

    Key design decisions:
    - Uses FOLLOWER time_boot_ms as the ONLY authoritative time source
    - Leader messages are checked for FRESHNESS, not drift
    - Rate verdict computed from rolling window AFTER settle gate
    - DRL mode: never abort, only emit quality metrics
    """

    # Configuration constants
    SETTLE_SAMPLES_MIN = 10  # Min samples before settle gate passes
    SETTLE_SIM_TIME_MIN_S = 2.0  # Min sim-time advance for settle
    RATE_WINDOW_SIZE = 20  # Rolling window for rate stats
    LEADER_STALE_THRESH_MS = 500  # Leader msg older than this = stale
    DT_MIN_S = 0.001  # Min dt clamp (1ms)
    DT_MAX_S = 0.5  # Max dt clamp (500ms)
    TIME_AUTH_DT_MAX_SAMPLES = 2000  # Rolling window for dt stats (sim-time)
    TIME_AUTH_STALE_GAP_THRESH_S = 0.2  # Gaps > this counted as "stale" (4 Hz detector)

    # OPTION A HARD GATING (Test 336+): Protect validity of high-speed runs
    # These thresholds ensure timing quality is sufficient for accurate test results.
    # Failures at these gates indicate the system cannot sustain the requested speedup
    # reliably, and test results would not be trustworthy.
    QUALITY_GATE_MIN = 0.95  # Minimum quality score to pass (0.0-1.0)
    MONOTONICITY_GATE_MAX = 0  # Maximum allowed monotonicity violations

    def __init__(
        self,
        expected_speedup: float = 1.0,
        drl_mode: bool = False,
        verbose: bool = True,
    ) -> None:
        """Initialize time authority.

        Args:
            expected_speedup: Expected SIM_SPEEDUP factor (for diagnostics)
            drl_mode: If True, never abort on timing issues
            verbose: If True, print status messages
        """
        self.expected_speedup = expected_speedup
        self.drl_mode = drl_mode
        self.verbose = verbose

        # Follower time tracking
        self._follower_tbms: int = 0
        self._follower_tbms_prev: int = 0
        self._sim_time_s: float = 0.0
        self._ready: bool = False
        self._mono_violations: int = 0

        # FD-008-B: Single-writer API state
        self._queue_active: bool = False  # Set True by controller when queue mode begins
        self._ta_update_violations: int = 0  # Count of rejected TA updates

        # Wall time tracking
        self._wall_start: float = time.monotonic()
        self._wall_last_update: float = time.monotonic()
        self._last_now_s: float = 0.0  # Monotonicity guarantee

        # Rate measurement
        self._rate_samples: List[float] = []
        self._settle_gate_passed: bool = False
        self._first_tbms: Optional[int] = None
        self._first_wall: Optional[float] = None

        # Leader freshness tracking
        self._leader_age_samples: List[float] = []
        self._leader_stale_count: int = 0

        # Logging control
        self._logged_ready: bool = False
        self._logged_settle: bool = False
        self._last_diag_log_wall: float = 0.0

        # TEST 387: Sim-time contract instrumentation
        self._dt_clamped_count: int = 0

        # Time-authority update metrics (for telemetry rate verification)
        self._time_authority_updates_total: int = 0
        self._time_authority_first_s: Optional[float] = None
        self._time_authority_last_s: Optional[float] = None

        # PRH-ATT Step 0B: Delegate dt/gap/stale tracking to MavlinkCadenceStats
        self._ta_cadence = MavlinkCadenceStats(
            stale_gap_thresh_s=self.TIME_AUTH_STALE_GAP_THRESH_S,
            window_s=5.0,
            max_dt_samples=self.TIME_AUTH_DT_MAX_SAMPLES,
            reset_on_backward=False,
        )

    def update_from_mavlink(self, time_boot_ms: int) -> bool:
        """Update from follower MAVLink time_boot_ms.

        Args:
            time_boot_ms: time_boot_ms field from follower MAVLink message

        Returns:
            True if update accepted, False if rejected (monotonicity violation)
        """
        if time_boot_ms is None or time_boot_ms < 0:
            return False

        wall_now = time.monotonic()

        # Track first sample for settle gate
        if self._first_tbms is None:
            self._first_tbms = time_boot_ms
            self._first_wall = wall_now

        # Monotonicity check
        if time_boot_ms < self._follower_tbms:
            self._mono_violations += 1
            # Allow small backwards (reordering), reject large jumps
            if self._follower_tbms - time_boot_ms > 1000:
                return False

        # Compute rate sample
        # IMPORTANT: Only update wall_last_update when we record a rate sample.
        # If we update it unconditionally, back-to-back messages with same tbms
        # will cause wall_delta to be shorter than sim_delta, inflating rates.
        if self._follower_tbms > 0:
            wall_delta = wall_now - self._wall_last_update
            sim_delta_s = (time_boot_ms - self._follower_tbms) / 1000.0

            # Only record rate if both deltas are significant
            # 1ms hard floor: prevents divide-by-near-zero on burst-drained messages
            # while accepting all legitimate inter-ATTITUDE wall deltas at any speedup
            if wall_delta > 0.001 and sim_delta_s > 0.001:
                rate = sim_delta_s / wall_delta
                self._rate_samples.append(rate)
                if len(self._rate_samples) > self.RATE_WINDOW_SIZE:
                    self._rate_samples.pop(0)
                # Only update wall reference when we record a rate sample
                self._wall_last_update = wall_now

        # Track time-authority update metrics before overwriting tbms
        self._track_time_authority_update(time_boot_ms)

        # Update state (tbms always updated)
        self._follower_tbms_prev = self._follower_tbms
        self._follower_tbms = time_boot_ms
        self._sim_time_s = time_boot_ms / 1000.0

        # Check ready condition
        if not self._ready and time_boot_ms > 0:
            self._ready = True
            if self.verbose and not self._logged_ready:
                self._logged_ready = True
                print(
                    f"[SIM_TIME_AUTHORITY] Ready: tbms={time_boot_ms}ms "
                    f"speedup_expected={self.expected_speedup}x"
                )

        # Check settle gate
        if not self._settle_gate_passed:
            samples_ok = len(self._rate_samples) >= self.SETTLE_SAMPLES_MIN
            time_ok = (
                self._sim_time_s - (self._first_tbms or 0) / 1000.0 >= self.SETTLE_SIM_TIME_MIN_S
            )
            if samples_ok and time_ok:
                self._settle_gate_passed = True
                # ChatGPT-5 R2: Reset gap/stale counters so they reflect steady-state only
                self._ta_cadence.reset_startup_counters()
                if self.verbose and not self._logged_settle:
                    self._logged_settle = True
                    rate_med = self._get_rate_median()
                    print(
                        f"[SIM_TIME_AUTHORITY] Settle gate PASSED: "
                        f"samples={len(self._rate_samples)} "
                        f"rate_median={rate_med:.2f}x"
                    )

        return True

    # =========================================================================
    # FD-008-B: Single-Writer API for Time Authority Updates
    # =========================================================================

    def set_queue_active(self, active: bool) -> None:
        """FD-008-B: Set queue-active flag (called by controller when queue mode begins).

        When queue is active, TA updates are only allowed from the tick scheduler
        source. Updates from other sources (settle, init) will be rejected to
        prevent multi-writer conflicts.

        Args:
            active: True when queue mode is active, False otherwise
        """
        self._queue_active = active
        if self.verbose:
            print(f"[SIM_TIME_AUTHORITY] FD-008-B: Queue active = {active}")

    def advance_from_time_boot_ms(self, ms: int, source: str) -> bool:
        """FD-008-B: Single-writer API for TA updates.

        This is the ONLY method that should be used to update the time authority
        during normal operation. It enforces single-writer semantics by rejecting
        updates from non-scheduler sources when queue mode is active.

        Args:
            ms: time_boot_ms from MAVLink ATTITUDE message
            source: Update source identifier:
                    - "tick_scheduler": Main loop queue drain (always allowed)
                    - "settle": Pre-queue settle phase (blocked when queue active)
                    - "init": Initial timebase setup (blocked when queue active)

        Returns:
            True if update accepted, False if rejected
        """
        # FD-008-B: Reject non-scheduler sources when queue is active
        if self._queue_active and source != "tick_scheduler":
            self._ta_update_violations += 1
            if self.verbose and self._ta_update_violations <= 5:
                print(
                    f"[SIM_TIME_AUTHORITY] FD-008-B REJECTED: source='{source}' "
                    f"while queue active (violation #{self._ta_update_violations})"
                )
            return False

        # Proceed with standard update
        return self.update_from_mavlink(ms)

    def get_ta_update_violations(self) -> int:
        """FD-008-B: Return count of rejected TA updates (for diagnostics)."""
        return self._ta_update_violations

    def is_queue_active(self) -> bool:
        """FD-008-B: Return whether queue mode is active."""
        return self._queue_active

    def now_s(self) -> float:
        """Get current sim-time in seconds.

        Returns sim-time from follower tbms, or wall-time fallback.
        Guarantees monotonically increasing values.
        """
        if self._ready:
            now = self._sim_time_s
        else:
            # Wall-time fallback
            now = time.monotonic() - self._wall_start

        # Monotonicity guarantee
        if now < self._last_now_s:
            now = self._last_now_s
        self._last_now_s = now
        return now

    def now_tbms(self) -> int:
        """Get current follower time_boot_ms (for leader timestamping)."""
        return self._follower_tbms

    def wall_elapsed_s(self) -> float:
        """Get wall-time elapsed since authority creation."""
        return time.monotonic() - self._wall_start

    def dt_clamped(self, prev_time_s: float) -> float:
        """Compute dt since a previous timestamp with sanity clamping."""
        self._dt_clamped_count += 1  # TEST 387: Instrumentation
        dt = self.now_s() - prev_time_s
        return max(self.DT_MIN_S, min(dt, self.DT_MAX_S))

    def get_dt_clamped_count(self) -> int:
        """Return count of dt_clamped() calls (TEST 387 instrumentation)."""
        return self._dt_clamped_count

    def is_ready(self) -> bool:
        """Return True if time authority is ready (has received tbms)."""
        return self._ready

    def is_settle_passed(self) -> bool:
        """Return True if settle gate has passed (rate is stable)."""
        return self._settle_gate_passed

    def check_leader_freshness(
        self, leader_msg_tbms: int, stale_thresh_ms: Optional[int] = None
    ) -> tuple:
        """Check leader message freshness relative to follower time.

        Args:
            leader_msg_tbms: time_boot_ms from leader's MAVLink message
            stale_thresh_ms: Override for staleness threshold

        Returns:
            (is_fresh: bool, age_ms: int)
        """
        if not self._ready or leader_msg_tbms <= 0:
            return False, 999999

        thresh = stale_thresh_ms or self.LEADER_STALE_THRESH_MS
        age_ms = self._follower_tbms - leader_msg_tbms

        # Track age stats
        self._leader_age_samples.append(age_ms)
        if len(self._leader_age_samples) > self.RATE_WINDOW_SIZE:
            self._leader_age_samples.pop(0)

        is_fresh = age_ms < thresh
        if not is_fresh:
            self._leader_stale_count += 1

        return is_fresh, age_ms

    def get_sync_verdict(self) -> tuple:
        """Get SYNC_VERDICT (PASS/FAIL) and reason.

        OPTION A HARD GATING (Test 336+):
        This method now enforces hard gates for quality_score and monotonicity
        violations to ensure the validity of high-speed test runs. These gates
        are applied AFTER the settle gate passes to avoid false failures during
        startup transients.

        Gate sequence (all must pass):
        1. Ready check (has received time_boot_ms)
        2. Settle gate (sufficient samples and sim-time advance)
        3. Monotonicity gate (no backwards time jumps)
        4. Quality gate (overall timing health >= 0.95)
        5. Rate drift check (within 25% of expected speedup)

        Returns:
            (pass: bool, reason: str, rate_median: float)
        """
        if not self._ready:
            return False, "NOT_READY", 0.0

        if not self._settle_gate_passed:
            return False, "SETTLE_PENDING", 0.0

        rate_med = self._get_rate_median()

        # OPTION A: Hard gate on monotonicity violations
        # Any backwards time jumps indicate unreliable timing
        if self._mono_violations > self.MONOTONICITY_GATE_MAX:
            return False, "MONOTONICITY", rate_med

        # OPTION A: Hard gate on quality score
        # Quality score incorporates rate accuracy, leader freshness, and monotonicity
        # A score below 0.95 indicates timing issues that would compromise test results
        metrics = self.get_quality_metrics()
        if metrics.quality_score < self.QUALITY_GATE_MIN:
            msg = f"QUALITY_LOW({metrics.quality_score:.2f})"
            return False, msg, rate_med

        # Check rate within tolerance (25% of expected)
        if self.expected_speedup > 0.1:
            rate_err = abs(rate_med - self.expected_speedup) / self.expected_speedup
            if rate_err > 0.25:
                msg = f"RATE_DRIFT({rate_med:.2f}x_vs_{self.expected_speedup:.2f}x)"
                return False, msg, rate_med

        return True, "OK", rate_med

    def get_quality_metrics(self) -> TimeQualityMetrics:
        """Get time quality metrics for DRL validation."""
        metrics = TimeQualityMetrics()
        metrics.ready = self._ready
        metrics.samples = len(self._rate_samples)
        metrics.settle_gate_passed = self._settle_gate_passed
        metrics.rate_expected = self.expected_speedup
        metrics.mono_violations = self._mono_violations

        if self._rate_samples:
            sorted_rates = sorted(self._rate_samples)
            metrics.rate_median = statistics.median(self._rate_samples)
            n = len(sorted_rates)
            metrics.rate_p05 = sorted_rates[max(0, int(n * 0.05))]
            metrics.rate_p95 = sorted_rates[min(n - 1, int(n * 0.95))]

        if self._leader_age_samples:
            sorted_ages = sorted(self._leader_age_samples)
            metrics.leader_age_median_ms = statistics.median(self._leader_age_samples)
            n = len(sorted_ages)
            metrics.leader_age_p95_ms = sorted_ages[min(n - 1, int(n * 0.95))]
        metrics.leader_stale_count = self._leader_stale_count

        # Compute quality score (0.0 = bad, 1.0 = perfect)
        score = 1.0
        if not self._ready:
            score = 0.0
        elif not self._settle_gate_passed:
            score = 0.2
        else:
            # Rate accuracy penalty
            if self.expected_speedup > 0.1:
                rate_err = abs(metrics.rate_median - self.expected_speedup) / self.expected_speedup
                score -= min(0.4, rate_err)

            # Leader staleness penalty
            if metrics.leader_age_p95_ms > 1000:
                score -= 0.2
            elif metrics.leader_age_p95_ms > 500:
                score -= 0.1

            # Monotonicity penalty
            if self._mono_violations > 10:
                score -= 0.2
            elif self._mono_violations > 0:
                score -= 0.05

        metrics.quality_score = max(0.0, score)
        return metrics

    def log_diagnostic(self, interval_s: float = 10.0) -> None:
        """Log diagnostic info (rate-limited)."""
        wall_now = time.monotonic()
        if wall_now - self._last_diag_log_wall < interval_s:
            return
        self._last_diag_log_wall = wall_now

        if not self._ready:
            print("[SIM_TIME_AUTHORITY] Waiting for follower time_boot_ms...")
            return

        rate_med = self._get_rate_median()
        age_med = statistics.median(self._leader_age_samples) if self._leader_age_samples else 0.0

        settle_str = "PASSED" if self._settle_gate_passed else "PENDING"
        print(
            f"[SIM_TIME_AUTHORITY] sim_t={self._sim_time_s:.1f}s "
            f"wall_t={self.wall_elapsed_s():.1f}s "
            f"rate={rate_med:.2f}x (exp={self.expected_speedup:.2f}x) "
            f"settle={settle_str} "
            f"leader_age_med={age_med:.0f}ms "
            f"mono_violations={self._mono_violations}"
        )

    def _get_rate_median(self) -> float:
        """Get median rate from rolling window."""
        if not self._rate_samples:
            return 0.0
        return statistics.median(self._rate_samples)

    def _track_time_authority_update(self, time_boot_ms: int) -> None:
        """Track telemetry update metrics derived from follower time_boot_ms.

        PRH-ATT Step 0B: Delegates dt/gap/stale tracking to MavlinkCadenceStats.
        Keeps _time_authority_updates_total, _first_s, _last_s for compatibility.
        """
        if time_boot_ms is None or time_boot_ms < 0:
            return

        sim_time_s = time_boot_ms / 1000.0

        # First sample path
        if self._time_authority_first_s is None:
            self._time_authority_first_s = sim_time_s
            self._time_authority_last_s = sim_time_s
            self._time_authority_updates_total = 1
            self._ta_cadence.observe_time_boot_ms(time_boot_ms)
            return

        if time_boot_ms <= self._follower_tbms:
            return

        # Delegate all dt/gap/stale tracking to cadence helper
        self._ta_cadence.observe_time_boot_ms(time_boot_ms)

        self._time_authority_updates_total += 1
        self._time_authority_last_s = sim_time_s

    def get_time_authority_update_hz_sim(self) -> float:
        """Return update rate in Hz computed over sim-time span.

        This uses the number of accepted time_boot_ms advances and the
        sim-time between the first and last update timestamps.
        """
        if self._time_authority_updates_total < 2:
            return 0.0
        if self._time_authority_first_s is None or self._time_authority_last_s is None:
            return 0.0
        span_s = self._time_authority_last_s - self._time_authority_first_s
        if span_s <= 0.0:
            return 0.0
        return (self._time_authority_updates_total - 1) / span_s

    def get_time_authority_dt_stats(self) -> tuple:
        """Return (min, median, p95, max) dt (sim-time) for time_boot_ms updates.

        PRH-ATT Step 0B: Delegates to MavlinkCadenceStats.dt_stats().
        """
        return self._ta_cadence.dt_stats()

    def get_time_authority_metrics(self) -> Dict[str, Any]:
        """Return time-authority update metrics for diagnostics.

        PRH-ATT Step 0B: Reads dt/gap/stale from MavlinkCadenceStats.
        All existing JSON keys preserved.
        """
        cm = self._ta_cadence.metrics()
        dt_min, dt_med, dt_p95, dt_max = self._ta_cadence.dt_stats()
        return {
            "time_authority_updates_total": self._time_authority_updates_total,
            "time_authority_first_s": self._time_authority_first_s,
            "time_authority_last_s": self._time_authority_last_s,
            "time_authority_update_hz_sim": self.get_time_authority_update_hz_sim(),
            "time_authority_dt_min": dt_min,
            "time_authority_dt_median": dt_med,
            "time_authority_dt_p95": dt_p95,
            "time_authority_dt_max": dt_max,
            "time_authority_dt_samples": max(0, self._time_authority_updates_total - 1),
            # Step 4.2: Gap/stale instrumentation (now from cadence helper)
            "time_authority_gap_max_s": cm["gap_max_s"],
            "time_authority_stale_gaps_over_0p2s": cm["stale_gaps"],
            # PRH-ATT: Windowed Hz from cadence helper
            "time_authority_min_window_hz": cm["min_window_hz"],
            # FD-008-B: Single-writer API metrics
            "ta_update_violations": self._ta_update_violations,
            "queue_active": self._queue_active,
        }


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================
_time_authority: Optional[SimTimeAuthority] = None


def get_time_authority() -> SimTimeAuthority:
    """Get the global SimTimeAuthority instance.

    Creates a default instance (speedup=1.0) if none has been set.
    This allows modules to access sim-time without requiring explicit
    initialization, while still allowing the test harness to configure
    the expected speedup via set_time_authority().

    Returns:
        The global SimTimeAuthority instance.
    """
    global _time_authority
    if _time_authority is None:
        _time_authority = SimTimeAuthority()
    return _time_authority


def set_time_authority(authority: SimTimeAuthority) -> None:
    """Set the global SimTimeAuthority instance.

    Call this during test harness initialization BEFORE any modules
    start using sim-time. Example:

        authority = SimTimeAuthority(expected_speedup=5.0, drl_mode=True)
        set_time_authority(authority)

    Args:
        authority: Configured SimTimeAuthority instance.
    """
    global _time_authority
    _time_authority = authority


# =============================================================================
# WALL-TIME GUARD HELPERS (Sim-Time Contract Enforcement)
# =============================================================================
# These functions provide instrumented wall-time access for ALLOWED non-decision
# uses only. They increment a global counter for sim-time contract auditing.
#
# ALLOWED USES (use these guarded functions):
#   - Logging timestamps for cross-process log correlation
#   - I/O rate limiting / flood guards
#   - Preflight timeouts (before sim-time available)
#   - Human-readable elapsed time display
#
# FORBIDDEN USES (use SimTimeAuthority.now_s() instead):
#   - Dwell timers (e.g., "hold for 5 seconds")
#   - Slew rate limits (e.g., "max 2 deg/sec")
#   - Gating decisions (e.g., "abort if timeout exceeded")
#   - Any logic that affects control behavior
#
# The guard counter (get_wall_time_guard_count()) enables auditing:
# if count increases faster than expected, wall-time is being misused.
# =============================================================================
_wall_time_guard_count: int = 0


def wall_now_s() -> float:
    """Wall-time (epoch seconds) for allowed non-decision uses.

    Use ONLY for:
    - Logging timestamps (cross-process correlation)
    - I/O rate limiting (flood guards)
    - Preflight timeouts (before sim-time is available)

    Increments guard counter for instrumentation.
    """
    global _wall_time_guard_count
    _wall_time_guard_count += 1
    return time.time()


def wall_monotonic_s() -> float:
    """Wall-time monotonic for allowed non-decision uses.

    Use ONLY for:
    - Wall-elapsed display (human readability)
    - I/O rate limiting (flood guards)

    Increments guard counter for instrumentation.
    """
    global _wall_time_guard_count
    _wall_time_guard_count += 1
    return time.monotonic()


def get_wall_time_guard_count() -> int:
    """Return count of guarded wall-time calls (TEST 387 instrumentation)."""
    return _wall_time_guard_count


def reset_wall_time_guard_count() -> None:
    """Reset guarded wall-time counter (for test isolation)."""
    global _wall_time_guard_count
    _wall_time_guard_count = 0


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("SimTimeAuthority self-test...")

    auth = SimTimeAuthority(expected_speedup=5.0, drl_mode=True)

    # Simulate time updates at 5x speedup
    base_tbms = 100000  # Start at 100 seconds
    for i in range(30):
        tbms = base_tbms + i * 250  # 250ms sim-time per 50ms wall-time = 5x
        auth.update_from_mavlink(tbms)
        time.sleep(0.05)

    # Check results
    metrics = auth.get_quality_metrics()
    print(f"Ready: {metrics.ready}")
    print(f"Settle passed: {metrics.settle_gate_passed}")
    print(f"Rate median: {metrics.rate_median:.2f}x (expected 5.0x)")
    print(f"Quality score: {metrics.quality_score:.2f}")

    passed, reason, rate = auth.get_sync_verdict()
    print(f"SYNC_VERDICT: {'PASS' if passed else 'FAIL'} ({reason})")

    print("Self-test complete.")
