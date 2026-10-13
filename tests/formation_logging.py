#!/usr/bin/env python3
"""
Formation Logging Module - Centralized Logging Utilities
=========================================================

Phase 7D.74: Originally extracted from predictive_formation_controller.py.
Now used by baseline_formation_controller.py.

This module provides a unified logging interface for the formation controller.
All log output goes through these functions to ensure consistent formatting.

Usage:
    from formation_logging import log, log_event, log_debug

    log("Simple message") # [1234567890.123] Simple message
    log_event("TAG", "Message") # [1234567890.123] [TAG] Message
    log_debug("DEBUG", "Verbose message") # Only if DEBUG_LOGGING_ENABLED
"""

import os
import sys
import time

# ============================================================================
# Phase 7D.74: Centralized Logging Configuration
# ============================================================================
# All debug flags can be overridden via environment variables.
# Example: FLIGHTDOCK_DEBUG_LOGGING=1 ./run_test_394.sh
# ============================================================================


def _env_bool(name: str, default: bool) -> bool:
    """Read boolean from environment variable (1/true/yes = True)."""
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


# Enable/disable debug-level logging
# Override: FLIGHTDOCK_DEBUG_LOGGING=1
DEBUG_LOGGING_ENABLED = _env_bool("FLIGHTDOCK_DEBUG_LOGGING", False)

# Enable/disable timestamp prefix
# Override: FLIGHTDOCK_LOG_TIMESTAMPS=0
TIMESTAMP_ENABLED = _env_bool("FLIGHTDOCK_LOG_TIMESTAMPS", True)

# Log flush behavior
# Override: FLIGHTDOCK_LOG_AUTOFLUSH=0
AUTO_FLUSH = _env_bool("FLIGHTDOCK_LOG_AUTOFLUSH", True)

# Rate limiting for high-frequency logs (tag -> last_log_time)
_log_rate_limits: dict = {}

# Default rate limit interval (seconds)
DEFAULT_RATE_LIMIT_S = 0.5


# ============================================================================
# Core Logging Functions
# ============================================================================


def log(msg: str) -> None:
    """
    Phase 7D.74: Timestamped logger for runtime messages.

    This is the primary logging function for the formation controller.
    Prepends a Unix timestamp for debugging timing issues.

    Args:
        msg: Message to log (may include [TAG] prefix already)
    """
    if TIMESTAMP_ENABLED:
        output = f"[{time.time():.3f}] {msg}"
    else:
        output = msg

    print(output, flush=AUTO_FLUSH)


def log_event(tag: str, msg: str) -> None:
    """
    Phase 7D.74: Tagged event logger with consistent formatting.

    Use this for structured log messages where the tag should be
    consistently formatted.

    Args:
        tag: Event tag (e.g., "7D22_HDG_SRC", "INTERCEPT", "SAFETY")
        msg: Message content
    """
    log(f"[{tag}] {msg}")


def log_debug(tag: str, msg: str) -> None:
    """
    Phase 7D.74: Debug-level logging (can be disabled).

    Use for verbose debugging output that should not appear in
    production runs.

    Args:
        tag: Debug tag
        msg: Debug message
    """
    if DEBUG_LOGGING_ENABLED:
        log(f"[DEBUG:{tag}] {msg}")


def log_warn(tag: str, msg: str) -> None:
    """
    Phase 7D.74: Warning-level logging with visual indicator.

    Args:
        tag: Warning tag
        msg: Warning message
    """
    log(f"⚠ [{tag}] {msg}")


def log_error(tag: str, msg: str) -> None:
    """
    Phase 7D.74: Error-level logging with visual indicator.

    Args:
        tag: Error tag
        msg: Error message
    """
    log(f"❌ [{tag}] {msg}")


def log_success(tag: str, msg: str) -> None:
    """
    Phase 7D.74: Success-level logging with visual indicator.

    Args:
        tag: Success tag
        msg: Success message
    """
    log(f"✓ [{tag}] {msg}")


def log_rate_limited(tag: str, msg: str, interval_s: float = None) -> bool:
    """
    Phase 7D.74: Rate-limited logging to prevent log spam.

    Returns True if the message was logged, False if rate-limited.

    Args:
        tag: Event tag (also used as rate limit key)
        msg: Message content
        interval_s: Min interval between logs (default: DEFAULT_RATE_LIMIT_S)

    Returns:
        True if logged, False if suppressed due to rate limiting
    """
    global _log_rate_limits

    if interval_s is None:
        interval_s = DEFAULT_RATE_LIMIT_S

    now = time.time()
    last_log = _log_rate_limits.get(tag, 0.0)

    if now - last_log >= interval_s:
        _log_rate_limits[tag] = now
        log_event(tag, msg)
        return True
    return False


def log_state_transition(from_state: str, to_state: str, reason: str = "") -> None:
    """
    Phase 7D.74: Log state machine transitions with consistent format.

    Args:
        from_state: Previous state name
        to_state: New state name
        reason: Optional reason for transition
    """
    if reason:
        log(f"[STATE] {from_state} -> {to_state} ({reason})")
    else:
        log(f"[STATE] {from_state} -> {to_state}")


def log_header(title: str, char: str = "=", width: int = 60) -> None:
    """
    Phase 7D.74: Log a visual header/separator.

    Args:
        title: Header title text
        char: Character to use for border (default "=")
        width: Total width of header (default 60)
    """
    border = char * width
    print(border)
    print(title)
    print(border)


def flush_logs() -> None:
    """Force flush of stdout and stderr."""
    sys.stdout.flush()
    sys.stderr.flush()


# ============================================================================
# Phase 7D.78: Specialized Phase Debug Loggers
# ============================================================================
# These provide detailed logging for specific flight phases.
# Enabled by default for development; disable in production for cleaner logs.
# Override: FLIGHTDOCK_CREEP_DEBUG=0 or FLIGHTDOCK_PRECONTACT_DEBUG=0
# ============================================================================

# Enable/disable phase-specific debug logging
CREEP_DEBUG_ENABLED = _env_bool("FLIGHTDOCK_CREEP_DEBUG", True)
PRECONTACT_DEBUG_ENABLED = _env_bool("FLIGHTDOCK_PRECONTACT_DEBUG", True)

# Rate limiting for phase debug logs (seconds)
CREEP_DEBUG_RATE_LIMIT_S = 0.5
PRECONTACT_DEBUG_RATE_LIMIT_S = 0.5

# Last log times for rate limiting
_creep_debug_last_log: float = 0.0
_precontact_debug_last_log: float = 0.0


def log_creep_debug(
    boom_x: float,
    boom_y: float,
    boom_z: float,
    vc_target: float,
    tas_cmd: float,
    msg: str = "",
    force: bool = False,
) -> bool:
    """
    Phase 7D.78: Specialized CREEP phase debug logger.

    Logs CREEP state with consistent format for debugging throttle/position
    oscillations. Rate-limited to avoid log spam.

    Args:
        boom_x: Longitudinal position (m, negative=behind)
        boom_y: Lateral offset (m)
        boom_z: Vertical offset (m)
        vc_target: Target closure rate (m/s)
        tas_cmd: Commanded true airspeed (m/s)
        msg: Optional additional message
        force: If True, bypass rate limiting

    Returns:
        True if logged, False if rate-limited
    """
    global _creep_debug_last_log

    if not CREEP_DEBUG_ENABLED and not force:
        return False

    now = time.time()
    if not force and (now - _creep_debug_last_log) < CREEP_DEBUG_RATE_LIMIT_S:
        return False

    _creep_debug_last_log = now

    extra = f" {msg}" if msg else ""
    log(
        f"[CREEP_DBG] bx={boom_x:+.2f} by={boom_y:+.2f} bz={boom_z:+.2f} "
        f"Vc={vc_target:+.3f} TAS={tas_cmd:.1f}{extra}"
    )
    return True


def log_precontact_debug(
    behind: float,
    cross: float,
    range_m: float,
    closure: float,
    in_box: bool,
    msg: str = "",
    force: bool = False,
) -> bool:
    """
    Phase 7D.78: Specialized HOLD_PRECONTACT phase debug logger.

    Logs precontact geometry with consistent format for debugging
    box entry/exit oscillations. Rate-limited to avoid log spam.

    Args:
        behind: Distance behind leader (m)
        cross: Cross-track offset (m)
        range_m: 3D range to leader (m)
        closure: Closure rate (m/s)
        in_box: Whether currently inside precontact box
        msg: Optional additional message
        force: If True, bypass rate limiting

    Returns:
        True if logged, False if rate-limited
    """
    global _precontact_debug_last_log

    if not PRECONTACT_DEBUG_ENABLED and not force:
        return False

    now = time.time()
    elapsed = now - _precontact_debug_last_log
    if not force and elapsed < PRECONTACT_DEBUG_RATE_LIMIT_S:
        return False

    _precontact_debug_last_log = now

    box_str = "IN" if in_box else "OUT"
    extra = f" {msg}" if msg else ""
    log(
        f"[PRECONTACT_DBG] behind={behind:+.1f} cross={cross:+.1f} "
        f"rng={range_m:.1f} Vc={closure:+.2f} box={box_str}{extra}"
    )
    return True


def log_creep_transition(
    from_state: str, to_state: str, boom_x: float, boom_y: float, boom_z: float, reason: str = ""
) -> None:
    """
    Phase 7D.78: Log CREEP state transitions with geometry context.

    Always logs (not rate-limited) for transition tracing.

    Args:
        from_state: Previous state name
        to_state: New state name
        boom_x: Longitudinal position at transition
        boom_y: Lateral offset at transition
        boom_z: Vertical offset at transition
        reason: Reason for transition
    """
    reason_str = f" ({reason})" if reason else ""
    log(
        f"[CREEP_TRANS] {from_state} -> {to_state} "
        f"boom=({boom_x:+.2f},{boom_y:+.2f},{boom_z:+.2f}){reason_str}"
    )


def log_precontact_transition(
    from_state: str, to_state: str, behind: float, cross: float, in_box: bool, reason: str = ""
) -> None:
    """
    Phase 7D.78: Log precontact state transitions with geometry context.

    Always logs (not rate-limited) for transition tracing.

    Args:
        from_state: Previous state name
        to_state: New state name
        behind: Distance behind leader at transition
        cross: Cross-track offset at transition
        in_box: Whether in precontact box at transition
        reason: Reason for transition
    """
    box_str = "IN_BOX" if in_box else "OUT_BOX"
    reason_str = f" ({reason})" if reason else ""
    log(
        f"[PRECONTACT_TRANS] {from_state} -> {to_state} "
        f"behind={behind:+.1f} cross={cross:+.1f} {box_str}{reason_str}"
    )


# ============================================================================
# Module Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Formation Logging Module - Self Test")
    print("=" * 60)

    log("Basic log message")
    log_event("TEST", "Event with tag")
    log_warn("WARN_TEST", "Warning message")
    log_error("ERR_TEST", "Error message")
    log_success("OK_TEST", "Success message")

    print("\n--- Rate limiting test ---")
    for i in range(5):
        logged = log_rate_limited("RATE_TEST", f"Attempt {i+1}")
        print(f"  Attempt {i+1}: {'logged' if logged else 'suppressed'}")
        time.sleep(0.2)

    print("\n--- State transition test ---")
    log_state_transition("INTERCEPT", "REJOIN_PRECONTACT", "range < 300m")

    print("\n--- Header test ---")
    log_header("PREFLIGHT CHECK")

    print("\n" + "=" * 60)
    print("Formation logging self-tests completed")
    print("=" * 60)
