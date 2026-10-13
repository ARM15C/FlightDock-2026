#!/usr/bin/env python3
"""
FlightDock Log Parsing Utilities
================================
Shared parsing functions and regex patterns for controller and SITL logs.
Uses only Python stdlib (no external dependencies).

Date: 2026-01-02
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Time extraction patterns
# TEST 387: Support new sim-time format [S=123.4s W=67.8s] and legacy [123.4s]
RE_ELAPSED_TIME = re.compile(r"\[S=\s*(\d+\.?\d*)s")  # Match [S=123.4s W=...]
RE_ELAPSED_TIME_LEGACY = re.compile(r"\[(\d+\.?\d*)s\]")  # Fallback for old logs
RE_UNIX_TIMESTAMP = re.compile(r"\[(\d{10}\.\d+)\]")

# State machine patterns
RE_STATE_TRANSITION = re.compile(r"STATE:\s*(\S+)\s*(?:→|->)\s*(\S+)(?:\s*\(([^)]*)\))?")

# Version/identity patterns
RE_VERSION = re.compile(r"Version:\s*(\S+)")
RE_SENSOR_MODE = re.compile(r"Close-Range Sensor Mode:\s*(\S+)")
RE_PRECONTACT_ENVELOPE = re.compile(r"Precontact Envelope:\s*(.+)")
RE_CONTACT_ENVELOPE = re.compile(r"Contact Envelope:\s*(.+)")

# Counter patterns
RE_PRECONTACT_HOLD = re.compile(r"\[COUNTER\]\s*Precontact hold #(\d+) completed \((\d+\.?\d*)s\)")
RE_CONTACT_ENTRY = re.compile(r"\[COUNTER\]\s*Contact envelope entry #(\d+)")
RE_CONTACT_DWELL = re.compile(r"\[COUNTER\]\s*Contact dwell time:\s*(\d+\.?\d*)s")

# Distance/geometry patterns
RE_RANGE = re.compile(r"range=(\d+\.?\d*)m")
RE_BEHIND_ERR = re.compile(r"behind_err=([+-]?\d+\.?\d*)m")
RE_CROSS_ERR = re.compile(r"cross_err=([+-]?\d+\.?\d*)m")
RE_BELOW_ERR = re.compile(r"below_err=([+-]?\d+\.?\d*)m")
RE_CLOSURE_VC = re.compile(r"Vc=([+-]?\d+\.?\d*)m/s")

# Safety/control event patterns
RE_OVERTAKE_PROTECT = re.compile(
    r"\[OVERTAKE_PROTECT\]\s*(TIER\d)?\s*behind_err=([+-]?\d+\.?\d*)m"
)
RE_NO_PASS_WARN = re.compile(r"\[NO_PASS_WARN\]")
RE_NO_PASS_ABORT = re.compile(r"\[NO_PASS_ABORT\]")
RE_CLOSURE_FLOOR = re.compile(r"\[CLOSURE_FLOOR\]")
RE_CLOSURE_GOV = re.compile(r"\[CLOSURE_GOV(?:ERNOR)?\]")
RE_CONTACT_BRAKE = re.compile(r"\[CONTACT_BRAKE\]")
RE_ALT_GATE = re.compile(r"\[ALT_GATE\]")
RE_ALT_WARNING = re.compile(r"\[ALT_WARNING\]")
RE_ALT_FALLBACK = re.compile(r"\[ALT_FALLBACK\]")
RE_PRECONTACT_VC_GATE = re.compile(r"\[PRECONTACT_VC_GATE\]")
RE_CONTACT_VC_GATE = re.compile(r"\[CONTACT_VC_GATE\]")
RE_TURN_GATE = re.compile(r"\[TURN_GATE\]|LEADER_TURN")
RE_OFFSET_SIDE = re.compile(r"\[OFFSET_SIDE\]")
RE_PRECISION_MODE = re.compile(r"\[PRECISION_MODE\]")
RE_PARAM_RESTORE_FAIL = re.compile(r"PARAM_RESTORE_FAIL")

# Generic warning patterns
RE_WARNING_SYMBOL = re.compile(r"⚠")
RE_ERROR = re.compile(r"ERROR", re.IGNORECASE)
RE_FAILSAFE = re.compile(r"FAILSAFE", re.IGNORECASE)

# SITL console patterns
RE_EKF = re.compile(r"EKF", re.IGNORECASE)
RE_ARMING = re.compile(r"arming|Arm|Disarm", re.IGNORECASE)
RE_TECS = re.compile(r"TECS")
RE_GPS = re.compile(r"GPS")
RE_MODE = re.compile(r"mode", re.IGNORECASE)
RE_PANIC = re.compile(r"panic|watchdog|ASSERT", re.IGNORECASE)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class StateTransition:
    """Single state machine transition."""

    line_num: int
    elapsed_s: float
    from_state: str
    to_state: str
    reason: str = ""
    duration_s: float = 0.0


@dataclass
class AlertEvent:
    """Single warning/error/alert event."""

    line_num: int
    elapsed_s: float
    tag: str
    severity: str  # INFO, WARNING, CRITICAL
    text: str
    expected: bool = True


@dataclass
class ControllerLogMetrics:
    """Aggregated metrics from controller log parsing."""

    # Identity
    version: str = ""
    sensor_mode: str = ""
    precontact_envelope: str = ""
    contact_envelope: str = ""

    # Duration
    duration_s: float = 0.0
    duration_min: float = 0.0

    # State machine
    state_transitions: List[StateTransition] = field(default_factory=list)
    phase_counts: Dict[str, int] = field(default_factory=dict)
    phase_times: Dict[str, float] = field(default_factory=dict)

    # Precontact metrics
    precontact_holds_completed: int = 0
    precontact_hold_durations: List[float] = field(default_factory=list)
    first_precontact_time_s: Optional[float] = None
    precontact_required_s: float = 5.0

    # Contact metrics
    contact_entries: int = 0
    contact_dwell_times: List[float] = field(default_factory=list)
    best_contact_dwell_s: float = 0.0
    total_contact_time_s: float = 0.0
    first_contact_time_s: Optional[float] = None
    contact_required_s: float = 7.0
    reached_hold_contact: bool = False
    reached_docking_creep: bool = False

    # Distance metrics
    min_range_m: float = float("inf")
    min_range_time_s: float = 0.0
    min_behind_err_m: float = float("inf")  # Most negative = worst overshoot
    min_behind_time_s: float = 0.0
    max_closure_near_contact: float = 0.0

    # Safety events
    alert_events: List[AlertEvent] = field(default_factory=list)
    alert_counts: Dict[str, int] = field(default_factory=dict)

    # Specific event counts
    overtake_protect_count: int = 0
    overtake_tier_counts: Dict[str, int] = field(default_factory=dict)
    no_pass_warn_count: int = 0
    no_pass_abort_count: int = 0
    closure_floor_count: int = 0
    closure_gov_count: int = 0
    contact_brake_count: int = 0
    alt_gate_count: int = 0
    alt_fallback_count: int = 0
    precontact_vc_gate_count: int = 0
    contact_vc_gate_count: int = 0
    turn_gate_count: int = 0

    # ALTITUDE_CAPTURE metrics
    altitude_capture_attempts: int = 0
    altitude_capture_success: int = 0
    altitude_capture_time_s: float = 0.0

    # Verdict
    verdict: str = "UNKNOWN"
    verdict_reason: str = ""


@dataclass
class SITLLogMetrics:
    """Aggregated metrics from SITL console log."""

    ekf_events: int = 0
    arming_events: int = 0
    tecs_events: int = 0
    gps_events: int = 0
    failsafe_events: int = 0
    error_events: int = 0
    panic_events: int = 0
    sample_lines: List[str] = field(default_factory=list)
    healthy: bool = True


# =============================================================================
# SEVERITY MAPPING
# =============================================================================

ALERT_SEVERITY_MAP = {
    "OFFSET_SIDE": ("INFO", True, "Normal geometry adjustment"),
    "NO_PASS_WARN": ("WARNING", True, "Overshoot warning, expected during recovery"),
    "NO_PASS_ABORT": ("CRITICAL", False, "Abort triggered - test failure"),
    "OVERTAKE_PROTECT": ("WARNING", True, "Speed reduction triggered"),
    "CLOSURE_FLOOR": ("INFO", True, "Min closure applied"),
    "CLOSURE_GOV": ("INFO", True, "Closure governor active"),
    "CONTACT_BRAKE": ("INFO", True, "Contact brake active"),
    "ALT_GATE": ("WARNING", True, "Altitude not captured"),
    "ALT_WARNING": ("WARNING", True, "Altitude warning"),
    "ALT_FALLBACK": ("INFO", True, "Altitude fallback commanded"),
    "PRECONTACT_VC_GATE": ("INFO", True, "Waiting for closure to settle"),
    "CONTACT_VC_GATE": ("INFO", True, "Waiting for closure to settle"),
    "TURN_GATE": ("INFO", True, "Leader turn detected"),
    "PRECISION_MODE": ("INFO", True, "Precision mode entered"),
    "PARAM_RESTORE_FAIL": ("CRITICAL", False, "Parameter restore failed"),
    "SPEED_DIAG": ("WARNING", True, "Speed tracking issue"),
    "ERROR": ("CRITICAL", False, "Generic error"),
    "FAILSAFE": ("CRITICAL", False, "Failsafe triggered"),
}


def get_alert_severity(tag: str) -> Tuple[str, bool, str]:
    """Get severity, expectedness, and note for an alert tag."""
    return ALERT_SEVERITY_MAP.get(tag, ("WARNING", True, "Unknown event type"))


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================


def extract_elapsed_time(line: str) -> Optional[float]:
    """Extract sim-time from line like '[S=123.4s W=67.8s]' or legacy '[123.4s]'.

    TEST 387: Updated to support new dual-timestamp format.
    """
    # Try new format first: [S=123.4s W=67.8s]
    m = RE_ELAPSED_TIME.search(line)
    if m:
        return float(m.group(1))
    # Fallback to legacy format: [123.4s]
    m = RE_ELAPSED_TIME_LEGACY.search(line)
    return float(m.group(1)) if m else None


def extract_test_id(filename: str) -> Optional[int]:
    """Extract test ID from filename like 'intercept_rejoin-327.log'."""
    m = re.search(r"intercept_rejoin-(\d+)", filename)
    return int(m.group(1)) if m else None


def parse_controller_log(log_path: str, max_alert_lines: int = 20) -> ControllerLogMetrics:
    """
    Parse controller log file and extract all metrics.

    Args:
        log_path: Path to controller log file
        max_alert_lines: Maximum number of alert lines to store

    Returns:
        ControllerLogMetrics with all extracted data
    """
    metrics = ControllerLogMetrics()
    path = Path(log_path)

    if not path.exists():
        metrics.verdict = "FILE_NOT_FOUND"
        return metrics

    last_elapsed = 0.0
    transitions_raw = []
    near_contact_closures = []

    with open(path, "r", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            # Track elapsed time
            elapsed = extract_elapsed_time(line)
            if elapsed is not None:
                last_elapsed = elapsed

            # === IDENTITY ===
            if "Version:" in line:
                m = RE_VERSION.search(line)
                if m:
                    metrics.version = m.group(1)

            if "Close-Range Sensor Mode:" in line:
                m = RE_SENSOR_MODE.search(line)
                if m:
                    metrics.sensor_mode = m.group(1)

            if "Precontact Envelope:" in line:
                m = RE_PRECONTACT_ENVELOPE.search(line)
                if m:
                    metrics.precontact_envelope = m.group(1)

            if "Contact Envelope:" in line:
                m = RE_CONTACT_ENVELOPE.search(line)
                if m:
                    metrics.contact_envelope = m.group(1)

            # === STATE TRANSITIONS ===
            if "STATE:" in line:
                m = RE_STATE_TRANSITION.search(line)
                if m:
                    reason = m.group(3) or ""
                    transitions_raw.append((ln, last_elapsed, m.group(1), m.group(2), reason))

            # === COUNTER EVENTS ===
            if "[COUNTER]" in line:
                # Precontact hold
                m = RE_PRECONTACT_HOLD.search(line)
                if m:
                    metrics.precontact_holds_completed += 1
                    metrics.precontact_hold_durations.append(float(m.group(2)))
                    if metrics.first_precontact_time_s is None:
                        metrics.first_precontact_time_s = last_elapsed

                # Contact entry
                m = RE_CONTACT_ENTRY.search(line)
                if m:
                    metrics.contact_entries += 1
                    if metrics.first_contact_time_s is None:
                        metrics.first_contact_time_s = last_elapsed

                # Contact dwell
                m = RE_CONTACT_DWELL.search(line)
                if m:
                    dwell = float(m.group(1))
                    metrics.contact_dwell_times.append(dwell)
                    metrics.total_contact_time_s += dwell
                    if dwell > metrics.best_contact_dwell_s:
                        metrics.best_contact_dwell_s = dwell

            # === DISTANCE METRICS ===
            m = RE_RANGE.search(line)
            if m:
                r = float(m.group(1))
                if r < metrics.min_range_m:
                    metrics.min_range_m = r
                    metrics.min_range_time_s = last_elapsed

            m = RE_BEHIND_ERR.search(line)
            if m:
                b = float(m.group(1))
                if b < metrics.min_behind_err_m:
                    metrics.min_behind_err_m = b
                    metrics.min_behind_time_s = last_elapsed

            # Track closure near contact
            m_behind = RE_BEHIND_ERR.search(line)
            m_vc = RE_CLOSURE_VC.search(line)
            if m_behind and m_vc:
                behind = float(m_behind.group(1))
                vc = float(m_vc.group(1))
                if abs(behind) < 3.0:
                    near_contact_closures.append(vc)
                    if vc > metrics.max_closure_near_contact:
                        metrics.max_closure_near_contact = vc

            # === SAFETY EVENTS ===
            # OVERTAKE_PROTECT
            if "[OVERTAKE_PROTECT]" in line:
                metrics.overtake_protect_count += 1
                if "TIER3" in line:
                    metrics.overtake_tier_counts["TIER3"] = (
                        metrics.overtake_tier_counts.get("TIER3", 0) + 1
                    )
                elif "TIER2" in line:
                    metrics.overtake_tier_counts["TIER2"] = (
                        metrics.overtake_tier_counts.get("TIER2", 0) + 1
                    )
                elif "TIER1" in line:
                    metrics.overtake_tier_counts["TIER1"] = (
                        metrics.overtake_tier_counts.get("TIER1", 0) + 1
                    )

            if RE_NO_PASS_WARN.search(line):
                metrics.no_pass_warn_count += 1
            if RE_NO_PASS_ABORT.search(line):
                metrics.no_pass_abort_count += 1
            if RE_CLOSURE_FLOOR.search(line):
                metrics.closure_floor_count += 1
            if RE_CLOSURE_GOV.search(line):
                metrics.closure_gov_count += 1
            if RE_CONTACT_BRAKE.search(line):
                metrics.contact_brake_count += 1
            if RE_ALT_GATE.search(line):
                metrics.alt_gate_count += 1
            if RE_ALT_FALLBACK.search(line):
                metrics.alt_fallback_count += 1
            if RE_PRECONTACT_VC_GATE.search(line):
                metrics.precontact_vc_gate_count += 1
            if RE_CONTACT_VC_GATE.search(line):
                metrics.contact_vc_gate_count += 1
            if RE_TURN_GATE.search(line):
                metrics.turn_gate_count += 1

            # === ALERT EVENTS (warnings/errors) ===
            alert_tag = None
            if "⚠" in line or "WARN" in line or "ABORT" in line:
                # Determine tag
                if "[NO_PASS_ABORT]" in line:
                    alert_tag = "NO_PASS_ABORT"
                elif "[NO_PASS_WARN]" in line:
                    alert_tag = "NO_PASS_WARN"
                elif "[OFFSET_SIDE]" in line:
                    alert_tag = "OFFSET_SIDE"
                elif "[SPEED_DIAG]" in line:
                    alert_tag = "SPEED_DIAG"
                elif "[ALT_GATE]" in line:
                    alert_tag = "ALT_GATE"
                elif "ERROR" in line.upper():
                    alert_tag = "ERROR"
                else:
                    alert_tag = "WARNING"

                if alert_tag and len(metrics.alert_events) < max_alert_lines:
                    severity, expected, _ = get_alert_severity(alert_tag)
                    metrics.alert_events.append(
                        AlertEvent(
                            line_num=ln,
                            elapsed_s=last_elapsed,
                            tag=alert_tag,
                            severity=severity,
                            text=line.strip()[:150],
                            expected=expected,
                        )
                    )
                metrics.alert_counts[alert_tag] = metrics.alert_counts.get(alert_tag, 0) + 1

    # === POST-PROCESSING ===
    metrics.duration_s = last_elapsed
    metrics.duration_min = last_elapsed / 60.0

    # Process state transitions
    for i, (ln, t, old, new, reason) in enumerate(transitions_raw):
        t2 = transitions_raw[i + 1][1] if i + 1 < len(transitions_raw) else last_elapsed
        dur = t2 - t
        metrics.state_transitions.append(
            StateTransition(
                line_num=ln,
                elapsed_s=t,
                from_state=old,
                to_state=new,
                reason=reason[:50],
                duration_s=dur,
            )
        )
        metrics.phase_counts[new] = metrics.phase_counts.get(new, 0) + 1
        metrics.phase_times[new] = metrics.phase_times.get(new, 0.0) + dur

    # Check specific phases reached
    phases_reached = set(metrics.phase_counts.keys())
    metrics.reached_hold_contact = "HOLD_CONTACT" in phases_reached
    metrics.reached_docking_creep = "DOCKING_CREEP" in phases_reached

    if "ALTITUDE_CAPTURE" in phases_reached:
        metrics.altitude_capture_attempts = metrics.phase_counts.get("ALTITUDE_CAPTURE", 0)
        metrics.altitude_capture_time_s = metrics.phase_times.get("ALTITUDE_CAPTURE", 0.0)
        # Count successes (transitions from ALT_CAPTURE to DOCKING_CREEP)
        for tr in metrics.state_transitions:
            if tr.from_state == "ALTITUDE_CAPTURE" and tr.to_state == "DOCKING_CREEP":
                metrics.altitude_capture_success += 1

    # === DETERMINE VERDICT ===
    if metrics.no_pass_abort_count > 0 and not metrics.reached_hold_contact:
        metrics.verdict = "FAIL_OVERTAKE"
        worst = metrics.min_behind_err_m
        metrics.verdict_reason = f"NO_PASS_ABORT triggered (worst behind: {worst:+.1f}m)"
    elif (
        metrics.contact_entries > 0 and metrics.best_contact_dwell_s >= metrics.contact_required_s
    ):
        metrics.verdict = "SUCCESS"
        best = metrics.best_contact_dwell_s
        req = metrics.contact_required_s
        metrics.verdict_reason = f"Contact dwell {best:.1f}s >= required {req}s"
    elif metrics.contact_entries > 0:
        metrics.verdict = "PARTIAL_CONTACT"
        best = metrics.best_contact_dwell_s
        req = metrics.contact_required_s
        metrics.verdict_reason = f"Contact entered but best dwell {best:.1f}s < req {req}s"
    elif metrics.precontact_holds_completed > 0:
        metrics.verdict = "FAIL_NO_CONTACT"
        metrics.verdict_reason = "Precontact achieved but no contact entries"
    elif metrics.min_range_m < 30.0:
        metrics.verdict = "FAIL_ALTITUDE"
        metrics.verdict_reason = "Got close but likely altitude issue"
    else:
        metrics.verdict = "FAIL_GEOMETRY"
        metrics.verdict_reason = "Did not reach precontact"

    return metrics


def parse_sitl_log(log_path: str, max_samples: int = 10) -> SITLLogMetrics:
    """
    Parse SITL console log for health indicators.

    Args:
        log_path: Path to SITL console log file
        max_samples: Maximum sample lines to store per category

    Returns:
        SITLLogMetrics with health summary
    """
    metrics = SITLLogMetrics()
    path = Path(log_path)

    if not path.exists():
        return metrics

    with open(path, "r", errors="replace") as f:
        for line in f:
            if RE_EKF.search(line):
                metrics.ekf_events += 1
                if len(metrics.sample_lines) < max_samples:
                    metrics.sample_lines.append(f"[EKF] {line.strip()[:100]}")

            if RE_ARMING.search(line):
                metrics.arming_events += 1

            if RE_TECS.search(line):
                metrics.tecs_events += 1
                if len(metrics.sample_lines) < max_samples:
                    metrics.sample_lines.append(f"[TECS] {line.strip()[:100]}")

            if RE_GPS.search(line):
                metrics.gps_events += 1

            if RE_FAILSAFE.search(line):
                metrics.failsafe_events += 1
                if len(metrics.sample_lines) < max_samples:
                    metrics.sample_lines.append(f"[FAILSAFE] {line.strip()[:100]}")

            if RE_ERROR.search(line):
                metrics.error_events += 1
                if len(metrics.sample_lines) < max_samples:
                    metrics.sample_lines.append(f"[ERROR] {line.strip()[:100]}")

            if RE_PANIC.search(line):
                metrics.panic_events += 1
                if len(metrics.sample_lines) < max_samples:
                    metrics.sample_lines.append(f"[PANIC] {line.strip()[:100]}")

    # Determine health
    metrics.healthy = (
        metrics.failsafe_events == 0 and metrics.panic_events == 0 and metrics.error_events < 5
    )

    return metrics


def metrics_to_dict(metrics: ControllerLogMetrics) -> Dict[str, Any]:
    """Convert ControllerLogMetrics to JSON-serializable dict."""
    return {
        "version": metrics.version,
        "sensor_mode": metrics.sensor_mode,
        "duration_s": metrics.duration_s,
        "duration_min": round(metrics.duration_min, 2),
        "verdict": metrics.verdict,
        "verdict_reason": metrics.verdict_reason,
        "precontact_holds_completed": metrics.precontact_holds_completed,
        "precontact_required_s": metrics.precontact_required_s,
        "first_precontact_time_s": metrics.first_precontact_time_s,
        "contact_entries": metrics.contact_entries,
        "contact_required_s": metrics.contact_required_s,
        "best_contact_dwell_s": metrics.best_contact_dwell_s,
        "total_contact_time_s": round(metrics.total_contact_time_s, 2),
        "first_contact_time_s": metrics.first_contact_time_s,
        "reached_hold_contact": metrics.reached_hold_contact,
        "reached_docking_creep": metrics.reached_docking_creep,
        "min_range_m": (
            round(metrics.min_range_m, 2) if metrics.min_range_m != float("inf") else None
        ),
        "min_range_time_s": round(metrics.min_range_time_s, 2),
        "min_behind_err_m": (
            round(metrics.min_behind_err_m, 2)
            if metrics.min_behind_err_m != float("inf")
            else None
        ),
        "min_behind_time_s": round(metrics.min_behind_time_s, 2),
        "max_closure_near_contact": round(metrics.max_closure_near_contact, 2),
        "no_pass_abort_count": metrics.no_pass_abort_count,
        "no_pass_warn_count": metrics.no_pass_warn_count,
        "overtake_protect_count": metrics.overtake_protect_count,
        "overtake_tier_counts": metrics.overtake_tier_counts,
        "alt_gate_count": metrics.alt_gate_count,
        "closure_floor_count": metrics.closure_floor_count,
        "altitude_capture_attempts": metrics.altitude_capture_attempts,
        "altitude_capture_success": metrics.altitude_capture_success,
        "altitude_capture_time_s": round(metrics.altitude_capture_time_s, 2),
        "phase_counts": metrics.phase_counts,
        "phase_times": {k: round(v, 2) for k, v in metrics.phase_times.items()},
        "alert_counts": metrics.alert_counts,
    }


def metrics_to_csv_row(test_id: int, metrics: ControllerLogMetrics) -> Dict[str, Any]:
    """Convert metrics to a flat dict suitable for CSV row."""
    return {
        "test_id": test_id,
        "version": metrics.version,
        "duration_s": round(metrics.duration_s, 1),
        "duration_min": round(metrics.duration_min, 1),
        "verdict": metrics.verdict,
        "precontact_holds": metrics.precontact_holds_completed,
        "contact_entries": metrics.contact_entries,
        "best_contact_dwell_s": round(metrics.best_contact_dwell_s, 1),
        "total_contact_time_s": round(metrics.total_contact_time_s, 1),
        "min_range_m": (
            round(metrics.min_range_m, 1) if metrics.min_range_m != float("inf") else ""
        ),
        "min_range_time_s": round(metrics.min_range_time_s, 1),
        "min_behind_err_m": (
            round(metrics.min_behind_err_m, 1) if metrics.min_behind_err_m != float("inf") else ""
        ),
        "no_pass_abort_count": metrics.no_pass_abort_count,
        "alt_gate_count": metrics.alt_gate_count,
        "reached_docking_creep": 1 if metrics.reached_docking_creep else 0,
        "reached_hold_contact": 1 if metrics.reached_hold_contact else 0,
    }


def _run_format_selftest():
    """TEST 387: Verify extract_elapsed_time() handles both formats."""
    test_cases = [
        # (input_line, expected_result)
        ("[S=123.4s W=67.8s] Some message", 123.4),  # New format
        ("[S= 0.0s W= 0.0s] Start", 0.0),  # New format with spaces
        ("[123.4s] Legacy message", 123.4),  # Legacy format
        ("[0.0s] Legacy start", 0.0),  # Legacy format zero
        ("No timestamp here", None),  # No match
        ("[1768682612.260] Unix only", None),  # Unix timestamp only
    ]

    failed = []
    for line, expected in test_cases:
        result = extract_elapsed_time(line)
        if result != expected:
            failed.append((line, expected, result))

    if failed:
        print("SELFTEST FAILED:")
        for line, expected, got in failed:
            print(f"  Input: {line!r}")
            print(f"  Expected: {expected}, Got: {got}")
        raise AssertionError(f"log_parsing selftest: {len(failed)} failures")

    print("log_parsing.py selftest: PASS (6 cases)")


if __name__ == "__main__":
    # TEST 387: Verify dual-format parsing before any other tests
    _run_format_selftest()

    # Quick test
    import sys

    if len(sys.argv) > 1:
        m = parse_controller_log(sys.argv[1])
        print(f"Version: {m.version}")
        print(f"Duration: {m.duration_min:.1f} min")
        print(f"Verdict: {m.verdict}")
        print(f"Precontact holds: {m.precontact_holds_completed}")
        print(f"Contact entries: {m.contact_entries}")
        print(f"Min range: {m.min_range_m:.1f}m")
