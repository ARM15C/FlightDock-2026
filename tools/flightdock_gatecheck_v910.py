#!/usr/bin/env python3
"""
FlightDock v9.10 Gate Checker

Validates that Gates 0-2 pass for a given run directory.
Gates:
  - Gate 0: Timebase Integrity (dt_tick_raw_s column exists, no regressions, valid rates)
  - Gate 1: Safety as Invariant (no no-pass violations, safety emergency logged properly)
  - Gate 2: State Machine Stability (HOLD debouncing, transition log exists with reasons)

Exit codes:
  0 = PASS (all gates passed)
  1 = ERROR (could not run check)
  2 = FAIL (one or more gates failed)

Usage:
    python tools/flightdock_gatecheck_v910.py --run-dir <RUN_DIR> --speedup 1x
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field

from common_io import ensure_output_dir
from typing import Dict, List, Optional, Tuple


@dataclass
class GateResult:
    """Result of a single gate check."""

    name: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def load_csv(path: str) -> List[Dict[str, str]]:
    """Load CSV file as list of dicts."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_json(path: str) -> Optional[Dict]:
    """Load JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def check_gate0_timebase(run_dir: str, speedup: str) -> GateResult:
    """
    Gate 0: Timebase Integrity

    FAILS if:
    - dt_tick_raw_s column is missing (HARD REQUIRE)
    - frac_small_dt > 0.05 (more than 5% of rows have dt <= 0.002s)
    - sim_time_regressions > 0
    - max_sim_dt_raw_s > 0.5
    - run_invalid_timebase == True
    - For 1x/3x: dt_raw_median not in [0.03, 0.08]
    - Skip rate extreme AND lag symptoms
    """
    result = GateResult(name="Gate0_Timebase", passed=True)

    # Load arbitration CSV
    arb_csv_path = os.path.join(run_dir, "speed_arbitration.csv")
    csv_data = load_csv(arb_csv_path)

    # R8: Don't early return on missing CSV - mark failed but let fatal_conditions check first
    # For short canary runs, CSV may exist but be empty (header only) if run didn't reach
    # formation flight states. Check file existence vs empty data distinction.
    arb_csv_exists = os.path.exists(arb_csv_path)
    if not csv_data:
        if not arb_csv_exists:
            result.passed = False
            result.failures.append(f"speed_arbitration.csv not found at {arb_csv_path}")
        else:
            # CSV exists but has no data rows - check run_summary for context
            # Note: ticks_executed_total counts ALL ticks from tick(), but CSV rows are only
            # written during formation flight states. Empty CSV is expected if run didn't
            # reach formation states (REJOIN_PRECONTACT, HOLD_PRECONTACT, etc.)
            run_summary = load_json(os.path.join(run_dir, "run_summary.json"))
            state_counts = run_summary.get("state_entry_counts", {}) if run_summary else {}

            # Formation states that write to CSV
            formation_states = {
                "REJOIN_PRECONTACT",
                "HOLD_PRECONTACT",
                "ALT_CAP_HOLD_RANGE",
                "ALTITUDE_CAPTURE",
                "DOCKING_CREEP",
                "HOLD_CONTACT",
            }
            entered_formation = any(state in state_counts for state in formation_states)

            if not entered_formation:
                # Expected for runs that didn't reach formation flight
                result.warnings.append(
                    "speed_arbitration.csv empty (run didn't reach formation states)"
                )
            else:
                # Reached formation states but no CSV rows - this is a bug
                result.passed = False
                result.failures.append(
                    "speed_arbitration.csv empty but formation states were entered (CSV write bug)"
                )
        # R8: Don't return early - main() checks fatal_conditions first and will short-circuit
        return result  # Still return, but main() ordering handles priority

    # HARD REQUIRE: dt_tick_raw_s column must exist
    if "dt_tick_raw_s" not in csv_data[0]:
        result.passed = False
        result.failures.append(
            "HARD FAIL: dt_tick_raw_s column missing - Gate 0 requires new timebase columns"
        )
        return result

    # Parse dt values
    dt_raw_values = []
    for row in csv_data:
        try:
            dt = float(row.get("dt_tick_raw_s", "0"))
            if dt > 0:
                dt_raw_values.append(dt)
        except (ValueError, TypeError):
            pass

    if not dt_raw_values:
        result.passed = False
        result.failures.append("No valid dt_tick_raw_s values found")
        return result

    # Compute metrics
    dt_raw_values.sort()
    n = len(dt_raw_values)
    dt_min = dt_raw_values[0]
    dt_max = dt_raw_values[-1]
    dt_median = dt_raw_values[n // 2]
    dt_p95 = dt_raw_values[int(n * 0.95)] if n > 20 else dt_max

    # Count small dt (likely fake/stale)
    small_dt_count = sum(1 for dt in dt_raw_values if dt <= 0.002)
    frac_small_dt = small_dt_count / len(dt_raw_values)

    result.metrics = {
        "dt_raw_min": dt_min,
        "dt_raw_median": dt_median,
        "dt_raw_p95": dt_p95,
        "dt_raw_max": dt_max,
        "frac_small_dt": frac_small_dt,
        "n_rows": n,
    }

    # Check from run_summary if available
    run_summary_path = os.path.join(run_dir, "run_summary.json")
    run_summary = load_json(run_summary_path)

    if run_summary:
        # v9.10 P1 Fix B: Check time_authority_update_hz_sim at 1x
        # This is a HARD gate - if telemetry rate is below 18Hz at 1x, the timebase
        # cannot provide sufficient resolution for formation control.
        time_auth_hz = run_summary.get("time_authority_update_hz_sim")
        if time_auth_hz is not None:
            result.metrics["time_authority_update_hz_sim"] = time_auth_hz
            result.metrics["time_authority_updates_total"] = run_summary.get(
                "time_authority_updates_total", 0
            )
            dt_median = run_summary.get("time_authority_dt_median")
            if dt_median is not None:
                result.metrics["time_authority_dt_median"] = dt_median

            # Gate0 early fail: time_authority_update_hz_sim < 18 at 1x
            if speedup == "1x" and time_auth_hz < 18.0:
                result.passed = False
                result.failures.append(
                    f"time_authority_update_hz_sim = {time_auth_hz:.1f} Hz < 18 Hz @ 1x (ATTITUDE rate too low)"
                )

        # Check run_invalid_timebase
        if run_summary.get("run_invalid_timebase", False):
            result.passed = False
            result.failures.append("run_invalid_timebase == True")

        # Check sim_time_regressions
        regressions = run_summary.get("sim_time_regressions_total", 0)
        if regressions > 0:
            result.passed = False
            result.failures.append(f"sim_time_regressions = {regressions} (must be 0)")
        result.metrics["sim_time_regressions"] = regressions

        # Check max_sim_dt_raw_s
        max_dt = run_summary.get("max_sim_dt_raw_s", 0)
        if max_dt > 0.5:
            result.passed = False
            result.failures.append(f"max_sim_dt_raw_s = {max_dt:.3f}s > 0.5s threshold")
        result.metrics["max_sim_dt_raw_s"] = max_dt

        # Check ticks_executed_total matches CSV row count
        # v9.10 FIX: Read from LAST CSV row (cumulative), not run_summary
        # This prevents counter drift if run_summary is stale
        try:
            ticks_executed = int(csv_data[-1].get("ticks_executed_total", 0))
        except (ValueError, KeyError, IndexError):
            ticks_executed = run_summary.get("ticks_executed_total", 0)

        # Rec 3: Account for CSV decimation stride when validating row count
        csv_stride = run_summary.get("csv_decimation_stride", 1) if run_summary else 1
        expected_rows = ticks_executed / max(1, csv_stride)
        result.metrics["csv_decimation_stride"] = csv_stride
        if ticks_executed > 0 and abs(expected_rows - n) > max(2, csv_stride):
            result.passed = False
            result.failures.append(
                f"ticks_executed_total ({ticks_executed}) / stride ({csv_stride}) "
                f"≈ {expected_rows:.0f} != CSV rows ({n}) - counter drift"
            )
        result.metrics["ticks_executed_total"] = ticks_executed

        # Skip rate check
        skipped_no_advance = run_summary.get("ticks_skipped_no_advance_total", 0)
        skipped_accumulating = run_summary.get("ticks_skipped_accumulating_total", 0)
        late_tick_count = run_summary.get("late_tick_count", 0)

        if ticks_executed > 0:
            skip_ratio = (skipped_no_advance + skipped_accumulating) / max(1, ticks_executed)
            result.metrics["skip_ratio"] = skip_ratio
            result.metrics["skipped_no_advance"] = skipped_no_advance
            result.metrics["skipped_accumulating"] = skipped_accumulating
            result.metrics["late_tick_count"] = late_tick_count

            # WARN if skip_ratio > 1.0, FAIL only if extreme AND lag symptoms
            if skip_ratio > 1.0:
                result.warnings.append(f"skip_ratio = {skip_ratio:.2f} > 1.0 (may be wasting CPU)")
            if skip_ratio > 10.0 and late_tick_count > 0:
                result.passed = False
                result.failures.append(
                    f"Extreme skip_ratio ({skip_ratio:.1f}) with late_tick_count ({late_tick_count}) - falling behind"
                )

        # v9.10 Gate0 Hardening: Tick throughput validation
        # Check ticks_executed_total ≈ duration_s × 20 (±5% tolerance)
        # This catches pathological "improvements" where skip_ratio looks good
        # but tick throughput collapsed.
        sim_elapsed = run_summary.get("sim_elapsed_s")
        if sim_elapsed is not None and sim_elapsed > 10.0:  # Only check for runs > 10s
            expected_ticks = sim_elapsed * 20.0  # 20 Hz target
            tolerance_pct = 0.05  # ±5%
            min_expected = expected_ticks * (1.0 - tolerance_pct)
            max_expected = expected_ticks * (1.0 + tolerance_pct)
            result.metrics["expected_ticks_20hz"] = expected_ticks
            result.metrics["tick_throughput_tolerance_pct"] = tolerance_pct * 100

            if ticks_executed < min_expected:
                # Tick throughput too low - formation control may be starved
                deficit_pct = (expected_ticks - ticks_executed) / expected_ticks * 100
                result.warnings.append(
                    f"Low tick throughput: {ticks_executed} vs expected {expected_ticks:.0f} "
                    f"(-{deficit_pct:.1f}%)"
                )
                # Only FAIL for severe deficit (> 20%) at 1x/3x speedups
                if speedup in ("1x", "3x") and deficit_pct > 20.0:
                    result.passed = False
                    result.failures.append(
                        f"Tick throughput severely degraded: {ticks_executed} ticks "
                        f"vs expected {expected_ticks:.0f} ({deficit_pct:.1f}% deficit)"
                    )

    # Check frac_small_dt
    if frac_small_dt > 0.05:
        result.passed = False
        result.failures.append(f"frac_small_dt = {frac_small_dt:.3f} > 0.05 (dt spam detected)")

    # For 1x/3x: check dt_raw_median is ~20Hz (0.03-0.08s)
    if speedup in ("1x", "3x"):
        if not (0.03 <= dt_median <= 0.08):
            result.passed = False
            result.failures.append(
                f"dt_raw_median = {dt_median:.4f}s not in [0.03, 0.08] for {speedup}"
            )

    # Rec 1: Composite time_authority_quality_score (0.0 = worst, 1.0 = perfect)
    # Combines: regression penalty, frac_small_dt penalty, dt_median deviation, tick throughput
    score = 1.0
    # Penalty: sim-time regressions (hard fail → score = 0)
    regressions = result.metrics.get("sim_time_regressions", 0)
    if regressions > 0:
        score = 0.0
    # Penalty: frac_small_dt (linear 0→0.05 maps to 1.0→0.5)
    score -= min(frac_small_dt / 0.05, 1.0) * 0.5
    # Penalty: dt_median deviation from 0.05s ideal (±0.03s → 0 penalty, beyond → up to 0.2)
    dt_ideal = 0.05
    dt_dev = abs(dt_median - dt_ideal)
    score -= min(dt_dev / 0.03, 1.0) * 0.2
    # Penalty: tick throughput deficit (if available)
    expected_ticks_val = result.metrics.get("expected_ticks_20hz")
    ticks_exec_val = result.metrics.get("ticks_executed_total", 0)
    if expected_ticks_val is not None and expected_ticks_val > 0 and ticks_exec_val > 0:
        deficit_frac = max(0, (expected_ticks_val - ticks_exec_val) / expected_ticks_val)
        score -= min(deficit_frac / 0.20, 1.0) * 0.3
    score = max(0.0, min(1.0, score))
    result.metrics["time_authority_quality_score"] = round(score, 3)

    return result


def check_gate0_backlog_time(run_dir: str, speedup: str) -> GateResult:
    """
    Gate 0 Extension: Backlog Time (Rec 5)

    FAILS if runtime_backlog_time_max_s exceeds speedup-dependent threshold.
    WARNS about backpressure drops.
    """
    result = GateResult(name="Gate0_BacklogTime", passed=True)

    run_summary_path = os.path.join(run_dir, "run_summary.json")
    run_summary = load_json(run_summary_path)

    if not run_summary:
        # No run_summary - skip this check (other gates handle missing summary)
        return result

    backlog_max_s = run_summary.get("runtime_backlog_time_max_s", 0.0)
    drops_total = run_summary.get("ticks_dropped_backpressure_total", 0)
    result.metrics["runtime_backlog_time_max_s"] = backlog_max_s
    result.metrics["runtime_backlog_time_max_depth"] = run_summary.get(
        "runtime_backlog_time_max_depth", 0
    )
    result.metrics["runtime_backlog_time_max_state"] = run_summary.get(
        "runtime_backlog_time_max_state", ""
    )
    result.metrics["ticks_dropped_backpressure_total"] = drops_total

    # Speedup-dependent thresholds
    BACKLOG_THRESHOLDS = {
        "1x": 0.25,
        "3x": 0.35,
        "5x": 0.75,
        "8x": 1.5,
        "10x": 2.0,
    }
    threshold = BACKLOG_THRESHOLDS.get(speedup, 0.75)
    result.metrics["backlog_threshold_s"] = threshold

    if backlog_max_s > threshold:
        if speedup in ("1x", "3x"):
            result.passed = False
            result.failures.append(
                f"runtime_backlog_time_max_s = {backlog_max_s:.3f}s > {threshold}s @ {speedup}"
            )
        else:
            result.warnings.append(
                f"runtime_backlog_time_max_s = {backlog_max_s:.3f}s > {threshold}s @ {speedup}"
            )

    # Backpressure drops
    if drops_total > 0:
        if speedup == "1x":
            result.passed = False
            result.failures.append(f"ticks_dropped_backpressure_total = {drops_total} @ {speedup}")
        elif speedup == "3x":
            result.warnings.append(f"ticks_dropped_backpressure_total = {drops_total} @ {speedup}")
        # 5x+: informational only

    return result


def check_gate1_safety(run_dir: str, speedup: str) -> GateResult:
    """
    Gate 1: Safety as Invariant

    FAILS if:
    - success_metrics.safety.hard_fail == true
    - success_metrics.outcome in {COLLISION, CRASH}

    Informational (no fail):
    - count_no_pass_violations (NOPASS guard removed in Chunk 2A)
    - min_behind_m (follower may now pass leader — expected)
    - safety_emergency_count
    """
    result = GateResult(name="Gate1_Safety", passed=True)

    run_summary_path = os.path.join(run_dir, "run_summary.json")
    run_summary = load_json(run_summary_path)

    if not run_summary:
        result.passed = False
        result.failures.append(f"run_summary.json not found at {run_summary_path}")
        return result

    # Check no-pass violations
    violations = run_summary.get("count_no_pass_violations", 0)
    result.metrics["count_no_pass_violations"] = violations

    # NOPASS guard removed (Chunk 2A) — violations are expected; record only
    # if violations > 0: # Disabled — NOPASS guard no longer active

    # Check min_behind_m (informational — follower may now pass leader)
    min_behind = run_summary.get("min_behind_m")
    if min_behind is None:
        min_behind = float("inf")  # Not computed yet - treat as no violation
    result.metrics["min_behind_m"] = min_behind
    # min_behind < 0 is now expected (NOPASS guard removed) — informational only

    # Check safety_emergency_count (informational)
    safety_emergency = run_summary.get("safety_emergency_count", 0)
    result.metrics["safety_emergency_count"] = safety_emergency

    # Rec 1: Check success_metrics.safety.hard_fail AND outcome
    sm = run_summary.get("success_metrics", {})
    safety = sm.get("safety", {})
    hard_fail = safety.get("hard_fail", False)
    hard_fail_reason = safety.get("hard_fail_reason", "")
    outcome = sm.get("outcome", "")
    result.metrics["success_metrics_hard_fail"] = hard_fail
    result.metrics["success_metrics_hard_fail_reason"] = hard_fail_reason
    result.metrics["success_metrics_outcome"] = outcome

    if hard_fail:
        result.passed = False
        result.failures.append(f"success_metrics.safety.hard_fail = true: {hard_fail_reason}")

    # Also fail on collision/crash outcomes even if hard_fail field is missing
    FAIL_OUTCOMES = {"COLLISION", "CRASH"}
    if outcome in FAIL_OUTCOMES:
        result.passed = False
        result.failures.append(f"success_metrics.outcome = {outcome}")

    return result


def check_gate2_state_machine(run_dir: str, speedup: str) -> GateResult:
    """
    Gate 2: State Machine Stability

    FAILS for 1x/3x if:
    - TRANSITION_SNAPSHOT.log is missing
    - TRANSITION_SNAPSHOT.log has < 3 transition lines
    - Bounce count > 2 after first HOLD entry
    """
    result = GateResult(name="Gate2_StateMachine", passed=True)

    # Check TRANSITION_SNAPSHOT.log
    transition_log_path = os.path.join(run_dir, "TRANSITION_SNAPSHOT.log")

    # For 1x/3x, transition log must exist
    if speedup in ("1x", "3x"):
        if not os.path.exists(transition_log_path):
            result.passed = False
            result.failures.append(f"TRANSITION_SNAPSHOT.log not found at {transition_log_path}")
            return result

        # Check minimum transition lines
        with open(transition_log_path, "r") as f:
            lines = [l.strip() for l in f if l.strip() and "TRANSITION" in l]

        result.metrics["transition_count"] = len(lines)

        if len(lines) < 3:
            result.passed = False
            result.failures.append(
                f"TRANSITION_SNAPSHOT.log has only {len(lines)} transitions (minimum 3)"
            )

    # Check run_summary for bounce count
    run_summary_path = os.path.join(run_dir, "run_summary.json")
    run_summary = load_json(run_summary_path)

    if run_summary:
        # R5: For high speedups (5x+), warn if tick count suspiciously low
        if speedup not in ("1x", "3x"):
            ticks = run_summary.get("ticks_executed_total", 0)
            result.metrics["ticks_executed_total"] = ticks
            expected_min = 100  # At least 100 ticks for any real run
            if ticks < expected_min:
                result.warnings.append(
                    f"Low tick count ({ticks}) at {speedup} - possible early termination"
                )

        # Check state entry counts for REJOIN/HOLD bouncing
        state_entries = run_summary.get("state_entry_counts", {})
        hold_entries = state_entries.get("HOLD_PRECONTACT", 0)
        rejoin_entries = state_entries.get("REJOIN_PRECONTACT", 0)

        result.metrics["hold_precontact_entries"] = hold_entries
        result.metrics["rejoin_precontact_entries"] = rejoin_entries

        # If HOLD was entered at least once, check bouncing
        if hold_entries > 0:
            # Bounce count is roughly REJOIN entries after first HOLD
            # Simplified: if REJOIN > HOLD + 2, we're bouncing too much
            if rejoin_entries > hold_entries + 2:
                bounce_count = rejoin_entries - hold_entries
                if speedup in ("1x", "3x") and bounce_count > 2:
                    result.passed = False
                    result.failures.append(
                        f"REJOIN/HOLD bounce count = {bounce_count} > 2 after first HOLD"
                    )
                else:
                    result.warnings.append(
                        f"REJOIN/HOLD bouncing detected: rejoin={rejoin_entries}, hold={hold_entries}"
                    )

        # Check dwell accumulation
        max_dwell = run_summary.get("precontact_max_dwell_s", 0)
        result.metrics["precontact_max_dwell_s"] = max_dwell

        # For required dwell runs, check minimum
        if run_summary.get("required_dwell", False) and max_dwell < 5.0:
            if speedup in ("1x", "3x"):
                result.passed = False
                result.failures.append(f"Max HOLD dwell = {max_dwell:.1f}s < 5.0s required")

        # §8 Step 3e: Precision ladder (informational, not gating)
        max_tier = run_summary.get("max_tier_achieved", "N/A")
        result.metrics["max_tier_achieved"] = max_tier
        m3a_coarse = run_summary.get("m3a_coarse_achieved", False)
        result.metrics["m3a_coarse_achieved"] = m3a_coarse
        m3a_coarse_dwell = run_summary.get("m3a_coarse_best_dwell_s", 0)
        result.metrics["m3a_coarse_best_dwell_s"] = m3a_coarse_dwell

        # Step 1a: New milestone tiers (informational, not gating)
        m3a_fine = run_summary.get("m3a_fine_achieved", False)
        result.metrics["m3a_fine_achieved"] = m3a_fine
        m3a_cd = run_summary.get("m3a_coarse_delta_achieved", False)
        result.metrics["m3a_coarse_delta_achieved"] = m3a_cd
        # Also extract from success_metrics milestones for dwell details
        sm = run_summary.get("success_metrics", {}).get("milestones", {})
        m3a_fine_data = sm.get("m3a_fine_contact", {})
        result.metrics["m3a_fine_best_dwell_s"] = m3a_fine_data.get("best_dwell_s", 0)
        m3a_cd_data = sm.get("m3a_coarse_delta_contact", {})
        result.metrics["m3a_coarse_delta_best_dwell_s"] = m3a_cd_data.get("best_dwell_s", 0)

        # Step 1b: TECS speed tracking error (informational, not gating)
        result.metrics["tecs_speed_err_rms_mps"] = run_summary.get("tecs_speed_err_rms_mps")
        result.metrics["tecs_speed_err_p95_mps"] = run_summary.get("tecs_speed_err_p95_mps")

        # M3 progress longitudinal (informational, not gating)
        m3_prog_long = run_summary.get("m3_progress_longitudinal_achieved", False)
        result.metrics["m3_progress_longitudinal_achieved"] = m3_prog_long
        m3_prog_long_dwell = run_summary.get("m3_progress_longitudinal_best_dwell_s", 0)
        result.metrics["m3_progress_longitudinal_best_dwell_s"] = m3_prog_long_dwell

    return result


def generate_report(results: List[GateResult], run_dir: str, speedup: str) -> str:
    """Generate markdown report of gate check results."""
    # L-DIAG: Check for diagnostic mode from run_summary
    run_summary_path = os.path.join(run_dir, "run_summary.json")
    diag_flags = []
    if os.path.exists(run_summary_path):
        try:
            with open(run_summary_path, "r") as f:
                _rs = json.load(f)
            diag_flags = _rs.get("diagnostic_flags_active", [])
        except (json.JSONDecodeError, OSError):
            pass

    is_diagnostic = len(diag_flags) > 0
    title = "# FlightDock v9.10 Gate Check Report"
    if is_diagnostic:
        title = "# [DIAGNOSTIC — NOT PRODUCTION VALID] FlightDock v9.10 Gate Check Report"

    lines = [
        title,
        f"",
        f"**Run Directory**: `{run_dir}`",
        f"**Speedup**: {speedup}",
        f"",
    ]

    if is_diagnostic:
        lines.extend(
            [
                f"> **WARNING: DIAGNOSTIC MODE ACTIVE**",
                f"> ",
                f"> Active flags: {', '.join(diag_flags)}",
                f"> ",
                f"> Safety guards are suppressed. Gate results are informational only.",
                f"",
            ]
        )

    lines.extend(
        [
            f"## Summary",
            f"",
        ]
    )

    all_passed = all(r.passed for r in results)
    overall = "PASS" if all_passed else "FAIL"
    lines.append(f"**Overall Result**: **{overall}**")
    if is_diagnostic:
        lines.append(f"**Diagnostic Mode**: true")
    lines.append("")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"- {r.name}: **{status}**")

    lines.append("")
    lines.append("## Gate Details")
    lines.append("")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"### {r.name} - {status}")
        lines.append("")

        if r.metrics:
            lines.append("**Metrics:**")
            for k, v in sorted(r.metrics.items()):
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")

        if r.failures:
            lines.append("**Failures:**")
            for f in r.failures:
                lines.append(f"- {f}")
            lines.append("")

        if r.warnings:
            lines.append("**Warnings:**")
            for w in r.warnings:
                lines.append(f"- {w}")
            lines.append("")

    return "\n".join(lines)


def check_fatal_conditions(run_dir: str, speedup: str) -> GateResult:
    """
    v9.10 C7: Check for fatal conditions FIRST (source of truth priority).

    FAILS if:
    - fatal_reason exists in run_summary.json
    - unhandled_exception exists in run_summary.json
    - tick_hz_sim below speedup-scoped threshold (R10)

    This check MUST run before all other gates per Appendix C.
    """
    result = GateResult(name="Fatal_Conditions", passed=True)

    run_summary_path = os.path.join(run_dir, "run_summary.json")
    run_summary = load_json(run_summary_path)

    if not run_summary:
        # No run_summary is not a fatal - other gates will handle this
        result.warnings.append("run_summary.json not found (other gates will check)")
        return result

    # Check for fatal_reason
    fatal_reason = run_summary.get("fatal_reason")
    if fatal_reason:
        result.passed = False
        result.failures.append(f"fatal_reason: {fatal_reason}")
        result.metrics["fatal_reason"] = fatal_reason

    # Check for unhandled_exception
    unhandled = run_summary.get("unhandled_exception")
    if unhandled:
        result.passed = False
        result.failures.append(f"unhandled_exception: {unhandled}")
        result.metrics["unhandled_exception"] = unhandled

    # v9.10 C4: Check tick rate from run_summary
    # R10: Speedup-scoped tick rate thresholds (sim Hz, not wall Hz)
    TICK_HZ_THRESHOLDS = {
        "1x": 18.0,  # Full rate required
        "3x": 15.0,  # Allow some slip at moderate speedup
        "5x": 10.0,  # CPU-bound, lower expectation
        "8x": 5.0,  # Stress test only
        "10x": 3.0,  # Extreme stress test
    }

    tick_hz = run_summary.get("tick_hz_sim")
    tick_rate_reason = run_summary.get("tick_rate_reason", "unknown")
    if tick_hz is not None:
        result.metrics["tick_hz_sim"] = tick_hz
        result.metrics["tick_rate_reason"] = tick_rate_reason

        # R10: Use speedup parameter passed from main()
        threshold = TICK_HZ_THRESHOLDS.get(speedup, 10.0)
        result.metrics["tick_hz_threshold"] = threshold
        result.metrics["speedup"] = speedup

        if tick_hz < threshold:
            result.passed = False
            result.failures.append(
                f"tick_hz_sim = {tick_hz:.1f} < {threshold:.1f} Hz @ {speedup} (starvation)"
            )
        else:
            result.metrics["tick_hz_pass"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="FlightDock v9.10 Gate Checker")
    parser.add_argument("--run-dir", required=True, help="Path to run output directory")
    parser.add_argument(
        "--speedup",
        required=True,
        choices=["1x", "1.5x", "2x", "2.5x", "3x", "5x", "8x", "10x"],
        help="Speedup factor for this run",
    )
    parser.add_argument(
        "--output", help="Output report path (default: <run-dir>/gate_report_v9.10.md)"
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    speedup = args.speedup

    if not os.path.isdir(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[GATECHECK] Checking {run_dir} at {speedup}...")

    # R3: Check fatal conditions FIRST (source of truth priority)
    # v9.10 C7: Fatal check runs before all other gates
    fatal_check = check_fatal_conditions(run_dir, speedup)
    if not fatal_check.passed:
        print(f"[GATECHECK] Fatal_Conditions: FAIL (primary failure)")
        for f in fatal_check.failures:
            print(f"  FAIL: {f}")
        # R3: Short-circuit if fatal - other gates will fail anyway
        # Still generate report for full context, but mark as failed

    # Run all gate checks
    results = [
        fatal_check,  # R3: Fatal conditions first (checked above)
        check_gate0_timebase(run_dir, speedup),
        check_gate0_backlog_time(run_dir, speedup),  # Rec 5: Backlog time gate
        check_gate1_safety(run_dir, speedup),
        check_gate2_state_machine(run_dir, speedup),
    ]

    # Generate report
    report = generate_report(results, run_dir, speedup)

    # Write report
    output_path = args.output or os.path.join(run_dir, "gate_report_v9.10.md")
    ensure_output_dir(output_path)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"[GATECHECK] Report written to {output_path}")

    # Print summary
    all_passed = all(r.passed for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[GATECHECK] {r.name}: {status}")
        for f in r.failures:
            print(f"  FAIL: {f}")
        for w in r.warnings:
            print(f"  WARN: {w}")

    if all_passed:
        print(f"[GATECHECK] Overall: PASS")
        sys.exit(0)
    else:
        print(f"[GATECHECK] Overall: FAIL")
        sys.exit(2)


if __name__ == "__main__":
    main()
