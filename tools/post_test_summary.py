#!/usr/bin/env python3
"""
FlightDock Post-Test Summary Tool
=================================
Parse controller and SITL logs to produce:
  A) 1-page Markdown summary
  B) JSON metrics file
  C) CSV index update for run comparison

Usage:
  python3 tools/post_test_summary.py \\
    --controller-log <path> \\
    --sitl-log <optional path> \\
    --out <optional output md path> \\
    --compare 308 324 326

Date: 2026-01-02
"""

import argparse
import csv
import json
import sys
from datetime import datetime

from common_io import ensure_output_dir, timestamp_str, write_json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our parsing module
try:
    from log_parsing import (
        ControllerLogMetrics,
        SITLLogMetrics,
        extract_test_id,
        get_alert_severity,
        metrics_to_csv_row,
        metrics_to_dict,
        parse_controller_log,
        parse_sitl_log,
    )
except ImportError:
    # Handle running from different directory
    sys.path.insert(0, str(Path(__file__).parent))
    from log_parsing import (
        ControllerLogMetrics,
        SITLLogMetrics,
        extract_test_id,
        get_alert_severity,
        metrics_to_csv_row,
        metrics_to_dict,
        parse_controller_log,
        parse_sitl_log,
    )


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_REPO_ROOT = Path(__file__).parent.parent
SUMMARIES_DIR = "analysis/summaries"
RUN_METRICS_DIR = "analysis/run_metrics"
RUNS_METRICS_CSV = "analysis/runs_metrics.csv"
LOGS_DIR = "logs/intercept_tests/test_logs"

# M3b docking-quality snapshot (RMS/p95/p99 behind/below/cross on BOTH perception
# and truth side + Δ, dwell both-sides + simultaneous, relative closure velocity,
# QuadPlane TRUE airspeed, safety guards). Additive columns; graceful if pandas or
# the leg's paired_tick/speed_arbitration data are absent.
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import m3b_snapshot as _m3b
except Exception:
    _m3b = None


def m3b_csv_fields(leg_dir):
    """Flat M3b metrics for one leg dir; {} if unavailable (graceful)."""
    if _m3b is None:
        return {}
    try:
        return _m3b.flatten(_m3b.m3b_snapshot(str(leg_dir)))
    except Exception:
        return {}


CSV_COLUMNS = [
    "test_id",
    "version",
    "duration_s",
    "duration_min",
    "verdict",
    "precontact_holds",
    "contact_entries",
    "best_contact_dwell_s",
    "total_contact_time_s",
    "min_range_m",
    "min_range_time_s",
    "min_behind_err_m",
    "no_pass_abort_count",
    "alt_gate_count",
    "reached_docking_creep",
    "reached_hold_contact",
] + (_m3b.FLAT_COLUMNS if _m3b is not None else [])


# =============================================================================
# MARKDOWN GENERATION
# =============================================================================


def generate_verdict_emoji(verdict: str) -> str:
    """Get emoji for verdict."""
    if verdict == "SUCCESS":
        return "✅"
    elif verdict == "PARTIAL_CONTACT":
        return "🔶"
    elif verdict.startswith("FAIL"):
        return "❌"
    return "❓"


def generate_markdown_summary(
    test_id: int,
    metrics: ControllerLogMetrics,
    sitl_metrics: Optional[SITLLogMetrics],
    comparison_data: Optional[Dict[int, Dict[str, Any]]],
    max_alert_lines: int = 10,
    log_path: str = "",
    success_metrics: Optional[Dict[str, Any]] = None,
    wake_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate 1-page Markdown summary."""
    lines = []

    # Header
    emoji = generate_verdict_emoji(metrics.verdict)
    lines.append(f"# FlightDock Post-Test Summary — Test {test_id} {emoji}")
    lines.append("")

    # Metadata
    file_mtime = ""
    if log_path:
        try:
            mtime = Path(log_path).stat().st_mtime
            file_mtime = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            file_mtime = "unknown"

    lines.append(f"- **Date/Time**: {file_mtime}")
    lines.append(f"- **Controller Version**: {metrics.version or 'unknown'}")
    lines.append(f"- **Duration**: {metrics.duration_s:.1f}s ({metrics.duration_min:.1f} min)")
    lines.append(f"- **Verdict**: **{metrics.verdict}** — {metrics.verdict_reason}")
    lines.append("")

    # KPI Snapshot
    lines.append("## KPI Snapshot")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    req_s = metrics.precontact_required_s
    holds = metrics.precontact_holds_completed
    lines.append(f"| Precontact holds ≥ {req_s}s | {holds} |")
    lines.append(f"| Contact entries | {metrics.contact_entries} |")

    contact_dwells_ge_req = sum(
        1 for d in metrics.contact_dwell_times if d >= metrics.contact_required_s
    )
    lines.append(f"| Contact dwells ≥ {metrics.contact_required_s}s | {contact_dwells_ge_req} |")
    lines.append(f"| Best contact dwell (s) | {metrics.best_contact_dwell_s:.1f} |")
    lines.append(f"| Total contact time (s) | {metrics.total_contact_time_s:.1f} |")

    fpt = metrics.first_precontact_time_s
    first_pre = f"{fpt:.1f}" if fpt else "N/A"
    lines.append(f"| Time to first precontact (s) | {first_pre} |")

    fct = metrics.first_contact_time_s
    first_contact = f"{fct:.1f}" if fct else "N/A"
    lines.append(f"| Time to first contact (s) | {first_contact} |")

    mr = metrics.min_range_m
    min_r = f"{mr:.1f}" if mr != float("inf") else "N/A"
    lines.append(f"| Min range (m) @ t (s) | {min_r} @ {metrics.min_range_time_s:.1f}s |")

    mb = metrics.min_behind_err_m
    min_b = f"{mb:+.1f}" if mb != float("inf") else "N/A"
    lines.append(f"| Worst behind_err (m) | {min_b} |")
    lines.append(f"| NO_PASS_ABORT | {metrics.no_pass_abort_count} |")
    lines.append(f"| NO_PASS_WARN | {metrics.no_pass_warn_count} |")
    lines.append(f"| ALT_GATE | {metrics.alt_gate_count} |")
    lines.append(f"| OVERTAKE_PROTECT | {metrics.overtake_protect_count} |")
    lines.append("")

    # State Timeline (top N)
    lines.append("## State Timeline (Top 15)")
    lines.append("")
    if metrics.state_transitions:
        lines.append("| # | Time(s) | Duration | Transition | Reason |")
        lines.append("|--:|-------:|---------:|------------|--------|")
        for i, tr in enumerate(metrics.state_transitions[:15]):
            lines.append(
                f"| {i} | {tr.elapsed_s:.1f} | {tr.duration_s:.1f}s | "
                f"{tr.from_state} → {tr.to_state} | {tr.reason[:30]} |"
            )
        if len(metrics.state_transitions) > 15:
            more = len(metrics.state_transitions) - 15
            lines.append(f"| ... | ... | ... | ({more} more) | ... |")
    else:
        lines.append("_No state transitions found._")
    lines.append("")

    # Phase summary
    lines.append("### Phase Time Summary")
    lines.append("")
    lines.append("| Phase | Entries | Total Time |")
    lines.append("|-------|-------:|----------:|")
    sorted_phases = sorted(metrics.phase_times.items(), key=lambda x: -x[1])[:8]
    for phase, dur in sorted_phases:
        cnt = metrics.phase_counts.get(phase, 0)
        lines.append(f"| {phase} | {cnt} | {dur:.1f}s ({dur/60:.1f}m) |")
    lines.append("")

    # Safety + Warning Ledger
    lines.append("## Safety + Warning Ledger")
    lines.append("")
    if metrics.alert_counts:
        lines.append("| Tag | Count | Severity | Note |")
        lines.append("|-----|------:|----------|------|")
        for tag, count in sorted(metrics.alert_counts.items(), key=lambda x: -x[1]):
            sev, expected, note = get_alert_severity(tag)
            exp_mark = "✓" if expected else "⚠"
            lines.append(f"| {tag} | {count} | {sev} {exp_mark} | {note} |")
        lines.append("")

        # Sample alert lines
        if metrics.alert_events:
            lines.append("### Sample Alert Lines")
            lines.append("")
            for ae in metrics.alert_events[:max_alert_lines]:
                lines.append(f"> `[{ae.elapsed_s:.1f}s]` {ae.tag}: {ae.text[:100]}")
            lines.append("")
    else:
        lines.append("_No warnings or alerts found._")
        lines.append("")

    # ALTITUDE_CAPTURE section
    if metrics.altitude_capture_attempts > 0:
        lines.append("## Altitude Capture")
        lines.append("")
        lines.append(f"- Attempts: {metrics.altitude_capture_attempts}")
        lines.append(f"- Successes: {metrics.altitude_capture_success}")
        lines.append(f"- Total time in ALTITUDE_CAPTURE: {metrics.altitude_capture_time_s:.1f}s")
        lines.append("")

    # OVERTAKE_PROTECT breakdown
    if metrics.overtake_protect_count > 0:
        lines.append("## OVERTAKE_PROTECT Breakdown")
        lines.append("")
        lines.append("| Tier | Count |")
        lines.append("|------|------:|")
        for tier in ["TIER1", "TIER2", "TIER3"]:
            cnt = metrics.overtake_tier_counts.get(tier, 0)
            if cnt > 0:
                lines.append(f"| {tier} | {cnt} |")
        lines.append("")

    # Formation Success Metrics (if available)
    if success_metrics:
        lines.append("## Formation Success Metrics")
        lines.append("")

        # Outcome
        outcome = success_metrics.get("outcome", "UNKNOWN")
        outcome_emoji = {
            "SUCCESS_CONTACT": "✅",
            "SUCCESS_PRECONTACT_ONLY": "🔶",
            "PLATEAU": "🛑",
            "OVERSHOOT": "❌",
            "COLLISION": "💥",
            "FAIL": "❌",
            "INCOMPLETE": "⏳",
        }.get(outcome, "❓")
        lines.append(f"**Outcome:** {outcome} {outcome_emoji}")

        # Show plateau note if detected (even if outcome is not PLATEAU)
        plateau = success_metrics.get("plateau", {})
        if plateau.get("detected"):
            first_s = plateau.get("first_detected_s")
            accum_s = plateau.get("accum_s", 0)
            window_s = plateau.get("window_s", 60)
            behind_min = plateau.get("behind_min_m", 20)
            first_str = f"{first_s:.1f}s" if first_s is not None else "?"
            lines.append("")
            lines.append(
                f"**Plateau detected:** first={first_str}, accum={accum_s:.1f}s, window={window_s}s, behind_min={behind_min}m"
            )

        lines.append("")

        # Milestones table
        milestones = success_metrics.get("milestones", {})
        lines.append("### Milestones")
        lines.append("")
        lines.append("| Milestone | Achieved | First Dwell (s) | Best Dwell (s) |")
        lines.append("|-----------|:--------:|----------------:|---------------:|")

        milestone_names = [
            ("m1_staging", "M1 Staging"),
            ("m2_precontact", "M2 Precontact"),
            ("m3a_coarse_contact", "M3a Coarse Contact"),
            ("m3b_tight_contact", "M3b Tight Contact"),
        ]
        for key, label in milestone_names:
            m = milestones.get(key, {})
            achieved = "✅" if m.get("achieved", False) else "—"
            first = m.get("first_dwell_complete_s")
            first_str = f"{first:.1f}" if first is not None else "—"
            best = m.get("best_dwell_s")
            best_str = f"{best:.1f}" if best is not None else "—"
            lines.append(f"| {label} | {achieved} | {first_str} | {best_str} |")
        lines.append("")

        # Safety section
        safety = success_metrics.get("safety", {})
        hard_fail = safety.get("hard_fail", False)
        if hard_fail:
            reason = safety.get("hard_fail_reason", "Unknown")
            lines.append(f"### Safety: ❌ HARD FAIL — {reason}")
        else:
            lines.append("### Safety: ✅ No hard failures")
        lines.append("")

        violations = safety.get("violations", [])
        if violations:
            lines.append("| Code | Time (s) | Details |")
            lines.append("|------|--------:|---------|")
            for v in violations[:10]:  # Limit to 10
                code = v.get("code", "UNKNOWN")
                t = v.get("first_time_s", 0)
                details_raw = v.get("details", "")
                # Handle dict/None/str
                if isinstance(details_raw, dict):
                    details = json.dumps(details_raw)[:50]
                elif details_raw is None:
                    details = ""
                else:
                    details = str(details_raw)[:50]
                lines.append(f"| {code} | {t:.1f} | {details} |")
            if len(violations) > 10:
                lines.append(f"| ... | ... | ({len(violations) - 10} more) |")
            lines.append("")

        # Safety Gating Stats (if available)
        gs = success_metrics.get("geometry_sanity", {})
        if gs:
            lines.append("### Safety Gating")
            lines.append("")
            eval_count = gs.get("safety_evaluated_count", 0)
            s1_count = gs.get("s1_evaluated_count", 0)
            s2_count = gs.get("s2_evaluated_count", 0)
            lines.append(f"- Evaluated: {eval_count} (S1: {s1_count}, S2: {s2_count})")
            far_count = gs.get("safety_skipped_far_count", 0)
            s1_state_count = gs.get("s1_skipped_state_count", 0)
            insane_count = gs.get("center_insane_count", 0)
            lines.append(
                f"- Skipped: far={far_count}, s1_state={s1_state_count}, insane={insane_count}"
            )
            lines.append("")

        # Integrity Stats (for DRL validation)
        integrity = success_metrics.get("integrity", {})
        if integrity:
            integrity_ok = integrity.get("integrity_ok", False)
            status_emoji = "✅" if integrity_ok else "❌"
            lines.append(f"### Integrity: {status_emoji} {'PASS' if integrity_ok else 'FAIL'}")
            lines.append("")
            total = integrity.get("sample_count_total", 0)
            valid_tip = integrity.get("sample_count_valid_tip", 0)
            valid_center = integrity.get("sample_count_valid_center", 0)
            pct_tip = integrity.get("percent_valid_tip")
            pct_center = integrity.get("percent_valid_center")
            pct_safety = integrity.get("percent_safety_evaluated")
            pct_tip_str = f"{pct_tip:.1f}%" if pct_tip is not None else "N/A"
            pct_center_str = f"{pct_center:.1f}%" if pct_center is not None else "N/A"
            pct_safety_str = f"{pct_safety:.1f}%" if pct_safety is not None else "N/A"
            lines.append(
                f"- Samples: {total} total, {valid_tip} valid tip ({pct_tip_str}), {valid_center} valid center ({pct_center_str})"
            )
            denom = integrity.get("safety_eval_denominator", "valid_center")
            denom_count = integrity.get("safety_eval_denominator_count", valid_center)
            lines.append(f"- Safety evaluated: {pct_safety_str} of {denom_count} {denom} samples")
            failed_checks = integrity.get("failed_checks", [])
            if failed_checks:
                # Format structured failed_checks: [{"check": "...", "value": ..., "threshold": ...}]
                check_strs = []
                for fc in failed_checks:
                    if isinstance(fc, dict):
                        check_strs.append(f"{fc['check']}={fc['value']}<{fc['threshold']}")
                    else:
                        check_strs.append(str(fc))  # Backward compat
                lines.append(f"- **Failed checks:** {', '.join(check_strs)}")
            lines.append("")

        # Stability summary
        stability = success_metrics.get("stability", {})
        if stability:
            lines.append("### Stability")
            lines.append("")

            # Precontact stats
            pc = stability.get("precontact", {})
            if pc:
                n = pc.get("sample_count", 0)
                rms_b = pc.get("rms_behind")
                rms_c = pc.get("rms_cross")
                rms_z = pc.get("rms_below")
                p2p_b = pc.get("p2p_behind")
                p2p_c = pc.get("p2p_cross")
                p2p_z = pc.get("p2p_below")
                if n > 0 and rms_b is not None:
                    lines.append(
                        f"**Precontact:** RMS(b/c/z)={rms_b:.2f}/{rms_c:.2f}/{rms_z:.2f}m, P2P(b/c/z)={p2p_b:.2f}/{p2p_c:.2f}/{p2p_z:.2f}m, N={n}"
                    )
                else:
                    lines.append("**Precontact:** No data")

            # Contact stats
            ct = stability.get("contact", {})
            if ct:
                n = ct.get("sample_count", 0)
                rms_b = ct.get("rms_behind")
                rms_c = ct.get("rms_cross")
                rms_z = ct.get("rms_below")
                p2p_b = ct.get("p2p_behind")
                p2p_c = ct.get("p2p_cross")
                p2p_z = ct.get("p2p_below")
                if n > 0 and rms_b is not None:
                    lines.append(
                        f"**Contact:** RMS(b/c/z)={rms_b:.2f}/{rms_c:.2f}/{rms_z:.2f}m, P2P(b/c/z)={p2p_b:.2f}/{p2p_c:.2f}/{p2p_z:.2f}m, N={n}"
                    )
                else:
                    lines.append("**Contact:** No data")
            lines.append("")

        # Range tracking
        range_data = success_metrics.get("range_tracking", {})
        if range_data:
            min_tip = range_data.get("min_range_tip_m")
            min_tip_t = range_data.get("min_range_tip_time_s")
            min_center = range_data.get("min_range_center_m")
            min_center_t = range_data.get("min_range_center_time_s")
            lines.append("### Range Tracking")
            lines.append("")
            if min_tip is not None:
                if min_tip_t is not None:
                    lines.append(f"- Min range (boom tip): {min_tip:.2f}m @ {min_tip_t:.1f}s")
                else:
                    lines.append(f"- Min range (boom tip): {min_tip:.2f}m")
            if min_center is not None:
                if min_center_t is not None:
                    lines.append(f"- Min range (center): {min_center:.2f}m @ {min_center_t:.1f}s")
                else:
                    lines.append(f"- Min range (center): {min_center:.2f}m")
            lines.append("")

    # SITL Health (if provided)
    if sitl_metrics:
        lines.append("## Autopilot Health (SITL Console)")
        lines.append("")
        if sitl_metrics.healthy:
            lines.append("✅ **Healthy** — No critical issues detected.")
        else:
            lines.append("⚠️ **Issues detected:**")

        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|------:|")
        lines.append(f"| EKF events | {sitl_metrics.ekf_events} |")
        lines.append(f"| Arming events | {sitl_metrics.arming_events} |")
        lines.append(f"| TECS events | {sitl_metrics.tecs_events} |")
        lines.append(f"| GPS events | {sitl_metrics.gps_events} |")
        lines.append(f"| Failsafe events | {sitl_metrics.failsafe_events} |")
        lines.append(f"| Error events | {sitl_metrics.error_events} |")
        lines.append(f"| Panic/watchdog | {sitl_metrics.panic_events} |")
        lines.append("")

        if sitl_metrics.sample_lines:
            lines.append("### Sample Lines")
            for sline in sitl_metrics.sample_lines[:5]:
                lines.append(f"> {sline}")
            lines.append("")
        else:
            lines.append("_No EKF/TECS/failsafe/error keywords detected in console log._")
            lines.append("")

    # Comparison vs Baselines
    if comparison_data:
        lines.append("## Comparison vs Baselines")
        lines.append("")
        hdrs = "|".join(f" Test {t} |" for t in comparison_data.keys())
        lines.append(f"| Metric | This Test |{hdrs}")
        lines.append("|--------|----------:|" + "|----------:|" * len(comparison_data))

        # Build rows
        def get_val(d, key, fmt=".1f"):
            v = d.get(key)
            if v is None or v == "":
                return "N/A"
            if isinstance(v, (int, float)):
                return f"{v:{fmt}}"
            return str(v)

        this_data = metrics_to_dict(metrics)

        rows = [
            ("Verdict", "verdict", "s"),
            ("Duration (min)", "duration_min", ".1f"),
            ("Precontact holds", "precontact_holds_completed", "d"),
            ("Contact entries", "contact_entries", "d"),
            ("Best dwell (s)", "best_contact_dwell_s", ".1f"),
            ("Total contact (s)", "total_contact_time_s", ".1f"),
            ("Min range (m)", "min_range_m", ".1f"),
            ("NO_PASS_ABORT", "no_pass_abort_count", "d"),
        ]

        for label, key, fmt in rows:
            this_val = get_val(this_data, key, fmt)
            comp_vals = [get_val(comparison_data[t], key, fmt) for t in comparison_data.keys()]
            lines.append(f"| {label} | {this_val} |" + "|".join(f" {v} |" for v in comp_vals))
        lines.append("")

        # Improvement/regression notes
        lines.append("### Notes")
        lines.append("")
        for t, d in comparison_data.items():
            this_contacts = metrics.contact_entries
            other_contacts = d.get("contact_entries", 0) or 0
            this_holds = metrics.precontact_holds_completed
            other_holds = d.get("precontact_holds_completed", 0) or 0

            if this_contacts > other_contacts:
                lines.append(
                    f"- **Improved** vs Test {t}: More contact entries "
                    f"({this_contacts} vs {other_contacts})"
                )
            elif this_contacts < other_contacts:
                lines.append(
                    f"- **Regressed** vs Test {t}: Fewer contact entries "
                    f"({this_contacts} vs {other_contacts})"
                )

            if this_holds > other_holds:
                lines.append(
                    f"- **Improved** vs Test {t}: More precontact holds "
                    f"({this_holds} vs {other_holds})"
                )
        lines.append("")

    # Wake Turbulence Zone Analysis (if available)
    if wake_stats is not None:
        wake_section = generate_wake_zone_summary_section(wake_stats)
        lines.append(wake_section)

    # Footer
    lines.append("---")
    lines.append(f"*Generated by post_test_summary.py on {timestamp_str()}*")

    return "\n".join(lines)


# =============================================================================
# WAKE TURBULENCE ZONE ANALYSIS
# =============================================================================


def analyze_wake_zone_events(controller_log_path: str) -> Dict[str, Any]:
    """
    Extract and analyze wake zone CORE entries and PERIPHERY observations from controller log.

    Parses the two-vortex-core model log format:
    - [WAKE] ENTERED CORE behind=... cross=... below=... intensity=... dist=... side=...
    - [WAKE] EXITED CORE dwell=...s
    - [WAKE] CORE behind=... intensity=... dist=... lat_clear=... vert_clear=...
    - [WAKE] PERIPHERY behind=... intensity=... dist=... lat_clear=... vert_clear=...

    Args:
        controller_log_path: Path to controller log file

    Returns:
        Dictionary with analysis results including:
            - total_core_entries: Number of CORE zone entries
            - total_time_in_core_s: Total dwell time in CORE
            - max_hazard_intensity: Maximum intensity observed
            - periphery_observations: Count of PERIPHERY log lines
            - min_dist_to_core_m: Minimum distance to any vortex core
            - min_dist_to_core_time_s: Timestamp when min distance occurred
            - percent_in_periphery_150m: Percentage in PERIPHERY when behind < 150m
            - entries: List of CORE entry event details
    """
    import re

    core_entries = []
    periphery_count = 0
    max_intensity = 0.0
    current_entry = None

    # New tracking variables
    min_dist_to_core_m = float("inf")
    min_dist_to_core_time_s = None
    core_observation_count = 0  # Periodic CORE logs (not entry/exit events)
    periphery_count_within_150m = 0  # PERIPHERY obs when behind < 150m
    total_obs_within_150m = 0  # All obs when behind < 150m

    try:
        with open(controller_log_path, "r") as f:
            for line in f:
                if "[WAKE]" not in line:
                    continue

                if "ENTERED CORE" in line:
                    # Parse CORE entry event
                    # Example: [WAKE] ENTERED CORE behind=50.2m cross=3.5m below=25.1m intensity=0.75 dist=1.2m side=right
                    match = re.search(
                        r"behind=([\d.-]+)m cross=([\d.-]+)m below=([\d.-]+)m "
                        r"intensity=([\d.]+) dist=([\d.]+)m side=(\w+)",
                        line,
                    )
                    if match:
                        # Extract timestamp from line prefix
                        time_match = re.search(r"S=\s*([\d.]+)", line)
                        timestamp = float(time_match.group(1)) if time_match else 0.0

                        current_entry = {
                            "entry_time_s": timestamp,
                            "behind_m": float(match.group(1)),
                            "cross_m": float(match.group(2)),
                            "below_m": float(match.group(3)),
                            "intensity": float(match.group(4)),
                            "dist_m": float(match.group(5)),
                            "side": match.group(6),
                        }
                        max_intensity = max(max_intensity, current_entry["intensity"])

                        # Track min distance from entry events too
                        dist = current_entry["dist_m"]
                        if dist < min_dist_to_core_m:
                            min_dist_to_core_m = dist
                            min_dist_to_core_time_s = timestamp

                elif "EXITED CORE" in line and current_entry:
                    # Parse CORE exit event
                    # Example: [WAKE] EXITED CORE dwell=12.5s
                    match = re.search(r"dwell=([\d.]+)s", line)
                    if match:
                        time_match = re.search(r"S=\s*([\d.]+)", line)
                        exit_time = float(time_match.group(1)) if time_match else 0.0

                        current_entry["exit_time_s"] = exit_time
                        current_entry["dwell_s"] = float(match.group(1))
                        core_entries.append(current_entry)
                        current_entry = None

                elif "[WAKE] PERIPHERY" in line:
                    # Count PERIPHERY observations
                    periphery_count += 1

                    # Parse: behind=X.Xm intensity=X.XX dist=X.Xm ...
                    # New format with behind=, or old format without
                    match = re.search(
                        r"behind=([\d.]+)m.*intensity=([\d.]+).*dist=([\d.]+)m", line
                    )
                    if match:
                        behind = float(match.group(1))
                        intensity = float(match.group(2))
                        dist = float(match.group(3))
                        max_intensity = max(max_intensity, intensity)

                        # Track min distance
                        if dist < min_dist_to_core_m:
                            min_dist_to_core_m = dist
                            time_match = re.search(r"S=\s*([\d.]+)", line)
                            min_dist_to_core_time_s = (
                                float(time_match.group(1)) if time_match else None
                            )

                        # Track percent in periphery within 150m
                        if behind < 150.0:
                            periphery_count_within_150m += 1
                            total_obs_within_150m += 1
                    else:
                        # Old format without behind= field
                        match_old = re.search(r"intensity=([\d.]+)", line)
                        if match_old:
                            max_intensity = max(max_intensity, float(match_old.group(1)))

                elif "[WAKE] CORE" in line and "ENTERED" not in line and "EXITED" not in line:
                    # Periodic CORE status log
                    core_observation_count += 1

                    # Parse: behind=X.Xm intensity=X.XX dist=X.Xm ...
                    match = re.search(
                        r"behind=([\d.]+)m.*intensity=([\d.]+).*dist=([\d.]+)m", line
                    )
                    if match:
                        behind = float(match.group(1))
                        intensity = float(match.group(2))
                        dist = float(match.group(3))
                        max_intensity = max(max_intensity, intensity)

                        if dist < min_dist_to_core_m:
                            min_dist_to_core_m = dist
                            time_match = re.search(r"S=\s*([\d.]+)", line)
                            min_dist_to_core_time_s = (
                                float(time_match.group(1)) if time_match else None
                            )

                        # Track within 150m (CORE counts toward total, not periphery)
                        if behind < 150.0:
                            total_obs_within_150m += 1
                    else:
                        # Old format without behind= field
                        match_old = re.search(r"intensity=([\d.]+)", line)
                        if match_old:
                            max_intensity = max(max_intensity, float(match_old.group(1)))

    except FileNotFoundError:
        pass  # Return empty stats if file not found

    # Handle unclosed entry (still in CORE at end of test)
    if current_entry:
        current_entry["exit_time_s"] = None
        current_entry["dwell_s"] = None
        core_entries.append(current_entry)

    # Compute summary statistics
    total_dwell = sum(e["dwell_s"] for e in core_entries if e["dwell_s"] is not None)

    # Compute percent time in periphery (within 150m range)
    percent_in_periphery_150m = (
        (periphery_count_within_150m / total_obs_within_150m * 100)
        if total_obs_within_150m > 0
        else 0.0
    )

    return {
        "total_core_entries": len(core_entries),
        "total_time_in_core_s": total_dwell,
        "max_hazard_intensity": max_intensity,
        "periphery_observations": periphery_count,
        "core_observation_count": core_observation_count,
        "min_dist_to_core_m": min_dist_to_core_m if min_dist_to_core_m != float("inf") else None,
        "min_dist_to_core_time_s": min_dist_to_core_time_s,
        "percent_in_periphery_150m": percent_in_periphery_150m,
        "periphery_count_within_150m": periphery_count_within_150m,
        "total_obs_within_150m": total_obs_within_150m,
        "entries": core_entries,
    }


def generate_wake_zone_summary_section(wake_stats: Dict[str, Any]) -> str:
    """
    Generate markdown summary section for wake zone analysis.

    Safety assessment is based on CORE entries (dangerous zone) only.
    PERIPHERY observations are logged as warnings but not counted as "in zone".

    Args:
        wake_stats: Output from analyze_wake_zone_events()

    Returns:
        Markdown-formatted summary string
    """
    lines = []
    lines.append("## Wake Turbulence Zone Analysis")
    lines.append("")

    core_entries = wake_stats["total_core_entries"]
    periphery_obs = wake_stats.get("periphery_observations", 0)

    if core_entries == 0 and periphery_obs == 0:
        lines.append("**OK** - No wake zone activity detected")
        lines.append("")
        return "\n".join(lines)

    # Safety assessment based on CORE entries
    if core_entries > 0:
        lines.append(f"**CRITICAL** - {core_entries} CORE zone entries detected!")
    elif periphery_obs > 0:
        lines.append(f"**WARNING** - Periphery observations ({periphery_obs}) but no CORE entries")

    lines.append("")
    lines.append(f"- Core entries: {core_entries}")
    lines.append(f"- Total time in core: {wake_stats['total_time_in_core_s']:.1f}s")
    lines.append(f"- Max hazard intensity: {wake_stats['max_hazard_intensity']:.2f}")
    lines.append(f"- Periphery observations: {periphery_obs}")

    # New metrics: min distance to core
    min_dist = wake_stats.get("min_dist_to_core_m")
    min_dist_t = wake_stats.get("min_dist_to_core_time_s")
    if min_dist is not None:
        time_str = f" @ {min_dist_t:.1f}s" if min_dist_t is not None else ""
        lines.append(f"- Min dist to core: {min_dist:.2f}m{time_str}")

    # New metrics: percent in periphery within 150m
    pct_periph = wake_stats.get("percent_in_periphery_150m", 0)
    periph_150m = wake_stats.get("periphery_count_within_150m", 0)
    total_150m = wake_stats.get("total_obs_within_150m", 0)
    if total_150m > 0:
        lines.append(
            f"- Percent in periphery (<150m): {pct_periph:.1f}% ({periph_150m}/{total_150m} obs)"
        )

    lines.append("")

    # Detailed CORE entry table
    if wake_stats["entries"]:
        lines.append("### Core Entry Details")
        lines.append("")
        lines.append(
            "| Entry (s) | Exit (s) | Dwell | Behind | Cross | Below | Intensity | Side |"
        )
        lines.append(
            "|-----------|----------|-------|--------|-------|-------|-----------|------|"
        )

        for entry in wake_stats["entries"][:10]:  # Limit to 10 entries
            exit_s = (
                f"{entry.get('exit_time_s', 0):.1f}" if entry.get("exit_time_s") else "ongoing"
            )
            dwell = f"{entry.get('dwell_s', 0):.1f}s" if entry.get("dwell_s") else "ongoing"
            lines.append(
                f"| {entry['entry_time_s']:.1f} | {exit_s} | {dwell} | "
                f"{entry['behind_m']:.1f} | {entry['cross_m']:.1f} | {entry['below_m']:.1f} | "
                f"{entry['intensity']:.2f} | {entry['side']} |"
            )

        if len(wake_stats["entries"]) > 10:
            lines.append(
                f"| ... | ... | ... | ... | ... | ... | ... | ({len(wake_stats['entries']) - 10} more) |"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# FILE I/O
# =============================================================================


def write_json_metrics(test_id: int, metrics: ControllerLogMetrics, output_path: Path) -> None:
    """Write JSON metrics file."""
    data = metrics_to_dict(metrics)
    data["test_id"] = test_id
    data["generated_at"] = datetime.now().isoformat()

    write_json(output_path, data)


def update_csv_index(test_id: int, metrics: ControllerLogMetrics, csv_path: Path,
                     extra: dict = None) -> None:
    """Update or append to CSV index file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = metrics_to_csv_row(test_id, metrics)
    if extra:
        row.update(extra)

    # Read existing data
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    # Check if row exists for this test_id
    found = False
    for i, existing in enumerate(existing_rows):
        if str(existing.get("test_id")) == str(test_id):
            # Update if same version or replace
            existing_rows[i] = row
            found = True
            break

    if not found:
        existing_rows.append(row)

    # Sort by test_id
    existing_rows.sort(key=lambda x: int(x.get("test_id", 0)))

    # Write back
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in existing_rows:
            # Ensure all columns present
            clean_row = {c: r.get(c, "") for c in CSV_COLUMNS}
            writer.writerow(clean_row)


def load_comparison_metrics(test_ids: List[int], repo_root: Path) -> Dict[int, Dict[str, Any]]:
    """Load metrics for comparison tests."""
    result = {}
    metrics_dir = repo_root / RUN_METRICS_DIR
    logs_dir = repo_root / LOGS_DIR

    for tid in test_ids:
        # Try JSON first
        json_path = metrics_dir / f"test_{tid}.json"
        if json_path.exists():
            with open(json_path) as f:
                result[tid] = json.load(f)
            continue

        # Try parsing log on-demand
        log_path = logs_dir / f"intercept_rejoin-{tid}.log"
        if log_path.exists():
            m = parse_controller_log(str(log_path))
            result[tid] = metrics_to_dict(m)
            # Save for future use
            write_json_metrics(tid, m, json_path)
            continue

        # Not found
        print(f"Warning: Could not find metrics for Test {tid}", file=sys.stderr)

    return result


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate post-test summary and metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 tools/post_test_summary.py \\
    --controller-log logs/.../intercept_rejoin-327.log

  # With SITL log and comparison
  python3 tools/post_test_summary.py \\
    --controller-log logs/.../intercept_rejoin-327.log \\
    --sitl-log logs/.../sitl_console_20260102_114321.log \\
    --compare 308 324 326
""",
    )

    parser.add_argument(
        "--controller-log",
        "-c",
        required=True,
        help="Path to controller log file (intercept_rejoin-XXX.log)",
    )
    parser.add_argument("--sitl-log", "-s", help="Path to SITL console log file (optional)")
    parser.add_argument(
        "--out",
        "-o",
        help="Output Markdown path (default: analysis/summaries/test_XXX_summary.md)",
    )
    parser.add_argument(
        "--compare", nargs="+", type=int, help="Test IDs to compare against (e.g., 308 324 326)"
    )
    parser.add_argument("--repo-root", help="Repository root directory (default: auto-detect)")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=10,
        help="Max alert lines to include in summary (default: 10)",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip updating CSV index")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages")
    parser.add_argument(
        "--run-summary",
        type=str,
        default=None,
        help="Explicit path to run_summary.json (overrides auto-discovery)",
    )

    args = parser.parse_args()

    # Determine repo root
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        repo_root = DEFAULT_REPO_ROOT

    # Extract test ID
    test_id = extract_test_id(args.controller_log)
    if test_id is None:
        print("Error: Could not extract test ID from log filename", file=sys.stderr)
        print("Expected format: intercept_rejoin-XXX.log", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Parsing Test {test_id}...")

    # Parse controller log
    metrics = parse_controller_log(args.controller_log, max_alert_lines=args.max_lines * 2)

    if metrics.verdict == "FILE_NOT_FOUND":
        print(f"Error: Controller log not found: {args.controller_log}", file=sys.stderr)
        sys.exit(1)

    # Parse SITL log (optional)
    sitl_metrics = None
    if args.sitl_log:
        sitl_metrics = parse_sitl_log(args.sitl_log)

    # Load success_metrics from run_summary.json
    success_metrics = None
    run_summary_path = None

    # Priority 1: Explicit --run-summary argument
    if args.run_summary:
        run_summary_path = Path(args.run_summary)
        if not run_summary_path.exists():
            if not args.quiet:
                print(
                    f"  Warning: --run-summary path not found: {run_summary_path}", file=sys.stderr
                )
            run_summary_path = None

    # Priority 2: Same directory as controller log
    if run_summary_path is None:
        log_dir = Path(args.controller_log).parent
        candidate = log_dir / "run_summary.json"
        if candidate.exists():
            run_summary_path = candidate

    # Priority 3: Fallback discovery from intercept_rejoin log naming
    # Pattern: intercept_rejoin-{test_id}_{timestamp}.log
    # Look for: run_summary-{test_id}_{timestamp}.json
    if run_summary_path is None:
        import re

        log_name = Path(args.controller_log).name
        match = re.match(r"intercept_rejoin-(\d+)_(\d{8}_\d{6})\.log", log_name)
        if match:
            test_id_str, timestamp_str = match.groups()
            log_dir = Path(args.controller_log).parent
            candidate = log_dir / f"run_summary-{test_id_str}_{timestamp_str}.json"
            if candidate.exists():
                run_summary_path = candidate
                if not args.quiet:
                    print(f"  Found run_summary via fallback: {run_summary_path}")

    # Load if found
    if run_summary_path and run_summary_path.exists():
        try:
            with open(run_summary_path, "r") as f:
                run_summary = json.load(f)
            success_metrics = run_summary.get("success_metrics")
            if success_metrics and not args.quiet:
                print(f"  Loaded success_metrics from {run_summary_path}")
        except Exception as e:
            if not args.quiet:
                print(f"  Warning: Could not load run_summary.json: {e}", file=sys.stderr)

    # Load comparison data
    comparison_data = None
    if args.compare:
        comparison_data = load_comparison_metrics(args.compare, repo_root)

    # Analyze wake zone events from controller log (two-vortex-core model)
    wake_stats = analyze_wake_zone_events(args.controller_log)
    if (
        wake_stats["total_core_entries"] > 0 or wake_stats.get("periphery_observations", 0) > 0
    ) and not args.quiet:
        print(
            f"  Wake zone: {wake_stats['total_core_entries']} CORE entries, "
            f"{wake_stats.get('periphery_observations', 0)} periphery obs "
            f"(max intensity: {wake_stats['max_hazard_intensity']:.2f})"
        )

    # Generate Markdown
    md_content = generate_markdown_summary(
        test_id=test_id,
        metrics=metrics,
        sitl_metrics=sitl_metrics,
        comparison_data=comparison_data,
        max_alert_lines=args.max_lines,
        log_path=args.controller_log,
        success_metrics=success_metrics,
        wake_stats=wake_stats,
    )

    # Determine output paths
    if args.out:
        md_path = Path(args.out)
    else:
        md_path = repo_root / SUMMARIES_DIR / f"test_{test_id}_summary.md"

    json_path = repo_root / RUN_METRICS_DIR / f"test_{test_id}.json"
    csv_path = repo_root / RUNS_METRICS_CSV

    # Write outputs
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write(md_content)

    write_json_metrics(test_id, metrics, json_path)

    if not args.no_csv:
        update_csv_index(test_id, metrics, csv_path,
                         extra=m3b_csv_fields(Path(args.controller_log).parent))

    if not args.quiet:
        print(f"✓ Summary written: {md_path}")
        print(f"✓ Metrics written: {json_path}")
        if not args.no_csv:
            print(f"✓ CSV updated: {csv_path}")
        print()
        print(f"Test {test_id} Verdict: {metrics.verdict}")
        print(f"  Duration: {metrics.duration_min:.1f} min")
        print(f"  Precontact holds: {metrics.precontact_holds_completed}")
        print(f"  Contact entries: {metrics.contact_entries}")
        print(f"  Min range: {metrics.min_range_m:.1f}m")


if __name__ == "__main__":
    main()
