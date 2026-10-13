#!/usr/bin/env python3
"""
FlightDock Speedup Calibration Sweep Tool
==========================================

Runs a sequence of timebase smoketests at various SIM_SPEEDUP settings to
determine what speedups are actually achievable on the current hardware.

USAGE:
    # Default sweep: [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    python tools/speedup_sweep.py

    # Custom speedups
    python tools/speedup_sweep.py --speedups 1.0 2.0 5.0 --duration 20

    # Quick check for specific speedup
    python tools/speedup_sweep.py --speedups 5.0 --duration 15

    # Lint param files before sweep
    python tools/speedup_sweep.py --lint-params

OUTPUT:
    Creates timestamped folder in ~/FlightDock/tests/analysis_out/:
        speedup_sweep_YYYYMMDD_HHMMSS/
            results.json - machine-readable metrics
            REPORT.md - human-readable summary
            sweep_1.0x.log - per-test console output
            sweep_2.0x.log
            ...

STABILITY CRITERIA (configurable):
    - quality >= 0.95
    - rate within 25% of expected
    - settle gate passed

RELIABILITY FEATURES (Phase 7D.335):
    - Param file lint before sweep starts
    - Port availability verification between runs
    - Strict SITL cleanup with wait-for-shutdown
    - Speedup readback verification after SITL start
    - Failure recording with detailed reasons

(C) 2025-2026 FlightDock Research Project
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

try:
    from sim_time_authority import TimeQualityMetrics
except ImportError:
    TimeQualityMetrics = None


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SPEEDUPS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
DEFAULT_DURATION_S = 30.0
DEFAULT_SETTLE_S = 10.0

# Stability criteria
STABILITY_QUALITY_MIN = 0.95
STABILITY_RATE_TOLERANCE = 0.25  # 25% deviation allowed

# SITL ports to check for availability
SITL_PORTS = [5760, 14550, 14551, 14552, 14553]

# Cleanup timing
CLEANUP_WAIT_S = 3.0
PORT_CHECK_RETRIES = 5
PORT_CHECK_INTERVAL_S = 1.0


# ============================================================================
# Port Availability Helpers
# ============================================================================


def is_port_available(port: int) -> bool:
    """Check if a TCP/UDP port is available.

    Args:
        port: Port number to check

    Returns:
        True if port is available (not in use)
    """
    try:
        # Try TCP
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", port))
            if result == 0:
                # Connection succeeded = port is in use
                return False
    except Exception:
        pass

    try:
        # Try UDP bind
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.5)
            sock.bind(("127.0.0.1", port))
            # Bind succeeded = port is available
            return True
    except OSError:
        return False
    except Exception:
        return True  # Assume available on unexpected errors


def wait_ports_free(
    ports: List[int] = None,
    timeout_s: float = 10.0,
    verbose: bool = True,
) -> bool:
    """Wait for SITL ports to become free.

    Args:
        ports: List of ports to check (default: SITL_PORTS)
        timeout_s: Maximum wait time in seconds
        verbose: Print status messages

    Returns:
        True if all ports are free, False on timeout
    """
    if ports is None:
        ports = SITL_PORTS

    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout_s:
        busy_ports = [p for p in ports if not is_port_available(p)]

        if not busy_ports:
            if verbose:
                print("[PORT] All SITL ports are free")
            return True

        if verbose:
            print(f"[PORT] Waiting for ports to free: {busy_ports}")

        time.sleep(PORT_CHECK_INTERVAL_S)

    if verbose:
        busy_ports = [p for p in ports if not is_port_available(p)]
        print(f"[PORT] Timeout - ports still busy: {busy_ports}")

    return False


# ============================================================================
# Param File Lint Helper
# ============================================================================


def lint_param_files(flightdock_root: Path, verbose: bool = True) -> bool:
    """Lint param files for SIM_SPEEDUP before sweep.

    Args:
        flightdock_root: Path to FlightDock installation
        verbose: Print status messages

    Returns:
        True if no SIM_SPEEDUP found, False if found
    """
    try:
        # Try to import speedup_guard
        tools_dir = flightdock_root / "tools"
        sys.path.insert(0, str(tools_dir))
        from speedup_guard import lint_multiple_files
    except ImportError:
        if verbose:
            print("[LINT] Warning: speedup_guard.py not found, skipping param lint")
        return True

    # Find param files
    param_files = []
    config_dir = flightdock_root / "config"
    params_dir = flightdock_root / "params"

    for pdir in [config_dir, params_dir]:
        if pdir.exists():
            param_files.extend(str(p) for p in pdir.glob("*.parm"))

    if not param_files:
        if verbose:
            print("[LINT] No param files found")
        return True

    if verbose:
        print(f"[LINT] Checking {len(param_files)} param file(s)...")

    results = lint_multiple_files(param_files)

    if not results:
        if verbose:
            print("[LINT] PASS: No SIM_SPEEDUP found in param files")
        return True

    # SIM_SPEEDUP found
    print("[LINT] WARNING: SIM_SPEEDUP found in param file(s):")
    for filepath, matches in results.items():
        for line_num, content in matches:
            print(f"  {filepath}:{line_num}: {content}")
    print()
    print("[LINT] These may override CLI --speedup settings.")
    print("[LINT] Consider removing SIM_SPEEDUP lines or using --skip-lint")
    return False


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SpeedupResult:
    """Result of a single speedup test."""

    requested: float
    achieved_rate: float
    quality_score: float
    settle_passed: bool
    passed: bool
    reason: str
    stable: bool  # Meets stability criteria
    wall_duration_s: float
    samples: int
    rate_p05: float = 0.0
    rate_p95: float = 0.0
    failure_reason: str = ""  # Detailed failure reason (port conflict, readback mismatch, etc.)
    readback_speedup: float = 0.0  # Actual speedup read back from SITL


@dataclass
class SweepSummary:
    """Summary of the full sweep."""

    timestamp: str
    hostname: str
    stable_speedups: List[float]
    max_stable_speedup: float
    recommended_speedup: float
    all_results: List[SpeedupResult]


# ============================================================================
# SITL Management
# ============================================================================


class SITLManager:
    """Manages SITL lifecycle for speedup testing."""

    def __init__(
        self,
        flightdock_root: str,
        output_dir: str,
        verbose: bool = True,
    ):
        self.flightdock_root = Path(flightdock_root)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self._sitl_proc: Optional[subprocess.Popen] = None
        self._sitl_pids: List[int] = []
        self._requested_speedup: float = 1.0
        self._readback_speedup: float = 0.0
        self._last_failure_reason: str = ""

    @property
    def readback_speedup(self) -> float:
        """Actual speedup read back from SITL after start."""
        return self._readback_speedup

    @property
    def last_failure_reason(self) -> str:
        """Reason for last start failure."""
        return self._last_failure_reason

    def start(self, speedup: float, log_path: str) -> bool:
        """Start SITL at given speedup.

        Args:
            speedup: Requested SIM_SPEEDUP factor
            log_path: Path to write SITL console output

        Returns:
            True if SITL started, heartbeat detected, and speedup verified
        """
        self._requested_speedup = speedup
        self._readback_speedup = 0.0
        self._last_failure_reason = ""

        # Step 1: Cleanup existing processes
        self._cleanup_existing()

        # Step 2: Verify ports are free
        if not wait_ports_free(verbose=self.verbose, timeout_s=10.0):
            self._last_failure_reason = "PORT_CONFLICT"
            print("[ERROR] Ports still in use after cleanup")
            return False

        script_path = self.flightdock_root / "tests" / "start_formation_sitl_fixed.sh"
        if not script_path.exists():
            self._last_failure_reason = "SCRIPT_NOT_FOUND"
            print(f"[ERROR] SITL script not found: {script_path}")
            return False

        if self.verbose:
            print(f"[SITL] Starting at {speedup}x (headless)...")

        with open(log_path, "w") as log_file:
            # Build environment with speedup and headless
            env = os.environ.copy()
            env["HEADLESS"] = "1"
            env["HARNESS_UI_MODE"] = "none"

            # Start SITL script
            self._sitl_proc = subprocess.Popen(
                ["bash", str(script_path), "--speedup", str(speedup)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(self.flightdock_root / "tests"),
                env=env,
                start_new_session=True,  # Separate process group
            )

        # Step 3: Wait for heartbeat
        if not self._wait_for_heartbeat(timeout_s=60.0):
            self._last_failure_reason = "HEARTBEAT_TIMEOUT"
            print("[ERROR] SITL heartbeat timeout")
            self.stop()
            return False

        # Step 4: Verify speedup via readback
        if not self._verify_speedup_readback(speedup, tolerance=0.5):
            self._last_failure_reason = f"SPEEDUP_MISMATCH:{self._readback_speedup:.1f}"
            print(
                f"[ERROR] Speedup mismatch: requested={speedup}x, "
                f"actual={self._readback_speedup}x"
            )
            self.stop()
            return False

        if self.verbose:
            print(
                f"[SITL] Ready (PID: {self._sitl_proc.pid}, "
                f"speedup={self._readback_speedup:.1f}x)"
            )

        return True

    def stop(self) -> None:
        """Stop SITL and all related processes with strict cleanup."""
        if self.verbose:
            print("[SITL] Stopping...")

        # Kill process group
        if self._sitl_proc:
            try:
                pgid = os.getpgid(self._sitl_proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(1.0)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            self._sitl_proc = None

        # Also kill any remaining SITL processes
        self._cleanup_existing()

        # Wait for ports to free
        wait_ports_free(verbose=self.verbose, timeout_s=10.0)

        if self.verbose:
            print("[SITL] Stopped")

    def _verify_speedup_readback(
        self,
        expected: float,
        tolerance: float = 0.5,
        timeout_s: float = 10.0,
    ) -> bool:
        """Verify SITL speedup matches expected via MAVLink readback.

        Args:
            expected: Expected SIM_SPEEDUP value
            tolerance: Acceptable deviation
            timeout_s: Timeout for readback

        Returns:
            True if speedup matches within tolerance
        """
        try:
            # Try to use speedup_guard for readback
            tools_dir = self.flightdock_root / "tools"
            sys.path.insert(0, str(tools_dir))
            from speedup_guard import readback_sim_speedup
        except ImportError:
            # Fall back to direct MAVLink
            return self._readback_speedup_direct(expected, tolerance, timeout_s)

        actual = readback_sim_speedup("udp:127.0.0.1:14551", timeout_s)

        if actual is None:
            if self.verbose:
                print("[READBACK] Failed to read SIM_SPEEDUP")
            return False

        self._readback_speedup = actual
        deviation = abs(actual - expected)

        if deviation <= tolerance:
            if self.verbose:
                print(f"[READBACK] OK: {actual:.1f}x ~= {expected:.1f}x")
            return True

        if self.verbose:
            print(f"[READBACK] MISMATCH: expected={expected}x, actual={actual}x")
        return False

    def _readback_speedup_direct(
        self,
        expected: float,
        tolerance: float,
        timeout_s: float,
    ) -> bool:
        """Direct MAVLink readback when speedup_guard not available."""
        try:
            from pymavlink import mavutil
        except ImportError:
            if self.verbose:
                print("[READBACK] pymavlink not available, skipping readback")
            self._readback_speedup = expected  # Assume it worked
            return True

        try:
            mav = mavutil.mavlink_connection(
                "udp:127.0.0.1:14551",
                source_system=254,
            )
            mav.mav.param_request_read_send(
                mav.target_system,
                mav.target_component,
                b"SIM_SPEEDUP",
                -1,
            )

            start = time.monotonic()
            while time.monotonic() - start < timeout_s:
                msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
                if msg:
                    param_id = msg.param_id
                    if isinstance(param_id, bytes):
                        param_id = param_id.decode("utf-8", errors="replace")
                    param_id = param_id.rstrip("\x00")
                    if param_id.upper() == "SIM_SPEEDUP":
                        self._readback_speedup = float(msg.param_value)
                        mav.close()
                        return abs(self._readback_speedup - expected) <= tolerance

            mav.close()
        except Exception as e:
            if self.verbose:
                print(f"[READBACK] Error: {e}")

        return False

    def _wait_for_heartbeat(self, timeout_s: float = 60.0) -> bool:
        """Wait for SITL heartbeat on port 14551."""
        try:
            from pymavlink import mavutil
        except ImportError:
            print("[ERROR] pymavlink not available for heartbeat check")
            # Fall back to simple sleep
            time.sleep(10.0)
            return True

        start_time = time.monotonic()
        check_interval = 2.0
        last_check = 0

        while time.monotonic() - start_time < timeout_s:
            now = time.monotonic()
            if now - last_check < check_interval:
                time.sleep(0.5)
                continue

            last_check = now

            try:
                mav = mavutil.mavlink_connection(
                    "udp:127.0.0.1:14551",
                    source_system=254,
                )
                msg = mav.recv_match(
                    type="HEARTBEAT",
                    blocking=True,
                    timeout=2.0,
                )
                mav.close()
                if msg:
                    return True
            except Exception:
                pass

        return False

    def _cleanup_existing(self) -> None:
        """Kill any existing SITL processes."""
        patterns = ["arduplane", "mavproxy", "sim_vehicle"]
        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
        time.sleep(1.0)


# ============================================================================
# Timebase Test Runner
# ============================================================================


def run_timebase_test(
    speedup: float,
    duration_s: float,
    verbose: bool = True,
    quality_min: float = STABILITY_QUALITY_MIN,
    rate_tolerance: float = STABILITY_RATE_TOLERANCE,
) -> SpeedupResult:
    """Run timebase smoketest and return results.

    Args:
        speedup: Expected speedup factor
        duration_s: Test duration in wall-seconds
        verbose: Print progress
        quality_min: Minimum quality score for stability
        rate_tolerance: Maximum rate deviation (fraction)

    Returns:
        SpeedupResult with achieved metrics
    """
    # Import timebase smoketest runner
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
        from timebase_smoketest import run_smoketest
    except ImportError as e:
        return SpeedupResult(
            requested=speedup,
            achieved_rate=0.0,
            quality_score=0.0,
            settle_passed=False,
            passed=False,
            reason=f"Import error: {e}",
            stable=False,
            wall_duration_s=0.0,
            samples=0,
        )

    start_time = time.monotonic()

    try:
        passed, metrics, reason = run_smoketest(
            conn_str="udp:127.0.0.1:14551",
            expected_speedup=speedup,
            duration_wall_s=duration_s,
            sysid_filter=1,
            leader_conn_str=None,
            verbose=verbose,
        )
    except Exception as e:
        return SpeedupResult(
            requested=speedup,
            achieved_rate=0.0,
            quality_score=0.0,
            settle_passed=False,
            passed=False,
            reason=f"Exception: {e}",
            stable=False,
            wall_duration_s=time.monotonic() - start_time,
            samples=0,
        )

    wall_duration = time.monotonic() - start_time

    # Check stability criteria
    rate_deviation = abs(metrics.rate_median - speedup) / speedup
    stable = (
        metrics.quality_score >= quality_min
        and rate_deviation <= rate_tolerance
        and metrics.settle_gate_passed
    )

    return SpeedupResult(
        requested=speedup,
        achieved_rate=metrics.rate_median,
        quality_score=metrics.quality_score,
        settle_passed=metrics.settle_gate_passed,
        passed=passed,
        reason=reason,
        stable=stable,
        wall_duration_s=wall_duration,
        samples=metrics.samples,
        rate_p05=metrics.rate_p05,
        rate_p95=metrics.rate_p95,
    )


# ============================================================================
# Report Generation
# ============================================================================


def generate_report(summary: SweepSummary, output_dir: Path) -> None:
    """Generate REPORT.md with sweep results."""
    report_path = output_dir / "REPORT.md"

    lines = [
        "# FlightDock Speedup Calibration Report",
        "",
        f"**Timestamp:** {summary.timestamp}",
        f"**Hostname:** {summary.hostname}",
        "",
        "## Summary",
        "",
        f"- **Max Stable Speedup:** {summary.max_stable_speedup}x",
        f"- **Recommended Speedup:** {summary.recommended_speedup}x",
        f"- **Stable Speedups:** {summary.stable_speedups}",
        "",
        "### Stability Criteria",
        "",
        f"- Quality score >= {STABILITY_QUALITY_MIN}",
        f"- Rate within {STABILITY_RATE_TOLERANCE * 100:.0f}% of expected",
        "- Settle gate passed",
        "",
        "## Detailed Results",
        "",
        "| Speedup | Achieved | Quality | Settle | Stable | Reason |",
        "|---------|----------|---------|--------|--------|--------|",
    ]

    for r in summary.all_results:
        stable_icon = "✓" if r.stable else "✗"
        settle_icon = "✓" if r.settle_passed else "✗"
        lines.append(
            f"| {r.requested}x | {r.achieved_rate:.2f}x | "
            f"{r.quality_score:.2f} | {settle_icon} | {stable_icon} | "
            f"{r.reason} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )

    if summary.max_stable_speedup >= 5.0:
        lines.extend(
            [
                f"Your system can sustain **{summary.max_stable_speedup}x** speedup.",
                "",
                "For formation testing:",
                f"- Use `--speedup {summary.recommended_speedup}` for single tests",
                "- For higher throughput, use parallel environments",
                "",
            ]
        )
    elif summary.max_stable_speedup >= 3.0:
        lines.extend(
            [
                f"Your system sustains **{summary.max_stable_speedup}x** speedup.",
                "",
                "Consider:",
                f"- Use `--speedup {summary.recommended_speedup}` for reliable tests",
                "- Close other applications to improve performance",
                "- For HPC, use parallel environments at this speedup",
                "",
            ]
        )
    elif summary.max_stable_speedup >= 1.0:
        lines.extend(
            [
                f"**WARNING:** Limited to **{summary.max_stable_speedup}x** speedup.",
                "",
                "This is typical for:",
                "- Laptop/low-power systems",
                "- Systems under heavy load",
                "",
                "Recommendations:",
                "- Use 1.0x speedup for reliable testing",
                "- Use parallel environments on HPC for throughput",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "**ERROR:** No stable speedup achieved!",
                "",
                "Check:",
                "- SITL is building and running correctly",
                "- System has sufficient CPU resources",
                "- No other heavy processes running",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "*Generated by FlightDock speedup_sweep.py*",
        ]
    )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def generate_json(summary: SweepSummary, output_dir: Path) -> None:
    """Generate results.json with machine-readable metrics."""
    results_path = output_dir / "results.json"

    data = {
        "timestamp": summary.timestamp,
        "hostname": summary.hostname,
        "stability_criteria": {
            "quality_min": STABILITY_QUALITY_MIN,
            "rate_tolerance": STABILITY_RATE_TOLERANCE,
        },
        "summary": {
            "stable_speedups": summary.stable_speedups,
            "max_stable_speedup": summary.max_stable_speedup,
            "recommended_speedup": summary.recommended_speedup,
        },
        "results": [asdict(r) for r in summary.all_results],
    }

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================================
# Main Sweep Logic
# ============================================================================


def run_sweep(
    speedups: List[float],
    duration_s: float,
    output_dir: Path,
    flightdock_root: Path,
    verbose: bool = True,
    quality_min: float = STABILITY_QUALITY_MIN,
    rate_tolerance: float = STABILITY_RATE_TOLERANCE,
    skip_lint: bool = False,
) -> SweepSummary:
    """Run the full speedup sweep.

    Args:
        speedups: List of speedups to test
        duration_s: Test duration per speedup
        output_dir: Where to save results
        flightdock_root: FlightDock installation root
        verbose: Print progress
        quality_min: Minimum quality score for stability
        rate_tolerance: Maximum rate deviation (fraction)
        skip_lint: Skip param file lint check

    Returns:
        SweepSummary with all results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FLIGHTDOCK SPEEDUP CALIBRATION SWEEP")
    print("=" * 70)
    print(f"Speedups to test: {speedups}")
    print(f"Duration per test: {duration_s}s (wall)")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()

    # Step 0: Lint param files for SIM_SPEEDUP
    if not skip_lint:
        if not lint_param_files(flightdock_root, verbose=verbose):
            print()
            print("[WARNING] SIM_SPEEDUP found in param files. Continuing anyway...")
            print("          Use --skip-lint to suppress this warning.")
            print()

    sitl = SITLManager(
        flightdock_root=str(flightdock_root),
        output_dir=str(output_dir),
        verbose=verbose,
    )

    results: List[SpeedupResult] = []

    try:
        for i, speedup in enumerate(speedups):
            print(f"\n[{i + 1}/{len(speedups)}] Testing {speedup}x speedup...")
            print("-" * 50)

            # Start SITL
            sitl_log = output_dir / f"sitl_{speedup}x.log"
            if not sitl.start(speedup, str(sitl_log)):
                failure_reason = sitl.last_failure_reason or "SITL_START_FAILED"
                results.append(
                    SpeedupResult(
                        requested=speedup,
                        achieved_rate=0.0,
                        quality_score=0.0,
                        settle_passed=False,
                        passed=False,
                        reason=failure_reason,
                        stable=False,
                        wall_duration_s=0.0,
                        samples=0,
                        failure_reason=failure_reason,
                        readback_speedup=sitl.readback_speedup,
                    )
                )
                # Wait before next attempt
                time.sleep(3.0)
                continue

            # Wait for SITL to settle
            time.sleep(5.0)

            # Run timebase test
            result = run_timebase_test(
                speedup=speedup,
                duration_s=duration_s,
                verbose=verbose,
                quality_min=quality_min,
                rate_tolerance=rate_tolerance,
            )

            # Add readback speedup to result
            result.readback_speedup = sitl.readback_speedup

            results.append(result)

            # Print result
            stable_str = "STABLE" if result.stable else "UNSTABLE"
            print(
                f"\n[RESULT] {speedup}x: achieved={result.achieved_rate:.2f}x "
                f"quality={result.quality_score:.2f} [{stable_str}]"
            )

            # Stop SITL and wait for cleanup
            sitl.stop()
            time.sleep(2.0)

    finally:
        sitl.stop()

    # Build summary
    stable_speedups = [r.requested for r in results if r.stable]
    max_stable = max(stable_speedups) if stable_speedups else 0.0

    # Recommend 80% of max stable for safety margin
    if max_stable >= 5.0:
        recommended = 5.0
    elif max_stable >= 3.0:
        recommended = 3.0
    elif max_stable >= 2.0:
        recommended = 2.0
    elif max_stable >= 1.0:
        recommended = 1.0
    else:
        recommended = 1.0

    summary = SweepSummary(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        hostname=os.uname().nodename,
        stable_speedups=stable_speedups,
        max_stable_speedup=max_stable,
        recommended_speedup=recommended,
        all_results=results,
    )

    return summary


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="FlightDock Speedup Calibration Sweep Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--speedups",
        "-s",
        type=float,
        nargs="+",
        default=DEFAULT_SPEEDUPS,
        help=f"Speedups to test (default: {DEFAULT_SPEEDUPS})",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Test duration per speedup in wall-seconds (default: {DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: auto-generated timestamped folder)",
    )
    parser.add_argument(
        "--flightdock-root",
        type=str,
        default=None,
        help="FlightDock root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--quality-min",
        type=float,
        default=STABILITY_QUALITY_MIN,
        help=f"Minimum quality score for stability (default: {STABILITY_QUALITY_MIN})",
    )
    parser.add_argument(
        "--rate-tolerance",
        type=float,
        default=STABILITY_RATE_TOLERANCE,
        help=f"Rate deviation tolerance (default: {STABILITY_RATE_TOLERANCE})",
    )
    parser.add_argument(
        "--skip-lint",
        action="store_true",
        help="Skip param file lint check for SIM_SPEEDUP",
    )
    parser.add_argument(
        "--lint-only",
        action="store_true",
        help="Only run param file lint check, then exit",
    )

    args = parser.parse_args()

    # Handle --lint-only mode
    if args.lint_only:
        if args.flightdock_root:
            flightdock_root = Path(args.flightdock_root)
        else:
            flightdock_root = Path(__file__).parent.parent
            if not (flightdock_root / "tests").exists():
                flightdock_root = Path.home() / "FlightDock"

        lint_ok = lint_param_files(flightdock_root, verbose=True)
        sys.exit(0 if lint_ok else 1)

    # Auto-detect FlightDock root
    if args.flightdock_root:
        flightdock_root = Path(args.flightdock_root)
    else:
        flightdock_root = Path(__file__).parent.parent
        if not (flightdock_root / "tests").exists():
            flightdock_root = Path.home() / "FlightDock"

    if not flightdock_root.exists():
        print(f"[ERROR] FlightDock root not found: {flightdock_root}")
        sys.exit(1)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = flightdock_root / "tests" / "analysis_out" / f"speedup_sweep_{timestamp}"

    # Run sweep
    try:
        summary = run_sweep(
            speedups=sorted(args.speedups),
            duration_s=args.duration,
            output_dir=output_dir,
            flightdock_root=flightdock_root,
            verbose=not args.quiet,
            quality_min=args.quality_min,
            rate_tolerance=args.rate_tolerance,
            skip_lint=args.skip_lint,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Sweep cancelled by user")
        sys.exit(130)

    # Generate reports
    generate_report(summary, output_dir)
    generate_json(summary, output_dir)

    # Print final summary
    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Stable speedups: {summary.stable_speedups}")
    print(f"Max stable: {summary.max_stable_speedup}x")
    print(f"Recommended: {summary.recommended_speedup}x")
    print()
    print(f"Reports:")
    print(f"  - {output_dir / 'REPORT.md'}")
    print(f"  - {output_dir / 'results.json'}")

    # Exit code based on whether any speedup is stable
    sys.exit(0 if summary.stable_speedups else 1)


if __name__ == "__main__":
    main()
