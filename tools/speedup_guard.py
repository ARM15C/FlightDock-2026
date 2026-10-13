#!/usr/bin/env python3
"""
FlightDock Speedup Guard - Hardening for SIM_SPEEDUP consistency
=================================================================

Provides:
1. Param file lint: Detect SIM_SPEEDUP lines in param files
2. Param file sanitizer: Generate sanitized param file without SIM_SPEEDUP
3. Runtime readback: Verify actual SIM_SPEEDUP matches requested via MAVLink

USAGE:
    # Lint param file (exit 1 if SIM_SPEEDUP found)
    python tools/speedup_guard.py lint config/quadplane_formation_mav_clean.parm

    # Lint with auto-fix (generate sanitized temp file)
    python tools/speedup_guard.py lint --sanitize config/quadplane.parm

    # Runtime readback check
    python tools/speedup_guard.py readback --conn udp:127.0.0.1:14551 --expected 5.0

EXIT CODES:
    0 = OK (no issues found / readback matches)
    1 = FAIL (SIM_SPEEDUP found / readback mismatch)
    2 = Error (connection failed, etc.)

(C) 2025-2026 FlightDock Research Project
"""

import argparse
import os
import sys
import tempfile
import time
from typing import List, Optional, Tuple

# Add tests directory for pymavlink access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))


# =============================================================================
# Param File Lint
# =============================================================================


def lint_param_file(path: str) -> List[Tuple[int, str]]:
    """Check param file for SIM_SPEEDUP lines.

    Args:
        path: Path to .parm file

    Returns:
        List of (line_number, line_content) tuples for each match
    """
    matches = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.upper().startswith("SIM_SPEEDUP"):
                    matches.append((line_num, stripped))
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Reading {path}: {e}", file=sys.stderr)
    return matches


def lint_multiple_files(paths: List[str]) -> dict:
    """Lint multiple param files.

    Args:
        paths: List of file paths to check

    Returns:
        Dict mapping path -> list of (line_num, content) matches
    """
    results = {}
    for path in paths:
        matches = lint_param_file(path)
        if matches:
            results[path] = matches
    return results


def sanitize_param_file(input_path: str, output_path: Optional[str] = None) -> str:
    """Remove SIM_SPEEDUP lines from param file.

    Args:
        input_path: Source .parm file
        output_path: Destination path (None = auto-generate temp file)

    Returns:
        Path to sanitized file
    """
    if output_path is None:
        # Create temp file in same directory for easy cleanup
        fd, output_path = tempfile.mkstemp(
            prefix="sanitized_",
            suffix=".parm",
            dir=os.path.dirname(input_path) or ".",
        )
        os.close(fd)

    removed_count = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                stripped = line.strip()
                if stripped.upper().startswith("SIM_SPEEDUP"):
                    removed_count += 1
                    # Write as comment to preserve line numbers for debugging
                    fout.write(f"# [SANITIZED] {line}")
                else:
                    fout.write(line)

    print(f"[SANITIZE] Removed {removed_count} SIM_SPEEDUP line(s)")
    print(f"[SANITIZE] Output: {output_path}")
    return output_path


# =============================================================================
# Runtime Readback
# =============================================================================


def readback_sim_speedup(
    conn_str: str,
    timeout_s: float = 10.0,
    retries: int = 3,
) -> Optional[float]:
    """Read SIM_SPEEDUP parameter from SITL via MAVLink.

    Args:
        conn_str: MAVLink connection string (e.g., 'udp:127.0.0.1:14551')
        timeout_s: Timeout in seconds per attempt
        retries: Number of connection attempts

    Returns:
        Current SIM_SPEEDUP value, or None if failed
    """
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[ERROR] pymavlink not installed", file=sys.stderr)
        return None

    for attempt in range(retries):
        if attempt > 0:
            print(f"[READBACK] Retry {attempt + 1}/{retries}...")
            time.sleep(1)

        try:
            mav = mavutil.mavlink_connection(
                conn_str,
                source_system=254,
                source_component=191 + attempt,  # Vary component ID to avoid conflicts
            )
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}", file=sys.stderr)
            continue

        # Wait for heartbeat first
        print(f"[READBACK] Connecting to {conn_str}...")
        try:
            hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout_s)
            if not hb:
                print("[WARN] No heartbeat received, retrying...", file=sys.stderr)
                mav.close()
                continue
        except Exception as e:
            print(f"[ERROR] Heartbeat wait failed: {e}", file=sys.stderr)
            mav.close()
            continue

        print("[READBACK] Heartbeat received, requesting SIM_SPEEDUP...")

        # Request SIM_SPEEDUP parameter multiple times for reliability
        param_name = b"SIM_SPEEDUP"
        for _ in range(3):
            mav.mav.param_request_read_send(
                mav.target_system,
                mav.target_component,
                param_name,
                -1,  # Use name, not index
            )
            time.sleep(0.1)

        # Wait for PARAM_VALUE response
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout_s:
            msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
            if msg is None:
                continue

            # Check if this is SIM_SPEEDUP
            param_id = msg.param_id
            if isinstance(param_id, bytes):
                param_id = param_id.decode("utf-8", errors="replace")
            param_id = param_id.rstrip("\x00")

            if param_id.upper() == "SIM_SPEEDUP":
                value = msg.param_value
                print(f"[READBACK] SIM_SPEEDUP = {value}")
                mav.close()
                return float(value)

        print("[WARN] Timeout waiting for PARAM_VALUE", file=sys.stderr)
        mav.close()

    print("[ERROR] All readback attempts failed", file=sys.stderr)
    return None


def verify_speedup(
    conn_str: str,
    expected_speedup: float,
    tolerance: float = 0.1,
    timeout_s: float = 10.0,
) -> Tuple[bool, str, Optional[float]]:
    """Verify SITL speedup matches expected value.

    Args:
        conn_str: MAVLink connection string
        expected_speedup: Expected SIM_SPEEDUP value
        tolerance: Acceptable deviation (absolute)
        timeout_s: Timeout in seconds

    Returns:
        (passed, reason, actual_value)
    """
    actual = readback_sim_speedup(conn_str, timeout_s)

    if actual is None:
        return False, "Readback failed (connection or timeout)", None

    deviation = abs(actual - expected_speedup)
    if deviation <= tolerance:
        return True, f"Match: {actual} ~= {expected_speedup}", actual

    # Mismatch - provide diagnostic message
    reason = (
        f"MISMATCH: requested={expected_speedup}, actual={actual}\n"
        f"\n"
        f"LIKELY CAUSES:\n"
        f"  1. Parameter file contains SIM_SPEEDUP line that overrides CLI\n"
        f"     Fix: Run 'python speedup_guard.py lint --sanitize <parm_file>'\n"
        f"  2. SITL eeprom.bin cache has stale SIM_SPEEDUP value\n"
        f"     Fix: Delete ~/FlightDock/ardupilot_formation/ArduPlane/eeprom.bin\n"
        f"  3. SITL was restarted but MAVProxy retained old parameter\n"
        f"     Fix: Fully kill SITL and MAVProxy, then restart\n"
    )
    return False, reason, actual


# =============================================================================
# Runtime Set
# =============================================================================


def set_sim_speedup(
    conn_str: str,
    speedup: float,
    timeout_s: float = 10.0,
) -> Tuple[bool, str]:
    """Set SIM_SPEEDUP parameter via MAVLink.

    Args:
        conn_str: MAVLink connection string (e.g., 'udp:127.0.0.1:14551')
        speedup: Desired SIM_SPEEDUP value
        timeout_s: Timeout in seconds

    Returns:
        (success, message)
    """
    try:
        from pymavlink import mavutil
    except ImportError:
        return False, "pymavlink not installed"

    try:
        mav = mavutil.mavlink_connection(
            conn_str,
            source_system=254,
            source_component=190,  # GCS component
        )
    except Exception as e:
        return False, f"Connection failed: {e}"

    # Wait for heartbeat first
    print(f"[SET] Connecting to {conn_str}...")
    try:
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout_s)
        if not hb:
            mav.close()
            return False, "No heartbeat received"
    except Exception as e:
        mav.close()
        return False, f"Heartbeat wait failed: {e}"

    print(f"[SET] Heartbeat received, setting SIM_SPEEDUP={speedup}...")

    # Send PARAM_SET multiple times for reliability
    param_name = b"SIM_SPEEDUP\x00\x00\x00\x00\x00"  # Pad to 16 chars
    for _ in range(3):
        mav.mav.param_set_send(
            mav.target_system,
            mav.target_component,
            param_name,
            float(speedup),
            9,  # MAV_PARAM_TYPE_REAL32
        )
        time.sleep(0.1)

    # Brief wait for parameter to take effect, then close
    # Don't wait for confirmation - readback will verify
    time.sleep(0.5)
    mav.close()
    return True, f"SIM_SPEEDUP set command sent (value={speedup})"


# =============================================================================
# CLI
# =============================================================================


def cmd_lint(args: argparse.Namespace) -> int:
    """Handle 'lint' subcommand."""
    results = lint_multiple_files(args.files)

    if not results:
        print("[LINT] PASS: No SIM_SPEEDUP found in any file")
        return 0

    # SIM_SPEEDUP found
    print("[LINT] FAIL: SIM_SPEEDUP found in parameter file(s):")
    print()
    for filepath, matches in results.items():
        print(f"  {filepath}:")
        for line_num, content in matches:
            print(f"    Line {line_num}: {content}")
    print()

    if args.sanitize:
        print("[LINT] --sanitize requested, generating sanitized files...")
        sanitized_files = []
        for filepath in results.keys():
            sanitized = sanitize_param_file(filepath)
            sanitized_files.append((filepath, sanitized))
            print(f"  {filepath} -> {sanitized}")

        # Print usage hint
        print()
        print("To use sanitized file(s), set environment variable:")
        for original, sanitized in sanitized_files:
            print(f"  PARAM_FILE_OVERRIDE={sanitized}")
        return 0  # Success after sanitization

    # No sanitize, just report and fail
    print("To fix, either:")
    print("  1. Remove SIM_SPEEDUP lines from parameter file(s)")
    print("  2. Run with --sanitize to auto-generate clean files")
    print("  3. Set HARNESS_SKIP_PARAM_LINT=1 to bypass (not recommended)")
    return 1


def cmd_readback(args: argparse.Namespace) -> int:
    """Handle 'readback' subcommand."""
    passed, reason, actual = verify_speedup(
        conn_str=args.conn,
        expected_speedup=args.expected,
        tolerance=args.tolerance,
        timeout_s=args.timeout,
    )

    if passed:
        print(f"[READBACK] PASS: {reason}")
        return 0

    print(f"[READBACK] FAIL: {reason}")
    return 1


def cmd_set(args: argparse.Namespace) -> int:
    """Handle 'set' subcommand."""
    success, message = set_sim_speedup(
        conn_str=args.conn,
        speedup=args.speedup,
        timeout_s=args.timeout,
    )

    if success:
        print(f"[SET] PASS: {message}")
        return 0

    print(f"[SET] FAIL: {message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FlightDock Speedup Guard - Hardening for SIM_SPEEDUP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # lint subcommand
    lint_parser = subparsers.add_parser(
        "lint",
        help="Check param file(s) for SIM_SPEEDUP lines",
    )
    lint_parser.add_argument(
        "files",
        nargs="+",
        help="Param file(s) to check",
    )
    lint_parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Generate sanitized temp file without SIM_SPEEDUP",
    )

    # readback subcommand
    readback_parser = subparsers.add_parser(
        "readback",
        help="Verify SITL speedup via MAVLink readback",
    )
    readback_parser.add_argument(
        "--conn",
        "-c",
        required=True,
        help="MAVLink connection string (e.g., udp:127.0.0.1:14551)",
    )
    readback_parser.add_argument(
        "--expected",
        "-e",
        type=float,
        required=True,
        help="Expected SIM_SPEEDUP value",
    )
    readback_parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=0.1,
        help="Acceptable deviation (default: 0.1)",
    )
    readback_parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds (default: 10)",
    )

    # set subcommand
    set_parser = subparsers.add_parser(
        "set",
        help="Set SIM_SPEEDUP via MAVLink",
    )
    set_parser.add_argument(
        "--conn",
        "-c",
        required=True,
        help="MAVLink connection string (e.g., udp:127.0.0.1:14551)",
    )
    set_parser.add_argument(
        "--speedup",
        "-s",
        type=float,
        required=True,
        help="Desired SIM_SPEEDUP value",
    )
    set_parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds (default: 10)",
    )

    args = parser.parse_args()

    if args.command == "lint":
        return cmd_lint(args)
    elif args.command == "readback":
        return cmd_readback(args)
    elif args.command == "set":
        return cmd_set(args)

    return 2


if __name__ == "__main__":
    sys.exit(main())
