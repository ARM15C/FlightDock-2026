#!/usr/bin/env python3
"""
FlightDock Config Snapshot Helper
=================================

Generates valid JSON config snapshot by extracting values from Python modules.
Called by lib_run_harness.sh to avoid fragile shell grep/sed.

Usage:
    python3 config_snapshot_helper.py write \\
        --test-id 338 \\
        --test-name test_338 \\
        --speedup 5.0 \\
        --sim-seconds 1800 \\
        --headless 1 \\
        --ui-mode none \\
        --leader-profile straight \\
        --timebase-gate 1 \\
        --drl-mode 0 \\
        --timebase-duration 30 \\
        --min-achieved-speedup-frac 0.8 \\
        --env-id 0 \\
        --port-offset 0 \\
        --conn udp:127.0.0.1:14551 \\
        --timesync-conn udpin:127.0.0.1:14553 \\
        --git-commit abc123 \\
        --harness-version 1.0.0 \\
        --output /path/to/config_snapshot.json

    # Validate existing JSON
    python3 config_snapshot_helper.py validate /path/to/config_snapshot.json

(C) 2025-2026 FlightDock Research Project
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional


def get_git_state(repo_dir: Optional[str] = None) -> Dict:
    """Capture git working tree state for traceability.

    Returns a dict with:
    - commit_full: Full commit hash
    - commit_short: Short (7-char) commit hash
    - working_tree_clean: True if no modified/untracked files
    - modified_files: List of modified/staged files
    - untracked_count: Number of untracked files
    - diff_stat: Summary of changes (e.g., "2 files changed, 15 insertions(+)")

    TEST 386: Enables permanent record of working tree state at test time.
    """
    result = {
        "commit_full": "unknown",
        "commit_short": "unknown",
        "working_tree_clean": True,
        "modified_files": [],
        "untracked_count": 0,
        "diff_stat": "",
    }

    cwd = repo_dir or os.getcwd()

    def run_git(cmd: List[str]) -> Optional[str]:
        """Run git command and return stdout, or None on error."""
        try:
            proc = subprocess.run(
                ["git"] + cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass
        return None

    # Get commit hashes
    commit_full = run_git(["rev-parse", "HEAD"])
    if commit_full:
        result["commit_full"] = commit_full
        result["commit_short"] = commit_full[:7]

    # Get porcelain status (modified and untracked files)
    status_output = run_git(["status", "--porcelain"])
    if status_output is not None:
        lines = status_output.split("\n") if status_output else []
        modified_files = []
        untracked_count = 0

        for line in lines:
            if not line:
                continue
            status_code = line[:2]
            file_path = line[3:]

            if status_code.startswith("?"):
                # Untracked file
                untracked_count += 1
            else:
                # Modified, staged, etc.
                modified_files.append(file_path)

        result["modified_files"] = modified_files
        result["untracked_count"] = untracked_count
        result["working_tree_clean"] = len(modified_files) == 0 and untracked_count == 0

    # Get diff stat (summary of changes, not full diff)
    diff_stat = run_git(["diff", "--stat", "--stat-width=80"])
    if diff_stat:
        # Extract just the summary line (last line with "X files changed, ...")
        lines = diff_stat.strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            if "changed" in last_line or "insertion" in last_line or "deletion" in last_line:
                result["diff_stat"] = last_line
            else:
                # Full stat output if no summary line
                result["diff_stat"] = diff_stat[:200]  # Truncate if too long

    return result


def get_controller_version(tests_dir: str) -> str:
    """Extract CONTROLLER_VERSION from baseline_formation_controller.py."""
    controller_path = os.path.join(tests_dir, "baseline_formation_controller.py")
    if not os.path.isfile(controller_path):
        return "unknown"

    try:
        with open(controller_path, "r") as f:
            for line in f:
                if "CONTROLLER_VERSION" in line and "=" in line:
                    # Extract string between quotes
                    if '"' in line:
                        parts = line.split('"')
                        if len(parts) >= 2:
                            return parts[1]
                    elif "'" in line:
                        parts = line.split("'")
                        if len(parts) >= 2:
                            return parts[1]
    except OSError:
        pass
    return "unknown"


def get_closure_taper_table(tests_dir: str) -> dict:
    """Compute closure taper table by calling compute_tapered_max_closure_alongtrack().

    Returns a dict with:
    - taper_values: dict mapping behind_err (m) to max_closure (m/s)
    - constants: dict of key taper parameters
    - test_341_enabled: bool indicating TEST341 instrumentation is present

    TEST 341: This MUST call the actual function to ensure config reflects runtime behavior.
    """
    taper_values = {}
    constants = {}
    test_341_enabled = False

    controller_path = os.path.join(tests_dir, "baseline_formation_controller.py")
    if not os.path.isfile(controller_path):
        return {"taper_values": {}, "constants": {}, "test_341_enabled": False}

    try:
        # Add tests_dir to path to import the controller module
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)

        # Import directly as a module (avoid caching issues)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "baseline_formation_controller", controller_path
        )
        controller_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller_module)

        # Get the taper function
        compute_taper = getattr(controller_module, "compute_tapered_max_closure_alongtrack", None)
        if compute_taper is None:
            return {"taper_values": {}, "constants": {}, "test_341_enabled": False}

        # Compute taper values at key points
        test_points = [5.0, 10.0, 20.0, 22.8, 35.0, 50.0, 100.0, 150.0]
        for behind_err in test_points:
            try:
                max_closure = compute_taper(behind_err)
                taper_values[str(behind_err)] = round(max_closure, 4)
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        # Extract constants
        constant_names = [
            "CLOSURE_TAPER_NEAR_RANGE_M",
            "CLOSURE_TAPER_FAR_RANGE_M",
            "MAX_CLOSURE_RATE_FAR_MS",
            "MAX_CLOSURE_RATE_NEAR_MS",
            "CLOSURE_TAPER_VERY_NEAR_ERR_M",
            "CLOSURE_TAPER_ULTRA_NEAR_ERR_M",
            "MAX_CLOSURE_RATE_VERY_NEAR_MS",
            "MAX_CLOSURE_RATE_ULTRA_NEAR_MS",
        ]
        for name in constant_names:
            val = getattr(controller_module, name, None)
            if val is not None:
                constants[name] = val

        # Check for TEST 341 instrumentation
        test_341_enabled = hasattr(controller_module, "get_taper_zone")

    except (ImportError, AttributeError, OSError, ValueError, TypeError) as e:
        # Silently fail - will return empty dict
        print(f"[CONFIG_SNAPSHOT] Warning: Could not compute taper table: {e}", file=sys.stderr)

    return {
        "taper_values": taper_values,
        "constants": constants,
        "test_341_enabled": test_341_enabled,
    }


def get_gating_thresholds(tests_dir: str) -> dict:
    """Extract gating thresholds from sim_time_authority.py.

    Returns a dict with numeric values (never empty or null).
    """
    # Defaults in case extraction fails
    quality_gate_min = 0.95
    monotonicity_gate_max = 0

    sim_time_path = os.path.join(tests_dir, "sim_time_authority.py")
    if os.path.isfile(sim_time_path):
        try:
            with open(sim_time_path, "r") as f:
                content = f.read()

            # Parse QUALITY_GATE_MIN
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("QUALITY_GATE_MIN") and "=" in line:
                    # Extract numeric value
                    parts = line.split("=")
                    if len(parts) >= 2:
                        val_part = parts[1].split("#")[0].strip()
                        try:
                            quality_gate_min = float(val_part)
                        except ValueError:
                            pass
                    break

            # Parse MONOTONICITY_GATE_MAX
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("MONOTONICITY_GATE_MAX") and "=" in line:
                    parts = line.split("=")
                    if len(parts) >= 2:
                        val_part = parts[1].split("#")[0].strip()
                        try:
                            monotonicity_gate_max = int(val_part)
                        except ValueError:
                            pass
                    break
        except OSError:
            pass

    return {
        "quality_gate_min": quality_gate_min,
        "monotonicity_gate_max": monotonicity_gate_max,
    }


def write_config_snapshot(args) -> int:
    """Write config_snapshot.json with validated values."""
    import time

    # Determine tests directory from output path
    out_dir = os.path.dirname(args.output)
    # Walk up to find tests/ directory
    tests_dir = out_dir
    for _ in range(5):  # Max 5 levels up
        parent = os.path.dirname(tests_dir)
        if os.path.basename(tests_dir) == "analysis_out":
            tests_dir = os.path.dirname(tests_dir)  # tests/
            break
        tests_dir = parent

    # Get controller version
    controller_version = get_controller_version(tests_dir)
    if hasattr(args, "controller_version") and args.controller_version:
        controller_version = args.controller_version

    # Get gating thresholds from Python source
    thresholds = get_gating_thresholds(tests_dir)

    # Get closure taper table (TEST 341+)
    taper_table = get_closure_taper_table(tests_dir)

    # TEST 386: Capture git state for traceability
    git_state = None
    if getattr(args, "capture_git_state", 1):
        # Use tests_dir parent (should be repo root)
        repo_dir = os.path.dirname(tests_dir) if tests_dir else None
        git_state = get_git_state(repo_dir)

    # Build config dict
    config = {
        "test_id": str(args.test_id),
        "test_name": args.test_name,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "speedup": float(args.speedup),
        "sim_seconds": int(args.sim_seconds),
        "headless": int(args.headless),
        "ui_mode": args.ui_mode,
        "leader_profile": args.leader_profile,
        "controller_version": controller_version,
        "git_commit": args.git_commit or "unknown",
        "harness_version": args.harness_version or "1.0.0",
        "gating_thresholds": {
            "quality_gate_min": thresholds["quality_gate_min"],
            "monotonicity_gate_max": thresholds["monotonicity_gate_max"],
            "rate_tolerance_pct": 0.25,
            "timebase_duration_s": int(args.timebase_duration),
            "min_achieved_speedup_frac": float(args.min_achieved_speedup_frac),
        },
        "env": {
            "env_id": int(args.env_id),
            "port_offset": int(args.port_offset),
            "conn": args.conn,
            "timesync_conn": args.timesync_conn,
        },
        "flags": {
            "timebase_gate": int(args.timebase_gate),
            "drl_mode": int(args.drl_mode),
        },
        "closure_taper": {
            "taper_values_ms": taper_table.get("taper_values", {}),
            "constants": taper_table.get("constants", {}),
            "test_341_instrumentation": taper_table.get("test_341_enabled", False),
        },
        # TEST 387 Matrix B: C++ handoff policy
        "cpp_handoff": {
            "policy": getattr(args, "cpp_handoff_policy", "always_on"),
            "enable_threshold_m": (
                30.0 if getattr(args, "cpp_handoff_policy", "") == "range_hysteresis" else None
            ),
            "disable_threshold_m": (
                40.0 if getattr(args, "cpp_handoff_policy", "") == "range_hysteresis" else None
            ),
        },
    }

    # TEST 386: Add git_state if captured
    if git_state is not None:
        config["git_state"] = git_state

    # Ensure output directory exists
    from common_io import ensure_output_dir

    ensure_output_dir(args.output)

    # Write atomically (temp file + rename)
    temp_path = args.output + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(temp_path, args.output)
        print(f"[CONFIG_SNAPSHOT] Written: {args.output}")

        # Self-check: verify we wrote valid JSON
        with open(args.output, "r") as f:
            json.load(f)  # Will raise on invalid JSON
        return 0

    except (json.JSONDecodeError, OSError) as e:
        print(f"[CONFIG_SNAPSHOT] ERROR: {e}", file=sys.stderr)
        return 1


def validate_config_snapshot(path: str) -> int:
    """Validate an existing config_snapshot.json file."""
    if not os.path.isfile(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 1

    try:
        with open(path, "r") as f:
            config = json.load(f)

        # Check required fields
        required = ["test_id", "speedup", "gating_thresholds"]
        missing = [k for k in required if k not in config]
        if missing:
            print(f"ERROR: Missing fields: {missing}", file=sys.stderr)
            return 1

        # Check gating_thresholds has numeric values
        gt = config.get("gating_thresholds", {})
        if not isinstance(gt.get("quality_gate_min"), (int, float)):
            print(
                f"ERROR: quality_gate_min is not numeric: {gt.get('quality_gate_min')}",
                file=sys.stderr,
            )
            return 1
        if not isinstance(gt.get("monotonicity_gate_max"), (int, float)):
            print(
                f"ERROR: monotonicity_gate_max is not numeric: {gt.get('monotonicity_gate_max')}",
                file=sys.stderr,
            )
            return 1

        print(f"OK: {path}")
        print(f"  quality_gate_min: {gt['quality_gate_min']}")
        print(f"  monotonicity_gate_max: {gt['monotonicity_gate_max']}")
        return 0

    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Config snapshot helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'write' subcommand
    write_parser = subparsers.add_parser("write", help="Write config snapshot JSON")
    write_parser.add_argument("--test-id", required=True, help="Test ID")
    write_parser.add_argument("--test-name", required=True, help="Test name")
    write_parser.add_argument("--speedup", type=float, required=True, help="Sim speedup")
    write_parser.add_argument("--sim-seconds", type=int, required=True, help="Sim seconds budget")
    write_parser.add_argument("--headless", type=int, default=1, help="Headless mode")
    write_parser.add_argument("--ui-mode", default="none", help="UI mode")
    write_parser.add_argument("--leader-profile", default="straight", help="Leader profile")
    write_parser.add_argument("--timebase-gate", type=int, default=1, help="Timebase gate")
    write_parser.add_argument("--drl-mode", type=int, default=0, help="DRL mode")
    write_parser.add_argument(
        "--timebase-duration", type=int, default=30, help="Timebase duration"
    )
    write_parser.add_argument(
        "--min-achieved-speedup-frac",
        type=float,
        default=0.8,
        help="Min achieved speedup fraction",
    )
    write_parser.add_argument("--env-id", type=int, default=0, help="Environment ID")
    write_parser.add_argument("--port-offset", type=int, default=0, help="Port offset")
    write_parser.add_argument("--conn", default="udp:127.0.0.1:14551", help="Connection string")
    write_parser.add_argument(
        "--timesync-conn", default="udpin:127.0.0.1:14553", help="Timesync connection"
    )
    write_parser.add_argument("--git-commit", default="unknown", help="Git commit hash")
    write_parser.add_argument("--harness-version", default="1.0.0", help="Harness version")
    write_parser.add_argument(
        "--controller-version", default="", help="Override controller version"
    )
    write_parser.add_argument("--output", required=True, help="Output JSON path")
    write_parser.add_argument(
        "--capture-git-state",
        type=int,
        default=1,
        help="Capture git state (0=disable, 1=enable, default=1)",
    )
    # TEST 387 Matrix B: Add cpp-handoff-policy
    write_parser.add_argument(
        "--cpp-handoff-policy",
        default="always_on",
        help="C++ handoff policy (always_on, always_off, range_hysteresis, state_gated)",
    )

    # 'validate' subcommand
    validate_parser = subparsers.add_parser("validate", help="Validate config snapshot JSON")
    validate_parser.add_argument("path", help="Path to config_snapshot.json")

    args = parser.parse_args()

    if args.command == "write":
        return write_config_snapshot(args)
    elif args.command == "validate":
        return validate_config_snapshot(args.path)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
