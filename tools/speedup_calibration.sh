#!/bin/bash
# =============================================================================
# FlightDock Speedup Calibration Tool
# =============================================================================
#
# Runs a SHORT calibration test at specified speedup to validate:
#   1. Timebase quality (Option A hard gating)
#   2. Telemetry freshness (leader broadcast)
#   3. Controller startup stability
#
# This is NOT a full mission test - it runs only long enough to validate
# infrastructure health at the given speedup.
#
# USAGE:
#   tools/speedup_calibration.sh --speedup 5.0
#   tools/speedup_calibration.sh --speedup 8.0 --headless
#
# OUTPUT:
#   - Creates artifact bundle in tests/analysis_out/speedup_calib_<X.X>x_<timestamp>/
#   - Includes timebase results, controller log, leader log, metrics JSON
#   - Exits 0 if PASS, nonzero if FAIL
#
# (C) 2025-2026 FlightDock Research Project
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(dirname "$SCRIPT_DIR")/tests"

# Source the shared harness library
if [[ ! -f "${TESTS_DIR}/lib_run_harness.sh" ]]; then
    echo "ERROR: lib_run_harness.sh not found in ${TESTS_DIR}"
    exit 1
fi
# shellcheck source=../tests/lib_run_harness.sh
source "${TESTS_DIR}/lib_run_harness.sh"

# =============================================================================
# CALIBRATION DEFAULTS
# =============================================================================
HARNESS_TEST_ID="999"                       # Use test ID 999 for calibration
HARNESS_TEST_NAME="speedup_calib"
HARNESS_LEADER_PROFILE="straight"           # Simple straight pattern
HARNESS_SPEEDUP="${HARNESS_SPEEDUP:-5.0}"
HARNESS_SIM_SECONDS="120"                   # SHORT window: 120s sim-time
HARNESS_TIMEBASE_GATE="1"                   # Always run timebase gate
HARNESS_TIMEBASE_DURATION="30"              # 30s timebase test
HARNESS_HEADLESS="${HARNESS_HEADLESS:-1}"   # Default headless
HARNESS_UI_MODE="${HARNESS_UI_MODE:-none}"
HARNESS_SITL_MODE="auto"                    # Auto-detect SITL

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
harness_parse_args "$@"

# Override test name to include speedup
HARNESS_TEST_NAME="speedup_calib_${HARNESS_SPEEDUP}x"

# =============================================================================
# INITIALIZE
# =============================================================================
# Pass a fake calling script path that points to tests directory
# This ensures harness computes all paths relative to tests/ not tools/
harness_init "${TESTS_DIR}/speedup_calibration_fake.sh"

# Print calibration header
echo ""
echo "============================================================="
echo "SPEEDUP CALIBRATION TEST"
echo "============================================================="
echo "  Speedup:           ${HARNESS_SPEEDUP}x"
echo "  Timebase duration: ${HARNESS_TIMEBASE_DURATION}s (wall)"
echo "  Controller window: ${HARNESS_SIM_SECONDS}s (sim)"
echo "  Output:            ${HARNESS_OUT_DIR}"
echo "============================================================="
echo ""

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
harness_preflight_checks

# =============================================================================
# CALIBRATION SEQUENCE
# =============================================================================

# 1. Ensure SITL is running
harness_ensure_sitl

# 2. Run timebase gate (30s wall-time)
#    This validates:
#      - Time sync quality >= 0.95
#      - No monotonicity violations
#      - Rate within 25% of expected
harness_run_timebase_gate

# If timebase gate failed, it already exited (nonzero code)
log_pass "Timebase gate PASSED at ${HARNESS_SPEEDUP}x"

# 3. Start leader broadcaster
harness_start_leader

# 4. Run SHORT controller window (120s sim-time)
#    This validates:
#      - Controller startup
#      - Leader telemetry freshness
#      - No early crashes
#      - Initial state transitions
log_info "Running SHORT controller window (${HARNESS_SIM_SECONDS}s sim-time)..."
harness_run_controller || true

# 5. Quick analysis
harness_analyze_quick

# =============================================================================
# CALIBRATION VERDICT
# =============================================================================
CONTROLLER_LOG="${HARNESS_OUT_DIR}/controller_${HARNESS_SPEEDUP}x.log"
VERDICT_FILE="${HARNESS_OUT_DIR}/calibration_verdict.txt"

echo ""
echo "============================================================="
echo "CALIBRATION VERDICT"
echo "============================================================="

# Check for failures
FAILED=0
FAIL_REASONS=()

# Check timebase verdict
if ! grep -q "SYNC_VERDICT.*PASS" "$CONTROLLER_LOG" 2>/dev/null; then
    FAILED=1
    FAIL_REASONS+=("sync_verdict_not_pass")
fi

# Check for persistent leader stale failsafe (after initial startup)
# Allow up to 5 leader stale events during startup transient
LEADER_STALE_COUNT=$(grep -c "leader_stale_failsafe" "$CONTROLLER_LOG" 2>/dev/null || echo "0")
if [[ "$LEADER_STALE_COUNT" -gt 5 ]]; then
    FAILED=1
    FAIL_REASONS+=("leader_stale_failsafe=${LEADER_STALE_COUNT}")
fi

# Check for controller crash/exception
# NOTE: Match actual Python exceptions, NOT log messages with "CRITICAL" level
# "[LEADER_FRESHNESS] CRITICAL:" is a legitimate log message, not an exception
if grep -qE "Traceback \(most recent call last\)|^[A-Za-z]+Error:|^[A-Za-z]+Exception:" "$CONTROLLER_LOG" 2>/dev/null; then
    FAILED=1
    FAIL_REASONS+=("controller_exception")
fi

# Check controller completed normally
CONTROLLER_EXIT=$(grep "controller_exit=" "${HARNESS_PIDS_FILE}" | cut -d= -f2 || echo "999")
if [[ "$CONTROLLER_EXIT" != "0" ]] && [[ "$CONTROLLER_EXIT" != "124" ]]; then
    # Exit 0 = normal, Exit 124 = timeout (expected for short window), other = error
    FAILED=1
    FAIL_REASONS+=("controller_exit=${CONTROLLER_EXIT}")
fi

# Write verdict
if [[ "$FAILED" == "0" ]]; then
    VERDICT="PASS"
    echo "  Status: ${GREEN}PASS${NC}"
    echo "  Speedup ${HARNESS_SPEEDUP}x is ACCEPTABLE for DRL infrastructure"
    echo "$VERDICT" > "$VERDICT_FILE"
    EXIT_CODE=0
else
    VERDICT="FAIL"
    echo "  Status: ${RED}FAIL${NC}"
    echo "  Reason: ${FAIL_REASONS[*]}"
    echo "  Speedup ${HARNESS_SPEEDUP}x is NOT ACCEPTABLE"
    echo "$VERDICT: ${FAIL_REASONS[*]}" > "$VERDICT_FILE"
    EXIT_CODE=1
fi

# Print key metrics
echo ""
echo "KEY METRICS:"
echo "  Leader stale count:  $LEADER_STALE_COUNT (threshold: <=5)"
echo "  Controller exit:     $CONTROLLER_EXIT (0=normal, 124=timeout/ok)"
echo ""

if grep -q "SYNC_VERDICT" "$CONTROLLER_LOG" 2>/dev/null; then
    echo "SYNC_VERDICT from controller log:"
    grep "SYNC_VERDICT" "$CONTROLLER_LOG" | tail -1
fi

echo "============================================================="
echo ""

# Print output location
echo "Artifact bundle: ${HARNESS_OUT_DIR}"
echo "Verdict: ${VERDICT_FILE}"
echo ""

exit $EXIT_CODE
