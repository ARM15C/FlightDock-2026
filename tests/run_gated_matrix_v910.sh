#!/bin/bash
# =============================================================================
# FlightDock v9.10 Gated Test Matrix Runner
# =============================================================================
#
# Runs formation controller tests at multiple speedups and validates each
# with the v9.10 gate checker. Uses lib_run_harness.sh for proper SITL
# lifecycle management, timebase gating, and leader broadcaster.
#
# Usage:
#   ./tools/run_gated_matrix_v910.sh [--duration SECONDS] [--speedups "1x 3x 5x"]
#
# Examples:
#   ./tools/run_gated_matrix_v910.sh --duration 600 --speedups "1x 3x 5x"
#   ./tools/run_gated_matrix_v910.sh --duration 300 --speedups "1x"
#   ./tools/run_gated_matrix_v910.sh --ui map --speedups "1x" # With GUI
#
# Gate Checker:
#   After each run, invokes flightdock_gatecheck_v910.py to validate:
#   - Gate 0: Timebase integrity (dt_tick_raw_s, no regressions)
#   - Gate 1: Safety invariant (no no-pass violations)
#   - Gate 2: State machine stability (TRANSITION_SNAPSHOT.log, debouncing)
#
# =============================================================================

set -euo pipefail

# =============================================================================
# MATRIX CONFIGURATION (before sourcing harness)
# =============================================================================
MATRIX_DURATION_S="${MATRIX_DURATION_S:-600}"
MATRIX_SPEEDUPS="${MATRIX_SPEEDUPS:-1x 3x 5x}"
MATRIX_TEST_ID="${MATRIX_TEST_ID:-v910}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$PROJECT_ROOT/tests"
TOOLS_DIR="$PROJECT_ROOT/tools"
GATE_CHECKER="$TOOLS_DIR/flightdock_gatecheck_v910.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# ARGUMENT PARSING (matrix-specific, before harness)
# =============================================================================
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration|--sim-seconds)
            MATRIX_DURATION_S="$2"
            shift 2
            ;;
        --speedups)
            MATRIX_SPEEDUPS="$2"
            shift 2
            ;;
        --test-id)
            MATRIX_TEST_ID="$2"
            shift 2
            ;;
        *)
            # Pass through to harness
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# =============================================================================
# BANNER
# =============================================================================
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}FlightDock v9.10 Gated Test Matrix${NC}"
echo -e "${CYAN}========================================${NC}"
echo "Duration:  ${MATRIX_DURATION_S}s sim-time per speedup"
echo "Speedups:  ${MATRIX_SPEEDUPS}"
echo "Test ID:   ${MATRIX_TEST_ID}"
echo "Gates:     0 (Timebase), 1 (Safety), 2 (StateMachine)"
echo ""

# Validate gate checker exists
if [[ ! -f "$GATE_CHECKER" ]]; then
    echo -e "${RED}[FATAL] Gate checker not found: $GATE_CHECKER${NC}"
    exit 1
fi

# =============================================================================
# RESULTS TRACKING
# =============================================================================
declare -A MATRIX_RESULTS
declare -A MATRIX_OUT_DIRS
MATRIX_START_TIME=$(date +%s)

# =============================================================================
# RUN SINGLE SPEEDUP TEST
# =============================================================================
run_single_speedup() {
    local speedup_tag="$1"  # e.g., "1x", "3x", "5x"
    local speedup_num="${speedup_tag%x}"  # e.g., "1", "3", "5"

    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}SPEEDUP: ${speedup_tag}${NC}"
    echo -e "${CYAN}========================================${NC}"

    # Reset harness state for fresh run
    unset HARNESS_INITIALIZED
    unset HARNESS_OUT_DIR
    unset HARNESS_SITL_OWNED
    unset HARNESS_LEADER_PID
    unset HARNESS_CLEANUP_DONE

    # Configure harness for this speedup
    export HARNESS_SPEEDUP="$speedup_num"
    export HARNESS_SIM_SECONDS="$MATRIX_DURATION_S"
    export HARNESS_TEST_ID="$MATRIX_TEST_ID"
    export HARNESS_TEST_NAME="${HARNESS_TEST_NAME:-test}"
    export HARNESS_TIMEBASE_GATE="1"
    export HARNESS_CPP_HANDOFF_POLICY="${HARNESS_CPP_HANDOFF_POLICY:-state_gated}"

    # Source harness library (in subshell-like context via function)
    # We need to source it fresh for each run
    source "${TESTS_DIR}/lib_run_harness.sh"

    # Parse any passthrough args
    if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
        harness_parse_args "${PASSTHROUGH_ARGS[@]}"
    fi

    # Initialize harness (creates output directory)
    harness_init "${BASH_SOURCE[0]}"

    # Store output directory for gate checking
    local out_dir="$HARNESS_OUT_DIR"
    MATRIX_OUT_DIRS[$speedup_tag]="$out_dir"

    echo ""
    echo "[MATRIX] Output: $out_dir"
    echo ""

    # Run the full test flow
    local run_ok=0

    # Disable exit-on-error for test execution (we want to continue matrix on failure)
    set +e

    # 1. Ensure SITL is running at correct speedup
    harness_ensure_sitl
    local sitl_exit=$?

    if [[ $sitl_exit -ne 0 ]]; then
        echo -e "${RED}[${speedup_tag}] SITL failed to start${NC}"
        MATRIX_RESULTS[$speedup_tag]="FAIL_SITL"
        set -e
        return 1
    fi

    # 1.5 Configure + validate ATTITUDE rate (PRH-ATT-002)
    harness_configure_attitude_rate
    local att_exit=$?

    if [[ $att_exit -ne 0 ]]; then
        echo -e "${RED}[${speedup_tag}] ATTITUDE rate preflight failed${NC}"
        MATRIX_RESULTS[$speedup_tag]="FAIL_ATTITUDE_RATE"
        set -e
        return 1
    fi

    # 2. Run timebase gate (validates speedup is actually working)
    harness_run_timebase_gate
    local timebase_exit=$?

    if [[ $timebase_exit -ne 0 ]]; then
        echo -e "${RED}[${speedup_tag}] Timebase gate failed${NC}"
        MATRIX_RESULTS[$speedup_tag]="FAIL_TIMEBASE"
        set -e
        return 1
    fi

    # 3. Start leader broadcaster
    harness_start_leader
    local leader_exit=$?

    if [[ $leader_exit -ne 0 ]]; then
        echo -e "${RED}[${speedup_tag}] Leader broadcaster failed${NC}"
        MATRIX_RESULTS[$speedup_tag]="FAIL_LEADER"
        set -e
        return 1
    fi

    # 4. Run controller
    harness_run_controller
    local ctrl_exit=$?

    # 5. Quick analysis (non-fatal - don't abort if analyze fails)
    harness_analyze_quick || true

    # R2: Validate critical artifacts exist before gate checking
    if [[ ! -f "${out_dir}/run_summary.json" ]]; then
        echo -e "${RED}[${speedup_tag}] MISSING: run_summary.json - artifacts not preserved${NC}"
        MATRIX_RESULTS[$speedup_tag]="FAIL_MISSING_ARTIFACTS"
        set -e
        return 1
    fi

    # 6. Run gate checker (captures all failure modes including fatal conditions)
    echo ""
    echo -e "${CYAN}[GATECHECK] Validating ${speedup_tag} run...${NC}"

    local gate_report="${out_dir}/gate_report_v9.10.md"

    # R1: Keep set +e through gate checker - defer set -e until after gate_exit captured
    python3 "$GATE_CHECKER" \
        --run-dir "$out_dir" \
        --speedup "$speedup_tag" \
        --output "$gate_report"
    local gate_exit=$?

    # R1: Re-enable exit-on-error AFTER gate checker exit code captured
    set -e

    # Determine result
    if [[ $gate_exit -eq 0 ]]; then
        echo -e "${GREEN}[${speedup_tag}] GATES PASS${NC}"
        MATRIX_RESULTS[$speedup_tag]="PASS"
        run_ok=1
    else
        echo -e "${RED}[${speedup_tag}] GATES FAIL${NC}"
        MATRIX_RESULTS[$speedup_tag]="FAIL_GATES"

        # Show gate report summary
        if [[ -f "$gate_report" ]]; then
            echo ""
            echo "Gate Report Summary:"
            grep -E "^- Gate|FAIL|PASS" "$gate_report" | head -10 || true
        fi
    fi

    # Cleanup for this run (kill leader, optionally SITL)
    if [[ -n "${HARNESS_LEADER_PID:-}" ]]; then
        kill -INT "$HARNESS_LEADER_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$HARNESS_LEADER_PID" 2>/dev/null || true
    fi

    # Kill SITL between runs to ensure clean speedup for next iteration
    harness_kill_sitl

    return $((1 - run_ok))
}

# =============================================================================
# MAIN MATRIX LOOP
# =============================================================================
echo ""
echo "Starting matrix runs..."
echo ""

for SPD in $MATRIX_SPEEDUPS; do
    run_single_speedup "$SPD" || true

    # Brief pause between runs
    sleep 2
done

# =============================================================================
# MATRIX SUMMARY
# =============================================================================
MATRIX_END_TIME=$(date +%s)
MATRIX_ELAPSED=$((MATRIX_END_TIME - MATRIX_START_TIME))

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}MATRIX RESULTS${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

ALL_PASS=true
PASS_COUNT=0
FAIL_COUNT=0

for SPD in $MATRIX_SPEEDUPS; do
    STATUS="${MATRIX_RESULTS[$SPD]:-NOT_RUN}"
    OUT_DIR="${MATRIX_OUT_DIRS[$SPD]:-N/A}"

    if [[ "$STATUS" == "PASS" ]]; then
        echo -e "${GREEN}${SPD}: PASS${NC}"
        echo "    Output: $OUT_DIR"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${RED}${SPD}: ${STATUS}${NC}"
        echo "    Output: $OUT_DIR"
        ALL_PASS=false
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "Duration: ${MATRIX_ELAPSED}s wall-time"
echo "Passed:   ${PASS_COUNT}"
echo "Failed:   ${FAIL_COUNT}"
echo ""

## Rec 5: Cross-speedup invariance comparison (only when 2+ speedups passed)
INVARIANCE_TOOL="$TOOLS_DIR/compare_speedup_invariance.py"
if [[ $PASS_COUNT -ge 2 && -f "$INVARIANCE_TOOL" ]]; then
    echo ""
    echo -e "${CYAN}--- Cross-Speedup Invariance Check ---${NC}"
    INVARIANCE_ARGS=()
    for SPD in $MATRIX_SPEEDUPS; do
        STATUS="${MATRIX_RESULTS[$SPD]:-NOT_RUN}"
        OUT_DIR="${MATRIX_OUT_DIRS[$SPD]:-}"
        if [[ "$STATUS" == "PASS" && -n "$OUT_DIR" ]]; then
            INVARIANCE_ARGS+=("$OUT_DIR")
        fi
    done
    if [[ ${#INVARIANCE_ARGS[@]} -ge 2 ]]; then
        INVARIANCE_REPORT="${INVARIANCE_ARGS[0]}/invariance_report.json"
        python3 "$INVARIANCE_TOOL" --run-dirs "${INVARIANCE_ARGS[@]}" \
            --output "$INVARIANCE_REPORT" 2>&1 || true
    fi
fi

if $ALL_PASS; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}OVERALL: PASS - All gates passed${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "v9.10 Gates 0-2 validated at all speedups."
    echo "Ready to proceed with DoD documentation."
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}OVERALL: FAIL - One or more gates failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Review gate reports for failure details:"
    for SPD in $MATRIX_SPEEDUPS; do
        OUT_DIR="${MATRIX_OUT_DIRS[$SPD]:-}"
        if [[ -n "$OUT_DIR" && -f "${OUT_DIR}/gate_report_v9.10.md" ]]; then
            echo "  cat ${OUT_DIR}/gate_report_v9.10.md"
        fi
    done
    exit 2
fi
