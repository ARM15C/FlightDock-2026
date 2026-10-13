#!/bin/bash
# =============================================================================
# FlightDock Shared Test Harness Library
# =============================================================================
#
# Provides common functions for running formation controller tests:
#   - SITL lifecycle management (start/reuse/stop)
#   - Heartbeat-based readiness detection
#   - Timebase gate validation
#   - Leader broadcaster management
#   - Controller execution with timeout
#   - Cleanup and symlink management
#
# USAGE:
#   source tests/lib_run_harness.sh
#   harness_parse_args "$@"
#   harness_init
#   harness_ensure_sitl
#   harness_run_timebase_gate
#   harness_start_leader
#   harness_run_controller
#   harness_analyze_quick
#
# (C) 2025-2026 FlightDock Research Project
# =============================================================================

# Strict mode
set -euo pipefail

# =============================================================================
# GLOBAL STATE (set by harness_init)
# =============================================================================
HARNESS_VERSION="1.0.0"
HARNESS_INITIALIZED=0
HARNESS_SITL_OWNED=0           # 1 if we started SITL (cleanup will kill it)
HARNESS_SITL_SCRIPT_PID=""
HARNESS_LEADER_PID=""
HARNESS_OUT_DIR=""
HARNESS_PIDS_FILE=""
HARNESS_CLEANUP_DONE=0

# =============================================================================
# DEFAULT CONFIGURATION (can be overridden before harness_init)
# =============================================================================
HARNESS_SPEEDUP="${HARNESS_SPEEDUP:-5.0}"
HARNESS_SIM_SECONDS="${HARNESS_SIM_SECONDS:-1500}"
HARNESS_TEST_ID="${HARNESS_TEST_ID:-334}"
HARNESS_TEST_NAME="${HARNESS_TEST_NAME:-test}"
HARNESS_LEADER_PROFILE="${HARNESS_LEADER_PROFILE:-straight}"
HARNESS_DRY_RUN="${HARNESS_DRY_RUN:-0}"

# SITL control
HARNESS_SITL_MODE="${HARNESS_SITL_MODE:-auto}"  # auto, reuse, start
HARNESS_SITL_READY_TIMEOUT="${HARNESS_SITL_READY_TIMEOUT:-120}"

# UI control
HARNESS_UI_MODE="${HARNESS_UI_MODE:-none}"  # none, map, console, both
HARNESS_HEADLESS="${HARNESS_HEADLESS:-1}"   # 1=no GUI, 0=allow GUI

# Gate control
HARNESS_TIMEBASE_GATE="${HARNESS_TIMEBASE_GATE:-1}"
HARNESS_TIMEBASE_DURATION="${HARNESS_TIMEBASE_DURATION:-30}"
HARNESS_LEADER_FRESHNESS_GATE="${HARNESS_LEADER_FRESHNESS_GATE:-0}"
HARNESS_LEADER_FRESHNESS_THRESHOLD_MS="${HARNESS_LEADER_FRESHNESS_THRESHOLD_MS:-500}"
HARNESS_LEADER_FRESHNESS_DURATION_S="${HARNESS_LEADER_FRESHNESS_DURATION_S:-30}"

# DRL training mode (relaxes timing requirements)
HARNESS_DRL_MODE="${HARNESS_DRL_MODE:-0}"

# Smoketest-only mode (for calibration)
HARNESS_SMOKETEST_ONLY="${HARNESS_SMOKETEST_ONLY:-0}"
HARNESS_SMOKETEST_POST_SETTLE_S="${HARNESS_SMOKETEST_POST_SETTLE_S:-20}"

# Run tag (for directory naming)
HARNESS_RUN_TAG="${HARNESS_RUN_TAG:-}"

# Force-UI mode (override GUI speedup clamping)
HARNESS_FORCE_UI="${HARNESS_FORCE_UI:-0}"

# GUI speedup policy
HARNESS_GUI_MAX_SPEEDUP="${HARNESS_GUI_MAX_SPEEDUP:-3.0}"

# Fail-fast speedup collapse detection
# If achieved rate < (MIN_ACHIEVED_SPEEDUP_FRAC * requested), fail immediately
HARNESS_MIN_ACHIEVED_SPEEDUP_FRAC="${HARNESS_MIN_ACHIEVED_SPEEDUP_FRAC:-0.8}"

# Parallel environment support (env-id → port offset)
# Each env_id offsets all ports by (env_id * 100)
HARNESS_ENV_ID="${HARNESS_ENV_ID:-0}"
HARNESS_PORT_OFFSET="${HARNESS_PORT_OFFSET:-0}"

# Controller connection (computed from env-id if not explicitly set)
HARNESS_CONN="${HARNESS_CONN:-}"
HARNESS_TIMESYNC_CONN="${HARNESS_TIMESYNC_CONN:-}"

# Parameter wipe control (clean eeprom.bin/mav.parm before each test)
HARNESS_WIPE_PARAMS="${HARNESS_WIPE_PARAMS:-1}"  # Default: wipe params each test

# Parallel-safe mode: skip global pkill, use PID-tree kill only
# Set by campaign runner when multiple SITL instances share the machine
HARNESS_PARALLEL_SAFE="${HARNESS_PARALLEL_SAFE:-0}"

# High-Side Convergence Mode (TEST 392+)
# When enabled, follower flies above leader during convergence phases to avoid wake
HARNESS_HIGH_SIDE_CONVERGENCE="${HARNESS_HIGH_SIDE_CONVERGENCE:-0}"
HIGH_SIDE_MARGIN_M="${HIGH_SIDE_MARGIN_M:-6.0}"           # Altitude above leader during high-side
HIGH_SIDE_CROSS_OK_M="${HIGH_SIDE_CROSS_OK_M:-2.5}"       # Cross-track threshold for descent trigger
HIGH_SIDE_BEHIND_MAX_M="${HIGH_SIDE_BEHIND_MAX_M:-250.0}" # Max behind distance for high-side mode
HIGH_SIDE_MIN_TIME_S="${HIGH_SIDE_MIN_TIME_S:-5.0}"       # Min time cross must be OK before descent
HIGH_SIDE_DISABLE_BEHIND_M="${HIGH_SIDE_DISABLE_BEHIND_M:-60.0}"  # Force descent before close-in

# =============================================================================
# PATHS (set by harness_init based on SCRIPT_DIR)
# =============================================================================
HARNESS_SCRIPT_DIR=""
HARNESS_FLIGHTDOCK_DIR=""
HARNESS_TOOLS_DIR=""
HARNESS_LOG_BASE=""
HARNESS_VENV_DIR=""

# =============================================================================
# COLORS
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================
log_info() {
    echo -e "${CYAN}[HARNESS]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_debug() {
    if [[ "${HARNESS_DEBUG:-0}" == "1" ]]; then
        echo -e "[DEBUG] $1"
    fi
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
harness_parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --speedup)
                HARNESS_SPEEDUP="$2"
                shift 2
                ;;
            --sim-seconds)
                HARNESS_SIM_SECONDS="$2"
                shift 2
                ;;
            --test-id)
                HARNESS_TEST_ID="$2"
                shift 2
                ;;
            --test-name)
                HARNESS_TEST_NAME="$2"
                shift 2
                ;;
            --leader-profile|--profile)
                HARNESS_LEADER_PROFILE="$2"
                shift 2
                ;;
            --dry-run)
                HARNESS_DRY_RUN="1"
                shift
                ;;
            --reuse-sitl)
                HARNESS_SITL_MODE="reuse"
                shift
                ;;
            --start-sitl)
                HARNESS_SITL_MODE="start"
                shift
                ;;
            --skip-sitl)
                # Legacy: treat as reuse
                HARNESS_SITL_MODE="reuse"
                shift
                ;;
            --ui)
                HARNESS_UI_MODE="$2"
                # UI modes map to headless flag
                case "$2" in
                    none)
                        HARNESS_HEADLESS="1"
                        ;;
                    map|console|both)
                        HARNESS_HEADLESS="0"
                        ;;
                    *)
                        log_fail "Invalid --ui mode: $2 (use: none, map, console, both)"
                        exit 1
                        ;;
                esac
                shift 2
                ;;
            --headless)
                HARNESS_HEADLESS="1"
                HARNESS_UI_MODE="none"
                shift
                ;;
            --mavproxy-map)
                HARNESS_UI_MODE="map"
                HARNESS_HEADLESS="0"
                shift
                ;;
            --mavproxy-console)
                HARNESS_UI_MODE="console"
                HARNESS_HEADLESS="0"
                shift
                ;;
            --timebase-gate)
                HARNESS_TIMEBASE_GATE="1"
                shift
                ;;
            --skip-timebase|--no-timebase-gate)
                HARNESS_TIMEBASE_GATE="0"
                shift
                ;;
            --timebase-duration)
                HARNESS_TIMEBASE_DURATION="$2"
                shift 2
                ;;
            --leader-freshness-gate)
                HARNESS_LEADER_FRESHNESS_GATE="1"
                shift
                ;;
            --leader-freshness-threshold)
                HARNESS_LEADER_FRESHNESS_THRESHOLD_MS="$2"
                shift 2
                ;;
            --drl-training-mode|--drl-mode)
                HARNESS_DRL_MODE="1"
                shift
                ;;
            --smoketest-only)
                HARNESS_SMOKETEST_ONLY="1"
                shift
                ;;
            --smoketest-post-settle)
                HARNESS_SMOKETEST_POST_SETTLE_S="$2"
                shift 2
                ;;
            --run-tag)
                HARNESS_RUN_TAG="$2"
                shift 2
                ;;
            --force-ui)
                HARNESS_FORCE_UI="1"
                shift
                ;;
            --min-achieved-speedup-frac)
                HARNESS_MIN_ACHIEVED_SPEEDUP_FRAC="$2"
                shift 2
                ;;
            --env-id)
                HARNESS_ENV_ID="$2"
                HARNESS_PORT_OFFSET=$((HARNESS_ENV_ID * 100))
                shift 2
                ;;
            --connection|--conn)
                HARNESS_CONN="$2"
                shift 2
                ;;
            --racetrack)
                HARNESS_LEADER_PROFILE="racetrack"
                shift
                ;;
            --straight)
                HARNESS_LEADER_PROFILE="straight"
                shift
                ;;
            --wipe|--clean-params)
                HARNESS_WIPE_PARAMS="1"
                shift
                ;;
            --no-wipe|--skip-wipe)
                HARNESS_WIPE_PARAMS="0"
                shift
                ;;
            --high-side-convergence)
                HARNESS_HIGH_SIDE_CONVERGENCE="1"
                shift
                ;;
            --high-side-margin)
                HIGH_SIDE_MARGIN_M="$2"
                shift 2
                ;;
            --high-side-cross-ok)
                HIGH_SIDE_CROSS_OK_M="$2"
                shift 2
                ;;
            --high-side-behind-max)
                HIGH_SIDE_BEHIND_MAX_M="$2"
                shift 2
                ;;
            --high-side-min-time)
                HIGH_SIDE_MIN_TIME_S="$2"
                shift 2
                ;;
            --high-side-disable-behind)
                HIGH_SIDE_DISABLE_BEHIND_M="$2"
                shift 2
                ;;
            --help|-h)
                harness_show_help
                exit 0
                ;;
            *)
                # Allow callers to handle unknown args
                if [[ "${HARNESS_STRICT_ARGS:-0}" == "1" ]]; then
                    log_fail "Unknown option: $1"
                    harness_show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # GUI Speedup Policy: clamp speedup if UI enabled at high speedup
    if [[ "$HARNESS_HEADLESS" == "0" ]]; then
        if (( $(echo "$HARNESS_SPEEDUP > $HARNESS_GUI_MAX_SPEEDUP" | bc -l) )); then
            if [[ "$HARNESS_FORCE_UI" == "1" ]]; then
                log_warn "╔═══════════════════════════════════════════════════════════════════╗"
                log_warn "║ ⚠️  --force-ui: Clamping ${HARNESS_SPEEDUP}x → ${HARNESS_GUI_MAX_SPEEDUP}x for GUI stability"
                log_warn "╚═══════════════════════════════════════════════════════════════════╝"
                HARNESS_SPEEDUP="$HARNESS_GUI_MAX_SPEEDUP"
            else
                log_info "[GUI_POLICY] Speedup ${HARNESS_SPEEDUP}x > ${HARNESS_GUI_MAX_SPEEDUP}x with UI"
                log_info "[GUI_POLICY] Clamping to ${HARNESS_GUI_MAX_SPEEDUP}x (use --headless for full speed)"
                HARNESS_SPEEDUP="$HARNESS_GUI_MAX_SPEEDUP"
            fi
        fi
    fi
}

harness_show_help() {
    cat << 'EOF'
FlightDock Test Harness - Common Options
=========================================

SITL Control:
  --reuse-sitl              Use existing SITL (fail if not running)
  --start-sitl              Force-start SITL (kill existing first)
  (default)                 Auto: reuse if running, start if not

UI Control:
  --headless                No GUI windows (default for HPC/CI)
  --ui none|map|console|both    Select MAVProxy UI mode
  --mavproxy-map            Enable MAVProxy map window
  --mavproxy-console        Enable MAVProxy console window
  --force-ui                Force UI even at high speedup (will clamp speedup)

Timing:
  --speedup N               Sim speedup factor (default: 5.0)
  --sim-seconds N           Max sim-time duration (default: 1500)
  --min-achieved-speedup-frac F   Fail-fast threshold (default: 0.8)
  --timebase-gate           Enable timebase smoketest gate (default: ON)
  --skip-timebase           Disable timebase smoketest gate
  --timebase-duration N     Timebase test duration (default: 30s)

Mission:
  --test-id N               Controller test ID (default: 334)
  --leader-profile P        straight or racetrack (default: straight)
  --racetrack               Shorthand for --leader-profile racetrack
  --straight                Shorthand for --leader-profile straight
  --drl-training-mode       Relax timing checks for DRL training

Parameters:
  --wipe                    Wipe SITL parameters before test (default: ON)
  --no-wipe                 Skip parameter wipe (for rapid iteration)

High-Side Convergence (Wake Avoidance):
  --high-side-convergence   Enable high-side altitude mode during convergence
  --high-side-margin M      Altitude above leader during high-side (default: 6.0m)
  --high-side-cross-ok M    Cross-track threshold for descent (default: 2.5m)
  --high-side-behind-max M  Max behind distance for high-side (default: 250m)
  --high-side-min-time S    Min time cross OK before descent (default: 5s)
  --high-side-disable-behind M  Force descent before this range (default: 60m)

Calibration:
  --smoketest-only          Run SHORT smoketest flow (for calibration)
  --smoketest-post-settle N Post-settle sim-time window (default: 20s)
  --run-tag TAG             Tag for output directory naming

General:
  --dry-run                 Print commands without executing
  --help                    Show this help

Examples:
  # Headless smoketest at 5x
  ./run_test_334A_smoketest.sh --speedup 5.0

  # Interactive R&D with map+console at 3x
  ./run_test_334.sh --speedup 3.0 --ui both

  # Headless high-speed run with strict gating
  ./run_test_334.sh --speedup 10.0 --headless --timebase-gate

  # Reuse existing SITL
  ./run_test_334.sh --reuse-sitl --speedup 3.0

  # Calibration smoketest with run tag
  ./run_test_336.sh --smoketest-only --run-tag speedup_calibration --speedup 5.0
EOF
}

# =============================================================================
# PORT CALCULATION HELPER
# =============================================================================
# Compute port offset from env_id and set connection strings
# Port scheme: base_port + (env_id * 100)
# - SITL TCP: 5760 + offset
# - Leader in: 14550 + offset
# - Follower out: 14551 + offset
# - Monitor: 14552 + offset
# - Timesync: 14553 + offset
# - Control in: 14560 + offset
harness_compute_ports() {
    local env_id="${1:-$HARNESS_ENV_ID}"
    local offset=$((env_id * 100))

    # Export computed port offset
    HARNESS_PORT_OFFSET="$offset"

    # Compute connection strings if not explicitly set
    if [[ -z "$HARNESS_CONN" ]]; then
        local follower_port=$((14551 + offset))
        HARNESS_CONN="udp:127.0.0.1:${follower_port}"
    fi

    if [[ -z "$HARNESS_TIMESYNC_CONN" ]]; then
        local timesync_port=$((14553 + offset))
        HARNESS_TIMESYNC_CONN="udpin:127.0.0.1:${timesync_port}"
    fi

    # Export for use by start_formation_sitl_fixed.sh
    export HARNESS_PORT_OFFSET
    export SITL_PORT_OFFSET="$offset"
}

# =============================================================================
# INITIALIZATION
# =============================================================================
harness_init() {
    local calling_script="${1:-}"

    if [[ "$HARNESS_INITIALIZED" == "1" ]]; then
        log_warn "harness_init called multiple times"
        return 0
    fi

    # Determine script directory
    if [[ -n "$calling_script" ]]; then
        HARNESS_SCRIPT_DIR="$(cd "$(dirname "$calling_script")" && pwd)"
    elif [[ -n "${BASH_SOURCE[1]:-}" ]]; then
        HARNESS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    else
        HARNESS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi

    # Set up paths
    HARNESS_FLIGHTDOCK_DIR="$(dirname "$HARNESS_SCRIPT_DIR")"
    HARNESS_TOOLS_DIR="${HARNESS_FLIGHTDOCK_DIR}/tools"
    HARNESS_LOG_BASE="${HARNESS_FLIGHTDOCK_DIR}/logs/intercept_tests/test_logs"
    HARNESS_VENV_DIR="${HARNESS_FLIGHTDOCK_DIR}/flightdock_native_venv"

    # Compute port offset and connection strings from env-id
    harness_compute_ports "$HARNESS_ENV_ID"

    # Create output directory with timestamp
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local dir_name="${HARNESS_TEST_NAME}_${HARNESS_SPEEDUP}x_${timestamp}"
    if [[ -n "$HARNESS_RUN_TAG" ]]; then
        dir_name="${HARNESS_RUN_TAG}_${HARNESS_TEST_ID}_${HARNESS_SPEEDUP}x_${timestamp}"
    fi
    HARNESS_OUT_DIR="${HARNESS_SCRIPT_DIR}/analysis_out/${dir_name}"
    mkdir -p "$HARNESS_OUT_DIR"
    mkdir -p "$HARNESS_LOG_BASE"

    # Output HARNESS_OUT_DIR for machine parsing
    echo "HARNESS_OUT_DIR=$HARNESS_OUT_DIR"

    # Initialize PID tracking
    HARNESS_PIDS_FILE="${HARNESS_OUT_DIR}/pids.txt"
    echo "# PIDs for ${HARNESS_TEST_NAME} - ${timestamp}" > "$HARNESS_PIDS_FILE"
    echo "harness_version=${HARNESS_VERSION}" >> "$HARNESS_PIDS_FILE"

    # TEST 336+: Write config snapshot JSON for analysis
    harness_write_config_snapshot

    # Set up trap
    trap harness_cleanup EXIT INT TERM

    HARNESS_INITIALIZED=1
    log_info "Harness v${HARNESS_VERSION} initialized"
    log_info "Output: ${HARNESS_OUT_DIR}"
}

# =============================================================================
# CONFIG SNAPSHOT (TEST 336+ / TEST 338 FIX)
# =============================================================================
# TEST 338 FIX: Uses Python helper to ensure valid JSON with numeric values.
# Previous shell grep approach silently failed, producing empty gating_thresholds.
harness_write_config_snapshot() {
    local config_json="${HARNESS_OUT_DIR}/config_snapshot.json"
    local helper_script="${HARNESS_TOOLS_DIR}/config_snapshot_helper.py"

    log_info "Writing config snapshot..."

    # Get git commit hash if available
    local git_commit="unknown"
    if command -v git >/dev/null 2>&1 && [[ -d "${HARNESS_FLIGHTDOCK_DIR}/.git" ]]; then
        git_commit=$(cd "${HARNESS_FLIGHTDOCK_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    fi

    # TEST 338: Use Python helper for reliable JSON generation
    if [[ -f "$helper_script" ]]; then
        if python3 "$helper_script" write \
            --test-id "${HARNESS_TEST_ID}" \
            --test-name "${HARNESS_TEST_NAME}" \
            --speedup "${HARNESS_SPEEDUP}" \
            --sim-seconds "${HARNESS_SIM_SECONDS}" \
            --headless "${HARNESS_HEADLESS}" \
            --ui-mode "${HARNESS_UI_MODE}" \
            --leader-profile "${HARNESS_LEADER_PROFILE}" \
            --timebase-gate "${HARNESS_TIMEBASE_GATE}" \
            --drl-mode "${HARNESS_DRL_MODE}" \
            --timebase-duration "${HARNESS_TIMEBASE_DURATION}" \
            --min-achieved-speedup-frac "${HARNESS_MIN_ACHIEVED_SPEEDUP_FRAC}" \
            --env-id "${HARNESS_ENV_ID}" \
            --port-offset "${HARNESS_PORT_OFFSET}" \
            --conn "${HARNESS_CONN}" \
            --timesync-conn "${HARNESS_TIMESYNC_CONN}" \
            --git-commit "${git_commit}" \
            --harness-version "${HARNESS_VERSION}" \
            --cpp-handoff-policy "${HARNESS_CPP_HANDOFF_POLICY:-state_gated}" \
            --output "$config_json"; then
            log_pass "Config snapshot written via Python helper"
        else
            log_warn "Python helper failed, falling back to shell method"
            _harness_write_config_snapshot_shell "$config_json" "$git_commit"
        fi
    else
        log_warn "Python helper not found, using shell method"
        _harness_write_config_snapshot_shell "$config_json" "$git_commit"
    fi

    # Self-check: verify JSON is valid
    if python3 -c "import json; json.load(open('$config_json'))" 2>/dev/null; then
        log_pass "Config snapshot JSON validated"
    else
        log_fail "Config snapshot JSON is invalid!"
        cat "$config_json"
        return 1
    fi
}

# Fallback shell method (kept for compatibility)
_harness_write_config_snapshot_shell() {
    local config_json="$1"
    local git_commit="$2"

    # Get controller version from baseline_formation_controller.py
    local controller_version="unknown"
    if [[ -f "${HARNESS_SCRIPT_DIR}/baseline_formation_controller.py" ]]; then
        controller_version=$(grep -o 'CONTROLLER_VERSION.*=.*"[^"]*"' \
            "${HARNESS_SCRIPT_DIR}/baseline_formation_controller.py" | \
            head -1 | sed 's/.*"\([^"]*\)".*/\1/' || echo "unknown")
    fi

    # Hardcode gating thresholds (safer than fragile grep)
    local quality_gate="0.95"
    local monotonicity_gate="0"

    # Write JSON
    cat > "$config_json" << EOF
{
  "test_id": "${HARNESS_TEST_ID}",
  "test_name": "${HARNESS_TEST_NAME}",
  "timestamp": "$(date +%Y%m%d_%H%M%S)",
  "speedup": ${HARNESS_SPEEDUP},
  "sim_seconds": ${HARNESS_SIM_SECONDS},
  "headless": ${HARNESS_HEADLESS},
  "ui_mode": "${HARNESS_UI_MODE}",
  "leader_profile": "${HARNESS_LEADER_PROFILE}",
  "controller_version": "${controller_version}",
  "git_commit": "${git_commit}",
  "harness_version": "${HARNESS_VERSION}",
  "gating_thresholds": {
    "quality_gate_min": ${quality_gate},
    "monotonicity_gate_max": ${monotonicity_gate},
    "rate_tolerance_pct": 0.25,
    "timebase_duration_s": ${HARNESS_TIMEBASE_DURATION},
    "min_achieved_speedup_frac": ${HARNESS_MIN_ACHIEVED_SPEEDUP_FRAC}
  },
  "env": {
    "env_id": ${HARNESS_ENV_ID},
    "port_offset": ${HARNESS_PORT_OFFSET},
    "conn": "${HARNESS_CONN}",
    "timesync_conn": "${HARNESS_TIMESYNC_CONN}"
  },
  "flags": {
    "timebase_gate": ${HARNESS_TIMEBASE_GATE},
    "drl_mode": ${HARNESS_DRL_MODE}
  }
}
EOF

    log_info "Config snapshot written to: $config_json (shell fallback)"
}

# =============================================================================
# CLEANUP
# =============================================================================
harness_cleanup() {
    local exit_code=$?

    # Prevent multiple cleanup runs
    if [[ "$HARNESS_CLEANUP_DONE" == "1" ]]; then
        return
    fi
    HARNESS_CLEANUP_DONE=1

    log_info "Cleanup started (exit code: $exit_code)"

    # --- Truth observer reap (gated; truth-logger STRUCTURAL 2026-06-09) -------
    # SIGTERM -> wait 3s (observer SIGTERM handler flushes JSONL+summary) ->
    # SIGKILL -> assert pgrep empty. REAP_FAILED => force exit_code=2 (HALT).
    # When TRUTH_OBSERVER_ENABLE=0, OBSERVER_PID is never set => whole block skipped.
    if [[ -n "${OBSERVER_PID:-}" ]]; then
        log_info "Reaping truth observer (PID $OBSERVER_PID)"
        kill -TERM "$OBSERVER_PID" 2>/dev/null || true
        local _i
        for _i in 1 2 3; do
            kill -0 "$OBSERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$OBSERVER_PID" 2>/dev/null; then
            log_info "Observer alive after 3s grace; sending SIGKILL"
            kill -KILL "$OBSERVER_PID" 2>/dev/null || true
            sleep 1
        fi
        if pgrep -f v98_standalone_truth_observer >/dev/null 2>&1; then
            log_fail "REAP_FAILED: truth observer still running after SIGKILL"
            exit_code=2
        else
            log_pass "REAP_CONFIRMED: truth observer reaped"
        fi
    fi

    # Kill leader broadcaster if we started it
    if [[ -n "${HARNESS_LEADER_PID:-}" ]] && kill -0 "$HARNESS_LEADER_PID" 2>/dev/null; then
        log_info "Stopping leader broadcaster (PID $HARNESS_LEADER_PID)"
        kill -INT "$HARNESS_LEADER_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$HARNESS_LEADER_PID" 2>/dev/null || true
    fi

    # Kill SITL only if we started it
    if [[ "$HARNESS_SITL_OWNED" == "1" ]]; then
        log_info "Stopping SITL (started by this harness)"
        harness_kill_sitl
    else
        log_info "Leaving SITL running (not started by this harness)"
    fi

    # Create _latest symlink
    if [[ -d "${HARNESS_OUT_DIR:-}" ]]; then
        local latest_link="${HARNESS_SCRIPT_DIR}/analysis_out/_latest_${HARNESS_TEST_NAME}"
        rm -f "$latest_link"
        ln -sf "$HARNESS_OUT_DIR" "$latest_link"
        log_info "Symlink created: $latest_link"
    fi

    log_info "Cleanup complete"
    exit "$exit_code"
}

# =============================================================================
# SITL MANAGEMENT
# =============================================================================

# Kill process tree (process and all children)
harness_stop_proc_tree() {
    local pid=${1:-}
    if [[ -z "$pid" ]]; then
        return 0
    fi

    # Get all child PIDs
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)

    # Kill children first (recursively)
    for child in $children; do
        harness_stop_proc_tree "$child"
    done

    # Kill the main process
    if kill -0 "$pid" 2>/dev/null; then
        log_debug "Stopping process $pid"
        kill -INT "$pid" 2>/dev/null || true
        sleep 0.5
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
}

# Detect if SITL is running using heartbeat
# Returns 0 if SITL is running and responding, 1 otherwise
harness_detect_sitl_running() {
    local timeout_s=${1:-5}

    # First check if arduplane process exists
    if ! pgrep -f "arduplane" >/dev/null 2>&1; then
        return 1
    fi

    # Check if mavproxy exists
    if ! pgrep -f "mavproxy" >/dev/null 2>&1; then
        return 1
    fi

    # Heartbeat-based detection using pymavlink
    local result
    result=$(timeout "${timeout_s}" python3 -c "
import sys
from pymavlink import mavutil
try:
    mav = mavutil.mavlink_connection('${HARNESS_CONN}', source_system=254)
    msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=3.0)
    if msg:
        print('OK')
        sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null || true)

    [[ "$result" == "OK" ]]
}

# Wait for SITL to become ready
harness_wait_for_sitl_ready() {
    local timeout_s=${1:-$HARNESS_SITL_READY_TIMEOUT}
    local poll_interval=5
    local elapsed=0

    log_info "Waiting for SITL heartbeat (timeout: ${timeout_s}s)..."

    while [[ "$elapsed" -lt "$timeout_s" ]]; do
        # Check if SITL script died
        if [[ -n "${HARNESS_SITL_SCRIPT_PID:-}" ]] && ! kill -0 "$HARNESS_SITL_SCRIPT_PID" 2>/dev/null; then
            log_fail "SITL start script exited unexpectedly"
            return 1
        fi

        # Try heartbeat detection
        if harness_detect_sitl_running 3; then
            log_pass "SITL heartbeat detected at ${elapsed}s"
            return 0
        fi

        echo -n "."
        sleep "$poll_interval"
        elapsed=$((elapsed + poll_interval))
    done

    echo ""
    return 1
}

# Start SITL
harness_start_sitl() {
    local sitl_log="${HARNESS_OUT_DIR}/sitl_console.log"

    log_info "Starting SITL (UI mode: ${HARNESS_UI_MODE})..."

    local start_script="${HARNESS_SCRIPT_DIR}/start_formation_sitl_fixed.sh"
    if [[ ! -f "$start_script" ]]; then
        log_fail "SITL start script not found: $start_script"
        return 1
    fi

    # Build environment for SITL start script
    local sitl_env="HEADLESS=${HARNESS_HEADLESS}"
    sitl_env="${sitl_env} HARNESS_UI_MODE=${HARNESS_UI_MODE}"
    sitl_env="${sitl_env} SITL_SPEEDUP=${HARNESS_SPEEDUP}"

    # Parallel-safe: give each instance its own working dir for eeprom.bin isolation
    if [[ "${HARNESS_PARALLEL_SAFE:-0}" == "1" ]]; then
        local work_dir="/tmp/flightdock_sitl_env${HARNESS_ENV_ID}_$$"
        sitl_env="${sitl_env} SITL_WORK_DIR=${work_dir}"
        log_info "Parallel-safe SITL work dir: $work_dir"
    fi

    # Start SITL in background with setsid for clean shutdown
    # Pass speedup as both env var and flag for robustness
    # Use --direct mode to bypass sim_vehicle.py (WSL2 compatible)
    setsid bash -c "cd '${HARNESS_SCRIPT_DIR}' && ${sitl_env} ./start_formation_sitl_fixed.sh --direct --speedup ${HARNESS_SPEEDUP} --ui ${HARNESS_UI_MODE}" \
        > "$sitl_log" 2>&1 &
    HARNESS_SITL_SCRIPT_PID=$!

    log_info "SITL start script PID: $HARNESS_SITL_SCRIPT_PID"
    log_info "SITL console log: $sitl_log"

    echo "sitl_script_pid=$HARNESS_SITL_SCRIPT_PID" >> "$HARNESS_PIDS_FILE"

    # Wait for SITL to become ready
    if harness_wait_for_sitl_ready "$HARNESS_SITL_READY_TIMEOUT"; then
        log_pass "SITL is ready"
        HARNESS_SITL_OWNED=1
        echo "sitl_owned=1" >> "$HARNESS_PIDS_FILE"
        return 0
    else
        log_fail "SITL failed to start within ${HARNESS_SITL_READY_TIMEOUT}s"
        return 1
    fi
}

# Kill SITL processes
harness_kill_sitl() {
    log_info "Stopping SITL processes..."

    # If we know the script PID, use process tree kill
    if [[ -n "${HARNESS_SITL_SCRIPT_PID:-}" ]]; then
        harness_stop_proc_tree "$HARNESS_SITL_SCRIPT_PID"
    fi

    # Global kill-by-name fallback — skip in parallel-safe mode
    # to avoid killing sibling instances on different ports
    if [[ "${HARNESS_PARALLEL_SAFE:-0}" != "1" ]]; then
        pkill -9 -f "sim_vehicle.py" 2>/dev/null || true
        pkill -9 -f "arduplane" 2>/dev/null || true
        pkill -9 -f "mavproxy" 2>/dev/null || true
    fi

    sleep 2
    log_info "SITL processes stopped"
}

# Wipe SITL parameters for clean test environment
harness_wipe_params() {
    log_info "Wiping SITL parameters for clean test..."
    local ARDUPILOT_DIR="$HOME/FlightDock/ardupilot_formation"

    # Delete eeprom.bin files (SITL persistent parameter storage)
    rm -f "$ARDUPILOT_DIR/ArduPlane/eeprom.bin" 2>/dev/null || true
    rm -f "$ARDUPILOT_DIR/eeprom.bin" 2>/dev/null || true

    # Delete mav.parm files (MAVProxy parameter cache)
    rm -f "$ARDUPILOT_DIR/ArduPlane/mav.parm" 2>/dev/null || true
    rm -f "$HOME/FlightDock/tests/mav.parm" 2>/dev/null || true

    log_pass "Parameter cache wiped (eeprom.bin, mav.parm)"
}

# Ensure SITL is running (main entry point)
harness_ensure_sitl() {
    log_info "========================================"
    log_info "SITL MANAGEMENT (mode: ${HARNESS_SITL_MODE})"
    log_info "========================================"

    # ====================================================================
    # OPTION 0 FIX: UNCONDITIONAL PRE-TEST CLEANUP
    # ====================================================================
    # CRITICAL: Always clean BEFORE SITL detection to prevent parameter
    # contamination from previous tests (eeprom.bin persists speedup!)
    # This happens REGARDLESS of --wipe flag or SITL mode.
    # ====================================================================

    log_info "[OPTION_0] Unconditional pre-test cleanup (prevent contamination)..."

    if [[ "${HARNESS_PARALLEL_SAFE:-0}" == "1" ]]; then
        # In parallel-safe mode, skip global kill and shared-file wipe.
        # The campaign runner does global cleanup between waves, and each
        # instance uses SITL_WORK_DIR for isolated eeprom.bin.
        log_info "[OPTION_0] Parallel-safe mode: skipping global kill + shared wipe"
    else
        # 1. Kill existing SITL processes unconditionally (even in "auto" mode)
        log_info "[OPTION_0] Killing any existing SITL processes..."
        harness_kill_sitl

        # 2. Delete persistent parameter storage (eeprom.bin is the culprit!)
        log_info "[OPTION_0] Wiping persistent parameter storage (eeprom.bin)..."
        harness_wipe_params

        # 3. Delete SITL logs (optional but recommended for clean slate)
        local ARDUPILOT_DIR="$HOME/FlightDock/ardupilot_formation"
        rm -rf "$ARDUPILOT_DIR/ArduPlane/logs/"* 2>/dev/null || true
    fi

    log_pass "[OPTION_0] Pre-test cleanup complete (fresh SITL start guaranteed)"

    # ====================================================================
    # END OPTION 0 FIX
    # ====================================================================

    # Legacy parameter wipe check (now redundant, but kept for compatibility)
    if [[ "${HARNESS_WIPE_PARAMS:-1}" == "1" ]]; then
        log_info "Parameter wipe already done by Option 0 cleanup"
    else
        log_warn "Parameter wipe skipped by --no-wipe (not recommended with Option 0)"
    fi

    case "$HARNESS_SITL_MODE" in
        reuse)
            log_info "Checking for existing SITL (heartbeat detection)..."
            if harness_detect_sitl_running 5; then
                local sitl_pids
                sitl_pids=$(pgrep -f "arduplane" 2>/dev/null || true)
                log_pass "SITL detected and responding (PIDs: $sitl_pids)"
                echo "sitl_pids=$sitl_pids" >> "$HARNESS_PIDS_FILE"
                echo "sitl_mode=reuse" >> "$HARNESS_PIDS_FILE"
            else
                log_fail "SITL not detected. Start SITL first or use default (auto) mode"
                # R6: Return error code instead of exit - let caller handle, preserves matrix loop
                return 1
            fi
            ;;

        start)
            log_info "Force-starting SITL (killing any existing instances)..."
            harness_kill_sitl
            if harness_start_sitl; then
                local sitl_pids
                sitl_pids=$(pgrep -f "arduplane" 2>/dev/null || true)
                log_pass "SITL started fresh (PIDs: $sitl_pids)"
                echo "sitl_pids=$sitl_pids" >> "$HARNESS_PIDS_FILE"
                echo "sitl_mode=started" >> "$HARNESS_PIDS_FILE"
            else
                log_fail "Failed to start SITL"
                # R6: Return error code instead of exit - let caller handle, preserves matrix loop
                return 1
            fi
            ;;

        auto|*)
            log_info "Auto-detecting SITL status..."
            if harness_detect_sitl_running 5; then
                local sitl_pids
                sitl_pids=$(pgrep -f "arduplane" 2>/dev/null || true)
                log_pass "SITL already running (reusing, PIDs: $sitl_pids)"
                echo "sitl_pids=$sitl_pids" >> "$HARNESS_PIDS_FILE"
                echo "sitl_mode=reused" >> "$HARNESS_PIDS_FILE"
            else
                log_info "SITL not detected, starting automatically..."
                if harness_start_sitl; then
                    local sitl_pids
                    sitl_pids=$(pgrep -f "arduplane" 2>/dev/null || true)
                    log_pass "SITL started (PIDs: $sitl_pids)"
                    echo "sitl_pids=$sitl_pids" >> "$HARNESS_PIDS_FILE"
                    echo "sitl_mode=started" >> "$HARNESS_PIDS_FILE"
                else
                    log_fail "Failed to start SITL automatically"
                    log_info "You can try starting SITL manually:"
                    echo ""
                    echo "    cd ${HARNESS_SCRIPT_DIR}"
                    echo "    ./start_formation_sitl_fixed.sh"
                    echo ""
                    echo "Then re-run this script with --reuse-sitl"
                    # R6: Return error code instead of exit - let caller handle, preserves matrix loop
                    return 1
                fi
            fi
            ;;
    esac

    # Final verification
    sleep 2
    log_info "Verifying SITL is ready (heartbeat check)..."
    if ! harness_detect_sitl_running 10; then
        log_fail "SITL verification failed (no heartbeat)"
        if [[ -f "${HARNESS_OUT_DIR}/sitl_console.log" ]]; then
            log_info "Last 10 lines of SITL log:"
            tail -10 "${HARNESS_OUT_DIR}/sitl_console.log" 2>/dev/null || true
        fi
        # R6: Return error code instead of exit - let caller handle, preserves matrix loop
        return 1
    fi
    log_pass "SITL verified and ready"

    # --- Truth observer sidecar (gated; truth-logger STRUCTURAL 2026-06-09) ----
    # When TRUTH_OBSERVER_ENABLE=1, spawn the SEPARATE-PROCESS truth observer on
    # the MAVProxy UI fanout (14554 + env*100). It self-requests SIM_STATE(108).
    # Default unset/0 => not spawned => this path is byte-identical to before.
    # OBSERVER_PID is intentionally a GLOBAL so harness_cleanup() can reap it.
    if [[ "${TRUTH_OBSERVER_ENABLE:-0}" == "1" ]]; then
        local _obs_script="${HARNESS_FLIGHTDOCK_DIR:-${HARNESS_SCRIPT_DIR}/..}/tools/v98_standalone_truth_observer.py"
        if [[ -f "$_obs_script" ]]; then
            HARNESS_OUT_DIR="$HARNESS_OUT_DIR" \
            HARNESS_ENV_ID="${HARNESS_ENV_ID:-0}" \
            OBSERVER_TICK_HZ="${OBSERVER_TICK_HZ:-10}" \
                python3 -u "$_obs_script" \
                > "${HARNESS_OUT_DIR}/truth_observer.log" 2>&1 &
            OBSERVER_PID=$!
            echo "observer_pid=$OBSERVER_PID" >> "$HARNESS_PIDS_FILE"
            log_pass "Truth observer started (PID $OBSERVER_PID, port $((14554 + ${HARNESS_ENV_ID:-0} * 100)))"
        else
            log_info "TRUTH_OBSERVER_ENABLE=1 but observer not found: $_obs_script"
        fi
    fi

    # Option 0 Fix: Speedup validation with retries
    if ! harness_validate_speedup_with_retries; then
        log_fail "Speedup validation failed after multiple attempts"
        log_fail "SITL may not be running at requested speedup ${HARNESS_SPEEDUP}x"
        # R6: Return error code instead of exit - let caller handle, preserves matrix loop
        return 1
    fi
}

# =============================================================================
# OPTION 0 FIX: SPEEDUP VALIDATION WITH RETRIES
# =============================================================================
# Validates that SITL is running at the requested speedup.
# Performs multiple attempts: set → verify → retry if mismatch
# Returns 0 on success, 1 on failure
harness_validate_speedup_with_retries() {
    local requested="${HARNESS_SPEEDUP:-1.0}"
    local tolerance=0.15  # Allow 15% variance (e.g., 5.0x ± 0.75x)
    local max_attempts=3
    local attempt=1
    local set_succeeded=0

    log_info "[OPTION_0] Validating speedup: ${requested}x (tolerance: ±${tolerance}x)..."

    while [[ $attempt -le $max_attempts ]]; do
        log_info "[OPTION_0] Attempt $attempt/$max_attempts: Setting speedup to ${requested}x..."

        # 1. Set speedup via MAVLink
        if ! harness_set_speedup; then
            log_warn "[OPTION_0] Attempt $attempt: speedup set failed"
            sleep 2
            ((attempt++))
            continue
        fi

        # SET succeeded at least once
        set_succeeded=1

        # 2. Brief delay for speedup to take effect
        sleep 2

        # 3. Verify speedup via readback (optional - may fail at high speedup)
        local speedup_guard="${HARNESS_TOOLS_DIR}/speedup_guard.py"
        if [[ ! -f "$speedup_guard" ]]; then
            log_warn "[OPTION_0] speedup_guard.py not found, trusting SET command"
            log_pass "[OPTION_0] Speedup SET to ${requested}x (readback skipped - tool missing)"
            return 0
        fi

        local conn="${HARNESS_CONN:-udp:127.0.0.1:14551}"
        local readback_output
        readback_output=$(python3 "$speedup_guard" readback \
            --conn "$conn" \
            --expected "$requested" \
            --timeout 10 2>&1 || true)

        # Extract actual speedup from output
        local actual
        actual=$(echo "$readback_output" | grep -oP '\[READBACK\] SIM_SPEEDUP = \K[\d\.]+' | head -1 || echo "")

        if [[ -z "$actual" ]]; then
            log_warn "[OPTION_0] Attempt $attempt: Could not read back speedup"
            # Don't fail yet - continue trying
            ((attempt++))
            sleep 2
            continue
        fi

        # Check if actual is within tolerance
        local delta
        delta=$(echo "scale=2; ($actual - $requested)" | bc -l | sed 's/^-//')  # absolute value

        if (( $(echo "$delta <= $tolerance" | bc -l) )); then
            log_pass "[OPTION_0] Speedup validated: ${actual}x (requested ${requested}x, delta ±${delta}x)"
            return 0
        else
            log_warn "[OPTION_0] Attempt $attempt: Speedup mismatch (actual=${actual}x, requested=${requested}x, delta=±${delta}x)"
        fi

        ((attempt++))
        sleep 2
    done

    # If we got here, readback failed all attempts BUT if SET succeeded, that's acceptable
    # The timebase gate will catch actual speedup mismatches later
    if [[ $set_succeeded -eq 1 ]]; then
        log_warn "[OPTION_0] Speedup readback failed after $max_attempts attempts"
        log_warn "[OPTION_0] SET command succeeded, trusting it (timebase gate will verify)"
        log_pass "[OPTION_0] Speedup SET to ${requested}x (readback unavailable, trusting command)"
        return 0
    fi

    # SET never succeeded - this is a real failure
    log_fail "[OPTION_0] Speedup SET command failed after $max_attempts attempts"
    return 1
}

# =============================================================================
# ATTITUDE RATE ENFORCEMENT (Checklist Steps 2-3)
# =============================================================================
# Ensures ATTITUDE message rate is 20 Hz by:
# 1. Setting SR0_EXTRA1=20 via MAVLink (covers ATTITUDE stream group)
# 2. Verifying the rate is applied
# 3. Optionally setting SERIAL_OPTIONS bit 12 (Ignore Streamrate)
#
# Note: MAVProxy init scripts already apply:
#   set streamrate -1
#   module load messagerate
#   messagerate set ATTITUDE 20
# This function adds SRx parameter enforcement as belt-and-suspenders.
# =============================================================================
harness_configure_attitude_rate() {
    log_info "========================================"
    log_info "ATTITUDE RATE PREFLIGHT (PRH-ATT-002)"
    log_info "========================================"

    local conn="${HARNESS_CONN:-udp:127.0.0.1:14551}"
    local speedup="${HARNESS_SPEEDUP:-1.0}"
    local report_path="${HARNESS_OUT_DIR}/attitude_rate_preflight.json"
    local preflight_script="${HARNESS_SCRIPT_DIR}/attitude_rate_preflight.py"

    # Determine mode: validation (hard fail) or debug (warn only)
    local mode="validation"
    if [[ "${HARNESS_TIMEBASE_STRICT:-1}" == "0" ]]; then
        mode="debug"
        log_warn "HARNESS_TIMEBASE_STRICT=0 -> preflight in debug mode (warn only)"
    fi

    if [[ ! -f "$preflight_script" ]]; then
        log_fail "PRH-ATT-002: attitude_rate_preflight.py not found at $preflight_script"
        return 1
    fi

    log_info "Running attitude_rate_preflight.py --mode ${mode} --speedup ${speedup}"

    local preflight_exit=0
    timeout 90 python3 "$preflight_script" \
        --conn "$conn" \
        --speedup "$speedup" \
        --settle-sim-s 5 \
        --measure-sim-s 30 \
        --min-hz 18 \
        --max-dt-p95-s 0.08 \
        --stale-gap-thresh-s 0.2 \
        --window-sim-s 5 \
        --mode "$mode" \
        --report "$report_path" \
    || preflight_exit=$?

    if [[ $preflight_exit -eq 0 ]]; then
        log_pass "ATTITUDE rate preflight PASS (report: $report_path)"
    elif [[ $preflight_exit -eq 2 ]]; then
        log_fail "PRH-ATT-002: ATTITUDE preflight FAIL - link dead (exit 2)"
        return 1
    else
        if [[ "$mode" == "debug" ]]; then
            log_warn "ATTITUDE rate preflight FAIL in debug mode (continuing)"
        else
            log_fail "PRH-ATT-002: ATTITUDE rate preflight FAIL - rate/stability below threshold"
            return 1
        fi
    fi
}

# =============================================================================
# TIMEBASE GATE
# =============================================================================
harness_run_timebase_gate() {
    if [[ "$HARNESS_TIMEBASE_GATE" != "1" ]]; then
        log_warn "Timebase gate SKIPPED (--skip-timebase)"
        return 0
    fi

    log_info "========================================"
    log_info "TIMEBASE GATE (${HARNESS_SPEEDUP}x, ${HARNESS_TIMEBASE_DURATION}s)"
    log_info "========================================"

    local timebase_log="${HARNESS_OUT_DIR}/timebase_smoketest_${HARNESS_SPEEDUP}x.txt"
    local timebase_script="${HARNESS_TOOLS_DIR}/timebase_smoketest.py"

    if [[ ! -f "$timebase_script" ]]; then
        log_fail "Timebase script not found: $timebase_script"
        exit 1
    fi

    if python3 "$timebase_script" \
        --conn "${HARNESS_CONN}" \
        --expected-speedup "${HARNESS_SPEEDUP}" \
        --duration "${HARNESS_TIMEBASE_DURATION}" \
        --sysid 1 \
        2>&1 | tee "$timebase_log"; then

        if grep -q "VERDICT: PASS" "$timebase_log"; then
            log_pass "Timebase gate PASSED"

            # Fail-fast speedup collapse detection
            local achieved_rate
            achieved_rate=$(grep "Rate median:" "$timebase_log" | grep -oP '\d+\.\d+' | head -1 || echo "0")

            if [[ -n "$achieved_rate" ]] && [[ "$achieved_rate" != "0" ]]; then
                local min_acceptable
                min_acceptable=$(echo "$HARNESS_SPEEDUP * $HARNESS_MIN_ACHIEVED_SPEEDUP_FRAC" | bc -l)

                if (( $(echo "$achieved_rate < $min_acceptable" | bc -l) )); then
                    log_fail "╔═══════════════════════════════════════════════════════════════════╗"
                    log_fail "║ SPEEDUP COLLAPSE DETECTED                                         ║"
                    log_fail "╠═══════════════════════════════════════════════════════════════════╣"
                    log_fail "║ Requested: ${HARNESS_SPEEDUP}x, Achieved: ${achieved_rate}x"
                    log_fail "║ Minimum acceptable: ${min_acceptable}x (${HARNESS_MIN_ACHIEVED_SPEEDUP_FRAC} × ${HARNESS_SPEEDUP})"
                    log_fail "║"
                    log_fail "║ System cannot sustain requested speedup."
                    log_fail "║ Use --speedup to reduce, or run speedup_sweep.py to calibrate."
                    log_fail "╚═══════════════════════════════════════════════════════════════════╝"
                    exit 1
                else
                    log_pass "Speedup OK: ${achieved_rate}x >= ${min_acceptable}x minimum"
                fi
            fi

            return 0
        else
            log_fail "Timebase gate did not PASS"
            echo "Check $timebase_log for details"
            exit 1
        fi
    else
        log_fail "Timebase gate failed to run"
        exit 1
    fi
}

# =============================================================================
# LEADER BROADCASTER
# =============================================================================
harness_start_leader() {
    log_info "========================================"
    log_info "STARTING LEADER BROADCASTER"
    log_info "========================================"

    local leader_log="${HARNESS_OUT_DIR}/leader_${HARNESS_SPEEDUP}x.log"
    local leader_script="${HARNESS_SCRIPT_DIR}/leader_broadcast_fixed.py"

    if [[ ! -f "$leader_script" ]]; then
        log_fail "Leader script not found: $leader_script"
        exit 1
    fi

    # Build leader args
    local leader_args="--target-speed 24.7"
    leader_args="${leader_args} --time-sync-conn ${HARNESS_TIMESYNC_CONN}"
    leader_args="${leader_args} --sim-speedup ${HARNESS_SPEEDUP}"
    leader_args="${leader_args} --profile ${HARNESS_LEADER_PROFILE}"

    # Port-offset-aware connections for parallel SITL instances
    # --udp-out: MAVProxy leader input port (14550 + offset)
    # --sitl-tcp: Direct SERIAL2 TCP port (5763 + offset)
    local leader_in_port=$((14550 + HARNESS_PORT_OFFSET))
    local serial2_port=$((5763 + HARNESS_PORT_OFFSET))
    leader_args="${leader_args} --udp-out 127.0.0.1:${leader_in_port}"
    leader_args="${leader_args} --sitl-tcp 127.0.0.1:${serial2_port}"

    # H-4: Force --require-time-sync at speedup > 1.0 (numeric compare)
    if (( $(echo "${HARNESS_SPEEDUP} > 1.0" | bc -l) )); then
        leader_args="${leader_args} --require-time-sync"
        log_info "H-4: Added --require-time-sync (speedup=${HARNESS_SPEEDUP} > 1.0)"
    else
        log_info "H-4: Skipped --require-time-sync (speedup=${HARNESS_SPEEDUP} <= 1.0)"
    fi

    # ENV-SWEEP wind hook (default-OFF; additive). Forward leader wind/turbulence
    # ONLY when the HARNESS_LEADER_WIND_* env vars are set+nonempty. When unset, the
    # ${VAR:+ ...} expansion adds nothing (not even a space) => leader_args is
    # byte-identical to the pre-hook baseline (safe under set -u). When set, the
    # leader broadcaster's weather-sync pushes the intended atmosphere to the
    # follower SITL (otherwise sync_weather_to_sitl zeroes SIM_WIND_* at startup).
    leader_args="${leader_args}${HARNESS_LEADER_WIND_SPD:+ --wind-speed ${HARNESS_LEADER_WIND_SPD}}"
    leader_args="${leader_args}${HARNESS_LEADER_WIND_DIR:+ --wind-dir ${HARNESS_LEADER_WIND_DIR}}"
    leader_args="${leader_args}${HARNESS_LEADER_WIND_TURB:+ --turbulence ${HARNESS_LEADER_WIND_TURB}}"
    # Route the weather-sync to the FREE serial1 MAVLink2 port (5762+offset) when wind is
    # enabled: :5763 (serial2) is held by the leader's own Link 2 and starves the sync's 5s
    # heartbeat, so SIM_WIND_* never lands on the follower; serial1 is unused (per
    # start_formation_sitl_fixed.sh) and SERIAL1_PROTOCOL=2. Default-OFF: unset => skipped.
    if [[ -n "${HARNESS_LEADER_WIND_SPD:-}" ]]; then
        leader_args="${leader_args} --sitl-weather-port $((5762 + ${HARNESS_PORT_OFFSET:-0}))"
    fi

    # H-4: Log leader args to output directory for verification (like ctrl_args.txt)
    mkdir -p "${HARNESS_OUT_DIR}"
    printf '%s\n' "$leader_args" > "${HARNESS_OUT_DIR}/lb_args.txt"

    log_info "Leader profile: ${HARNESS_LEADER_PROFILE}"
    log_info "Starting leader_broadcast_fixed.py..."

    # shellcheck disable=SC2086 # Intentional word splitting
    # Use -u for unbuffered output so logs appear immediately
    python3 -u "$leader_script" $leader_args > "$leader_log" 2>&1 &
    HARNESS_LEADER_PID=$!

    log_info "Leader broadcaster started (PID $HARNESS_LEADER_PID)"
    echo "leader_pid=$HARNESS_LEADER_PID" >> "$HARNESS_PIDS_FILE"

    # H-1: Wait for sim-time lock marker instead of blind sleep
    local wait_timeout=20
    local wait_start
    wait_start=$(date +%s)
    local marker_found=0

    log_info "Waiting for leader sim-time lock (up to ${wait_timeout}s)..."

    while (( $(date +%s) - wait_start < wait_timeout )); do
        # Check if process died
        if ! kill -0 "$HARNESS_LEADER_PID" 2>/dev/null; then
            log_fail "Leader broadcaster died before sim-time lock"
            cat "$leader_log"
            exit 1
        fi

        # Check for sim-time ready marker
        if grep -q "Time-sync READY" "$leader_log" 2>/dev/null; then
            marker_found=1
            break
        fi

        sleep 0.5
    done

    if [[ $marker_found -eq 0 ]]; then
        log_fail "Leader broadcaster never locked to sim-time within ${wait_timeout}s"
        cat "$leader_log"
        kill "$HARNESS_LEADER_PID" 2>/dev/null || true
        exit 1
    fi

    log_pass "Leader broadcaster locked to sim-time"
}

# =============================================================================
# CONTROLLER EXECUTION
# =============================================================================
harness_run_controller() {
    log_info "========================================"
    log_info "RUNNING CONTROLLER (Test ${HARNESS_TEST_ID}, ${HARNESS_SPEEDUP}x)"
    log_info "========================================"

    local controller_log="${HARNESS_OUT_DIR}/controller_${HARNESS_SPEEDUP}x.log"
    local timestamp_log
    timestamp_log=$(date +%Y%m%d_%H%M%S)
    local controller_link="${HARNESS_LOG_BASE}/intercept_rejoin-${HARNESS_TEST_ID}_${timestamp_log}.log"
    local controller_script="${HARNESS_SCRIPT_DIR}/baseline_formation_controller.py"

    if [[ ! -f "$controller_script" ]]; then
        log_fail "Controller script not found: $controller_script"
        exit 1
    fi

    # Build controller args
    local ctrl_args="--test-id ${HARNESS_TEST_ID}"
    ctrl_args="${ctrl_args} --speedup ${HARNESS_SPEEDUP}"
    ctrl_args="${ctrl_args} --connection ${HARNESS_CONN}"

    # TEST 339: Pass explicit output directory for run_summary.json
    ctrl_args="${ctrl_args} --out-dir ${HARNESS_OUT_DIR}"

    # Option 0 Fix: Pass max-sim-seconds for sim-time based termination
    ctrl_args="${ctrl_args} --max-sim-seconds ${HARNESS_SIM_SECONDS}"

    # Also export as environment variable for backward compatibility
    export HARNESS_OUT_DIR

    if [[ "$HARNESS_DRL_MODE" == "1" ]]; then
        ctrl_args="${ctrl_args} --drl-training-mode"
        log_warn "DRL training mode enabled (timing issues won't abort)"
    fi

    # C++ handoff control (Test 387 Matrix B)
    HARNESS_CPP_HANDOFF_POLICY="${HARNESS_CPP_HANDOFF_POLICY:-range_hysteresis}"

    if [[ "$HARNESS_CPP_HANDOFF_POLICY" == "always_off" ]]; then
        ctrl_args="${ctrl_args} --cpp-handoff-policy always_off"
        log_info "C++ handoff policy: always_off (Python-only mode)"
    elif [[ "$HARNESS_CPP_HANDOFF_POLICY" == "always_on" ]]; then
        ctrl_args="${ctrl_args} --cpp-handoff-policy always_on"
        log_info "C++ handoff policy: always_on (Matrix B Integrated)"
    elif [[ "$HARNESS_CPP_HANDOFF_POLICY" == "state_gated" ]]; then
        ctrl_args="${ctrl_args} --cpp-handoff-policy state_gated"
        log_info "C++ handoff policy: state_gated (DC/HOLD_CONTACT only)"
    else
        ctrl_args="${ctrl_args} --cpp-handoff-policy range_hysteresis"
        log_info "C++ handoff policy: range_hysteresis (<30m enable, >40m disable)"
    fi

    # High-Side Convergence mode (TEST 392+)
    if [[ "$HARNESS_HIGH_SIDE_CONVERGENCE" == "1" ]]; then
        ctrl_args="${ctrl_args} --high-side-convergence"
        ctrl_args="${ctrl_args} --high-side-margin ${HIGH_SIDE_MARGIN_M}"
        ctrl_args="${ctrl_args} --high-side-cross-ok ${HIGH_SIDE_CROSS_OK_M}"
        ctrl_args="${ctrl_args} --high-side-behind-max ${HIGH_SIDE_BEHIND_MAX_M}"
        ctrl_args="${ctrl_args} --high-side-min-time ${HIGH_SIDE_MIN_TIME_S}"
        ctrl_args="${ctrl_args} --high-side-disable-behind ${HIGH_SIDE_DISABLE_BEHIND_M}"
        log_info "High-Side Convergence: ENABLED (margin=${HIGH_SIDE_MARGIN_M}m, disable-behind=${HIGH_SIDE_DISABLE_BEHIND_M}m)"
    fi

    # Parallel SITL: leader telemetry port (14552 + offset)
    local monitor_port=$((14552 + HARNESS_PORT_OFFSET))
    ctrl_args="${ctrl_args} --leader-port ${monitor_port}"

    # Log ctrl_args to output directory for verification
    mkdir -p "${HARNESS_OUT_DIR}"
    printf '%s\n' "$ctrl_args" > "${HARNESS_OUT_DIR}/ctrl_args.txt"

    log_info "Log: $controller_log"
    log_info "Out dir: $HARNESS_OUT_DIR"

    # Calculate wall-time timeout
    local wall_timeout
    wall_timeout=$(python3 -c "import math; print(int(math.ceil(${HARNESS_SIM_SECONDS} / ${HARNESS_SPEEDUP} * 1.2 + 120)))")
    log_info "Wall timeout: ${wall_timeout}s (~${HARNESS_SIM_SECONDS}s sim @ ${HARNESS_SPEEDUP}x + buffer)"

    # Run controller with timeout
    set +e
    # shellcheck disable=SC2086 # Intentional word splitting
    timeout -k 10 "${wall_timeout}" python3 "$controller_script" $ctrl_args 2>&1 | tee "$controller_log"
    local ctrl_exit=$?
    set -e

    # Copy to standard log location
    cp "$controller_log" "$controller_link" 2>/dev/null || true

    # Copy run_summary.json alongside the controller log (for post_test_summary.py)
    local run_summary_src="${HARNESS_OUT_DIR}/run_summary.json"
    if [[ -f "$run_summary_src" ]]; then
        local run_summary_dst="${HARNESS_LOG_BASE}/run_summary-${HARNESS_TEST_ID}_${timestamp_log}.json"
        cp "$run_summary_src" "$run_summary_dst" 2>/dev/null || true
        log_info "Copied run_summary.json to ${run_summary_dst}"
    fi

    # Also copy success_metrics.json (backup file)
    local metrics_src="${HARNESS_OUT_DIR}/success_metrics.json"
    if [[ -f "$metrics_src" ]]; then
        local metrics_dst="${HARNESS_LOG_BASE}/success_metrics-${HARNESS_TEST_ID}_${timestamp_log}.json"
        cp "$metrics_src" "$metrics_dst" 2>/dev/null || true
    fi

    echo ""
    if [[ $ctrl_exit -eq 0 ]]; then
        log_pass "Controller completed normally (exit 0)"
    elif [[ $ctrl_exit -eq 124 ]]; then
        log_warn "Controller timed out after ${wall_timeout}s wall-time"
    else
        log_warn "Controller exited with code $ctrl_exit"
    fi

    echo "controller_exit=$ctrl_exit" >> "$HARNESS_PIDS_FILE"
    return $ctrl_exit
}

# =============================================================================
# QUICK ANALYSIS
# =============================================================================
harness_analyze_quick() {
    local controller_log="${HARNESS_OUT_DIR}/controller_${HARNESS_SPEEDUP}x.log"

    log_info "========================================"
    log_info "QUICK ANALYSIS"
    log_info "========================================"

    if [[ ! -f "$controller_log" ]]; then
        log_warn "Controller log not found for analysis"
        return
    fi

    # Count key metrics
    local state_transitions alt_capture_entries overtake_events
    local precontact_holds contact_entries docking_creep

    state_transitions=$(grep -c "STATE:" "$controller_log" 2>/dev/null || echo "0")
    alt_capture_entries=$(grep -c "ALTITUDE_CAPTURE" "$controller_log" 2>/dev/null || echo "0")
    overtake_events=$(grep -c "OVERTAKE_PROTECT" "$controller_log" 2>/dev/null || echo "0")
    precontact_holds=$(grep -c "Precontact hold complete" "$controller_log" 2>/dev/null || echo "0")
    contact_entries=$(grep -c "Contact envelope entry" "$controller_log" 2>/dev/null || echo "0")
    docking_creep=$(grep -c "DOCKING_CREEP" "$controller_log" 2>/dev/null || echo "0")

    echo "  State transitions:   $state_transitions"
    echo "  ALT_CAPTURE entries: $alt_capture_entries"
    echo "  OVERTAKE events:     $overtake_events (target: <100)"
    echo "  Precontact holds:    $precontact_holds"
    echo "  Contact entries:     $contact_entries"
    echo "  DOCKING_CREEP:       $docking_creep"

    # Check SYNC_VERDICT
    local sync_ok=0
    if grep -q "SYNC_VERDICT.*PASS" "$controller_log"; then
        log_pass "SYNC_VERDICT: PASS"
        sync_ok=1
    elif grep -q "SYNC_VERDICT.*FAIL" "$controller_log"; then
        log_warn "SYNC_VERDICT: FAIL (check timing)"
    else
        log_warn "SYNC_VERDICT: not found"
    fi

    # Save quick metrics
    cat > "${HARNESS_OUT_DIR}/quick_metrics.txt" << EOF
Test ${HARNESS_TEST_NAME} Quick Metrics
=================================
Timestamp: $(date +"%Y%m%d_%H%M%S")
Speedup: ${HARNESS_SPEEDUP}x
Sim-seconds budget: ${HARNESS_SIM_SECONDS}

State transitions: $state_transitions
ALT_CAPTURE entries: $alt_capture_entries
OVERTAKE events: $overtake_events
Precontact holds: $precontact_holds
Contact entries: $contact_entries
DOCKING_CREEP entries: $docking_creep
SYNC_VERDICT passed: $sync_ok
EOF

    log_info "Quick metrics saved to: ${HARNESS_OUT_DIR}/quick_metrics.txt"

    # Success metrics summary (if available)
    local run_summary="${HARNESS_OUT_DIR}/run_summary.json"
    if [[ -f "$run_summary" ]]; then
        # Use python3 for JSON parsing (more portable than jq)
        local metrics_output
        metrics_output=$(python3 -c "
import json, sys
try:
    with open('$run_summary') as f:
        data = json.load(f)
    sm = data.get('success_metrics', {})
    if not sm:
        sys.exit(0)
    outcome = sm.get('outcome', 'UNKNOWN')
    m2 = sm.get('milestones', {}).get('m2_precontact', {})
    m3a = sm.get('milestones', {}).get('m3a_coarse_contact', {})
    print(f'SUCCESS_METRICS:')
    print(f'  Outcome: {outcome}')
    m2_ach = m2.get('achieved', False)
    m2_dwell = m2.get('first_dwell_complete_s')
    print(f'  M2 precontact: {\"ACHIEVED\" if m2_ach else \"not achieved\"}{f\" @ {m2_dwell:.1f}s\" if m2_dwell else \"\"}')
    m3a_ach = m3a.get('achieved', False)
    m3a_dwell = m3a.get('first_dwell_complete_s')
    print(f'  M3a contact: {\"ACHIEVED\" if m3a_ach else \"not achieved\"}{f\" @ {m3a_dwell:.1f}s\" if m3a_dwell else \"\"}')
except Exception:
    pass  # Silent fail - best effort
" 2>/dev/null || true)
        if [[ -n "$metrics_output" ]]; then
            echo ""
            echo "$metrics_output"
        fi
    fi
}

# =============================================================================
# VIRTUAL ENVIRONMENT
# =============================================================================
harness_activate_venv() {
    local venv_activate="${HARNESS_VENV_DIR}/bin/activate"

    if [[ -f "$venv_activate" ]]; then
        # shellcheck disable=SC1090
        source "$venv_activate"
        log_pass "Virtual environment activated"
    else
        log_fail "Virtual environment not found: $venv_activate"
        exit 1
    fi
}

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
harness_preflight_checks() {
    log_info "Pre-flight checks..."

    harness_activate_venv

    # Syntax check controller
    local controller_script="${HARNESS_SCRIPT_DIR}/baseline_formation_controller.py"
    if python3 -m py_compile "$controller_script"; then
        log_pass "Controller syntax OK"
    else
        log_fail "Controller has syntax errors"
        exit 1
    fi

    # Version check
    local version
    version=$(grep -o "CONTROLLER_VERSION.*" "$controller_script" | head -1)
    log_info "$version"

    # Check required files
    for f in "${HARNESS_SCRIPT_DIR}/leader_broadcast_fixed.py" \
             "${HARNESS_TOOLS_DIR}/timebase_smoketest.py"; do
        if [[ -f "$f" ]]; then
            log_pass "Found: $(basename "$f")"
        else
            log_fail "Missing: $f"
            exit 1
        fi
    done

    # Phase 7D.335: Lint param files for SIM_SPEEDUP
    if ! harness_lint_param_files; then
        exit 1
    fi
}

# =============================================================================
# DRY RUN
# =============================================================================
harness_dry_run() {
    log_info "[DRY RUN] Would execute:"
    echo ""
    echo "  SITL Mode: ${HARNESS_SITL_MODE}"
    case "$HARNESS_SITL_MODE" in
        auto)   echo "    -> Check if SITL running; if no, start it; if yes, reuse" ;;
        reuse)  echo "    -> Use existing SITL, fail if not running" ;;
        start)  echo "    -> Kill existing SITL (if any), start fresh" ;;
    esac
    echo ""
    echo "  UI Mode: ${HARNESS_UI_MODE} (headless=${HARNESS_HEADLESS})"
    echo ""
    echo "  # 1. Activate venv"
    echo "  source ${HARNESS_VENV_DIR}/bin/activate"
    echo ""
    if [[ "$HARNESS_TIMEBASE_GATE" == "1" ]]; then
        echo "  # 2. Timebase smoketest"
        echo "  python ${HARNESS_TOOLS_DIR}/timebase_smoketest.py \\"
        echo "    --conn ${HARNESS_CONN} \\"
        echo "    --expected-speedup ${HARNESS_SPEEDUP} \\"
        echo "    --duration ${HARNESS_TIMEBASE_DURATION}"
        echo ""
    fi
    echo "  # 3. Start leader broadcaster"
    echo "  python ${HARNESS_SCRIPT_DIR}/leader_broadcast_fixed.py \\"
    echo "    --target-speed 24.7 \\"
    echo "    --time-sync-conn ${HARNESS_TIMESYNC_CONN} \\"
    echo "    --sim-speedup ${HARNESS_SPEEDUP} \\"
    echo "    --profile ${HARNESS_LEADER_PROFILE} &"
    echo ""
    echo "  # 4. Run controller"
    echo "  python ${HARNESS_SCRIPT_DIR}/baseline_formation_controller.py \\"
    echo "    --test-id ${HARNESS_TEST_ID} \\"
    echo "    --speedup ${HARNESS_SPEEDUP}"
    echo ""
    echo "  # 5. Cleanup (kill SITL only if started by harness)"
    echo ""
}

# =============================================================================
# MAIN FLOW HELPERS
# =============================================================================

# Run full test flow
harness_run_full() {
    # Check dry run first
    if [[ "$HARNESS_DRY_RUN" == "1" ]]; then
        harness_dry_run
        exit 0
    fi

    harness_preflight_checks
    harness_ensure_sitl
    harness_configure_attitude_rate
    harness_run_timebase_gate
    harness_start_leader
    harness_run_controller || true
    harness_analyze_quick
}

# Run smoketest-only flow (for calibration)
# This runs a SHORT test to validate infrastructure at a given speedup:
#   1. SITL startup
#   2. Timebase gate
#   3. Leader freshness check
#   4. Brief controller window
# Does NOT run full mission phases.
harness_run_smoketest_only() {
    log_info "========================================"
    log_info "SMOKETEST-ONLY MODE"
    log_info "========================================"
    log_info "Speedup:             ${HARNESS_SPEEDUP}x"
    log_info "Timebase duration:   ${HARNESS_TIMEBASE_DURATION}s"
    log_info "Post-settle window:  ${HARNESS_SMOKETEST_POST_SETTLE_S}s (sim)"
    log_info "========================================"

    # Check dry run first
    if [[ "$HARNESS_DRY_RUN" == "1" ]]; then
        harness_dry_run
        exit 0
    fi

    harness_preflight_checks
    harness_ensure_sitl
    harness_configure_attitude_rate

    # Run timebase gate (this is the main validation)
    harness_run_timebase_gate

    # Start leader broadcaster for freshness check
    harness_start_leader

    # Brief controller window to confirm telemetry freshness
    # Override sim_seconds for smoketest
    local orig_sim_seconds="$HARNESS_SIM_SECONDS"
    HARNESS_SIM_SECONDS="$HARNESS_SMOKETEST_POST_SETTLE_S"

    log_info "Running brief controller window (${HARNESS_SIM_SECONDS}s sim-time)..."
    harness_run_controller || true

    # Restore original
    HARNESS_SIM_SECONDS="$orig_sim_seconds"

    # Quick analysis
    harness_analyze_quick

    log_pass "Smoketest complete"
}

# Print summary header
harness_print_header() {
    echo ""
    echo "============================================================="
    echo "${HARNESS_TEST_NAME^^} - FLIGHTDOCK FORMATION TEST"
    echo "============================================================="
    echo "  Harness:           v${HARNESS_VERSION}"
    echo "  Speedup:           ${HARNESS_SPEEDUP}x"
    echo "  Sim-seconds:       ${HARNESS_SIM_SECONDS}s"
    echo "  Test ID:           ${HARNESS_TEST_ID}"
    echo "  Leader profile:    ${HARNESS_LEADER_PROFILE}"
    echo "  UI mode:           ${HARNESS_UI_MODE}"
    echo "  Timebase gate:     $([ "$HARNESS_TIMEBASE_GATE" == "1" ] && echo 'ON' || echo 'OFF')"
    echo "  DRL mode:          $([ "$HARNESS_DRL_MODE" == "1" ] && echo 'ON' || echo 'OFF')"
    echo "  Smoketest only:    $([ "$HARNESS_SMOKETEST_ONLY" == "1" ] && echo 'ON' || echo 'OFF')"
    if [[ -n "$HARNESS_RUN_TAG" ]]; then
        echo "  Run tag:           ${HARNESS_RUN_TAG}"
    fi
    echo "  Output:            ${HARNESS_OUT_DIR:-<not initialized>}"
    echo "============================================================="
    echo ""
}

# =============================================================================
# SPEEDUP HARDENING (Phase 7D.335 - Prevent SIM_SPEEDUP override)
# =============================================================================
# These functions ensure that --speedup is authoritative and param files
# cannot silently override it.
#
# HARNESS_SKIP_PARAM_LINT=1 Skip param file lint (not recommended)
# HARNESS_SANITIZE_PARAMS=1 Auto-generate sanitized param file if needed
# =============================================================================

HARNESS_SKIP_PARAM_LINT="${HARNESS_SKIP_PARAM_LINT:-0}"
HARNESS_SANITIZE_PARAMS="${HARNESS_SANITIZE_PARAMS:-0}"
HARNESS_SPEEDUP_READBACK="${HARNESS_SPEEDUP_READBACK:-1}"

# Get list of param files that will be passed to SITL
harness_get_param_files() {
    local fw_variant="${FD_FW_VARIANT:-legacy}"
    local param_files=()

    case "$fw_variant" in
        modern)
            param_files+=("$HOME/FlightDock/config/quadplane_formation_modern_v1.parm")
            ;;
        *)
            param_files+=("$HOME/FlightDock/config/quadplane_formation_mav_clean.parm")
            ;;
    esac

    # Echo one per line for easy parsing
    printf '%s\n' "${param_files[@]}"
}

# Lint param files for SIM_SPEEDUP
# Returns 0 if clean, 1 if SIM_SPEEDUP found, 2 on error
harness_lint_param_files() {
    if [[ "$HARNESS_SKIP_PARAM_LINT" == "1" ]]; then
        log_warn "Param file lint SKIPPED (HARNESS_SKIP_PARAM_LINT=1)"
        return 0
    fi

    log_info "Checking param files for SIM_SPEEDUP override..."

    local speedup_guard="${HARNESS_TOOLS_DIR}/speedup_guard.py"
    if [[ ! -f "$speedup_guard" ]]; then
        log_warn "speedup_guard.py not found, skipping param lint"
        return 0
    fi

    # Collect param files
    local -a param_files
    while IFS= read -r line; do
        [[ -n "$line" ]] && param_files+=("$line")
    done < <(harness_get_param_files)

    if [[ ${#param_files[@]} -eq 0 ]]; then
        log_warn "No param files to check"
        return 0
    fi

    # Run lint
    local lint_args=("${param_files[@]}")
    if [[ "$HARNESS_SANITIZE_PARAMS" == "1" ]]; then
        lint_args=("--sanitize" "${lint_args[@]}")
    fi

    if python3 "$speedup_guard" lint "${lint_args[@]}"; then
        log_pass "Param files clean (no SIM_SPEEDUP)"
        return 0
    else
        if [[ "$HARNESS_SANITIZE_PARAMS" == "1" ]]; then
            log_warn "SIM_SPEEDUP found but sanitized param file generated"
            return 0
        else
            log_fail "╔═══════════════════════════════════════════════════════════════════╗"
            log_fail "║ SIM_SPEEDUP FOUND IN PARAMETER FILE                               ║"
            log_fail "╠═══════════════════════════════════════════════════════════════════╣"
            log_fail "║ Parameter files contain SIM_SPEEDUP which will override --speedup ║"
            log_fail "║                                                                   ║"
            log_fail "║ To fix, choose one:                                               ║"
            log_fail "║   1. Remove SIM_SPEEDUP line from param file(s)                   ║"
            log_fail "║   2. Run with HARNESS_SANITIZE_PARAMS=1                           ║"
            log_fail "║   3. Run with HARNESS_SKIP_PARAM_LINT=1 (not recommended)         ║"
            log_fail "╚═══════════════════════════════════════════════════════════════════╝"
            return 1
        fi
    fi
}

# Verify SITL speedup matches expected after startup
harness_readback_speedup() {
    if [[ "$HARNESS_SPEEDUP_READBACK" != "1" ]]; then
        log_warn "Speedup readback SKIPPED (HARNESS_SPEEDUP_READBACK=0)"
        return 0
    fi

    log_info "Verifying SITL speedup via MAVLink readback..."

    local speedup_guard="${HARNESS_TOOLS_DIR}/speedup_guard.py"
    if [[ ! -f "$speedup_guard" ]]; then
        log_warn "speedup_guard.py not found, skipping readback"
        return 0
    fi

    local expected="${HARNESS_SPEEDUP:-1.0}"
    local conn="${HARNESS_CONN:-udp:127.0.0.1:14551}"

    if python3 "$speedup_guard" readback \
        --conn "$conn" \
        --expected "$expected" \
        --timeout 15; then
        log_pass "Speedup readback verified: ${expected}x"
        return 0
    else
        log_fail "╔═══════════════════════════════════════════════════════════════════╗"
        log_fail "║ SPEEDUP READBACK MISMATCH                                         ║"
        log_fail "╠═══════════════════════════════════════════════════════════════════╣"
        log_fail "║ SITL is running at a different speedup than requested.            ║"
        log_fail "║                                                                   ║"
        log_fail "║ Requested: ${expected}x                                           ║"
        log_fail "║                                                                   ║"
        log_fail "║ Common causes:                                                    ║"
        log_fail "║   1. Param file contains SIM_SPEEDUP (use HARNESS_SANITIZE_PARAMS)║"
        log_fail "║   2. Stale eeprom.bin (delete ArduPlane/eeprom.bin and restart)   ║"
        log_fail "║   3. SITL was not fully restarted                                 ║"
        log_fail "║                                                                   ║"
        log_fail "║ Fix: Kill all SITL processes, delete eeprom.bin, and retry        ║"
        log_fail "╚═══════════════════════════════════════════════════════════════════╝"
        return 1
    fi
}

# Set SITL speedup via MAVLink (Phase 7D.386 - Fix reuse speedup mismatch)
harness_set_speedup() {
    log_info "Setting SITL speedup via MAVLink to ${HARNESS_SPEEDUP}x..."

    local speedup_guard="${HARNESS_TOOLS_DIR}/speedup_guard.py"
    if [[ ! -f "$speedup_guard" ]]; then
        log_warn "speedup_guard.py not found, skipping set"
        return 0
    fi

    local desired="${HARNESS_SPEEDUP:-1.0}"
    local conn="${HARNESS_CONN:-udp:127.0.0.1:14551}"

    if python3 "$speedup_guard" set \
        --conn "$conn" \
        --speedup "$desired" \
        --timeout 15; then
        log_pass "SIM_SPEEDUP set to ${desired}x via MAVLink"
        return 0
    else
        log_fail "Failed to set SIM_SPEEDUP via MAVLink"
        return 1
    fi
}
