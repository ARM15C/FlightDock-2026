#!/bin/bash
# =============================================================================
# FlightDock Formation SITL Startup Script (Phase 7B Baseline + Speed Hook)
# =============================================================================
# Starts ArduPlane SITL with formation controller and correct MAVLink routing
# Default: 4.7-dev fork (phase7b-baseline-speed-hook branch) with Option-3FORM envelope
#
# UI MODES:
#   HEADLESS=1 No GUI windows (for HPC/CI)
#   HARNESS_UI_MODE=none Equivalent to HEADLESS=1
#   HARNESS_UI_MODE=map Map window only
#   HARNESS_UI_MODE=console Console window only
#   HARNESS_UI_MODE=both Map + console windows (default for interactive)
#
# FLAGS:
#   --headless Same as HEADLESS=1
#   --ui none|map|console|both Set UI mode
#   --mavproxy-map Enable map window
#   --mavproxy-console Enable console window
#   --force-ui Force UI even at high speedup (will clamp speedup)
#
# GUI SPEEDUP POLICY:
#   When UI is requested and speedup > GUI_MAX_SPEEDUP (3.0x):
#   - By default: clamp speedup to GUI_MAX_SPEEDUP and warn
#   - With --force-ui: allow UI but still clamp and warn loudly
#   - With --headless: always disable UI (no clamping needed)
#
# =============================================================================

set -euo pipefail  # Exit on error, unset var, or pipe failure

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
# Default speedup (can be overridden by --speedup flag)
SITL_SPEEDUP="${SITL_SPEEDUP:-1}"
FORCE_UI="${FORCE_UI:-0}"

# Direct mode: bypass sim_vehicle.py (fixes WSL2 terminal spawning issues)
DIRECT_MODE="${DIRECT_MODE:-0}"

# Parameter wipe: delete eeprom.bin/mav.parm before starting (default: from env or 0)
WIPE_PARAMS="${HARNESS_WIPE_PARAMS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless)
            HEADLESS=1
            shift
            ;;
        --speedup)
            SITL_SPEEDUP="$2"
            shift 2
            ;;
        --ui)
            HARNESS_UI_MODE="$2"
            shift 2
            ;;
        --mavproxy-map)
            HARNESS_UI_MODE="${HARNESS_UI_MODE:-map}"
            [[ "$HARNESS_UI_MODE" == "console" ]] && HARNESS_UI_MODE="both"
            shift
            ;;
        --mavproxy-console)
            HARNESS_UI_MODE="${HARNESS_UI_MODE:-console}"
            [[ "$HARNESS_UI_MODE" == "map" ]] && HARNESS_UI_MODE="both"
            shift
            ;;
        --force-ui)
            FORCE_UI=1
            shift
            ;;
        --direct)
            # Direct mode: run ArduPlane and MAVProxy directly (WSL2 compatible)
            DIRECT_MODE=1
            shift
            ;;
        --wipe|--clean-params)
            # Wipe SITL parameters before starting
            WIPE_PARAMS=1
            shift
            ;;
        --no-wipe)
            WIPE_PARAMS=0
            shift
            ;;
        *)
            # Ignore unknown args
            shift
            ;;
    esac
done

# Apply HARNESS_UI_MODE to HEADLESS if set
if [[ -n "${HARNESS_UI_MODE:-}" ]]; then
    case "$HARNESS_UI_MODE" in
        none)
            HEADLESS=1
            ;;
        map|console|both)
            HEADLESS=0
            ;;
    esac
fi

echo "Starting FlightDock Formation SITL..."
echo ""

# Activate Python virtual environment (required for MAVProxy)
echo "Activating virtual environment..."
# shellcheck source=/dev/null
export PYTHONPATH="${PYTHONPATH:-}"   # T-62: venv activate:69 derefs $PYTHONPATH unguarded; set -u would abort here
source ~/FlightDock/flightdock_native_venv/bin/activate || {
    echo "ERROR: Could not activate virtual environment"
    echo "Expected at: ~/FlightDock/flightdock_native_venv/bin/activate"
    exit 1
}
echo "✓ Virtual environment activated"
echo ""

# ========== FIRMWARE VARIANT SELECTION ==========
# FD_FW_VARIANT environment variable controls which ArduPlane build to use:
#   legacy (default) → ~/FlightDock/ardupilot_formation (4.7-dev FlightDock fork)
#   modern → ~/ardupilot-plane-modern (upstream Plane-stable)
# ================================================

# Default to legacy if FD_FW_VARIANT is not set
FD_FW_VARIANT="${FD_FW_VARIANT:-legacy}"

case "$FD_FW_VARIANT" in
    legacy)
        ARDUPILOT_DIR="$HOME/FlightDock/ardupilot_formation"
        FIRMWARE_DESC="legacy (4.7-dev FlightDock fork)"
        ;;
    modern)
        ARDUPILOT_DIR="$HOME/ardupilot-plane-modern"
        FIRMWARE_DESC="modern (upstream Plane-stable)"
        ;;
    *)
        echo "ERROR: Invalid FD_FW_VARIANT='$FD_FW_VARIANT'"
        echo "Valid options: legacy, modern"
        exit 1
        ;;
esac

echo "═══════════════════════════════════════════════════════════════════"
echo "  Using firmware variant: $FIRMWARE_DESC"
echo "  ArduPlane directory: $ARDUPILOT_DIR"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# ========== DEPLOYMENT HARDENING (Test 241: Binary Verification) ==========
# Test 240 analysis revealed binary deployment mystery: [PD] signature was in
# binary but log showed 0 [PD] entries. Root cause: sim_vehicle.py may cache
# old binary if SITL process still running.
#
# Fix: Kill existing SITL, verify binary, check signatures BEFORE launch.
# =========================================================================

echo "Deployment Hardening..."

# Compute port offset early so Step 1 can scope kills to this env only.
# Full PORT_OFFSET setup happens later (line ~423) but we need it here.
_EARLY_PORT_OFFSET="${SITL_PORT_OFFSET:-${HARNESS_PORT_OFFSET:-0}}"
_EARLY_TCP_PORT=$((5760 + _EARLY_PORT_OFFSET))

# Step 1: Kill SITL processes belonging to THIS env only (port-scoped).
# ALL offsets including 0 are port-scoped. When offset=0, tcp=5760.
# Previous else-branch used global pkill for offset=0, destroying other
# parallel envs during SubprocVecEnv auto-reset. (C006 Phase 2 fix)
echo "  [DIAG] Step 1 (port-scoped, offset=$_EARLY_PORT_OFFSET, tcp=$_EARLY_TCP_PORT)"
SCOPE_PIDS=$(pgrep -fa "serial0=tcp:${_EARLY_TCP_PORT}" 2>/dev/null || true)
if [ -n "$SCOPE_PIDS" ]; then
    echo "  [DIAG] Scoped ArduPlane: $SCOPE_PIDS"
    pkill -9 -f "serial0=tcp:${_EARLY_TCP_PORT}" 2>/dev/null || true
fi
# Kill MAVProxy instances on our port range
MAVP_PIDS=$(pgrep -fa "master.*tcp:.*:${_EARLY_TCP_PORT}" 2>/dev/null || true)
if [ -n "$MAVP_PIDS" ]; then
    echo "  [DIAG] Scoped MAVProxy: $MAVP_PIDS"
    pkill -9 -f "master.*tcp:.*:${_EARLY_TCP_PORT}" 2>/dev/null || true
fi
if [ -n "$SCOPE_PIDS" ] || [ -n "$MAVP_PIDS" ]; then
    sleep 1
    echo "  ✓ Killed port-scoped SITL processes (offset=$_EARLY_PORT_OFFSET)"
else
    echo "  ✓ No SITL processes found for offset=$_EARLY_PORT_OFFSET"
fi

# Step 2: Verify binary exists
BINARY_PATH="$ARDUPILOT_DIR/build/sitl/bin/arduplane"
if [ ! -f "$BINARY_PATH" ]; then
    echo "ERROR: ArduPlane binary not found at: $BINARY_PATH"
    echo "Build with: cd $ARDUPILOT_DIR && ./waf configure --board sitl && ./waf plane"
    exit 1
fi
echo "  ✓ Binary exists: $BINARY_PATH"

# Step 3: Check for [PD] signature (Test 241 velocity-projection fix)
if strings "$BINARY_PATH" | grep -q '\[PD\]'; then
    echo "  ✓ [PD] signature found in binary (C++ PD controller active)"
else
    echo "WARNING: [PD] signature NOT found in binary"
    echo "  This may indicate the binary was built before Test 241 fixes"
    echo "  Rebuild with: cd $ARDUPILOT_DIR && ./waf plane"
fi

# Step 4: Display binary timestamp for verification
echo "  Binary modified: $(stat -c '%y' "$BINARY_PATH" | cut -d. -f1)"

# Step 5: Verify source is not newer than binary (rebuild needed?)
SOURCE_PATH="$ARDUPILOT_DIR/libraries/AP_Formation/formation_controller.cpp"
if [ -f "$SOURCE_PATH" ]; then
    if [ "$SOURCE_PATH" -nt "$BINARY_PATH" ]; then
        echo "WARNING: Source is newer than binary - rebuild recommended!"
        echo "  Source: $(stat -c '%y' "$SOURCE_PATH" | cut -d. -f1)"
        echo "  Binary: $(stat -c '%y' "$BINARY_PATH" | cut -d. -f1)"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting. Rebuild with: cd $ARDUPILOT_DIR && ./waf plane"
            exit 1
        fi
    else
        echo "  ✓ Binary is up-to-date with source"
    fi
fi

echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Parameter wipe (TEST 384: clean environment)
if [ "${WIPE_PARAMS:-0}" = "1" ]; then
    echo "Wiping SITL parameter cache..."
    rm -f "$ARDUPILOT_DIR/ArduPlane/eeprom.bin" 2>/dev/null || true
    rm -f "$ARDUPILOT_DIR/eeprom.bin" 2>/dev/null || true
    rm -f "$ARDUPILOT_DIR/ArduPlane/mav.parm" 2>/dev/null || true
    rm -f "$HOME/FlightDock/tests/mav.parm" 2>/dev/null || true
    echo "  ✓ Parameter cache wiped (eeprom.bin, mav.parm)"
    echo ""
fi

# Navigate to ArduPlane directory
cd "$ARDUPILOT_DIR/ArduPlane" || {
    echo "ERROR: Could not find ArduPlane directory"
    echo "Expected at: $ARDUPILOT_DIR/ArduPlane"
    echo ""
    if [ "$FD_FW_VARIANT" = "modern" ]; then
        echo "To use modern firmware, ensure you have built it:"
        echo "  cd ~/ardupilot-plane-modern"
        echo "  git checkout Plane-stable  # or desired tag/branch"
        echo "  ./waf configure --board sitl"
        echo "  ./waf plane"
    else
        echo "Please update the path in this script or create a symlink"
    fi
    exit 1
}

echo "Launching ArduPlane SITL with Formation Controller Support..."
echo "  Vehicle: ArduPlane QuadPlane (sysid=1)"
echo "  Location: -35.363261, 149.165230, 584m MSL"
echo ""
echo "  MAVLink Port Architecture:"
echo "    TCP 5760        - SITL main port (MAVProxy master connection)"
echo "    UDP 14550       - MAVProxy INPUT from leader_broadcast_fixed.py (sysid=2)"
echo "                      → leader_broadcast sends to udpout:127.0.0.1:14550"
echo "                      → MAVProxy listens on udpin:0.0.0.0:14550"
echo "    UDP 14551       - MAVProxy OUTPUT for follower telemetry (sysid=1)"
echo "                      → MAVProxy sends to udp:127.0.0.1:14551"
echo "                      → Optional GCS monitoring port"
echo "    UDP 14560       - MAVProxy INPUT for follower control commands (sysid=1)"
echo "                      → MAVProxy listens on udpin:0.0.0.0:14560"
echo "                      → DroneKit/test scripts send to udpout:127.0.0.1:14560"
echo "                      → BIDIRECTIONAL communication channel"
echo "    UDP 14552       - MAVProxy OUTPUT for lead telemetry monitoring (sysid=2 filter)"
echo "                      → MAVProxy sends to udp:127.0.0.1:14552"
echo "                      → LeadTelemetryFeed listens on udpin:0.0.0.0:14552"
echo "    UDP 14554       - MAVProxy OUTPUT for UI (optional Map/Console)"
echo "    UDP 14553       - MAVProxy OUTPUT for leader time-sync (Test 328 SIM_SPEEDUP)"
echo "                      → MAVProxy sends to udp:127.0.0.1:14553"
echo "                      → leader_broadcast_fixed.py --time-sync-conn listens here"
echo "                      → Provides time_boot_ms for sim-time synchronization"
echo ""

# Select parameter file based on firmware variant
if [ "$FD_FW_VARIANT" = "modern" ]; then
    PARAM_FILE="$HOME/FlightDock/config/quadplane_formation_modern_v1.parm"
    PARAM_DESC="modern 4.6.2 (TRIM_ARSPD_CM, ARSPD_FBW_MIN/MAX, no FORM_*)"
else
    PARAM_FILE="$HOME/FlightDock/config/quadplane_formation_mav_clean.parm"
    PARAM_DESC="legacy Phase 7B (AIRSPEED_MIN/MAX/CRUISE + SCALING_SPEED, includes FORM_*)"
fi

echo "Using parameter file: $PARAM_DESC"
echo "  Path: $PARAM_FILE"
echo ""

# Verify parameter file exists
if [ ! -f "$PARAM_FILE" ]; then
    echo "ERROR: Parameter file not found: $PARAM_FILE"
    echo "Please ensure the parameter file exists in ~/FlightDock/config/"
    exit 1
fi

echo "  Parameter Fixes Applied (Nov 16-19, 2025):"
echo "    ✓ RC_OPTIONS = 0 (was 32 - blocked throttle arming)"
echo "    ✓ ARMING_CHECK removed (uses SITL default = 1)"
echo "    ✓ ARMING_REQUIRE removed (uses SITL default = 1)"
echo "    ✓ Q_WVANE_ENABLE = 0 (weathervane disabled for SITL)"
echo "    ✓ Q_ASSIST_SPEED = 8 m/s (VTOL assist below this speed)"
echo "    ✓ Q_GUIDED_MODE = 1 (FW flight in GUIDED mode above Q_ASSIST_SPEED)"
echo "    ✓ FENCE_ENABLE = 0 (geofence disabled for formation tests)"
echo ""

# Start SITL with MAVLink configuration for formation testing
# CRITICAL SYSTEM ID SEPARATION:
#   - QuadPlane SITL runs as sysid=1
#   - leader_broadcast_fixed.py sends as sysid=2 to udp:14550 (MAVProxy input)
#   - baseline_formation_controller.py connects to udp:14551 (QuadPlane control)
#   - LeadTelemetryFeed listens on udp:14552 (lead telemetry, filters for sysid=2)

echo "Starting sim_vehicle.py..."
echo "(This will open MAVProxy console and map in new windows)"
echo ""

# ========== USAGE EXAMPLES ==========
#
# Example 1: FW Cruise Test (verify VTOL→FW transition)
# Terminal 1: ./start_formation_sitl_fixed.sh
# Terminal 2:
#   cd ~/FlightDock/tests
#   RUN_DIR=~/FlightDock/logs/fw_cruise_tests/$(date +"%Y%m%d_%H%M%S")
#   mkdir -p "$RUN_DIR"
#   python3 baseline_formation_controller.py --fw-cruise-test | tee "$RUN_DIR/fw_cruise_baseline.log"
#   python3 analyze_fw_cruise_logs.py "$RUN_DIR"
#
# Example 2: Full Mission (VTOL takeoff -> FW transition -> intercept)
# Terminal 1: ./start_formation_sitl_fixed.sh
# Terminal 2: python3 leader_broadcast_fixed.py --target-speed 24.7 \
#             --time-sync-conn=udpin:127.0.0.1:14553 --sim-speedup=3
# Terminal 3:
#   cd ~/FlightDock/tests
#   RUN_DIR=~/FlightDock/logs/intercept_tests/$(date +"%Y%m%d_%H%M%S")
#   mkdir -p "$RUN_DIR"
#   python3 baseline_formation_controller.py --sim-speedup=3 | tee "$RUN_DIR/intercept_rejoin.log"
#
# Example 3: Skip-Takeoff (for mid-flight restart - controller verifies airborne)
# NOTE: --intercept-rejoin-test is safe even if vehicle is on-ground; controller
#       will auto-fallback to normal takeoff if not airborne and FW-ready.
# Terminal 3:
#   python3 baseline_formation_controller.py --intercept-rejoin-test --sim-speedup=3
#
# Example 4: Formation Flight (full pattern-aware controller)
# Terminal 1: ./start_formation_sitl_fixed.sh
# Terminal 2: python3 leader_broadcast_fixed.py --target-speed 24.7 \
#             --time-sync-conn=udpin:127.0.0.1:14553 --sim-speedup=3
# Terminal 3: python3 baseline_formation_controller.py --sim-speedup=3
#
# ==========================================

# ========== NETWORK CONFIGURATION ==========
# Port 14550: INPUT ONLY (leader_broadcast_fixed.py → MAVProxy)
#
# CRITICAL: We do NOT use --out=<addr>:14550 because that would create a
# feedback loop where MAVProxy sends its own telemetry back to itself on
# the same port it's listening on.
#
# Correct configuration:
#   - MAVProxy: --cmd="link add udpin:0.0.0.0:14550" (listen on 14550)
#   - Broadcaster: udpout:127.0.0.1:14550 (send to 14550)
#
# If you need MAVProxy to send telemetry to an external GCS (QGroundControl,
# Mission Planner, etc.), use a DIFFERENT port:
#   - Example: --out=udp:<gcs-host>:14550 (send to GCS on different host)
#   - Or: --out=udp:127.0.0.1:14560 (send to GCS on different port)
#
# For all-WSL2 deployment (default), no additional --out needed.
# ==========================================

echo "  Port 14550: INPUT from leader_broadcast_fixed.py (no feedback loop)"
echo "    TCP 5763        - DIRECT SITL serial port for weather sync (Phase 7C.22)"
echo "                      → SITL SERIAL2 (auto-bound by SITL) - bypasses MAVProxy"
echo "                      → leader_broadcast_fixed.py connects to tcp:127.0.0.1:5763"
echo "                      → PARAM_SET goes directly to SITL for processing"
echo ""

# C++ FormationController plugin status (Phase 7D.97)
# NOTE: FORM_ENABLE only exists in legacy firmware (4.7-dev FlightDock fork)
# Python controller enforces FORM_ENABLE=0 at connect time via PARAM_SET + confirm
if [ "$FD_FW_VARIANT" = "legacy" ]; then
    echo "C++ FormationController: Python will enforce FORM_ENABLE=0 at connect"
    echo "  → Python sends PARAM_REQUEST_READ to verify current value"
    echo "  → If non-zero, Python sends PARAM_SET with confirmed param_type"
    echo "  → Controller blocks until PARAM_VALUE confirms FORM_ENABLE=0"
    echo ""
else
    echo "Modern firmware: C++ FormationController not present"
    echo ""
fi

# MSL Configuration: Load MAVProxy initialization script for MSL-first altitude display
# Select init script based on HEADLESS mode
if [[ "${HEADLESS:-0}" == "1" ]]; then
    MSL_INIT_SCRIPT="$HOME/FlightDock/config/mavproxy_headless_init.scr"
else
    MSL_INIT_SCRIPT="$HOME/FlightDock/config/mavproxy_msl_init.scr"
fi

# ========== FORM_ENABLE: Python handles this now (Phase 7D.97) ==========
# The Python controller now verifies/sets FORM_ENABLE=0 at startup using
# _verify_form_enable_at_startup() which:
#   1. Sends PARAM_REQUEST_READ to get the current value
#   2. If value != 0, sends PARAM_SET with correct param_type
#   3. Waits for PARAM_VALUE confirmation before any control
#
# This replaced the fragile MAVProxy "param set" command that ran before
# the parameter list was downloaded (causing "Unable to find parameter"
# warnings). The defaults file also sets FORM_ENABLE=0 as a safety net.
# ========================================================================

# =============================================================================
# PORT OFFSET SUPPORT (for parallel environments)
# =============================================================================
# SITL_PORT_OFFSET (or HARNESS_PORT_OFFSET) offsets all ports by given amount.
# This allows multiple SITL instances to run in parallel without port conflicts.
# Port scheme: base_port + offset
#   - SITL TCP: 5760 + offset
#   - Leader in: 14550 + offset
#   - Follower out: 14551 + offset
#   - Monitor: 14552 + offset
#   - Timesync: 14553 + offset
#   - Control in: 14560 + offset
# =============================================================================
PORT_OFFSET="${SITL_PORT_OFFSET:-${HARNESS_PORT_OFFSET:-0}}"

# Compute actual port numbers
PORT_FOLLOWER=$((14551 + PORT_OFFSET))
PORT_MONITOR=$((14552 + PORT_OFFSET))
PORT_TIMESYNC=$((14553 + PORT_OFFSET))
PORT_UI=$((14554 + PORT_OFFSET))
PORT_LEADER_IN=$((14550 + PORT_OFFSET))
PORT_CONTROL_IN=$((14560 + PORT_OFFSET))

# #512 P-ARRIVE probe tap (PREREG_512 §2). Default INERT: unless
# FD_PHASELOC_PROBE=1, PHASELOC_OUT is empty and the MAVProxy command line is
# byte-identical to pre-#512. When armed it adds ONE extra --out, placed
# immediately after the monitor out so both copies leave MAVProxy in the same
# loop iteration -- that adjacency is what lets the listener's arrival stamp
# stand in for the controller's (they differ by the cost of one sendto).
PORT_PHASELOC=$((14556 + PORT_OFFSET))
PHASELOC_OUT=""
if [ "${FD_PHASELOC_PROBE:-0}" = "1" ]; then
    PHASELOC_OUT="--out=udp:127.0.0.1:${PORT_PHASELOC} "
    echo "  [#512] P-ARRIVE tap ARMED on udp:127.0.0.1:${PORT_PHASELOC}"
fi

if [[ "$PORT_OFFSET" -ne 0 ]]; then
    echo "Port offset: $PORT_OFFSET"
    echo "  Follower:  $PORT_FOLLOWER"
    echo "  Monitor:   $PORT_MONITOR"
    echo "  Timesync:  $PORT_TIMESYNC"
    echo "  Leader:    $PORT_LEADER_IN"
    echo "  Control:   $PORT_CONTROL_IN"
    echo ""
fi

if [ -f "$MSL_INIT_SCRIPT" ]; then
    echo "Loading MAVProxy MSL initialization script..."
    echo "  Path: $MSL_INIT_SCRIPT"
    echo ""
    # Fix A (Test 230): Enforce GUIDED mode after links are added and script is loaded
    # FLTMODE_CH=0 is already set in param file to disable RC mode switching
    # Adding explicit "mode GUIDED" ensures we start in GUIDED regardless of RC state
    # Test 328: Added --out for time-sync; Test 334: Port offset support
    MAVPROXY_CMD="--out=udp:127.0.0.1:${PORT_FOLLOWER} --out=udp:127.0.0.1:${PORT_MONITOR} ${PHASELOC_OUT}--out=udp:127.0.0.1:${PORT_TIMESYNC} --out=udp:127.0.0.1:${PORT_UI} --cmd=\"link add udpin:0.0.0.0:${PORT_LEADER_IN}; link add udpin:0.0.0.0:${PORT_CONTROL_IN}; script $MSL_INIT_SCRIPT; mode GUIDED\""
else
    echo "⚠️  MAVProxy MSL init script not found at: $MSL_INIT_SCRIPT"
    echo "   Continuing without MSL-specific MAVProxy configuration"
    echo ""
    # Fix A (Test 230): Enforce GUIDED mode even without MSL script
    # Test 328: Added --out for time-sync; Test 334: Port offset support
    MAVPROXY_CMD="--out=udp:127.0.0.1:${PORT_FOLLOWER} --out=udp:127.0.0.1:${PORT_MONITOR} ${PHASELOC_OUT}--out=udp:127.0.0.1:${PORT_TIMESYNC} --out=udp:127.0.0.1:${PORT_UI} --cmd=\"link add udpin:0.0.0.0:${PORT_LEADER_IN}; link add udpin:0.0.0.0:${PORT_CONTROL_IN}; mode GUIDED\""
fi

# Phase 7C.22: Direct SITL TCP port for weather sync (bypasses MAVProxy)
# ============================================================================
# SERIAL2 is already configured for MAVLink2 in the param file:
#   SERIAL2_PROTOCOL=2 (MAVLink2)
#
# SITL automatically binds SERIAL2 to TCP port 5763, which we use for
# direct weather sync bypassing MAVProxy.
#
# Architecture:
#   leader_broadcast_fixed.py → tcp:127.0.0.1:5763 → SITL SERIAL2 → param applied
#   (no MAVProxy in the path!)
# ============================================================================

# Add --non-interactive for headless mode (prevents MAVProxy from quitting without stdin)
if [[ "${HEADLESS:-0}" == "1" ]]; then
    MAVPROXY_CMD="--non-interactive $MAVPROXY_CMD"
    echo "Adding --non-interactive for headless MAVProxy operation"
fi

# =============================================================================
# BUILD GUI OPTIONS
# =============================================================================
# Supports fine-grained control via HARNESS_UI_MODE or legacy HEADLESS flag
#   HARNESS_UI_MODE=none → No windows (headless)
#   HARNESS_UI_MODE=map → Map window only
#   HARNESS_UI_MODE=console → Console window only
#   HARNESS_UI_MODE=both → Map + console (full interactive)
#   HEADLESS=1 (legacy) → No windows

GUI_OPTS=""
UI_MODE="${HARNESS_UI_MODE:-}"

# If no UI mode set, default based on HEADLESS flag
if [[ -z "$UI_MODE" ]]; then
    if [[ "${HEADLESS:-0}" == "1" ]]; then
        UI_MODE="none"
    else
        UI_MODE="both"
    fi
fi

case "$UI_MODE" in
    none)
        GUI_OPTS=""
        echo "UI mode: none (headless, no windows)"
        ;;
    map)
        GUI_OPTS="--map"
        echo "UI mode: map only"
        ;;
    console)
        GUI_OPTS="--console"
        echo "UI mode: console only"
        ;;
    both)
        GUI_OPTS="--console --map"
        echo "UI mode: console + map (full interactive)"
        ;;
    *)
        echo "WARNING: Unknown UI_MODE='$UI_MODE', defaulting to headless"
        GUI_OPTS=""
        ;;
esac

# =============================================================================
# GUI SPEEDUP POLICY (Phase 7D.334)
# =============================================================================
# GUI windows can't keep up with fast simulation (>3x) and cause instability.
# Policy:
#   - If UI requested AND speedup > GUI_MAX_SPEEDUP:
#     - Clamp speedup to GUI_MAX_SPEEDUP and warn
#     - If --force-ui: allow but warn loudly about stability risk
#   - If --headless: disable UI (no clamping needed)
#
GUI_MAX_SPEEDUP=3.0

if [[ -n "$SITL_SPEEDUP" ]] && [[ -n "$GUI_OPTS" ]]; then
    # UI is requested and we have a speedup value
    if (( $(echo "$SITL_SPEEDUP > $GUI_MAX_SPEEDUP" | bc -l) )); then
        if [[ "${FORCE_UI:-0}" == "1" ]]; then
            # Force-UI mode: allow UI but clamp speedup and warn
            echo ""
            echo "╔═══════════════════════════════════════════════════════════════════╗"
            echo "║ ⚠️  WARNING: --force-ui ENABLED                                    ║"
            echo "╠═══════════════════════════════════════════════════════════════════╣"
            echo "║ Requested speedup: ${SITL_SPEEDUP}x with GUI enabled              ║"
            echo "║ Clamped to: ${GUI_MAX_SPEEDUP}x (GUI_MAX_SPEEDUP)                  ║"
            echo "║                                                                   ║"
            echo "║ GUI at high speedup causes:                                       ║"
            echo "║   - MAVProxy instability and crashes                              ║"
            echo "║   - Timing drift and simulation quality degradation               ║"
            echo "║   - Unreliable test results                                       ║"
            echo "║                                                                   ║"
            echo "║ For production tests, use --headless mode.                        ║"
            echo "╚═══════════════════════════════════════════════════════════════════╝"
            echo ""
            SITL_SPEEDUP="$GUI_MAX_SPEEDUP"
        else
            # Default policy: clamp and warn
            echo ""
            echo "[GUI_POLICY] Speedup ${SITL_SPEEDUP}x > ${GUI_MAX_SPEEDUP}x with UI enabled"
            echo "[GUI_POLICY] Clamping speedup to ${GUI_MAX_SPEEDUP}x for GUI stability"
            echo "[GUI_POLICY] Use --headless for full speedup, or --force-ui to override"
            echo ""
            SITL_SPEEDUP="$GUI_MAX_SPEEDUP"
        fi
    fi
fi

# Pass speedup as-is (ArduPilot SITL accepts float via strtof())
SITL_SPEEDUP_INT="$SITL_SPEEDUP"
echo "SITL speedup: ${SITL_SPEEDUP_INT}x"

# =============================================================================
# DIRECT MODE: Run ArduPlane and MAVProxy directly (WSL2 compatible)
# =============================================================================
# This bypasses sim_vehicle.py which relies on run_in_terminal_window.sh
# that fails in WSL2. Instead, we run ArduPlane and MAVProxy directly.
#
# Benefits:
#   - Works reliably in WSL2
#   - No terminal window spawning issues
#   - Simpler process management
#
# Usage:
#   ./start_formation_sitl_fixed.sh --direct --speedup 3.0
#   ./start_formation_sitl_fixed.sh --direct --headless --speedup 5.0
# =============================================================================

if [[ "$DIRECT_MODE" == "1" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  DIRECT MODE: Running ArduPlane and MAVProxy directly"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""

    # Home location for SITL
    SITL_HOME="-35.363261,149.165230,584,353"

    # Default param file for quadplane
    DEFAULT_PARAMS="$ARDUPILOT_DIR/Tools/autotest/default_params/quadplane.parm"

    # Compute SITL TCP port (base + offset)
    SITL_TCP_PORT=$((5760 + PORT_OFFSET))

    echo "Starting ArduPlane binary directly..."
    echo "  Binary: $BINARY_PATH"
    echo "  Home: $SITL_HOME"
    echo "  Speedup: ${SITL_SPEEDUP_INT}x"
    echo "  TCP port: $SITL_TCP_PORT"
    echo ""

    # Start ArduPlane in background
    # Parallel port isolation:
    #   --serial0: MAVProxy master TCP (5760 + offset)
    #   --serial1: ADSB sim / unused TCP (5762 + offset)
    #   --serial2: Weather sync / leader_broadcast TCP (5763 + offset)
    # All three must be explicitly set to avoid port collisions between
    # parallel SITL instances. Using --base-port changes SERIAL0 to :wait
    # mode which causes simulation pacing issues, so we use --serialN directly.
    #
    # SITL_WORK_DIR: when set, run ArduPlane from a separate directory
    # so each parallel instance gets its own eeprom.bin/logs
    SERIAL1_PORT=$((5762 + PORT_OFFSET))
    SERIAL2_PORT=$((5763 + PORT_OFFSET))
    # Always wipe eeprom on SITL boot to prevent stale param contamination.
    # Without this, failed runs write back to eeprom.bin/mav.parm and poison
    # subsequent boots (e.g. SIM_SPEEDUP=2.0 persists when requesting 1.0x).
    # --defaults reloads from param files, so no configuration is lost.
    # NOTE: --no-wipe CLI flag only suppresses the pre-launch rm (line 233),
    # not this SITL-level --wipe. This is intentional — SITL eeprom must always
    # be clean to prevent cross-run contamination (PRH-ATT-004).
    WIPE_FLAG="--wipe"
    if [[ -n "${SITL_WORK_DIR:-}" ]]; then
        mkdir -p "$SITL_WORK_DIR"
        echo "  Work dir: $SITL_WORK_DIR (parallel-safe)"
        pushd "$SITL_WORK_DIR" > /dev/null
    fi

    # #309 P1: JSON-lockstep backend swap. Unset/native => quadplane + no extra
    # args => argv byte-identical to pre-P1. json => external-FDM model +
    # --sim-port-out (FD_FDM_PORT, default 9002). Empty array expands to ZERO
    # words under bash 5.2 set -u (safe since bash 4.4) => no phantom arg native.
    if [ "${FD_TRAIN_BACKEND:-native}" = "json" ]; then
        SITL_MODEL="JSON:127.0.0.1"
        SIM_PORT_ARGS=(--sim-port-out "${FD_FDM_PORT:-9002}")
    else
        SITL_MODEL="quadplane"
        SIM_PORT_ARGS=()
    fi

    # #362 pin-test: optional fixed SITL epoch (default-OFF). Same empty-array
    # idiom as SIM_PORT_ARGS above => unset/empty => ZERO words => arduplane argv
    # byte-identical to pre-#362 (native + json). Set => one --start-time <int>
    # pair (integer UNIX seconds; atoi in SITL_cmdline.cpp) so SITL start_time_UTC
    # is pinned instead of wall-seeded from gettimeofday().
    if [ -n "${FD_SITL_START_TIME:-}" ]; then
        START_TIME_ARGS=(--start-time "$FD_SITL_START_TIME")
    else
        START_TIME_ARGS=()
    fi

    "$BINARY_PATH" \
        --model "$SITL_MODEL" \
        "${SIM_PORT_ARGS[@]}" \
        "${START_TIME_ARGS[@]}" \
        --speedup "$SITL_SPEEDUP_INT" \
        --home "$SITL_HOME" \
        --serial0=tcp:$SITL_TCP_PORT \
        --serial1=tcp:$SERIAL1_PORT \
        --serial2=tcp:$SERIAL2_PORT \
        --defaults "$DEFAULT_PARAMS,$PARAM_FILE${FD_EXTRA_DEFAULTS:+,$FD_EXTRA_DEFAULTS}" \
        $WIPE_FLAG \
        &
    ARDUPLANE_PID=$!

    if [[ -n "${SITL_WORK_DIR:-}" ]]; then
        popd > /dev/null
    fi
    echo "ArduPlane started with PID: $ARDUPLANE_PID"

    # Wait for ArduPlane to start listening
    echo "Waiting for ArduPlane to start..."
    sleep 5

    # Verify ArduPlane is running
    if ! kill -0 "$ARDUPLANE_PID" 2>/dev/null; then
        echo "ERROR: ArduPlane failed to start"
        exit 1
    fi
    echo "✓ ArduPlane is running"

    # Build MAVProxy command
    echo ""
    echo "Starting MAVProxy..."
    echo "  Master: tcp:127.0.0.1:$SITL_TCP_PORT"
    echo "  Follower port: $PORT_FOLLOWER"
    echo "  Monitor port: $PORT_MONITOR"
    echo "  Timesync port: $PORT_TIMESYNC"
    echo ""

    # Build MAVProxy args
    MAVPROXY_FULL_CMD="mavproxy.py --master=tcp:127.0.0.1:$SITL_TCP_PORT"
    MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --out=udp:127.0.0.1:${PORT_FOLLOWER}"
    MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --out=udp:127.0.0.1:${PORT_MONITOR}"
    # #512 P-ARRIVE tap -- DIRECT_MODE path. This is the builder the TRAINING
    # assembly actually executes; the single-line MAVPROXY_CMD sites above are
    # the sim_vehicle.py path. Patching only those left the tap absent from the
    # live MAVProxy (caught by the G-1 pilot gate, 2026-07-21).
    if [ -n "${PHASELOC_OUT}" ]; then
        MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --out=udp:127.0.0.1:${PORT_PHASELOC}"
    fi
    MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --out=udp:127.0.0.1:${PORT_TIMESYNC}"
    MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --out=udp:127.0.0.1:${PORT_UI}"

    # Add GUI options if not headless
    if [[ "${HEADLESS:-0}" != "1" ]]; then
        case "$UI_MODE" in
            map)
                MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --map"
                ;;
            console)
                MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --console"
                ;;
            both)
                MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --console --map"
                ;;
        esac
    fi

    # Add link commands for leader input and control input
    if [ -f "$MSL_INIT_SCRIPT" ]; then
        MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --cmd=\"link add udpin:0.0.0.0:${PORT_LEADER_IN}; link add udpin:0.0.0.0:${PORT_CONTROL_IN}; script $MSL_INIT_SCRIPT; mode GUIDED\""
    else
        MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --cmd=\"link add udpin:0.0.0.0:${PORT_LEADER_IN}; link add udpin:0.0.0.0:${PORT_CONTROL_IN}; mode GUIDED\""
    fi

    # Add non-interactive for headless
    if [[ "${HEADLESS:-0}" == "1" ]]; then
        MAVPROXY_FULL_CMD="$MAVPROXY_FULL_CMD --non-interactive"
    fi

    echo "MAVProxy command:"
    echo "  $MAVPROXY_FULL_CMD"
    echo ""

    # Function to cleanup on exit
    cleanup_direct() {
        echo ""
        echo "Cleaning up..."
        kill -INT "$ARDUPLANE_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$ARDUPLANE_PID" 2>/dev/null || true
        echo "SITL stopped."
        deactivate 2>/dev/null || true
    }
    trap cleanup_direct EXIT INT TERM

    # Run MAVProxy - either directly (headless) or in xterm (UI mode)
    if [[ "${HEADLESS:-0}" == "1" || "$UI_MODE" == "none" ]]; then
        # Headless mode: run MAVProxy directly with non-interactive
        echo "Running MAVProxy in headless mode..."
        eval "$MAVPROXY_FULL_CMD"
    else
        # UI mode: launch MAVProxy in xterm so it gets a proper TTY
        echo "Launching MAVProxy in xterm window..."

        # Write the MAVProxy command to a temp script (avoids quoting issues)
        MAVPROXY_SCRIPT=$(mktemp /tmp/mavproxy_ui_XXXXXX.sh)
        cat > "$MAVPROXY_SCRIPT" << MAVPROXY_EOF
#!/bin/bash
source ~/FlightDock/flightdock_native_venv/bin/activate
$MAVPROXY_FULL_CMD
echo ""
echo "MAVProxy exited. Press Enter to close..."
read
MAVPROXY_EOF
        chmod +x "$MAVPROXY_SCRIPT"
        echo "MAVProxy script: $MAVPROXY_SCRIPT"

        # Launch xterm with the script
        xterm -title "MAVProxy SITL" -geometry 120x40 -e "$MAVPROXY_SCRIPT" &
        XTERM_PID=$!
        echo "xterm PID: $XTERM_PID"

        # Wait for xterm/MAVProxy to fully start
        sleep 3

        # Keep script running while SITL is alive (so cleanup trap works)
        echo ""
        echo "MAVProxy running in xterm window. SITL will stay running."
        echo "Press Ctrl+C or close xterm to stop."
        echo ""

        # Wait for ArduPlane to exit (keeps script alive for cleanup trap)
        wait "$ARDUPLANE_PID" 2>/dev/null || true

        # Cleanup temp script
        rm -f "$MAVPROXY_SCRIPT" 2>/dev/null || true
    fi

    # If MAVProxy exits, cleanup will run
    exit 0
fi

# =============================================================================
# STANDARD MODE: Use sim_vehicle.py (original behavior)
# =============================================================================

# shellcheck disable=SC2086 # Intentional word splitting for GUI_OPTS
python3 ../Tools/autotest/sim_vehicle.py -v ArduPlane -f quadplane \
    --custom-location=-35.363261,149.165230,584,353 \
    --speedup "$SITL_SPEEDUP_INT" \
    $GUI_OPTS \
    --add-param-file="$PARAM_FILE" \
    --mavproxy-args="$MAVPROXY_CMD"

# If SITL exits, deactivate venv
echo ""
echo "SITL exited. Deactivating virtual environment..."
deactivate 2>/dev/null || true
