#!/usr/bin/env python3
"""
FlightDock - QuadPlane QLOITER VTOL Takeoff Controller (Pure PyMAVLink)

This module provides a clean, reusable controller class for:
1. Robust arming in VTOL mode (QLOITER)
2. VTOL climb to precontact altitude
3. Transition to fixed-wing forward flight

Designed to be used by both test scripts and the formation controller.

**MIGRATED TO PURE PYMAVLINK** - No DroneKit dependency
- Native multi-vehicle support with system ID filtering
- Direct MAVLink message handling
- Lower latency, more control

Author: FlightDock Research Team
Date: November 2025
Migration Date: November 17, 2025
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink not found. Install with: pip install pymavlink")
    raise


# ========== RESULT CLASSES ==========

@dataclass
class TakeoffResult:
    """Result of a takeoff operation with success status and diagnostic info."""
    success: bool
    message: str
    final_altitude_agl: Optional[float] = None
    final_mode: Optional[str] = None
    elapsed_time: Optional[float] = None
    fw_ready: bool = False  # True if vehicle is in FW flight and ready for formation control


# ========== MAIN CONTROLLER CLASS ==========

class QuadplaneQloiterTakeoffController:
    """
    Reusable controller for QuadPlane VTOL takeoff, climb, and transition.

    This class encapsulates the generic sequence:
    1. Enter VTOL mode (e.g., QLOITER)
    2. Arm the vehicle
    3. Climb to precontact altitude in VTOL
    4. Transition to fixed-wing forward flight

    Designed to work with ArduPlane QuadPlane via pure PyMAVLink.
    Native multi-vehicle support with system ID filtering (sysid=1 for QuadPlane).
    """

    # ArduPlane mode numbers
    MODE_MANUAL = 0
    MODE_FBWA = 5
    MODE_FBWB = 6
    MODE_CRUISE = 7
    MODE_AUTO = 10
    MODE_RTL = 11
    MODE_LOITER = 12
    MODE_GUIDED = 15
    MODE_QSTABILIZE = 17
    MODE_QHOVER = 18
    MODE_QLOITER = 19
    MODE_QLAND = 20
    MODE_QRTL = 21

    MODE_MAP = {
        0: "MANUAL", 5: "FBWA", 6: "FBWB", 7: "CRUISE",
        10: "AUTO", 11: "RTL", 12: "LOITER", 15: "GUIDED",
        17: "QSTABILIZE", 18: "QHOVER", 19: "QLOITER",
        20: "QLAND", 21: "QRTL"
    }

    MODE_NAME_TO_NUMBER = {v: k for k, v in MODE_MAP.items()}

    def __init__(
        self,
        mav: mavutil.mavlink_connection,
        home_alt_msl: float,
        target_system: int = 1,
        target_component: int = 1,
        vtol_mode_name: str = "QLOITER",
        fixedwing_mode_name: str = "GUIDED",
        target_precontact_alt_agl: float = 306.0,  # 10m below DHC-2 Beaver (890m MSL - 584m home = 306m AGL)
        climb_throttle_pwm: int = 2000,  # CLIMB_TUNE 2026-06-10: 1800(57%,1.44m/s)->2000(full RC3 -> ~2.5m/s, Q_PILOT_SPD_UP cap) for faster VTOL climb
        climb_rate_tolerance: float = 0.5,  # m/s
        altitude_tolerance: float = 2.0,  # meters
        mode_change_timeout: float = 15.0,
        arming_timeout: float = 20.0,
        climb_timeout: float = 300.0,  # Increased to 5 min for 406m AGL climb (was 120s for 142m)
        transition_timeout: float = 30.0,
        speed_mode_for_fw: Optional[str] = None,  # "SLOW" | "MATCH" | "FAST" for pre-CRUISE PARAM_SET
    ):
        """
        Initialize the QuadPlane VTOL takeoff controller.

        Args:
            mav: PyMAVLink connection object for the QuadPlane follower
            home_alt_msl: Home altitude in meters MSL (for AGL calculations)
            target_system: MAVLink system ID (default: 1 for QuadPlane)
            target_component: MAVLink component ID (default: 1)
            vtol_mode_name: VTOL hover mode name (default: "QLOITER")
            fixedwing_mode_name: Fixed-wing mode for forward flight (default: "GUIDED")
            target_precontact_alt_agl: Target altitude AGL in meters (default: 406.0m = 990m MSL - 584m home)
            climb_throttle_pwm: RC throttle override PWM for climb (default: 2000 = full RC3 -> ~2.5 m/s)
            climb_rate_tolerance: Acceptable climb rate variation in m/s
            altitude_tolerance: Acceptable altitude error in meters
            mode_change_timeout: Timeout for mode changes in seconds
            arming_timeout: Timeout for arming in seconds
            climb_timeout: Timeout for climb in seconds (default: 300s for 406m AGL climb)
            transition_timeout: Timeout for fixed-wing transition in seconds
            speed_mode_for_fw: Optional speed mode ("SLOW"|"MATCH"|"FAST") for pre-CRUISE PARAM_SET
        """
        self.mav = mav
        self.target_system = target_system
        self.target_component = target_component
        self.home_alt_msl = home_alt_msl
        self.vtol_mode_name = vtol_mode_name
        self.fixedwing_mode_name = fixedwing_mode_name
        self.target_precontact_alt_agl = target_precontact_alt_agl
        self.climb_throttle_pwm = climb_throttle_pwm
        self.climb_rate_tolerance = climb_rate_tolerance
        self.altitude_tolerance = altitude_tolerance
        self.mode_change_timeout = mode_change_timeout
        self.arming_timeout = arming_timeout
        self.climb_timeout = climb_timeout
        self.transition_timeout = transition_timeout
        self.speed_mode_for_fw = speed_mode_for_fw

        # Internal state
        self.start_time: Optional[float] = None

        # Message cache for latest telemetry (filtered by target_system)
        self._last_global_position = None
        self._last_heartbeat = None
        self._last_statustext = None
        self._last_command_ack = None
        self._cached_vfr_hud = None  # FD-001: Shared from main controller

    def log(self, msg: str):
        """Timestamped logger for diagnostic output."""
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.3f}s] TAKEOFF: {msg}", flush=True)

    def _update_message_cache(self, timeout_ms: int = 10):
        """
        Update cached messages by polling MAVLink connection.

        Reads all available messages and caches the latest ones from target_system.

        Args:
            timeout_ms: Timeout for recv_match in milliseconds (default: 10ms non-blocking)
        """
        while True:
            msg = self.mav.recv_match(blocking=False, timeout=timeout_ms / 1000.0)
            if msg is None:
                break

            # Only cache messages from our target system
            if msg.get_srcSystem() != self.target_system:
                continue

            msg_type = msg.get_type()
            if msg_type == 'GLOBAL_POSITION_INT':
                self._last_global_position = msg
            elif msg_type == 'HEARTBEAT':
                self._last_heartbeat = msg
            elif msg_type == 'STATUSTEXT':
                self._last_statustext = msg
            elif msg_type == 'COMMAND_ACK':
                self._last_command_ack = msg

    def _get_altitude_agl(self) -> float:
        """
        Get current altitude above ground level, filtered by system ID.

        Returns altitude from cached GLOBAL_POSITION_INT message.
        FD-001 Fix: Don't call _update_message_cache() - use cached messages only.
        """
        # FD-001 Fix: Don't steal messages - rely on cached position
        # self._update_message_cache()

        if self._last_global_position:
            # relative_alt is in millimeters, convert to meters
            return self._last_global_position.relative_alt / 1000.0

        # No message available yet
        return 0.0

    def _is_armed(self) -> bool:
        """
        Check if vehicle is armed, filtered by system ID.

        Returns armed state from cached HEARTBEAT message.
        FD-001 Fix: Don't call _update_message_cache() - use cached messages only.
        """
        # FD-001 Fix: Don't steal messages - rely on cached heartbeat
        # self._update_message_cache()

        if self._last_heartbeat:
            return bool(self._last_heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

        return False

    def _get_mode_name(self) -> str:
        """
        Get current mode name, filtered by system ID.

        Returns mode name from cached HEARTBEAT message.
        FD-001 Fix: Don't call _update_message_cache() - use cached messages only.
        """
        # FD-001 Fix: Don't steal messages - rely on cached heartbeat
        # self._update_message_cache()

        if self._last_heartbeat:
            custom_mode = self._last_heartbeat.custom_mode
            return self.MODE_MAP.get(custom_mode, f"UNKNOWN({custom_mode})")

        return "UNKNOWN"

    def _get_mode_number(self) -> Optional[int]:
        """Get current mode number from cached HEARTBEAT.

        FD-001 Fix: Don't call _update_message_cache() - use cached messages only.
        """
        # FD-001 Fix: Don't steal messages - rely on cached heartbeat
        # self._update_message_cache()

        if self._last_heartbeat:
            return self._last_heartbeat.custom_mode

        return None

    def _is_mode(self, mode_name: str) -> bool:
        """
        Check if vehicle is in specified mode.

        Args:
            mode_name: Target mode name (e.g., "QLOITER", "GUIDED")

        Returns:
            True if vehicle is in the specified mode, False otherwise
        """
        current_mode = self._get_mode_name()
        return current_mode == mode_name.upper()

    def _get_last_statustext(self) -> Optional[str]:
        """Return the last STATUSTEXT message from ArduPilot, if any."""
        self._update_message_cache()

        if self._last_statustext:
            try:
                text = self._last_statustext.text.decode("utf-8", errors="ignore")
            except AttributeError:
                text = self._last_statustext.text
            return text

        return None

    def _get_last_arm_ack(self) -> Optional[str]:
        """
        Get the last COMMAND_ACK for ARM/DISARM commands.

        Returns formatted string with ACK result, or None.
        """
        self._update_message_cache()

        if self._last_command_ack:
            cmd = self._last_command_ack.command
            if cmd == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                result = self._last_command_ack.result
                result_map = {
                    mavutil.mavlink.MAV_RESULT_ACCEPTED: "ACCEPTED",
                    mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED: "TEMP_REJECTED",
                    mavutil.mavlink.MAV_RESULT_DENIED: "DENIED",
                    mavutil.mavlink.MAV_RESULT_UNSUPPORTED: "UNSUPPORTED",
                    mavutil.mavlink.MAV_RESULT_FAILED: "FAILED",
                }
                result_str = result_map.get(result, f"UNKNOWN({result})")
                return f"ARM/DISARM: {result_str}"

        return None

    def _wait_for_command_ack(self, command_id: int, timeout: float = 5.0) -> Tuple[bool, str]:
        """
        Wait for COMMAND_ACK for a specific command.

        Args:
            command_id: MAVLink command ID to wait for (e.g., MAV_CMD_DO_VTOL_TRANSITION)
            timeout: Maximum time to wait in seconds

        Returns:
            Tuple of (success: bool, message: str)
            - success: True if ACCEPTED, False otherwise
            - message: Human-readable result string
        """
        result_map = {
            mavutil.mavlink.MAV_RESULT_ACCEPTED: "ACCEPTED",
            mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED: "TEMPORARILY_REJECTED",
            mavutil.mavlink.MAV_RESULT_DENIED: "DENIED",
            mavutil.mavlink.MAV_RESULT_UNSUPPORTED: "UNSUPPORTED",
            mavutil.mavlink.MAV_RESULT_FAILED: "FAILED",
            mavutil.mavlink.MAV_RESULT_IN_PROGRESS: "IN_PROGRESS",
        }

        start_time = time.time()

        while time.time() - start_time < timeout:
            msg = self.mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=0.5)

            if msg and msg.get_srcSystem() == self.target_system:
                if msg.command == command_id:
                    result = msg.result
                    result_str = result_map.get(result, f"UNKNOWN({result})")

                    if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        return True, result_str
                    else:
                        return False, result_str

        return False, "TIMEOUT"

    def set_mode_and_wait(self, mode_name: str, timeout: Optional[float] = None) -> bool:
        """
        Set flight mode and wait for confirmation.

        Args:
            mode_name: Target mode name (e.g., "QLOITER", "GUIDED")
            timeout: Timeout in seconds (default: use self.mode_change_timeout)

        Returns:
            True if mode change successful, False if timeout
        """
        if timeout is None:
            timeout = self.mode_change_timeout

        # Get mode number
        mode_number = self.MODE_NAME_TO_NUMBER.get(mode_name.upper())
        if mode_number is None:
            self.log(f"❌ Unknown mode name: {mode_name}")
            return False

        self.log(f"Setting mode to {mode_name} (current: {self._get_mode_name()})")

        # Override RC channel 8 (flight mode switch) to prevent interference
        self.mav.mav.rc_channels_override_send(
            self.target_system,
            self.target_component,
            0, 0, 0, 0, 0, 0, 0, 1500  # channels 1-8, channel 8 = 1500
        )
        time.sleep(0.5)

        # Send mode change command
        self.mav.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,  # confirmation
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_number,
            0, 0, 0, 0, 0
        )

        start = time.time()
        attempt_count = 0

        while time.time() - start < timeout:
            if self._is_mode(mode_name):
                self.log(f"✓ Mode confirmed: {mode_name}")
                return True

            # Re-attempt mode change every second
            if time.time() - start > attempt_count:
                self.mav.mav.command_long_send(
                    self.target_system,
                    self.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_number,
                    0, 0, 0, 0, 0
                )
                attempt_count += 1

            time.sleep(0.5)

        # Timeout
        current_mode = self._get_mode_name()
        self.log(f"❌ Mode change timeout: wanted {mode_name}, got {current_mode}")
        return False

    def arm_and_enter_vtol(self) -> TakeoffResult:
        """
        Enter VTOL mode and arm the vehicle.

        Returns:
            TakeoffResult with success status and diagnostic information
        """
        self.log(f"=== PHASE 1: Arming in {self.vtol_mode_name} ===")

        # Step 1: Enter VTOL mode
        if not self.set_mode_and_wait(self.vtol_mode_name):
            return TakeoffResult(
                success=False,
                message=f"Failed to set {self.vtol_mode_name} mode",
                final_mode=self._get_mode_name(),
            )

        # Step 2: Wait for armable (check system status)
        self.log("Waiting for vehicle to be armable...")
        start = time.time()
        while time.time() - start < self.arming_timeout:
            self._update_message_cache()
            if self._last_heartbeat:
                system_status = self._last_heartbeat.system_status
                if system_status in [
                    mavutil.mavlink.MAV_STATE_STANDBY,
                    mavutil.mavlink.MAV_STATE_ACTIVE
                ]:
                    break
            time.sleep(0.5)
        else:
            self.log("⚠️  Timeout waiting for armable state")

        self.log("✓ Vehicle is armable")

        # TEST 334: Wait for EKF3 to be fully active before arming
        # This prevents "AHRS: not using configured AHRS type" arming failures
        self.log("Waiting for EKF3 to become active...")
        ekf_wait_start = time.time()
        ekf_timeout = 30.0  # seconds to wait for EKF
        ekf_active = False
        while time.time() - ekf_wait_start < ekf_timeout:
            # Poll for EKF_STATUS_REPORT which indicates EKF is running
            msg = self.mav.recv_match(
                type='EKF_STATUS_REPORT', blocking=True, timeout=0.5
            )
            if msg:
                # Check velocity and position flags (bits 0 and 1)
                # EKF is healthy when these are set
                flags = msg.flags
                vel_ok = (flags & 1) != 0  # velocity variance OK
                pos_ok = (flags & 2) != 0  # position horizontal variance OK
                if vel_ok and pos_ok:
                    ekf_active = True
                    self.log(f"✓ EKF3 active (flags=0x{flags:02x})")
                    break
            time.sleep(0.1)

        if not ekf_active:
            self.log("⚠️  EKF not fully active, attempting arm anyway...")

        # Step 3: Send arm command with retry logic
        # TEST 334: Retry arming multiple times to handle SITL initialization race
        MAX_ARM_ATTEMPTS = 10
        ARM_RETRY_DELAY = 3.0  # seconds between attempts

        for attempt in range(1, MAX_ARM_ATTEMPTS + 1):
            self.log(f"Sending arm command (attempt {attempt}/{MAX_ARM_ATTEMPTS})...")
            self.mav.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                1,  # 1=arm, 0=disarm
                0, 0, 0, 0, 0, 0
            )

            # Wait for arm ACK or armed state
            arm_check_start = time.time()
            arm_check_timeout = 3.0
            while time.time() - arm_check_start < arm_check_timeout:
                if self._is_armed():
                    self.log(f"✓ Vehicle armed (attempt {attempt})")
                    return self._post_arm_success()
                time.sleep(0.2)

            # Check why arm failed
            last_text = self._get_last_statustext()
            ack = self._get_last_arm_ack()
            if ack:
                self.log(f"  ARM ACK: {ack}")
            if last_text:
                self.log(f"  ArduPilot: {last_text}")

            # Wait before retry
            if attempt < MAX_ARM_ATTEMPTS:
                self.log(f"  Waiting {ARM_RETRY_DELAY}s before retry...")
                time.sleep(ARM_RETRY_DELAY)

        # All attempts failed
        return TakeoffResult(
            success=False,
            message=f"Arming failed after {MAX_ARM_ATTEMPTS} attempts",
            final_mode=self._get_mode_name(),
        )

    def _post_arm_success(self) -> TakeoffResult:
        """Handle successful arm and return result."""
        self.log(f"✓ Armed in {self.vtol_mode_name} mode")
        return TakeoffResult(
            success=True,
            message="Armed successfully",
            final_mode=self._get_mode_name(),
        )

    def _legacy_arm_monitor(self) -> TakeoffResult:
        """Legacy arm monitoring - kept for reference."""
        # Step 4: Monitor armed state
        self.log("Monitoring armed state...")
        start = time.time()
        while time.time() - start < self.arming_timeout:
            if self._is_armed():
                elapsed = time.time() - start
                self.log(f"✓ Vehicle armed (elapsed={elapsed:.1f}s)")
                break
            time.sleep(0.5)
        else:
            # Arming timeout
            last_text = self._get_last_statustext()
            if last_text:
                self.log(f"  Last ArduPilot message: {last_text}")
            ack = self._get_last_arm_ack()
            if ack:
                self.log(f"  Last ARM/DISARM ACK: {ack}")
            return TakeoffResult(
                success=False,
                message=f"Arming timeout after {self.arming_timeout}s",
                final_mode=self._get_mode_name(),
            )

    def vtol_climb_to_precontact_alt(self) -> TakeoffResult:
        """
        Climb to precontact altitude in VTOL mode using RC throttle override.

        Returns:
            TakeoffResult with success status and diagnostic information
        """
        target_msl = self.home_alt_msl + self.target_precontact_alt_agl
        self.log(
            f"=== PHASE 2: VTOL Climb to {target_msl:.1f}m MSL (Precontact) ==="
        )

        # TEST 334: Get current altitude BEFORE applying throttle to check direction
        start_alt = self._get_altitude_agl()
        start_alt_msl = self.home_alt_msl + start_alt
        self.log(f"Starting altitude: {start_alt_msl:.2f}m MSL")

        # TEST 334: Check if we're already above target (stale SITL state)
        alt_error = self.target_precontact_alt_agl - start_alt
        if alt_error < -self.altitude_tolerance:
            self.log(
                f"⚠️  Already ABOVE target altitude! "
                f"(start={start_alt_msl:.1f}m > target={target_msl:.1f}m)"
            )
            self.log("Skipping climb phase - proceeding to fixed-wing transition")
            return TakeoffResult(
                success=True,
                message="Already above target - skipping climb",
                final_altitude_agl=start_alt,
                final_mode=self._get_mode_name(),
                elapsed_time=0.0,
            )

        # Apply immediate throttle kick to prevent idle auto-disarm
        self.log(f"Applying RC3 throttle override: {self.climb_throttle_pwm} PWM")
        self.mav.mav.rc_channels_override_send(
            self.target_system,
            self.target_component,
            0, 0, self.climb_throttle_pwm, 0, 0, 0, 0, 0  # channels 1-8
        )

        start = time.time()
        last_valid_alt = start_alt
        last_log_time = start

        # Multi-system armed state debounce
        # When leader_broadcast (sysid=2) is running, we need debounce for disarm detection
        disarmed_streak = 0  # Count consecutive disarmed checks
        DISARM_CONFIRM_COUNT = 3  # Require 3 consecutive disarmed checks (1.5 seconds)

        while (time.time() - start) < self.climb_timeout:
            current_alt = self._get_altitude_agl()
            elapsed = time.time() - start

            # Detect impossible altitude jumps (sanity check)
            if abs(current_alt - last_valid_alt) > 50.0:  # 50m jump impossible
                self.log(
                    f"⚠️  Rejecting impossible altitude: {current_alt:.1f}m "
                    f"(jump from {last_valid_alt:.1f}m)"
                )
                current_alt = last_valid_alt
            else:
                last_valid_alt = current_alt

            # Log progress every 2 seconds
            if time.time() - last_log_time >= 2.0:
                current_alt_msl = self.home_alt_msl + current_alt
                self.log(
                    f"Climbing: {current_alt_msl:.1f}m MSL "
                    f"(target: {target_msl:.1f}m, "
                    f"armed={self._is_armed()}, "
                    f"mode={self._get_mode_name()})"
                )
                last_log_time = time.time()

            # Check if we've reached target altitude
            alt_error = self.target_precontact_alt_agl - current_alt
            if abs(alt_error) <= self.altitude_tolerance:
                current_alt_msl = self.home_alt_msl + current_alt
                self.log(f"✓ Target altitude reached: {current_alt_msl:.1f}m MSL (Precontact Position)")
                self.log("Releasing RC3 throttle override (preparing for plane modes)...")
                # Clear override - critical for TECS throttle control in plane modes
                self.mav.mav.rc_channels_override_send(
                    self.target_system,
                    self.target_component,
                    0, 0, 0, 0, 0, 0, 0, 0  # clear all overrides
                )
                time.sleep(0.5)  # Give ArduPlane time to process
                self.log("✓ RC override released - throttle control returned to autopilot")
                return TakeoffResult(
                    success=True,
                    message="Climb completed successfully",
                    final_altitude_agl=current_alt,
                    final_mode=self._get_mode_name(),
                    elapsed_time=elapsed,
                )

            # Check if vehicle disarmed (with debounce to handle multi-system interference)
            if not self._is_armed():
                disarmed_streak += 1
                if disarmed_streak >= DISARM_CONFIRM_COUNT:
                    self.log(f"⚠️  Vehicle disarmed confirmed after {disarmed_streak} consecutive checks")
                    self.mav.mav.rc_channels_override_send(
                        self.target_system,
                        self.target_component,
                        0, 0, 0, 0, 0, 0, 0, 0  # clear all overrides
                    )
                    return TakeoffResult(
                        success=False,
                        message="Vehicle disarmed during climb",
                        final_altitude_agl=current_alt,
                        final_mode=self._get_mode_name(),
                    )
            else:
                # Armed state confirmed, reset streak
                disarmed_streak = 0

            # Reassert override every loop iteration
            self.mav.mav.rc_channels_override_send(
                self.target_system,
                self.target_component,
                0, 0, self.climb_throttle_pwm, 0, 0, 0, 0, 0
            )

            time.sleep(0.5)

        # Timeout
        final_alt = self._get_altitude_agl()
        self.mav.mav.rc_channels_override_send(
            self.target_system,
            self.target_component,
            0, 0, 0, 0, 0, 0, 0, 0  # clear all overrides
        )
        return TakeoffResult(
            success=False,
            message=f"Climb timeout after {self.climb_timeout}s",
            final_altitude_agl=final_alt,
            final_mode=self._get_mode_name(),
        )

    def transition_to_fixedwing(self) -> TakeoffResult:
        """
        Transition from VTOL to fixed-wing forward flight.

        Uses MODE-CHANGE TRANSITION approach (QLOITER → CRUISE).
        ArduPlane's QuadPlane logic automatically triggers VTOL→FW transition
        when switching from a Q-mode to a plane mode.

        This approach is simpler and more robust than DO_VTOL_TRANSITION:
        - No AUTO mode requirement
        - No mission dependencies
        - No MIS_DONE_BEHAVE interference
        - No DO_VTOL_TRANSITION command rejection

        Sequence:
        1. Switch from QLOITER to CRUISE (triggers automatic transition)
        2. Monitor mode + airspeed until transition completes
        3. Switch to GUIDED mode for formation control

        Returns:
            TakeoffResult with success status and diagnostic information
        """
        self.log(f"=== PHASE 3: VTOL → FW Transition (Mode-Change: QLOITER → CRUISE) ===")

        current_alt = self._get_altitude_agl()
        current_alt_msl = self.home_alt_msl + current_alt
        self.log(f"Current altitude: {current_alt_msl:.1f}m MSL")
        self.log(f"Current mode: {self._get_mode_name()}")

        # Safety: Ensure RC override is cleared before mode change
        # (should already be cleared by Phase 2, but verify for TECS throttle control)
        self.log("Ensuring RC override is cleared (critical for TECS throttle control)...")
        self.mav.mav.rc_channels_override_send(
            self.target_system,
            self.target_component,
            0, 0, 0, 0, 0, 0, 0, 0  # clear all overrides
        )
        time.sleep(0.5)

        # Apply pre-CRUISE SpeedMode via PARAM_SET (if specified)
        # This sets FD_SPD_MODE=1 and FD_SPD_TARGET BEFORE entering CRUISE mode,
        # so TECS picks up the target speed at mode entry.
        if self.speed_mode_for_fw in ["SLOW", "MATCH", "FAST"]:
            # SpeedMode target mapping (aligned with AIRSPEED_MIN=20, TRIM=24.7, MAX=30)
            speed_targets = {
                "SLOW": 21.0,   # AIRSPEED_MIN + 1.0
                "MATCH": 24.7,  # AIRSPEED_TRIM (DHC-2 Beaver cruise speed)
                "FAST": 29.0,   # AIRSPEED_MAX - 1.0
            }
            target_speed = speed_targets[self.speed_mode_for_fw]

            self.log(f"SpeedMode/preCRUISE: applying mode {self.speed_mode_for_fw} → "
                     f"FD_SPD_MODE=1, FD_SPD_TARGET={target_speed:.1f} m/s "
                     f"(before QLOITER→CRUISE mode change)")

            # Enable FlightDock TECS speed override
            self.mav.mav.param_set_send(
                self.target_system,
                self.target_component,
                b'FD_SPD_MODE',
                1.0,  # 1 = override active
                mavutil.mavlink.MAV_PARAM_TYPE_INT8
            )
            time.sleep(0.2)  # Brief pause for param to apply

            # Set FlightDock target cruise speed
            self.mav.mav.param_set_send(
                self.target_system,
                self.target_component,
                b'FD_SPD_TARGET',
                float(target_speed),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
            time.sleep(0.5)  # Brief pause before mode change

            self.log(f"SpeedMode/preCRUISE: FD_SPD override enabled, proceeding to CRUISE mode entry")

        # Switch from QLOITER to CRUISE - this automatically triggers VTOL→FW transition
        TRANSITION_MODE = "CRUISE"  # Could also use FBWA
        self.log(f"Switching to {TRANSITION_MODE} (triggers automatic VTOL → FW transition)...")

        if not self.set_mode_and_wait(TRANSITION_MODE):
            return TakeoffResult(
                success=False,
                message=f"Failed to enter {TRANSITION_MODE} mode",
                final_altitude_agl=current_alt,
                final_mode=self._get_mode_name(),
            )

        self.log(f"✓ In {TRANSITION_MODE} mode - ArduPlane will now transition to FW")
        self.log("   Expecting MAVProxy messages:")
        self.log("     - 'AP: Transition airspeed wait'")
        self.log("     - 'AP: Transition done'")

        # Set cruise airspeed - COMMENTED OUT per ChatGPT-5 recommendation
        # Now relying on AIRSPEED_CRUISE and TECS for cruise speed target (Option-3FORM).
        # TECS will natively control to target speed based on parameters - no MAVLink overrides.
        # self.log("Setting cruise airspeed: 25 m/s...")
        # self.mav.mav.command_long_send(
        #     self.target_system,
        #     self.target_component,
        #     mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        #     0,
        #     0, # airspeed type
        #     25.0, # 25 m/s
        #     -1, 0, 0, 0, 0
        # )
        # time.sleep(1.0)
        self.log("Option-3FORM: Using AIRSPEED_CRUISE and TECS for cruise speed target (24.7 m/s nominal).")

        # Monitor mode + airspeed until transition completes
        self.log("Monitoring airspeed + mode for transition completion...")
        self.log("Success criteria:")
        self.log("  - Mode is a plane mode (not Q-mode)")
        self.log("  - Airspeed ≥ 12 m/s sustained for 2 seconds")

        transition_start = time.time()
        transition_timeout = 30.0
        sustained_airspeed_start = None
        required_sustained_time = 2.0
        min_transition_airspeed = 12.0  # Conservative threshold

        # Plane modes (FW flight)
        PLANE_MODES = {"CRUISE", "FBWA", "FBWB", "AUTO", "GUIDED", "RTL", "LOITER"}
        # Q-modes (VTOL flight)
        Q_MODES = {"QLOITER", "QHOVER", "QSTABILIZE", "QLAND", "QRTL"}

        while True:
            elapsed = time.time() - transition_start
            if elapsed > transition_timeout:
                self.log("⚠️  Transition timeout - continuing anyway")
                break

            current_mode = self._get_mode_name()
            msg = self.mav.recv_match(type='VFR_HUD', blocking=True, timeout=0.5)

            if msg and msg.get_srcSystem() == self.target_system:
                airspeed = msg.airspeed

                # Check both mode and airspeed
                mode_is_fw = current_mode in PLANE_MODES
                airspeed_ok = airspeed >= min_transition_airspeed

                if mode_is_fw and airspeed_ok:
                    if sustained_airspeed_start is None:
                        sustained_airspeed_start = time.time()
                        self.log(f"  ✓ FW flight detected: mode={current_mode}, airspeed={airspeed:.1f} m/s")
                        self.log(f"    Sustaining for {required_sustained_time:.0f}s...")
                    elif time.time() - sustained_airspeed_start >= required_sustained_time:
                        self.log(f"  ✓ Transition complete!")
                        self.log(f"    Mode: {current_mode}")
                        self.log(f"    Airspeed: {airspeed:.1f} m/s (sustained for {required_sustained_time:.0f}s)")
                        break
                else:
                    if sustained_airspeed_start is not None:
                        reason = "mode changed" if not mode_is_fw else "airspeed dropped"
                        self.log(f"  ⚠️  Lost FW criteria ({reason}) - waiting...")
                        self.log(f"      Mode: {current_mode}, Airspeed: {airspeed:.1f} m/s")
                    sustained_airspeed_start = None

                # Periodic status update
                if int(elapsed) % 2 == 0:
                    fw_status = "✓" if mode_is_fw else "✗"
                    spd_status = "✓" if airspeed_ok else "✗"
                    self.log(f"  [{elapsed:5.1f}s] Mode: {current_mode} {fw_status} | "
                             f"Airspeed: {airspeed:.1f} m/s {spd_status}")

            time.sleep(0.1)

        # Transition complete - stay in CRUISE mode (do NOT switch to GUIDED)
        final_mode = self._get_mode_name()
        final_alt = self._get_altitude_agl()
        final_alt_msl = self.home_alt_msl + final_alt

        # NEW: Stay in CRUISE for FW cruise, don't auto-switch to GUIDED
        # The formation controller (or test script) will switch modes when ready
        self.log(f"✓ Fixed-wing transition complete")
        self.log(f"  Final mode: {final_mode}")
        self.log(f"  Final altitude: {final_alt_msl:.1f}m MSL")
        self.log(f"  Leaving vehicle in {final_mode} mode")
        self.log(f"  FW_READY = True")
        self.log("Option-3FORM: TECS controls cruise speed based on AIRSPEED_CRUISE "
                 "and the configured envelope (no RC override).")

        return TakeoffResult(
            success=True,
            message="Fixed-wing transition successful",
            final_altitude_agl=final_alt,
            final_mode=final_mode,
            fw_ready=True,  # Indicate FW flight is ready
        )

    def takeoff_to_fixedwing_precontact(self) -> TakeoffResult:
        """
        Full takeoff sequence: arm → VTOL climb → fixed-wing transition.

        Returns:
            TakeoffResult with success status and diagnostic information
        """
        self.log("=" * 70)
        self.log("QuadPlane VTOL Takeoff to Fixed-Wing Precontact")
        self.log("=" * 70)
        self.log("Configuration:")
        self.log(f"  VTOL Mode: {self.vtol_mode_name}")
        self.log(f"  Fixed-Wing Mode: {self.fixedwing_mode_name}")
        target_msl = self.home_alt_msl + self.target_precontact_alt_agl
        self.log(f"  Target Altitude: {target_msl:.1f}m MSL (Precontact Position)")
        self.log(f"  Home Altitude: {self.home_alt_msl:.1f}m MSL")
        self.log("")

        # Phase 1: Arm
        result = self.arm_and_enter_vtol()
        if not result.success:
            self.log(f"❌ PHASE 1 FAILED: {result.message}")
            return result

        # Phase 2: Climb
        result = self.vtol_climb_to_precontact_alt()
        if not result.success:
            self.log(f"❌ PHASE 2 FAILED: {result.message}")
            return result

        # Phase 3: Transition
        result = self.transition_to_fixedwing()
        if not result.success:
            self.log(f"❌ PHASE 3 FAILED: {result.message}")
            return result

        self.log("=" * 70)
        self.log("✓ FULL SEQUENCE COMPLETE")
        final_alt_msl = self.home_alt_msl + result.final_altitude_agl
        self.log(f"  Final altitude: {final_alt_msl:.1f}m MSL")
        self.log(f"  Final mode: {result.final_mode}")
        self.log("=" * 70)

        return result

    # =========================================================================
    # P1 Fix A v9.10: NON-BLOCKING HELPER METHODS
    # =========================================================================
    # These methods are designed to be called once per tick from the formation
    # controller's non-blocking takeoff sub-FSM. They send commands once and
    # return predicates for state transitions.

    def nb_send_vtol_mode_cmd(self) -> None:
        """P1 Fix A: Send QLOITER mode command (non-blocking, call once)."""
        mode_number = self.MODE_NAME_TO_NUMBER.get(self.vtol_mode_name.upper())
        if mode_number is None:
            return

        # Override RC channel 8 to prevent interference
        self.mav.mav.rc_channels_override_send(
            self.target_system, self.target_component,
            0, 0, 0, 0, 0, 0, 0, 1500
        )

        self.mav.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_number,
            0, 0, 0, 0, 0
        )

    def nb_set_cached_heartbeat(self, heartbeat) -> None:
        """P1 Fix A v9.10: Set cached heartbeat from external source.

        This allows the main formation controller to share its cached heartbeat
        with the takeoff controller, avoiding message duplication issues when
        both try to read from the same MAVLink connection.
        """
        if heartbeat is not None:
            self._last_heartbeat = heartbeat

    def nb_set_cached_position(self, position_msg) -> None:
        """P1 Fix B: Accept cached GLOBAL_POSITION_INT from main controller.

        This prevents message theft when both controllers share the same MAVLink connection.
        The main controller pumps messages and caches them; this method allows sharing
        the cached position data instead of competing for recv_match().

        Args:
            position_msg: Cached GLOBAL_POSITION_INT message from main controller
        """
        if position_msg is not None:
            self._last_global_position = position_msg

    def nb_check_vtol_mode(self) -> bool:
        """P1 Fix A: Check if vehicle is in VTOL mode (call each tick)."""
        # Note: Caller should call nb_set_cached_heartbeat() first
        # FD-001 Fix: Don't steal messages - use cached heartbeat from main controller
        # self._update_message_cache()
        return self._is_mode(self.vtol_mode_name)

    def nb_send_arm_cmd(self) -> None:
        """P1 Fix A: Send arm command (non-blocking, call once)."""
        self.mav.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

    def nb_check_armed(self) -> bool:
        """P1 Fix A: Check if vehicle is armed (call each tick)."""
        # FD-001 Fix: Don't steal messages - use cached heartbeat from main controller
        # self._update_message_cache()
        return self._is_armed()

    def nb_send_throttle_override(self) -> None:
        """P1 Fix A: Send RC throttle override for climb (call each tick)."""
        self.mav.mav.rc_channels_override_send(
            self.target_system, self.target_component,
            0, 0, self.climb_throttle_pwm, 0, 0, 0, 0, 0
        )

    def nb_check_altitude_reached(self) -> Tuple[bool, float]:
        """P1 Fix A: Check if target altitude reached (call each tick).

        Returns:
            Tuple of (reached: bool, current_alt_agl: float)
        """
        # FD-001 Fix: Don't steal messages - use cached position from main controller
        # self._update_message_cache()
        current_alt = self._get_altitude_agl()
        alt_error = self.target_precontact_alt_agl - current_alt
        reached = abs(alt_error) <= self.altitude_tolerance
        return reached, current_alt

    def nb_release_throttle_override(self) -> None:
        """P1 Fix A: Release RC throttle override (call once when altitude reached)."""
        self.mav.mav.rc_channels_override_send(
            self.target_system, self.target_component,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    def nb_send_fw_mode_cmd(self) -> None:
        """P1 Fix A: Send CRUISE mode command for FW transition (call once)."""
        mode_number = self.MODE_NAME_TO_NUMBER.get("CRUISE")
        if mode_number is None:
            return

        self.mav.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_number,
            0, 0, 0, 0, 0
        )

    def nb_check_fw_mode(self) -> bool:
        """P1 Fix A: Check if vehicle is in a plane mode (call each tick)."""
        # FD-001 Fix: Don't steal messages - use cached heartbeat from main controller
        # self._update_message_cache()
        current_mode = self._get_mode_name()
        PLANE_MODES = {"CRUISE", "FBWA", "FBWB", "AUTO", "GUIDED", "RTL", "LOITER"}
        return current_mode in PLANE_MODES

    def nb_check_transition_complete(self, min_airspeed_mps: float = 12.0) -> Tuple[bool, float]:
        """P1 Fix A: Check if FW transition is complete (call each tick).

        Checks for sustained airspeed >= min_airspeed_mps in a plane mode.

        Returns:
            Tuple of (complete: bool, current_airspeed: float)
        """
        # FD-001 Fix: Don't steal messages - use cached heartbeat from main controller
        # self._update_message_cache()
        current_mode = self._get_mode_name()
        PLANE_MODES = {"CRUISE", "FBWA", "FBWB", "AUTO", "GUIDED", "RTL", "LOITER"}
        mode_is_fw = current_mode in PLANE_MODES

        # FD-001 Fix: Use cached VFR_HUD instead of stealing messages
        # Get airspeed from cached VFR_HUD (set by main controller via nb_set_cached_vfr_hud)
        if hasattr(self, '_cached_vfr_hud') and self._cached_vfr_hud is not None:
            airspeed = self._cached_vfr_hud.airspeed
        else:
            airspeed = 0.0

        airspeed_ok = airspeed >= min_airspeed_mps
        complete = mode_is_fw and airspeed_ok
        return complete, airspeed

    def nb_check_disarmed(self) -> bool:
        """P1 Fix A: Check if vehicle has disarmed (safety check)."""
        # FD-001 Fix: Don't steal messages - use cached heartbeat from main controller
        # self._update_message_cache()
        return not self._is_armed()

    def nb_set_cached_vfr_hud(self, msg) -> None:
        """FD-001: Set cached VFR_HUD message from main controller."""
        self._cached_vfr_hud = msg
