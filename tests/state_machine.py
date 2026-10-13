#!/usr/bin/env python3
"""
FlightDock State Machine Module (Phase 3 Refactor)
===================================================

Table-driven state machine with explicit entry/exit actions, handler
dispatch, and transition audit trail.

Replaces the scattered transition_to() / if-elif dispatch chain in
baseline_formation_controller.py with a centralized StateMachine class.

Design decisions:
- Entry/exit actions are functions in lookup tables (not subclass overrides)
- Handlers remain as methods on the controller (called via dispatch table)
- skip_exit flag handles cases where handlers pre-run exit cleanup
- TransitionRecord provides a full audit trail

STATE TRANSITION RULES:
    The formation controller follows this nominal progression:

    INIT -> WEATHER_CHECK -> VTOL_TAKEOFF -> FW_TRANSITION -> FW_STABILIZE
         -> INTERCEPT -> LEAD_TURN -> STERN_CONVERSION -> REJOIN_PRECONTACT
         -> HOLD_PRECONTACT -> ALTITUDE_CAPTURE -> DOCKING_CREEP
         -> HOLD_CONTACT -> SUCCESS

    Exception transitions (can occur from any non-terminal state):
    - Any state -> EMERGENCY_BREAKAWAY (safety limit exceeded)
    - Any state -> DOCKING_ABORT (envelope violation or timeout)
    - Any state -> FAIL (unrecoverable error)

    Retry transitions:
    - HOLD_PRECONTACT -> REJOIN_PRECONTACT (re-attempt precontact)
    - DOCKING_CREEP -> REJOIN_PRECONTACT (creep abort, retry from rejoin)
    - HOLD_CONTACT -> REJOIN_PRECONTACT (contact lost, re-acquire)

    Terminal states: SUCCESS, FAIL (no further transitions allowed)

USAGE:
    from state_machine import StateMachine

    # In controller __init__:
    self.sm = StateMachine(self, log_fn=log)

    # Replace transition_to:
    def transition_to(self, new_state, reason="", skip_exit=False):
        self.sm.transition_to(new_state, reason, skip_exit=skip_exit)

    # Replace run_state_machine:
    def run_state_machine(self):
        if not self.sm.tick():
            self.running = False

(C) 2025-2026 FlightDock Research Project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "ControllerState",
    "TransitionRecord",
    "StateMachineState",
    "StateMachine",
    "HANDLER_TABLE",
    "TERMINAL_STATES",
    "STATE_ENTRY_ACTIONS",
    "STATE_EXIT_ACTIONS",
    "UNIVERSAL_ENTRY_ACTIONS",
]


# =============================================================================
# STATE ENUM (canonical definition, re-exported for convenience)
# =============================================================================
class ControllerState(Enum):
    """Controller state machine states.

    Fighter intercept flow:
    INTERCEPT -> LEAD_TURN -> STERN_CONVERSION -> REJOIN_PRECONTACT

    - INTERCEPT: Maintain lateral offset, fly towards intercept point
    - LEAD_TURN: Execute turn at TA~130 deg to roll out behind target
    - STERN_CONVERSION: Fine-tune position at 200m behind target

    P1 Fix A v9.10: Non-blocking VTOL takeoff sub-FSM states
    TAKEOFF_SET_VTOL_MODE -> TAKEOFF_ARMING -> TAKEOFF_CLIMBING
    -> TAKEOFF_SET_FW_MODE -> TAKEOFF_TRANSITION_MONITOR -> FW_TRANSITION
    """

    INIT = auto()
    WEATHER_CHECK = auto()
    VTOL_TAKEOFF = auto()  # Legacy (kept for compatibility, transitions immediately)

    # P1 Fix A v9.10: Non-blocking takeoff sub-states
    TAKEOFF_SET_VTOL_MODE = auto()  # Send QLOITER mode, wait for confirmation
    TAKEOFF_ARMING = auto()  # Send arm cmd, wait for armed state
    TAKEOFF_CLIMBING = auto()  # RC throttle override, wait for altitude
    TAKEOFF_SET_FW_MODE = auto()  # Send CRUISE mode, wait for confirmation
    TAKEOFF_TRANSITION_MONITOR = auto()  # Monitor airspeed until FW flight

    FW_TRANSITION = auto()
    FW_STABILIZE = auto()
    INTERCEPT = auto()
    LEAD_TURN = auto()
    STERN_CONVERSION = auto()
    REJOIN_PRECONTACT = auto()
    HOLD_PRECONTACT = auto()
    ALTITUDE_CAPTURE = auto()
    ALT_CAP_HOLD_RANGE = auto()  # P1 v9.4: Boom-tip vertical alignment
    DOCKING_CREEP = auto()
    HOLD_CONTACT = auto()
    SUCCESS = auto()
    EMERGENCY_BREAKAWAY = auto()
    DOCKING_ABORT = auto()
    FAIL = auto()


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class TransitionRecord:
    """Audit trail entry for a single state transition."""

    sim_time_s: float
    from_state: ControllerState
    to_state: ControllerState
    reason: str
    entry_actions_run: List[str]
    exit_actions_run: List[str]


@dataclass
class StateMachineState:
    """Internal state machine bookkeeping."""

    current: ControllerState = ControllerState.INIT
    entry_time_s: float = 0.0
    entry_count: Dict[str, int] = field(default_factory=dict)
    transition_count: int = 0


# =============================================================================
# ACTION TYPE
# =============================================================================
# An action is a callable: (controller, sim_now_s, other_state) -> description_str
# other_state is the state being transitioned FROM (for entry) or TO (for exit)
ActionFn = Callable[[Any, float, ControllerState], str]


# =============================================================================
# UNIVERSAL ENTRY ACTIONS (run on ALL transitions)
# =============================================================================
def _entry_reset_state_timer(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Reset state_entry_time to current sim time."""
    ctrl.state_entry_time = sim_now
    return "reset_state_timer"


def _entry_increment_state_count(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Increment entry count for the new state."""
    state_name = ctrl.state.name
    ctrl._state_entry_counts[state_name] = ctrl._state_entry_counts.get(state_name, 0) + 1
    return "increment_state_count"


UNIVERSAL_ENTRY_ACTIONS: List[ActionFn] = [
    _entry_reset_state_timer,
    _entry_increment_state_count,
]


# =============================================================================
# PER-STATE ENTRY ACTIONS
# =============================================================================
def _entry_fw_transition_init(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Initialize FW_TRANSITION tracking variables."""
    ctrl._fw_trans_sustained_start = None
    ctrl._fw_trans_retry_count = 0
    return "fw_transition_init"


def _entry_fw_stabilize_init(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Initialize FW_STABILIZE guided mode tracking."""
    ctrl._guided_mode_confirmed = False
    return "fw_stabilize_init"


def _entry_altitude_capture_init(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Clear ALTITUDE_CAPTURE timers for fresh capture attempt."""
    ctrl._alt_capture_start_time = None
    ctrl._alt_capture_stable_start = None
    ctrl._alt_capture_standoff_stable_start = None
    return "altitude_capture_init"


def _entry_hold_contact_init(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Initialize HOLD_CONTACT: increment counter, reset integrator."""
    ctrl.contact_entry_count += 1
    ctrl.current_contact_entry_time = sim_now
    ctrl._reset_hold_contact_integrator()
    return "hold_contact_init"


def _entry_rejoin_precontact_init(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Clear NO_PASS_ABORT state on entering REJOIN_PRECONTACT."""
    ctrl._no_pass_abort_start = None
    ctrl._no_pass_abort_triggered = False
    return "rejoin_precontact_init"


def _entry_alt_cap_hold_range_init(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Clear ALT_CAP_HOLD_RANGE accumulators for fresh capture attempt."""
    ctrl._alt_cap_stable_s = 0.0
    ctrl._alt_cap_violation_accum_s = 0.0
    ctrl._alt_cap_log_accum_s = 0.0
    return "alt_cap_hold_range_init"


def _entry_docking_creep_init(ctrl: Any, sim_now: float, from_state: ControllerState) -> str:
    """Option A: Reset sub-phase to VERT_STABILIZE on DOCKING_CREEP entry."""
    ctrl._dc_phase = "DC_PHASE1_VERT_STABILIZE"
    ctrl._dc_stab_accum_s = 0.0
    ctrl._dc_phase2_lockout = False
    ctrl._dc_sink_bad_accum_s = 0.0
    ctrl._dc_phase1_enter_count += 1
    ctrl.control_laws.reset_closure_deficit_integrator()  # Fix 3B: Clear stale integrator state
    ctrl._tecs_spdw_ramp_active = False
    return "docking_creep_phase_init"


STATE_ENTRY_ACTIONS: Dict[ControllerState, List[ActionFn]] = {
    ControllerState.FW_TRANSITION: [_entry_fw_transition_init],
    ControllerState.FW_STABILIZE: [_entry_fw_stabilize_init],
    ControllerState.ALTITUDE_CAPTURE: [_entry_altitude_capture_init],
    ControllerState.HOLD_CONTACT: [_entry_hold_contact_init],
    ControllerState.REJOIN_PRECONTACT: [_entry_rejoin_precontact_init],
    ControllerState.ALT_CAP_HOLD_RANGE: [_entry_alt_cap_hold_range_init],
    ControllerState.DOCKING_CREEP: [_entry_docking_creep_init],
}


# =============================================================================
# PER-STATE EXIT ACTIONS
#
# Note on try/except/pass pattern: The hasattr() check guards against missing
# attributes, but a race condition could delete the attribute between the check
# and the del. The try/except silently handles this edge case. This is
# intentional defensive programming, not a stub that needs implementation.
# =============================================================================
def _exit_fw_transition_cleanup(ctrl: Any, sim_now: float, to_state: ControllerState) -> str:
    """Clean up FW_TRANSITION tracking variables."""
    if hasattr(ctrl, "_fw_trans_sustained_start"):
        try:
            del ctrl._fw_trans_sustained_start
        except AttributeError:
            pass  # Attribute deleted between hasattr check and del (rare race)
    if hasattr(ctrl, "_fw_trans_retry_count"):
        try:
            del ctrl._fw_trans_retry_count
        except AttributeError:
            pass  # Attribute deleted between hasattr check and del (rare race)
    return "fw_transition_cleanup"


def _exit_fw_stabilize_cleanup(ctrl: Any, sim_now: float, to_state: ControllerState) -> str:
    """Clean up FW_STABILIZE guided mode flag."""
    if hasattr(ctrl, "_guided_mode_confirmed"):
        try:
            del ctrl._guided_mode_confirmed
        except AttributeError:
            pass  # Attribute deleted between hasattr check and del (rare race)
    return "fw_stabilize_cleanup"


def _exit_altitude_capture_cleanup(ctrl: Any, sim_now: float, to_state: ControllerState) -> str:
    """Full ALTITUDE_CAPTURE exit: restore TECS/QASSIST, clear timers.

    Delegates to the controller's existing _exit_altitude_capture() method
    which handles TECS/QASSIST restoration and timer cleanup.
    """
    ctrl._exit_altitude_capture("state_machine_exit")
    return "altitude_capture_cleanup"


def _exit_hold_contact_cleanup(ctrl: Any, sim_now: float, to_state: ControllerState) -> str:
    """Full HOLD_CONTACT exit: dwell accounting, integrator/bias reset.

    Delegates to the controller's existing _exit_hold_contact() method.
    """
    ctrl._exit_hold_contact("state_machine_exit")
    return "hold_contact_cleanup"


def _exit_docking_creep_cleanup(ctrl: Any, sim_now: float, to_state: ControllerState) -> str:
    """Option E: Restore TECS_SPDWEIGHT to cruise on DOCKING_CREEP exit."""
    ctrl._exit_docking_creep("state_machine_exit")
    return "docking_creep_tecs_restore"


STATE_EXIT_ACTIONS: Dict[ControllerState, List[ActionFn]] = {
    ControllerState.FW_TRANSITION: [_exit_fw_transition_cleanup],
    ControllerState.FW_STABILIZE: [_exit_fw_stabilize_cleanup],
    ControllerState.ALTITUDE_CAPTURE: [_exit_altitude_capture_cleanup],
    ControllerState.HOLD_CONTACT: [_exit_hold_contact_cleanup],
    ControllerState.DOCKING_CREEP: [_exit_docking_creep_cleanup],
}


# =============================================================================
# HANDLER DISPATCH TABLE
# =============================================================================
HANDLER_TABLE: Dict[ControllerState, str] = {
    ControllerState.WEATHER_CHECK: "run_weather_check",
    ControllerState.VTOL_TAKEOFF: "run_vtol_takeoff",
    # P1 Fix A v9.10: Non-blocking takeoff sub-states
    ControllerState.TAKEOFF_SET_VTOL_MODE: "_handle_takeoff_set_vtol_mode",
    ControllerState.TAKEOFF_ARMING: "_handle_takeoff_arming",
    ControllerState.TAKEOFF_CLIMBING: "_handle_takeoff_climbing",
    ControllerState.TAKEOFF_SET_FW_MODE: "_handle_takeoff_set_fw_mode",
    ControllerState.TAKEOFF_TRANSITION_MONITOR: "_handle_takeoff_transition_monitor",
    ControllerState.FW_TRANSITION: "run_fw_transition",
    ControllerState.FW_STABILIZE: "run_fw_stabilize",
    ControllerState.INTERCEPT: "run_intercept",
    ControllerState.LEAD_TURN: "run_lead_turn",
    ControllerState.STERN_CONVERSION: "run_stern_conversion",
    ControllerState.REJOIN_PRECONTACT: "run_rejoin_precontact",
    ControllerState.HOLD_PRECONTACT: "run_hold_precontact",
    ControllerState.ALTITUDE_CAPTURE: "run_altitude_capture",
    ControllerState.ALT_CAP_HOLD_RANGE: "run_alt_cap_hold_range",
    ControllerState.DOCKING_CREEP: "run_docking_creep",
    ControllerState.HOLD_CONTACT: "run_hold_contact",
    # v9.10 Gate 1: Add EMERGENCY_BREAKAWAY handler for no-pass recovery
    # Note: Controller uses _handle_emergency_breakaway via its own dispatch table
    ControllerState.EMERGENCY_BREAKAWAY: "_handle_emergency_breakaway",
}

TERMINAL_STATES = {ControllerState.SUCCESS, ControllerState.FAIL}


# =============================================================================
# STATE MACHINE CLASS
# =============================================================================
class StateMachine:
    """Table-driven state machine with entry/exit actions and audit trail.

    Centralizes:
    - State transitions with entry/exit action execution
    - Handler dispatch via table lookup
    - Transition audit trail for debugging
    """

    def __init__(self, controller: Any, log_fn: Optional[Callable[[str], None]] = None) -> None:
        """Initialize state machine.

        Args:
            controller: The formation controller instance (provides handler methods
                        and state variables).
            log_fn: Logging function. Defaults to no-op.
        """
        self._ctrl = controller
        self._log = log_fn or (lambda msg: None)
        self._state = StateMachineState()
        self._transition_log: List[TransitionRecord] = []

    @property
    def current_state(self) -> ControllerState:
        """Current state."""
        return self._state.current

    @current_state.setter
    def current_state(self, value: ControllerState) -> None:
        """Set current state (for initialization only, bypasses actions)."""
        self._state.current = value

    @property
    def entry_time_s(self) -> float:
        """Time the current state was entered."""
        return self._state.entry_time_s

    @property
    def transition_count(self) -> int:
        """Total number of transitions executed."""
        return self._state.transition_count

    def transition_to(
        self, new_state: ControllerState, reason: str = "", skip_exit: bool = False
    ) -> None:
        """Execute state transition with entry/exit actions.

        Args:
            new_state: Target state to transition to.
            reason: Human-readable reason for audit trail.
            skip_exit: If True, skip exit actions for the old state. Use when
                       the handler has already run its own exit cleanup (e.g.,
                       _exit_hold_contact called before transition).
        """
        old_state = self._state.current
        sim_now = self._get_sim_now()

        # 1. EXIT actions (unless skip_exit or same-state)
        exit_actions_run: List[str] = []
        if not skip_exit and old_state != new_state and old_state in STATE_EXIT_ACTIONS:
            for action in STATE_EXIT_ACTIONS[old_state]:
                desc = action(self._ctrl, sim_now, new_state)
                exit_actions_run.append(desc)

        # 2. Update state on both SM and controller
        self._state.current = new_state
        self._ctrl.state = new_state

        # 3. UNIVERSAL entry actions
        entry_actions_run: List[str] = []
        for action in UNIVERSAL_ENTRY_ACTIONS:
            desc = action(self._ctrl, sim_now, old_state)
            entry_actions_run.append(desc)

        # 4. STATE-SPECIFIC entry actions
        if new_state in STATE_ENTRY_ACTIONS:
            for action in STATE_ENTRY_ACTIONS[new_state]:
                desc = action(self._ctrl, sim_now, old_state)
                entry_actions_run.append(desc)

        # 5. Update bookkeeping
        self._state.entry_time_s = sim_now
        self._state.transition_count += 1

        # 6. Log transition
        self._log(f"STATE: {old_state.name} -> {new_state.name} ({reason})")

        # 7. Audit trail
        self._transition_log.append(
            TransitionRecord(
                sim_time_s=sim_now,
                from_state=old_state,
                to_state=new_state,
                reason=reason,
                entry_actions_run=entry_actions_run,
                exit_actions_run=exit_actions_run,
            )
        )

    def tick(self) -> bool:
        """Execute one handler tick for the current state.

        Returns:
            True if the state machine should continue running.
            False if a terminal state (SUCCESS/FAIL) has been reached.
        """
        state = self._state.current
        if state in TERMINAL_STATES:
            if state == ControllerState.SUCCESS:
                self._log(
                    f"SUCCESS! Minimum range achieved: " f"{self._ctrl.min_range_achieved:.2f}m"
                )
            elif state == ControllerState.FAIL:
                self._log(
                    f"FAILED. Minimum range achieved: " f"{self._ctrl.min_range_achieved:.2f}m"
                )
            return False

        handler_name = HANDLER_TABLE.get(state)
        if handler_name:
            handler = getattr(self._ctrl, handler_name, None)
            if handler:
                handler()
            else:
                self._log(f"WARNING: No handler method '{handler_name}' on controller")
        else:
            self._log(f"WARNING: No handler registered for state {state.name}")

        # Check if handler caused a terminal transition
        return self._state.current not in TERMINAL_STATES

    def get_transition_log(self) -> List[TransitionRecord]:
        """Get full transition audit trail."""
        return list(self._transition_log)

    def get_last_transition(self) -> Optional[TransitionRecord]:
        """Get the most recent transition record."""
        return self._transition_log[-1] if self._transition_log else None

    def get_entry_count(self, state: ControllerState) -> int:
        """Get the number of times a state has been entered."""
        return self._state.entry_count.get(state.name, 0)

    def _get_sim_now(self) -> float:
        """Get sim time from controller tick timestamp (single-source).

        v9.10: Use tick_time_s() instead of sim_now_s() to ensure single-source
        timebase. This returns the timestamp of the most recent EXECUTED tick,
        not raw sim-time which could have advanced since the tick started.

        For unit testing, mock controllers should implement tick_time_s() or
        use the fallback _t_sim_s attribute.
        """
        # v9.10: Prefer tick_time_s() (safe single-source accessor)
        if hasattr(self._ctrl, "tick_time_s"):
            return self._ctrl.tick_time_s()
        # Fallback for unit testing with mock controllers
        if hasattr(self._ctrl, "_t_sim_s"):
            return self._ctrl._t_sim_s
        if hasattr(self._ctrl, "_sim_time"):
            return self._ctrl._sim_time
        return 0.0
