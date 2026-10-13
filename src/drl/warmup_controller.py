"""Warmup controller — runs baseline to HOLD_PRECONTACT (M2), then stops.

Subclass of BaselineFormationController that overrides transition_to()
to detect M2 arrival and exit the main loop cleanly. After run_warmup()
returns, the caller takes ownership of MAVLink connections, time authority,
geometry calculator, and last-known telemetry state.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

# --- Path setup (same as __init__.py) ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TESTS_DIR = os.path.join(_REPO_ROOT, "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from baseline_formation_controller import BaselineFormationController
from state_machine import ControllerState


class WarmupController(BaselineFormationController):
    """Subclass that stops the main loop when HOLD_PRECONTACT (M2) is reached.

    After ``run_warmup()`` returns, the caller takes ownership of:
      - ``self.follower`` — MAVConnection (with cached telemetry)
      - ``self.leader_rx`` — LeaderTelemetryReceiver (with ``last_state``)
      - ``self._time_auth`` — SimTimeAuthority (already global)
      - ``self.geom_calc`` — GeometryCalculator (with correct home position)
      - ``self.control_laws`` — FormationControlLaws (with correct speedup_factor)
      - ``self.home_lat``, ``self.home_lon``, ``self.home_alt_msl``
    """

    def __init__(
        self,
        follower_conn_str: str,
        speedup: float,
        leader_port: int = 14552,
        out_dir: Optional[str] = None,
        stop_state: str = "M2",
    ) -> None:
        # BaselineFormationController.__init__ creates FormationControlLaws
        # with expected_speedup=1.0 (line 2105/2181). We fix that below.
        super().__init__(follower_conn_str=follower_conn_str)

        # Override properties that main() normally sets from CLI args
        # (matches lines 9865-9869 of baseline_formation_controller.py)
        self.expected_speedup = speedup
        self.control_laws.speedup_factor = max(1.0, speedup)
        self._leader_port = leader_port
        self._cpp_handoff_enabled = False
        self._cpp_handoff_policy = "always_off"
        self._out_dir = out_dir
        self._warmup_reached_target = False
        self._stop_states = self._resolve_stop_states(stop_state)
        self._sim_time_reporter: Optional[Callable[[float], None]] = None

    @staticmethod
    def _resolve_stop_states(target: str) -> frozenset:
        """Map target shorthand to the set of ControllerStates that end warmup."""
        if target == "M1":
            return frozenset({ControllerState.REJOIN_PRECONTACT})
        elif target == "M2":
            return frozenset({ControllerState.HOLD_PRECONTACT})
        else:
            raise ValueError(
                f"Unknown warmup stop_state: {target!r}. Expected 'M1' or 'M2'."
            )

    def _execute_tick(self, queue_time_boot_ms=None) -> bool:
        """Override to report sim-time progress for conditional heartbeat (v13)."""
        result = super()._execute_tick(queue_time_boot_ms)
        if result and self._sim_time_reporter is not None:
            try:
                self._sim_time_reporter(self._t_sim_s)
            except Exception:
                pass  # Never let reporter failure break warmup
        return result

    def transition_to(
        self,
        new_state: ControllerState,
        reason: str = "",
        skip_exit: bool = False,
    ) -> None:
        """Override to detect target state and exit the main loop."""
        super().transition_to(new_state, reason, skip_exit)
        if new_state in self._stop_states:
            self._warmup_reached_target = True
            self.running = False  # Breaks the ``while self.running`` loop

    def run_warmup(
        self,
        max_sim_seconds: float = 600.0,
        sim_time_reporter: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Run baseline to target state (M1 or M2), then return.

        Call chain: ``run_warmup()`` → ``run()`` → ``connect()`` (7-step
        preflight) → main control loop → ``transition_to(target_state)``
        → ``self.running = False``.

        The target state is configured via the ``stop_state`` parameter
        passed to ``__init__``:
          - ``"M1"`` → stops at ``REJOIN_PRECONTACT`` (~190 m behind leader)
          - ``"M2"`` → stops at ``HOLD_PRECONTACT`` (~10 m behind leader)

        Args:
            max_sim_seconds: Maximum sim-time before timeout (default 600 s,
                enough for M1 staging ~450 s + M2 approach ~60 s at 2x).
            sim_time_reporter: Optional callback invoked each tick with the
                current sim-time (v13 Commit 11 — conditional heartbeat).

        Raises:
            TimeoutError: If target state not reached within *max_sim_seconds*.
        """
        # v13 Commit 11: Store reporter for tick-level callback
        self._sim_time_reporter = sim_time_reporter
        # PRH-ATT-004 Fix 3A: Skip Hz watchdog during DRL warmup.
        # The watchdog is a runtime safety mechanism for the v9.10 formation
        # controller — timing dips during warmup INTERCEPT are non-critical.
        self._drl_mode = True
        self.run(skip_takeoff=False, max_sim_seconds=max_sim_seconds)
        if not self._warmup_reached_target:
            raise TimeoutError(
                f"Warmup failed to reach target state "
                f"(final state: {self.state})"
            )
        # Switch time authority to DRL mode (never abort on timing issues)
        if self._time_auth is not None:
            self._time_auth.drl_mode = True
