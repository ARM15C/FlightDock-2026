"""DRL controller reference — full FSM baseline as a callable library.

Subclass of BaselineFormationController that overrides MAVLink output
to capture commands instead of transmitting. After warmup reaches M2,
the controller stays alive and drives tick-by-tick from the DRL env
via ``tick_once()``.

Key differences from WarmupController:
  - Does NOT stop at M2. Instead activates DRL mode (command capture).
  - FSM transitions naturally through DOCKING_CREEP → HOLD_CONTACT.
  - ``send_guided_commands()`` captures hdg/alt/spd, monkey-patched
    follower methods suppress actual MAVLink sends.
  - ``_send_speed_command()`` suppresses DoubleSendError in DRL mode.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from typing import Callable, Optional

# --- Path setup (same as warmup_controller.py) ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TESTS_DIR = os.path.join(_REPO_ROOT, "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from baseline_formation_controller import (
    BaselineFormationController,
    LeaderTelemetryReceiver,
    log,
)
from state_machine import ControllerState
from .residual_action import BaselineCommand

_FD_DRLREF_POSTTICK_PARITY = os.environ.get(
    "FD_DRLREF_POSTTICK_PARITY", "0").strip() == "1"

# T-396 (GO A): verify the station geometry before accepting the frozen FSM's
# M2 declaration.
#
# The frozen controller declares M2 on ENTRY into HOLD_PRECONTACT, and that
# entry criterion is the FSM CAPTURE box (+/-4.0 behind, +/-8.0 cross, +/-5.0
# below about the 10 m pre-contact station) -- deliberately wide, and far
# wider than the START-LINE box the scored window is supposed to open inside
# (+/-1.5 behind, +/-2.0 cross, +/-4.0 below, formation_config
# PRECONTACT_*_OUTER_TOL_M). So "in position" could be declared while still
# ~3 m aft, and the scored window opened there: +2.97 m median across 33 legs
# (START_OFFSET_CENSUS.json), 26/33 outside the start-line box.
#
# R7.1 is untouched: the FSM, its entry criterion and every transition are
# exactly as they were. What changes is only WHEN THIS SUBCLASS ACCEPTS the
# declaration. Withholding the latch leaves ``self.running`` True, so the
# frozen controller simply keeps flying itself to station on its OWN
# transmission path with every guard live -- no env-side commanding, no new
# failure path. If it never converges, run() ends on its existing sim-time
# limit and run_warmup() raises the TimeoutError it already raises, which the
# env's existing retry loop already handles.
#
# Ships OFF. Armed by the T-396 validation smoke; default-ON is a separate
# that requires that smoke's evidence.
_FD_M2_GEOM_VERIFY = os.environ.get("FD_M2_GEOM_VERIFY", "0").strip() == "1"


class DRLControllerRef(BaselineFormationController):
    """Full FSM baseline controller driven tick-by-tick from the DRL env.

    After ``run_warmup()`` returns, the caller drives the controller via
    ``tick_once()``. Each call pumps MAVLink, executes queued ticks, and
    returns the captured baseline command (speed, heading, altitude).

    MAVLink ownership:
      - ``follower.mav`` read: DRLControllerRef (sole reader)
      - ``follower.mav`` write: DRL env (sends final residual-modified cmds)
      - ``leader_rx``: DRLControllerRef (reads geometry for FSM)
    """

    def __init__(
        self,
        follower_conn_str: str,
        speedup: float,
        leader_port: int = 14552,
        out_dir: Optional[str] = None,
        episode_index: int = 0,
    ) -> None:
        # BaselineFormationController.__init__ creates FormationControlLaws
        # with expected_speedup=1.0 (line 2105/2181). We fix that below.
        super().__init__(follower_conn_str=follower_conn_str)

        # Override properties that main() normally sets from CLI args
        # (matches WarmupController pattern, lines 49-56)
        self.expected_speedup = speedup
        self.control_laws.speedup_factor = max(1.0, speedup)
        self._leader_port = leader_port
        # #329: OFF keeps legacy always_off; ON inherits parent state_gated.
        if not _FD_DRLREF_POSTTICK_PARITY:
            self._cpp_handoff_enabled = False
            self._cpp_handoff_policy = "always_off"
        self._out_dir = out_dir
        # Rev16 §6 — episode_index threaded via constructor for flush_history
        # correlation; assigned AFTER super().__init__() so any parent default
        # is overridden by the caller-supplied value.
        self._episode_index = episode_index

        # DRL mode state
        self._drl_active: bool = False
        self._captured_hdg_cmd: float = 0.0
        self._captured_alt_cmd: float = 0.0
        self._captured_spd_cmd: float = 24.7
        self._command_captured_this_tick: bool = False
        self._warmup_reached_target: bool = False
        # T-396: M2 acceptance witness. `rejects` counts every tick the
        # handoff was HELD, so "verified" can never be inferred from silence.
        self._m2_verify_rejects: int = 0
        self._m2_latch_source: str = ""
        self._m2_latch_witness: Optional[dict] = None
        self._sim_time_reporter: Optional[Callable[[float], None]] = None

        # C3n: Reference-level position offsets (meters)
        self._offset_behind: float = 0.0
        self._offset_cross: float = 0.0
        self._offset_below: float = 0.0

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Override to rebind leader_rx to env-correct UDP port.

        Parent's connect() at baseline_formation_controller.py:2632
        unconditionally instantiates LeaderTelemetryReceiver() with the
        default port 14552. For env_id >= 1, MAVProxy publishes leader
        telemetry on env-offset ports, so the default-port receiver
        gets 0 packets and FW_STABILIZE handler silent-returns at
        leader_state=None forever. We let parent connect normally,
        then replace leader_rx with a port-correct instance.

        Diagnostic provenance:
          analysis/fw_stabilize_rootcause_20260430T051230Z/
        """
        if not super().connect():
            return False
        if self._leader_port != 14552:
            self.leader_rx = LeaderTelemetryReceiver(
                listen_port=self._leader_port,
            )
            if not self.leader_rx.connect():
                return False
        return True

    def transition_to(
        self,
        new_state: ControllerState,
        reason: str = "",
        skip_exit: bool = False,
    ) -> None:
        """Override to activate DRL mode when M2 is reached.

        Unlike WarmupController (which stops the loop at M2), we keep the
        controller alive and switch to command-capture mode. All subsequent
        FSM transitions (DOCKING_CREEP, HOLD_CONTACT, etc.) proceed normally.
        """
        super().transition_to(new_state, reason, skip_exit)
        if not self._drl_active and new_state == ControllerState.HOLD_PRECONTACT:
            self._maybe_latch_m2(source="transition")

    # ------------------------------------------------------------------
    # T-396 (GO A): geometry-verified M2 acceptance
    # ------------------------------------------------------------------

    def m2_station_check(self) -> Optional[dict]:
        """Is the PERCEIVED geometry inside the start-line box right now?

        Reads ``get_geometry()`` -- this subclass's own perception estimate,
        the same one the FSM flies on. No truth field is touched (G4 Tier-0).
        Tolerances come from ``formation_config``, the single source
        (T-386/T-389); this method never re-forks a tolerance value.

        Returns the witness dict, or None when geometry is unavailable (which
        is NOT a pass -- UNSAMPLED is not CLEAN).
        """
        import formation_config as _fc
        try:
            geom = self.get_geometry()
        except Exception:
            geom = None
        if geom is None:
            return None
        w = {
            "behind_err_m": float(geom.behind_m) - _fc.PRECONTACT_BEHIND_M,
            "cross_err_m": float(geom.cross_m) - _fc.PRECONTACT_CROSS_M,
            "below_err_m": float(geom.below_m) - _fc.PRECONTACT_BELOW_M,
        }
        w["in_station"] = bool(
            abs(w["behind_err_m"]) <= _fc.PRECONTACT_BEHIND_OUTER_TOL_M
            and abs(w["cross_err_m"]) <= _fc.PRECONTACT_CROSS_OUTER_TOL_M
            and abs(w["below_err_m"]) <= _fc.PRECONTACT_BELOW_OUTER_TOL_M)
        return w

    def _maybe_latch_m2(self, source: str = "") -> bool:
        """Accept the FSM's M2 declaration iff the station geometry agrees.

        ``source`` is "transition" (FSM just entered HOLD_PRECONTACT) or
        "tick" (the per-tick re-check). The re-check exists because the
        aircraft can converge WITHOUT any further transition -- it is already
        in HOLD_PRECONTACT, so a transition-only latch would never fire again
        and every leg would burn its full warmup budget and then time out.
        """
        if self._drl_active:
            return False
        if not _FD_M2_GEOM_VERIFY:
            self._m2_latch_source = source
            self._warmup_reached_target = True
            self._activate_drl_mode()
            return True
        w = self.m2_station_check()
        if w is None or not w["in_station"]:
            self._m2_verify_rejects += 1
            if self._m2_verify_rejects in (1, 50) or (
                    self._m2_verify_rejects % 500 == 0):
                log(f"[M2_VERIFY] holding handoff ({source}): {w} "
                    f"rejects={self._m2_verify_rejects}")
            return False
        self._m2_latch_source = source
        self._m2_latch_witness = w
        log(f"[M2_VERIFY] station verified via {source} after "
            f"{self._m2_verify_rejects} held tick(s): {w}")
        self._warmup_reached_target = True
        self._activate_drl_mode()
        return True

    def _execute_tick(self, queue_time_boot_ms=None) -> bool:
        """Override to report sim-time progress (same as WarmupController)."""
        result = super()._execute_tick(queue_time_boot_ms)
        # T-396 re-check: the aircraft converges while ALREADY in
        # HOLD_PRECONTACT, with no further transition to hook. Without this,
        # a held handoff could never be released.
        if (result and _FD_M2_GEOM_VERIFY and not self._drl_active
                and self.state == ControllerState.HOLD_PRECONTACT):
            self._maybe_latch_m2(source="tick")
        if result and self._sim_time_reporter is not None:
            try:
                self._sim_time_reporter(self._t_sim_s)
            except Exception:
                pass  # Never let reporter failure break tick
        return result

    def send_guided_commands(
        self,
        hdg_cmd: float,
        alt_cmd: float,
        spd_cmd: float,
        range_m: float = 100.0,
        fol_alt: Optional[float] = None,
    ) -> None:
        """Capture baseline commands; suppress MAVLink sends in DRL mode.

        During warmup (pre-M2), passes through to the real implementation.
        In DRL mode, captures hdg/alt/spd as the baseline command, then runs
        super() with monkey-patched follower methods (no-ops) so all internal
        state tracking (rate limiting, altitude watchdog, etc.) executes normally.
        """
        if not self._drl_active:
            super().send_guided_commands(hdg_cmd, alt_cmd, spd_cmd, range_m, fol_alt)
            return

        # Capture the controller's intended commands as the baseline
        self._captured_hdg_cmd = hdg_cmd
        self._captured_alt_cmd = alt_cmd
        self._captured_spd_cmd = spd_cmd
        self._command_captured_this_tick = True

        # Run base class logic for internal state tracking (rate limiting
        # timestamps, altitude watchdog, speed override consumption, etc.)
        # Follower sends are monkey-patched to no-ops in _activate_drl_mode().
        super().send_guided_commands(hdg_cmd, alt_cmd, spd_cmd, range_m, fol_alt)

    def _send_speed_command(
        self, spd_cmd: float, source: str, bypass_latch: bool = False,
    ) -> None:
        """Capture speed; suppress MAVLink send and DoubleSendError in DRL mode.

        During warmup, passes through to the real implementation.
        In DRL mode, replicates tracking state without the actual send or
        the DoubleSendError (which is harmless when commands are captured).
        """
        if not self._drl_active:
            super()._send_speed_command(spd_cmd, source, bypass_latch)
            return

        # Suppress DoubleSendError — in capture mode, double sends are harmless
        if self._speed_cmd_sent_this_tick and not bypass_latch:
            return

        # Replicate tracking logic (mirrors base class lines 4794-4805)
        if not self._speed_cmd_sent_this_tick:
            self._speed_cmd_sent_this_tick = True
            self._first_send_source = source
            self._first_send_speed_mps = spd_cmd
            self._first_send_bypass = bypass_latch

        self._speed_cmd_sent_last = spd_cmd
        self._speed_cmd_last_send_time = self._t_sim_s
        self._last_do_change_speed_time = self._t_sim_s
        self._last_do_change_speed_cmd = spd_cmd

    # ------------------------------------------------------------------
    # DRL mode activation
    # ------------------------------------------------------------------

    def _activate_drl_mode(self) -> None:
        """Transition from warmup to DRL-driven command capture.

        Called when FSM reaches HOLD_PRECONTACT. Breaks the warmup run()
        loop and monkey-patches follower MAVLink sends to no-ops so that
        super().send_guided_commands() runs all internal logic without
        transmitting.
        """
        self._drl_active = True
        self.running = False  # Break warmup's run() loop
        self._drl_mode = True  # Skip timing watchdogs

        # Monkey-patch follower MAVLink sends to no-ops.
        # super().send_guided_commands() calls these internally; they must
        # execute without error but not transmit.
        self.follower.send_guided_heading = lambda hdg, alt: None
        self.follower.send_guided_change_altitude = lambda alt, rate=0.0: True
        self.follower.send_guided_speed = lambda spd: None

    # ------------------------------------------------------------------
    # DRL env interface
    # ------------------------------------------------------------------

    def tick_once(self) -> Optional[BaselineCommand]:
        """Drive one (or more) FSM ticks from the DRL env.

        Pumps MAVLink, drains all queued ATTITUDE timestamps (may be >1
        if SITL ran ahead between env sub-ticks), and returns the last
        captured baseline command.

        Returns None if no tick was executed (no new ATTITUDE messages).
        """
        self._command_captured_this_tick = False

        # C3a: bounded-snapshot drain with in-loop re-pump.
        # Snapshot queue depth before loop so new entries enqueued by the
        # in-loop pump are processed on the NEXT tick_once() call (preserves
        # single-tick atomicity). The in-loop pump refreshes follower position
        # cache, attitude queue, and leader telemetry between drain iterations,
        # forcing _tick_counter advancement to invalidate get_geometry()'s
        # cache (baseline_formation_controller.py:3238-3240) on every iter.
        self._pump_mavlink_messages(blocking=False)
        drain_n = len(self.follower._attitude_time_queue)
        for _ in range(drain_n):
            if not self.follower._attitude_time_queue:
                break
            t = self.follower._attitude_time_queue.popleft()
            ok = self._execute_tick(queue_time_boot_ms=t)
            # #329 parity: replicate run()'s post-tick block (ctrl:9366-9381).
            if _FD_DRLREF_POSTTICK_PARITY and ok:
                try:
                    geom = self.get_geometry()
                    self._check_cpp_handoff(geom)
                    if geom is not None:
                        lead_state = (self.leader_rx.get_state()
                                      if self.leader_rx else None)
                        self._check_wake_zone(geom, lead_state)
                    self._update_success_metrics(geom)
                except Exception as e:
                    log(f"[POSTTICK_PARITY] exception: {e!r}")
                    raise
            self._pump_mavlink_messages(blocking=False)
            self._tick_counter += 1  # Invalidate get_geometry() cache

        # === Phase 5c.A.6 ENTRYPOINT_DIAG (eval-path tick_end) ===
        try:
            from . import entrypoint_diag as _ed
            if _ed.is_enabled():
                _ed.emit_event(
                    "tick_end_drl",
                    path="eval",
                    tick_source="tick_once",
                    drain_n=drain_n,
                    fsm_state=getattr(self.state, "name", str(getattr(self, "state", ""))),
                    sim_t=getattr(self, "_t_sim_s", ""),
                    speed_cmd=getattr(self, "_captured_spd_cmd", ""),
                    alt_cmd=getattr(self, "_captured_alt_cmd", ""),
                    command_type="captured" if self._command_captured_this_tick else "none",
                )
        except Exception:
            pass

        if not self._command_captured_this_tick:
            return None  # No tick produced a command

        return BaselineCommand(
            speed_ms=self._captured_spd_cmd,
            heading_deg=self._captured_hdg_cmd,
            altitude_m=self._captured_alt_cmd,
        )

    def run_warmup(
        self,
        max_sim_seconds: float = 600.0,
        sim_time_reporter: Optional[Callable[[float], None]] = None,
        skip_takeoff: bool = False,
    ) -> None:
        """Run baseline to HOLD_PRECONTACT (M2), then return in DRL mode.

        Identical to WarmupController.run_warmup() except that the controller
        stays alive in command-capture mode after M2 is reached.

        Args:
            skip_takeoff: threaded to the frozen R7.1 ``run()`` param (#309 P1
                Option-1 addendum, operator-authorized 2026-07-04). Default
                False == pre-addendum byte-identical (ground VTOL takeoff).
                The json-lockstep backend passes True: its FDM spawns AIRBORNE
                at cruise (the census-validated substrate), where a ground
                arm/takeoff cannot complete (measured TAKEOFF_ARMING timeout
                3/3 x2). True == start in INTERCEPT state (R7.1 run():9089).

        Raises:
            TimeoutError: If M2 not reached within *max_sim_seconds*.
        """
        self._sim_time_reporter = sim_time_reporter
        # PRH-ATT-004 Fix 3A: Skip Hz watchdog during warmup
        self._drl_mode = True
        self.run(skip_takeoff=skip_takeoff, max_sim_seconds=max_sim_seconds)
        if not self._warmup_reached_target:
            raise TimeoutError(
                f"Warmup failed to reach HOLD_PRECONTACT "
                f"(final state: {self.state})"
            )
        # Switch time authority to DRL mode (never abort on timing issues)
        if self._time_auth is not None:
            self._time_auth.drl_mode = True
        self._revive_arbitration_csv()

    def _revive_arbitration_csv(self) -> None:
        """Re-open speed_arbitration.csv for the DRL-flown phase (ARBREVIVE).

        WHY THIS EXISTS [MEASURED 2026-08-16]: the frozen controller opens the
        CSV inside ``run()`` and closes it in ``run()``'s ``finally``. Warmup
        ends by breaking that loop (``_activate_drl_mode`` sets
        ``running = False``), so the writer is nulled the moment warmup
        returns and every later ``_write_arbitration_csv_row`` hits the
        fail-silent guard and returns. The consequence is that the ENTIRE
        DRL-flown acquisition and docking phase is uninstrumented: banked legs
        end on exactly ONE ``HOLD_PRECONTACT`` row, and the ALT_CAP_HOLD_RANGE
        and DOCKING_CREEP arbitration call sites are structurally unreachable
        in this venue. That is why W15 leg 47318's precontact stall could not
        be diagnosed from its telemetry.

        This mirrors the UWB-diagnostic revival the env already performs for
        the same reason (env_formation_sitl.py, "run()'s finally just closed
        it"). LOGGING ONLY -- no control path reads this file.

        R7.1 IS NOT TOUCHED: the frozen controller's own methods are CALLED,
        never edited. The warmup rows are BANKED aside first because
        ``_init_arbitration_csv`` opens with mode 'w' and would otherwise
        truncate them.
        """
        out_dir = getattr(self, "_out_dir", None)
        if not out_dir:
            return                      # mirrors the frozen guard at ctrl:7161
        try:
            csv_path = os.path.join(out_dir, "speed_arbitration.csv")
            if os.path.exists(csv_path):
                banked = os.path.join(out_dir, "speed_arbitration_warmup.csv")
                if not os.path.exists(banked):      # never overwrite a bank
                    os.replace(csv_path, banked)
            self._init_arbitration_csv()
        except Exception as exc:        # noqa: BLE001 - instrumentation only
            log("[ARB_CSV] revival for the DRL phase failed "
                "(post-handoff ticks stay unlogged): %s" % exc)

    def get_fsm_state(self) -> ControllerState:
        """Expose current FSM state to DRL env."""
        return self.state

    # ------------------------------------------------------------------
    # C3n: Reference-level position offsets (RFL)
    # ------------------------------------------------------------------

    def set_position_offset(
        self, behind_m: float, cross_m: float, below_m: float,
    ) -> None:
        """Set position offsets for reference-level residual (C3n).

        The next call to get_geometry() will apply these offsets, making
        the controller perceive modified relative positions.

        Sign convention (verified from formation_control_laws.py:523/598/702):
          offset > 0 => modified_behind = behind_m - offset => smaller behind
          => negative behind_error => negative p_term => lower speed
          => follower drifts further behind => CONSERVATIVE.
        """
        self._offset_behind = behind_m
        self._offset_cross = cross_m
        self._offset_below = below_m

    def clear_position_offset(self) -> None:
        """Reset all position offsets to zero (call at episode start / pre-gate)."""
        self._offset_behind = 0.0
        self._offset_cross = 0.0
        self._offset_below = 0.0

    def get_geometry(self, now_s=None):
        """Override to inject C3n position offsets into perceived geometry.

        Calls the base class to obtain true geometry, then returns a NEW
        RelativeGeometry with position offsets applied. Velocity components
        are unchanged — the offset is a static target shift, not a velocity
        perturbation. When all offsets are zero, returns the base result
        unchanged (invariant I1).

        Explicit full construction (NOT dataclasses.replace) ensures:
        (a) every field accounted for, (b) if RelativeGeometry gains a field,
        construction fails at runtime — safe failure, not silent inconsistency.

        All 14 FSM call sites use self.get_geometry() (verified lines 3090-8810
        in baseline_formation_controller.py). Zero bypass paths.
        """
        from geometry import RelativeGeometry

        geom = super().get_geometry(now_s)
        if geom is None:
            return None
        if (self._offset_behind == 0.0
                and self._offset_cross == 0.0
                and self._offset_below == 0.0):
            return geom  # Fast path: exact passthrough (I1)

        mod_b = geom.behind_m - self._offset_behind
        mod_c = geom.cross_m - self._offset_cross
        mod_l = geom.below_m - self._offset_below

        # C3r: preserve original target_aspect_deg to isolate heading pipeline.
        # Heading computation uses atan2(-cross_m, behind_m) directly from the
        # geometry struct. Recomputing target_aspect_deg from modified coords
        # caused the C3n heading slew artifact (cross offset → heading P-term
        # perturbation → slew rate limiter throttle → degraded dwell).
        return RelativeGeometry(
            behind_m=mod_b,
            cross_m=mod_c,
            below_m=mod_l,
            range_m=math.sqrt(mod_b ** 2 + mod_c ** 2 + mod_l ** 2),
            target_aspect_deg=geom.target_aspect_deg,  # C3r: heading isolation
            closure_rate_ms=geom.closure_rate_ms,
            vel_cross_m_s=geom.vel_cross_m_s,
            vel_below_m_s=geom.vel_below_m_s,
            source=geom.source,
        )
