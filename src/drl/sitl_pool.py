"""Pre-warmed SITL pool — eliminates per-episode warmup overhead.

Architecture
------------
The pool maintains N "slots", each running an independent SITL warmup in a
background thread. When ``acquire()`` is called, it returns a ready-to-use
``WarmupResult`` containing all state needed to start a DRL episode.

Each slot cycles through:
  1. Launch SITL + leader broadcaster
  2. Run WarmupController to M1/M2
  3. Package result into WarmupResult
  4. Wait until acquired by an env.reset()
  5. Repeat from step 1

The pool is designed for use with SB3 SubprocVecEnv, where each worker
process creates its own PooledSITLEnv that talks to the shared pool.

Threading Model
---------------
The pool manager runs in a single process (the main training process or
a dedicated daemon). Each slot runs warmup in a separate thread. This
works because the warmup is I/O-bound (waiting on MAVLink messages from
the SITL C++ process), not CPU-bound.

Port Isolation
--------------
Each slot gets a unique port offset (slot_id * PORT_STRIDE) to prevent
interference between concurrent SITL instances.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, List, Optional, Tuple

import logging

from . import config
from ._process_utils import pdeathsig_preexec

logger = logging.getLogger(__name__)

# Baseline imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TESTS_DIR = os.path.join(_REPO_ROOT, "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_SITL_SCRIPT = os.path.join(_TESTS_DIR, "start_formation_sitl_fixed.sh")
_LEADER_SCRIPT = os.path.join(_TESTS_DIR, "leader_broadcast_fixed.py")


@dataclass
class WarmupResult:
    """Everything the DRL env needs from a completed warmup.

    The env takes ownership of all handles; the pool slot releases them.
    """
    # MAVLink connection to follower SITL (pymavlink connection object)
    mav: Any = None
    # SimTimeAuthority instance
    time_auth: Any = None
    # GeometryCalculator with correct home position
    geom_calc: Any = None
    # FormationControlLaws with correct speedup_factor
    control_laws: Any = None
    # LeaderTelemetryReceiver (kept alive for DRL phase)
    leader_rx: Any = None
    # Home altitude MSL (for AGL safety checks)
    home_alt_msl: Optional[float] = None
    # Last known follower/leader telemetry for initial obs construction
    follower_lat: float = 0.0
    follower_lon: float = 0.0
    follower_vn: float = 0.0
    follower_ve: float = 0.0
    follower_airspeed_ms: float = 24.7
    follower_alt_msl_m: float = 890.0
    follower_heading_deg: float = 0.0
    follower_pitch_deg: float = 0.0
    follower_roll_deg: float = 0.0
    follower_vd_ms: float = 0.0
    leader_lat: float = 0.0
    leader_lon: float = 0.0
    leader_vn: float = 0.0
    leader_ve: float = 0.0
    leader_airspeed_ms: float = 24.7
    leader_heading_deg: float = 0.0
    leader_alt_msl_m: float = 890.0
    leader_vd_ms: float = 0.0
    # Subprocess handles (env takes ownership for cleanup)
    sitl_proc: Optional[subprocess.Popen] = None
    leader_proc: Optional[subprocess.Popen] = None
    leader_log_file: Any = None
    # Port offset (for orphan cleanup)
    port_offset: int = 0
    # Metadata
    warmup_target: str = "M1"
    slot_id: int = 0
    warmup_wall_seconds: float = 0.0


class _PoolSlot:
    """A single warmup slot — runs in its own thread.

    Lifecycle:
      1. _warmup_loop() runs continuously in a daemon thread
      2. Each iteration: launch SITL → warmup → put result in _ready queue
      3. Main thread calls acquire() → gets result from _ready queue
      4. Slot starts next warmup iteration immediately
    """

    def __init__(
        self,
        slot_id: int,
        speedup: float,
        warmup_target: str = "M1",
        wind_speed_ms: float = 0.0,
        wind_dir_deg: float = 0.0,
        wind_turb: float = 0.0,
        ui_mode: str = "none",
    ) -> None:
        self.slot_id = slot_id
        self.speedup = speedup
        self.warmup_target = warmup_target
        self.wind_speed_ms = wind_speed_ms
        self.wind_dir_deg = wind_dir_deg
        self.wind_turb = wind_turb
        self.ui_mode = ui_mode

        self._port_offset = slot_id * config.PORT_STRIDE
        self._follower_port = 14551 + self._port_offset
        self._leader_port = 14550 + self._port_offset
        self._timesync_port = 14553 + self._port_offset

        self._ready: Queue[WarmupResult] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warmup_count = 0
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self._permanently_failed = False

    def start(self) -> None:
        """Start the background warmup thread."""
        self._thread = threading.Thread(
            target=self._warmup_loop,
            name=f"pool-slot-{self.slot_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def acquire(self, timeout: float = 300.0) -> WarmupResult:
        """Get a pre-warmed SITL instance (blocks until ready).

        Parameters
        ----------
        timeout: float
            Max seconds to wait for a ready instance.

        Returns
        -------
        WarmupResult
            Ready-to-use SITL instance with all state.

        Raises
        ------
        TimeoutError
            If no instance ready within timeout.
        """
        try:
            return self._ready.get(timeout=timeout)
        except Empty:
            raise TimeoutError(
                f"Pool slot {self.slot_id}: no warmup ready within {timeout}s"
            )

    @property
    def is_ready(self) -> bool:
        """True if a warmed instance is available without blocking."""
        return not self._ready.empty()

    def update_domain_params(
        self,
        warmup_target: str = "M1",
        wind_speed_ms: float = 0.0,
        wind_dir_deg: float = 0.0,
        wind_turb: float = 0.0,
    ) -> None:
        """Update domain randomisation params for next warmup cycle."""
        self.warmup_target = warmup_target
        self.wind_speed_ms = wind_speed_ms
        self.wind_dir_deg = wind_dir_deg
        self.wind_turb = wind_turb

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _warmup_loop(self) -> None:
        """Continuous warmup loop — runs in daemon thread."""
        while not self._stop.is_set() and not self._permanently_failed:
            try:
                result = self._do_one_warmup()
                if result is not None:
                    self._consecutive_failures = 0  # Reset on success
                    # Block until acquired (or stop signal)
                    while not self._stop.is_set():
                        try:
                            self._ready.put(result, timeout=1.0)
                            break
                        except Exception:
                            logger.debug(
                                "Failed to put warmup result in queue, retrying", exc_info=True)
                            continue
                else:
                    self._consecutive_failures += 1
            except Exception as e:
                self._consecutive_failures += 1
                tag = f"[pool-slot-{self.slot_id}]"
                logger.error(f"{tag} Warmup failed ({self._consecutive_failures}/"
                             f"{self._max_consecutive_failures}): {e}")
                # Brief pause before retry to avoid tight error loops
                self._stop.wait(5.0)

            if self._consecutive_failures >= self._max_consecutive_failures:
                tag = f"[pool-slot-{self.slot_id}]"
                logger.error(f"{tag} PERMANENTLY FAILED after "
                             f"{self._max_consecutive_failures} consecutive failures. "
                             f"Slot disabled.")
                self._permanently_failed = True

    def _do_one_warmup(self) -> Optional[WarmupResult]:
        """Execute one full warmup cycle."""
        tag = f"[pool-slot-{self.slot_id}]"
        t0 = time.monotonic()
        self._warmup_count += 1

        # 1. Kill any orphan processes from previous iteration
        self._kill_orphans()

        # 2. Launch SITL
        sitl_proc = self._launch_sitl()
        if sitl_proc is None:
            return None

        # 3. Launch leader broadcaster
        leader_proc, leader_log_file = self._launch_leader()
        if leader_proc is None:
            self._kill_proc(sitl_proc)
            return None

        # 4. Run warmup controller
        try:
            from .warmup_controller import WarmupController
            import math as _math

            ctrl = WarmupController(
                follower_conn_str=f"udp:127.0.0.1:{self._follower_port}",
                speedup=self.speedup,
                leader_port=14552 + self._port_offset,
                out_dir=None,
                stop_state=self.warmup_target,
            )
            ctrl.run_warmup(max_sim_seconds=600.0)

            # 5. Package result
            result = WarmupResult(
                mav=ctrl.follower.mav,
                time_auth=ctrl._time_auth,
                geom_calc=ctrl.geom_calc,
                control_laws=ctrl.control_laws,
                leader_rx=ctrl.leader_rx,
                home_alt_msl=getattr(ctrl, "home_alt_msl", None),
                sitl_proc=sitl_proc,
                leader_proc=leader_proc,
                leader_log_file=leader_log_file,
                port_offset=self._port_offset,
                warmup_target=self.warmup_target,
                slot_id=self.slot_id,
                warmup_wall_seconds=time.monotonic() - t0,
            )

            # Extract follower state
            follower_airspeed = 24.7
            if ctrl.follower._last_vfr_hud:
                follower_airspeed = ctrl.follower._last_vfr_hud.airspeed

            if ctrl.follower._last_position:
                pos = ctrl.follower._last_position
                result.follower_lat = pos.lat / 1e7
                result.follower_lon = pos.lon / 1e7
                result.follower_vn = pos.vx / 100.0
                result.follower_ve = pos.vy / 100.0
                result.follower_airspeed_ms = follower_airspeed
                result.follower_alt_msl_m = pos.alt / 1000.0
                result.follower_heading_deg = pos.hdg / 100.0
                result.follower_vd_ms = pos.vz / 100.0

            if ctrl.follower._last_attitude:
                att = ctrl.follower._last_attitude
                result.follower_pitch_deg = _math.degrees(att.pitch)
                result.follower_roll_deg = _math.degrees(att.roll)

            # Extract leader state
            if ctrl.leader_rx and ctrl.leader_rx.last_state:
                ls = ctrl.leader_rx.last_state
                result.leader_lat = ls.lat
                result.leader_lon = ls.lon
                result.leader_vn = ls.vn_ms
                result.leader_ve = ls.ve_ms
                result.leader_airspeed_ms = (
                    ls.true_airspeed_ms if ls.true_airspeed_ms > 0 else 24.7
                )
                result.leader_heading_deg = ls.heading_deg
                result.leader_alt_msl_m = ls.alt_msl
                result.leader_vd_ms = ls.vd_ms

            elapsed = time.monotonic() - t0
            logger.info(f"{tag} Warmup #{self._warmup_count} ready: "
                        f"{self.warmup_target} reached in {elapsed:.1f}s wall")
            return result

        except Exception as e:
            logger.error(f"{tag} Warmup controller error: {e}")
            self._kill_proc(sitl_proc)
            self._kill_proc(leader_proc)
            if leader_log_file is not None:
                try:
                    leader_log_file.close()
                except Exception:
                    logger.debug("Failed to close leader log file", exc_info=True)
            return None

    def _launch_sitl(self) -> Optional[subprocess.Popen]:
        """Launch ArduPilot SITL."""
        env = os.environ.copy()
        env["HARNESS_PORT_OFFSET"] = str(self._port_offset)
        cmd = [
            "bash", _SITL_SCRIPT,
            "--direct",
            "--ui", self.ui_mode,
            "--speedup", str(self.speedup),
        ]
        try:
            return subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=pdeathsig_preexec,
            )
        except Exception as e:
            logger.error(f"[pool-slot-{self.slot_id}] SITL launch error: {e}")
            return None

    def _launch_leader(self) -> Tuple[Optional[subprocess.Popen], Any]:
        """Launch leader broadcaster."""
        leader_out = f"127.0.0.1:{self._leader_port}"
        timesync = f"udpin:127.0.0.1:{self._timesync_port}"
        sitl_tcp = f"127.0.0.1:{5763 + self._port_offset}"
        cmd = [
            sys.executable, _LEADER_SCRIPT,
            "--udp-out", leader_out,
            "--sitl-tcp", sitl_tcp,
            "--time-sync-conn", timesync,
            "--sim-speedup", str(self.speedup),
            "--wind-speed", str(self.wind_speed_ms),
            "--wind-dir", str(self.wind_dir_deg),
        ]
        # T-451: legacy SIM_WIND_TURB knob emitted only when set -- see the
        # note at env_formation_sitl.py:_launch_leader. Value-guarded, so the
        # native/legacy tables (wind_turb 0.1-0.5) are byte-identical.
        if float(self.wind_turb) > 0.0:
            cmd += ["--turbulence", str(self.wind_turb)]
        log_dir = os.path.join(_REPO_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        leader_log = os.path.join(log_dir, f"leader_pool_slot{self.slot_id}.log")
        try:
            log_file = open(leader_log, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=pdeathsig_preexec,
            )
            return proc, log_file
        except Exception as e:
            logger.error(f"[pool-slot-{self.slot_id}] Leader launch error: {e}")
            return None, None

    @staticmethod
    def _kill_proc(proc: Optional[subprocess.Popen]) -> None:
        """Kill a subprocess by process group."""
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def _kill_orphans(self) -> None:
        """Kill orphan processes from this slot's port range.

        Uses UID-scoped pkill (-u $USER) for multi-user safety (v15 audit).
        """
        base_port = 5760 + self._port_offset
        user = os.environ.get("USER", "")
        user_args = ["-u", user] if user else []
        patterns = [
            f"serial0=tcp:{base_port}",
            f"tcp:127.0.0.1:{base_port}",
            f"leader_broadcast_fixed.py.*{self._leader_port}",
        ]
        for pat in patterns:
            try:
                subprocess.run(
                    ["pkill", "-9"] + user_args + ["-f", pat],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except Exception:
                logger.debug("Failed to pkill orphan pattern %s", pat, exc_info=True)


class SITLPool:
    """Manages a pool of pre-warmed SITL instances.

    Parameters
    ----------
    n_envs: int
        Number of active DRL environments (determines number of pool slots).
    pool_size: int
        Number of pool slots. Should be >= n_envs. Extra slots provide
        buffer for overlapping warmup with stepping. Default: n_envs + 2.
    speedup: float
        SITL speedup factor.
    warmup_target: str
        Default warmup target ("M1" or "M2").
    """

    def __init__(
        self,
        n_envs: int,
        pool_size: Optional[int] = None,
        speedup: float = 2.0,
        warmup_target: str = "M1",
        ui_mode: str = "none",
    ) -> None:
        if pool_size is None:
            pool_size = n_envs + 2  # 2 buffer slots
        self._n_envs = n_envs
        self._pool_size = pool_size
        self._speedup = speedup
        self._warmup_target = warmup_target
        self._ui_mode = ui_mode

        # Create slots with non-overlapping port offsets
        # Active env slots: 0, 1, ..., n_envs-1
        # Buffer slots: n_envs, n_envs+1, ...
        self._slots: List[_PoolSlot] = []
        for i in range(pool_size):
            self._slots.append(
                _PoolSlot(
                    slot_id=i,
                    speedup=speedup,
                    warmup_target=warmup_target,
                    ui_mode=ui_mode,
                )
            )

        self._started = False

    def start(self) -> None:
        """Start all pool slots (begins background warmup)."""
        if self._started:
            return
        logger.info(f"[SITLPool] Starting {self._pool_size} warmup slots "
                    f"(speedup={self._speedup}x, target={self._warmup_target})")
        for slot in self._slots:
            slot.start()
            # Stagger launches slightly to avoid port conflicts
            time.sleep(2.0)
        self._started = True

    def stop(self) -> None:
        """Stop all pool slots and clean up."""
        for slot in self._slots:
            slot.stop()
        self._started = False

    def acquire(self, env_id: int, timeout: float = 300.0) -> WarmupResult:
        """Acquire a pre-warmed SITL instance for the given env_id.

        Tries the env's dedicated slot first, then falls back to any
        available slot.

        Parameters
        ----------
        env_id: int
            Environment ID (0-based).
        timeout: float
            Maximum wait in seconds.

        Returns
        -------
        WarmupResult
            Ready-to-use instance.
        """
        # Try dedicated slot first
        if env_id < len(self._slots) and self._slots[env_id].is_ready:
            return self._slots[env_id].acquire(timeout=timeout)

        # Try any ready slot
        for slot in self._slots:
            if slot.is_ready:
                return slot.acquire(timeout=1.0)

        # No ready slots — wait on dedicated slot
        slot = self._slots[env_id % len(self._slots)]
        return slot.acquire(timeout=timeout)

    def update_domain_params(
        self,
        warmup_target: str = "M1",
        wind_speed_ms: float = 0.0,
        wind_dir_deg: float = 0.0,
        wind_turb: float = 0.0,
    ) -> None:
        """Update domain randomisation for all pool slots.

        Called when curriculum stage changes to update warmup target
        (M1 → M2) and wind conditions.
        """
        for slot in self._slots:
            slot.update_domain_params(
                warmup_target=warmup_target,
                wind_speed_ms=wind_speed_ms,
                wind_dir_deg=wind_dir_deg,
                wind_turb=wind_turb,
            )

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def ready_count(self) -> int:
        """Number of slots with ready instances."""
        return sum(1 for s in self._slots if s.is_ready)
