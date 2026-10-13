"""Option-C DRL environment — wraps Method C harness as subprocess.

This is a structural SCAFFOLD authored under operator dispatch
2026-05-02 GO_OPTION_C_IMPLEMENTATION. Implementation details are TODO;
the file is structured to pass py_compile and import cleanly so subsequent
dispatches can fill in implementations incrementally without restructuring.

Architectural rationale (see analysis/option_c_feasibility_audit_20260502_180437/
reports/option_c_feasibility_audit.md for full audit):

- Method C harness (tests/run_gated_matrix_v910.sh / tests/lib_run_harness.sh /
  tests/start_formation_sitl_fixed.sh) runs UNCHANGED as a per-episode subprocess.
- baseline_formation_controller.py runs UNCHANGED as a subprocess (R7.1 preserved,
  zero waiver needed).
- observation_builder.py runs UNCHANGED in-env (R7.2 preserved). All 42-D
  observation inputs derivable from MAVLink telemetry + computed geometry + UWB.
- reward_mrrl.py runs UNCHANGED in-env.
- Residual injection point: PORT_CONTROL_IN = 14560 + (HARNESS_PORT_OFFSET).
  Already a MAVProxy INPUT link in tests/start_formation_sitl_fixed.sh:468,475.
  DRL env writes residual-modified DO_REPOSITION / DO_CHANGE_SPEED commands to
  this UDP port; MAVProxy routes to ArduPlane SITL.
- For zero-residual mode, DRL env sends nothing — controller's commands flow
  through unchanged → identical to Method C anchor (96/96 M3b STRICT proven).

Required working-tree patches at launch (verify with git diff --stat):
- tests/leader_broadcast_fixed.py F3 (+23/-2): leader port-offset routing
- tests/baseline_formation_controller.py F6 Option A (+3/-1): controller leader port wiring

NO modifications to:
- tests/baseline_formation_controller.py beyond F6 (R7.1)
- src/drl/observation_builder.py (R7.2)
- scripts/eval_baseline_zero_residual.py (R7.3)

References:
- Phase 1 RCA: analysis/phase1_rca_audit_20260502_142011/
- Phase 2 RCA: analysis/phase2_mechanism_audit_20260502_145524/
- Phase 3 RCA: analysis/phase3_mechanism_audit_20260502_151600/
- Option C feasibility: analysis/option_c_feasibility_20260502_180437/
"""

from __future__ import annotations

import json
import os
import queue
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from pymavlink import mavutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    mavutil = None  # smoke test will surface this if MAVLink path exercised

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore[no-redef]
    from gym import spaces  # type: ignore[no-redef]

from . import config
from .observation_builder import (
    EnvelopeFlags,
    FollowerState,
    LeaderState,
    ObservationBuilder,
    SensorInfo,
)
from .reward_mrrl import MRRLRewardComputer

# Geometry library lives outside src/drl; resolved via project-root sys.path
# (set by smoke driver and by training entry point).
import math  # noqa: E402

import sys as _sys  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "tests") not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT / "tests"))

from geometry import (  # type: ignore[import-not-found] # noqa: E402
    BOOM_TIP_BELOW_OFFSET_M,
    GeometryCalculator,
    RelativeGeometry,
)

try:  # Phase 4 I1: constants for OPTION_C_ENV_FUSION_TICK_LOG init record.
    from relative_pose.uwb_gps_fusion import (  # noqa: E402
        UWB_RANGE_SIGMA_M, GPS_POS_SIGMA_M, GPS_VEL_SIGMA_M_S,
        INNOVATION_GATE_SIGMA, P_FLOOR_POS_M2, P_FLOOR_VEL_M2S2,
    )
except Exception:
    UWB_RANGE_SIGMA_M = GPS_POS_SIGMA_M = GPS_VEL_SIGMA_M_S = None  # type: ignore
    INNOVATION_GATE_SIGMA = P_FLOOR_POS_M2 = P_FLOOR_VEL_M2S2 = None  # type: ignore


class _FusionTraceProxy:
    """Phase 4 I1 instrumentation: pre/post JSONL trace around RelativePoseFusion
    update_gps/update_uwb. Installed only when OPTION_C_ENV_FUSION_TICK_LOG=1."""
    def __init__(self, inner, env): self._inner, self._env = inner, env
    def __getattr__(self, n): return getattr(self._inner, n)
    def _emit(self, site, **kw):
        env = self._env
        if os.environ.get("OPTION_C_ENV_FUSION_TICK_LOG") != "1": return
        h = getattr(env, "_eft_handle", None)
        if h is None:
            d = Path(os.environ.get("OPTION_C_ENV_FUSION_TICK_LOG_DIR", "/tmp"))
            d.mkdir(parents=True, exist_ok=True)
            h = open(d / f"env_fusion_update_trace_env{env.env_id}.jsonl", "w", buffering=65536)
            env._eft_handle = h
        f = self._inner; raw = getattr(env, "_last_raw_geom", None); g = getattr(env, "_last_geom", None)
        rec = {"site": site, "env_id": env.env_id, "tick_idx": getattr(env, "_tick_count", None),
            "fsm_state": str(getattr(env, "_fsm_state", None)),
            "raw_truth_behind_m": raw.behind_m if raw else None,
            "raw_truth_cross_m": raw.cross_m if raw else None,
            "raw_truth_below_m": raw.below_m if raw else None,
            "env_fused_behind_m": g.behind_m if g else None,
            "env_fused_cross_m": g.cross_m if g else None,
            "env_fused_below_m": g.below_m if g else None,
            "last_predict_time": f.last_predict_time, "last_gps_time": f.last_gps_time,
            "last_uwb_time": f.last_uwb_time, "gps_update_count": f.gps_update_count,
            "uwb_update_count": f.uwb_update_count, "uwb_rejected_count": f.uwb_rejected_count}
        rec.update(kw)
        try: h.write(json.dumps(rec, default=str) + "\n")
        except Exception: pass
    def update_gps(self, m):
        xb = list(self._inner.x); Pb = [self._inner.P[i][i] for i in range(6)]
        rt = (self._env._last_raw_geom.behind_m if getattr(self._env, "_last_raw_geom", None) else None)
        dtg = (m.timestamp - self._inner.last_gps_time) if self._inner.last_gps_time is not None else None
        self._emit("gps_pre", sim_t=m.timestamp, x_before=xb, P_diag_before=Pb,
            gps_z_behind_m=m.behind_m, gps_z_cross_m=m.cross_m, gps_z_below_m=m.below_m,
            gps_z_vbehind_m_s=m.vel_behind_m_s, gps_z_vcross_m_s=m.vel_cross_m_s, gps_z_vbelow_m_s=m.vel_below_m_s,
            z0_minus_truth=(m.behind_m - rt) if rt is not None else None,
            dt_gps_actual=dtg, gps_innovation_behind_expected=(m.behind_m - xb[0]))
        ok = self._inner.update_gps(m)
        xa = list(self._inner.x); Pa = [self._inner.P[i][i] for i in range(6)]
        self._emit("gps_post", sim_t=m.timestamp, x_after=xa, P_diag_after=Pa, accepted=bool(ok))
        return ok
    def update_uwb(self, m):
        xb = list(self._inner.x); Pb = [self._inner.P[i][i] for i in range(6)]
        pr = (xb[0]**2 + xb[1]**2 + xb[2]**2) ** 0.5
        rb = self._inner.uwb_rejected_count
        dtu = (m.timestamp - self._inner.last_uwb_time) if self._inner.last_uwb_time is not None else None
        self._emit("uwb_pre", sim_t=m.timestamp, x_before=xb, P_diag_before=Pb,
            uwb_range_submitted_m=m.range_m, uwb_predicted_range_pre_m=pr,
            uwb_innovation_expected=(m.range_m - pr), uwb_H_x_weight=(xb[0] / max(pr, 0.1)),
            dt_uwb_actual=dtu)
        ok = self._inner.update_uwb(m)
        xa = list(self._inner.x); Pa = [self._inner.P[i][i] for i in range(6)]
        ra = self._inner.uwb_rejected_count
        d0 = xa[0] - xb[0]
        self._emit("uwb_post", sim_t=m.timestamp, x_after=xa, P_diag_after=Pa,
            x0_delta_actual=d0, x0_delta_sign=("positive" if d0 > 0 else "negative"),
            uwb_accepted=(ra == rb))
        return ok


import logging  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# Constants — Option C port + timing layout
# ============================================================================

# Port assignments (must match tests/start_formation_sitl_fixed.sh:438-448).
# All ports are offset by HARNESS_PORT_OFFSET = ENV_ID * PORT_STRIDE.
BASE_FOLLOWER_PORT = 14551       # PORT_FOLLOWER (MAVProxy → controller process)
BASE_CONTROL_IN_PORT = 14560     # PORT_CONTROL_IN (MAVProxy INPUT — residual injection)
BASE_LEADER_PORT = 14550         # PORT_LEADER_IN (MAVProxy INPUT — leader telemetry)
BASE_MONITOR_PORT = 14552        # PORT_MONITOR (MAVProxy → LeadTelemetryFeed in controller)
BASE_TIMESYNC_PORT = 14553       # PORT_TIMESYNC (MAVProxy → leader time-sync)
BASE_UI_PORT = 14554             # PORT_UI (MAVProxy → optional UI/Map; UNBOUND by default)
BASE_UWB_TRUTH_PORT = 14555      # PORT_UWB_TRUTH (MAVProxy → UWB sim truth feed)
PORT_STRIDE = 100                # HARNESS_PORT_OFFSET = ENV_ID * PORT_STRIDE

# Option C reads telemetry from the UI port (14554) because PORT_FOLLOWER (14551)
# and PORT_MONITOR (14552) are already bound by the in-subprocess
# baseline_formation_controller.py — UDP bind would EADDRINUSE. UI is an
# unbound MAVProxy --out endpoint that fans out the same MAVLink stream
# (see tests/start_formation_sitl_fixed.sh:441,468,475).
TELEMETRY_LISTEN_PORT_BASE = BASE_UI_PORT

# DRL timing
ACTION_REPEAT = 5                # config.ACTION_REPEAT — 4Hz DRL decisions over 20Hz
SIM_STEP_HZ = 20                 # config.TICK_HZ
EPISODE_SIM_DURATION_S = 900.0   # config.EPISODE_DURATION_S mirror (EPDUR-751); equality pinned by test_epdur_751
WARMUP_TIMEOUT_S = 480.0         # abort if HOLD_PRECONTACT not reached (sim seconds)

# FSM stdout sniffer pattern. baseline_formation_controller.py emits log lines
# like:
#   [1777746785.984] [S= 702.616s W=2740.705s] HOLD_PRECONTACT: range=10.0m ...
# (see tests/baseline_formation_controller.py:4081 log_continuous_diagnostics).
# We grep for "HOLD_PRECONTACT:" substring as the warmup-complete signal.
# State transitions to TRANSITION_SNAPSHOT.log (file, not stdout) — do NOT
# match "[TRANSITION]" because it never appears on stdout.
FSM_HOLD_PRECONTACT_PATTERN = "HOLD_PRECONTACT:"

# Residual injection rate (matches Method C ACK-throttled ~2Hz heading rate)
RESIDUAL_INJECTION_INTERVAL_S = 0.50

# Subprocess SIGTERM grace before SIGKILL escalation
SUBPROCESS_TERM_GRACE_S = 10.0

# SITL home position (matches tests/start_formation_sitl_fixed.sh `--home` arg).
# Used to instantiate GeometryCalculator before the first GLOBAL_POSITION_INT_sys1
# arrives. Once telemetry is flowing, the calculator's local-NED projection is
# anchored at this origin, so the absolute lat/lon need not match exactly — what
# matters is that *both* follower and leader are projected with the same origin.
SITL_HOME_LAT = -35.363261
SITL_HOME_LON = 149.165230
SITL_HOME_ALT_MSL = 584.0

# Boom geometry: at full extension, the DHC-2 boom hangs at 45° below
# horizontal — slant length 6.1 m projects to 4.31 m horizontal + 4.31 m
# vertical (per milestone spec / tests/geometry.py:146 / formation_config.py:99-110).
# `compute_relative` interprets boom_length_m as the HORIZONTAL projection only;
# vertical is passed separately via boom_tip_offset_m=BOOM_TIP_BELOW_OFFSET_M.
# Passing the slant length 6.1 m here over-projects horizontally by 1.79 m
# (Phase 3.6-H finding 2026-05-03).
BOOM_HORIZONTAL_PROJECTION_M = 4.31


# ============================================================================
# OPTION_C_FULL_FIELD_AUDIT helpers (2026-05-05)
# ============================================================================

def _serialize_mavlink_msg(msg: Any) -> Optional[Dict[str, Any]]:
    """Serialize a pymavlink message to a JSON-safe dict of standard scalar
    fields. Returns None if the object does not look like a MAVLink message
    (no get_fieldnames). Used by OPTION_C_FULL_FIELD_AUDIT to dump every
    cached MAVLink message field per tick.
    """
    if not hasattr(msg, "get_fieldnames"):
        return None
    out: Dict[str, Any] = {}
    try:
        fnames = msg.get_fieldnames()
    except Exception:
        return None
    for fn in fnames:
        try:
            v = getattr(msg, fn, None)
        except Exception:
            v = None
        if v is None or isinstance(v, (int, float, bool, str)):
            out[fn] = v
        elif isinstance(v, (list, tuple)):
            try:
                out[fn] = [
                    x if isinstance(x, (int, float, bool, str, type(None))) else repr(x)[:80]
                    for x in list(v)[:32]  # cap arrays at 32 entries
                ]
            except Exception:
                out[fn] = None
        elif isinstance(v, bytes):
            try:
                out[fn] = v.decode("utf-8", errors="replace")[:200]
            except Exception:
                out[fn] = repr(v)[:200]
        else:
            out[fn] = repr(v)[:200]
    return out


def _serialize_telemetry_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Walk the env's _telemetry_cache and emit a JSON-safe snapshot.
    Each MAVLink-message-typed value is serialized via _serialize_mavlink_msg.
    Non-MAVLink keys (sim_time_s, _n_msgs_drained, etc.) are passed through
    if they are JSON-safe scalars; otherwise they are repr'd.
    """
    out: Dict[str, Any] = {}
    for k, v in cache.items():
        if k is None:
            continue
        # MAVLink message?
        ser = _serialize_mavlink_msg(v)
        if ser is not None:
            out[k] = ser
            continue
        # Scalar passthrough
        if v is None or isinstance(v, (int, float, bool, str)):
            out[k] = v
        else:
            try:
                out[k] = repr(v)[:200]
            except Exception:
                out[k] = None
    return out


# ============================================================================
# MethodCFormationEnv — gym.Env wrapping Method C harness
# ============================================================================

class MethodCFormationEnv(gym.Env):
    """gym.Env wrapper around Method C harness for Option-C DRL training.

    Each reset() launches a fresh Method C subprocess stack (SITL + leader +
    MAVProxy + baseline_formation_controller) and waits for the controller
    FSM to reach HOLD_PRECONTACT. step() injects residual commands via
    PORT_CONTROL_IN and reads telemetry via the follower MAVLink stream.

    SCAFFOLD STATUS: structural placeholder. Method bodies marked TODO are
    not implemented; the file passes py_compile + import only.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id: int = 0,
        leader_profile: str = "straight",
        speedup: float = 1.0,
        force_zero_residual: bool = False,
        sitl_work_dir: Optional[str] = None,
        episode_sim_duration_s: float = EPISODE_SIM_DURATION_S,
        warmup_timeout_s: float = WARMUP_TIMEOUT_S,
        residual_injection_interval_s: float = RESIDUAL_INJECTION_INTERVAL_S,
        curriculum_stage: int = 0,
    ) -> None:
        super().__init__()

        # --- Identity / configuration ---
        self.env_id = env_id
        self.leader_profile = leader_profile
        self.speedup = speedup
        self.force_zero_residual = force_zero_residual
        self.episode_sim_duration_s = episode_sim_duration_s
        self.warmup_timeout_s = warmup_timeout_s
        self.residual_injection_interval_s = residual_injection_interval_s
        self.curriculum_stage = curriculum_stage

        # --- Derived ports (must match start_formation_sitl_fixed.sh:438-448) ---
        self._port_offset = env_id * PORT_STRIDE
        self._port_follower = BASE_FOLLOWER_PORT + self._port_offset
        self._port_control_in = BASE_CONTROL_IN_PORT + self._port_offset
        self._port_leader = BASE_LEADER_PORT + self._port_offset
        self._port_monitor = BASE_MONITOR_PORT + self._port_offset
        self._port_timesync = BASE_TIMESYNC_PORT + self._port_offset
        self._port_ui = BASE_UI_PORT + self._port_offset
        self._port_uwb_truth = BASE_UWB_TRUTH_PORT + self._port_offset
        self._port_telemetry_listen = TELEMETRY_LISTEN_PORT_BASE + self._port_offset

        # --- Per-env workdir (parallel-safe SITL isolation) ---
        if sitl_work_dir is None:
            sitl_work_dir = f"/tmp/flightdock_sitl_env{env_id}"
        self.sitl_work_dir = Path(sitl_work_dir)

        # --- Components (reused unmodified) ---
        self._obs_builder = ObservationBuilder(
            speedup_factor=self.speedup,
            wind_channels=True,
        )
        self._reward_computer = MRRLRewardComputer(stage=self.curriculum_stage)

        # --- Subprocess handles (populated on reset) ---
        self._harness_proc: Optional[subprocess.Popen] = None
        # NOTE: SITL/MAVProxy/leader/controller are spawned by start_formation_sitl_fixed.sh
        # as a single subprocess group; we track only the top-level harness PID and
        # rely on start_new_session=True for signal propagation.

        # 2026-05-08: HARNESS_OUT_DIR captured from harness stdout (echoed at
        # tests/lib_run_harness.sh:502). Used by _emit_truth_summary to write
        # run_summary_env_truth.json next to the controller's run_summary.json
        # so post-run dual-frame rollup can pair them per-episode.
        self._harness_out_dir: Optional[Path] = None
        self._truth_summary_emitted: bool = False

        # --- MAVLink connections (populated on connect) ---
        self._follower_conn = None  # pymavlink connection — read telemetry
        self._control_conn = None   # pymavlink connection — write residual commands

        # --- FSM stdout sniffer state ---
        self._fsm_state: str = "INIT"
        self._fsm_hold_precontact_seen = threading.Event()
        self._fsm_thread: Optional[threading.Thread] = None
        self._fsm_stop_event = threading.Event()
        self._stdout_lines_queue: "queue.Queue[str]" = queue.Queue(maxsize=10000)

        # --- Telemetry caches (populated by _pump_telemetry) ---
        self._telemetry_cache: Dict[str, Any] = {}
        self._first_heartbeat_seen = threading.Event()
        # Per-sysid receive counts (Phase 3.5 diagnostic).
        self._sysid_msg_count: Dict[int, int] = {}
        # Diagnostic JSONL trace handle (set in reset() if OPTION_C_GEOM_DIAG=1).
        self._geom_diag_path: Optional[Path] = None
        self._geom_diag_handle: Optional[Any] = None
        # Phase 3.6-F: timestamp-alignment diagnostic. Per-message receipt
        # timestamps captured in _pump_telemetry; per-step skew_ms written to
        # JSONL by _build_geometry when OPTION_C_TIMESTAMP_DIAG=1. No-op when
        # the env var is unset (handle stays None; recv-log writes are O(1)
        # dict ops with no observable effect on geometry computation).
        self._msg_recv_log: Dict[str, Tuple[float, Optional[float]]] = {}
        self._ts_diag_path: Optional[Path] = None
        self._ts_diag_handle: Optional[Any] = None
        # Phase 3.6-J: paired-tick (truth-side) diagnostic. Per sub-tick of step(),
        # if OPTION_C_PAIRED_TICK_DIAG=1, emit a JSONL line with truth-side
        # behind/cross/below/range from compute_relative output for post-hoc
        # correlation with controller's [SPD_ARB] perceived-side log emissions.
        # No-op when env var unset (handle stays None).
        self._paired_tick_diag_path: Optional[Path] = None
        self._paired_tick_diag_handle: Optional[Any] = None
        # OPTION_C_FULL_FIELD_AUDIT 2026-05-05: comprehensive per-tick
        # perception-vs-truth audit. Captures the entire _telemetry_cache
        # snapshot (every cached MAVLink message, every standard scalar field),
        # all SIM_STATE / GPI / ATTITUDE / VFR_HUD fields per sysid, geometry
        # NED-raw + post-projection outputs, full fusion P/K/innovation/x state.
        # Default-OFF; handle stays None when OPTION_C_FULL_FIELD_AUDIT is unset.
        # Block-buffered I/O (65 KB) to avoid I/O thrash on simultaneous emitters.
        self._full_field_audit_path: Optional[Path] = None
        self._full_field_audit_handle: Optional[Any] = None
        # Phase 3.6-A: one-time warning latch when SIM_STATE not received and
        # we fall back to GLOBAL_POSITION_INT (GPS-derived, may diverge ~6m).
        self._sim_state_warned: bool = False

        # Step C 2026-05-04: per-tick truth-side snapshot for perception
        # integrity audit. Populated only when OPTION_C_PERCEPTION_INTEGRITY_AUDIT=1.
        # None when audit gate is unset (default).
        self._last_sim_truth_audit: Optional[Dict[str, float]] = None

        # Step D 2026-05-04: Kalman-internals audit handles (Captures 4+5) +
        # V3 periodic-reset state. All default-OFF; gated on env vars.
        self._step_d_audit_enabled: bool = False
        self._kalman_audit_dir: Optional[Path] = None
        self._input_consistency_path: Optional[Path] = None
        self._input_consistency_handle: Optional[Any] = None
        self._leader_packet_path: Optional[Path] = None
        self._leader_packet_handle: Optional[Any] = None
        # V3 periodic-reset wall-clock state. Interval set from
        # OPTION_C_RESET_PERIODIC_S env var; 0.0 last-trigger means never fired.
        self._periodic_reset_interval_s: Optional[float] = None
        try:
            _periodic_s = os.environ.get("OPTION_C_RESET_PERIODIC_S")
            if _periodic_s:
                self._periodic_reset_interval_s = float(_periodic_s)
        except (ValueError, TypeError):
            self._periodic_reset_interval_s = None
        self._last_periodic_reset_wall_t: float = 0.0

        # --- Geometry calculator (single instance, anchored at SITL home) ---
        # Phase 3.6-A SIM_STATE-direct path: _build_geometry passes follower
        # position from SIM_STATE truth (no uwb_range kwarg) so compute_relative
        # returns geometry referenced to the leader boom tip without the
        # internal Kalman fusion's UWB tightening branch being engaged.
        self._geom_calc = GeometryCalculator(
            home_lat=SITL_HOME_LAT,
            home_lon=SITL_HOME_LON,
            home_alt_msl=SITL_HOME_ALT_MSL,
        )
        self._last_geom: Optional[RelativeGeometry] = None
        # PATH F+ additive diagnostic 2026-05-03: parallel raw-truth GeometryCalculator
        # for tracker-channel separation. fusion=None routes compute_relative through
        # the raw-GPS branch (tests/geometry.py L829-832), yielding direct SIM_STATE
        # follower + leader-broadcast geometry without Kalman smoothing. Used by the
        # paired-tick emitter to log raw_truth_* alongside env_fused_*.
        self._geom_calc_raw = GeometryCalculator(
            home_lat=SITL_HOME_LAT,
            home_lon=SITL_HOME_LON,
            home_alt_msl=SITL_HOME_ALT_MSL,
        )
        self._geom_calc_raw.fusion = None
        self._last_raw_geom: Optional[RelativeGeometry] = None
        # Probe A (follower-decomp 2026-05-03): a SECOND no-fusion GeometryCalculator
        # used to compute geometry from GLOBAL_POSITION_INT_sys1 follower input
        # (ArduPilot's onboard EKF state) instead of SIM_STATE truth. Lets the
        # paired-tick log emit a fourth tracker channel `gpi_direct_*` so the
        # raw_truth − gpi_direct gap quantifies ArduPilot EKF vs truth, and
        # the gpi_direct − env_fused gap isolates RelativePoseFusion's effect on
        # the same input controller's fusion sees. Gated on OPTION_C_GPI_DIRECT_DIAG=1
        # to keep emission cost zero when not testing.
        self._geom_calc_gpi = GeometryCalculator(
            home_lat=SITL_HOME_LAT,
            home_lon=SITL_HOME_LON,
            home_alt_msl=SITL_HOME_ALT_MSL,
        )
        self._geom_calc_gpi.fusion = None
        self._last_gpi_geom: Optional[RelativeGeometry] = None
        self._last_step_dt_s: float = 0.25  # ACTION_REPEAT(5) × SIM_STEP_HZ(20)^-1

        # --- Episode-state accumulators (reset() zeroes; updated by step()) ---
        self._in_m3b: bool = False
        self._continuous_m3b_dwell_s: float = 0.0
        self._best_m3b_dwell_s: float = 0.0
        self._m3b_entry_count: int = 0
        self._min_range_m: float = float("inf")
        self._prev_applied_residual: np.ndarray = np.zeros(3, dtype=np.float32)
        self._safety_violation_count: int = 0

        # --- Episode bookkeeping ---
        self._episode_start_sim_s: float = 0.0
        self._tick_count: int = 0
        self._last_residual_send_time: float = 0.0
        self._last_baseline_cmd: Optional[Dict[str, float]] = None
        self._last_telemetry: Optional[Dict[str, Any]] = None

        # --- Observation / action spaces (must match env_formation_sitl.py:278-286) ---
        self.observation_space = spaces.Box(
            low=-config.OBS_CLIP,
            high=config.OBS_CLIP,
            shape=(config.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.ACTION_DIM,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _build_harness_env(self) -> Dict[str, str]:
        """Construct env-var dict for harness subprocess invocation."""
        env = os.environ.copy()
        env["HARNESS_PORT_OFFSET"] = str(self._port_offset)
        env["HARNESS_LEADER_PROFILE"] = self.leader_profile
        env["HARNESS_SPEEDUP"] = str(self.speedup)
        env["HARNESS_SIM_SECONDS"] = str(int(self.episode_sim_duration_s))
        env["HARNESS_PARALLEL_SAFE"] = "1"
        env["HARNESS_UWB_ENABLE"] = "1"
        env["HARNESS_UWB_TRUTH_SOURCE"] = "sim_state"
        env["HARNESS_UWB_RATE_HZ"] = "10"
        env["SITL_WORK_DIR"] = str(self.sitl_work_dir)
        env["HARNESS_ENV_ID"] = str(self.env_id)
        return env

    def _launch_harness(self) -> bool:
        """Launch Method C subprocess stack via run_gated_matrix_v910.sh.

        run_gated_matrix orchestrates SITL → leader_broadcast → controller in
        sequence (see tests/run_gated_matrix_v910.sh:105-225 run_single_speedup).
        We Popen it as a single process group; SIGTERM to the group propagates
        to all children via the harness_cleanup EXIT INT TERM trap
        (tests/lib_run_harness.sh:513).

        Returns True when controller stdout emits HOLD_PRECONTACT marker;
        False on warmup_timeout_s wall budget exhaustion.
        """
        project_root = Path(__file__).resolve().parents[2]
        harness_script = project_root / "tests" / "run_gated_matrix_v910.sh"
        if not harness_script.exists():
            raise FileNotFoundError(f"harness script missing: {harness_script}")

        env = self._build_harness_env()
        # PYTHONUNBUFFERED forces line-buffered stdout in child python procs
        # (controller, leader). Without it, log lines get coalesced behind
        # the default ~8KB pipe buffer and HOLD_PRECONTACT detection lags.
        env["PYTHONUNBUFFERED"] = "1"

        # Per-lane SITL_WORK_DIR (matches lib_run_harness.sh:790
        # /tmp/flightdock_sitl_env{env_id}_$$ pattern; we use a stable suffix
        # so close() can clean up reliably without knowing the shell PID).
        self.sitl_work_dir.mkdir(parents=True, exist_ok=True)

        # Wall budget: warmup_timeout_s sim seconds + a 30s SITL boot margin.
        # At 1.0× speedup, sim s ≈ wall s.
        wall_budget_s = self.warmup_timeout_s / max(self.speedup, 1.0) + 30.0
        # Pass --duration in sim seconds; --speedups "1x" — single-run mode.
        cmd = [
            "bash",
            str(harness_script),
            "--duration",
            str(int(self.episode_sim_duration_s)),
            "--speedups",
            f"{int(self.speedup)}x" if self.speedup == int(self.speedup) else f"{self.speedup}x",
        ]

        self._fsm_hold_precontact_seen.clear()
        self._fsm_stop_event.clear()

        # start_new_session=True creates new pgid so SIGTERM to that pgid
        # cleanly propagates to all descendants (SITL, MAVProxy, leader,
        # controller). Combined cleanup trap inside run_gated_matrix
        # ensures graceful shutdown.
        self._harness_proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,                     # line-buffered
            universal_newlines=True,
            start_new_session=True,
        )

        # Spawn daemonized sniffer thread reading subprocess stdout.
        self._fsm_thread = threading.Thread(
            target=self._fsm_sniffer_loop,
            name=f"MethodC-FSM-env{self.env_id}",
            daemon=True,
        )
        self._fsm_thread.start()

        # Wait for HOLD_PRECONTACT or wall timeout.
        deadline = time.monotonic() + wall_budget_s
        check_interval_s = 1.0
        while time.monotonic() < deadline:
            if self._fsm_hold_precontact_seen.wait(timeout=check_interval_s):
                self._fsm_state = "HOLD_PRECONTACT"
                # Step D V2 FSM-gated covariance floor — propagate warmup-end
                # phase to env-side fusion. Env doesn't observe controller
                # transitions in real time; fusion stays at HOLD_PRECONTACT
                # for the rest of the episode (acceptable since post-warmup
                # states are all in the floor allowlist). No-op when gate unset.
                if os.environ.get("OPTION_C_FSM_GATED_FLOOR") == "1":
                    if (getattr(self, "_geom_calc", None) is not None
                            and getattr(self._geom_calc, "fusion", None) is not None
                            and hasattr(self._geom_calc.fusion, "set_fsm_phase")):
                        self._geom_calc.fusion.set_fsm_phase("HOLD_PRECONTACT")
                return True
            # Detect early subprocess death.
            if self._harness_proc.poll() is not None:
                rc = self._harness_proc.returncode
                self._fsm_state = f"DIED_RC={rc}"
                return False

        # Warmup timeout — bail out (caller will _terminate_harness on False).
        return False

    def _fsm_sniffer_loop(self) -> None:
        """Daemon thread: read subprocess stdout, set hold_precontact event on match."""
        if self._harness_proc is None or self._harness_proc.stdout is None:
            return
        try:
            for line in iter(self._harness_proc.stdout.readline, ""):
                if self._fsm_stop_event.is_set():
                    break
                if not line:
                    break
                # Buffer for diagnostic dump; non-blocking put.
                try:
                    self._stdout_lines_queue.put_nowait(line.rstrip("\n"))
                except queue.Full:
                    pass
                if FSM_HOLD_PRECONTACT_PATTERN in line:
                    self._fsm_hold_precontact_seen.set()
                # 2026-05-08: capture HARNESS_OUT_DIR for dual-frame rollup.
                # Harness echoes `HARNESS_OUT_DIR=<path>` once at init (line ~502
                # of tests/lib_run_harness.sh). Latch on first match per episode.
                if self._harness_out_dir is None and "HARNESS_OUT_DIR=" in line:
                    try:
                        cand = line.split("HARNESS_OUT_DIR=", 1)[1].strip()
                        if cand:
                            self._harness_out_dir = Path(cand)
                    except (IndexError, ValueError):
                        pass
        except (ValueError, OSError):
            # Pipe closed during teardown — expected.
            pass

    def _connect_mavlink(self) -> None:
        """Open pymavlink UDP connections after harness launch.

        Telemetry: bind UI port (14554+offset) — MAVProxy --out broadcasts the
        full MAVLink stream there with no in-process consumer (PORT_FOLLOWER
        and PORT_MONITOR are owned by baseline_formation_controller.py and
        LeadTelemetryFeed respectively; binding either would EADDRINUSE).

        Control: udpout to PORT_CONTROL_IN (14560+offset), the MAVProxy INPUT
        link configured for external command injection
        (tests/start_formation_sitl_fixed.sh:443,468,475).
        """
        if mavutil is None:
            raise RuntimeError(
                "pymavlink unavailable; install pymavlink before opening MAVLink"
            )

        # Open telemetry RX on UI port. Use udpin:0.0.0.0 so SO_REUSEADDR
        # behaviour is consistent across platforms.
        listen_uri = f"udpin:0.0.0.0:{self._port_telemetry_listen}"
        self._follower_conn = mavutil.mavlink_connection(
            listen_uri,
            source_system=255,         # GCS-style (avoids sysid clash with FCU=1, leader=2)
            source_component=190,      # MAV_COMP_ID_USER1 — distinct from MAVProxy
            autoreconnect=False,
            input=True,
            dialect="ardupilotmega",
        )

        # Open control TX to PORT_CONTROL_IN. udpout sends without binding;
        # MAVProxy receives via its `link add udpin:...` clause.
        control_uri = f"udpout:127.0.0.1:{self._port_control_in}"
        self._control_conn = mavutil.mavlink_connection(
            control_uri,
            source_system=255,
            source_component=190,
            autoreconnect=False,
            input=False,
            dialect="ardupilotmega",
        )

        # Wait for first HEARTBEAT (max 30s wall) — confirms MAVProxy fanout
        # is online and our UDP bind is receiving traffic.
        self._first_heartbeat_seen.clear()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            msg = self._follower_conn.recv_match(
                type="HEARTBEAT", blocking=False
            )
            if msg is not None:
                self._first_heartbeat_seen.set()
                self._telemetry_cache["HEARTBEAT"] = msg
                return
            time.sleep(0.1)

        # Heartbeat not seen — soft failure; smoke test will surface this
        # via empty telemetry cache. We do NOT raise here so that the smoke
        # can validate downstream lifecycle paths (close cleanup, etc.).

    def _terminate_harness(self) -> None:
        """SIGTERM all subprocess handles in reverse launch order.

        TODO:
        - Stop FSM sniffer thread (set _fsm_stop_event)
        - SIGTERM controller subprocess (if separately tracked)
        - SIGTERM harness top-level process group (kills SITL + MAVProxy + leader)
        - Wait SUBPROCESS_TERM_GRACE_S, then SIGKILL if still alive
        - Close MAVLink connections
        - Verify lockfiles cleaned
        """
        # Defensive cleanup of MAVLink first (release ports)
        if self._follower_conn is not None:
            try:
                self._follower_conn.close()
            except Exception:
                pass
            self._follower_conn = None
        if self._control_conn is not None:
            try:
                self._control_conn.close()
            except Exception:
                pass
            self._control_conn = None

        # Stop FSM sniffer
        self._fsm_stop_event.set()
        if self._fsm_thread is not None and self._fsm_thread.is_alive():
            self._fsm_thread.join(timeout=2.0)
        self._fsm_thread = None

        # SIGTERM harness process group
        if self._harness_proc is not None and self._harness_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._harness_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._harness_proc.wait(timeout=SUBPROCESS_TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._harness_proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self._harness_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        self._harness_proc = None

    # ------------------------------------------------------------------
    # Telemetry / observation
    # ------------------------------------------------------------------

    def _pump_telemetry(self) -> Dict[str, Any]:
        """Non-blocking MAVLink drain. Updates per-(sysid,type) cache.

        Reads up to MAX_PUMP_MSGS messages per call. Records each into
        self._telemetry_cache keyed by (sysid, msg_type) so follower
        (sysid=1) and leader (sysid=2) can be distinguished.

        Returns the cache dict (caller may mutate; we return a reference).
        """
        if self._follower_conn is None:
            return self._telemetry_cache

        # Drain ALL pending messages (Phase 3.5 fix: previously capped at 32,
        # which under heavy MAVProxy fanout meant we accumulated backlog in
        # the OS UDP receive buffer → ~217s stale telemetry observed in
        # Phase 3 verify run).
        n_msgs_drained = 0
        MAX_PUMP_MSGS = 4096  # high cap; only break on empty
        for _ in range(MAX_PUMP_MSGS):
            msg = self._follower_conn.recv_match(blocking=False)
            if msg is None:
                break
            n_msgs_drained += 1
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue
            sysid = getattr(msg, "_header", None)
            sysid = sysid.srcSystem if sysid is not None else 0
            self._sysid_msg_count[sysid] = self._sysid_msg_count.get(sysid, 0) + 1
            key = f"{mtype}_sys{sysid}"
            self._telemetry_cache[key] = msg
            # Also keep latest by type for convenience.
            self._telemetry_cache[mtype] = msg
            # Phase 3.6-F: record receipt monotonic-time + msg time_boot_ms for
            # SIM_STATE and leader GLOBAL_POSITION_INT (sysid=2). Used by
            # _build_geometry to compute follower/leader sample-time skew when
            # OPTION_C_TIMESTAMP_DIAG=1; no-op when env var unset.
            if key in ("SIM_STATE_sys1", "SIM_STATE_sys0", "SIM_STATE",
                       "GLOBAL_POSITION_INT_sys2"):
                self._msg_recv_log[key] = (
                    time.monotonic(),
                    getattr(msg, "time_boot_ms", None),
                )

        self._telemetry_cache["_n_msgs_drained"] = n_msgs_drained

        # Best-effort sim_time extraction: MAVLink SYSTEM_TIME or use wall time.
        sim_time_msg = self._telemetry_cache.get("SYSTEM_TIME")
        if sim_time_msg is not None:
            sim_time_s = getattr(sim_time_msg, "time_boot_ms", 0) / 1000.0
        else:
            sim_time_s = time.monotonic() - getattr(
                self, "_episode_start_wall", time.monotonic()
            )
        self._telemetry_cache["sim_time_s"] = sim_time_s
        return self._telemetry_cache

    def _get_sim_state(self, telemetry: Dict[str, Any]):
        """Phase 3.6-A: return cached SIM_STATE message or None.

        SITL emits SIM_STATE with sysid=1 typically, but pymavlink may strip
        sysid for SITL-internal messages (sysid=0/None). Check all common keys.
        SIM_STATE provides physics-truth follower position (lat/lon in degrees,
        alt in meters MSL) — matches the controller's UWB-fused tracker view,
        bypassing GPS drift that injected ~6m error in Phase 3.5 verification.
        """
        return (
            telemetry.get("SIM_STATE_sys1")
            or telemetry.get("SIM_STATE_sys0")
            or telemetry.get("SIM_STATE")
        )

    def _build_geometry(
        self, telemetry: Dict[str, Any]
    ) -> Optional[RelativeGeometry]:
        """Compute RelativeGeometry in boom-tip-Δ frame from MAVLink telemetry.

        Phase 3.6-A SIM_STATE-direct path: follower position is sourced from
        physics-truth SIM_STATE so the env's geometry view does not incur GPS
        noise (~6m vs truth). compute_relative is called WITHOUT the uwb_range
        kwarg, so the internal Kalman fusion's UWB tightening branch is not
        engaged.

        Reads sysid-keyed cache:
          - SIM_STATE → follower physics-truth (PRIMARY follower position source
            via Fix C1: lat_int / 1e7 from ardupilotmega dialect; falls back to
            float lat/lon, then to GLOBAL_POSITION_INT_sys1 with one-time WARNING)
          - GLOBAL_POSITION_INT_sys1 → follower NED velocities (SIM_STATE has
            none) + fallback follower position when SIM_STATE absent
          - GLOBAL_POSITION_INT_sys2 → leader lat/lon/alt + NED velocity + heading
            (leader has no SITL backing, so SIM_STATE for leader doesn't exist)

        Returns RelativeGeometry referenced to the boom TIP at the corrected
        horizontal projection (BOOM_HORIZONTAL_PROJECTION_M=4.31m) plus
        BOOM_TIP_BELOW_OFFSET_M=4.31m vertical drop (Phase 3.6-H 2026-05-03).
        Boom is fully extended by sim ~15s; env's reset returns at sim ~477s,
        so always post-extension when env steps.
        """
        lead = telemetry.get("GLOBAL_POSITION_INT_sys2")
        foll = telemetry.get("GLOBAL_POSITION_INT_sys1")
        if lead is None or foll is None:
            return self._last_geom

        # ---- Follower position: SIM_STATE-first per Phase 3.6-A Fix C1 ----
        # PRIMARY: SIM_STATE.lat_int / .lon_int (degE7 int32 — ardupilotmega
        #   canonical source; pymavlink float lat/lon is not populated as
        #   degrees by SITL — int32 fields are authoritative).
        # FALLBACK 1: SIM_STATE.lat / .lon (float "degrees" — older schemas).
        # FALLBACK 2: GLOBAL_POSITION_INT_sys1 (GPS, ~6m std vs truth) with
        #   one-time WARNING via self._sim_state_warned latch.
        sim = self._get_sim_state(telemetry)
        foll_lat: Optional[float] = None
        foll_lon: Optional[float] = None
        foll_alt: Optional[float] = None
        _sim_vn: Optional[float] = None
        _sim_ve: Optional[float] = None
        _sim_vd: Optional[float] = None
        # Step D 2026-05-04: track follower-position source for input-consistency
        # audit (Capture 4). Default-OFF: variable set always but consumed only
        # by the audit log writer below.
        foll_pos_source: str = "none"
        if sim is not None:
            lat_int = getattr(sim, "lat_int", None)
            lon_int = getattr(sim, "lon_int", None)
            if lat_int is not None and lon_int is not None:
                foll_lat = lat_int / 1e7
                foll_lon = lon_int / 1e7
                foll_pos_source = "sim_state.lat_int"
            else:
                foll_lat = float(sim.lat) / 1e7
                foll_lon = float(sim.lon) / 1e7
                foll_pos_source = "sim_state.lat_float_e7"
            foll_alt = float(sim.alt)
            _sim_vn = float(getattr(sim, "vn", 0.0))
            _sim_ve = float(getattr(sim, "ve", 0.0))
            _sim_vd = float(getattr(sim, "vd", 0.0))
            # Step C 2026-05-04: snapshot SIM_STATE truth fields for perception
            # integrity audit. Defensive getattr defaults handle older SITL
            # schemas. Default-OFF (audit gate unset → no snapshot).
            if os.environ.get("OPTION_C_PERCEPTION_INTEGRITY_AUDIT", "1") != "0":
                self._last_sim_truth_audit = {
                    "vn": float(getattr(sim, "vn", 0.0)),
                    "ve": float(getattr(sim, "ve", 0.0)),
                    "vd": float(getattr(sim, "vd", 0.0)),
                    "roll": float(getattr(sim, "roll", 0.0)),
                    "pitch": float(getattr(sim, "pitch", 0.0)),
                    "yaw": float(getattr(sim, "yaw", 0.0)),
                    "xgyro": float(getattr(sim, "xgyro", 0.0)),
                    "ygyro": float(getattr(sim, "ygyro", 0.0)),
                    "zgyro": float(getattr(sim, "zgyro", 0.0)),
                }
        if foll_lat is None or foll_lon is None or foll_alt is None:
            if not self._sim_state_warned:
                logger.warning(
                    f"[env_id={self.env_id}] SIM_STATE not received; falling "
                    "back to GLOBAL_POSITION_INT_sys1 GPS (~6m noise vs truth). "
                    "Verify HARNESS_UWB_TRUTH_SOURCE=sim_state and MAVProxy --out=14554 fanout."
                )
                self._sim_state_warned = True
            foll_lat = foll.lat / 1e7
            foll_lon = foll.lon / 1e7
            foll_alt = foll.alt / 1000.0
            foll_pos_source = "gpi_fallback"
        # Follower velocity: gate USE_SIMSTATE_VEL_FOR_ENV_FUSION=1 to substitute
        # SIM_STATE vn/ve/vd (truth, m/s) for GPI vx/vy/vz (EKF, cm/s).
        _use_sv = (
            os.environ.get("USE_SIMSTATE_VEL_FOR_ENV_FUSION") == "1"
            and _sim_vn is not None
        )
        foll_vn = _sim_vn if _use_sv else foll.vx / 100.0
        foll_ve = _sim_ve if _use_sv else foll.vy / 100.0
        foll_vd = _sim_vd if _use_sv else foll.vz / 100.0
        # Phase 4 Option 3: stash for downstream channel-routing gates (reward/eval source assertion).
        self._foll_pos_source = foll_pos_source
        self._use_simstate_vel = _use_sv
        self._foll_lat_deg = foll_lat
        self._foll_lon_deg = foll_lon

        # ---- Leader position (sysid=2 from leader_broadcast_fixed.py) ----
        # Leader has no SITL backing — leader_broadcast emits only
        # GLOBAL_POSITION_INT/VFR_HUD/ATTITUDE; no SIM_STATE for leader exists.
        lead_lat = lead.lat / 1e7
        lead_lon = lead.lon / 1e7
        lead_alt = lead.alt / 1000.0
        lead_vn = lead.vx / 100.0
        lead_ve = lead.vy / 100.0
        lead_vd = lead.vz / 100.0
        lead_hdg = lead.hdg / 100.0  # cdeg → deg

        try:
            geom = self._geom_calc.compute_relative(
                foll_lat=foll_lat,
                foll_lon=foll_lon,
                foll_alt=foll_alt,
                foll_vn=foll_vn,
                foll_ve=foll_ve,
                foll_vd=foll_vd,
                lead_lat=lead_lat,
                lead_lon=lead_lon,
                lead_alt=lead_alt,
                lead_vn=lead_vn,
                lead_ve=lead_ve,
                lead_vd=lead_vd,
                lead_heading_deg=lead_hdg,
                # Phase 3.6-H 2026-05-03: pass HORIZONTAL projection (4.31m), not slant
                # length (6.1m). compute_relative treats boom_length_m as horizontal-only;
                # vertical is handled separately via boom_tip_offset_m. Boom hangs at 45°
                # at full extension — horizontal = vertical = 4.31m, slant = 6.10m
                # (matches tests/geometry.py:146 BOOM_TIP_BELOW_OFFSET_M = 4.31).
                # Always post-extension by env reset at sim ~477s.
                boom_length_m=BOOM_HORIZONTAL_PROJECTION_M,
                timestamp_s=telemetry.get("sim_time_s", time.monotonic()),
                boom_tip_offset_m=BOOM_TIP_BELOW_OFFSET_M,
            )
        except Exception:
            return self._last_geom

        if geom is not None:
            self._last_geom = geom
            self._min_range_m = min(self._min_range_m, geom.range_m)

        # PATH F+ additive diagnostic 2026-05-03: parallel raw-truth compute on the
        # same SIM_STATE / leader-broadcast inputs, with no Kalman fusion. Stashes
        # on self._last_raw_geom for the paired-tick emitter to log alongside fused.
        try:
            raw_geom = self._geom_calc_raw.compute_relative(
                foll_lat=foll_lat,
                foll_lon=foll_lon,
                foll_alt=foll_alt,
                foll_vn=foll_vn,
                foll_ve=foll_ve,
                foll_vd=foll_vd,
                lead_lat=lead_lat,
                lead_lon=lead_lon,
                lead_alt=lead_alt,
                lead_vn=lead_vn,
                lead_ve=lead_ve,
                lead_vd=lead_vd,
                lead_heading_deg=lead_hdg,
                boom_length_m=BOOM_HORIZONTAL_PROJECTION_M,
                timestamp_s=telemetry.get("sim_time_s", time.monotonic()),
                boom_tip_offset_m=BOOM_TIP_BELOW_OFFSET_M,
            )
            if raw_geom is not None:
                self._last_raw_geom = raw_geom
        except Exception:
            pass

        # === STEP D AUDIT (Captures 4+5): per-tick input-consistency + leader-packet
        # writes. Default-OFF; both handles None unless OPTION_C_KALMAN_INTERNALS_AUDIT
        # was set at reset(). raw_geom is no-fusion truth (geom_calc_raw); geom is
        # post-fusion (geom_calc with full Kalman). The gap quantifies the fusion
        # algorithm's effect on truth-grade input — Phase 4 Analysis F's discriminator
        # for STEP_D_INPUT_CONSTRUCTION_BUG_CONFIRMED vs in-Kalman defect.
        if self._step_d_audit_enabled:
            sim_t_now = telemetry.get("sim_time_s")
            wall_t_now = time.time()
            if self._leader_packet_handle is not None:
                try:
                    lp_rec = {
                        "tick_idx": self._tick_count,
                        "sim_time_s": sim_t_now,
                        "wall_t": wall_t_now,
                        "lead_lat_int_e7": int(getattr(lead, "lat", 0)),
                        "lead_lon_int_e7": int(getattr(lead, "lon", 0)),
                        "lead_alt_mm": int(getattr(lead, "alt", 0)),
                        "lead_vx_cms": int(getattr(lead, "vx", 0)),
                        "lead_vy_cms": int(getattr(lead, "vy", 0)),
                        "lead_vz_cms": int(getattr(lead, "vz", 0)),
                        "lead_hdg_cdeg": int(getattr(lead, "hdg", 0)),
                        "lead_time_boot_ms": int(getattr(lead, "time_boot_ms", 0)),
                    }
                    self._leader_packet_handle.write(json.dumps(lp_rec) + "\n")
                except Exception:
                    pass
            if self._input_consistency_handle is not None:
                try:
                    ic_rec = {
                        "tick_idx": self._tick_count,
                        "sim_time_s": sim_t_now,
                        "wall_t": wall_t_now,
                        # Inputs to compute_relative (which constructs GPSMeasurement
                        # for fusion.update_gps inside tests/geometry.py:805-814).
                        "foll_lat": foll_lat, "foll_lon": foll_lon, "foll_alt": foll_alt,
                        "foll_vn": foll_vn, "foll_ve": foll_ve, "foll_vd": foll_vd,
                        "foll_pos_source": foll_pos_source,
                        "lead_lat": lead_lat, "lead_lon": lead_lon, "lead_alt": lead_alt,
                        "lead_vn": lead_vn, "lead_ve": lead_ve, "lead_vd": lead_vd,
                        "lead_hdg_deg": lead_hdg,
                        "boom_length_m": BOOM_HORIZONTAL_PROJECTION_M,
                        "boom_tip_offset_m": BOOM_TIP_BELOW_OFFSET_M,
                        # Outputs: env-fused geom vs raw-truth no-fusion geom.
                        "geom_fused_behind_m": geom.behind_m if geom is not None else None,
                        "geom_fused_cross_m": geom.cross_m if geom is not None else None,
                        "geom_fused_below_m": geom.below_m if geom is not None else None,
                        "geom_fused_range_m": geom.range_m if geom is not None else None,
                        "raw_truth_behind_m": (
                            self._last_raw_geom.behind_m
                            if self._last_raw_geom is not None else None
                        ),
                        "raw_truth_cross_m": (
                            self._last_raw_geom.cross_m
                            if self._last_raw_geom is not None else None
                        ),
                        "raw_truth_below_m": (
                            self._last_raw_geom.below_m
                            if self._last_raw_geom is not None else None
                        ),
                        "raw_truth_range_m": (
                            self._last_raw_geom.range_m
                            if self._last_raw_geom is not None else None
                        ),
                    }
                    self._input_consistency_handle.write(json.dumps(ic_rec) + "\n")
                except Exception:
                    pass

        # Probe A (follower-decomp 2026-05-03): parallel GPI_SYS1_DIRECT compute.
        # Same leader-broadcast inputs, but follower position from
        # GLOBAL_POSITION_INT_sys1 (ArduPilot EKF estimate) instead of SIM_STATE
        # truth. No Kalman. Stashes on self._last_gpi_geom for the paired-tick
        # emitter to log alongside raw_truth_* and env_fused_*.
        if (os.environ.get("OPTION_C_GPI_DIRECT_DIAG") == "1"
                or os.environ.get("OPTION_C_ACTOR_OBS_FROM_GPI") == "1"):
            try:
                gpi_lat = foll.lat / 1e7
                gpi_lon = foll.lon / 1e7
                gpi_alt = foll.alt / 1000.0
                gpi_geom = self._geom_calc_gpi.compute_relative(
                    foll_lat=gpi_lat,
                    foll_lon=gpi_lon,
                    foll_alt=gpi_alt,
                    foll_vn=foll_vn,
                    foll_ve=foll_ve,
                    foll_vd=foll_vd,
                    lead_lat=lead_lat,
                    lead_lon=lead_lon,
                    lead_alt=lead_alt,
                    lead_vn=lead_vn,
                    lead_ve=lead_ve,
                    lead_vd=lead_vd,
                    lead_heading_deg=lead_hdg,
                    boom_length_m=BOOM_HORIZONTAL_PROJECTION_M,
                    timestamp_s=telemetry.get("sim_time_s", time.monotonic()),
                    boom_tip_offset_m=BOOM_TIP_BELOW_OFFSET_M,
                )
                if gpi_geom is not None:
                    self._last_gpi_geom = gpi_geom
            except Exception:
                pass

        # Phase 3.5 diagnostic — write raw inputs + computed geometry to JSONL
        # for offline staleness/divergence analysis. No-op if not enabled.
        if self._geom_diag_handle is not None:
            try:
                rec = {
                    "wall": time.time(),
                    "tick_count": self._tick_count,
                    "sim_time_s_env": telemetry.get("sim_time_s"),
                    "n_msgs_drained_this_pump": telemetry.get("_n_msgs_drained"),
                    "sysid_msg_counts": dict(self._sysid_msg_count),
                    "foll_lat": foll_lat, "foll_lon": foll_lon, "foll_alt": foll_alt,
                    "foll_vn": foll_vn, "foll_ve": foll_ve, "foll_vd": foll_vd,
                    "lead_lat": lead_lat, "lead_lon": lead_lon, "lead_alt": lead_alt,
                    "lead_vn": lead_vn, "lead_ve": lead_ve, "lead_vd": lead_vd,
                    "lead_hdg": lead_hdg,
                    "foll_time_boot_ms": getattr(foll, "time_boot_ms", None),
                    "lead_time_boot_ms": getattr(lead, "time_boot_ms", None),
                    "geom_behind_m": geom.behind_m if geom else None,
                    "geom_cross_m": geom.cross_m if geom else None,
                    "geom_below_m": geom.below_m if geom else None,
                    "geom_range_m": geom.range_m if geom else None,
                    "geom_closure_rate_ms": geom.closure_rate_ms if geom else None,
                }
                self._geom_diag_handle.write(json.dumps(rec) + "\n")
                # Flush every 50 records to keep file inspectable mid-run.
                if self._tick_count % 50 == 0:
                    self._geom_diag_handle.flush()
            except Exception:
                pass

        # Phase 3.6-F timestamp-alignment diagnostic: write per-step record of
        # follower vs leader message age (ms since receipt) and skew_ms
        # (leader_age - follower_age). Gated on OPTION_C_TIMESTAMP_DIAG=1.
        if self._ts_diag_handle is not None:
            try:
                now_mono = time.monotonic()
                foll_recv = (
                    self._msg_recv_log.get("SIM_STATE_sys1")
                    or self._msg_recv_log.get("SIM_STATE_sys0")
                    or self._msg_recv_log.get("SIM_STATE")
                )
                lead_recv = self._msg_recv_log.get("GLOBAL_POSITION_INT_sys2")
                foll_age_ms = (
                    (now_mono - foll_recv[0]) * 1000.0 if foll_recv else None
                )
                lead_age_ms = (
                    (now_mono - lead_recv[0]) * 1000.0 if lead_recv else None
                )
                skew_ms = (
                    (lead_age_ms - foll_age_ms)
                    if (foll_age_ms is not None and lead_age_ms is not None)
                    else None
                )
                ts_rec = {
                    "wall": time.time(),
                    "tick_count": self._tick_count,
                    "sim_time_s": telemetry.get("sim_time_s"),
                    "follower_msg_age_ms": foll_age_ms,
                    "leader_msg_age_ms": lead_age_ms,
                    "skew_ms": skew_ms,
                    "foll_time_boot_ms": foll_recv[1] if foll_recv else None,
                    "lead_time_boot_ms": lead_recv[1] if lead_recv else None,
                    "behind_m": geom.behind_m if geom else None,
                    "cross_m": geom.cross_m if geom else None,
                    "below_m": geom.below_m if geom else None,
                    "range_m": geom.range_m if geom else None,
                    "min_range_m_so_far": self._min_range_m,
                }
                self._ts_diag_handle.write(json.dumps(ts_rec) + "\n")
                if self._tick_count % 50 == 0:
                    self._ts_diag_handle.flush()
            except Exception:
                pass

        return geom

    def _get_obs(self, telemetry: Dict[str, Any]) -> np.ndarray:
        """Build the (config.OBS_DIM,) observation via observation_builder.build
        (R7.2 unmodified).

        Constructs FollowerState/LeaderState/EnvelopeFlags/SensorInfo from
        telemetry + computed geometry. If telemetry incomplete, returns zeros
        (matches Method B fallback at env_formation_sitl.py:868).
        """
        geom = self._build_geometry(telemetry)
        foll_pos = telemetry.get("GLOBAL_POSITION_INT_sys1")
        lead_pos = telemetry.get("GLOBAL_POSITION_INT_sys2")
        if geom is None or foll_pos is None or lead_pos is None:
            return np.zeros(config.OBS_DIM, dtype=np.float32)

        foll_vfr = telemetry.get("VFR_HUD_sys1") or telemetry.get("VFR_HUD")
        foll_att = telemetry.get("ATTITUDE_sys1") or telemetry.get("ATTITUDE")
        lead_vfr = telemetry.get("VFR_HUD_sys2")

        foll_airspeed = foll_vfr.airspeed if foll_vfr is not None else 24.7
        foll_throttle = float(foll_vfr.throttle) if foll_vfr is not None else 50.0
        foll_pitch = (math.degrees(foll_att.pitch) if foll_att is not None else 0.0)
        foll_roll = (math.degrees(foll_att.roll) if foll_att is not None else 0.0)

        # Phase 3.6-A: prefer SIM_STATE for follower altitude/heading/Vd to match
        # the geometry source. SIM_STATE.yaw is rad; .alt is m MSL; .vd is m/s NED.
        sim = self._get_sim_state(telemetry)
        if sim is not None:
            foll_alt_msl = float(sim.alt)
            foll_heading_deg = math.degrees(float(sim.yaw)) % 360.0
            foll_vd_ms = float(sim.vd)
        else:
            foll_alt_msl = foll_pos.alt / 1000.0
            foll_heading_deg = foll_pos.hdg / 100.0
            foll_vd_ms = foll_pos.vz / 100.0

        follower = FollowerState(
            airspeed_ms=foll_airspeed,
            alt_msl_m=foll_alt_msl,
            heading_deg=foll_heading_deg,
            pitch_deg=foll_pitch,
            roll_deg=foll_roll,
            vd_ms=foll_vd_ms,
            throttle_pct=foll_throttle,
        )
        leader = LeaderState(
            airspeed_ms=(lead_vfr.airspeed if lead_vfr is not None else 24.7),
            heading_deg=lead_pos.hdg / 100.0,
            alt_msl_m=lead_pos.alt / 1000.0,
            vd_ms=lead_pos.vz / 100.0,
        )

        # Envelope flags from boom-tip-Δ frame thresholds.
        in_inner = (
            abs(geom.behind_m - config.PRECONTACT_BEHIND_M
                if hasattr(config, "PRECONTACT_BEHIND_M") else geom.behind_m - 10.0) <= 1.2
            and abs(geom.cross_m) <= 1.0
            and abs(geom.below_m - 10.0) <= 1.2
        )
        in_outer = (
            abs(geom.behind_m - 10.0) <= 5.0
            and abs(geom.cross_m) <= 5.0
            and abs(geom.below_m - 10.0) <= 5.0
        )
        envelope = EnvelopeFlags(in_inner=in_inner, in_outer=in_outer)
        sensor = SensorInfo(mode_id=1, uwb_valid=True)  # FUSED, UWB ON

        # Phase id (matches env_formation_sitl.py:_get_phase_id).
        if geom.behind_m > 30.0:
            phase_id = 1
        elif geom.behind_m > 5.0:
            phase_id = 2
        else:
            phase_id = 3

        # Targets per FSM phase (precontact targets for HOLD_PRECONTACT and
        # earlier; contact targets when controller in DOCKING_CREEP+).
        # Without controller introspection, we infer from geom: behind<5m → contact.
        if geom.behind_m < 5.0:
            target_behind = 0.0
            target_below = 0.0
        else:
            target_behind = 10.0
            target_below = 10.0

        # Milestone tier from envelope errors (matches _compute_milestone_tier).
        behind_err = geom.behind_m - target_behind
        cross_err = geom.cross_m
        below_err = geom.below_m - target_below
        tier = 0
        if hasattr(config, "MILESTONE_THRESHOLDS"):
            for t in (1, 2, 3, 4):
                b_tol, c_tol, bl_tol = config.MILESTONE_THRESHOLDS[t]
                if (abs(behind_err) <= b_tol and abs(cross_err) <= c_tol
                        and abs(below_err) <= bl_tol):
                    tier = t
                else:
                    break

        # Bearing follower→leader (degrees, 0=North, CW positive).
        dlat = lead_pos.lat / 1e7 - foll_pos.lat / 1e7
        dlon = lead_pos.lon / 1e7 - foll_pos.lon / 1e7
        bearing_to_leader = math.degrees(math.atan2(dlon, dlat)) % 360.0

        try:
            obs = self._obs_builder.build(
                geom=(self._get_obs_geom() or geom),
                follower=follower,
                leader=leader,
                phase_id=phase_id,
                envelope=envelope,
                sensor=sensor,
                dt_s=self._last_step_dt_s,
                continuous_contact_m3b_dwell_s=self._continuous_m3b_dwell_s,
                milestone_tier=tier,
                target_behind=target_behind,
                target_below=target_below,
                speed_delta_ms=0.0,
                max_closure_ms=getattr(config, "DEFAULT_MAX_CLOSURE_MS", 4.0),
                bearing_to_leader_deg=bearing_to_leader,
                wake_hazard_intensity=0.0,
            )
        except Exception:
            return np.zeros(config.OBS_DIM, dtype=np.float32)

        # T-37: FAIL LOUD, never pad. The previous
        # code zero-padded or truncated a mismatched vector to config.OBS_DIM
        # HISTORICAL: on the premise "observation_builder may return (39,) on legacy
        # modes". That premise is FALSE since D6.1 Amendment-O — the builder
        # always emits config.OBS_DIM — so the branch could only ever fire on a
        # real defect, and silently manufacturing three zero channels is
        # precisely the silent mis-index the schema hash exists to prevent.
        if obs.shape != (config.OBS_DIM,):
            raise RuntimeError(
                "[T-37] observation shape %s != expected (%d,). Refusing to "
                "pad or truncate: a mismatch here is a defect, not a legacy "
                "mode. Check config.OBS_DIM against observation_builder."
                % (obs.shape, config.OBS_DIM))
        return obs.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Residual injection
    # ------------------------------------------------------------------

    def _action_to_deltas(
        self, action: np.ndarray
    ) -> Tuple[float, float, float]:
        """Map ACTION_DIM=3 normalised action to (Δspeed_ms, Δhdg_deg, Δalt_m).

        Action layout (ACTION_DIM=3 restoration 2026-05-06):
        action[0]=speed (m/s), action[1]=heading (deg), action[2]=altitude (m).
        Heading delta is further clamped against config.CLAMP_HEADING_*_DEG
        (±3 deg per tick) inside ResidualActionProcessor.process().
        """
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scales = config.ACTION_SCALES
        speed_delta_ms = float(clipped[0]) * float(scales[0])
        hdg_delta_deg = float(clipped[1]) * float(scales[1])
        alt_delta_m = float(clipped[2]) * float(scales[2])
        return speed_delta_ms, hdg_delta_deg, alt_delta_m

    def _inject_residual(
        self,
        speed_delta_ms: float,
        heading_delta_deg: float,
        alt_delta_m: float,
    ) -> None:
        """Send residual-modified command to PORT_CONTROL_IN via MAVProxy.

        Bypassed entirely if force_zero_residual=True (controller commands
        flow through MAVProxy unchanged, matching Method C anchor exactly).

        Otherwise:
        - Read latest baseline command from MAVLink stream (DO_REPOSITION /
          DO_CHANGE_SPEED visible in follower telemetry tail)
        - Apply residual: spd_cmd_modified = spd_baseline + speed_delta_ms;
          alt_cmd_modified = alt_baseline + alt_delta_m;
          hdg_cmd_modified = hdg_baseline + heading_delta_deg
        - Rate-limit at residual_injection_interval_s (matches Method C ~2Hz)
        - Send via _control_conn:
          - DO_CHANGE_SPEED COMMAND_LONG (param speed_ms)
          - DO_REPOSITION COMMAND_INT (carrot point 1000m ahead in heading dir,
            param4=heading_deg, param.z=alt_msl)

        Phase 3 implementation: emits DO_CHANGE_SPEED only (heading/altitude
        residuals are no-op'd this dispatch since they require carrot-point
        computation matching baseline_formation_controller.py:577-651).
        """
        if self.force_zero_residual or self._control_conn is None:
            return  # Controller commands flow through unchanged

        # Read latest baseline speed from follower telemetry (VFR_HUD.airspeed).
        # If unavailable, use cruise default 24.7 m/s.
        vfr = self._telemetry_cache.get("VFR_HUD_sys1") or self._telemetry_cache.get("VFR_HUD")
        baseline_spd_ms = vfr.airspeed if vfr is not None else 24.7
        target_spd_ms = float(np.clip(
            baseline_spd_ms + speed_delta_ms,
            config.TAS_MIN_SAFE,
            config.TAS_MAX_SAFE,
        ))

        try:
            mav = self._control_conn.mav
            mav.command_long_send(
                target_system=1,           # follower (sysid=1)
                target_component=1,        # autopilot
                command=mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                confirmation=0,
                param1=0.0,                # 0=airspeed, 1=groundspeed
                param2=target_spd_ms,
                param3=-1.0,               # throttle: -1 = no change
                param4=0.0, param5=0.0, param6=0.0, param7=0.0,
            )
        except Exception:
            # Best-effort emit; failure to send is non-fatal under
            # force_zero_residual smoke regime.
            pass

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------

    # Phase 4 Option 3: per-consumer geometry source helpers. Default ENV_FUSED_SIM
    # (= self._last_geom). SIM_TRUTH/GPI sources gated via env vars. No silent
    # GPI-fallback masquerade as SIM_TRUTH (assertion under OPTION_C_ASSERT_SIMTRUTH_SOURCE=1).
    def _get_obs_geom(self):
        if os.environ.get("OPTION_C_ACTOR_OBS_FROM_GPI") == "1":
            if self._last_gpi_geom is not None:
                self._last_actor_obs_source = "GPI_DIRECT"
                return self._last_gpi_geom
            self._last_actor_obs_source = "ENV_FUSED_SIM_FALLBACK"
            return self._last_geom
        self._last_actor_obs_source = "ENV_FUSED_SIM"
        return self._last_geom

    def _get_reward_geom(self):
        if os.environ.get("OPTION_C_REWARD_FROM_SIMTRUTH") == "1":
            self._assert_simtruth_source(channel="reward")
            self._last_reward_source = ("SIM_TRUTH_NO_FUSION"
                                        if self._last_raw_geom is not None
                                        else "SIM_TRUTH_NO_FUSION_NULL")
            return self._last_raw_geom
        self._last_reward_source = "ENV_FUSED_SIM"
        return self._last_geom

    def _get_eval_geom(self):
        if os.environ.get("OPTION_C_EVAL_FROM_SIMTRUTH") == "1":
            self._assert_simtruth_source(channel="eval")
            self._last_eval_source = ("SIM_TRUTH_NO_FUSION"
                                      if self._last_raw_geom is not None
                                      else "SIM_TRUTH_NO_FUSION_NULL")
            return self._last_raw_geom
        self._last_eval_source = "ENV_FUSED_SIM"
        return self._last_geom

    def _assert_simtruth_source(self, channel: str) -> None:
        # G4 2026-05-05: accept lat_float_e7 with geofence sanity; gpi_fallback strictly invalid.
        if os.environ.get("OPTION_C_ASSERT_SIMTRUTH_SOURCE") != "1":
            return
        src = getattr(self, "_foll_pos_source", None)
        use_vel = bool(getattr(self, "_use_simstate_vel", False))
        lat = getattr(self, "_foll_lat_deg", None)
        lon = getattr(self, "_foll_lon_deg", None)
        fail, kind = None, "invalid"
        if not use_vel:
            fail = "vel_disabled"
        elif src == "sim_state.lat_int":
            kind = src
        elif src == "sim_state.lat_float_e7":
            if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)):
                fail = "non_finite"
            elif not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                fail = "lat_lon_range"
            elif abs(lat - SITL_HOME_LAT) > 1.0 or abs(lon - SITL_HOME_LON) > 1.0:
                fail = "geofence"
            else:
                kind = src
        else:
            fail = "gpi_fallback" if src == "gpi_fallback" else ("missing_source" if src in (None, "none") else "unknown_source")
        self._simtruth_assert_last = fail is None
        self._simtruth_source_kind = kind
        self._simtruth_source_degraded = fail is not None or src == "gpi_fallback"
        self._simtruth_assert_fail_reason = fail
        cnt = getattr(self, "_simtruth_kind_counts", None) or {}
        cnt[kind] = cnt.get(kind, 0) + 1
        self._simtruth_kind_counts = cnt
        if fail is not None:
            logger.error(f"[SIMTRUTH_SOURCE_ASSERT] FAIL channel={channel} foll_pos_source={src} use_simstate_vel={use_vel} reason={fail}")

    def _compute_reward(
        self, telemetry: Dict[str, Any], action: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward via reward_mrrl.MRRLRewardComputer.compute (REUSE).

        Reads geometry + episode-state accumulators (in_m3b, dwell, min_range)
        and threads them through the v3 Section 3 reward signature.
        """
        geom = self._get_reward_geom()
        if geom is None:
            return 0.0, {}

        target_behind = 0.0 if geom.behind_m < 5.0 else 10.0
        target_below = 0.0 if geom.behind_m < 5.0 else 10.0
        behind_err = geom.behind_m - target_behind
        cross_err = geom.cross_m
        below_err = geom.below_m - target_below

        # Build applied_residual (3,): [speed, heading, altitude]. In zero-
        # residual smoke, this is all zeros. In non-zero mode, post-clamp deltas.
        if self.force_zero_residual:
            applied_residual = np.zeros(3, dtype=np.float32)
        else:
            spd_d, hdg_d, alt_d = self._action_to_deltas(action)
            applied_residual = np.array([spd_d, hdg_d, alt_d], dtype=np.float32)

        # Tier from envelope errors (matches _get_obs).
        tier = 0
        if hasattr(config, "MILESTONE_THRESHOLDS"):
            for t in (1, 2, 3, 4):
                b_tol, c_tol, bl_tol = config.MILESTONE_THRESHOLDS[t]
                if (abs(behind_err) <= b_tol and abs(cross_err) <= c_tol
                        and abs(below_err) <= bl_tol):
                    tier = t
                else:
                    break

        in_m3a_coarse = tier >= 2
        in_m3a_fine = tier >= 3

        # R6 Gate B: drop R_passthrough term under force_zero_residual when gated.
        _r6_disable_pt = os.environ.get("OPTION_C_R6_DISABLE_PASSTHROUGH_ZERO_RESIDUAL") == "1"
        self._r6_passthrough_disabled = bool(_r6_disable_pt and self.force_zero_residual)
        _raw_for_reward = (None if self._r6_passthrough_disabled
                           else np.asarray(action, dtype=np.float32))

        try:
            total, components = self._reward_computer.compute(
                behind_error_m=behind_err,
                cross_error_m=cross_err,
                below_error_m=below_err,
                applied_residual=applied_residual,
                prev_applied_residual=self._prev_applied_residual,
                in_m3b=self._in_m3b,
                continuous_dwell_s=self._continuous_m3b_dwell_s,
                guard_fires=0,
                safety_violations=self._safety_violation_count,
                is_terminal=False,
                terminal_type="",
                wake_hazard_intensity=0.0,
                in_m3a_coarse=in_m3a_coarse,
                in_m3a_fine=in_m3a_fine,
                continuous_m3a_fine_dwell_s=0.0,
                boom_tip_range_m=geom.range_m,
                curriculum_stage=self.curriculum_stage,
                raw_action=_raw_for_reward,
                in_contact_m3b=self._in_m3b,
                continuous_contact_m3b_dwell_s=self._continuous_m3b_dwell_s,
                dt_s=self._last_step_dt_s,
                milestone_tier=tier,
                closure_rate_ms=geom.closure_rate_ms,
                policy_speed_delta=float(applied_residual[0]),
            )
        except Exception:
            return 0.0, {}

        self._prev_applied_residual = applied_residual
        return float(total), components

    def _is_done(
        self, telemetry: Dict[str, Any]
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """Episode termination check + safety gates + TIMEOUT.

        terminated = safety violation (SAFETY_RANGE / SAFETY_EMERG / collision)
        truncated = sim_time elapsed >= episode_sim_duration_s
        info = milestone flags + dwell stats + terminal_type
        """
        sim_time_s = telemetry.get("sim_time_s", 0.0)
        elapsed = sim_time_s - self._episode_start_sim_s
        info: Dict[str, Any] = {
            "terminal_type": "RUNNING",
            "sim_time_s": sim_time_s,
            "episode_elapsed_s": elapsed,
            "in_m3b": self._in_m3b,
            "continuous_m3b_dwell_s": self._continuous_m3b_dwell_s,
            "best_m3b_dwell_s": self._best_m3b_dwell_s,
            "m3b_entry_count": self._m3b_entry_count,
            "min_range_m": self._min_range_m if self._min_range_m != float("inf") else None,
            # Phase 4 Option 3: channel-routing labels (per-consumer source tracking).
            "actor_obs_source": getattr(self, "_last_actor_obs_source", "ENV_FUSED_SIM"),
            "reward_source": getattr(self, "_last_reward_source", "ENV_FUSED_SIM"),
            "eval_source": getattr(self, "_last_eval_source", "ENV_FUSED_SIM"),
            "simtruth_source_valid": getattr(self, "_simtruth_assert_last", None),
            "simtruth_source_kind": getattr(self, "_simtruth_source_kind", None),
            "simtruth_source_degraded": getattr(self, "_simtruth_source_degraded", None),
            "simtruth_assert_fail_reason": getattr(self, "_simtruth_assert_fail_reason", None),
            "foll_pos_source": getattr(self, "_foll_pos_source", None),
            "use_simstate_vel": getattr(self, "_use_simstate_vel", None),
        }

        # Fix A: subprocess-exit detection. If run_gated_matrix wrapper has
        # exited (controller's --max-sim-seconds reached), no new telemetry
        # will arrive; treat as truncated. Without this, _is_done loops
        # forever after wrapper exit (Phase 3 verification hung 17 min on
        # frozen geometry, hitting script wall budget).
        if self._harness_proc is not None and self._harness_proc.poll() is not None:
            info["terminal_type"] = "HARNESS_EXIT"
            info["harness_rc"] = self._harness_proc.returncode
            return False, True, info

        geom = self._last_geom
        if geom is not None:
            # SAFETY_RANGE: range exceeded stage cap.
            stage_safety_range = config.STAGE_SAFETY_RANGE.get(
                self.curriculum_stage,
                getattr(config, "HARD_RANGE_MAX_M", 800.0),
            )
            if geom.range_m > stage_safety_range:
                self._safety_violation_count += 1
                info["terminal_type"] = "SAFETY_RANGE"
                info["safety_range_m"] = stage_safety_range
                info["range_at_term_m"] = geom.range_m
                return True, False, info

        # Follower altitude / descent / TAS bounds.
        foll = telemetry.get("GLOBAL_POSITION_INT_sys1")
        vfr = telemetry.get("VFR_HUD_sys1") or telemetry.get("VFR_HUD")
        if foll is not None:
            alt_agl = foll.relative_alt / 1000.0  # mm → m
            if alt_agl < config.SAFETY_ALT_FLOOR_AGL_M:
                self._safety_violation_count += 1
                info["terminal_type"] = "SAFETY_ALT_FLOOR"
                info["alt_agl_m"] = alt_agl
                return True, False, info
            vd = foll.vz / 100.0  # cm/s → m/s, NED-D positive=down
            if vd > config.SAFETY_DESCENT_RATE_MAX_MS:
                self._safety_violation_count += 1
                info["terminal_type"] = "SAFETY_DESCENT"
                info["vd_ms"] = vd
                return True, False, info

        if vfr is not None:
            tas = vfr.airspeed
            if tas < config.TAS_MIN_SAFE or tas > config.TAS_MAX_SAFE:
                self._safety_violation_count += 1
                info["terminal_type"] = "SAFETY_TAS"
                info["tas_ms"] = tas
                return True, False, info

        # TIMEOUT (truncation).
        if elapsed >= self.episode_sim_duration_s:
            info["terminal_type"] = "TIMEOUT"
            return False, True, info

        return False, False, info

    def _update_m3b_dwell(self, geom: Optional[RelativeGeometry], dt_s: float) -> None:
        """Update boom-tip-Δ M3b STRICT dwell accumulators.

        M3b STRICT envelope is referenced to the boom-tip ORIGIN
        (`abs(geom.behind_m) ≤ 0.7m`), NOT to a target offset. Matches
        Method B's invariant at `env_formation_sitl.py:146-148`. The
        RelativeGeometry already has boom-tip as origin (geometry.py:236
        applies BOOM_STOWED_LENGTH_M / BOOM_TIP_BELOW_OFFSET_M).
        """
        if geom is None:
            return
        in_m3b_now = (
            abs(geom.behind_m) <= config.CONTACT_M3B_BEHIND_TOL
            and abs(geom.cross_m) <= config.CONTACT_M3B_CROSS_TOL
            and abs(geom.below_m) <= config.CONTACT_M3B_BELOW_TOL
        )
        if in_m3b_now:
            if not self._in_m3b:
                self._m3b_entry_count += 1
            self._continuous_m3b_dwell_s += dt_s
            self._best_m3b_dwell_s = max(
                self._best_m3b_dwell_s, self._continuous_m3b_dwell_s
            )
        else:
            self._continuous_m3b_dwell_s = 0.0
        self._in_m3b = in_m3b_now

    # ------------------------------------------------------------------
    # Dual-frame M3b reporting (2026-05-08)
    # ------------------------------------------------------------------

    def _emit_truth_summary(self) -> Optional[Path]:
        """Write run_summary_env_truth.json with truth-side M3b state.

        Lands next to the controller's run_summary.json (HARNESS_OUT_DIR captured
        from harness stdout). Falls back to OPTION_C_TRUTH_SUMMARY_DIR or
        OPTION_C_PAIRED_TICK_DIAG_DIR if HARNESS_OUT_DIR was not echoed.
        Returns the path written, or None if no usable target dir was found.
        Idempotent within an episode via self._truth_summary_emitted.
        """
        if self._truth_summary_emitted:
            return None
        target_dir: Optional[Path] = self._harness_out_dir
        if target_dir is None:
            fallback = (
                os.environ.get("OPTION_C_TRUTH_SUMMARY_DIR")
                or os.environ.get("OPTION_C_PAIRED_TICK_DIAG_DIR")
            )
            if fallback:
                target_dir = Path(fallback)
        if target_dir is None:
            return None
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        rec = {
            "schema_version": "1.0",
            "source": "src/drl/env_method_c.py:_emit_truth_summary",
            "env_id": self.env_id,
            "frame": "boom_tip_delta",
            "m3b_entries": int(self._m3b_entry_count),
            "m3b_best_dwell_s": float(self._best_m3b_dwell_s),
            "m3b_continuous_dwell_s_at_end": float(self._continuous_m3b_dwell_s),
            "in_m3b_at_end": bool(self._in_m3b),
            "min_range_m": (
                float(self._min_range_m) if self._min_range_m != float("inf") else None
            ),
            "thresholds": {
                "behind_max_m": float(config.CONTACT_M3B_BEHIND_TOL),
                "below_max_m": float(config.CONTACT_M3B_BELOW_TOL),
                "cross_max_m": float(config.CONTACT_M3B_CROSS_TOL),
                "dwell_req_s": 3.0,
            },
        }
        # If the paired-tick log exists, surface its path so the rollup tool
        # can compute disparity statistics without a separate hint env var.
        if self._paired_tick_diag_path is not None:
            try:
                rec["paired_tick_log_path"] = str(self._paired_tick_diag_path)
            except Exception:
                pass
        out_path = target_dir / "run_summary_env_truth.json"
        try:
            with open(out_path, "w") as f:
                json.dump(rec, f, indent=2)
        except OSError as exc:
            logger.warning(
                f"[env_id={self.env_id}] _emit_truth_summary write failed: {exc}"
            )
            return None
        self._truth_summary_emitted = True
        logger.info(f"[TRUTH_SUMMARY] wrote {out_path}")
        return out_path

    def _invoke_rollup(self, truth_path: Optional[Path]) -> Optional[Path]:
        """Run tools/m3b_dual_frame_rollup.py against the captured paths.

        Best-effort: any failure is logged and swallowed. Returns the rollup
        output path on success, else None.
        """
        if self._harness_out_dir is None:
            return None
        rollup_script = _PROJECT_ROOT / "tools" / "m3b_dual_frame_rollup.py"
        if not rollup_script.exists():
            return None
        controller_summary = self._harness_out_dir / "run_summary.json"
        out_path = self._harness_out_dir / "m3b_dual_frame_summary.json"
        cmd = [
            "python3", str(rollup_script),
            "--output", str(out_path),
        ]
        if controller_summary.exists():
            cmd.extend(["--controller-summary", str(controller_summary)])
        if truth_path and truth_path.exists():
            cmd.extend(["--env-truth", str(truth_path)])
        if (
            self._paired_tick_diag_path is not None
            and self._paired_tick_diag_path.exists()
        ):
            cmd.extend(["--paired-tick", str(self._paired_tick_diag_path)])
        try:
            subprocess.run(cmd, check=False, timeout=30)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(f"[env_id={self.env_id}] rollup invocation failed: {exc}")
            return None
        return out_path if out_path.exists() else None

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Terminate prior subprocess stack, launch fresh harness, return initial obs."""
        super().reset(seed=seed)

        # 2026-05-08: dual-frame M3b reporting — emit truth-side summary +
        # invoke rollup BEFORE tearing down prior harness, so artifacts land
        # next to the controller's run_summary.json before any cleanup.
        try:
            truth_path = self._emit_truth_summary()
            self._invoke_rollup(truth_path)
        except Exception as _e:
            logger.warning(f"[env_id={self.env_id}] dual-frame emit failed: {_e}")

        # Defensive teardown
        self._terminate_harness()

        # Reset dual-frame latch + harness_out_dir for the next episode.
        self._truth_summary_emitted = False
        self._harness_out_dir = None

        # G4 2026-05-05: emit per-episode SIMTRUTH source-kind summary.
        if getattr(self, "_simtruth_kind_counts", None):
            logger.info(f"[SIMTRUTH_SOURCE_KIND_EPISODE] env_id={self.env_id} counts={dict(self._simtruth_kind_counts)}")
        self._simtruth_kind_counts = {}

        # Reset internal state
        self._obs_builder.reset()
        self._tick_count = 0
        self._last_residual_send_time = 0.0
        self._last_baseline_cmd = None
        self._last_telemetry = None
        self._fsm_state = "INIT"
        self._fsm_stop_event = threading.Event()
        self._fsm_hold_precontact_seen = threading.Event()
        self._first_heartbeat_seen = threading.Event()
        self._telemetry_cache = {}
        self._episode_start_wall = time.monotonic()

        # Reset episode accumulators
        self._last_geom = None
        self._in_m3b = False
        self._continuous_m3b_dwell_s = 0.0
        self._best_m3b_dwell_s = 0.0
        self._m3b_entry_count = 0
        self._min_range_m = float("inf")
        self._prev_applied_residual = np.zeros(3, dtype=np.float32)
        self._safety_violation_count = 0
        self._sysid_msg_count = {}
        self._sim_state_warned = False  # Phase 3.6-A: one-time fallback warning

        # Phase 3.5 diagnostic: open JSONL trace if OPTION_C_GEOM_DIAG=1.
        if self._geom_diag_handle is not None:
            try:
                self._geom_diag_handle.close()
            except Exception:
                pass
            self._geom_diag_handle = None
        if os.environ.get("OPTION_C_GEOM_DIAG") == "1":
            diag_dir = Path(
                os.environ.get(
                    "OPTION_C_GEOM_DIAG_DIR",
                    "analysis/option_c_phase3_diag",
                )
            )
            diag_dir.mkdir(parents=True, exist_ok=True)
            self._geom_diag_path = diag_dir / f"geom_trace_env{self.env_id}.jsonl"
            self._geom_diag_handle = open(self._geom_diag_path, "w")

        # Phase 3.6-F timestamp-alignment diagnostic: open JSONL trace if
        # OPTION_C_TIMESTAMP_DIAG=1. Clear per-message receipt log so prior
        # episode's timestamps don't bleed into this one.
        self._msg_recv_log = {}
        if self._ts_diag_handle is not None:
            try:
                self._ts_diag_handle.close()
            except Exception:
                pass
            self._ts_diag_handle = None
        if os.environ.get("OPTION_C_TIMESTAMP_DIAG") == "1":
            ts_dir = Path(
                os.environ.get(
                    "OPTION_C_TIMESTAMP_DIAG_DIR",
                    "analysis/option_c_phase36f",
                )
            )
            ts_dir.mkdir(parents=True, exist_ok=True)
            self._ts_diag_path = ts_dir / f"timestamp_diag_env{self.env_id}.jsonl"
            self._ts_diag_handle = open(self._ts_diag_path, "w")

        # Phase 3.6-J: open paired-tick (truth-side) JSONL handle if
        # OPTION_C_PAIRED_TICK_DIAG=1. Pattern parity with above.
        if self._paired_tick_diag_handle is not None:
            try:
                self._paired_tick_diag_handle.close()
            except Exception:
                pass
            self._paired_tick_diag_handle = None
        if os.environ.get("OPTION_C_PAIRED_TICK_DIAG") == "1":
            pt_dir = Path(
                os.environ.get(
                    "OPTION_C_PAIRED_TICK_DIAG_DIR",
                    "analysis/option_c_phase36j",
                )
            )
            pt_dir.mkdir(parents=True, exist_ok=True)
            self._paired_tick_diag_path = pt_dir / f"paired_tick_log_env{self.env_id}.jsonl"
            self._paired_tick_diag_handle = open(self._paired_tick_diag_path, "w")

        # OPTION_C_FULL_FIELD_AUDIT 2026-05-05: comprehensive per-tick emitter.
        # Pattern parity with paired-tick handle above. Default-OFF.
        if self._full_field_audit_handle is not None:
            try:
                self._full_field_audit_handle.close()
            except Exception:
                pass
            self._full_field_audit_handle = None
        if os.environ.get("OPTION_C_FULL_FIELD_AUDIT") == "1":
            ff_dir = Path(
                os.environ.get(
                    "OPTION_C_FULL_FIELD_AUDIT_DIR",
                    "analysis/option_c_full_field_audit",
                )
            )
            ff_dir.mkdir(parents=True, exist_ok=True)
            self._full_field_audit_path = ff_dir / f"full_field_log_env{self.env_id}.jsonl"
            # Block-buffered (65 KB) — flushed on close() to avoid per-line
            # I/O thrash that would perturb timestamp_diag observations.
            self._full_field_audit_handle = open(
                self._full_field_audit_path, "w", buffering=65536
            )

        # Step D 2026-05-04: open Kalman-internals audit handles (Captures 4+5).
        # Pattern parity with paired-tick handle above. Default-OFF.
        if self._input_consistency_handle is not None:
            try:
                self._input_consistency_handle.close()
            except Exception:
                pass
            self._input_consistency_handle = None
        if self._leader_packet_handle is not None:
            try:
                self._leader_packet_handle.close()
            except Exception:
                pass
            self._leader_packet_handle = None
        if os.environ.get("OPTION_C_KALMAN_INTERNALS_AUDIT") == "1":
            kd_str = os.environ.get("OPTION_C_KALMAN_INTERNALS_DIR")
            if kd_str:
                kd = Path(kd_str)
                if kd.exists():
                    self._kalman_audit_dir = kd
                    self._step_d_audit_enabled = True
                    self._input_consistency_path = (
                        kd / f"input_consistency_log_env{self.env_id}.jsonl"
                    )
                    self._input_consistency_handle = open(self._input_consistency_path, "w")
                    self._leader_packet_path = (
                        kd / f"leader_packet_log_env{self.env_id}.jsonl"
                    )
                    self._leader_packet_handle = open(self._leader_packet_path, "w")
                    logger.info(
                        f"[KALMAN_AUDIT] env-side handles open: "
                        f"input_consistency={self._input_consistency_path.name}, "
                        f"leader_packet={self._leader_packet_path.name}"
                    )

        # Launch harness (subprocess + warmup wait)
        ok = self._launch_harness()
        if not ok:
            raise RuntimeError(
                f"[env_id={self.env_id}] Method C warmup timed out "
                f"({self.warmup_timeout_s}s sim budget exhausted)"
            )

        # Open MAVLink connections
        self._connect_mavlink()

        # Initial telemetry pump + obs
        telemetry = self._pump_telemetry()
        self._last_telemetry = telemetry
        self._episode_start_sim_s = telemetry.get("sim_time_s", 0.0)
        obs = self._get_obs(telemetry)

        # Step B 2026-05-04: dual fusion-reset hooks. Both env-var-gated; default OFF.
        # Env-side reset is in-process; controller-side reset is triggered via
        # file-marker polled by controller at top of _execute_tick().
        self._maybe_reset_env_fusion(telemetry.get("sim_time_s", 0.0))
        self._maybe_trigger_controller_reset()

        # R6 gate-status marker: emit once per episode for grep-discoverability in lane.log.
        _r6_b = os.environ.get("OPTION_C_R6_DISABLE_PASSTHROUGH_ZERO_RESIDUAL") == "1"
        print(f"[R6_GATE_STATUS] env_id={self.env_id} disable_passthrough_zero_residual={int(_r6_b and self.force_zero_residual)} "
              f"stage0_rebalance_active={int(getattr(config, 'R6_STAGE0_REBALANCE_ACTIVE', False))} "
              f"w_m3a_coarse={config.STAGE_WEIGHTS[0].w_m3a_coarse} w_dwell={config.STAGE_WEIGHTS[0].w_dwell}", flush=True)
        return obs, {"env_id": self.env_id, "leader_profile": self.leader_profile}

    def _maybe_reset_env_fusion(self, now_sim_s: float, force: bool = False) -> None:
        """Step B reset hook (env-side, in-process). Env-var-gated by default.

        When OPTION_C_RESET_ENV_FUSION_AT_EPISODE_START=1, snapshots the current
        env-side fusion state via get_state() and re-initializes the filter using
        those positions, with covariance reflated to default (100m² positions,
        10 (m/s)² velocities; see uwb_gps_fusion.py:688-690). Velocities reset to 0.
        Default behavior unchanged when env var unset (early-return).

        Step D 2026-05-04: when force=True (V3 periodic-reset path), bypass
        the env-var gate. JSONL record gets trigger="periodic" tag; otherwise
        trigger="episode_start".
        """
        if not force and os.environ.get("OPTION_C_RESET_ENV_FUSION_AT_EPISODE_START") != "1":
            return
        if self._geom_calc is None or self._geom_calc.fusion is None:
            return
        try:
            state_pre = self._geom_calc.fusion.get_state(now_s=now_sim_s)
            p_pre = [self._geom_calc.fusion.P[i][i] for i in range(3)]
            gps_pre = self._geom_calc.fusion.gps_update_count
            uwb_rej_pre = self._geom_calc.fusion.uwb_rejected_count
            self._geom_calc.fusion.reset(
                initial_behind=state_pre.behind_m,
                initial_cross=state_pre.cross_m,
                initial_below=state_pre.below_m,
            )
            p_post = [self._geom_calc.fusion.P[i][i] for i in range(3)]
            log_dir = os.environ.get("OPTION_C_RESET_EVENT_LOG_DIR", "/tmp")
            log_path = os.path.join(log_dir, f"reset_event_log_env{self.env_id}.jsonl")
            with open(log_path, "a") as fh:
                fh.write(json.dumps({
                    "side": "env", "env_id": self.env_id,
                    "trigger": "periodic" if force else "episode_start",
                    "sim_t": now_sim_s, "wall_t": time.time(),
                    "init_state_behind_m": state_pre.behind_m,
                    "init_state_cross_m": state_pre.cross_m,
                    "init_state_below_m": state_pre.below_m,
                    "p_pre_diag": p_pre, "p_post_diag": p_post,
                    "gps_updates_pre": gps_pre, "gps_updates_post": 0,
                    "uwb_rejected_pre": uwb_rej_pre, "uwb_rejected_post": 0,
                }) + "\n")
            _trig = "periodic" if force else "episode_start"
            print(f"[RESET_ENV trigger={_trig}] sim_t={now_sim_s:.3f} "
                  f"init_behind={state_pre.behind_m:.3f}m "
                  f"P_pre={p_pre[0]:.4f} P_post={p_post[0]:.4f}", flush=True)
            # Phase 4 I1: install fusion-trace proxy (idempotent) + emit INIT.
            if os.environ.get("OPTION_C_ENV_FUSION_TICK_LOG") == "1":
                if not isinstance(self._geom_calc.fusion, _FusionTraceProxy):
                    self._geom_calc.fusion = _FusionTraceProxy(self._geom_calc.fusion, self)
                f = self._geom_calc.fusion._inner
                self._geom_calc.fusion._emit("init", sim_t=now_sim_s, trigger=_trig,
                    x_init=list(f.x), P_diag_init=[f.P[i][i] for i in range(6)],
                    R_uwb_configured=UWB_RANGE_SIGMA_M, GPS_POS_SIGMA=GPS_POS_SIGMA_M,
                    GPS_VEL_SIGMA=GPS_VEL_SIGMA_M_S, INNOVATION_GATE_SIGMA=INNOVATION_GATE_SIGMA,
                    P_FLOOR_POS_M2=P_FLOOR_POS_M2, P_FLOOR_VEL_M2S2=P_FLOOR_VEL_M2S2)
        except Exception as e:
            print(f"[RESET_ENV] FAILED: {type(e).__name__}: {e}", flush=True)

    def _maybe_trigger_controller_reset(self, force: bool = False) -> None:
        """Step B reset hook (controller-side trigger via file-marker). Env-var-gated by default.

        When OPTION_C_RESET_CONTROLLER_FUSION_AT_EPISODE_START=1, writes a flag
        file that the controller polls at top of _execute_tick(). Sleeps briefly
        for controller to notice and apply, then polls for flag deletion as ack.
        Default behavior unchanged when env var unset.

        Step D 2026-05-04: when force=True (V3 periodic-reset path), bypass
        env-var gate AND skip the ack-wait loop (we are inside step()'s sub-tick
        loop and cannot block 0.5s every interval).
        """
        if not force and os.environ.get("OPTION_C_RESET_CONTROLLER_FUSION_AT_EPISODE_START") != "1":
            return
        flag_path = os.environ.get(
            "OPTION_C_RESET_FLAG_PATH",
            f"/tmp/method_c_reset_request_env{self.env_id}.flag",
        )
        try:
            with open(flag_path, "w") as fh:
                fh.write(json.dumps({
                    "requested_at_wall_s": time.time(),
                    "env_id": self.env_id,
                    "trigger": "periodic" if force else "episode_start",
                }) + "\n")
            if not force:
                time.sleep(0.5)
                ack = "TIMEOUT"
                for _ in range(10):
                    if not os.path.exists(flag_path):
                        ack = "YES"
                        break
                    time.sleep(0.05)
                print(f"[RESET_TRIGGER] env_id={self.env_id} flag={flag_path} "
                      f"requested=YES ack={ack}", flush=True)
            else:
                print(f"[RESET_TRIGGER trigger=periodic] env_id={self.env_id} "
                      f"flag={flag_path} requested=YES (no ack-wait)", flush=True)
        except Exception as e:
            print(f"[RESET_TRIGGER] FAILED: {type(e).__name__}: {e}", flush=True)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one DRL decision (ACTION_REPEAT sub-ticks at SIM_STEP_HZ)."""
        # ACTION_DIM=3 zero-equivalence probe gate (2026-05-06): when set,
        # overwrite the incoming action with np.zeros(ACTION_DIM) BEFORE the
        # FZR short-circuit. This exercises the real residual pipeline with
        # zero inputs (vs FZR=True which bypasses residual injection entirely).
        # Default-off when env-var unset.
        if os.environ.get("OPTION_C_FORCE_ZERO_3D_ACTION") == "1":
            action = np.zeros(config.ACTION_DIM, dtype=np.float32)
        sub_dt_s = 1.0 / SIM_STEP_HZ / max(self.speedup, 1.0)
        # Step D 2026-05-04: V3 periodic-reset hook. When OPTION_C_RESET_PERIODIC_S
        # is set (interval seconds), fire env+ctrl fusion reset every interval
        # via existing _maybe_reset_env_fusion(force=True) and
        # _maybe_trigger_controller_reset(force=True). Default-OFF when
        # _periodic_reset_interval_s is None (env var unset).
        if self._periodic_reset_interval_s is not None:
            wall_t_now = time.time()
            if wall_t_now - self._last_periodic_reset_wall_t >= self._periodic_reset_interval_s:
                now_sim = (
                    self._last_telemetry.get("sim_time_s", 0.0)
                    if self._last_telemetry is not None else 0.0
                )
                self._maybe_reset_env_fusion(now_sim, force=True)
                self._maybe_trigger_controller_reset(force=True)
                self._last_periodic_reset_wall_t = wall_t_now
        for _ in range(ACTION_REPEAT):
            self._tick_count += 1
            telemetry = self._pump_telemetry()
            self._last_telemetry = telemetry
            # Update geometry + dwell every sub-tick for accurate dwell timing.
            geom = self._build_geometry(telemetry)
            # Phase 4 Option 3: eval/milestone tracker uses _get_eval_geom() to permit
            # SIM_TRUTH-grounded milestone evaluation under OPTION_C_EVAL_FROM_SIMTRUTH=1.
            self._update_m3b_dwell(self._get_eval_geom() or geom, sub_dt_s)
            # Phase 3.6-J: emit paired-tick truth-side JSONL line if enabled.
            # Default-off (handle is None unless OPTION_C_PAIRED_TICK_DIAG=1
            # at reset()). For post-hoc correlation with controller's
            # [SPD_ARB] perceived-side log emissions by sim_time_s.
            if self._paired_tick_diag_handle is not None and geom is not None:
                try:
                    raw = self._last_raw_geom
                    gpi = self._last_gpi_geom
                    pt_rec = {
                        "tick_idx": self._tick_count,
                        "utc_ms": int(time.time() * 1000),
                        "sim_time_s": telemetry.get("sim_time_s"),
                        # Backward-compat keys (semantically env-fused; PATH F+ audit
                        # 2026-05-03 confirmed these were never raw truth).
                        "truth_behind_m": geom.behind_m,
                        "truth_cross_m": geom.cross_m,
                        "truth_below_m": geom.below_m,
                        "truth_range_m": geom.range_m,
                        "in_m3b_truth": self._in_m3b,
                        # PATH F+ tracker-channel separation
                        "env_fused_behind_m": geom.behind_m,
                        "env_fused_cross_m": geom.cross_m,
                        "env_fused_below_m": geom.below_m,
                        "env_fused_range_m": geom.range_m,
                        "raw_truth_behind_m": raw.behind_m if raw else None,
                        "raw_truth_cross_m": raw.cross_m if raw else None,
                        "raw_truth_below_m": raw.below_m if raw else None,
                        "raw_truth_range_m": raw.range_m if raw else None,
                        # Probe A 4th channel: GPI_SYS1_DIRECT (follower from
                        # GLOBAL_POSITION_INT_sys1 = ArduPilot EKF state, no Kalman)
                        "gpi_direct_behind_m": gpi.behind_m if gpi else None,
                        "gpi_direct_cross_m": gpi.cross_m if gpi else None,
                        "gpi_direct_below_m": gpi.below_m if gpi else None,
                        "gpi_direct_range_m": gpi.range_m if gpi else None,
                    }
                    # Step C 2026-05-04: perception integrity audit keys
                    # (velocity / attitude / angular-rate truth-vs-perceived).
                    # Default-OFF (audit gate unset → keys absent).
                    if os.environ.get("OPTION_C_PERCEPTION_INTEGRITY_AUDIT", "1") != "0":
                        sim_truth = self._last_sim_truth_audit or {}
                        foll_att = telemetry.get("ATTITUDE_sys1") or telemetry.get("ATTITUDE")
                        foll_pos = telemetry.get("GLOBAL_POSITION_INT_sys1")
                        lead_pos = telemetry.get("GLOBAL_POSITION_INT_sys2")
                        lead_hdg_rad = math.radians(lead_pos.hdg / 100.0) if lead_pos else 0.0
                        cos_h, sin_h = math.cos(lead_hdg_rad), math.sin(lead_hdg_rad)
                        # Velocity truth (SIM_STATE NED) vs GPI EKF (cm/s ÷100 = m/s NED)
                        truth_vn = sim_truth.get("vn", 0.0)
                        truth_ve = sim_truth.get("ve", 0.0)
                        truth_vd = sim_truth.get("vd", 0.0)
                        gpi_vx = (foll_pos.vx / 100.0) if foll_pos else 0.0
                        gpi_vy = (foll_pos.vy / 100.0) if foll_pos else 0.0
                        gpi_vz = (foll_pos.vz / 100.0) if foll_pos else 0.0
                        dvn = truth_vn - gpi_vx
                        dve = truth_ve - gpi_vy
                        dvd = truth_vd - gpi_vz
                        # Project velocity-error into leader-body frame (same
                        # rotation as tests/geometry.py:797-799 for relative-vel)
                        dv_behind = -(dvn * cos_h + dve * sin_h)
                        dv_cross = -dvn * sin_h + dve * cos_h
                        dv_below = dvd
                        # Attitude truth (SIM_STATE rad) vs ATTITUDE_sys1 (rad)
                        att_roll = float(foll_att.roll) if foll_att else 0.0
                        att_pitch = float(foll_att.pitch) if foll_att else 0.0
                        att_yaw = float(foll_att.yaw) if foll_att else 0.0
                        truth_roll = sim_truth.get("roll", 0.0)
                        truth_pitch = sim_truth.get("pitch", 0.0)
                        truth_yaw = sim_truth.get("yaw", 0.0)
                        # Angular rates (rad/s)
                        truth_xgyro = sim_truth.get("xgyro", 0.0)
                        truth_ygyro = sim_truth.get("ygyro", 0.0)
                        truth_zgyro = sim_truth.get("zgyro", 0.0)
                        att_rollspeed = float(getattr(foll_att, "rollspeed", 0.0)) if foll_att else 0.0
                        att_pitchspeed = float(getattr(foll_att, "pitchspeed", 0.0)) if foll_att else 0.0
                        att_yawspeed = float(getattr(foll_att, "yawspeed", 0.0)) if foll_att else 0.0
                        # Env-side fusion velocity state (truth-vs-fused-vel comparison)
                        env_fusion_vx = env_fusion_vy = env_fusion_vz = None
                        if self._geom_calc and self._geom_calc.fusion:
                            env_fusion_vx = self._geom_calc.fusion.x[3]
                            env_fusion_vy = self._geom_calc.fusion.x[4]
                            env_fusion_vz = self._geom_calc.fusion.x[5]
                        pt_rec.update({
                            "leader_heading_rad": lead_hdg_rad,
                            "fsm_state": str(self._fsm_state),
                            "truth_vn_ms": truth_vn, "truth_ve_ms": truth_ve, "truth_vd_ms": truth_vd,
                            "gpi_vx_ms": gpi_vx, "gpi_vy_ms": gpi_vy, "gpi_vz_ms": gpi_vz,
                            "delta_vel_n_ms": dvn, "delta_vel_e_ms": dve, "delta_vel_d_ms": dvd,
                            "delta_vel_behind_ms": dv_behind, "delta_vel_cross_ms": dv_cross, "delta_vel_below_ms": dv_below,
                            "truth_roll_rad": truth_roll, "truth_pitch_rad": truth_pitch, "truth_yaw_rad": truth_yaw,
                            "att_roll_rad": att_roll, "att_pitch_rad": att_pitch, "att_yaw_rad": att_yaw,
                            "delta_yaw_rad": truth_yaw - att_yaw,
                            "truth_xgyro_rs": truth_xgyro, "truth_ygyro_rs": truth_ygyro, "truth_zgyro_rs": truth_zgyro,
                            "att_rollspeed_rs": att_rollspeed, "att_pitchspeed_rs": att_pitchspeed, "att_yawspeed_rs": att_yawspeed,
                            "delta_gyro_x": truth_xgyro - att_rollspeed,
                            "delta_gyro_y": truth_ygyro - att_pitchspeed,
                            "delta_gyro_z": truth_zgyro - att_yawspeed,
                            "env_fusion_vx_state_ms": env_fusion_vx,
                            "env_fusion_vy_state_ms": env_fusion_vy,
                            "env_fusion_vz_state_ms": env_fusion_vz,
                        })
                    self._paired_tick_diag_handle.write(json.dumps(pt_rec) + "\n")
                except Exception:
                    pass

            # OPTION_C_FULL_FIELD_AUDIT 2026-05-05: comprehensive per-tick
            # perception-vs-truth emitter. Default-OFF; handle is None unless
            # OPTION_C_FULL_FIELD_AUDIT=1 was set at reset(). Writes a single
            # JSONL line per tick capturing the entire telemetry cache, all
            # SIM_STATE/GPI/ATTITUDE/VFR_HUD fields per sysid, geometry NED-raw
            # inputs and post-projection outputs, and full fusion internals
            # (P matrix, x state, K, innovation, anchor counters). Block-
            # buffered to avoid I/O thrash that would perturb timestamp-skew
            # measurements when this gate is run alongside others.
            if self._full_field_audit_handle is not None and geom is not None:
                try:
                    raw_g = self._last_raw_geom
                    gpi_g = self._last_gpi_geom
                    fus = self._geom_calc.fusion if self._geom_calc else None
                    sim_truth = self._last_sim_truth_audit or {}
                    foll_pos = telemetry.get("GLOBAL_POSITION_INT_sys1")
                    lead_pos = telemetry.get("GLOBAL_POSITION_INT_sys2")
                    foll_att = telemetry.get("ATTITUDE_sys1") or telemetry.get("ATTITUDE")
                    lead_att = telemetry.get("ATTITUDE_sys2")
                    foll_vfr = telemetry.get("VFR_HUD_sys1") or telemetry.get("VFR_HUD")
                    lead_vfr = telemetry.get("VFR_HUD_sys2")
                    sim_state_msg = (
                        telemetry.get("SIM_STATE_sys1")
                        or telemetry.get("SIM_STATE_sys0")
                        or telemetry.get("SIM_STATE")
                    )
                    full_rec = {
                        "tick_idx": self._tick_count,
                        "utc_ms": int(time.time() * 1000),
                        "wall_t": time.time(),
                        "sim_time_s": telemetry.get("sim_time_s"),
                        "env_id": self.env_id,
                        "fsm_state": str(self._fsm_state),
                        # Comprehensive cache snapshot — every cached MAVLink message,
                        # every standard scalar field. JSON-safe via helper.
                        "mav_cache_snapshot": _serialize_telemetry_cache(
                            self._telemetry_cache
                        ),
                        # Truth side (SIM_STATE) — flat fields for quick-grep.
                        "sim_truth": {
                            "lat_int": getattr(sim_state_msg, "lat_int", None) if sim_state_msg else None,
                            "lon_int": getattr(sim_state_msg, "lon_int", None) if sim_state_msg else None,
                            "lat_float_e7_fallback": getattr(sim_state_msg, "lat", None) if sim_state_msg else None,
                            "lon_float_e7_fallback": getattr(sim_state_msg, "lon", None) if sim_state_msg else None,
                            "alt": getattr(sim_state_msg, "alt", None) if sim_state_msg else None,
                            "vn": sim_truth.get("vn"),
                            "ve": sim_truth.get("ve"),
                            "vd": sim_truth.get("vd"),
                            "roll": sim_truth.get("roll"),
                            "pitch": sim_truth.get("pitch"),
                            "yaw": sim_truth.get("yaw"),
                            "xgyro": sim_truth.get("xgyro"),
                            "ygyro": sim_truth.get("ygyro"),
                            "zgyro": sim_truth.get("zgyro"),
                        },
                        # Perception transport stage (sys1/sys2)
                        "gpi_sys1": _serialize_mavlink_msg(foll_pos),
                        "gpi_sys2": _serialize_mavlink_msg(lead_pos),
                        "attitude_sys1": _serialize_mavlink_msg(foll_att),
                        "attitude_sys2": _serialize_mavlink_msg(lead_att),
                        "vfr_hud_sys1": _serialize_mavlink_msg(foll_vfr),
                        "vfr_hud_sys2": _serialize_mavlink_msg(lead_vfr),
                        # Geometry stages (raw NED inputs implicitly captured via
                        # gpi_sys1/sys2 above; analyzer reconstructs leader-body-
                        # rotated intermediate from those + lead_hdg).
                        "geom_stages": {
                            "boom_length_m": BOOM_HORIZONTAL_PROJECTION_M,
                            "boom_tip_below_offset_m": BOOM_TIP_BELOW_OFFSET_M,
                            "lead_hdg_deg": (lead_pos.hdg / 100.0) if lead_pos else None,
                            "boom_tip_projected": {
                                "behind_m": geom.behind_m,
                                "cross_m": geom.cross_m,
                                "below_m": geom.below_m,
                                "range_m": geom.range_m,
                            },
                            "raw_truth": {
                                "behind_m": raw_g.behind_m if raw_g else None,
                                "cross_m": raw_g.cross_m if raw_g else None,
                                "below_m": raw_g.below_m if raw_g else None,
                                "range_m": raw_g.range_m if raw_g else None,
                            },
                            "gpi_direct": {
                                "behind_m": gpi_g.behind_m if gpi_g else None,
                                "cross_m": gpi_g.cross_m if gpi_g else None,
                                "below_m": gpi_g.below_m if gpi_g else None,
                                "range_m": gpi_g.range_m if gpi_g else None,
                            },
                        },
                        # Fusion internals — full P, x, K, innovation, counters.
                        "fusion_internals": {
                            "x": list(fus.x) if fus is not None else None,
                            "P_diag": (
                                [fus.P[i][i] for i in range(6)]
                                if fus is not None else None
                            ),
                            "P_full_row_major": (
                                [v for row in fus.P for v in row]
                                if fus is not None else None
                            ),
                            "gps_update_count": (
                                fus.gps_update_count if fus is not None else None
                            ),
                            "uwb_update_count": (
                                fus.uwb_update_count if fus is not None else None
                            ),
                            "uwb_rejected_count": (
                                fus.uwb_rejected_count if fus is not None else None
                            ),
                            "last_innovation_range": (
                                fus.last_innovation_range if fus is not None else None
                            ),
                            "last_predict_time": (
                                fus.last_predict_time if fus is not None else None
                            ),
                            "last_gps_time": (
                                fus.last_gps_time if fus is not None else None
                            ),
                            "last_uwb_time": (
                                fus.last_uwb_time if fus is not None else None
                            ),
                            "uwb_updates_by_anchor": (
                                dict(fus._uwb_updates_by_anchor) if fus is not None else None
                            ),
                            "uwb_rejected_by_anchor": (
                                dict(fus._uwb_rejected_by_anchor) if fus is not None else None
                            ),
                            "last_innovation_by_anchor": (
                                dict(fus._last_innovation_by_anchor) if fus is not None else None
                            ),
                        },
                    }
                    self._full_field_audit_handle.write(json.dumps(full_rec) + "\n")
                except Exception:
                    pass

            if not self.force_zero_residual:
                deltas = self._action_to_deltas(action)
                # Rate-limit injection
                now = telemetry.get("sim_time_s", time.monotonic())
                if (now - self._last_residual_send_time
                        >= self.residual_injection_interval_s):
                    self._inject_residual(*deltas)
                    self._last_residual_send_time = now
            time.sleep(sub_dt_s)
        self._last_step_dt_s = sub_dt_s * ACTION_REPEAT

        obs = self._get_obs(telemetry)
        reward, _components = self._compute_reward(telemetry, action)
        terminated, truncated, info = self._is_done(telemetry)
        info["env_id"] = self.env_id
        info["tick_count"] = self._tick_count
        # R6 telemetry: per-step gate state (cheap booleans/floats; default-off invariant preserved).
        info["r6_passthrough_disabled_zero_residual"] = bool(getattr(self, "_r6_passthrough_disabled", False))
        info["r6_stage0_rebalance_active"] = bool(getattr(config, "R6_STAGE0_REBALANCE_ACTIVE", False))
        info["r6_w_m3a_coarse"] = float(config.STAGE_WEIGHTS[0].w_m3a_coarse)
        info["r6_w_dwell"] = float(config.STAGE_WEIGHTS[0].w_dwell)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Terminate subprocess stack and clean up."""
        # 2026-05-08: dual-frame M3b reporting — emit truth-side summary +
        # invoke rollup BEFORE teardown so artifacts persist past close().
        try:
            truth_path = self._emit_truth_summary()
            self._invoke_rollup(truth_path)
        except Exception as _e:
            logger.warning(f"[env_id={self.env_id}] dual-frame emit failed: {_e}")
        self._terminate_harness()
        if self._geom_diag_handle is not None:
            try:
                self._geom_diag_handle.flush()
                self._geom_diag_handle.close()
            except Exception:
                pass
            self._geom_diag_handle = None
        # Phase 3.6-F: flush + close timestamp-alignment JSONL handle.
        if self._ts_diag_handle is not None:
            try:
                self._ts_diag_handle.flush()
                self._ts_diag_handle.close()
            except Exception:
                pass
            self._ts_diag_handle = None
        # Phase 3.6-J: flush + close paired-tick (truth-side) JSONL handle.
        if self._paired_tick_diag_handle is not None:
            try:
                self._paired_tick_diag_handle.flush()
                self._paired_tick_diag_handle.close()
            except Exception:
                pass
            self._paired_tick_diag_handle = None
        # Step D 2026-05-04: flush + close Kalman-internals audit handles.
        if self._input_consistency_handle is not None:
            try:
                self._input_consistency_handle.flush()
                self._input_consistency_handle.close()
            except Exception:
                pass
            self._input_consistency_handle = None
        if self._leader_packet_handle is not None:
            try:
                self._leader_packet_handle.flush()
                self._leader_packet_handle.close()
            except Exception:
                pass
            self._leader_packet_handle = None


# ============================================================================
# Module-level smoke test (parse + import check; no SITL launch)
# ============================================================================

if __name__ == "__main__":
    # Parse-only: instantiate without launching harness.
    # All subprocess methods are NotImplementedError, so reset() will raise.
    # This block exists only to verify the class can be constructed and the
    # observation/action spaces match expectations.
    env = MethodCFormationEnv(
        env_id=0,
        leader_profile="straight",
        speedup=1.0,
        force_zero_residual=True,
    )
    print(f"observation_space = {env.observation_space}")
    print(f"action_space      = {env.action_space}")
    print(f"port_control_in   = {env._port_control_in}")
    print(f"port_follower     = {env._port_follower}")
    print(f"port_leader       = {env._port_leader}")
    print(f"sitl_work_dir     = {env.sitl_work_dir}")
    print("SCAFFOLD_PARSE_OK — reset() / step() will raise NotImplementedError")
    env.close()
