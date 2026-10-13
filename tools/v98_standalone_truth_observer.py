#!/usr/bin/env python3
"""v98 standalone truth observer — sidecar for non-DRL Method C harness runs.

Subscribes to MAVProxy's UI fanout port (14554 + HARNESS_PORT_OFFSET), drains
MAVLink, computes boom-tip-frame RAW-TRUTH geometry per tick from SIM_STATE
(Phase 3.6-A Fix C1: lat_int / 1e7) + leader broadcast, and emits
`paired_tick_log_env0.jsonl` + `run_summary_env_truth.json` into HARNESS_OUT_DIR.

The standalone harness's controller-perceived view is already in
`speed_arbitration.csv` (20 Hz). The dual-frame rollup tool
(`tools/m3b_dual_frame_rollup.py`) joins these two artifacts by `sim_time_s`
to produce the side-by-side perceived/truth M3b summary.

Authored 2026-05-08 to extend the dual-frame infrastructure (env_method_c-only
through May 8 morning) to standalone Method C runs without env_method_c wrap.
R7.1 / R7.2 / R7.3 untouched — observer is purely additive.

Configuration (env vars):
  HARNESS_OUT_DIR Required. Output directory.
  HARNESS_ENV_ID Default 0. Port offset = ENV_ID * 100.
  HARNESS_UWB_ENABLE Informational only (observer doesn't consume UWB).
  OBSERVER_TICK_HZ Default 10.0. Emission rate.
  OBSERVER_LOG_LEVEL Default INFO.

Shutdown: SIGTERM (or SIGINT) → flush + close JSONL handles, write summary.
"""
from __future__ import annotations

import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Make tests/ + src/ importable for shared geometry + config + fusion constants.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT / "tests", _PROJECT_ROOT / "src", _PROJECT_ROOT / "tools", _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from pymavlink import mavutil  # type: ignore[import-not-found]
except ImportError:
    print("[truth_observer] pymavlink not available; aborting.", file=sys.stderr)
    sys.exit(2)

from geometry import (  # type: ignore[import-not-found]
    BOOM_TIP_BELOW_OFFSET_M,
    GeometryCalculator,
)
# Truth-logger STRUCTURAL (2026-06-09): mirror the canonical milestone envelope
# definitions from the perceived-side tracker. IMPORT-ONLY (read, never write) —
# formation_success_metrics.py is the single source of truth for thresholds.
from formation_success_metrics import MilestoneThresholds  # type: ignore[import-not-found]

# CSV2_R7K_FRAME_VERIFICATION_LOG: additive raw-GPS frame-quartet logging hook
# (non-frozen, additive only; does not touch the truth path).
from diagnostic_frame_log import frame_quartet  # type: ignore[import-not-found]

# Constants — mirror env_method_c.py for cross-pipeline statistical comparability.
BASE_UI_PORT = 14554
PORT_STRIDE = 100
SITL_HOME_LAT = -35.363261
SITL_HOME_LON = 149.165230
SITL_HOME_ALT_MSL = 584.0
# CSV2 frame-fix (2026-06-10): import the canonical horizontal projection from
# formation_config (single source of truth shared with the controller via import *),
# replacing the prior local literal. Same value (4.31), now the identical object.
from formation_config import BOOM_HORIZONTAL_PROJECTION_M  # type: ignore[import-not-found] # noqa: E402

# --- SIM_STATE(108) self-request (truth-logger STRUCTURAL, 2026-06-09) ---------
# The standalone harness only streams SIMSTATE(164) by default and requests
# SIM_STATE(108) solely under the controller's _FD_UWB_TRUTH_SOURCED gate. The
# observer requests SIM_STATE(108) @ 10 Hz itself so truth never depends on that
# gate. The follower SITL MAVLink link is ALWAYS sysid=1 on the observer's 14554
# fanout port → target_system / target_component are HARDCODED to 1 (no HEARTBEAT
# discovery). Constants are factored out so the unit test can assert all 5 fields.
SET_MESSAGE_INTERVAL_CMD = 511   # MAV_CMD_SET_MESSAGE_INTERVAL
SIM_STATE_MSG_ID = 108           # SIM_STATE
SIM_STATE_INTERVAL_US = 100000   # microseconds → 10 Hz
SIM_STATE_TARGET_SYSTEM = 1      # follower autopilot (always sysid=1 on 14554)
SIM_STATE_TARGET_COMPONENT = 1   # flight controller


def sim_state_request_params() -> Dict[str, int]:
    """Return the 5 load-bearing COMMAND_LONG fields for the SIM_STATE(108)
    self-request, as a plain dict the unit test can assert on without a live
    MAVLink connection. command=511, param1=108, param2=100000(µs=10Hz),
    target_system=1, target_component=1."""
    return {
        "command": SET_MESSAGE_INTERVAL_CMD,
        "param1": SIM_STATE_MSG_ID,
        "param2": SIM_STATE_INTERVAL_US,
        "target_system": SIM_STATE_TARGET_SYSTEM,
        "target_component": SIM_STATE_TARGET_COMPONENT,
    }


# --- Milestone envelope predicates --------------------------------------------
# Mirror tests/formation_success_metrics.py EXACTLY (the code, not the dataclass
# comments — the code uses <= on every axis; comments saying "< tol" are stale).
# Args are (behind, cross, below) in the boom-tip RELATIVE frame, matching the
# controller's (behind_tip_m, cross_tip_m, below_tip_m) ordering. M3a/M3b deltas
# ARE the boom-tip-frame coordinates (delta_X = X_tip), so |X| <= tol.
_MT = MilestoneThresholds()


def m1_in_envelope(behind: float, cross: float, below: float, t: MilestoneThresholds = _MT) -> bool:
    return (
        t.m1_behind_min <= behind <= t.m1_behind_max
        and abs(cross) <= t.m1_cross_tol
        and abs(below) <= t.m1_below_tol
    )


def m2_in_envelope(behind: float, cross: float, below: float, t: MilestoneThresholds = _MT) -> bool:
    return (
        (t.m2_behind_target - t.m2_behind_tol) <= behind <= (t.m2_behind_target + t.m2_behind_tol)
        and abs(cross) <= t.m2_cross_tol
        and (t.m2_below_target - t.m2_below_tol) <= below <= (t.m2_below_target + t.m2_below_tol)
    )


def m3a_coarse_in_envelope(behind: float, cross: float, below: float, t: MilestoneThresholds = _MT) -> bool:
    return (
        abs(behind) <= t.m3a_coarse_delta_behind_tol
        and abs(below) <= t.m3a_coarse_delta_below_tol
        and abs(cross) <= t.m3a_coarse_delta_cross_tol
    )


def m3a_fine_in_envelope(behind: float, cross: float, below: float, t: MilestoneThresholds = _MT) -> bool:
    return (
        abs(behind) <= t.m3a_delta_behind_tol
        and abs(below) <= t.m3a_delta_below_tol
        and abs(cross) <= t.m3a_delta_cross_tol
    )


def m3b_in_envelope(behind: float, cross: float, below: float, t: MilestoneThresholds = _MT) -> bool:
    return (
        abs(behind) <= t.m3b_delta_behind_tol
        and abs(below) <= t.m3b_delta_below_tol
        and abs(cross) <= t.m3b_delta_cross_tol
    )


# Ladder order: (name, predicate, dwell_req_s, entry_only). M1 is entry-only
# (no dwell — achieved on first entry, matching FormationSuccessMetrics).
MILESTONE_LADDER = (
    ("m1", m1_in_envelope, 0.0, True),
    ("m2", m2_in_envelope, _MT.m2_dwell_s, False),
    ("m3a_coarse", m3a_coarse_in_envelope, _MT.m3a_coarse_dwell_s, False),
    ("m3a_fine", m3a_fine_in_envelope, _MT.m3a_dwell_s, False),
    ("m3b", m3b_in_envelope, _MT.m3b_dwell_s, False),
)


class _DwellTracker:
    """Continuous-dwell / best-dwell / entry-count for one truth-frame milestone.

    entry_only milestones (M1) mark achieved on first entry (no dwell requirement);
    dwell milestones mark achieved once best continuous dwell >= dwell_req_s.
    Pure-geometric: NOT gated by controller state (the observer measures truth
    reality independent of the controller's perception/state machine).
    """

    __slots__ = (
        "name", "dwell_req_s", "entry_only", "in_env",
        "continuous_s", "best_s", "entries", "achieved", "first_entry_tick",
    )

    def __init__(self, name: str, dwell_req_s: float, entry_only: bool = False) -> None:
        self.name = name
        self.dwell_req_s = float(dwell_req_s)
        self.entry_only = bool(entry_only)
        self.in_env = False
        self.continuous_s = 0.0
        self.best_s = 0.0
        self.entries = 0
        self.achieved = False
        self.first_entry_tick: Optional[int] = None

    def update(self, in_env: bool, dt_s: float, tick_idx: int) -> bool:
        if in_env:
            if not self.in_env:
                self.entries += 1
                if self.first_entry_tick is None:
                    self.first_entry_tick = tick_idx
            self.continuous_s += dt_s
            if self.continuous_s > self.best_s:
                self.best_s = self.continuous_s
            if self.entry_only:
                self.achieved = True
            elif self.best_s >= self.dwell_req_s:
                self.achieved = True
        else:
            self.continuous_s = 0.0
        self.in_env = in_env
        return in_env

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": int(self.entries),
            "best_dwell_s": float(self.best_s),
            "continuous_dwell_s_at_end": float(self.continuous_s),
            "in_env_at_end": bool(self.in_env),
            "achieved": bool(self.achieved),
            "dwell_req_s": float(self.dwell_req_s),
            "first_entry_tick": self.first_entry_tick,
        }


logger = logging.getLogger("truth_observer")


def _setup_logging() -> None:
    level_name = os.environ.get("OBSERVER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


def _get_sim_state(cache: Dict[str, Any]):
    """Mirror env_method_c._get_sim_state — sysid may be 0/1/None."""
    return (
        cache.get("SIM_STATE_sys1")
        or cache.get("SIM_STATE_sys0")
        or cache.get("SIM_STATE")
    )


class TruthObserver:
    """Listens to MAVProxy UI fanout, emits raw-truth boom-tip geometry."""

    def __init__(self) -> None:
        self.out_dir = Path(os.environ["HARNESS_OUT_DIR"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env_id = int(os.environ.get("HARNESS_ENV_ID", "0"))
        port_offset = self.env_id * PORT_STRIDE
        self.listen_port = BASE_UI_PORT + port_offset
        self.tick_hz = float(os.environ.get("OBSERVER_TICK_HZ", "10.0"))
        self.tick_dt = 1.0 / self.tick_hz
        self.uwb_enabled = os.environ.get("HARNESS_UWB_ENABLE", "0") == "1"

        self.geom_calc = GeometryCalculator(
            home_lat=SITL_HOME_LAT,
            home_lon=SITL_HOME_LON,
            home_alt_msl=SITL_HOME_ALT_MSL,
        )
        # Disable Kalman fusion — raw-truth path only.
        self.geom_calc.fusion = None

        self.conn = None
        self.cache: Dict[str, Any] = {}
        self.tick_idx = 0
        self.sim_state_warned = False

        # Full truth-frame milestone ladder (M1/M2/M3a-coarse/M3a-fine/M3b),
        # each with continuous + best dwell. Mirrors the perceived-side ladder.
        self._trackers: Dict[str, _DwellTracker] = {
            name: _DwellTracker(name, dwell_req_s, entry_only)
            for (name, _pred, dwell_req_s, entry_only) in MILESTONE_LADDER
        }
        # Count ticks where truth fell back from SIM_STATE(108) to GPI GPS.
        # Healthy run (108 streaming) => 0. Surfaced in run_summary_env_truth.json.
        self._sim_state_fallback_ticks = 0
        self._min_range_m = float("inf")
        self._min_behind_abs_m = float("inf")
        self._min_cross_abs_m = float("inf")
        self._min_below_abs_m = float("inf")

        # File handles.
        self.paired_log_path = self.out_dir / "paired_tick_log_env0.jsonl"
        self.paired_log_handle = open(self.paired_log_path, "w", buffering=65536)

        self._stop = False

    def shutdown(self, signum: int = 0, frame=None) -> None:
        """SIGTERM/SIGINT handler (registered in main()). Sets the shutdown flag;
        the run() loop exits between ticks and calls _finalize() which flushes +
        closes the paired-tick JSONL and writes the summary. This is what makes
        the harness reap's SIGTERM -> 3 s wait produce complete, un-truncated
        artifacts (no separate atexit needed)."""
        if self._stop:
            return
        self._stop = True
        m3b = self._trackers["m3b"]
        logger.info(
            f"shutdown signal={signum} ticks={self.tick_idx} "
            f"best_m3b_dwell_s={m3b.best_s:.3f} m3b_entries={m3b.entries} "
            f"sim_state_fallback_ticks={self._sim_state_fallback_ticks}"
        )

    def _open_connection(self) -> None:
        listen_uri = f"udpin:0.0.0.0:{self.listen_port}"
        logger.info(f"opening MAVLink listen socket: {listen_uri}")
        self.conn = mavutil.mavlink_connection(
            listen_uri,
            source_system=254,
            source_component=190,
            autoreconnect=False,
            input=True,
            dialect="ardupilotmega",
        )

    def _drain(self) -> None:
        """Drain pending MAVLink messages into self.cache."""
        if self.conn is None:
            return
        for _ in range(4096):
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue
            hdr = getattr(msg, "_header", None)
            sysid = hdr.srcSystem if hdr is not None else 0
            self.cache[f"{mtype}_sys{sysid}"] = msg
            self.cache[mtype] = msg

    def _build_raw_geometry(self):
        """Compute boom-tip-frame raw-truth RelativeGeometry from cache."""
        lead = self.cache.get("GLOBAL_POSITION_INT_sys2")
        foll = self.cache.get("GLOBAL_POSITION_INT_sys1")
        if lead is None or foll is None:
            return None

        sim = _get_sim_state(self.cache)
        foll_lat: Optional[float] = None
        foll_lon: Optional[float] = None
        foll_alt: Optional[float] = None
        if sim is not None:
            lat_int = getattr(sim, "lat_int", None)
            lon_int = getattr(sim, "lon_int", None)
            if lat_int is not None and lon_int is not None:
                foll_lat = lat_int / 1e7
                foll_lon = lon_int / 1e7
            else:
                foll_lat = float(sim.lat) / 1e7
                foll_lon = float(sim.lon) / 1e7
            foll_alt = float(sim.alt)
        if foll_lat is None or foll_lon is None or foll_alt is None:
            self._sim_state_fallback_ticks += 1
            if not self.sim_state_warned:
                logger.warning(
                    "SIM_STATE not received; falling back to GLOBAL_POSITION_INT_sys1 "
                    "GPS (~6m noise vs truth). Verify MAVProxy --out=14554 fanout "
                    "and that the SIM_STATE(108) self-request reached the autopilot."
                )
                self.sim_state_warned = True
            foll_lat = foll.lat / 1e7
            foll_lon = foll.lon / 1e7
            foll_alt = foll.alt / 1000.0

        # GPI velocities (cm/s → m/s); SIM_STATE has none.
        foll_vn = foll.vx / 100.0
        foll_ve = foll.vy / 100.0
        foll_vd = foll.vz / 100.0

        lead_lat = lead.lat / 1e7
        lead_lon = lead.lon / 1e7
        lead_alt = lead.alt / 1000.0
        lead_vn = lead.vx / 100.0
        lead_ve = lead.vy / 100.0
        lead_vd = lead.vz / 100.0
        lead_hdg = lead.hdg / 100.0

        # STEP-4 Hunk-O (2026-06-11, gated OBS_LEADER_EXTRAP default OFF):
        # leader-hop C3 alignment for the truth instrument. CLOCK SOURCE
        # : age = follower GPI time_boot_ms - leader GPI
        # time_boot_ms — BOTH AP boot clocks on the shared sim epoch (step-3
        # verified) — aligning the leader to the follower-sample instant.
        # Never time.time()/time.monotonic() for age. The helper itself is
        # additionally gated by LEADER_EXTRAP_ENABLE (default OFF inside).
        if os.environ.get("OBS_LEADER_EXTRAP", "0").strip() == "1":
            from leader_extrapolator import extrapolate_leader_latlon
            _age_ms = float(getattr(foll, "time_boot_ms", 0)) - float(
                getattr(lead, "time_boot_ms", 0))
            lead_lat, lead_lon = extrapolate_leader_latlon(
                lead_lat, lead_lon, lead_vn, lead_ve, _age_ms)

        try:
            return self.geom_calc.compute_relative(
                foll_lat=foll_lat, foll_lon=foll_lon, foll_alt=foll_alt,
                foll_vn=foll_vn, foll_ve=foll_ve, foll_vd=foll_vd,
                lead_lat=lead_lat, lead_lon=lead_lon, lead_alt=lead_alt,
                lead_vn=lead_vn, lead_ve=lead_ve, lead_vd=lead_vd,
                lead_heading_deg=lead_hdg,
                boom_length_m=BOOM_HORIZONTAL_PROJECTION_M,
                timestamp_s=time.monotonic(),
                boom_tip_offset_m=BOOM_TIP_BELOW_OFFSET_M,
            )
        except Exception as exc:
            logger.warning(f"compute_relative raised: {exc}")
            return None

    def _update_ladder(self, geom, dt_s: float) -> Dict[str, bool]:
        """Update every truth-frame milestone tracker (M1..M3b) for this tick.

        Returns {milestone_name: in_env_bool}. When geom is None all memberships
        are False (no truth this tick). Pure-geometric — envelope membership +
        dwell are computed from truth geometry alone, NOT gated by controller
        state, so the observer measures truth reality independently.
        """
        if geom is None:
            for tr in self._trackers.values():
                tr.update(False, dt_s, self.tick_idx)
            return {name: False for name in self._trackers}

        b = float(geom.behind_m)
        c = float(geom.cross_m)
        z = float(geom.below_m)
        flags: Dict[str, bool] = {}
        for (name, pred, _dwell_req, _entry_only) in MILESTONE_LADDER:
            in_env = bool(pred(b, c, z))
            self._trackers[name].update(in_env, dt_s, self.tick_idx)
            flags[name] = in_env

        # Aggregate axis minima (absolute) + min range.
        ab, ac, az = abs(b), abs(c), abs(z)
        if ab < self._min_behind_abs_m:
            self._min_behind_abs_m = ab
        if ac < self._min_cross_abs_m:
            self._min_cross_abs_m = ac
        if az < self._min_below_abs_m:
            self._min_below_abs_m = az
        if geom.range_m < self._min_range_m:
            self._min_range_m = geom.range_m
        return flags

    def _raw_cg_quartet(self):
        """CSV2_R7K (additive, non-frozen): raw-GPS follower (GP_INT_sys1) -> leader
        (GP_INT_sys2) CG-to-CG and CG-to-boom-tip geometry, fusion-free, via
        diagnostic_frame_log.frame_quartet. Uses v98's OWN boom constants
        (BOOM_HORIZONTAL_PROJECTION_M / BOOM_TIP_BELOW_OFFSET_M) and home so the
        raw_cg_boom basis matches SIM_TRUTH by construction (differs only by
        follower source: raw GPS here vs SIM_STATE in _build_raw_geometry).
        Returns (raw_cg_cg, raw_cg_boom) dicts or (None, None); NEVER raises."""
        try:
            lead = self.cache.get("GLOBAL_POSITION_INT_sys2")
            foll = self.cache.get("GLOBAL_POSITION_INT_sys1")
            if lead is None or foll is None:
                return None, None
            q = frame_quartet(
                foll_llh=(foll.lat / 1e7, foll.lon / 1e7, foll.alt / 1000.0),
                lead_llh=(lead.lat / 1e7, lead.lon / 1e7, lead.alt / 1000.0),
                lead_hdg_deg=lead.hdg / 100.0,
                home_llh=(SITL_HOME_LAT, SITL_HOME_LON, SITL_HOME_ALT_MSL),
                boom_horiz_m=BOOM_HORIZONTAL_PROJECTION_M,
                boom_below_m=BOOM_TIP_BELOW_OFFSET_M,
            )
            return q["raw_cg_cg"], q["raw_cg_boom"]
        except Exception as exc:
            logger.debug(f"raw_cg_quartet failed: {exc}")
            return None, None

    def _emit_paired_tick(
        self, geom, sim_time_s: Optional[float], flags: Dict[str, bool]
    ) -> None:
        rec = {
            "tick_idx": self.tick_idx,
            "utc_ms": int(time.time() * 1000),
            "sim_time_s": sim_time_s,
            "raw_truth_behind_m": geom.behind_m if geom is not None else None,
            "raw_truth_cross_m": geom.cross_m if geom is not None else None,
            "raw_truth_below_m": geom.below_m if geom is not None else None,
            "raw_truth_range_m": geom.range_m if geom is not None else None,
            # Backward-compat single flag + full truth ladder for this tick.
            "in_m3b_truth": bool(flags.get("m3b", False)),
            "truth_milestones": {k: bool(v) for k, v in flags.items()},
            "uwb_enabled": self.uwb_enabled,
        }
        # CSV2_R7K additive fields (never alter the truth path above).
        rec["raw_cg_cg"], rec["raw_cg_boom"] = self._raw_cg_quartet()
        try:
            self.paired_log_handle.write(json.dumps(rec) + "\n")
        except Exception as exc:
            logger.warning(f"paired_tick write failed: {exc}")

    def _request_sim_state_stream(self) -> None:
        """Self-request SIM_STATE(108) @ 10 Hz via SET_MESSAGE_INTERVAL so truth
        does not depend on the controller's _FD_UWB_TRUTH_SOURCED gate. Fields are
        the hardcoded sysid=1/comp=1 params from sim_state_request_params()."""
        if self.conn is None:
            return
        p = sim_state_request_params()
        try:
            self.conn.mav.command_long_send(
                p["target_system"],
                p["target_component"],
                p["command"],
                0,  # confirmation
                float(p["param1"]),
                float(p["param2"]),
                0.0, 0.0, 0.0, 0.0, 0.0,
            )
            logger.info(
                f"sent SET_MESSAGE_INTERVAL SIM_STATE({p['param1']}) "
                f"@ {1e6 / p['param2']:.1f} Hz to sys={p['target_system']} "
                f"comp={p['target_component']}"
            )
        except Exception as exc:
            logger.warning(f"SIM_STATE(108) self-request failed: {exc}")

    def run(self) -> None:
        self._open_connection()
        # Optional warmup: wait briefly for first heartbeat.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not self._stop:
            self._drain()
            if "HEARTBEAT" in self.cache:
                logger.info("first HEARTBEAT received")
                break
            time.sleep(0.1)

        # Self-request SIM_STATE(108) @ 10 Hz (truth-logger STRUCTURAL). Sent once
        # after the link is up; if no heartbeat arrived we still try once.
        self._request_sim_state_stream()

        next_tick_t = time.monotonic()
        while not self._stop:
            self._drain()
            geom = self._build_raw_geometry()
            sim_time_msg = self.cache.get("SYSTEM_TIME") or self.cache.get("SYSTEM_TIME_sys1")
            sim_time_s: Optional[float] = None
            if sim_time_msg is not None:
                try:
                    sim_time_s = float(sim_time_msg.time_boot_ms) / 1000.0
                except Exception:
                    sim_time_s = None
            flags = self._update_ladder(geom, self.tick_dt)
            self._emit_paired_tick(geom, sim_time_s, flags)
            self.tick_idx += 1
            next_tick_t += self.tick_dt
            sleep_s = next_tick_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Slipped — skip catch-up to avoid burst-firing.
                next_tick_t = time.monotonic()

        self._finalize()

    def _finalize(self) -> None:
        try:
            self.paired_log_handle.flush()
            self.paired_log_handle.close()
        except Exception:
            pass

        m3b = self._trackers["m3b"]
        summary = {
            "schema_version": "2.0",
            "source": "tools/v98_standalone_truth_observer.py:_finalize",
            "env_id": self.env_id,
            "frame": "boom_tip_delta",
            "uwb_enabled": self.uwb_enabled,
            "tick_count": self.tick_idx,
            "tick_hz": self.tick_hz,
            # SIM_STATE(108) health: 0 == truth never fell back to GPI GPS.
            "sim_state_fallback_ticks": int(self._sim_state_fallback_ticks),
            # Backward-compat top-level M3b keys (rollup _extract_truth consumes these).
            "m3b_entries": int(m3b.entries),
            "m3b_best_dwell_s": float(m3b.best_s),
            "m3b_continuous_dwell_s_at_end": float(m3b.continuous_s),
            "in_m3b_at_end": bool(m3b.in_env),
            # Full truth-frame ladder (M1/M2/M3a-coarse/M3a-fine/M3b).
            "milestones_truth": {
                name: tr.to_dict() for name, tr in self._trackers.items()
            },
            "min_range_m": (
                None if math.isinf(self._min_range_m) else float(self._min_range_m)
            ),
            "min_behind_abs_m": (
                None if math.isinf(self._min_behind_abs_m) else float(self._min_behind_abs_m)
            ),
            "min_cross_abs_m": (
                None if math.isinf(self._min_cross_abs_m) else float(self._min_cross_abs_m)
            ),
            "min_below_abs_m": (
                None if math.isinf(self._min_below_abs_m) else float(self._min_below_abs_m)
            ),
            # Thresholds reconciled to formation_success_metrics.MilestoneThresholds
            # (single source). M3b shown for back-compat; full ladder under
            # milestones_truth[*].dwell_req_s.
            "thresholds": {
                "behind_max_m": float(_MT.m3b_delta_behind_tol),
                "below_max_m": float(_MT.m3b_delta_below_tol),
                "cross_max_m": float(_MT.m3b_delta_cross_tol),
                "dwell_req_s": float(_MT.m3b_dwell_s),
            },
            "paired_tick_log_path": str(self.paired_log_path),
        }
        out_path = self.out_dir / "run_summary_env_truth.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"wrote {out_path} (best_m3b_dwell_s={m3b.best_s:.3f})")


def main() -> int:
    _setup_logging()
    if "HARNESS_OUT_DIR" not in os.environ:
        print(
            "[truth_observer] HARNESS_OUT_DIR env var required; aborting.",
            file=sys.stderr,
        )
        return 2
    obs = TruthObserver()
    signal.signal(signal.SIGTERM, obs.shutdown)
    signal.signal(signal.SIGINT, obs.shutdown)
    try:
        obs.run()
    except Exception as exc:
        logger.exception(f"observer crashed: {exc}")
        obs._finalize()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
