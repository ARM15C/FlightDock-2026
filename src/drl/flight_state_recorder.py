#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: copy per-tick flight telemetry to one gzipped CSV per episode.
#   Exists because on 2026-08-20 an eval wave banked 35 legs whose airspeed was
#   absent for a median 6.3% of ticks (max 100%) while the env silently
#   substituted 24.7 m/s -- the leader's cruise speed, mid-envelope, so
#   SAFETY_AIRSPEED could not fire. The legs were voided: with the guards
#   disarmed we do not know what they flew.
#
# WHY IT READS THE RAW MESSAGE CACHE, NOT FollowerState. FollowerState carries
#   the SUBSTITUTED value. A recorder reading it would faithfully log 24.7 and
#   teach us nothing. This reads the pymavlink cache, so a missing message is
#   recorded as EMPTY because it is missing, and the value the env actually
#   used is written in a parallel `*_used` column. The DISAGREEMENT between the
#   two columns is the fabrication detector, and it costs nothing extra.
#
# THIS MODULE PERFORMS NO ARITHMETIC. Every column is a copy of a value that
#   already exists, in its RAW WIRE UNIT (mm, centidegrees, cm/s, radians, degE7)
#   -- not even a unit conversion happens here. All derivation (AGL, degrees,
#   the fabrication mask, missingness rates) belongs to the offline
#   fabrication_census.py. This is the addendum's "no metric computation inside
#   env" rule, taken literally so a reader can verify it by inspection.
#
# PUBLIC API: open_leg(out_dir, episode_idx, header) -> Recorder|None
#             Recorder.write_row(**sources) · Recorder.close_leg()
# SIDE-EFFECTS: opens / writes / closes exactly one file per episode under the
#   directory it is given. No network, no process spawn, no mutation of any
#   input object, no import of any policy-path module.
# FAIL-SOFT: every entry point swallows its own exceptions. A recorder fault
#   must never kill a flying leg (detector A6). A leg that records nothing is
#   caught downstream by the row-count gate (detector A2), never here.
# CAPS: file soft 180 / hard 220 code-lines; function cap 35.
# COVERAGE: tests/unit/test_flight_state_recorder.py, detectors A1-A6 both ways.
# =============================================================================
"""Per-tick flight-state recorder. Copies only; see header block above."""
import csv
import gzip
import logging
import os

from . import leader_cache_probe as _probe   # TELEMPROBE-775, read-only (relative: dual import path)

logger = logging.getLogger(__name__)

# (column, attribute) pairs. The attribute is read with getattr(..., None), so a
# message the autopilot never sent yields an EMPTY cell, never a placeholder.
_VFR = (("vfr_tas_mps", "airspeed"), ("vfr_gs_mps", "groundspeed"),
        ("vfr_alt_m", "alt"), ("vfr_climb_mps", "climb"),
        ("vfr_throttle_pct", "throttle"), ("vfr_hdg_deg", "heading"))
_ATT = (("att_roll_rad", "roll"), ("att_pitch_rad", "pitch"),
        ("att_yaw_rad", "yaw"), ("att_p_rps", "rollspeed"),
        ("att_q_rps", "pitchspeed"), ("att_r_rps", "yawspeed"),
        ("att_tbms", "time_boot_ms"))
_GPI = (("gpi_lat_e7", "lat"), ("gpi_lon_e7", "lon"), ("gpi_alt_mm", "alt"),
        ("gpi_relalt_mm", "relative_alt"), ("gpi_hdg_cdeg", "hdg"),
        ("gpi_vx_cms", "vx"), ("gpi_vy_cms", "vy"), ("gpi_vz_cms", "vz"),
        ("gpi_tbms", "time_boot_ms"))
_HB = (("hb_custom_mode", "custom_mode"), ("hb_base_mode", "base_mode"))
_NAV = (("nav_roll_deg", "nav_roll"), ("nav_pitch_deg", "nav_pitch"),
        ("nav_bearing_cdeg", "nav_bearing"),
        ("nav_target_bearing_cdeg", "target_bearing"),
        ("nav_wp_dist_m", "wp_dist"), ("nav_alt_error_m", "alt_error"),
        ("nav_aspd_error_mps", "aspd_error"),
        ("nav_xtrack_error_m", "xtrack_error"))
# SIM_STATE truth attitude + IMU: parsed into the cache by pymavlink and
# discarded by _latest_new_sim_state, which returns position/velocity only.
_SIM = (("roll_rad", "roll"), ("pitch_rad", "pitch"), ("yaw_rad", "yaw"),
        ("xacc", "xacc"), ("yacc", "yacc"), ("zacc", "zacc"),
        ("xgyro", "xgyro"), ("ygyro", "ygyro"), ("zgyro", "zgyro"))
# What the env actually fed the guards and the observation builder.
_USED = ("airspeed_ms", "alt_msl_m", "heading_deg", "pitch_deg", "roll_deg",
         "vd_ms", "throttle_pct")
_GEOM = ("behind_m", "cross_m", "below_m", "range_m", "target_aspect_deg",
         "closure_rate_ms", "vel_cross_m_s", "vel_below_m_s", "source")
_GUARD = ("_guard_altitude_ceiling_fire_count", "_guard_hard_overtake_fire_count",
          "_guard_speed_cap_fire_count", "_guard_env_fallback_count",
          "_count_no_pass_violations", "_safety_emergency_active")
_IDENT = ("timestamp_wall_s", "sim_time_s", "episode_idx", "env_id",
          "tick_idx", "fsm_state", "terminal_reason", "home_alt_msl")


def _cols(prefix, specs):
    return ["%s_%s" % (prefix, c) for c, _ in specs]


def fieldnames():
    """The column order of record. Stable; readers may rely on it."""
    out = list(_IDENT)
    for pfx, specs in (("f", _VFR), ("f", _ATT), ("f", _GPI), ("f", _HB),
                       ("f", _NAV), ("t", _SIM), ("l", _VFR), ("l", _ATT),
                       ("l", _GPI)):
        out += _cols(pfx, specs)
    out += ["f_used_%s" % c for c in _USED]
    out += ["l_used_%s" % c for c in ("airspeed_ms", "heading_deg",
                                      "alt_msl_m", "vd_ms")]
    for ch in ("raw", "env", "ctrl"):
        out += ["%s_%s" % (ch, c) for c in _GEOM]
    out += ["cmd_speed_mps", "cmd_heading_deg", "cmd_alt_msl_m"]
    out += ["res_policy_0", "res_policy_1", "res_policy_2",
            "res_applied_0", "res_applied_1", "res_applied_2", "res_source"]
    out += list(_GUARD)
    return out


def _pull(row, prefix, specs, msg):
    """Copy named attributes off one cached message. No conversion."""
    for col, attr in specs:
        row["%s_%s" % (prefix, col)] = (None if msg is None
                                        else getattr(msg, attr, None))


def _cached(mav, msg_type, sysid=None):
    """One message out of a pymavlink cache, optionally sysid-filtered.

    pymavlink caches EVERY received message regardless of whether recv_match's
    caller kept it, which is why the leader's ATTITUDE -- dropped unread by the
    R7.1-frozen receiver -- is reachable here without touching that file."""
    # TELEMWIRE-776 [MEASURED by TELEMPROBE-775, 3 legs x 2 probes]: the
    # leader receiver's mavfile LOCKS onto the first vehicle heartbeat it sees
    # (sysid 1, the follower, shares the UDP port), and pymavlink's `messages`
    # PROPERTY serves only sysid_state[self.sysid]. So with a sysid asked for,
    # read THAT sysid's cache directly; the property is kept for the unfiltered
    # follower path and as the fallback for caches without sysid_state.
    try:
        cache = getattr(mav, "messages", {})
        if sysid is not None and hasattr(mav, "sysid_state"):
            st = mav.sysid_state.get(sysid)
            cache = getattr(st, "messages", {}) if st is not None else {}
        msg = cache.get(msg_type)
    except Exception:
        return None
    if msg is None or sysid is None:
        return msg
    try:
        return msg if msg.get_srcSystem() == sysid else None
    except Exception:
        return None


class Recorder:
    """One open episode file. Construct via open_leg()."""

    def __init__(self, fh, writer, path):
        self._fh, self._w, self.path, self.rows = fh, writer, path, 0

    def write_row(self, ident, f_mav=None, l_mav=None, f_used=None,
                  l_used=None, geoms=None, cmds=None, res=None, ctrl=None):
        """Emit one tick. Never raises (detector A6)."""
        try:
            row = dict.fromkeys(self._w.fieldnames)
            row.update(ident or {})
            _pull(row, "f", _VFR, _cached(f_mav, "VFR_HUD"))
            _pull(row, "f", _ATT, _cached(f_mav, "ATTITUDE"))
            _pull(row, "f", _GPI, _cached(f_mav, "GLOBAL_POSITION_INT"))
            _pull(row, "f", _HB, _cached(f_mav, "HEARTBEAT"))
            _pull(row, "f", _NAV, _cached(f_mav, "NAV_CONTROLLER_OUTPUT"))
            _pull(row, "t", _SIM, _cached(f_mav, "SIM_STATE"))
            for pfx, specs, mt in (("l", _VFR, "VFR_HUD"),
                                   ("l", _ATT, "ATTITUDE"),
                                   ("l", _GPI, "GLOBAL_POSITION_INT")):
                _pull(row, pfx, specs, _cached(l_mav, mt, sysid=2))
            for col in _USED:
                row["f_used_%s" % col] = getattr(f_used, col, None)
            for col in ("airspeed_ms", "heading_deg", "alt_msl_m", "vd_ms"):
                row["l_used_%s" % col] = getattr(l_used, col, None)
            for ch, g in (geoms or {}).items():
                for col in _GEOM:
                    row["%s_%s" % (ch, col)] = getattr(g, col, None)
            row.update(cmds or {})
            row.update(res or {})
            for col in _GUARD:
                row[col] = getattr(ctrl, col, None)
            if self.rows in _probe.PROBE_ROWS:          # TELEMPROBE-775
                _probe.probe_once(self.path, self.rows, ctrl, l_mav)
            self._w.writerow(row)
            self.rows += 1
        except Exception:
            logger.debug("flight_state row failed", exc_info=True)

    def close_leg(self):
        """Flush and close. Never raises."""
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            logger.debug("flight_state close failed", exc_info=True)


def open_leg(out_dir, episode_idx, header=None):
    """Open one episode file. Returns None if disabled or unopenable.

    Returning None is not an error path the caller must branch on -- the env
    holds Optional[Recorder] and skips when it is None, exactly as it already
    does for the paired CSV."""
    if os.environ.get("FD_FLIGHT_STATE_LOG", "1") == "0":
        return None
    try:
        gz = os.environ.get("FD_FLIGHT_STATE_GZIP", "1") != "0"
        name = "flight_state_ep%04d.csv%s" % (episode_idx, ".gz" if gz else "")
        path = os.path.join(str(out_dir), name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fh = (gzip.open(path, "wt", newline="") if gz
              else open(path, "w", newline=""))
        if header:
            fh.write("".join("# %s=%s\n" % kv for kv in sorted(header.items())))
        w = csv.DictWriter(fh, fieldnames=fieldnames(), extrasaction="ignore")
        w.writeheader()
        return Recorder(fh, w, path)
    except Exception:
        logger.debug("flight_state open failed", exc_info=True)
        return None
