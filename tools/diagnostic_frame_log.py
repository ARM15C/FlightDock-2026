#!/usr/bin/env python3
"""diagnostic_frame_log — pure frame-geometry helper for CSV2_R7K_FRAME_VERIFICATION_LOG.

RESPONSIBILITY
    Compute, from RAW positions (no fusion, no EKF), the follower-relative
    geometry to the leader in two reference frames, for side-by-side
    comparison against the controller's FUSED estimate and SIM ground truth:
      - raw_cg_cg: follower CG -> leader CG (NO boom transform)
      - raw_cg_boom: follower CG -> leader boom tip (boom transform applied)
    Used by tools/v98_standalone_truth_observer.py (additive logging hook) and
    tools/analyze_frame_quartet.py (offline four-way comparison).

    Boom geometry: the DHC-2 boom tip is 4.31 m BEHIND (aft) and 4.31 m BELOW
    the leader CG (6.10 m slant at 45 deg). This is BEHIND, not forward of, the
    CG (M2: 14.31 m behind CG = 10 m behind boom tip => boom 4.31 m behind CG;
    code constant BOOM_TIP_OFFSET_M = 4.31 behind/below). Frame conventions
    mirror tests/geometry.py:767-791 and tests/baseline_formation_controller.py
    :3328-3348 sign-for-sign so raw_cg_boom shares the SIM_TRUTH basis.

    Sign conventions (NED, leader body frame):
      behind = -(rel_n*cos(psi) + rel_e*sin(psi)) (>0 => follower behind leader)
      cross = -rel_n*sin(psi) + rel_e*cos(psi)
      below = rel_d (>0 => follower BELOW ref;
               NED down-positive. A follower at leader-CG altitude is ABOVE the
               boom tip => raw_cg_boom.below < 0 there; a docking pose has the
               follower below the boom tip => below > 0.)
    Altitude (Option B, absolute MSL): d = -(alt_msl - home_alt_msl); aircraft
    above home => d < 0. This matches v98 SIM_TRUTH (absolute MSL + home 584.0).

PUBLIC API
    frame_quartet(foll_llh, lead_llh, lead_hdg_deg, home_llh,
                  boom_horiz_m, boom_below_m) -> dict
    _quartet_ned(foll_ned, lead_ned, hdg_deg, boom_horiz_m, boom_below_m) -> dict
    _ned(lat, lon, alt_msl, home_lat, home_lon, home_alt_msl) -> (n, e, d)
    _body_bcb(rel_n, rel_e, rel_d, hdg_deg) -> (behind, cross, below)

SIDE EFFECTS
    NONE. Pure functions: no I/O, no filesystem, no subprocess, no fusion call,
    no global state. Deterministic.

CAPS
    file soft 200 / hard 350 LoC; function LoC cap 60.

COVERAGE
    tests/unit/test_diagnostic_frame_log.py (cases i, ii, iii, iii-b, iv, v, vi, vii).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

# WGS-ish spherical earth radius; matches tests/geometry.py / controller (R_EARTH).
R_EARTH_M: float = 6371000.0

Vec3 = Tuple[float, float, float]


def _ned(
    lat: float,
    lon: float,
    alt_msl: float,
    home_lat: float,
    home_lon: float,
    home_alt_msl: float,
) -> Vec3:
    """Convert (lat, lon, alt_msl) -> local NED (n, e, d) about home.

    d = -(alt_msl - home_alt_msl); NED down-positive; aircraft above home => d < 0.
    Longitude scaled by cos(home_lat) (matches controller:3322-3323/3336-3337).
    """
    n = math.radians(lat - home_lat) * R_EARTH_M
    e = math.radians(lon - home_lon) * R_EARTH_M * math.cos(math.radians(home_lat))
    d = -(alt_msl - home_alt_msl)
    return (n, e, d)


def _body_bcb(rel_n: float, rel_e: float, rel_d: float, hdg_deg: float) -> Vec3:
    """Rotate a follower-relative NED vector into leader body (behind, cross, below).

    Mirrors tests/geometry.py:782-791 / controller:3346-3348 sign-for-sign.
    """
    psi = math.radians(hdg_deg)
    cos_h = math.cos(psi)
    sin_h = math.sin(psi)
    behind = -(rel_n * cos_h + rel_e * sin_h)
    cross = -rel_n * sin_h + rel_e * cos_h
    below = rel_d
    return (behind, cross, below)


def _quartet_ned(
    foll_ned: Vec3,
    lead_ned: Vec3,
    hdg_deg: float,
    boom_horiz_m: float,
    boom_below_m: float,
) -> Dict[str, Dict[str, float]]:
    """Compute raw_cg_cg and raw_cg_boom (behind/cross/below) from NED positions.

    raw_cg_cg: rel = follower - leader CG (no boom)
    raw_cg_boom: boom tip = leader CG offset BEHIND+BELOW; rel = follower - boom
                  boom_n = lead_n - boom_horiz*cos(psi) (aft; matches geometry.py:769-771)
                  boom_e = lead_e - boom_horiz*sin(psi)
                  boom_d = lead_d + boom_below (below; geometry.py:775)
    """
    fn, fe, fd = foll_ned
    ln, le, ld = lead_ned
    psi = math.radians(hdg_deg)
    cos_h = math.cos(psi)
    sin_h = math.sin(psi)

    cg = _body_bcb(fn - ln, fe - le, fd - ld, hdg_deg)

    boom_n = ln - boom_horiz_m * cos_h
    boom_e = le - boom_horiz_m * sin_h
    boom_d = ld + boom_below_m
    boom = _body_bcb(fn - boom_n, fe - boom_e, fd - boom_d, hdg_deg)

    return {
        "raw_cg_cg": {"behind": cg[0], "cross": cg[1], "below": cg[2]},
        "raw_cg_boom": {"behind": boom[0], "cross": boom[1], "below": boom[2]},
    }


def frame_quartet(
    foll_llh: Vec3,
    lead_llh: Vec3,
    lead_hdg_deg: float,
    home_llh: Vec3,
    boom_horiz_m: float,
    boom_below_m: float,
) -> Dict[str, Dict[str, float]]:
    """Compute {raw_cg_cg, raw_cg_boom} from raw lat/lon/alt_msl inputs.

    foll_llh / lead_llh / home_llh are (lat, lon, alt_msl). Altitude is ABSOLUTE
    MSL (Option B) referenced to home_alt_msl, matching v98 SIM_TRUTH. The caller
    (v98 hook) passes its OWN absolute-MSL alt values + boom constants so the
    raw_cg_boom basis is identical to SIM_TRUTH (differing only by follower source).
    """
    home_lat, home_lon, home_alt = home_llh
    foll_ned = _ned(foll_llh[0], foll_llh[1], foll_llh[2], home_lat, home_lon, home_alt)
    lead_ned = _ned(lead_llh[0], lead_llh[1], lead_llh[2], home_lat, home_lon, home_alt)
    return _quartet_ned(foll_ned, lead_ned, lead_hdg_deg, boom_horiz_m, boom_below_m)
