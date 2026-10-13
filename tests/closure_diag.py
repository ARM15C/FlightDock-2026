"""TIER2 truth CG-CG closure projection — pure math helper (dispatch 3, 2026-07-02).

Responsibility: convert two TRUTH states (follower pos/vel + leader pos/vel;
lat/lon in deg, alt in m AMSL, velocities in NED m/s, down-positive) into the
scalar closure rate along the CG-to-CG line (positive = closing) and the CG-CG
range. Consumed ONLY by the speed_arbitration.csv diagnostic writer in
baseline_formation_controller.py when FD_TRUTH_CLOSURE_DIAG=1 — NEVER by
obs/actor/critic/reward/replay/policy/guard (Tier-2 spec leakage stance).

Public API: truth_closure_cgcg(...) -> Optional[Tuple[closure_mps, range_m]].
Side-effects: NONE (no I/O, no env reads, no state; imports math/typing only).
Caps: file soft 60 / hard 80 LoC; per-function <= 30 LoC.
Coverage: tests/unit/test_truth_closure_diag.py (analytic projection, sign
convention, degenerate range, equivalence to -d(range)/dt).
"""
import math
from typing import Optional, Tuple

# WGS-84 mean meters-per-degree-latitude; flat-earth projection is exact to
# sub-mm at docking-corridor ranges (< 100 m).
_M_PER_DEG_LAT = 111319.4908


def truth_closure_cgcg(
        f_lat: float, f_lon: float, f_alt_msl: float,
        f_vn: float, f_ve: float, f_vd: float,
        l_lat: float, l_lon: float, l_alt_msl: float,
        l_vn: float, l_ve: float, l_vd: float,
        min_range_m: float = 0.05) -> Optional[Tuple[float, float]]:
    """Closure rate (+ = closing) and range on the follower->leader CG-CG line.

    closure = (v_follower - v_leader). r_hat, with r_hat = unit(p_leader -
    p_follower) in NED — algebraically identical to -d|r|/dt, so it matches the
    safety guard's range-differentiation sign convention (positive = range
    decreasing). Returns None when range < min_range_m (unit vector degenerate).
    """
    dn = (l_lat - f_lat) * _M_PER_DEG_LAT
    de = (l_lon - f_lon) * _M_PER_DEG_LAT * math.cos(
        math.radians(0.5 * (f_lat + l_lat)))
    dd = f_alt_msl - l_alt_msl  # NED down = -alt
    rng = math.sqrt(dn * dn + de * de + dd * dd)
    if rng < min_range_m:
        return None
    closure = ((f_vn - l_vn) * dn
               + (f_ve - l_ve) * de
               + (f_vd - l_vd) * dd) / rng
    return (closure, rng)
