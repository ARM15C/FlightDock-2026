# =============================================================================
# RESPONSIBILITY: Pure arithmetic for FD_CTRL_START_EPOCH_QUANT (#326, R7.1
#   re-auth 2026-07-05): snap the controller mission-start epoch to a fixed
#   sim-time grid so INIT->INTERCEPT fires at a bit-identical tbms every run.
#   compute_target_ms picks the next STRICT grid multiple after the last tbms
#   fed to the time authority at sync-verdict PASS (strictly-after rule: an
#   exact-on-grid input still advances one quantum, so the target is always
#   ahead of the authority and never overshot pre-snap). first_at_or_after
#   scans the (monotonic, FD-001) ATTITUDE tbms queue for the first entry at
#   or past the target — queue CONTENT is lockstep-deterministic, so the
#   result is wall-independent. format_snap_log renders the controller's snap
#   log line (kept here so the format is unit-covered).
# PUBLIC API: compute_target_ms(last_fed_tbms_ms, q_ms=1000) -> int;
#   first_at_or_after(queue, target_ms) -> Optional[int];
#   format_snap_log(target_ms, t_star_ms, raw_pass_s) -> str.
# SIDE-EFFECTS: none (pure functions; no I/O, no env reads, no imports beyond
#   typing).
# CAPS: file soft 40 / hard 55 LoC; per-function <= 12 LoC.
# COVERAGE: tests/unit/test_ctrl_start_epoch_quant.py (grid snap incl.
#   exact-on-grid, scan hit/miss/empty/deque, coercion, log format).
# =============================================================================
from typing import Iterable, Optional


def compute_target_ms(last_fed_tbms_ms, q_ms: int = 1000) -> int:
    """Next strict grid multiple after last_fed_tbms_ms (grid anchored at 0)."""
    t = int(last_fed_tbms_ms)
    q = int(q_ms)
    return (t // q + 1) * q


def first_at_or_after(queue: Iterable, target_ms) -> Optional[int]:
    """First tbms >= target_ms in queue order (monotonic queue => minimum)."""
    t = int(target_ms)
    for tb in queue:
        if tb >= t:
            return int(tb)
    return None


def format_snap_log(target_ms, t_star_ms, raw_pass_s) -> str:
    """The controller's snap log line (format frozen for extractors)."""
    return (f"[SIM_TIME] START_EPOCH_QUANT: snapped to {target_ms / 1000.0:.3f}s "
            f"(t*={t_star_ms}ms, raw PASS {raw_pass_s:.3f}s)")
