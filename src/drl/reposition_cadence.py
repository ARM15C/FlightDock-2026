# =============================================================================
# RESPONSIBILITY: FIX-α (#328) pure decision arithmetic — forced DO_REPOSITION
#   re-sends re-arm on the interval rate-limit. #327 RCA [MEASURED]: once the
#   ALT_CAP climb lags, the env's persistent-alt-error forcing bypasses the
#   interval limit EVERY tick (22.6 Hz re-sends -> ArduPlane realizes ~0
#   climb), while the standalone's working pattern is interval-limited
#   (~7.1 Hz -> 0.44 m/s climb, docks). Genuine altitude-TARGET-change steps
#   stay immediate (they must fire the forced reposition + GCA pair at the
#   step); persistent-error forcing may not exceed the interval cadence.
#   Flag-off returns True unconditionally = legacy byte-identical behavior.
# PUBLIC API: force_send_allowed(rearm_on, alt_target_changed, now_s,
#   last_send_s, min_interval_s) -> bool.
# SIDE-EFFECTS: none (pure function; no I/O, no env reads, no imports).
# CAPS: file soft 40 / hard 55 LoC; per-function <= 12 LoC.
# COVERAGE: tests/unit/test_reposition_cadence.py (off-identity, target-change
#   immediacy, inside/past/boundary interval, float coercion).
# =============================================================================


def force_send_allowed(rearm_on, alt_target_changed, now_s, last_send_s,
                       min_interval_s):
    """FIX-α: may a FORCED reposition send proceed at this tick?

    Target-change steps are always immediate; with the re-arm flag on,
    persistent-error forcing is bounded by the interval rate-limit (the
    standalone's measured working cadence). Flag off => always True (legacy).
    """
    if not rearm_on or alt_target_changed:
        return True
    return (float(now_s) - float(last_send_s)) >= float(min_interval_s)
