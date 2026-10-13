# =============================================================================
# RESPONSIBILITY: Option B (#308 C+B dispatch) — pure resolver for the leader
#   trajectory-epoch latch under the deterministic engagement gate. Decides
#   whether the epoch that leader_broadcast_fixed.py latches into
#   pattern_start_time/state_start_time/last_sim_time at gate-bite is the
#   wall-drained follower crossing sample (legacy) or the frozen T_ENGAGE
#   constant (quantized — kills the SG-D4 ±25-50 ms epoch draw at its source,
#   since T_ENGAGE already sits on the 1 ms tbms/FDM frame grid).
# PUBLIC API: resolve_engage_epoch(quant, t_engage, current_sim_time)
#   -> (epoch: float, log_suffix: str)
# SIDE-EFFECTS: none (pure function; no I/O, no globals, no clock reads).
# CAPS: file soft 60 / hard 80 LoC; per-function <= 30 LoC.
# COVERAGE: tests/unit/test_engage_epoch_quant.py (unit smoke, runs in the
#   standard `pytest tests/unit -x -q` sweep).
# =============================================================================


def resolve_engage_epoch(quant, t_engage, current_sim_time):
    """Resolve the trajectory-epoch value + log suffix at gate-bite.

    quant is True AND the gate is armed (t_engage is not None) => snap the
    epoch to the frozen T_ENGAGE constant so identical-seed runs latch a
    bit-identical trajectory epoch; the wall-drained crossing sample is
    preserved in the log suffix as a diagnostic so the residual jitter stays
    measurable. Otherwise (quant OFF, or gate unarmed) => legacy path: the
    epoch IS the wall-latched current_sim_time and the suffix is empty, so the
    default log line is byte-identical to prior behavior.

    Returns (epoch, log_suffix). The suffix is appended AFTER the trailing "s"
    of the "[LB-1] Sim-time initialized (engage) at {epoch:.2f}s" line, so the
    adjudicator's INIT_RE (which stops at the first "s") parses the snapped
    epoch while the raw crossing remains human/tool-readable in-line.
    """
    if quant and t_engage is not None:
        return float(t_engage), f" (raw {current_sim_time:.2f}s)"
    return current_sim_time, ""
