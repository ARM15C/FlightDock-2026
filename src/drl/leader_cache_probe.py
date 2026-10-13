#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: TELEMPROBE-775 -- a one-shot, read-only DIAGNOSTIC of the
#   recorder's leader access path. The recorder's 22 l_* columns are empty on
#   every banked leg while the env's own leader_age_ms is live; the fault is
#   therefore in how the recorder reads pymavlink's cache, and it could not be
#   determined statically (two root causes already asserted and withdrawn).
#   This writes what the recorder can actually SEE at the handle it was
#   given: is ctrl None, is leader_rx None, is the mav the SAME object the
#   recorder reads, what type is it, which sysid did it lock onto, which
#   sysids hold messages, and which message types each holds.
#   It decides NOTHING and changes NOTHING; the fix is a separate dispatch.
# PUBLIC API: probe(ctrl, l_mav) -> dict
#             probe_once(rec_path, row_idx, ctrl, l_mav) -> str|None (file)
#             main(argv) -> int (--selfcheck only)
# SIDE-EFFECTS: writes leader_cache_probe_ep<NNNN>_r<row>.json beside the
#   flight-state file. No network, no mutation of any input, never raises.
# FAIL DIRECTION: `readable` is False and `error` is set whenever an attribute
#   could not be inspected. An unreadable probe is INCONCLUSIVE, never a
#   verdict -- the JSON says so in its own field.
# CAPS: file soft 60 / hard 80 code-lines; function cap 30.
#   SOFT OVERAGE DISCLOSED, NOT RAISED: 71 code-lines at landing (eighth
#   consecutive low LoC estimate); hard 80 clear.
# COVERAGE: tests/unit/test_leader_cache_probe.py + --selfcheck, both ways.
# =============================================================================
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)
PROBE_ROWS = (0, 1200)       # first tick, and ~60 s in at 20 Hz


def _sysid_table(mav):
    out = {}
    for sid, st in getattr(mav, "sysid_state", {}).items():
        msgs = getattr(st, "messages", {})
        out[str(sid)] = sorted(msgs.keys())[:40]
    return out


def probe(ctrl, l_mav):
    """Everything the recorder can see at the handle it was handed."""
    r = {"readable": False, "dispatch": "TELEMPROBE-775"}
    try:
        lrx = getattr(ctrl, "leader_rx", None)
        r["ctrl_is_none"] = ctrl is None
        r["leader_rx_is_none"] = lrx is None
        r["l_mav_is_none"] = l_mav is None
        r["l_mav_is_leader_rx_mav"] = (lrx is not None
                                       and getattr(lrx, "mav", None) is l_mav)
        r["l_mav_type"] = type(l_mav).__name__ if l_mav is not None else None
        if l_mav is not None:
            r["mav_sysid"] = getattr(l_mav, "sysid", None)
            r["mav_target_system"] = getattr(l_mav, "target_system", None)
            r["mav_count"] = getattr(l_mav, "mav_count", None)
            r["sysid_state_keys"] = sorted(
                str(k) for k in getattr(l_mav, "sysid_state", {}).keys())
            r["messages_types_at_selected_sysid"] = sorted(
                getattr(l_mav, "messages", {}).keys())[:40]
            vfr = getattr(l_mav, "messages", {}).get("VFR_HUD")
            r["vfr_hud_src_sysid"] = (vfr.get_srcSystem()
                                      if vfr is not None else None)
            r["per_sysid_types"] = _sysid_table(l_mav)
        r["readable"] = True
    except Exception as e:                       # noqa: BLE001 - diagnostic
        r["error"] = "%s: %s" % (type(e).__name__, e)
    return r


def probe_once(rec_path, row_idx, ctrl, l_mav):
    """Write the probe beside the flight-state file. Never raises."""
    try:
        base = os.path.basename(rec_path).split(".")[0]     # flight_state_epNNNN
        name = "leader_cache_probe_%s_r%d.json" % (base.replace("flight_state_", ""), row_idx)
        path = os.path.join(os.path.dirname(rec_path), name)
        d = probe(ctrl, l_mav); d["row_idx"] = row_idx
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1, default=str)
        return path
    except Exception:
        logger.debug("leader_cache_probe failed", exc_info=True)
        return None


def _selfcheck():
    class _Raising:
        @property
        def sysid_state(self):
            raise RuntimeError("boom")
    d = probe(None, None)
    assert d["readable"] and d["ctrl_is_none"] and d["l_mav_is_none"]
    d = probe(None, _Raising())
    assert d["readable"] is False and "boom" in d["error"]
    assert probe_once("/nonexistent/dir/flight_state_ep0001.csv.gz", 0, None, None) is None
    print("selfcheck OK"); return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--selfcheck"]:
        return _selfcheck()
    print("usage: leader_cache_probe.py --selfcheck"); return 3


if __name__ == "__main__":
    sys.exit(main())
