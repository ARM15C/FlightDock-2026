"""Phase 5c.A.6 — entrypoint cadence/command discriminator diagnostic helper.

Captures per-tick / per-step rows in a CSV for paired comparison between:
  - gated subprocess path: BaselineFormationController.run() main loop
  - eval/training path: FormationSITLEnv.step() -> DRLControllerRef.tick_once()

Gated by env var OPTION_C_ENTRYPOINT_DIAG=1. Behaviorally invisible when unset
(returns immediately at the first conditional, no allocations, no I/O).

Output: <OPTION_C_ENTRYPOINT_DIAG_DIR>/entrypoint_diag.csv

Single emit helper used at 7 insertion sites across:
  - tests/baseline_formation_controller.py (R7.1 SCOPED WAIVER)
  - src/drl/drl_controller_ref.py
  - src/drl/env_formation_sitl.py
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA = [
    "wall_t",
    "sim_t",
    "time_boot_ms",
    "env_id",
    "path",
    "tick_source",
    "fsm_state",
    "event_type",
    "step_idx",
    "drain_n",
    "queue_age_ms",
    "ctrl_fused_behind",
    "ctrl_fused_cross",
    "ctrl_fused_below",
    "env_geom_behind",
    "env_geom_cross",
    "env_geom_below",
    "truth_behind",
    "truth_cross",
    "truth_below",
    "range_m",
    "follower_alt",
    "leader_alt",
    "alt_cmd",
    "alt_err",
    # OP-S3 instrumentation amendment: exact belly-clamp
    # binding telemetry — replaces the alt-cmd-delta proxy for readout 1.
    "alt_cmd_pre_belly_clamp",
    "alt_cmd_post_belly_clamp",
    "belly_clamp_active",
    "belly_clamp_limit",
    "below_err",
    "tecs_spdw_target",
    "tecs_spdw_current",
    "speed_cmd",
    "max_closure",
    "vc_gate_reason",
    "spd_arb_reason",
    "command_type",
    "command_seq",
    "command_suppressed",
    "mavlink_send_result",
]

_lock = threading.Lock()
_writer: Optional[csv.DictWriter] = None
_file_handle = None
_initialized = False
_enabled_cache: Optional[bool] = None


def _enabled() -> bool:
    global _enabled_cache
    if _enabled_cache is None:
        _enabled_cache = os.environ.get("OPTION_C_ENTRYPOINT_DIAG", "0") == "1"
    return _enabled_cache


def _ensure_writer() -> bool:
    global _writer, _file_handle, _initialized
    if _initialized:
        return _writer is not None
    with _lock:
        if _initialized:
            return _writer is not None
        _initialized = True
        out_dir_str = os.environ.get("OPTION_C_ENTRYPOINT_DIAG_DIR", "")
        if not out_dir_str:
            out_dir_str = f"/tmp/entrypoint_diag_{os.getpid()}"
        try:
            out_dir = Path(out_dir_str)
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "entrypoint_diag.csv"
            _file_handle = open(csv_path, "a", newline="", buffering=1)
            need_header = csv_path.stat().st_size == 0
            _writer = csv.DictWriter(_file_handle, fieldnames=SCHEMA)
            if need_header:
                _writer.writeheader()
                _file_handle.flush()
            logger.info("[entrypoint_diag] writing to %s", csv_path)
        except Exception as e:
            logger.error("[entrypoint_diag] failed to open writer: %s", e)
            _writer = None
            _file_handle = None
            return False
    return _writer is not None


def emit_event(event_type: str, *, path: str = "unknown", **fields: Any) -> None:
    """Emit a diagnostic row. No-op when OPTION_C_ENTRYPOINT_DIAG is not set.

    All fields are best-effort; missing fields default to empty string.
    Lock-guarded against concurrent emits from multiple Python threads.
    """
    if not _enabled():
        return
    if not _ensure_writer():
        return
    row = {col: fields.get(col, "") for col in SCHEMA}
    row["wall_t"] = f"{time.time():.6f}"
    row["event_type"] = event_type
    row["path"] = path
    with _lock:
        try:
            _writer.writerow(row)
            _file_handle.flush()
        except Exception as e:
            logger.error("[entrypoint_diag] write failed: %s", e)


def is_enabled() -> bool:
    return _enabled()


def resolve_path() -> str:
    """Return the configured 'path' label for this process (gated|eval|unknown)."""
    return os.environ.get("OPTION_C_ENTRYPOINT_DIAG_PATH", "unknown")
