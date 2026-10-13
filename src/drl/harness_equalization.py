"""Phase 5c.A.5 — eval/training harness equalization preflight.

Brings the FormationSITLEnv launch path to gated_matrix preflight parity by
running PRH-ATT-002 attitude-rate verification and the timebase smoketest as
subprocesses against the env's MAVProxy fan-out, in the window AFTER the
leader broadcaster is up but BEFORE _run_warmup constructs DRLControllerRef
(which would bind udp:14551+offset and EADDRINUSE-collide with the
preflights).

Invocation cadence: once per FormationSITLEnv.reset(), gated by
FLIGHTDOCK_HARNESS_EQUALIZATION env var.

CLI signatures mirror tests/lib_run_harness.sh:
  - tests/attitude_rate_preflight.py (lib:1216-1227, validation mode)
  - tools/timebase_smoketest.py (lib:1265-1270)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
ATTITUDE_SCRIPT = _REPO_ROOT / "tests" / "attitude_rate_preflight.py"
TIMEBASE_SCRIPT = _REPO_ROOT / "tools" / "timebase_smoketest.py"

ATTITUDE_TIMEOUT_S = 90
TIMEBASE_TIMEOUT_S = 60
TIMEBASE_DURATION_S = 10.0


def _resolve_artifact_dir(out_dir: Optional[str], ep_idx: Optional[int],
                          env_id: int) -> Path:
    if out_dir:
        base = Path(out_dir)
        if ep_idx is not None and base.name and not base.name.startswith("episode_"):
            base = base / f"episode_{ep_idx:03d}_env{env_id}"
        target = base / "equalization"
    else:
        target = Path(f"/tmp/flightdock_harness_equalization_{os.getpid()}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _run_attitude_rate(conn: str, speedup: float, artifact_dir: Path) -> dict:
    report_path = artifact_dir / "attitude_rate_preflight.json"
    stdout_path = artifact_dir / "attitude_rate_preflight.stdout.log"
    cmd = [
        sys.executable, str(ATTITUDE_SCRIPT),
        "--conn", conn,
        "--speedup", str(speedup),
        "--settle-sim-s", "5",
        "--measure-sim-s", "30",
        "--min-hz", "18",
        "--max-dt-p95-s", "0.08",
        "--stale-gap-thresh-s", "0.2",
        "--window-sim-s", "5",
        "--mode", "validation",
        "--report", str(report_path),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, timeout=ATTITUDE_TIMEOUT_S,
            capture_output=True, text=True,
        )
        rc = proc.returncode
        stdout_path.write_text(
            f"=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n"
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(
            f"=== TIMEOUT after {ATTITUDE_TIMEOUT_S}s ===\n"
            f"stdout: {exc.stdout}\nstderr: {exc.stderr}\n"
        )
        rc = -1

    elapsed = time.time() - t0
    report_data = None
    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report_data = {"parse_error": True}

    if rc == 0:
        verdict = "PASS"
    elif rc == 2:
        verdict = "LINK_DEAD"
    elif rc == -1:
        verdict = "TIMEOUT"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "exit_code": rc,
        "elapsed_s": round(elapsed, 2),
        "report_path": str(report_path),
        "stdout_path": str(stdout_path),
        "report_data": report_data,
    }


def _run_timebase(conn: str, speedup: float, artifact_dir: Path) -> dict:
    log_path = artifact_dir / f"timebase_smoketest_{speedup}x.txt"
    cmd = [
        sys.executable, str(TIMEBASE_SCRIPT),
        "--conn", conn,
        "--expected-speedup", str(speedup),
        "--duration", str(TIMEBASE_DURATION_S),
        "--sysid", "1",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, timeout=TIMEBASE_TIMEOUT_S,
            capture_output=True, text=True,
        )
        rc = proc.returncode
        log_path.write_text(
            f"=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n"
        )
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(
            f"=== TIMEOUT after {TIMEBASE_TIMEOUT_S}s ===\n"
            f"stdout: {exc.stdout}\nstderr: {exc.stderr}\n"
        )
        rc = -1

    elapsed = time.time() - t0
    log_text = log_path.read_text() if log_path.exists() else ""
    verdict_pass = (rc == 0) and ("VERDICT: PASS" in log_text)

    if verdict_pass:
        verdict = "PASS"
    elif rc == -1:
        verdict = "TIMEOUT"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "exit_code": rc,
        "elapsed_s": round(elapsed, 2),
        "log_path": str(log_path),
    }


def run_preflight(*, follower_port: int, speedup: float,
                  out_dir: Optional[str], ep_idx: Optional[int],
                  env_id: int) -> dict:
    """Run gated_matrix-equivalent preflight against the env's MAVProxy fan-out.

    Raises OSError on FAIL/LINK_DEAD/TIMEOUT so the caller's
    MAX_WARMUP_RETRIES retry loop catches it.

    Bypass: FLIGHTDOCK_HARNESS_EQUALIZATION="0" returns BYPASSED immediately.
    """
    flag = os.environ.get("FLIGHTDOCK_HARNESS_EQUALIZATION", "0")
    if flag == "0":
        return {"verdict": "BYPASSED", "reason": "FLIGHTDOCK_HARNESS_EQUALIZATION=0"}

    artifact_dir = _resolve_artifact_dir(out_dir, ep_idx, env_id)
    summary_path = artifact_dir / "harness_equalization_summary.json"
    conn = f"udp:127.0.0.1:{follower_port}"

    logger.info(
        "[harness_equalization] env_id=%s conn=%s speedup=%s artifact_dir=%s",
        env_id, conn, speedup, artifact_dir,
    )

    summary = {
        "env_id": env_id,
        "follower_port": follower_port,
        "speedup": speedup,
        "conn": conn,
        "artifact_dir": str(artifact_dir),
        "ep_idx": ep_idx,
        "timestamp": time.time(),
    }

    attitude = _run_attitude_rate(conn, speedup, artifact_dir)
    summary["attitude"] = attitude
    if attitude["verdict"] != "PASS":
        summary["verdict"] = "FAIL"
        summary["fail_stage"] = "attitude"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.error(
            "[harness_equalization] ATTITUDE preflight %s (rc=%s, %.1fs); "
            "raising OSError to caller's retry loop",
            attitude["verdict"], attitude["exit_code"], attitude["elapsed_s"],
        )
        raise OSError(
            f"harness_equalization ATTITUDE {attitude['verdict']} "
            f"(rc={attitude['exit_code']}, env_id={env_id}); "
            f"see {attitude['stdout_path']}"
        )

    timebase = _run_timebase(conn, speedup, artifact_dir)
    summary["timebase"] = timebase
    if timebase["verdict"] != "PASS":
        summary["verdict"] = "FAIL"
        summary["fail_stage"] = "timebase"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.error(
            "[harness_equalization] TIMEBASE preflight %s (rc=%s, %.1fs); "
            "raising OSError to caller's retry loop",
            timebase["verdict"], timebase["exit_code"], timebase["elapsed_s"],
        )
        raise OSError(
            f"harness_equalization TIMEBASE {timebase['verdict']} "
            f"(rc={timebase['exit_code']}, env_id={env_id}); "
            f"see {timebase['log_path']}"
        )

    summary["verdict"] = "PASS"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "[harness_equalization] PASS env_id=%s attitude=%.1fs timebase=%.1fs "
        "summary=%s",
        env_id, attitude["elapsed_s"], timebase["elapsed_s"], summary_path,
    )
    return summary
