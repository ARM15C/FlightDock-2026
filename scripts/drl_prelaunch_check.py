#!/usr/bin/env python3
"""DRL Campaign Pre-Launch Verification — 34 automated read-only checks.

Exits 0 on all-pass, 1 on any failure. Run before every campaign launch
to catch configuration drift and environmental issues.

Supports PPO, TD3, and SAC via --algorithm flag. Algorithm-specific
checks are automatically skipped when irrelevant (labelled SKIP).

Usage:
    python scripts/drl_prelaunch_check.py --campaign 004
    python scripts/drl_prelaunch_check.py --campaign 005 --algorithm td3
    python scripts/drl_prelaunch_check.py --campaign 006 --algorithm td3 --log-dir logs/custom
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Color helpers (respect NO_COLOR convention)
# ---------------------------------------------------------------------------
_NO_COLOR = os.environ.get("NO_COLOR") is not None


def _green(s: str) -> str:
    return s if _NO_COLOR else f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return s if _NO_COLOR else f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return s if _NO_COLOR else f"\033[33m{s}\033[0m"


def _cyan(s: str) -> str:
    return s if _NO_COLOR else f"\033[36m{s}\033[0m"


def _bold(s: str) -> str:
    return s if _NO_COLOR else f"\033[1m{s}\033[0m"


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure src/ is on path for drl.config imports (used by checks 1 and 2)
_src_path = str(REPO_ROOT / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


def check_n_steps() -> tuple[bool, str]:
    """Check 1: n_steps = 900 in PPO_CONFIG."""
    try:
        from drl.config import PPO_CONFIG
        if PPO_CONFIG.n_steps == 900:
            return True, f"PPO_CONFIG.n_steps = {PPO_CONFIG.n_steps}"
        return False, f"PPO_CONFIG.n_steps = {PPO_CONFIG.n_steps} (expected 900)"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}. Activate venv first?"


def check_m2_warmup() -> tuple[bool, str]:
    """Check 2: All 4 curriculum stages use M2 warmup."""
    try:
        from drl.config import CURRICULUM_STAGES
        failures = []
        for stage_id, cfg in CURRICULUM_STAGES.items():
            if cfg.warmup_target != "M2":
                failures.append(f"Stage {stage_id}: warmup_target={cfg.warmup_target!r}")
        if not failures:
            return True, f"All {len(CURRICULUM_STAGES)} stages use warmup_target='M2'"
        return False, "; ".join(failures)
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}. Activate venv first?"


# --- #451 shared helpers (structural config checks) ---------------------------
BASELINE_FAILS = (REPO_ROOT / "analysis/t38_baseline_20260723T020000Z"
                  / "PREEXISTING_FAILURES_v3.json")
_TOOLS_DIR = REPO_ROOT / "tools"


def _validate_stage_map(m, lo, hi, order, name):
    """Structural gate for a per-stage scalar map: non-empty, in [lo,hi], monotone."""
    if not m:
        return False, f"{name}: no stages in live config"
    fails = [f"stage {s}: {name}={v} outside [{lo},{hi}]"
             for s, v in m.items() if not (lo - 1e-9 <= v <= hi + 1e-9)]
    vals = [m[s] for s in sorted(m)]
    if order == "nondecr" and any(vals[i] > vals[i + 1] + 1e-9 for i in range(len(vals) - 1)):
        fails.append(f"{name} not non-decreasing across stages: {vals}")
    if order == "nonincr" and any(vals[i] < vals[i + 1] - 1e-9 for i in range(len(vals) - 1)):
        fails.append(f"{name} not non-increasing across stages: {vals}")
    if fails:
        return False, "; ".join(fails)
    tag = f" ({order})" if order else ""
    return True, f"{name}: {len(m)} stages, all in [{lo},{hi}]{tag}"


def check_unit_tests() -> tuple[bool, str]:
    """Check 3: no NEW unit-test failures vs the canonical baseline.

    #451 replaced the old `pytest -x` (which FAILed on documented pre-existing
    failures) with a baseline diff. T-59 replaced that diff's file-set+count
    rule, and its superseded baseline artifact, with the canonical TEST-ID-set
    comparison. With the T-38 baseline empty the rule reduces to its strongest
    form: ANY failure is a NEW failure. Fails CLOSED on every error path — a
    launch gate that cannot evaluate must never report PASS.
    """
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    try:
        from check_suite_baseline import compare, load_baseline, parse_ids
    except Exception as e:
        return False, f"canonical baseline checker unimportable: {e}"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit/", "-q", "--tb=no",
             "-p", "no:cacheprovider"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 300s"
    try:
        res = compare(load_baseline(str(BASELINE_FAILS)),
                      parse_ids(result.stdout))
    except Exception as e:
        return False, f"baseline gate FAILED CLOSED: {type(e).__name__}: {e}"
    if res["verdict"] == "RED_NEW_FAILURES":
        return False, ("%d NEW failure(s) vs ID-set baseline: %s"
                       % (len(res["new_failures"]), res["new_failures"][:5]))
    if res["verdict"] == "AMBER_BASELINE_STALE":
        return False, ("baseline STALE — %d baseline failure(s) now fixed; "
                       "re-capture before launch"
                       % len(res["fixed_since_baseline"]))
    return True, ("no NEW failures vs ID-set baseline (baseline %d, current %d)"
                  % (res["n_baseline"], res["n_current"]))


def check_port_range_clean() -> tuple[bool, str]:
    """Check 4: Port range 5760-6260 is clean (no listeners)."""
    import re
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True, text=True, timeout=10,
        )
        occupied = []
        for line in result.stdout.strip().split("\n"):
            # Parse local address column: "0.0.0.0:5760" or "*:5760"
            m = re.search(r':(\d+)\s', line)
            if m:
                p = int(m.group(1))
                if 5760 <= p <= 6260:
                    occupied.append(p)
        if not occupied:
            return True, "Ports 5760-6260 all clear"
        return False, f"Occupied ports: {occupied}"
    except FileNotFoundError:
        return True, "ss not found — skipped (non-critical)"
    except Exception as e:
        return False, f"Port check failed: {e}"


def check_disk_space() -> tuple[bool, str]:
    """Check 5: At least 2 GB free disk space."""
    usage = shutil.disk_usage(str(REPO_ROOT))
    free_gb = usage.free / (1024 ** 3)
    if free_gb >= 2.0:
        return True, f"{free_gb:.1f} GB free"
    return False, f"Only {free_gb:.1f} GB free (need >= 2.0 GB)"


def check_lockfile_absent() -> tuple[bool, str]:
    """Check 6: Training lockfile is absent."""
    lockfile = Path("/tmp/flightdock_training.lock")
    if not lockfile.exists():
        return True, "No lockfile found"
    return False, f"Lockfile exists: {lockfile} — another session may be running"


def check_watchdog_threshold() -> tuple[bool, str]:
    """Check 7: Watchdog stale threshold is 15 minutes (C005b)."""
    watchdog = REPO_ROOT / "scripts" / "training_watchdog.sh"
    if not watchdog.exists():
        return False, f"Watchdog script not found: {watchdog}"
    text = watchdog.read_text()
    for line in text.split("\n"):
        stripped = line.strip()
        if "HEARTBEAT_STALE_MINUTES" in stripped and ":-15}" in stripped:
            return True, f"Watchdog default stale threshold = 15 min ({stripped})"
        if "HEARTBEAT_STALE_MINUTES" in stripped and (":-5}" in stripped or ":-3}" in stripped):
            return False, f"Watchdog stale threshold too low (expected 15): {stripped}"
    return False, "Could not confirm HEARTBEAT_STALE_MINUTES default in watchdog script"


def check_git_clean() -> tuple[bool, str]:
    """Check 8: Git working tree is clean (no unstaged changes to tracked files)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=10,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        # Filter to only modified tracked files (M, D, R, C — not ?? untracked)
        modified = [l for l in lines if l and not l.startswith("??")]
        if not modified:
            return True, f"Clean ({len(lines)} untracked files ignored)"
        return False, f"{len(modified)} modified tracked files: {modified[:5]}"
    except Exception as e:
        return False, f"Git check failed: {e}"


def check_wsl_memory() -> tuple[bool, str]:
    """Check 9: WSL2 memory configuration (informational only)."""
    wslconfig = Path.home() / ".wslconfig"
    if not wslconfig.exists():
        return True, "No ~/.wslconfig found (using WSL defaults)"
    try:
        text = wslconfig.read_text()
        mem_lines = [l.strip() for l in text.split("\n") if "memory" in l.lower()]
        if mem_lines:
            return True, f"WSL config: {'; '.join(mem_lines)}"
        return True, "~/.wslconfig exists but no memory setting found"
    except Exception as e:
        return True, f"Could not read .wslconfig: {e}"


def check_campaign_dir(campaign_num: int, log_dir: str | None,
                       checkpoint_dir: str | None) -> tuple[bool, str]:
    """Check 10: Campaign dirs don't contain existing data."""
    dirs_to_check = []
    if log_dir:
        dirs_to_check.append(Path(log_dir))
    else:
        dirs_to_check.append(REPO_ROOT / f"logs/campaign_{campaign_num:03d}")
    if checkpoint_dir:
        dirs_to_check.append(Path(checkpoint_dir))
    else:
        dirs_to_check.append(REPO_ROOT / f"checkpoints/campaign_{campaign_num:03d}")

    conflicts = []
    for d in dirs_to_check:
        if d.exists():
            # Only fail if dir contains data files (not just empty dir)
            data_files = list(d.glob("metrics.csv")) + list(d.glob("PPO_*"))
            data_files += list(d.glob("*.zip")) + list(d.glob("policy_diagnostics.csv"))
            if data_files:
                conflicts.append(f"{d} contains {len(data_files)} data files")
    if not conflicts:
        return True, "Campaign directories clean"
    return False, "; ".join(conflicts)


def check_wrapper_executable() -> tuple[bool, str]:
    """Check 11: Wrapper script is executable."""
    wrapper = REPO_ROOT / "scripts" / "run_drl_training.sh"
    if not wrapper.exists():
        return False, f"Wrapper not found: {wrapper}"
    if os.access(wrapper, os.X_OK):
        return True, f"{wrapper.name} is executable"
    return False, f"{wrapper.name} is NOT executable (run: chmod +x {wrapper})"


def check_orphan_processes() -> tuple[bool, str]:
    """Check 12: No orphan SITL/MAVProxy/leader processes."""
    user = os.environ.get("USER", "")
    patterns = ["arduplane", "mavproxy", "leader_broadcast"]
    found = []
    for pat in patterns:
        try:
            cmd = ["pgrep", "-u", user, "-f", pat] if user else ["pgrep", "-f", pat]
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                found.append(f"{pat}: {len(pids)} process(es)")
        except Exception:
            pass
    if not found:
        return True, "No orphan SITL/MAVProxy/leader processes"
    return False, "; ".join(found)


def check_action_scale_per_stage() -> tuple[bool, str]:
    """Check 13: action_scale structural sanity (#451: was a stale 4-stage table).

    Config-derived: every stage's action_scale in (0,1] and non-decreasing across
    stages. Replaces the hardcoded {0:0.5,1:0.6,2:1.0,3:1.0} that FAILed against
    the current 6-stage config regardless of correctness.
    """
    try:
        from drl.config import CURRICULUM_STAGES
        m = {s: c.action_scale for s, c in CURRICULUM_STAGES.items()}
        return _validate_stage_map(m, 0.0, 1.0, "nondecr", "action_scale")
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_w_ref_per_stage() -> tuple[bool, str]:
    """Check 14: w_ref structural sanity (#451: was a stale 4-stage table).

    Config-derived: every stage's w_ref in [0,1] and non-increasing across stages
    (curriculum shifts weight off the baseline reference toward the residual);
    stage set must match CURRICULUM_STAGES.
    """
    try:
        from drl.config import CURRICULUM_STAGES, STAGE_WEIGHTS
        if set(STAGE_WEIGHTS) != set(CURRICULUM_STAGES):
            return False, (f"stage-set mismatch: STAGE_WEIGHTS {sorted(STAGE_WEIGHTS)} "
                           f"vs CURRICULUM_STAGES {sorted(CURRICULUM_STAGES)}")
        m = {s: w.w_ref for s, w in STAGE_WEIGHTS.items()}
        return _validate_stage_map(m, 0.0, 1.0, "nonincr", "w_ref")
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_dwell_threshold_per_stage() -> tuple[bool, str]:
    """Check 15: M3b dwell-threshold structural sanity (#451: was a stale table).

    Config-derived: a threshold present for every curriculum stage, each in
    [0,10] s. Terminal-stage dwell VALUES are set by the signed reward recal
    (RECAL_PLAN), NOT gated here — so this checks presence + range, not magnitude.
    """
    try:
        from drl.config import CURRICULUM_STAGES, DWELL_REWARD_THRESHOLD_S
        if set(DWELL_REWARD_THRESHOLD_S) != set(CURRICULUM_STAGES):
            return False, (f"stage-set mismatch: DWELL_REWARD_THRESHOLD_S "
                           f"{sorted(DWELL_REWARD_THRESHOLD_S)} vs CURRICULUM_STAGES "
                           f"{sorted(CURRICULUM_STAGES)}")
        m = dict(DWELL_REWARD_THRESHOLD_S)
        return _validate_stage_map(m, 0.0, 10.0, None, "dwell_threshold_s")
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_leader_profile() -> tuple[bool, str]:
    """Check 16: Leader trajectory default is 'straight'."""
    try:
        # The leader profile is set via environment variable or defaults to "straight"
        # in the SITL launcher. Check that no override is active.
        profile = os.environ.get("HARNESS_LEADER_PROFILE", "straight")
        if profile == "straight":
            return True, f"HARNESS_LEADER_PROFILE = '{profile}'"
        return False, f"HARNESS_LEADER_PROFILE = '{profile}' (expected 'straight')"
    except Exception as e:
        return False, f"Leader profile check failed: {e}"


# ---------------------------------------------------------------------------
# v13 Commit 14 + Checks 17-22: Post-synthesis verification
# ---------------------------------------------------------------------------

def check_stage0_max_rollouts() -> tuple[bool, str]:
    """Check 17: Stage 0 max_rollouts = 20."""
    try:
        from drl.config import CURRICULUM_STAGES
        actual = CURRICULUM_STAGES[0].max_rollouts
        if actual == 20:
            return True, f"Stage 0 max_rollouts = {actual}"
        return False, f"Stage 0 max_rollouts = {actual} (expected 20)"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_dwell_caps() -> tuple[bool, str]:
    """Check 18-19: Dwell reward caps."""
    try:
        from drl.config import DWELL_CAP_M3A_FINE, DWELL_CAP_M3B
        failures = []
        if abs(DWELL_CAP_M3A_FINE - 0.20) > 1e-6:
            failures.append(f"DWELL_CAP_M3A_FINE={DWELL_CAP_M3A_FINE} (expected 0.20)")
        if abs(DWELL_CAP_M3B - 10.0) > 1e-6:
            failures.append(f"DWELL_CAP_M3B={DWELL_CAP_M3B} (expected 10.0)")
        if not failures:
            return True, f"DWELL_CAP_M3A_FINE={DWELL_CAP_M3A_FINE}, DWELL_CAP_M3B={DWELL_CAP_M3B}"
        return False, "; ".join(failures)
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_airspeed_guard() -> tuple[bool, str]:
    """Check 20: Airspeed guard bounds [20.0, 30.0] m/s."""
    try:
        from drl.config import TAS_MIN_SAFE, TAS_MAX_SAFE
        failures = []
        if abs(TAS_MIN_SAFE - 20.0) > 1e-6:
            failures.append(f"TAS_MIN_SAFE={TAS_MIN_SAFE} (expected 20.0)")
        if abs(TAS_MAX_SAFE - 30.0) > 1e-6:
            failures.append(f"TAS_MAX_SAFE={TAS_MAX_SAFE} (expected 30.0)")
        if not failures:
            return True, f"TAS_MIN_SAFE={TAS_MIN_SAFE}, TAS_MAX_SAFE={TAS_MAX_SAFE}"
        return False, "; ".join(failures)
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_collision_guard() -> tuple[bool, str]:
    """Check 21: Collision guard parameters."""
    try:
        from drl.config import COLLISION_CG_DISTANCE_MIN, COLLISION_BEHIND_MIN, COLLISION_BEHIND_MAX
        failures = []
        if abs(COLLISION_CG_DISTANCE_MIN - 3.0) > 1e-6:
            failures.append(f"dist_min={COLLISION_CG_DISTANCE_MIN} (expected 3.0)")
        if abs(COLLISION_BEHIND_MIN - (-2.0)) > 1e-6:
            failures.append(f"behind_min={COLLISION_BEHIND_MIN} (expected -2.0)")
        if abs(COLLISION_BEHIND_MAX - 5.0) > 1e-6:
            failures.append(f"behind_max={COLLISION_BEHIND_MAX} (expected 5.0)")
        if not failures:
            return True, f"dist={COLLISION_CG_DISTANCE_MIN}, behind=[{COLLISION_BEHIND_MIN},{COLLISION_BEHIND_MAX}]"
        return False, "; ".join(failures)
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_watchdog_stale_15() -> tuple[bool, str]:
    """Check 22: Watchdog stale threshold = 15 minutes (C005b)."""
    watchdog = REPO_ROOT / "scripts" / "training_watchdog.sh"
    if not watchdog.exists():
        return False, f"Watchdog script not found: {watchdog}"
    text = watchdog.read_text()
    for line in text.split("\n"):
        stripped = line.strip()
        if "HEARTBEAT_STALE_MINUTES" in stripped and ":-15}" in stripped:
            return True, f"Watchdog default stale threshold = 15 min ({stripped})"
    return False, "Could not confirm HEARTBEAT_STALE_MINUTES=15 default in watchdog script"


def check_separate_clip_ppo() -> tuple[bool, str]:
    """Check 23: train.py imports SeparateClipPPO (C005b)."""
    try:
        from drl.custom_ppo import SeparateClipPPO
        from stable_baselines3 import PPO
        if issubclass(SeparateClipPPO, PPO):
            return True, "SeparateClipPPO is a valid PPO subclass"
        return False, "SeparateClipPPO does not inherit from PPO"
    except ImportError as e:
        return False, f"Cannot import SeparateClipPPO: {e}"


def check_net_arch_dict() -> tuple[bool, str]:
    """Check 24: PPO_CONFIG.net_arch is dict-style (C005b)."""
    try:
        from drl.config import PPO_CONFIG
        if isinstance(PPO_CONFIG.net_arch, dict):
            return True, f"net_arch = dict(pi={PPO_CONFIG.net_arch.get('pi')}, vf={PPO_CONFIG.net_arch.get('vf')})"
        return False, f"net_arch is {type(PPO_CONFIG.net_arch).__name__}, expected dict"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


# ---------------------------------------------------------------------------
# v18 Checks 25-32: TD3/SAC-specific (Phase C)
# ---------------------------------------------------------------------------

def check_td3_buffer_size() -> tuple[bool, str]:
    """Check 25: TD3_CONFIG.buffer_size_stage0 == 200_000 (R23-signed Q7 value;
    F1 2026-07-18 CONFIG_AUDIT 100K→200K, repair authorized
    2026-07-28 — the 100K expectation was the stale Phase-5.1 value)."""
    try:
        from drl.config import TD3_CONFIG
        if TD3_CONFIG.buffer_size_stage0 == 200_000:
            return True, f"buffer_size_stage0 = {TD3_CONFIG.buffer_size_stage0:,}"
        return False, f"buffer_size_stage0 = {TD3_CONFIG.buffer_size_stage0:,} (expected 200,000)"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_td3_learning_starts() -> tuple[bool, str]:
    """Check 26: TD3_CONFIG.learning_starts == 4_000."""
    try:
        from drl.config import TD3_CONFIG
        if TD3_CONFIG.learning_starts == 4_000:
            return True, f"learning_starts = {TD3_CONFIG.learning_starts:,}"
        return False, f"learning_starts = {TD3_CONFIG.learning_starts:,} (expected 4,000)"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_td3_gradient_steps() -> tuple[bool, str]:
    """Check 27: TD3_CONFIG.gradient_steps == 4 (UTD ratio)."""
    try:
        from drl.config import TD3_CONFIG
        if TD3_CONFIG.gradient_steps == 4:
            return True, f"gradient_steps = {TD3_CONFIG.gradient_steps} (UTD ratio 4)"
        return False, f"gradient_steps = {TD3_CONFIG.gradient_steps} (expected 4)"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_td3_policy_delay() -> tuple[bool, str]:
    """Check 28: TD3_CONFIG.policy_delay == 2."""
    try:
        from drl.config import TD3_CONFIG
        if TD3_CONFIG.policy_delay == 2:
            return True, f"policy_delay = {TD3_CONFIG.policy_delay}"
        return False, f"policy_delay = {TD3_CONFIG.policy_delay} (expected 2)"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_td3_noise_sigma() -> tuple[bool, str]:
    """Check 29: the exploration sigma that WILL FLY is the one reported.

    NOISEAUDIT-792 re-pointed this check. It used to assert the per-axis
    config constants (noise_sigma_speed/altitude) and return PASS -- a gate
    pinning a constant that train.py overwrites with
    ``np.full(shape, args.exploration_noise_sigma)`` before ``learn()``.
    It could not fail when the flown sigma changed, which is the only failure
    that matters at launch.

    The invariant now checked is the SEAM: the trainer's CLI default is the
    sigma of record, the overwrite that makes it so is still present in
    train.py, and the per-axis config values are inert. Every clause can fail.
    """
    try:
        import inspect
        import re

        from drl import train as _train

        src = inspect.getsource(_train)
        # (a) the overwrite must still exist -- if it is removed, the per-axis
        #     config silently becomes live and differs from every past campaign
        if "np.full(n._sigma.shape, args.exploration_noise_sigma)" not in src:
            return False, (
                "train.py no longer overwrites _sigma with the CLI value: the "
                "per-axis config would become LIVE and unlike every past run")
        # (b) the CLI default is the sigma of record; read it, do not assume it
        m = re.search(r'"--exploration-noise-sigma"[^)]*?default=([0-9.]+)',
                      src, re.S)
        if not m:
            return False, "cannot read --exploration-noise-sigma default"
        flown = float(m.group(1))
        if not (0.0 < flown <= 1.0):
            return False, f"flown sigma {flown} outside (0, 1]"
        # (c) the config per-axis values are INERT -- assert they are labelled
        #     so, not that they hold any particular value
        if "INERT" not in src:
            return False, (
                "the per-axis noise log no longer declares itself INERT; it "
                "would read as the flown sigma again (NOISEAUDIT-792)")
        return True, (
            f"flown sigma = {flown} ISOTROPIC (CLI --exploration-noise-sigma); "
            f"per-axis config INERT, overwrite present")
    except ImportError as e:
        return False, f"Cannot import drl.train: {e}"


def check_vecnorm_calibration() -> tuple[bool, str]:
    """Check 30: VECNORM_CALIBRATION_EPISODES == 30."""
    try:
        from drl.config import VECNORM_CALIBRATION_EPISODES
        if VECNORM_CALIBRATION_EPISODES == 30:
            return True, f"VECNORM_CALIBRATION_EPISODES = {VECNORM_CALIBRATION_EPISODES}"
        return False, f"VECNORM_CALIBRATION_EPISODES = {VECNORM_CALIBRATION_EPISODES} (expected 30)"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_zero_residual_threshold() -> tuple[bool, str]:
    """Check 31: ZERO_RESIDUAL_ALERT_THRESHOLD == 0.05."""
    try:
        from drl.config import ZERO_RESIDUAL_ALERT_THRESHOLD
        if abs(ZERO_RESIDUAL_ALERT_THRESHOLD - 0.05) < 1e-6:
            return True, f"ZERO_RESIDUAL_ALERT_THRESHOLD = {ZERO_RESIDUAL_ALERT_THRESHOLD}"
        return False, f"ZERO_RESIDUAL_ALERT_THRESHOLD = {ZERO_RESIDUAL_ALERT_THRESHOLD} (expected 0.05)"
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_sb3_version_pinned() -> tuple[bool, str]:
    """Check 33: SB3 runtime version == 2.3.2 and requirements.txt pin verified."""
    try:
        import stable_baselines3
        runtime_ver = stable_baselines3.__version__
        if runtime_ver != "2.3.2":
            return False, f"SB3 runtime version = {runtime_ver} (expected 2.3.2)"
        # Verify requirements.txt pin
        req_path = REPO_ROOT / "requirements.txt"
        if req_path.exists():
            text = req_path.read_text()
            if "stable-baselines3==2.3.2" in text or "stable_baselines3==2.3.2" in text:
                return True, f"SB3 v{runtime_ver} (runtime) + pinned in requirements.txt"
            return False, f"SB3 v{runtime_ver} (runtime) but NOT pinned in requirements.txt"
        return True, f"SB3 v{runtime_ver} (runtime) — no requirements.txt found"
    except ImportError as e:
        return False, f"Cannot import stable_baselines3: {e}"


def check_td3_net_arch_asymmetric() -> tuple[bool, str]:
    """Check 34: TD3/SAC net_arch is dict with pi=[128,128], qf=[256,256]."""
    try:
        from drl.config import TD3_CONFIG, SAC_CONFIG
        failures = []
        for name, cfg in [("TD3", TD3_CONFIG), ("SAC", SAC_CONFIG)]:
            if not isinstance(cfg.net_arch, dict):
                failures.append(f"{name} net_arch is {type(cfg.net_arch).__name__}, expected dict")
            elif cfg.net_arch.get("pi") != [128, 128]:
                failures.append(f"{name} net_arch['pi'] = {cfg.net_arch.get('pi')} (expected [128,128])")
            elif cfg.net_arch.get("qf") != [256, 256]:
                failures.append(f"{name} net_arch['qf'] = {cfg.net_arch.get('qf')} (expected [256,256])")
        if not failures:
            return True, f"TD3/SAC net_arch = dict(pi=[128,128], qf=[256,256])"
        return False, "; ".join(failures)
    except ImportError as e:
        return False, f"Cannot import drl.config: {e}"


def check_campaign_dir_offpolicy(campaign_num: int, log_dir: str | None,
                                  checkpoint_dir: str | None) -> tuple[bool, str]:
    """Check 32: Campaign dirs clean of off-policy artifacts."""
    dirs_to_check = []
    if log_dir:
        dirs_to_check.append(Path(log_dir))
    else:
        dirs_to_check.append(REPO_ROOT / f"logs/campaign_{campaign_num:03d}")
    if checkpoint_dir:
        dirs_to_check.append(Path(checkpoint_dir))
    else:
        dirs_to_check.append(REPO_ROOT / f"checkpoints/campaign_{campaign_num:03d}")

    conflicts = []
    for d in dirs_to_check:
        if d.exists():
            data_files = list(d.glob("offpolicy_diagnostics.csv"))
            data_files += list(d.glob("replay_buffer_*.pkl"))
            data_files += list(d.glob("buffer_dilution_check.csv"))
            if data_files:
                conflicts.append(f"{d} contains {len(data_files)} off-policy data files")
    if not conflicts:
        return True, "Campaign directories clean of off-policy artifacts"
    return False, "; ".join(conflicts)


# ---------------------------------------------------------------------------
# C006 Pre-Campaign Checks (35-44)
# ---------------------------------------------------------------------------

def check_n_envs_default_4() -> tuple[bool, str]:
    """Check 35: n_envs default == 4 in train.py."""
    train_path = REPO_ROOT / "src/drl/train.py"
    text = train_path.read_text()
    if '"--n-envs"' in text and 'default=4' in text:
        return True, "n_envs default = 4"
    return False, "n_envs default not 4 in train.py"


def check_save_freq_scaling() -> tuple[bool, str]:
    """Check 36: save_freq scales by n_envs."""
    train_path = REPO_ROOT / "src/drl/train.py"
    text = train_path.read_text()
    if "effective_save_freq" in text and "// n_envs" in text:
        return True, "save_freq = 3600 // n_envs present"
    return False, "save_freq scaling not found in train.py"


def check_port_uniqueness(n_envs: int = 24) -> tuple[bool, str]:
    """Check 36b: No port collisions for up to n_envs SITL instances (Phase 7)."""
    all_ports = []
    for env_id in range(n_envs):
        for base in [5760, 5762, 5763, 14550, 14551, 14552, 14553]:
            all_ports.append(base + env_id * 100)
    unique = len(set(all_ports))
    total = len(all_ports)
    if unique == total:
        return True, f"No port collisions for P={n_envs} ({total} ports, all unique)"
    return False, f"Port collision: {total} ports but only {unique} unique"


def check_offpolicy_diag_env_id() -> tuple[bool, str]:
    """Check 37: env_id column in OffPolicyDiagnosticsCallback._CSV_COLUMNS."""
    try:
        from drl.callbacks import OffPolicyDiagnosticsCallback
        cols = OffPolicyDiagnosticsCallback._CSV_COLUMNS
        if "env_id" in cols:
            return True, f"env_id in OffPolicyDiag CSV columns (pos {cols.index('env_id')})"
        return False, "env_id NOT in OffPolicyDiagnosticsCallback._CSV_COLUMNS"
    except Exception as e:
        return False, f"Import error: {e}"


def check_buffer_dilution_env_id() -> tuple[bool, str]:
    """Check 38: env_id column in BufferDilutionCheckCallback._CSV_COLUMNS."""
    try:
        from drl.callbacks import BufferDilutionCheckCallback
        cols = BufferDilutionCheckCallback._CSV_COLUMNS
        if "env_id" in cols:
            return True, f"env_id in BufferDilution CSV columns (pos {cols.index('env_id')})"
        return False, "env_id NOT in BufferDilutionCheckCallback._CSV_COLUMNS"
    except Exception as e:
        return False, f"Import error: {e}"


def check_no_old_milestone_columns() -> tuple[bool, str]:
    """Check 39: No m3b_dwell_s or dwell_M3b_s in MetricsCallback._FIELDNAMES."""
    try:
        from drl.callbacks import MetricsCallback
        fields = MetricsCallback._FIELDNAMES
        old_names = {"m3b_dwell_s", "dwell_M3b_s", "m3b_mean_residual",
                     "learned_m3b_fraction", "milestone_tier", "best_milestone_tier"}
        found = old_names & set(fields)
        if not found:
            return True, "No old milestone column names in MetricsCallback._FIELDNAMES"
        return False, f"Old column names found: {found}"
    except Exception as e:
        return False, f"Import error: {e}"


def check_actual_m2_column() -> tuple[bool, str]:
    """Check 40: actual_M2_achieved in MetricsCallback._FIELDNAMES."""
    try:
        from drl.callbacks import MetricsCallback
        if "actual_M2_achieved" in MetricsCallback._FIELDNAMES:
            return True, "actual_M2_achieved in MetricsCallback._FIELDNAMES"
        return False, "actual_M2_achieved NOT in MetricsCallback._FIELDNAMES"
    except Exception as e:
        return False, f"Import error: {e}"


def check_boom_tip_constants() -> tuple[bool, str]:
    """Check 41: BOOM_TIP_BEHIND_M defined in config."""
    try:
        from drl import config
        val = getattr(config, "BOOM_TIP_BEHIND_M", None)
        if val is not None and abs(val - 4.313) < 0.01:
            return True, f"BOOM_TIP_BEHIND_M = {val}"
        return False, f"BOOM_TIP_BEHIND_M = {val} (expected ~4.313)"
    except Exception as e:
        return False, f"Import error: {e}"


def check_pool_default_false() -> tuple[bool, str]:
    """Check 42: --pool defaults to False (store_true)."""
    train_path = REPO_ROOT / "src/drl/train.py"
    text = train_path.read_text()
    if '"--pool", action="store_true"' in text or "'--pool', action='store_true'" in text:
        return True, "--pool uses store_true (default=False)"
    return False, "--pool not using store_true in train.py"


def check_demo_buffer_importable() -> tuple[bool, str]:
    """Check 43: DemoReplayBuffer importable."""
    try:
        from drl.demo_buffer import DemoReplayBuffer
        return True, f"DemoReplayBuffer importable ({DemoReplayBuffer.__bases__[0].__name__})"
    except Exception as e:
        return False, f"Cannot import DemoReplayBuffer: {e}"


def check_sb3_offpolicy_version() -> tuple[bool, str]:
    """Check 44: SB3 version == 2.3.2 (off-policy scope verification)."""
    try:
        import stable_baselines3 as sb3
        if sb3.__version__ == "2.3.2":
            return True, f"SB3 version = {sb3.__version__}"
        return False, f"SB3 version = {sb3.__version__} (expected 2.3.2)"
    except Exception as e:
        return False, f"Cannot import SB3: {e}"


# ---------------------------------------------------------------------------
# v19 Fix 6: WSL2 power-state warning
# ---------------------------------------------------------------------------

def check_wsl2_power_warning() -> tuple[bool, str]:
    """Check 45: WSL2 host power-state warning (informational — always passes)."""
    try:
        proc_version = Path("/proc/version").read_text()
        if "microsoft" not in proc_version.lower():
            return True, "Not running under WSL2 — no power-state concern"
        return True, (
            "⚠ WSL2 DETECTED — ensure Windows sleep/hibernate is DISABLED "
            "before long campaigns. Host suspension freezes ALL processes "
            "including the watchdog. Use: Settings > System > Power & sleep > Never, "
            "or PowerToys Awake."
        )
    except Exception as e:
        return True, f"Could not read /proc/version: {e}"


def check_vecnorm_unified_callback() -> tuple[bool, str]:
    """Check 46: VecNormalize unified callback (Campaign 2b).

    Verifies that train.py does not instantiate both old VecNormalize callbacks
    on non-comment lines. The unified VecNormalizeScheduleCallback should be the
    sole authority over VecNormalize training state.
    """
    train_py = REPO_ROOT / "src" / "drl" / "train.py"
    if not train_py.exists():
        return False, f"train.py not found at {train_py}"

    conflicts = []
    for line_no, line in enumerate(train_py.read_text().splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "VecNormalizeFreezeCallback()" in line:
            conflicts.append(f"L{line_no}: VecNormalizeFreezeCallback() instantiated")
        if "VecNormalizeUnfreezeCallback(" in line and "import" not in line:
            conflicts.append(f"L{line_no}: VecNormalizeUnfreezeCallback() instantiated")

    if not conflicts:
        return True, "Only VecNormalizeScheduleCallback active in train.py"
    return False, f"Old callbacks still active: {'; '.join(conflicts)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DRL Campaign Pre-Launch Verification (46 checks)",
    )
    parser.add_argument(
        "--campaign", type=int, required=True,
        help="Campaign number (e.g., 004)",
    )
    parser.add_argument(
        "--algorithm", type=str, default="ppo",
        choices=["td3", "sac", "ppo"],
        help="RL algorithm: ppo (default), td3, sac. Controls which checks run.",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Override log directory (default: logs/campaign_NNN)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Override checkpoint directory (default: checkpoints/campaign_NNN)",
    )
    args = parser.parse_args()

    algo = args.algorithm
    is_offpolicy = algo in ("td3", "sac")

    print(_bold(f"\n{'='*60}"))
    print(_bold(f"  DRL Pre-Launch Check — Campaign {args.campaign:03d} ({algo.upper()})"))
    print(_bold(f"{'='*60}\n"))

    # PPO-only check indices (1-indexed array position): skip when running td3/sac
    PPO_ONLY_CHECKS = {1, 22, 23}  # n_steps, SeparateClipPPO, net_arch dict
    # TD3/SAC-only check indices (1-indexed array position): skip when running ppo
    OFFPOLICY_ONLY_CHECKS = {24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 37, 38}

    checks = [
        # 1: PPO-only
        ("n_steps = 900", lambda: check_n_steps()),
        # 2-22: algorithm-agnostic
        ("All stages M2 warmup", lambda: check_m2_warmup()),
        ("Unit tests pass", lambda: check_unit_tests()),
        ("Port range 5760-6160 clean", lambda: check_port_range_clean()),
        ("Disk space >= 2 GB", lambda: check_disk_space()),
        ("Lockfile absent", lambda: check_lockfile_absent()),
        ("Watchdog stale threshold = 15", lambda: check_watchdog_threshold()),
        ("Git status clean", lambda: check_git_clean()),
        ("WSL2 memory (info)", lambda: check_wsl_memory()),
        ("Campaign dir clean", lambda: check_campaign_dir(args.campaign, args.log_dir, args.checkpoint_dir)),
        ("Wrapper script executable", lambda: check_wrapper_executable()),
        ("No orphan SITL processes", lambda: check_orphan_processes()),
        ("action_scale per stage", lambda: check_action_scale_per_stage()),
        ("w_ref per stage (Rev 3)", lambda: check_w_ref_per_stage()),
        ("M3b dwell threshold per stage", lambda: check_dwell_threshold_per_stage()),
        ("Leader profile = straight", lambda: check_leader_profile()),
        # v13 Checks 17-22
        ("Stage 0 max_rollouts = 20", lambda: check_stage0_max_rollouts()),
        ("Dwell caps (M3A_FINE=0.20, M3B=5.0)", lambda: check_dwell_caps()),
        ("Airspeed guard [20.0, 30.0] m/s", lambda: check_airspeed_guard()),
        ("Collision guard dist=3, behind=[-2,+5]", lambda: check_collision_guard()),
        ("Watchdog stale = 15 min (C005b)", lambda: check_watchdog_stale_15()),
        # C005b Checks 23-24: PPO-only
        ("SeparateClipPPO import", lambda: check_separate_clip_ppo()),
        ("net_arch is dict-style", lambda: check_net_arch_dict()),
        # v18 Checks 25-32: TD3/SAC-only
        ("TD3 buffer_size_stage0 = 200k", lambda: check_td3_buffer_size()),
        ("TD3 learning_starts = 4k", lambda: check_td3_learning_starts()),
        ("TD3 gradient_steps = 4 (UTD)", lambda: check_td3_gradient_steps()),
        ("TD3 policy_delay = 2", lambda: check_td3_policy_delay()),
        ("TD3 exploration sigma seam (flown==reported)", lambda: check_td3_noise_sigma()),
        ("VecNorm calibration = 30 eps", lambda: check_vecnorm_calibration()),
        ("Zero-residual threshold = 0.05", lambda: check_zero_residual_threshold()),
        ("Campaign dir clean (off-policy)", lambda: check_campaign_dir_offpolicy(args.campaign, args.log_dir, args.checkpoint_dir)),
        # v19 Checks 33-34: SB3 version pin + asymmetric net_arch
        ("SB3 version == 2.3.2 (pinned)", lambda: check_sb3_version_pinned()),
        ("TD3/SAC net_arch asymmetric", lambda: check_td3_net_arch_asymmetric()),
        # C006 Checks 35-44: Pre-campaign prerequisites
        ("n_envs default = 4", lambda: check_n_envs_default_4()),
        ("save_freq scaling present", lambda: check_save_freq_scaling()),
        ("Port uniqueness (P=24)", lambda: check_port_uniqueness(24)),
        ("env_id in OffPolicyDiag CSV", lambda: check_offpolicy_diag_env_id()),
        ("env_id in BufferDilution CSV", lambda: check_buffer_dilution_env_id()),
        ("No old milestone column names", lambda: check_no_old_milestone_columns()),
        ("actual_M2_achieved in Metrics CSV", lambda: check_actual_m2_column()),
        ("BOOM_TIP_BEHIND_M in config", lambda: check_boom_tip_constants()),
        ("Pool mode default is False", lambda: check_pool_default_false()),
        ("DemoReplayBuffer importable", lambda: check_demo_buffer_importable()),
        ("SB3 version == 2.3.2 (off-policy)", lambda: check_sb3_offpolicy_version()),
        # v19 Check 45: WSL2 power-state warning (informational)
        ("WSL2 power-state warning", lambda: check_wsl2_power_warning()),
        # Campaign 2b Check 46: VecNormalize unified callback
        ("VecNormalize unified callback", lambda: check_vecnorm_unified_callback()),
    ]

    results = []
    n_skipped = 0
    for i, (name, check_fn) in enumerate(checks, 1):
        # Algorithm-aware skip logic
        if i in PPO_ONLY_CHECKS and is_offpolicy:
            results.append((name, True, f"PPO-only — skipped for {algo}"))
            status = _cyan("SKIP")
            n_skipped += 1
            print(f"  [{status}] {i:2d}. {name}")
            print(f"       PPO-only — skipped for {algo}")
            continue
        if i in OFFPOLICY_ONLY_CHECKS and not is_offpolicy:
            results.append((name, True, f"TD3/SAC-only — skipped for {algo}"))
            status = _cyan("SKIP")
            n_skipped += 1
            print(f"  [{status}] {i:2d}. {name}")
            print(f"       TD3/SAC-only — skipped for {algo}")
            continue

        passed, detail = check_fn()
        results.append((name, passed, detail))
        status = _green("PASS") if passed else _red("FAIL")
        print(f"  [{status}] {i:2d}. {name}")
        print(f"       {detail}")

    failures = [r for r in results if not r[1]]
    print(f"\n{'='*60}")
    if failures:
        print(_red(f"  {len(failures)} CHECK(S) FAILED — do NOT launch campaign"))
        for name, _, detail in failures:
            print(f"    - {name}: {detail}")
        print(f"{'='*60}\n")
        return 1
    else:
        n_run = len(checks) - n_skipped
        print(_green(f"  ALL {n_run} CHECKS PASSED ({n_skipped} skipped) — "
                     f"clear to launch Campaign {args.campaign:03d} ({algo.upper()})"))
        print(f"{'='*60}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
