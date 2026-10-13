#!/usr/bin/env python3
"""FlightDock DRL Training Entry Point.

CLI for launching TD3 (default), SAC, or PPO training with VecNormalize.

Algorithms:
  - TD3 (default): Off-policy, deterministic policy + replay buffer.
    5-20x sample efficiency vs PPO for expensive SITL episodes.
  - SAC: Off-policy stochastic fallback with automatic entropy tuning.
  - PPO: On-policy (via SeparateClipPPO) — preserved for defense comparison.

Warmup optimization approaches (enabled for all algorithms):
  - Approach G: Zero-residual policy init (near-zero output)
  - Approach F: Stage-dependent safety ranges (wider early, tighter late)
  - Approach D: Pre-warmed SITL pool (--pool flag, background warmup)

Usage:
    # TD3 (default — off-policy pivot):
    python src/drl/train.py \\
        --algorithm td3 \\
        --total-timesteps 1728000 \\
        --n-envs 3 \\
        --speedup 2.0 \\
        --checkpoint-dir checkpoints/campaign_006 \\
        --log-dir logs/campaign_006

    # PPO (defense comparison):
    python src/drl/train.py \\
        --algorithm ppo \\
        --total-timesteps 1728000 \\
        --n-envs 4 \\
        --speedup 2.0 \\
        --checkpoint-dir checkpoints/ppo_baseline \\
        --log-dir logs/ppo_baseline
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import json
import logging
import multiprocessing
import os
import signal as _signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Ensure src/ and tests/ are on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for d in ("src", "tests"):
    p = os.path.join(_REPO_ROOT, d)
    if p not in sys.path:
        sys.path.insert(0, p)

from stable_baselines3 import TD3, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from drl.custom_ppo import SeparateClipPPO as PPO  # C005b: separate actor/critic gradient clipping
from drl.custom_td3 import CustomClippedTD3, LayerNormTD3Policy  # C006 Phase 3/5.1

from drl import config
from drl.callbacks import (
    ActorFreezeCallback,
    AtomicOffPolicyCheckpointCallback,
    BehavioralEarlyStopCallback,
    BufferDilutionCheckCallback,
    CheckpointCallback,
    ConfigVerificationCallback,
    CriticDivergenceAbortCallback,
    H12ConjunctionCallback,
    StructuralCeilingCallback,
    NoiseAnnealCallback,
    PassthroughMonitorCallback,
    CurriculumCallback,
    HeartbeatCallback,
    MetricsCallback,
    OffPolicyDiagnosticsCallback,
    PolicyDiagnosticsCallback,
    QValueDistributionCallback,
    RewardMonitorCallback,
    CheckpointGatedRollbackCallback,
    TrainingHealthCallback,
    VecNormalizeScheduleCallback,
    ZeroResidualFallbackCallback,
)
from drl.curriculum import CurriculumManager
from drl.env_formation_sitl import FormationSITLEnv


# ---------------------------------------------------------------------------
# P1-b (FD_CURRICULUM_V3): A5 atomic training-state snapshot
# ---------------------------------------------------------------------------

class AtomicSnapshotter:
    """A5: atomic snapshot of the training state at a v3 gate freeze.

    Captures the model archive (actor/critics/targets/optimizers via
    model.save), VecNormalize statistics, RNG states (python/numpy/torch),
    the curriculum sampler cursor, and the replay-buffer IDENTITY (id/size/
    pos/full — the buffer's contents are checkpointed by the existing
    AtomicOffPolicyCheckpointCallback, never duplicated here), plus a sha256
    manifest of every artifact. Written to <root>/.<tag>.tmp then atomically
    renamed to <root>/<tag> — a crashed save never leaves a plausible-looking
    partial snapshot.
    """

    def __init__(self, root: str, curriculum=None) -> None:
        self._root = root
        self._curriculum = curriculum

    @staticmethod
    def _sha256(path: str) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def save(self, tag: str, model, curriculum=None) -> str:
        import pickle
        import random as _random
        import shutil
        import torch as th
        cur = curriculum if curriculum is not None else self._curriculum
        tmp = os.path.join(self._root, f".{tag}.tmp")
        final = os.path.join(self._root, tag)
        for p in (tmp, final):
            if os.path.isdir(p):
                shutil.rmtree(p)
        os.makedirs(tmp, exist_ok=True)
        model.save(os.path.join(tmp, "model.zip"))
        vn = model.get_vec_normalize_env()
        if vn is not None:
            vn.save(os.path.join(tmp, "vecnormalize.pkl"))
        with open(os.path.join(tmp, "rng_state.pkl"), "wb") as fh:
            pickle.dump({"python": _random.getstate(),
                         "numpy": np.random.get_state(),
                         "torch": th.get_rng_state()}, fh)
        cur_state = (cur.get_v3_sampler_state()
                     if cur is not None and hasattr(cur, "get_v3_sampler_state")
                     else {})
        with open(os.path.join(tmp, "curriculum_state.json"), "w") as fh:
            json.dump(cur_state, fh, indent=1)
        rb = getattr(model, "replay_buffer", None)
        replay_meta = ({"id": id(rb), "buffer_size": int(rb.buffer_size),
                        "pos": int(rb.pos), "full": bool(rb.full)}
                       if rb is not None else None)
        with open(os.path.join(tmp, "replay_meta.json"), "w") as fh:
            json.dump(replay_meta, fh, indent=1)
        manifest = {"tag": tag,
                    "num_timesteps": int(model.num_timesteps),
                    "files": {}}
        for name in sorted(os.listdir(tmp)):
            manifest["files"][name] = self._sha256(os.path.join(tmp, name))
        with open(os.path.join(tmp, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        os.replace(tmp, final)
        return final

    def restore(self, tag: str, model, curriculum=None) -> dict:
        """Restore RNG states, VecNormalize statistics, and the curriculum
        sampler cursor from a snapshot onto LIVE objects (the model archive
        itself is loaded by the normal resume path). Returns the manifest."""
        import pickle
        import random as _random
        import torch as th
        snap = os.path.join(self._root, tag)
        with open(os.path.join(snap, "rng_state.pkl"), "rb") as fh:
            rng = pickle.load(fh)
        _random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        th.set_rng_state(rng["torch"])
        vn = model.get_vec_normalize_env() if model is not None else None
        vn_path = os.path.join(snap, "vecnormalize.pkl")
        if vn is not None and os.path.exists(vn_path):
            with open(vn_path, "rb") as fh:
                loaded = pickle.load(fh)
            vn.obs_rms = loaded.obs_rms
            vn.ret_rms = loaded.ret_rms
            vn.training = loaded.training
        cur = curriculum if curriculum is not None else self._curriculum
        if cur is not None and hasattr(cur, "set_v3_sampler_state"):
            with open(os.path.join(snap, "curriculum_state.json")) as fh:
                state = json.load(fh)
            if state:
                cur.set_v3_sampler_state(state)
        with open(os.path.join(snap, "manifest.json")) as fh:
            return json.load(fh)


# ---------------------------------------------------------------------------
# Process cleanup utilities (v9 reliability layers 2/3)
# ---------------------------------------------------------------------------

# Module-level state for signal handler and atexit
_vec_env_ref = None  # Set after vec_env creation; used by signal handler
_n_envs_for_cleanup = None  # Set after argparse; used by atexit handler
_PID_FILE = Path(config.PID_FILE_PATH)
_HEARTBEAT_FILE = Path(config.HEARTBEAT_FILE_PATH)


def _kill_all_sitl_by_port(n_envs: int, port_stride: int = config.PORT_STRIDE) -> int:
    """Kill any SITL/MAVProxy/leader processes matching our port range.

    Returns the number of pkill invocations that found a match.
    """
    killed = 0
    # UID-scoped pkill — only kill processes owned by the current user (v11).
    user = os.environ.get("USER", "")
    user_args = ["-u", user] if user else []
    for env_id in range(n_envs):
        base_port = config.SITL_SERIAL_PORT_BASE + env_id * port_stride
        leader_port = config.PORT_BASE_LEADER + env_id * port_stride
        patterns = [
            f"serial0=tcp:{base_port}",                   # ArduPlane SITL
            f"tcp:127.0.0.1:{base_port}",                 # MAVProxy
            f"leader_broadcast_fixed.py.*{leader_port}",   # leader broadcast
        ]
        for pat in patterns:
            try:
                result = subprocess.run(
                    ["pkill", "-9"] + user_args + ["-f", pat],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                if result.returncode == 0:
                    killed += 1
                    logger.info(f"[Cleanup] Killed process matching: {pat}")
            except Exception:
                logger.debug("Failed to pkill pattern %s", pat, exc_info=True)
    # Kill stale forkserver processes from prior crashed runs (v10 Change 3).
    try:
        result = subprocess.run(
            ["pkill", "-9"] + user_args + ["-f", "from multiprocessing.forkserver import main"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
        if result.returncode == 0:
            killed += 1
            logger.info("[Cleanup] Killed Python forkserver process")
    except Exception:
        logger.debug("Failed to pkill forkserver process", exc_info=True)
    return killed


def _close_with_timeout(vec_env, timeout: float = config.VECENV_CLOSE_TIMEOUT_S) -> None:
    """Close vec_env with a hard timeout to prevent indefinite hangs.

    SB3's SubprocVecEnv.close() uses process.join() with no timeout.
    If a worker is alive but stuck, join() blocks forever. This wrapper
    runs close() in a daemon thread and falls back to SIGKILL if it
    times out.

    Cross-model review: 4/4 models identified this as the single most
    critical gap in the original plan.
    """
    thread = threading.Thread(target=vec_env.close, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        logger.warning(f"[Cleanup] vec_env.close() hung for {timeout}s — "
                       f"killing workers directly")
        # Unwrap all VecEnvWrapper layers to reach SubprocVecEnv.processes
        inner = vec_env
        while hasattr(inner, 'venv'):
            inner = inner.venv
        for proc in getattr(inner, 'processes', []):
            try:
                proc.kill()  # SIGKILL — immediate
            except Exception:
                logger.debug("Failed to kill worker process", exc_info=True)
        # Do NOT try to join again — the close thread may be stuck on a
        # pipe operation. daemon=True ensures it dies with the process.


def _atexit_cleanup():
    """Last-resort cleanup — runs on normal exit, sys.exit(), unhandled exceptions.

    Does NOT fire on SIGKILL or os._exit().
    """
    # Gate to MainProcess only. Forked SubprocVecEnv workers inherit
    # atexit handlers, but only the main process should run the port sweep.
    if multiprocessing.current_process().name != 'MainProcess':
        return
    _PID_FILE.unlink(missing_ok=True)
    _HEARTBEAT_FILE.unlink(missing_ok=True)
    Path("/tmp/flightdock_warmup_active").unlink(missing_ok=True)
    if _n_envs_for_cleanup is not None:
        _kill_all_sitl_by_port(_n_envs_for_cleanup, port_stride=config.PORT_STRIDE)


atexit.register(_atexit_cleanup)


# ---------------------------------------------------------------------------
# P1 Rev 3 — final_run_summary.json emission
# ---------------------------------------------------------------------------
#
# State dict populated at launch; helpers emit the 9-field exit-time
# artifact atomically (temp + rename), idempotently (re-emit with same
# payload is a no-op), and best-effort (exceptions during write are
# logged; never mask original exit code / signal semantics).

_flightdock_run_summary_state: dict = {
    "start_wall_clock": None,         # time.monotonic() at learn entry
    "model": None,                    # SB3 model reference
    "args": None,                     # argparse Namespace
    "noise_resume_offset": 0,
    "noise_schedule_mode": "absolute_progress",
    "caught_signal": None,            # int signum if caught, else None
    "summary_path": None,             # log_dir/final_run_summary.json
    "emitted": False,                 # idempotency latch
}


def _extract_sigma_exit(model):
    """Robust sigma extraction from SB3 action_noise. Returns float or None."""
    try:
        noise = getattr(model, "action_noise", None)
        if noise is None:
            return None
        for attr in ("_sigma", "sigma"):
            if hasattr(noise, attr):
                arr = np.atleast_1d(
                    np.asarray(getattr(noise, attr), dtype=float))
                return float(arr.flat[0])
        inner_list = getattr(noise, "noises", None)
        if inner_list:
            inner = inner_list[0]
            for attr in ("_sigma", "sigma"):
                if hasattr(inner, attr):
                    arr = np.atleast_1d(
                        np.asarray(getattr(inner, attr), dtype=float))
                    return float(arr.flat[0])
    except Exception:
        return None
    return None


def _termination_to_exit_path(termination_reason):
    """Legacy exit_path alias: top-level class for termination_reason."""
    if termination_reason.startswith("signal_"):
        return "signal"
    if termination_reason == "exception":
        return "exception"
    if termination_reason == "guard_abort":
        return "guard_abort"
    return "normal"


def _classify_termination(final_ts, target_ts, h12_fired,
                          early_stop_fired=False):
    """E-2/E-4: name the reason model.learn() returned (pure, unit-tested).

    A guard halt returns from learn() exactly like a budget stop; before E-2
    both were labeled budget_exhaustion_partial and the halt was only
    discoverable via forensic files. H12 outranks the behavioral stop (the
    critic-divergence signature is the more specific finding).
    """
    if h12_fired:
        return "h12_divergence_halt"
    if early_stop_fired:
        return "behavioral_regression_halt"
    if target_ts > 0 and final_ts < target_ts:
        return "budget_exhaustion_partial"
    return "normal_budget_exhaustion"


def _emit_final_run_summary(termination_reason):
    """Emit final_run_summary.json atomically, idempotently, best-effort.

    Operator cautions honored:
      - Atomic: write `<path>.tmp.<pid>` then os.replace.
      - Idempotent: if file already exists with identical payload, no-op.
      - Best-effort: any exception during emission is swallowed (logged
        at DEBUG); never re-raised so exit code / signal semantics are
        preserved.
    """
    state = _flightdock_run_summary_state
    if state.get("emitted"):
        return
    path = state.get("summary_path")
    if not path:
        return
    tmp = None
    try:
        model = state.get("model")
        noise_offset = int(state.get("noise_resume_offset", 0) or 0)
        final_ts = 0
        if model is not None:
            final_ts = int(getattr(model, "num_timesteps", 0) or 0)
        effective = max(0, final_ts - noise_offset)
        sigma_exit = (
            _extract_sigma_exit(model) if model is not None else None)
        caught_sig = state.get("caught_signal")
        caught_field = None
        if caught_sig is not None:
            try:
                caught_field = int(caught_sig)
            except (TypeError, ValueError):
                caught_field = str(caught_sig)
        wall_clock = 0.0
        start = state.get("start_wall_clock")
        if start is not None:
            try:
                wall_clock = float(time.monotonic() - start)
            except Exception:
                wall_clock = 0.0
        payload = {
            "summary_schema_version": 1,
            "final_num_timesteps": final_ts,
            "effective_noise_progress_steps": effective,
            "sigma_exit": sigma_exit,
            "noise_schedule_mode": state.get(
                "noise_schedule_mode", "absolute_progress"),
            "resume_offset_steps_used": (
                noise_offset if noise_offset > 0 else None),
            "termination_reason": termination_reason,
            "caught_signal": caught_field,
            "exit_path": _termination_to_exit_path(termination_reason),
            "wall_clock_seconds": wall_clock,
        }
        body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if os.path.exists(path):
            try:
                with open(path, "r") as _f:
                    if _f.read() == body:
                        state["emitted"] = True
                        return
            except OSError:
                pass
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as _f:
            _f.write(body)
        os.replace(tmp, path)
        tmp = None
        state["emitted"] = True
    except Exception:
        try:
            logger.debug(
                "[final_run_summary] emission failed (best-effort)",
                exc_info=True)
        except Exception:
            pass
        # Never re-raise — best-effort per operator caution.
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT: emit final_run_summary, close env, re-raise."""
    logger.warning(f"[Signal] Received {_signal.Signals(signum).name}, cleaning up...")
    # (P1 Rev 3) Emit final_run_summary BEFORE cleanup + re-raise so the
    # artifact survives even if later cleanup steps fault. Best-effort;
    # signal semantics / exit code are never altered by this path.
    try:
        _flightdock_run_summary_state["caught_signal"] = int(signum)
        _emit_final_run_summary(
            f"signal_{_signal.Signals(signum).name}")
    except Exception:
        pass
    if _vec_env_ref is not None:
        try:
            _close_with_timeout(_vec_env_ref, timeout=config.SIGTERM_TIMEOUT_S)
        except Exception:
            logger.debug("Failed to close vec_env in signal handler", exc_info=True)
    if _n_envs_for_cleanup is not None:
        _kill_all_sitl_by_port(_n_envs_for_cleanup, port_stride=config.PORT_STRIDE)
    # Re-raise with default handler for correct exit code (128 + signum)
    _signal.signal(signum, _signal.SIG_DFL)
    os.kill(os.getpid(), signum)


_signal.signal(_signal.SIGTERM, _signal_handler)
_signal.signal(_signal.SIGINT, _signal_handler)


class HaltSentinelCallback(BaseCallback):
    """L7.6 §6 D1 fix: polls log_dir/HALT_REQUESTED, sends SIGTERM-to-self on detection.

    Routes through _signal_handler above. See docs/C3X_D1_SENTINEL_FIX.md."""
    def __init__(self, halt_path: str, check_every: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.halt_path = halt_path
        self.check_every = check_every
        self._tripped = False

    def _on_step(self) -> bool:
        if self._tripped:
            return False
        if self.n_calls % self.check_every != 0:
            return True
        if os.path.exists(self.halt_path):
            self._tripped = True
            logger.warning(f"[HALT] HALT_REQUESTED sentinel detected at {self.halt_path}; initiating graceful shutdown.")
            os.kill(os.getpid(), _signal.SIGTERM)
            return False
        return True


# ---------------------------------------------------------------------------
# Zero-residual initialization (Approach G — algorithm-specific)
# ---------------------------------------------------------------------------

def init_zero_residual_ppo(model: PPO) -> None:
    """Initialise PPO policy for MRRL: near-zero mean, low exploration noise.

    In Multi-Resolution Residual Learning (MRRL), action=[0,0,0] means
    "follow baseline exactly." Standard PPO initialisation (log_std=0,
    random action_net) produces large residuals that cause safety
    terminations within 2-5 steps.

    This function:
      1. Zeros the action_net bias and sets small weights (approx zero output)
      2. Sets log_std to LOG_STD_INIT=-0.7 (sigma=0.50, entropy +0.42 nats)
    """
    policy = model.policy
    # --- Zero the action net (last layer of actor) ---
    action_net = policy.action_net
    assert isinstance(action_net, nn.Linear), (
        f"Unexpected action_net type: {type(action_net)}. "
        f"Orthogonal init requires nn.Linear."
    )
    if hasattr(action_net, 'weight'):
        nn.init.orthogonal_(action_net.weight, gain=config.ACTION_NET_GAIN)
        if action_net.bias is not None:
            nn.init.zeros_(action_net.bias)
    # --- Set low exploration noise ---
    if hasattr(policy, 'log_std'):
        with torch.no_grad():
            policy.log_std.fill_(config.LOG_STD_INIT)
    logger.info(f"  Zero-residual PPO init: log_std={config.LOG_STD_INIT:.2f} "
                f"(sigma={torch.exp(torch.tensor(config.LOG_STD_INIT)):.3f}), "
                f"action_net gain={config.ACTION_NET_GAIN}")


def init_zero_residual_td3(model: TD3) -> None:
    """Initialise TD3 actor for MRRL: zero deterministic output.

    TD3's actor.mu is Sequential ending with [nn.Linear, nn.Tanh].
    Zeroing mu[-2] (the last Linear before Tanh) makes tanh(0)=0,
    so the actor outputs exactly [0,0,0] = pure baseline passthrough.

    CRITICAL: After zeroing, sync actor_target so that target Q-values
    are computed with matching weights. Without this, the critic learns
    from inconsistent target actions during the first learning_starts steps.

    Architecture empirically verified (SB3 2.3.2 with net_arch=dict(pi=[128,128])):
      mu[0] = Linear(39, 128)
      mu[1] = ReLU()
      mu[2] = Linear(128, 128)
      mu[3] = ReLU()
      mu[4] = Linear(128, 3) <-- zero this
      mu[5] = Tanh()
    """
    mu = model.actor.mu
    # Find the last Linear layer (mu[-2] when Tanh is present)
    last_linear = mu[-2]
    assert isinstance(last_linear, nn.Linear), (
        f"Expected nn.Linear at mu[-2], got {type(last_linear).__name__}. "
        f"TD3 actor structure may have changed — run hard gate diagnostic."
    )
    with torch.no_grad():
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
    # Sync target actor — CRITICAL for consistent critic learning
    model.actor_target.load_state_dict(model.actor.state_dict())
    logger.info("  Zero-residual TD3 init: mu[-2] zeroed, actor_target synced")


def reset_critic_in_place(model) -> dict:
    """CPRIME-808 R2: rebuild the critic and
    critic_target from scratch with the SAME architecture and a fresh
    optimizer, leaving the actor and actor_target byte-identical.

    The replay-lineage rule: a clamp change (CRRL_BPC_FLOOR_NEG) makes the
    Q-function trained under the old plant obsolete, while the actor may
    remain as a labelled warm start. Architecture is NOT changed here (the
    LayerNorm path is the separate --critic-layernorm fallback).

    Returns a witness dict the launcher asserts against the log:
    actor_sha16 (unchanged), critic_sha16_before/after (must differ),
    target_synced (True), n_critic_params.
    """
    import hashlib
    def _sha16(module):
        h = hashlib.sha256()
        for k, v in module.state_dict().items():
            h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]
    pol = model.policy
    actor_before = _sha16(pol.actor)
    critic_before = _sha16(pol.critic)
    if pol.share_features_extractor:
        raise RuntimeError("--reset-critic: share_features_extractor=True is "
                           "unsupported (critic would share the actor's extractor)")
    pol.critic = pol.make_critic(features_extractor=None)
    pol.critic_target = pol.make_critic(features_extractor=None)
    pol.critic_target.load_state_dict(pol.critic.state_dict())
    pol.critic.optimizer = pol.optimizer_class(
        pol.critic.parameters(), lr=model.lr_schedule(1), **pol.optimizer_kwargs)
    pol.critic_target.set_training_mode(False)
    model._create_aliases()  # TD3 caches .critic/.critic_target on the algorithm
    critic_after = _sha16(pol.critic)
    assert _sha16(pol.actor) == actor_before, "--reset-critic touched the actor"
    assert critic_after != critic_before, "--reset-critic produced the same critic"
    assert _sha16(pol.critic_target) == critic_after, "critic_target not synced"
    w = {"actor_sha16": actor_before, "critic_sha16_before": critic_before,
         "critic_sha16_after": critic_after, "target_synced": True,
         "n_critic_params": sum(p.numel() for p in pol.critic.parameters())}
    logger.info("[RESET_CRITIC] critic + critic_target re-initialised in place "
                "(same architecture), fresh optimizer; actor unchanged "
                "actor_sha16=%s critic %s -> %s n_params=%d",
                w["actor_sha16"], critic_before, critic_after, w["n_critic_params"])
    return w


def transfer_critic_weights_to_layernorm(
    new_model: TD3,
    old_model: TD3,
    shrink: float | None = None,
    perturb_scale: float = 0.01,
) -> None:
    """Map old critic (no LayerNorm) weights to new critic (with LayerNorm).

    Linear layers matched positionally: old qfN has K+1 Linears in Sequential
    (K hidden + 1 output), new qfN has K+1 Linears interleaved with K LayerNorms.
    LayerNorm layers keep defaults (weight=1, bias=0).

    If shrink is not None, apply S&P to hidden Linears only (not output layer):
        theta' = shrink * theta_old + perturb_scale * N(0,1)
    """
    for idx in range(new_model.critic.n_critics):
        old_seq = getattr(old_model.critic, f"qf{idx}")
        new_seq = getattr(new_model.critic, f"qf{idx}")
        old_linears = [m for m in old_seq if isinstance(m, nn.Linear)]
        new_linears = [m for m in new_seq if isinstance(m, nn.Linear)]
        assert len(old_linears) == len(new_linears), (
            f"qf{idx}: {len(old_linears)} old Linears vs {len(new_linears)} new — "
            f"net_arch mismatch between checkpoint and fresh model")
        with torch.no_grad():
            for i, (old_l, new_l) in enumerate(zip(old_linears, new_linears)):
                is_output = (i == len(old_linears) - 1)
                if shrink is not None and not is_output:
                    new_l.weight.copy_(
                        shrink * old_l.weight
                        + perturb_scale * torch.randn_like(old_l.weight))
                    new_l.bias.copy_(
                        shrink * old_l.bias
                        + perturb_scale * torch.randn_like(old_l.bias))
                else:
                    new_l.weight.copy_(old_l.weight)
                    new_l.bias.copy_(old_l.bias)
        logger.info("  qf%d: %d Linear layers transferred%s",
                    idx, len(old_linears),
                    f" (S&P shrink={shrink})" if shrink is not None else "")
    # Sync target critics
    new_model.critic_target.load_state_dict(new_model.critic.state_dict())


def init_zero_residual_sac(model: SAC) -> None:
    """Initialise SAC actor for MRRL: zero mean output.

    SAC's actor.mu is a single nn.Linear (not Sequential).
    Zeroing it makes the mean output [0,0,0]. The log_std layer is
    left untouched — SAC's automatic entropy tuning will adjust
    exploration noise via the learned alpha coefficient.

    No target actor to sync (SAC only has critic targets).
    """
    mu = model.actor.mu
    assert isinstance(mu, nn.Linear), (
        f"Expected nn.Linear for SAC actor.mu, got {type(mu).__name__}. "
        f"SAC actor structure may have changed."
    )
    with torch.no_grad():
        nn.init.zeros_(mu.weight)
        nn.init.zeros_(mu.bias)
    logger.info("  Zero-residual SAC init: actor.mu zeroed (log_std untouched)")


def linear_schedule(initial: float, final: float) -> Callable[[float], float]:
    """Linear learning rate schedule (SB3 convention).

    Parameters
    ----------
    initial: float
        Starting learning rate.
    final: float
        Ending learning rate.

    Returns
    -------
    Callable
        Function of progress_remaining (1.0 -> 0.0) returning current LR.
    """

    def func(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return func


def make_env(
    env_id: int, speedup: float, curriculum: CurriculumManager,
    ui_mode: str = "none", pool=None, wind_channels: bool = True,
    seed: Optional[int] = None,
) -> Callable:
    """Factory function for SubprocVecEnv — one env per subprocess.

    Parameters
    ----------
    pool: SITLPool or None
        If provided, creates PooledFormationSITLEnv instead.
    wind_channels: bool
        INERT since D6.1 Amendment-O (SIGNED 2026-07-18): the wind channels
        were DELETED and the observation is always 42-D (config.OBS_DIM).
        The flag is threaded through for caller compatibility only and
        cannot produce the OBS_DIM_LEGACY space.
    seed: Optional[int]
        Phase G: per-env seed (additive combine with env_id). Threaded to
        FormationSITLEnv constructor for worker-local Python random + numpy
        seeding in SubprocVecEnv forkserver children. None = unseeded.
    """
    # Capture runtime config mutations for propagation to forkserver
    # SubprocVecEnv workers. Forkserver children import config.py fresh and
    # do NOT see main-process config mutations. Closure captures ensure
    # the mutations are re-applied in each child before env construction.
    _cfg_residual_mode = config.RESIDUAL_MODE
    _cfg_rfl_activation_tier = config.RFL_ACTIVATION_TIER
    _cfg_force_zero = config.FORCE_ZERO_RESIDUAL
    _cfg_crrl_enabled = config.CRRL_ENABLED
    _cfg_exit_behind_mult = config.EXIT_BEHIND_MULTIPLIER
    _cfg_passthrough_thresh = config.PASSTHROUGH_L2_THRESHOLD
    _cfg_obs_norms_35 = config.OBS_NORMS[35]
    _cfg_obs_norms_36 = config.OBS_NORMS[36]
    _cfg_obs_norms_37 = config.OBS_NORMS[37]
    # Capture all intervention params for forkserver propagation
    _cfg_w_barrier = config.W_BARRIER
    _cfg_w_barrier_outside = config.W_BARRIER_OUTSIDE
    _cfg_cbf_violation_scale = config.CBF_VIOLATION_SCALE
    _cfg_barrier_proximity_gate = config.BARRIER_PROXIMITY_GATE
    _cfg_barrier_velocity_ema = config.BARRIER_VELOCITY_EMA
    _cfg_forward_bias_ms = config.FORWARD_BIAS_MS
    # FCLAMP-20260901: captured HERE and re-applied in _init below because the
    # override runs in the PARENT and the residual clamp is applied in the
    # WORKER. Omitting the propagation would leave the flag set in the process
    # that does not fly and unset in the one that does -- a silent no-op that
    # reads as "the clamp made no difference".
    _cfg_clamp_behind_positive_ms = config.CLAMP_BEHIND_POSITIVE_MS
    _cfg_bias_smooth_alpha = config.BIAS_SMOOTH_ALPHA
    _cfg_ema_alpha_speed = config.EMA_ALPHA_SPEED
    _cfg_speed_slew_rate = config.SPEED_SLEW_RATE_MPS2
    _cfg_inject_zero = config.INJECT_ZERO_ACTION
    _cfg_velocity_gate_k = config.VELOCITY_GATE_K
    _cfg_w_hold_fwd = config.W_HOLD_FWD
    _cfg_caps_actor_lambda = config.CAPS_ACTOR_LAMBDA  # Was missing from propagation

    def _init():
        # Re-apply runtime config mutations in child process
        from drl import config as _cfg
        _cfg.RESIDUAL_MODE = _cfg_residual_mode
        _cfg.RFL_ACTIVATION_TIER = _cfg_rfl_activation_tier
        _cfg.FORCE_ZERO_RESIDUAL = _cfg_force_zero
        _cfg.CRRL_ENABLED = _cfg_crrl_enabled
        _cfg.EXIT_BEHIND_MULTIPLIER = _cfg_exit_behind_mult
        _cfg.PASSTHROUGH_L2_THRESHOLD = _cfg_passthrough_thresh
        _cfg.OBS_NORMS[35] = _cfg_obs_norms_35
        _cfg.OBS_NORMS[36] = _cfg_obs_norms_36
        _cfg.OBS_NORMS[37] = _cfg_obs_norms_37
        # Propagate intervention params to worker
        _cfg.W_BARRIER = _cfg_w_barrier
        _cfg.W_BARRIER_OUTSIDE = _cfg_w_barrier_outside
        _cfg.CBF_VIOLATION_SCALE = _cfg_cbf_violation_scale
        _cfg.BARRIER_PROXIMITY_GATE = _cfg_barrier_proximity_gate
        _cfg.BARRIER_VELOCITY_EMA = _cfg_barrier_velocity_ema
        _cfg.FORWARD_BIAS_MS = _cfg_forward_bias_ms
        _cfg.CLAMP_BEHIND_POSITIVE_MS = _cfg_clamp_behind_positive_ms
        _cfg.BIAS_SMOOTH_ALPHA = _cfg_bias_smooth_alpha
        _cfg.EMA_ALPHA_SPEED = _cfg_ema_alpha_speed
        _cfg.SPEED_SLEW_RATE_MPS2 = _cfg_speed_slew_rate
        _cfg.INJECT_ZERO_ACTION = _cfg_inject_zero
        _cfg.VELOCITY_GATE_K = _cfg_velocity_gate_k
        _cfg.W_HOLD_FWD = _cfg_w_hold_fwd
        _cfg.CAPS_ACTOR_LAMBDA = _cfg_caps_actor_lambda  # Propagate to workers

        # C006 Phase 7: Stagger SITL launches to avoid port contention
        if env_id > 0:
            import time
            time.sleep(env_id * 8)
        # Phase G seed chain: additive combine (seed + env_id) so each
        # SubprocVecEnv worker gets a distinct but reproducible stream.
        # D1 (FD_DR_SEED_V2): when ON, derive a collision-free per-worker base
        # seed via SeedSequence (fixes the additive (seed,env_id) collision; e.g.
        # seed=0/env_id=42 vs seed=42/env_id=0) and seed the pooled env too. When
        # OFF, the legacy additive combine runs and the pooled env stays unseeded
        # (byte-identical). See docs/PHASE_G_IMPLEMENTATION_REPORT.md.
        _v2_seed = os.environ.get("FD_DR_SEED_V2", "0") == "1"
        if _v2_seed and seed is not None:
            _seed_arg = int(np.random.SeedSequence([int(seed), int(env_id)]).generate_state(1)[0])
        else:
            _seed_arg = None if seed is None else seed + env_id
        if pool is not None:
            from drl.env_pooled import PooledFormationSITLEnv
            return PooledFormationSITLEnv(
                env_id=env_id,
                pool=pool,
                speedup=speedup,
                curriculum_manager=curriculum,
                ui_mode=ui_mode,
                seed=(_seed_arg if _v2_seed else None),
            )
        return FormationSITLEnv(
            env_id=env_id,
            speedup=speedup,
            curriculum_manager=curriculum,
            ui_mode=ui_mode,
            wind_channels=wind_channels,
            seed=_seed_arg,
        )

    return _init


# ---------------------------------------------------------------------------
# Model creation dispatch (v18 — algorithm-aware)
# ---------------------------------------------------------------------------

def effective_learning_starts_fn(td3_cfg, n_envs, override=None):
    """F2 (CONFIG_AUDIT 2026-07-18): per-env scaling capped at a hardware-
    independent total so the first gradient lands at the same transition
    count on any box (unbounded 4000*n_envs = 64k@n16 exceeded S0's 60k)."""
    if override is not None:
        return override
    return min(td3_cfg.learning_starts * n_envs,
               td3_cfg.learning_starts_total_cap)


def effective_gradient_steps_fn(td3_cfg, n_envs):
    """F3 (CONFIG_AUDIT 2026-07-18): hold grads-per-transition constant
    across hardware (fixed gradient_steps=4 made UTD silently 4/n_envs)."""
    return max(1, int(round(td3_cfg.grads_per_transition * n_envs)))


def _create_td3(vec_env, td3_cfg: config.TD3Config, stage: int,
                log_dir: str, args: argparse.Namespace) -> TD3:
    """Create TD3 model with per-dimension action noise and zero-residual init."""
    # Per-dimension exploration noise: [speed, heading, altitude]
    # (ACTION_DIM=3 cross-axis restored 2026-05-06)
    n_actions = vec_env.action_space.shape[0]
    noise_sigma = np.array([
        td3_cfg.noise_sigma_speed,
        td3_cfg.noise_sigma_heading,
        td3_cfg.noise_sigma_altitude,
    ])
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_sigma,
    )

    # Buffer size is stage-dependent (smaller for Stage 0 exploration)
    buffer_size = (td3_cfg.buffer_size_stage0 if stage == 0
                   else td3_cfg.buffer_size_stage1_plus)

    # F2 (CONFIG_AUDIT): per-env scaling capped hardware-independently
    # (supersedes unbounded v19 P2 scaling; CLI override still wins).
    effective_learning_starts = effective_learning_starts_fn(
        td3_cfg, vec_env.num_envs, args.learning_starts)
    eff_gs = effective_gradient_steps_fn(td3_cfg, vec_env.num_envs)  # F3
    logger.info(f"[F2/F3] effective learning_starts={effective_learning_starts} "
                f"gradient_steps={eff_gs} (n_envs={vec_env.num_envs}, "
                f"cap={td3_cfg.learning_starts_total_cap})")

    # Apply CLI overrides (fall back to config defaults)
    eff_lr = args.learning_rate if args.learning_rate is not None else td3_cfg.learning_rate
    eff_tau = args.tau if args.tau is not None else td3_cfg.tau
    eff_pd = args.policy_delay if args.policy_delay is not None else td3_cfg.policy_delay
    eff_gamma = args.gamma if args.gamma is not None else td3_cfg.gamma

    # Use CustomClippedTD3 when gradient clipping requested
    ModelClass = CustomClippedTD3 if args.max_grad_norm_critic is not None else TD3
    extra_kw = {}
    if args.max_grad_norm_critic is not None:
        extra_kw["max_grad_norm_critic"] = args.max_grad_norm_critic
        logger.info(f"[GRAD_CLIP] Fresh-start CustomClippedTD3 with "
                    f"max_grad_norm_critic={args.max_grad_norm_critic}")

    # P1-B: CAPS temporal smoothness
    if args.caps_lambda_t > 0.0:
        extra_kw["caps_lambda_t"] = args.caps_lambda_t
        logger.info(f"[CAPS] caps_lambda_t={args.caps_lambda_t}")
        if ModelClass is not CustomClippedTD3:
            ModelClass = CustomClippedTD3
            extra_kw.setdefault("max_grad_norm_critic", 0.5)
    if args.caps_actor_lambda > 0.0:
        extra_kw["caps_actor_lambda"] = args.caps_actor_lambda
        logger.info(f"[CAPS] caps_actor_lambda={args.caps_actor_lambda}")
        if ModelClass is not CustomClippedTD3:
            ModelClass = CustomClippedTD3
            extra_kw.setdefault("max_grad_norm_critic", 0.5)

    # Use LayerNormTD3Policy when --critic-layernorm is set
    policy_cls = LayerNormTD3Policy if args.critic_layernorm else "MlpPolicy"

    model = ModelClass(
        policy=policy_cls,
        env=vec_env,
        learning_rate=eff_lr,
        buffer_size=buffer_size,
        learning_starts=effective_learning_starts,
        batch_size=td3_cfg.batch_size,
        tau=eff_tau,
        gamma=eff_gamma,
        train_freq=(td3_cfg.train_freq, "step"),
        gradient_steps=eff_gs,  # F3: hardware-decoupled UTD
        action_noise=action_noise,
        policy_delay=eff_pd,
        target_policy_noise=td3_cfg.target_policy_noise,
        target_noise_clip=td3_cfg.target_noise_clip,
        policy_kwargs=dict(
            net_arch=td3_cfg.net_arch,
        ),
        verbose=1,
        tensorboard_log=log_dir,
        **extra_kw,
    )

    # Approach G: zero-residual init for MRRL
    init_zero_residual_td3(model)

    # Log actor init state (confirms fresh init vs transfer)
    _mu_norm = model.policy.actor.mu[-2].weight.data.norm().item()
    logger.info(f"  Actor initialized fresh (zero-residual). "
                f"Output layer weight norm: {_mu_norm:.6f}")
    logger.info(f"  {ModelClass.__name__} created: buffer={buffer_size}, batch={td3_cfg.batch_size}, "
                f"learning_starts={effective_learning_starts} "
                f"({td3_cfg.learning_starts} × {vec_env.num_envs} envs), "
                f"gradient_steps={td3_cfg.gradient_steps}, "
                f"noise_sigma={noise_sigma.tolist()}")
    return model


def _create_sac(vec_env, sac_cfg: config.SACConfig, stage: int,
                log_dir: str, args: argparse.Namespace) -> SAC:
    """Create SAC model with automatic entropy tuning and zero-residual init."""
    buffer_size = (sac_cfg.buffer_size_stage0 if stage == 0
                   else sac_cfg.buffer_size_stage1_plus)

    # Scale learning_starts by n_envs (v19 P2, same rationale as TD3)
    if args.learning_starts is not None:
        effective_learning_starts = args.learning_starts
    else:
        effective_learning_starts = sac_cfg.learning_starts * vec_env.num_envs

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=sac_cfg.learning_rate,
        buffer_size=buffer_size,
        learning_starts=effective_learning_starts,
        batch_size=sac_cfg.batch_size,
        tau=sac_cfg.tau,
        gamma=sac_cfg.gamma,
        train_freq=(sac_cfg.train_freq, "step"),
        gradient_steps=sac_cfg.gradient_steps,
        ent_coef=sac_cfg.ent_coef,
        target_entropy=sac_cfg.target_entropy,
        policy_kwargs=dict(
            net_arch=sac_cfg.net_arch,
        ),
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Approach G: zero-residual init for MRRL
    init_zero_residual_sac(model)

    logger.info(f"  SAC created: buffer={buffer_size}, batch={sac_cfg.batch_size}, "
                f"learning_starts={effective_learning_starts} "
                f"({sac_cfg.learning_starts} × {vec_env.num_envs} envs), "
                f"ent_coef={sac_cfg.ent_coef}")
    return model


def _create_ppo(vec_env, ppo_cfg: config.PPOConfig, n_envs: int,
                log_dir: str) -> PPO:
    """Create PPO model with SeparateClipPPO and zero-residual init."""
    effective_batch_size = ppo_cfg.n_steps * n_envs

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(ppo_cfg.lr_initial, ppo_cfg.lr_final),
        n_steps=ppo_cfg.n_steps,
        batch_size=effective_batch_size,
        n_epochs=ppo_cfg.n_epochs,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        target_kl=ppo_cfg.target_kl,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        normalize_advantage=ppo_cfg.normalize_advantage,
        policy_kwargs=dict(
            net_arch=ppo_cfg.net_arch,
            activation_fn=nn.Tanh,
        ),
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Approach G: zero-residual init for MRRL
    init_zero_residual_ppo(model)

    logger.info(f"  PPO created: n_steps={ppo_cfg.n_steps}, "
                f"batch={effective_batch_size}, "
                f"clip={ppo_cfg.clip_range}, ent={ppo_cfg.ent_coef}")
    return model


def _resolve_or_default_profile(args) -> Dict[str, Any]:
    """Patch A.1 (Phase 5): resolve --profile arg with fail-safe default.

    If args.profile is None, default to clean_straight_zero_noise (the perception-validation
    profile) with a warning. If args.profile is set but unknown, fail-loud with SystemExit(2).
    Mutates args.profile in place when defaulting. Returns the resolved profile config dict.
    """
    if args.profile is None:
        logger.warning(
            "[PROFILE] --profile not specified; defaulting to "
            "'clean_straight_zero_noise'. Pass --profile explicitly to suppress this warning."
        )
        args.profile = "clean_straight_zero_noise"
    profile_cfg = config.CONDITION_PROFILES.get(args.profile)
    if profile_cfg is None:
        logger.error("[FATAL] Unknown profile %r.", args.profile)
        sys.exit(2)
    return profile_cfg


# ---------------------------------------------------------------------------
# T-457 launch preconditions (class STRUCTURAL, 2026-08-12)
#
# Two configurations that used to be prevented only by a human remembering to
# write a line in a launch script are now refused by construction, at the
# learner entry, BEFORE any SITL is spawned or any model is loaded:
#
#   G4 FD_COLLISION_TRUTH_CHANNEL=1 in a TRAINING process. With the flag on,
#       SIM_STATE truth decides an episode boundary (env_formation_sitl.py
#       :7094-7096 -> :7109-7114) whose terminal_type is passed straight into
#       the reward (:2771 -> :2789-2790 -> reward_mrrl.py:347-348, priced
#       R_TERMINAL_SAFETY = -10.0). That is truth reaching the learner: G4
#       Tier-0. Correct and untouched on the EVAL path, which never runs this.
#
#   TMUX A training launch outside tmux dies with its ssh session (the setsid'd
#       python survives but its tee writers are wrapper children -> EPIPE, and
#       the rc capture is lost). The standing directive has existed since
#       2026-07-12; until now the only enforcement lived in one hand-copied
#       block in an analysis-dir script, so any new wrapper reproduced the bug.
#
# Both are sys.exit(), never assert: an assert vanishes under `python -O`, and
# a guard removable by a command-line switch cannot fire. Both live here in
# train.py as well as in scripts/run_drl_training.sh so that a hand-written
# wrapper calling train.py directly cannot bypass them.
# ---------------------------------------------------------------------------

#: Exit code when the G4 truth-channel precondition refuses a training launch.
#: NOT 42: the wrapper's own regression suite already uses a child exit of 42
#: as an arbitrary sentinel (test_exit_code_preserved_on_sys_exit_42), and the
#: wrapper faithfully propagates a child's rc, so a G4 refusal and that
#: sentinel would be indistinguishable in wrapper_exit.txt.
G4_TRUTH_CHANNEL_REFUSED_RC = 43
#: Exit code when the tmux precondition refuses a training launch. Matches the
#: rc the pre-existing G-TMUX block in w1418_relaunch_s15b.sh already used.
TMUX_REFUSED_RC = 9


def _truth_channel_ack_is_valid(ack: str) -> bool:
    """True only for a ledger task id: 'T-' followed by digits.

    Deliberately rejects '1', 'true', 'yes' and blanks. The whole point of the
    declaration is that it cannot be produced by an environment variable
    leaking in from an eval shell; it has to be authored against a.
    """
    ack = (ack or "").strip()
    return ack.startswith("T-") and ack[2:].isdigit()


def check_g4_truth_channel_precondition() -> None:
    """Refuse a training launch that would feed truth into the reward.

    No-op (the overwhelmingly common case) when FD_COLLISION_TRUTH_CHANNEL is
    unset or 0. Raises SystemExit(G4_TRUTH_CHANNEL_REFUSED_RC) when the flag is
    on without a valid ledger-tagged declaration.
    """
    if not config.COLLISION_TRUTH_CHANNEL_ENABLE:
        return
    ack = config.COLLISION_TRUTH_CHANNEL_TRAIN_ACK
    if not _truth_channel_ack_is_valid(ack):
        logger.error(
            "[FATAL][G4] FD_COLLISION_TRUTH_CHANNEL=1 in a TRAINING process.\n"
            "  With the flag on, SIM_STATE truth decides the SAFETY_COLLISION\n"
            "  episode boundary, and that terminal is priced by the reward\n"
            "  (R_TERMINAL_SAFETY). Truth would reach the learner -- G4 Tier-0.\n"
            "  The flag is CORRECT for eval/gate legs; it is refused here.\n"
            "\n"
            "  Refusing before any SITL spawn or model load. Either:\n"
            "    * unset FD_COLLISION_TRUTH_CHANNEL for this training run, or\n"
            "    * set FD_COLLISION_TRUTH_CHANNEL_TRAIN_ACK=T-<n> naming the\n"
            "      ledger row that carries the signed G4 review for this run.\n"
            "  Declaration seen: %r (must be 'T-' + digits).",
            ack,
        )
        sys.exit(G4_TRUTH_CHANNEL_REFUSED_RC)
    logger.warning(
        "[G4-EXCEPTION] FD_COLLISION_TRUTH_CHANNEL=1 accepted in TRAINING "
        "under declaration %s. Truth decides the SAFETY_COLLISION boundary "
        "and the reward prices it. This run is NOT G4-clean by construction "
        "and every artifact it produces inherits that caveat.", ack,
    )


def check_g4_align_center_precondition() -> None:
    """Refuse a training launch that carries the Fix A truth-seam flag.

    FD_RAW_ALIGN_CENTER (FIXAELECT-717) is REQUIRED on every eval leg and is
    therefore exported by the gate relay shell -- which is exactly the shell a
    training run gets launched from by hand. env_contract R4 already bars it,
    but that assertion binds only when FD_CURRICULUM_V3=1; this one is
    UNCONDITIONAL, so a non-v3 launch cannot inherit the flag silently.

    No-op (the overwhelmingly common case) when the flag is unset or 0. Raises
    SystemExit(G4_TRUTH_CHANNEL_REFUSED_RC) when it is armed without a valid
    ledger-tagged declaration. Sharing rc 43 with the collision-channel guard
    is deliberate: it is the same refusal CLASS (a truth-seam knob armed in
    training), a second code would fragment triage, and the FATAL line names
    which knob tripped.
    """
    if not config.RAW_ALIGN_CENTER_ENABLE:
        return
    ack = config.RAW_ALIGN_CENTER_TRAIN_ACK
    if not _truth_channel_ack_is_valid(ack):
        logger.error(
            "[FATAL][G4] FD_RAW_ALIGN_CENTER=1 in a TRAINING process.\n"
            "  The flag re-centres the raw/truth align lattice. It is CORRECT\n"
            "  and REQUIRED on eval/gate legs (FIXAELECT-717); it is refused\n"
            "  here, because a learner must not run on a truth-seam substrate\n"
            "  that only the EVAL role is elected to carry.\n"
            "\n"
            "  Refusing before any SITL spawn or model load. Either:\n"
            "    * unset FD_RAW_ALIGN_CENTER for this training run, or\n"
            "    * set FD_RAW_ALIGN_CENTER_TRAIN_ACK=T-<n> naming the ledger\n"
            "      row that carries the signed G4 review for this run.\n"
            "  Declaration seen: %r (must be 'T-' + digits).",
            ack,
        )
        sys.exit(G4_TRUTH_CHANNEL_REFUSED_RC)
    logger.warning(
        "[G4-EXCEPTION] FD_RAW_ALIGN_CENTER=1 accepted in TRAINING under "
        "declaration %s. The learner is running on the EVAL truth-seam "
        "configuration; every artifact it produces inherits that caveat.", ack,
    )


def check_tmux_precondition(unsafe: bool = False) -> None:
    """Refuse a training launch that is not inside a tmux session.

    ``unsafe`` mirrors train.py's existing --unsafe development escape (the
    same switch that already waives the wrapper requirement): it downgrades the
    refusal to a warning so a developer can run a throwaway launch. It is not a
    campaign path.
    """
    if os.environ.get("TMUX", ""):
        return
    if unsafe:
        logger.warning(
            "[TMUX] Not inside tmux, but --unsafe was passed. Proceeding. "
            "This launch dies with its terminal -- development only, never a "
            "campaign.")
        return
    logger.error(
        "[FATAL][TMUX] Training must run inside a tmux session.\n"
        "  Outside tmux the run dies with its ssh session: the setsid'd python\n"
        "  survives but its tee writers are wrapper children (EPIPE) and the\n"
        "  exit code is lost, so the campaign ends with no verdict and no log.\n"
        "\n"
        "  Refusing before any SITL spawn or model load. Launch as:\n"
        "    tmux new-session -d -s <name> 'scripts/run_drl_training.sh ...'\n"
        "  Development-only override: --unsafe.")
    sys.exit(TMUX_REFUSED_RC)


def main() -> None:
    # F-1: fail closed on the curriculum-v3 environment contract BEFORE any
    # SITL spawn. Only binds when FD_CURRICULUM_V3=1 (v1/v2 paths untouched);
    # role=train asserts the D7 line, an A7 sigma triple, rehearsal ENABLED,
    # and no truth-side knob armed. FD_ENV_CONTRACT_ENFORCE=0 downgrades to a
    # warning. See src/drl/env_contract.py.
    if os.environ.get("FD_CURRICULUM_V3", "0").strip() == "1":
        from src.drl import env_contract
        env_contract.assert_contract(env_contract.TRAIN)

    parser = argparse.ArgumentParser(
        description="FlightDock DRL Training (TD3/SAC/PPO + Residual MRRL)",
    )
    parser.add_argument(
        "--algorithm", type=str, default="td3",
        choices=["td3", "sac", "ppo"],
        help="RL algorithm: td3 (default, off-policy), sac (off-policy fallback), "
             "ppo (on-policy, defense comparison)",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1_728_000,
        help="Total training timesteps (default: 1728000 for Stages 0-2)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel SITL environments (default: 4, max: 4)",
    )
    parser.add_argument(
        "--speedup", type=float, default=2.0,
        help="SITL speedup factor (default: 2.0)",
    )
    parser.add_argument(
        "--stage", type=int, default=0,
        help="Initial curriculum stage (default: 0)",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to model zip to resume from",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility (default: None = entropy-seeded).",
    )
    parser.add_argument(
        "--freeze-normalize", action="store_true",
        help="Freeze VecNormalize stats (for Stage 3 fine-tuning)",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run identifier; auto-sets checkpoint-dir to "
             "checkpoints/{run_name}/ if --checkpoint-dir not explicit",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/default",
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/default",
        help="Directory for training logs and metrics",
    )
    parser.add_argument(
        "--ui", type=str, default="none",
        choices=["none", "map", "console", "both"],
        help="MAVProxy UI mode (default: none = headless)",
    )
    parser.add_argument(
        "--pool", action="store_true",
        help="Use pre-warmed SITL pool (Approach D) — background warmup",
    )
    parser.add_argument(
        "--pool-size", type=int, default=None,
        help="Pool size (default: n_envs + 2 buffer slots)",
    )
    parser.add_argument(
        "--auto-stop-on-flat", action="store_true", default=False,
        help="Cleanly stop training if no learning signal is detected "
             "(flat r_prox + low explained_variance). See RewardMonitorCallback.",
    )
    parser.add_argument(
        "--no-curriculum", action="store_true", default=False,
        help="Disable curriculum advancement. Train at fixed --stage for "
             "entire run. Used for RQ3 non-curriculum comparison.",
    )
    # Patch A 2026-05-06: explicit condition profile (required for training).
    parser.add_argument(
        "--profile", type=str, default=None,
        choices=sorted(config.CONDITION_PROFILES.keys()),
        help="Runtime condition profile. Required for training entry points; "
             "abort if unresolved. Choices: clean_straight_zero_noise (clean "
             "validation), stage0_canonical (Stage-0 racetrack legacy), "
             "curriculum (defer to CURRICULUM_STAGES + sample_domain_randomization).",
    )
    parser.add_argument(
        "--unsafe", action="store_true", default=False,
        help="Skip wrapper script check. Development only — disables SIGKILL "
             "orphan protection. Do NOT use for unattended training.",
    )
    # PPO-specific escalation triggers (ignored for td3/sac)
    parser.add_argument(
        "--ent-coef", type=float, default=None,
        help="Override PPO entropy coefficient (PPO only). "
             "Escalation trigger: 0.02 -> 0.04 at rollout 10.",
    )
    parser.add_argument(
        "--action-scale", type=float, default=None,
        help="Override curriculum stage action_scale. "
             "Escalation trigger: 0.5 -> 0.8 at rollout 15.",
    )
    # Reference-level residual mode
    parser.add_argument(
        "--residual-mode", type=str, default=None,
        choices=["actuator", "reference"],
        help="Residual mode. 'actuator' = C3m legacy, 'reference' = C3r RFL.",
    )
    parser.add_argument(
        "--rfl-activation-tier", type=int, default=None,
        help="RFL phase-gate activation tier (default: 3 = M3a_fine).",
    )
    parser.add_argument(
        "--force-zero-residual", action="store_true", default=False,
        help="Hard-force applied_residual=[0,0,0] in both modes. "
             "For Phase 0 re-anchor only. Bypasses model output entirely.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override learning rate on resume. SB3's TD3.load() restores lr "
             "from checkpoint, ignoring config.py. Use this flag to change lr "
             "between campaign phases. Example: --learning-rate 5e-5",
    )
    # C006 Phase 3: critic stabilization + hyperparameter overrides on resume
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Override target network tau on resume (default: checkpoint value). "
             "Phase 3 uses --tau 0.001 (5x slower target updates).",
    )
    parser.add_argument(
        "--policy-delay", type=int, default=None,
        help="Override policy delay on resume (default: checkpoint value). "
             "Phase 3 uses --policy-delay 3.",
    )
    parser.add_argument(
        "--critic-layernorm", action="store_true",
        help="Use LayerNormTD3Policy (critic LayerNorm, DAWN paper). "
             "On resume, reinitializes critic with LayerNorm while transferring "
             "actor weights from checkpoint. TD3 only.",
    )
    parser.add_argument(
        "--unfreeze-normalize-for", type=int, default=0,
        help="Temporarily unfreeze VecNormalize for N episodes after resume "
             "(recalibrate running stats for new reward distribution).",
    )
    parser.add_argument(
        "--actor-freeze-until-critic-stable", action="store_true",
        help="Freeze actor until critic loss stabilizes below threshold. "
             "Prevents learned helplessness during critic calibration.",
    )
    parser.add_argument(
        "--actor-freeze-min-steps", type=int, default=40000,
        help="Minimum steps before actor can unfreeze (Phase 5.1 default: 40000).",
    )
    parser.add_argument(
        "--actor-freeze-max-steps", type=int, default=80000,
        help="Force actor unfreeze at this step count (Phase 5.1 default: 80000).",
    )
    parser.add_argument(
        "--actor-freeze-critic-threshold", type=float, default=15.0,
        help="Critic loss threshold for actor unfreeze (Phase 6 default: 15.0).",
    )
    parser.add_argument(
        "--fresh-buffer", action="store_true", default=False,
        help="Create a fresh 200K replay buffer, discarding any checkpoint buffer. "
             "Phase 6: avoids SB3 buffer pickle lock on LayerNorm resume.",
    )
    # Phase 5.1: gamma override, critic abort threshold, gradient clipping
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Override discount factor on resume (Phase 5.1: --gamma 0.99).",
    )
    parser.add_argument(
        "--critic-abort-threshold", type=float, default=500.0,
        help="Critic loss rolling-mean abort threshold (default: 500.0). "
             "C3s: raised from 100.0 — C3r aborted at 101.6 rolling mean "
             "before first gate checkpoint. Per-episode critic was stable (33-49). "
             "GUARDWIN (2026-08-26): default 200 -> 500. With a one-episode window, "
             "200 sits BELOW the worst episode-mean critic loss ever measured (256, "
             "analysis/guardprice_20260826T2000Z); every launcher already passes 500.",
    )
    parser.add_argument(
        "--critic-abort-window", type=int, default=1000,
        help="Readings averaged by the critic guard's rolling-mean limb (default: 1000). "
             "Each reading is one train() call = gradient_steps batches = ONE batch at "
             "n_envs=1, so 1000 is about one episode. GUARDWIN (2026-08-26): was a "
             "hard-coded 5 -- five batches, ~0.25 s -- which killed three healthy legs in "
             "36 h on single-batch spikes (RB-61). 5 restores the legacy window.",
    )
    parser.add_argument(
        "--critic-hard-ceiling", type=float, default=500.0,
        help="Critic loss hard ceiling (default: 500.0). RB-67: was a hardcoded literal "
             "at the construction site; the RB-61 flag never reached it.",
    )
    parser.add_argument(
        "--critic-hard-ceiling-persist", type=int, default=3,
        help="Consecutive over-ceiling samples required before the hard ceiling aborts "
             "(default: 3 = 12 batches at n_envs=4). RB-67: one sample of 503.7 against a "
             "rolling mean of 251.8 ended Candidate-A leg 3; 1 restores the legacy trigger.",
    )
    parser.add_argument(
        "--max-grad-norm-critic", type=float, default=1.0,
        help="Critic gradient clipping norm via CustomClippedTD3 (Phase 5.1: 0.5).",
    )
    # Phase 7: config verification
    parser.add_argument(
        "--config-verify", action="store_true", default=False,
        help="Enable config verification callback (Phase 7). Writes "
             "config_snapshot.json to log-dir and hard-aborts on mismatch.",
    )
    # Phase 7: buffer seeding
    parser.add_argument(
        "--buffer-seed-steps", type=int, default=0,
        help="Collect N transitions from current policy into replay buffer "
             "before training (Phase 7: 30000). Runs at P=1.",
    )
    # True Phase 0 warmup — zero-action baseline transitions
    parser.add_argument(
        "--phase0-warmup-steps", type=int, default=0,
        help="Pre-populate replay buffer with N zero-action transitions "
             "(true PID baseline). Addresses DAWN cold-start pathology. "
             "Runs at P=1. Sets learning_starts=0 to prevent SB3 random "
             "exploration from diluting the clean baseline data.",
    )
    # Warm restart flags
    parser.add_argument(
        "--transfer-actor-weights", action="store_true", default=False,
        help="Transfer actor weights from checkpoint on --critic-layernorm resume. "
             "C3t: preserves W2 actor instead of zero-residual init.",
    )
    parser.add_argument(
        "--critic-shrink-perturb", type=float, default=None,
        help="Shrink factor for Shrink & Perturb on critic hidden layers during "
             "--critic-layernorm resume. theta'=alpha*theta_old + 0.01*N(0,1). "
             "C3t: 0.8 (retain 80%% old weights). None=no S&P.",
    )
    parser.add_argument(
        "--reset-critic", action="store_true", default=False,
        help="CPRIME-808 R2: on --resume-from (td3, plain policy), re-initialise "
             "critic + critic_target with the SAME architecture and a fresh "
             "optimizer; actor/actor_target unchanged. Use with --fresh-buffer "
             "after a clamp change (new replay lineage).")
    parser.add_argument(
        "--reset-critic-optimizer", action="store_true", default=False,
        help="Reset critic Adam optimizer state after --critic-layernorm resume. "
             "Clears stale momentum/variance from pre-LayerNorm architecture.",
    )
    # Phase 7: exploration noise annealing
    parser.add_argument(
        "--exploration-noise-sigma", type=float, default=0.25,
        help="Initial exploration noise sigma for TD3 (Phase 7: 0.25).",
    )
    parser.add_argument(
        "--exploration-noise-sigma-final", type=float, default=0.15,
        help="Final exploration noise sigma after annealing (Phase 7: 0.15).",
    )
    parser.add_argument(
        "--exploration-noise-anneal-steps", type=int, default=200000,
        help="Anneal noise sigma over this many timesteps (Phase 7: 200000).",
    )
    parser.add_argument(
        "--reset-exploration-schedule-on-resume", action="store_true",
        default=False,
        help="H4-Alpha: re-base the exploration-noise anneal schedule to "
             "the resume checkpoint's num_timesteps. Without this flag, the "
             "anneal progress is computed against absolute num_timesteps and "
             "a resumed run inherits the parent's schedule position, which "
             "collapses the initial sigma override. Opt-in; default preserves "
             "pre-H4 behavior exactly.",
    )
    # Campaign 3: reward weight override
    parser.add_argument(
        "--w-contact-dwell", type=float, default=None,
        help="Override w_contact_dwell in all STAGE_WEIGHTS entries. "
             "Campaign 3: amplify dwell reward (e.g. 2.5).",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=None,
        help="Override effective learning_starts (no n_envs multiplication). "
             "Default: config.learning_starts × n_envs. "
             "Phase 7B: --learning-starts 1000 (warm buffer, skip cold-start delay).",
    )
    # Campaign 3: P0-A zone-based w_ref
    parser.add_argument(
        "--zone-based-wref", action="store_true",
        default=None, dest="zone_based_wref",
        help="Enable zone-based w_ref modulation (Campaign 3 P0-A).",
    )
    parser.add_argument(
        "--no-zone-based-wref", action="store_false",
        dest="zone_based_wref",
        help="Disable zone-based w_ref modulation.",
    )
    parser.add_argument(
        "--w-ref-inner", type=float, default=None,
        help="W_REF_INNER: penalty weight inside docking envelope (P0-A).",
    )
    parser.add_argument(
        "--w-ref-outer", type=float, default=None,
        help="W_REF_OUTER: penalty weight far from envelope (P0-A).",
    )
    # Campaign 3: P0-B r_prox envelope suppression
    parser.add_argument(
        "--prox-envelope-suppression", type=float, default=None,
        help="R_PROX_ENVELOPE_SUPPRESSION factor (Campaign 3 P0-B).",
    )
    # Campaign 3: P1-A action rate penalty
    parser.add_argument(
        "--w-action-rate", type=float, default=None,
        help="Override w_action_rate in all STAGE_WEIGHTS entries (P1-A).",
    )
    # Campaign 3: P1-B CAPS temporal smoothness
    parser.add_argument(
        "--caps-lambda-t", type=float, default=0.0,
        help="CAPS temporal smoothness lambda for actor loss (P1-B, 0.0=disabled).",
    )
    # Campaign 3c: contact dwell exponent (power-law ramp)
    parser.add_argument(
        "--contact-dwell-exponent", type=float, default=None,
        help="Override CONTACT_DWELL_EXPONENT (k for r_contact_dwell power-law). "
             "k=1.0 = linear (C3b), k=2.0 = quadratic (C3c default).",
    )
    # Campaign 3h: exit penalty configuration
    parser.add_argument(
        "--w-exit-penalty", type=float, default=None,
        help="Override w_exit_penalty in all STAGE_WEIGHTS entries (C3h).",
    )
    parser.add_argument(
        "--exit-penalty-dwell-threshold", type=float, default=None,
        help="Override EXIT_DWELL_THRESHOLD_S (C3h). Upper bound for r_exit penalty.",
    )
    parser.add_argument(
        "--exit-penalty-lower-gate", type=float, default=None,
        help="Override EXIT_DWELL_LOWER_GATE_S (C3h). Exits below this are noise, not penalized.",
    )
    # Campaign 3i: reward pivot CLI args
    parser.add_argument(
        "--no-approach-suppression-inside-m3b", action="store_true", default=False,
        help="Disable r_approach suppression inside M3b (C3i default: suppressed).",
    )
    parser.add_argument(
        "--exit-behind-multiplier", type=float, default=None,
        help="Override EXIT_BEHIND_MULTIPLIER (C3i). Behind-axis exits get Nx penalty.",
    )
    parser.add_argument(
        "--dwell-ramp-floor", type=float, default=None,
        help="Override DWELL_RAMP_FLOOR (C3i). Linear floor for early dwell gradient.",
    )
    # Campaign 3j: MRRL-aligned reward architecture
    parser.add_argument(
        "--approach-taper-midpoint", type=float, default=None,
        help="Sigmoid midpoint for r_approach taper (meters, default 10.0).",
    )
    parser.add_argument(
        "--approach-taper-steepness", type=float, default=None,
        help="Sigmoid steepness k for r_approach taper (default 1.0).",
    )
    parser.add_argument(
        "--w-overspeed", type=float, default=None,
        help="Weight for overspeed penalty (default 1.0, conservative).",
    )
    parser.add_argument(
        "--overspeed-corridor-outer", type=float, default=None,
        help="Outer boundary of overspeed corridor (meters, default 10.0).",
    )
    parser.add_argument(
        "--overspeed-corridor-inner", type=float, default=None,
        help="Inner boundary of overspeed corridor (meters, default 0.7).",
    )
    # Campaign 3: P2 observation space
    parser.add_argument(
        "--obs-wind-channels", action="store_true",
        default=True, dest="obs_wind_channels",
        help="INERT since D6.1 Amendment-O — the wind channels were deleted "
             "and the observation is always 42-D. Accepted for compatibility.",
    )
    parser.add_argument(
        "--no-obs-wind-channels", action="store_false",
        dest="obs_wind_channels",
        help="INERT since D6.1 Amendment-O — the wind channels were deleted "
             "and the observation is always 42-D. Accepted for compatibility; "
             "it cannot restore the pre-Campaign-3 OBS_DIM_LEGACY space.",
    )
    # Campaign 3: P3-CRRL residual clipping
    parser.add_argument(
        "--crrl-beta", type=float, default=None,
        help="CRRL beta: residual bounded to +/-beta*|u_PID_delta| (P3).",
    )
    parser.add_argument(
        "--crrl", action="store_true",
        default=None, dest="crrl",
        help="Enable CRRL residual clipping (Campaign 3 P3).",
    )
    parser.add_argument(
        "--no-crrl", action="store_false", dest="crrl",
        help="Disable CRRL residual clipping.",
    )
    parser.add_argument(
        "--start-env-id", type=int, default=0,
        help="Starting env_id for port isolation (default: 0). "
             "Use >=24 to avoid port conflicts with a P=24 training run.",
    )
    # C2: training health monitoring
    parser.add_argument(
        "--enable-health-monitoring", action="store_true", default=False,
        help="Enable TrainingHealthCallback for multi-trigger health monitoring "
             "(T1: collisions, T2: critic loss, T3: Q1 mean). Off-policy only.",
    )
    parser.add_argument(
        "--enable-checkpoint-gating", action="store_true", default=False,
        help="Enable CheckpointGatedRollbackCallback for automatic rollback "
             "when M3b rate drops below 50%% of rolling best (H3). Off-policy only.",
    )
    # Campaign 3: progress monitor
    parser.add_argument(
        "--enable-progress-monitor", action="store_true", default=False,
        help="Enable TrainingProgressMonitorCallback (Campaign 3 early stopping).",
    )
    parser.add_argument(
        "--progress-failure-threshold", type=float, default=0.05,
        help="M3b failure threshold for progress monitor (default: 0.05).",
    )
    # Coordinated intervention stack
    parser.add_argument(
        "--w-barrier", type=float, default=None,
        help="Weight for barrier reward inside M3b (C3o V1, default 0.0=disabled).",
    )
    parser.add_argument(
        "--w-barrier-outside", type=float, default=None,
        help="Weight for barrier reward outside M3b (approach zone, C3o V1).",
    )
    parser.add_argument(
        "--cbf-violation-scale", type=float, default=None,
        help="CBF violation penalty multiplier (C3o V1, default 2.0).",
    )
    parser.add_argument(
        "--barrier-proximity-gate", type=float, default=None,
        help="Barrier proximity gate in error space (C3o V1, default 1.5).",
    )
    parser.add_argument(
        "--barrier-velocity-ema", type=float, default=None,
        help="Barrier velocity EMA alpha (C3o V1, default 0.3).",
    )
    parser.add_argument(
        "--velocity-gate-k", type=float, default=None,
        help="Sigmoid velocity gate sharpness k (C3q V1, default 15.0).",
    )
    parser.add_argument(
        "--forward-bias-ms", type=float, default=None,
        help="Forward speed bias in m/s (C3o V3, default 0.0=disabled).",
    )
    parser.add_argument(
        "--clamp-behind-positive-ms", type=float, default=None,
        help="FCLAMP-20260901: upper bound on the policy's ACCELERATING speed "
             "residual, m/s (config default 0.10). Omitted = no override, so "
             "behaviour is byte-identical to the pin. Exists so the clamp is a "
             "per-run election recorded in the config snapshot, letting two "
             "boxes fly DIFFERENT clamps on IDENTICAL code -- a config-constant "
             "edit would have put the two arms on different pins, which is the "
             "wrong-harness hazard this campaign is explicitly guarding.",
    )
    parser.add_argument(
        "--w-hold-fwd", type=float, default=None,
        help="C3q-B: Additive braking penalty weight inside in_m3b (default 0.5).",
    )
    parser.add_argument(
        "--bias-smooth-alpha", type=float, default=None,
        help="Bias EMA smoothing alpha (C3o V3, default 0.15).",
    )
    parser.add_argument(
        "--ema-alpha-speed", type=float, default=None,
        help="EMA alpha for policy speed delta (C3o V4, 1.0=passthrough).",
    )
    parser.add_argument(
        "--speed-slew-rate", type=float, default=None,
        help="Speed slew rate limit in m/s^2 (C3o V4, 100.0=unlimited).",
    )
    parser.add_argument(
        "--caps-actor-lambda", type=float, default=0.0,
        help="CAPS actor-loss temporal smoothness lambda (C3o V2, 0.0=disabled).",
    )
    parser.add_argument(
        "--inject-zero-action", action="store_true", default=False,
        help="Inject literal [0,0,0] action through normal path (C3o identity proof).",
    )

    args = parser.parse_args()

    profile_cfg = _resolve_or_default_profile(args)
    # If profile is non-curriculum, override CURRICULUM_STAGES[args.stage] with profile values.
    # The curriculum-DR sampler will then return clean values regardless of stage index.
    if args.profile != "curriculum":
        _stage_cfg = config.CURRICULUM_STAGES[args.stage]
        config.CURRICULUM_STAGES[args.stage] = dataclasses.replace(
            _stage_cfg,
            wind_max_ms=profile_cfg["wind_max_ms"],
            wind_turb=profile_cfg["wind_turb"],
            gps_accuracy=profile_cfg["gps_accuracy"],
        )
        logger.info(
            "[PROFILE] %s applied to stage %d: "
            "wind_max_ms=%.2f wind_turb=%.2f gps_accuracy=%.3f leader_profile=%r",
            args.profile, args.stage,
            profile_cfg["wind_max_ms"], profile_cfg["wind_turb"],
            profile_cfg["gps_accuracy"], profile_cfg["leader_profile"],
        )
        # leader_profile override is plumbed via HARNESS_LEADER_PROFILE env var
        # consumed by env_formation_sitl.py:554-560 (existing route-α path).
        if profile_cfg["leader_profile"]:
            os.environ["HARNESS_LEADER_PROFILE"] = profile_cfg["leader_profile"]

    # Per-campaign checkpoint isolation: auto-derive from --run-name
    if args.run_name and args.checkpoint_dir == "checkpoints/default":
        args.checkpoint_dir = f"checkpoints/{args.run_name}"

    # C006 P1→P7: n_envs bounds (P7 cloud targets P=8-20 on c7i.16xlarge)
    assert 1 <= args.n_envs <= 24, (
        f"n_envs={args.n_envs} out of range [1, 24]. "
        f"P>24 risks port collisions above userspace ceiling."
    )
    logger.info(f"n_envs={args.n_envs}")

    # T-457 launch preconditions — fail-closed, and FIRST in the cluster so a
    # refusal happens before anything else in main() can spawn or allocate.
    # All three are far above vec_env creation and model construction; see the
    # module-level block above for the rationale on each.
    check_tmux_precondition(unsafe=args.unsafe)
    check_g4_truth_channel_precondition()
    check_g4_align_center_precondition()

    # P1-b (FD_CURRICULUM_V3) launch preconditions — fail-closed at parse time.
    if config.FD_CURRICULUM_V3:
        assert os.environ.get("FD_TRAIN_BACKEND", "") == "json", (
            "FD_CURRICULUM_V3 requires FD_TRAIN_BACKEND=json — wind is "
            "physically live only on the FDM Dryden channel (#465).")
        assert os.environ.get("FD_DR_SEED_V2", "0") == "1", (
            "FD_CURRICULUM_V3 requires FD_DR_SEED_V2=1 (deterministic "
            "per-episode sampler; the v3 sampler refuses the global-RNG path).")
        assert args.seed is not None, (
            "FD_CURRICULUM_V3 requires --seed (per-episode SeedSequence "
            "lineage feeds the sampler and the FDM wind seed).")
        assert not args.no_curriculum, (
            "FD_CURRICULUM_V3 with --no-curriculum is contradictory.")

    if args.seed is not None:
        set_random_seed(args.seed, using_cuda=False)
        logger.info(f"[SEED] Global RNG seeded with {args.seed}")

    algorithm = args.algorithm
    is_offpolicy = algorithm in ("td3", "sac")

    # Warn if PPO-only flags are set with non-PPO algorithm
    if is_offpolicy:
        if args.ent_coef is not None:
            print(f"WARNING: --ent-coef is a PPO-only flag, ignored for {algorithm}",
                  file=sys.stderr)

    # --- Wrapper enforcement gate (v10 Change 2) ---
    if not args.unsafe and os.environ.get("FLIGHTDOCK_WRAPPER") != "1":
        logger.error(
            "[FATAL] Running without wrapper script. Orphan cleanup on\n"
            "  SIGKILL/OOM will not work. This is not optional.\n"
            "\n"
            "  Use:  ./scripts/run_drl_training.sh [args...]\n"
            "  Or:   python train.py --unsafe [args...]  (development only)\n"
            "\n"
            "  See DRL_PROCESS_RELIABILITY_HARDENING_v10.md for rationale."
        )
        sys.exit(1)

    # Write PID file and initial heartbeat for watchdog (v10 Change 4)
    _PID_FILE.write_text(str(os.getpid()))
    _HEARTBEAT_FILE.touch()

    # Set module-level state for atexit/signal cleanup
    global _n_envs_for_cleanup
    _n_envs_for_cleanup = args.n_envs

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Layer 2: Preflight cleanup — kill stale SITL processes from crashed runs
    logger.info("[Preflight] Killing stale SITL processes on our port range...")
    n_killed = _kill_all_sitl_by_port(args.n_envs, port_stride=config.PORT_STRIDE)
    if n_killed > 0:
        logger.info(
            f"[Preflight] Killed {n_killed} stale processes — sleeping 2s for port release")
        time.sleep(2.0)
    else:
        logger.info("[Preflight] No stale processes found — port range clean")

    logger.info("FlightDock DRL Training")
    logger.info(f"  algorithm:    {algorithm}")
    logger.info(f"  timesteps:    {args.total_timesteps:,}")
    logger.info(f"  n_envs:       {args.n_envs}")
    logger.info(f"  speedup:      {args.speedup}x")
    logger.info(f"  stage:        {args.stage}")
    logger.info(f"  resume:       {args.resume_from or 'None'}")
    logger.info(f"  checkpoints:  {args.checkpoint_dir}")
    logger.info(f"  logs:         {args.log_dir}")
    logger.info(f"  ui:           {args.ui}")
    logger.info(f"  pool:         {args.pool} (size={args.pool_size or 'auto'})")

    # Campaign 3: w_contact_dwell CLI override (before env/reward creation)
    if args.w_contact_dwell is not None:
        from dataclasses import replace as dc_replace
        for stage_id in config.STAGE_WEIGHTS:
            old = config.STAGE_WEIGHTS[stage_id]
            config.STAGE_WEIGHTS[stage_id] = dc_replace(
                old, w_contact_dwell=args.w_contact_dwell)
        logger.info(f"[OVERRIDE] w_contact_dwell → {args.w_contact_dwell} "
                    f"(all {len(config.STAGE_WEIGHTS)} stages)")

    # action_scale CLI override (all algorithms)
    if args.action_scale is not None:
        _stage_cfg = config.CURRICULUM_STAGES[args.stage]
        config.CURRICULUM_STAGES[args.stage] = dataclasses.replace(
            _stage_cfg, action_scale=args.action_scale,
        )
        logger.info(f"[OVERRIDE] stage {args.stage} action_scale → {args.action_scale}")

    # Campaign 3: P0-A zone-based w_ref overrides
    if args.zone_based_wref is not None:
        config.ZONE_BASED_WREF_ENABLED = args.zone_based_wref
        logger.info(f"[OVERRIDE] ZONE_BASED_WREF_ENABLED -> {args.zone_based_wref}")
    if args.w_ref_inner is not None:
        config.W_REF_INNER = args.w_ref_inner
        logger.info(f"[OVERRIDE] W_REF_INNER -> {args.w_ref_inner}")
    if args.w_ref_outer is not None:
        config.W_REF_OUTER = args.w_ref_outer
        logger.info(f"[OVERRIDE] W_REF_OUTER -> {args.w_ref_outer}")
    # Campaign 3: P0-B override
    if args.prox_envelope_suppression is not None:
        config.R_PROX_ENVELOPE_SUPPRESSION = args.prox_envelope_suppression
        logger.info(f"[OVERRIDE] R_PROX_ENVELOPE_SUPPRESSION -> {args.prox_envelope_suppression}")
    # Campaign 3: P1-A w_action_rate override (same pattern as w_contact_dwell)
    if args.w_action_rate is not None:
        from dataclasses import replace as dc_replace
        for stage_id in config.STAGE_WEIGHTS:
            old = config.STAGE_WEIGHTS[stage_id]
            config.STAGE_WEIGHTS[stage_id] = dc_replace(
                old, w_action_rate=args.w_action_rate)
        logger.info(f"[OVERRIDE] w_action_rate -> {args.w_action_rate} "
                    f"(all {len(config.STAGE_WEIGHTS)} stages)")
    # Campaign 3: P3-CRRL overrides
    if args.crrl is not None:
        config.CRRL_ENABLED = args.crrl
        logger.info(f"[OVERRIDE] CRRL_ENABLED -> {args.crrl}")
    if args.crrl_beta is not None:
        config.CRRL_BETA = args.crrl_beta
        logger.info(f"[OVERRIDE] CRRL_BETA -> {args.crrl_beta}")
    # Campaign 3c: contact dwell exponent override
    if args.contact_dwell_exponent is not None:
        config.CONTACT_DWELL_EXPONENT = args.contact_dwell_exponent
        logger.info(f"[OVERRIDE] CONTACT_DWELL_EXPONENT -> {args.contact_dwell_exponent}")
    # Campaign 3h: exit penalty overrides
    if args.w_exit_penalty is not None:
        from dataclasses import replace as dc_replace
        for stage_id in config.STAGE_WEIGHTS:
            old = config.STAGE_WEIGHTS[stage_id]
            config.STAGE_WEIGHTS[stage_id] = dc_replace(
                old, w_exit_penalty=args.w_exit_penalty)
        logger.info(f"[OVERRIDE] w_exit_penalty -> {args.w_exit_penalty} "
                    f"(all {len(config.STAGE_WEIGHTS)} stages)")
    if args.exit_penalty_dwell_threshold is not None:
        config.EXIT_DWELL_THRESHOLD_S = args.exit_penalty_dwell_threshold
        logger.info(f"[OVERRIDE] EXIT_DWELL_THRESHOLD_S -> "
                    f"{args.exit_penalty_dwell_threshold}")
    if args.exit_penalty_lower_gate is not None:
        config.EXIT_DWELL_LOWER_GATE_S = args.exit_penalty_lower_gate
        logger.info(f"[OVERRIDE] EXIT_DWELL_LOWER_GATE_S -> "
                    f"{args.exit_penalty_lower_gate}")
    # Campaign 3i: reward pivot overrides
    if args.no_approach_suppression_inside_m3b:
        config.APPROACH_SUPPRESSION_INSIDE_M3B = False
        logger.info("[OVERRIDE] APPROACH_SUPPRESSION_INSIDE_M3B -> False")
    if args.exit_behind_multiplier is not None:
        config.EXIT_BEHIND_MULTIPLIER = args.exit_behind_multiplier
        logger.info(f"[OVERRIDE] EXIT_BEHIND_MULTIPLIER -> {args.exit_behind_multiplier}")
    if args.dwell_ramp_floor is not None:
        config.DWELL_RAMP_FLOOR = args.dwell_ramp_floor
        logger.info(f"[OVERRIDE] DWELL_RAMP_FLOOR -> {args.dwell_ramp_floor}")
    # Campaign 3j: MRRL-aligned reward overrides
    if args.approach_taper_midpoint is not None:
        config.APPROACH_TAPER_MIDPOINT = args.approach_taper_midpoint
        logger.info(f"[OVERRIDE] APPROACH_TAPER_MIDPOINT -> {args.approach_taper_midpoint}")
    if args.approach_taper_steepness is not None:
        config.APPROACH_TAPER_STEEPNESS = args.approach_taper_steepness
        logger.info(f"[OVERRIDE] APPROACH_TAPER_STEEPNESS -> {args.approach_taper_steepness}")
    if args.w_overspeed is not None:
        from dataclasses import replace as dc_replace
        for stage_id in config.STAGE_WEIGHTS:
            old = config.STAGE_WEIGHTS[stage_id]
            config.STAGE_WEIGHTS[stage_id] = dc_replace(
                old, w_overspeed=args.w_overspeed)
        logger.info(f"[OVERRIDE] w_overspeed -> {args.w_overspeed} "
                    f"(all {len(config.STAGE_WEIGHTS)} stages)")
    if args.overspeed_corridor_outer is not None:
        config.OVERSPEED_CORRIDOR_OUTER = args.overspeed_corridor_outer
        logger.info(f"[OVERRIDE] OVERSPEED_CORRIDOR_OUTER -> {args.overspeed_corridor_outer}")
    if args.overspeed_corridor_inner is not None:
        config.OVERSPEED_CORRIDOR_INNER = args.overspeed_corridor_inner
        logger.info(f"[OVERRIDE] OVERSPEED_CORRIDOR_INNER -> {args.overspeed_corridor_inner}")
    # Velocity gate override
    if args.velocity_gate_k is not None:
        config.VELOCITY_GATE_K = args.velocity_gate_k
        logger.info(f"[OVERRIDE] VELOCITY_GATE_K -> {args.velocity_gate_k}")

    # Residual mode + force-zero overrides
    if args.residual_mode is not None:
        config.RESIDUAL_MODE = args.residual_mode
        logger.info(f"[OVERRIDE] RESIDUAL_MODE -> {args.residual_mode}")
    if args.rfl_activation_tier is not None:
        config.RFL_ACTIVATION_TIER = args.rfl_activation_tier
        logger.info(f"[OVERRIDE] RFL_ACTIVATION_TIER -> {args.rfl_activation_tier}")
    if args.force_zero_residual:
        config.FORCE_ZERO_RESIDUAL = True
        logger.info("[OVERRIDE] FORCE_ZERO_RESIDUAL -> True")
    if args.inject_zero_action:
        config.INJECT_ZERO_ACTION = True
        logger.info("[OVERRIDE] INJECT_ZERO_ACTION -> True")

    # Coordinated intervention stack overrides
    if args.w_barrier is not None:
        config.W_BARRIER = args.w_barrier
        logger.info(f"[OVERRIDE] W_BARRIER -> {args.w_barrier}")
    if args.w_barrier_outside is not None:
        config.W_BARRIER_OUTSIDE = args.w_barrier_outside
        logger.info(f"[OVERRIDE] W_BARRIER_OUTSIDE -> {args.w_barrier_outside}")
    if args.cbf_violation_scale is not None:
        config.CBF_VIOLATION_SCALE = args.cbf_violation_scale
    if args.barrier_proximity_gate is not None:
        config.BARRIER_PROXIMITY_GATE = args.barrier_proximity_gate
    if args.barrier_velocity_ema is not None:
        config.BARRIER_VELOCITY_EMA = args.barrier_velocity_ema
    if args.forward_bias_ms is not None:
        config.FORWARD_BIAS_MS = args.forward_bias_ms
        logger.info(f"[OVERRIDE] FORWARD_BIAS_MS -> {args.forward_bias_ms}")
    if args.clamp_behind_positive_ms is not None:
        config.CLAMP_BEHIND_POSITIVE_MS = args.clamp_behind_positive_ms
        logger.info("[OVERRIDE] CLAMP_BEHIND_POSITIVE_MS -> "
                    f"{args.clamp_behind_positive_ms}")
    if args.bias_smooth_alpha is not None:
        config.BIAS_SMOOTH_ALPHA = args.bias_smooth_alpha
    if args.ema_alpha_speed is not None:
        config.EMA_ALPHA_SPEED = args.ema_alpha_speed
        logger.info(f"[OVERRIDE] EMA_ALPHA_SPEED -> {args.ema_alpha_speed}")
    if args.speed_slew_rate is not None:
        config.SPEED_SLEW_RATE_MPS2 = args.speed_slew_rate
        logger.info(f"[OVERRIDE] SPEED_SLEW_RATE_MPS2 -> {args.speed_slew_rate}")
    if args.w_hold_fwd is not None:
        config.W_HOLD_FWD = args.w_hold_fwd
        logger.info(f"[OVERRIDE] W_HOLD_FWD -> {args.w_hold_fwd}")
    # CAPS write-back — was the only CLI param missing config propagation.
    # Prior: config.CAPS_ACTOR_LAMBDA stayed at config.py default (0.01) even
    # when CLI set --caps-actor-lambda 0.0, making config_snapshot.json misleading.
    config.CAPS_ACTOR_LAMBDA = args.caps_actor_lambda
    logger.info(f"[OVERRIDE] CAPS_ACTOR_LAMBDA -> {args.caps_actor_lambda}")

    # === Centralized reference-mode derived mutations ===
    # ALL mode-dependent config changes in ONE place, AFTER all CLI overrides.
    # No mode-conditional logic in config.py — config.py holds only raw defaults.
    if config.RESIDUAL_MODE == "reference":
        config.CRRL_ENABLED = False  # CRRL inapplicable to position offsets
        # OBS_NORMS for action memory channels — use scale if > 0, else 1.0.
        # Patch D 2026-05-06: ACTION_DIM=3 (cross-axis restored). ACTION_SCALES_REFERENCE is
        # [behind, cross, below] (3-element); previous 2-element layout was C3X Intervention 1.
        config.OBS_NORMS[35] = config.ACTION_SCALES_REFERENCE[0] or 1.0  # behind
        config.OBS_NORMS[36] = config.ACTION_SCALES_REFERENCE[1] or 1.0  # cross (was hardcoded 1.0 under 2D amputation)
        config.OBS_NORMS[37] = config.ACTION_SCALES_REFERENCE[2] or 1.0  # below (was index [1] under 2D layout)
        config.PASSTHROUGH_L2_THRESHOLD = 0.025  # meters (not mixed units)
        # Default action_scale=1.0 in reference mode (unless explicitly overridden)
        if args.action_scale is None:
            _stage_cfg = config.CURRICULUM_STAGES[args.stage]
            config.CURRICULUM_STAGES[args.stage] = dataclasses.replace(
                _stage_cfg, action_scale=1.0,
            )
            logger.info(f"[REFERENCE MODE] stage {args.stage} action_scale → 1.0 (default)")
        logger.info("[REFERENCE MODE] Applied derived config mutations: "
                    "CRRL_ENABLED=False, OBS_NORMS[35:38] updated, "
                    f"PASSTHROUGH_L2_THRESHOLD={config.PASSTHROUGH_L2_THRESHOLD}")

    # Curriculum
    curriculum = CurriculumManager(initial_stage=args.stage)

    # C006 P5: Hard-disable pool mode to prevent P>n_envs violation.
    # Pool mode creates extra SITL instances that could exceed the validated
    # parallelism level. Disabled until pool-mode is re-validated at P>1.
    if args.pool:
        pool_size = args.pool_size or (args.n_envs + 2)
        assert pool_size <= args.n_envs, (
            f"Pool size {pool_size} > n_envs {args.n_envs}. "
            f"Pool mode disabled for C006 to prevent P>{args.n_envs} violation."
        )

    # Optional: Start SITL pool (Approach D)
    pool = None
    if args.pool:
        from drl.sitl_pool import SITLPool

        warmup_target = config.CURRICULUM_STAGES[args.stage].warmup_target
        pool = SITLPool(
            n_envs=args.n_envs,
            pool_size=args.pool_size,
            speedup=args.speedup,
            warmup_target=warmup_target,
            ui_mode=args.ui,
        )
        pool.start()
        logger.info("[SITLPool] Waiting for initial warmup slots to be ready...")
        # Give pool time to warm up at least n_envs slots
        deadline = time.monotonic() + 600  # 10 min max
        while pool.ready_count < args.n_envs and time.monotonic() < deadline:
            time.sleep(5.0)
            logger.info(f"  {pool.ready_count}/{args.n_envs} slots ready...")
        logger.info(f"[SITLPool] {pool.ready_count}/{args.n_envs} slots ready. Proceeding.")

    # Create vectorized environment
    # Pool mode uses DummyVecEnv (in-process) because the SITLPool contains
    # threading primitives (Queue, Event, Thread) that cannot be pickled
    # across SubprocVecEnv process boundaries. DummyVecEnv runs all envs
    # in the main process, which is fine because the per-env step() is
    # I/O-bound (waiting on MAVLink) not CPU-bound.
    env_fns = [
        make_env(env_id=args.start_env_id + i, speedup=args.speedup,
                 curriculum=curriculum, ui_mode=args.ui, pool=pool,
                 wind_channels=args.obs_wind_channels,
                 seed=args.seed)
        for i in range(args.n_envs)
    ]

    # Gamma for VecNormalize — use algorithm-specific config
    if algorithm == "td3":
        gamma = config.TD3_CONFIG.gamma
    elif algorithm == "sac":
        gamma = config.SAC_CONFIG.gamma
    else:
        gamma = config.PPO_CONFIG.gamma

    # CONSTRAINT: SubprocVecEnv MUST be created from the main thread.
    # PR_SET_PDEATHSIG fires when the CREATING THREAD dies, not the process.
    # Moving creation to a transient worker thread would cause premature kills.
    if pool is not None:
        logger.info("  Using DummyVecEnv (pool mode — non-picklable pool object)")
        vec_env = DummyVecEnv(env_fns)
    elif args.n_envs == 1:
        # P=1: DummyVecEnv — env runs in-process. No forkserver, no IPC.
        logger.info("  Using DummyVecEnv (n_envs=1 — in-process, no forkserver)")
        vec_env = DummyVecEnv(env_fns)
    else:
        # P>1: SubprocVecEnv with forkserver. Forkserver gap exists for
        # SIGKILL — wrapper script + watchdog provide external containment.
        vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        gamma=gamma,
    )
    logger.info("  VecNormalize: FRESH (norm_obs=True, norm_reward=False)")

    # --- Model creation / resume ---
    if args.resume_from:
        logger.info(f"Loading model from {args.resume_from}")

        # Resolve checkpoint companion files (VecNormalize, replay buffer)
        resume_dir = os.path.dirname(args.resume_from)
        model_basename = os.path.basename(args.resume_from)
        if model_basename == "model.zip":
            # best_policy_final/ uses bare model.zip — companion files have
            # simple names without the model_ prefix (Campaign 3 §5.47 fix)
            norm_file = os.path.join(resume_dir, "vecnormalize.pkl")
            buffer_file = os.path.join(resume_dir, "replay_buffer.pkl")
        else:
            resume_base = model_basename.replace(
                "model_", "vecnormalize_").replace(".zip", ".pkl")
            norm_file = os.path.join(resume_dir, resume_base)
            buffer_base = model_basename.replace(
                "model_", "replay_buffer_").replace(".zip", ".pkl")
            buffer_file = os.path.join(resume_dir, buffer_base)

        if args.critic_layernorm and algorithm == "td3":
            # --- LayerNorm critic reinitialization path ---
            # Load checkpoint to extract trained actor weights, then create
            # a fresh TD3 with LayerNormTD3Policy. Critic starts randomly
            # initialized (LayerNorm architecture incompatible with checkpoint).
            logger.info("  LayerNorm critic path: loading actor weights from checkpoint...")
            q3_model = TD3.load(args.resume_from, device="cpu")

            td3_cfg = config.TD3_CONFIG
            n_actions = vec_env.action_space.shape[0]
            # ACTION_DIM=3 (cross-axis restored 2026-05-06):
            # 3-element noise sigma [speed, heading, altitude].
            noise_sigma = np.array([
                td3_cfg.noise_sigma_speed,
                td3_cfg.noise_sigma_heading,
                td3_cfg.noise_sigma_altitude,
            ])
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=noise_sigma,
            )
            # F2 (CONFIG_AUDIT): capped scaling, same rule as _create_td3.
            effective_learning_starts = effective_learning_starts_fn(
                td3_cfg, vec_env.num_envs, args.learning_starts)
            buffer_size = (td3_cfg.buffer_size_stage0 if args.stage == 0
                           else td3_cfg.buffer_size_stage1_plus)

            # Use `is not None` — `or` silently drops zero-valued CLI args.
            eff_lr = args.learning_rate if args.learning_rate is not None else td3_cfg.learning_rate
            eff_tau = args.tau if args.tau is not None else td3_cfg.tau
            eff_pd = args.policy_delay if args.policy_delay is not None else td3_cfg.policy_delay
            # Phase 5.1: gamma override
            eff_gamma = args.gamma if args.gamma is not None else td3_cfg.gamma

            # Phase 5.1: Use CustomClippedTD3 when gradient clipping requested
            ModelClass = CustomClippedTD3 if args.max_grad_norm_critic is not None else TD3
            extra_kw = {}
            if args.max_grad_norm_critic is not None:
                extra_kw["max_grad_norm_critic"] = args.max_grad_norm_critic
                logger.info(f"[GRAD_CLIP] Using CustomClippedTD3 with "
                            f"max_grad_norm_critic={args.max_grad_norm_critic}")

            # P1-B: CAPS temporal smoothness on LayerNorm resume path
            if args.caps_lambda_t > 0.0:
                extra_kw["caps_lambda_t"] = args.caps_lambda_t
                if ModelClass is not CustomClippedTD3:
                    ModelClass = CustomClippedTD3
                    extra_kw.setdefault("max_grad_norm_critic", 0.5)
                logger.info(f"[CAPS] caps_lambda_t={args.caps_lambda_t} on LayerNorm resume")

            model = ModelClass(
                policy=LayerNormTD3Policy,
                env=vec_env,
                learning_rate=eff_lr,
                buffer_size=buffer_size,
                learning_starts=effective_learning_starts,
                batch_size=td3_cfg.batch_size,
                tau=eff_tau,
                gamma=eff_gamma,
                train_freq=(td3_cfg.train_freq, "step"),
                gradient_steps=effective_gradient_steps_fn(td3_cfg, vec_env.num_envs),  # F3
                action_noise=action_noise,
                policy_delay=eff_pd,
                target_policy_noise=td3_cfg.target_policy_noise,
                target_noise_clip=td3_cfg.target_noise_clip,
                policy_kwargs=dict(net_arch=td3_cfg.net_arch),
                verbose=1,
                tensorboard_log=args.log_dir,
                **extra_kw,
            )

            # --- Actor weights ---
            if args.transfer_actor_weights:
                # Transfer W2 actor weights to new LayerNorm model
                model.actor.load_state_dict(q3_model.actor.state_dict())
                model.actor_target.load_state_dict(model.actor.state_dict())
                for (k1, v1), (k2, v2) in zip(
                        model.actor.state_dict().items(),
                        model.actor_target.state_dict().items()):
                    assert torch.equal(v1, v2), f"Actor/target mismatch at {k1}"
                logger.info("  Actor weights transferred from checkpoint, "
                            "target synced and verified.")
            else:
                # Fresh init with zero-residual bias
                init_zero_residual_td3(model)
                logger.info("  Fresh zero-residual init (no actor transfer).")

            # --- Critic weights ---
            if args.critic_shrink_perturb is not None:
                transfer_critic_weights_to_layernorm(
                    new_model=model, old_model=q3_model,
                    shrink=args.critic_shrink_perturb,
                )
                for (k1, v1), (k2, v2) in zip(
                        model.critic.state_dict().items(),
                        model.critic_target.state_dict().items()):
                    assert torch.equal(v1, v2), f"Critic/target mismatch at {k1}"
                logger.info("  Critic weights transferred with S&P (shrink=%.2f), "
                            "targets synced and verified.",
                            args.critic_shrink_perturb)
            else:
                logger.info("  Critic randomly initialized (no transfer).")

            # --- Critic optimizer ---
            if args.reset_critic_optimizer:
                model.critic.optimizer = type(model.critic.optimizer)(
                    model.critic.parameters(),
                    **{k: v for k, v in model.critic.optimizer.defaults.items()},
                )
                logger.info("  Critic optimizer reset (fresh Adam state).")

            # Dissertation artifact: Q-value baseline post-S&P, pre-training
            with torch.no_grad():
                _test_obs = torch.zeros(
                    1, model.observation_space.shape[0], device=model.device)
                _test_act = torch.zeros(
                    1, model.action_space.shape[0], device=model.device)
                _q1, _q2 = model.critic(_test_obs, _test_act)
                logger.info("[Q_BASELINE] Post-S&P Q-values at zero state: "
                            "Q1=%.4f, Q2=%.4f", _q1.item(), _q2.item())

            del q3_model

        else:
            # --- Standard resume path (no LayerNorm) ---
            # Build custom_objects for hyperparameter overrides on resume.
            # SB3's .load() restores ALL hyperparams from the checkpoint .zip.
            # config.py values are IGNORED on resume. custom_objects is the
            # officially documented SB3 mechanism for post-load overrides.
            custom_objects = {}
            if args.learning_rate is not None:
                custom_objects["learning_rate"] = args.learning_rate
                logger.info(
                    f"Learning rate override via --learning-rate: "
                    f"{args.learning_rate} (will replace checkpoint's saved value)"
                )
            if args.tau is not None:
                custom_objects["tau"] = args.tau
                logger.info(f"  tau override via --tau: {args.tau}")
            if args.policy_delay is not None:
                custom_objects["policy_delay"] = args.policy_delay
                logger.info(f"  policy_delay override via --policy-delay: "
                            f"{args.policy_delay}")
            # Phase 5.1: gamma override via custom_objects
            if args.gamma is not None:
                custom_objects["gamma"] = args.gamma
                logger.info(f"  gamma override via --gamma: {args.gamma}")
            if args.learning_starts is not None:
                custom_objects["learning_starts"] = args.learning_starts
                logger.info(f"  learning_starts override via --learning-starts: "
                            f"{args.learning_starts} (replaces checkpoint's saved value)")

            co = custom_objects if custom_objects else None

            if algorithm == "td3":
                model = TD3.load(args.resume_from, env=vec_env, custom_objects=co)
            elif algorithm == "sac":
                model = SAC.load(args.resume_from, env=vec_env, custom_objects=co)
            else:
                model = PPO.load(args.resume_from, env=vec_env, custom_objects=co)

            # CPRIME-808 R2: same-architecture critic reset on the plain path.
            if args.reset_critic:
                assert algorithm == "td3", "--reset-critic is TD3-only"
                reset_critic_witness = reset_critic_in_place(model)

            # Phase 5.1 / 7A: promote TD3 to CustomClippedTD3 for critic
            # gradient clipping. Changing __class__ avoids the pickle failure
            # caused by a bound-method monkey-patch (the bound method kept a
            # reference to `self`, creating a circular chain that included the
            # env's thread locks).
            if args.max_grad_norm_critic is not None and algorithm == "td3":
                model.max_grad_norm_critic = args.max_grad_norm_critic
                model.__class__ = CustomClippedTD3
                logger.info(f"[GRAD_CLIP] Using CustomClippedTD3 with "
                            f"max_grad_norm_critic={args.max_grad_norm_critic}")

            # P1-B: CAPS on resume
            if args.caps_lambda_t > 0.0 and algorithm == "td3":
                model.caps_lambda_t = args.caps_lambda_t
                if not isinstance(model, CustomClippedTD3):
                    model.max_grad_norm_critic = 0.5
                    model.__class__ = CustomClippedTD3
                logger.info(f"[CAPS] caps_lambda_t={args.caps_lambda_t} on resume")

            # Verify overrides took effect
            actual_lr = model.learning_rate
            logger.info(f"Effective learning rate after load: {actual_lr}")
            if args.learning_rate is not None:
                assert abs(actual_lr - args.learning_rate) < 1e-10, (
                    f"Learning rate override FAILED: expected {args.learning_rate}, "
                    f"got {actual_lr}. DO NOT PROCEED — wrong lr."
                )
            if args.tau is not None and hasattr(model, 'tau'):
                assert abs(model.tau - args.tau) < 1e-10, (
                    f"tau override FAILED: expected {args.tau}, got {model.tau}"
                )
            if args.policy_delay is not None and hasattr(model, 'policy_delay'):
                assert model.policy_delay == args.policy_delay, (
                    f"policy_delay override FAILED: expected {args.policy_delay}, "
                    f"got {model.policy_delay}"
                )

        if args.learning_starts is not None and hasattr(model, 'learning_starts'):
            assert model.learning_starts == args.learning_starts, (
                f"learning_starts override FAILED: expected {args.learning_starts}, "
                f"got {model.learning_starts}. DO NOT PROCEED."
            )
            logger.info(f"  learning_starts verified: {model.learning_starts}")

        # Phase 5.1: gamma assertion (both paths — LayerNorm and standard)
        if args.gamma is not None and hasattr(model, 'gamma'):
            assert abs(model.gamma - args.gamma) < 1e-10, (
                f"gamma override FAILED: expected {args.gamma}, got {model.gamma}")
            logger.info(f"  gamma verified: {model.gamma}")

        # Load VecNormalize stats if available
        if os.path.exists(norm_file):
            logger.info(f"Loading VecNormalize from {norm_file}")
            vec_env = VecNormalize.load(norm_file, vec_env.venv)
            model.set_env(vec_env)

        # Off-policy resume: load or create replay buffer
        if is_offpolicy:
            if args.fresh_buffer:
                from stable_baselines3.common.buffers import ReplayBuffer
                # Phase 7A: Scale buffer with parallelism. At P>8, 200K provides only
                # ~11K slots/env causing rapid FIFO eviction and distribution shift.
                fresh_size = config.BUFFER_SIZE_HIGH_P if args.n_envs > 8 else config.BUFFER_SIZE_DEFAULT
                print(f"[BUFFER] Size set to {fresh_size:,} (n_envs={args.n_envs})")
                model.replay_buffer = ReplayBuffer(
                    buffer_size=fresh_size,
                    observation_space=model.observation_space,
                    action_space=model.action_space,
                    device=model.device,
                    n_envs=model.n_envs,
                )
                print(f"[BUFFER] Fresh buffer created: size={fresh_size}")
            elif os.path.exists(buffer_file):
                logger.info(f"Loading replay buffer from {buffer_file}")
                model.load_replay_buffer(buffer_file)
                old_buf = model.replay_buffer
                # Handle n_envs mismatch (e.g. buffer saved at P=18, resuming at P=24)
                if old_buf.n_envs != model.n_envs:
                    from stable_baselines3.common.buffers import ReplayBuffer
                    old_n = old_buf.n_envs
                    new_n = model.n_envs
                    valid = old_buf.buffer_size if old_buf.full else old_buf.pos
                    total = valid * old_n
                    # Flatten old (valid, old_n, ...) -> (total, ...) then re-chunk to (slots, new_n, ...)
                    new_slots = int(np.ceil(total / new_n))
                    pad = new_slots * new_n - total
                    def _rebucket(arr):
                        # Infer trailing dims from array shape (skip buffer_size, n_envs)
                        trailing = arr.shape[2:]
                        flat = arr[:valid].reshape(total, *trailing)
                        if pad > 0:
                            flat = np.concatenate([flat, np.zeros((pad, *trailing), dtype=flat.dtype)])
                        return flat.reshape(new_slots, new_n, *trailing)
                    # SB3 divides buffer_size by n_envs internally, so pass
                    # new_slots * new_n to get at least new_slots rows.
                    raw_size = max(old_buf.buffer_size * old_n, new_slots * new_n)
                    new_buf = ReplayBuffer(
                        buffer_size=raw_size,
                        observation_space=model.observation_space,
                        action_space=model.action_space,
                        device=model.device,
                        n_envs=new_n,
                    )
                    actual_cap = new_buf.buffer_size  # after SB3's //n_envs
                    new_buf.observations[:new_slots] = _rebucket(old_buf.observations)
                    new_buf.next_observations[:new_slots] = _rebucket(old_buf.next_observations)
                    new_buf.actions[:new_slots] = _rebucket(old_buf.actions)
                    new_buf.rewards[:new_slots] = _rebucket(old_buf.rewards)
                    new_buf.dones[:new_slots] = _rebucket(old_buf.dones)
                    new_buf.pos = new_slots % actual_cap
                    new_buf.full = new_slots >= actual_cap
                    model.replay_buffer = new_buf
                    logger.info(f"  [BUFFER] n_envs mismatch: rebucketed {old_n}->"
                                f"{new_n}, {new_buf.size()} slots ({total} transitions)")
                logger.info(f"  Replay buffer: {model.replay_buffer.size()} transitions")
            else:
                logger.warning(f"No replay buffer found at {buffer_file} — "
                               f"starting with empty buffer")

        if args.freeze_normalize:
            logger.info("Freezing VecNormalize stats (training=False)")
            vec_env.training = False
    else:
        # PPO-specific CLI overrides (ignored for td3/sac)
        if algorithm == "ppo":
            ppo_cfg = config.PPO_CONFIG
            if args.ent_coef is not None:
                ppo_cfg = dataclasses.replace(ppo_cfg, ent_coef=args.ent_coef)
                logger.info(f"  [CLI override] ent_coef = {args.ent_coef}")

        # Create model based on algorithm
        if algorithm == "td3":
            model = _create_td3(vec_env, config.TD3_CONFIG, args.stage, args.log_dir, args)
        elif algorithm == "sac":
            model = _create_sac(vec_env, config.SAC_CONFIG, args.stage, args.log_dir, args)
        else:
            model = _create_ppo(vec_env, ppo_cfg, args.n_envs, args.log_dir)

    # Phase 6: Model-class assertion
    print(f"[MODEL] Class: {type(model).__name__}")
    if args.max_grad_norm_critic is not None:
        assert isinstance(model, CustomClippedTD3) or hasattr(model, 'max_grad_norm_critic'), (
            f"--max-grad-norm-critic set but model is {type(model).__name__} "
            f"without max_grad_norm_critic attribute"
        )

    # Store reference for signal handler cleanup — AFTER resume block,
    # because --resume-from may reassign vec_env to a new VecNormalize.
    global _vec_env_ref
    _vec_env_ref = vec_env

    # Phase 7A: Rebuild VectorizedActionNoise if n_envs changed from checkpoint
    if (is_offpolicy and args.resume_from
            and hasattr(model, "action_noise") and model.action_noise is not None):
        from stable_baselines3.common.noise import VectorizedActionNoise
        noise = model.action_noise
        if isinstance(noise, VectorizedActionNoise) and noise.n_envs != args.n_envs:
            old_n = noise.n_envs
            base = noise.noises[0]  # NormalActionNoise with _sigma
            rebuilt = VectorizedActionNoise(base, args.n_envs)
            model.action_noise = rebuilt
            logger.info(f"  [NOISE] Rebuilt VectorizedActionNoise: {old_n} -> {args.n_envs}")

    # C006 P3/P10: Log effective UTD ratio and action noise for reproducibility
    if is_offpolicy:
        algo_cfg = config.TD3_CONFIG if algorithm == "td3" else config.SAC_CONFIG
        logger.info(f"  Effective UTD: {algo_cfg.gradient_steps}/{args.n_envs} = "
                    f"{algo_cfg.gradient_steps / args.n_envs:.2f}")
        if algorithm == "td3":
            # NOISEAUDIT-792: these per-axis config values seed the noise array
            # (shape + initial value) but are OVERWRITTEN below by
            # np.full(shape, args.exploration_noise_sigma) before learn() --
            # what flies is ISOTROPIC. Logging them as the flown sigma was
            # false in every archived campaign log. Labelled INERT here; the
            # authoritative line is emitted after the overwrite.
            logger.info(f"  Action noise σ config per-axis (INERT, "
                        f"overwritten before learn()): "
                        f"speed={algo_cfg.noise_sigma_speed}, "
                        f"heading={algo_cfg.noise_sigma_heading}, "
                        f"altitude={algo_cfg.noise_sigma_altitude}")

    # Phase 6: Configuration verification banner
    if is_offpolicy and args.resume_from:
        buffer_status = "FRESH" if args.fresh_buffer else "LOADED"
        print("=" * 60)
        print("[PHASE6] Configuration verification:")
        print(f"  gamma = {model.gamma}")
        print(f"  max_grad_norm_critic = {getattr(model, 'max_grad_norm_critic', 'N/A')}")
        print(f"  critic_abort_threshold = {args.critic_abort_threshold}")
        print(f"  critic_hard_ceiling = {args.critic_hard_ceiling}")
        print(f"  critic_hard_ceiling_persist = {args.critic_hard_ceiling_persist}")
        print(f"  actor_freeze_min_steps = {args.actor_freeze_min_steps}")
        print(f"  actor_freeze_max_steps = {args.actor_freeze_max_steps}")
        print(f"  actor_freeze_critic_threshold = {args.actor_freeze_critic_threshold}")
        print(f"  R_APPROACH_THRESHOLD_M = {config.R_APPROACH_THRESHOLD_M}")
        print(f"  R_APPROACH_MAX = {config.R_APPROACH_MAX}")
        print(f"  w_ref Stage 0 = {config.STAGE_WEIGHTS[0].w_ref}")
        print(f"  [BUFFER] {buffer_status} — size={model.replay_buffer.buffer_size}")
        print(f"  learning_rate = {model.learning_rate}")
        print(f"  tau = {model.tau}")
        print(f"  policy_delay = {model.policy_delay}")
        print(f"  model_class = {type(model).__name__}")
        print("=" * 60)

    # --- Callbacks (algorithm-aware) ---
    is_resume = bool(args.resume_from)
    callbacks = [
        MetricsCallback(
            log_path=os.path.join(args.log_dir, "metrics.csv"),
            resume=is_resume,
        ),
        HeartbeatCallback(interval=100),
        HaltSentinelCallback(
            halt_path=os.path.join(args.log_dir, "HALT_REQUESTED"),
            check_every=100,
        ),
    ]

    n_envs = args.n_envs

    if is_offpolicy:
        # Off-policy: Atomic checkpoint with replay buffer saving (v19 P3)
        # C006 P3: Scale save_freq by n_envs — each env contributes steps in parallel,
        # so total timesteps accumulate n_envs× faster.
        base_save_freq = 3600
        effective_save_freq = max(base_save_freq // n_envs, 1)
        callbacks.append(AtomicOffPolicyCheckpointCallback(
            save_freq=effective_save_freq,
            save_path=args.checkpoint_dir,
            name_prefix="model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        ))
        callbacks.append(OffPolicyDiagnosticsCallback(
            log_dir=args.log_dir,
            algorithm=algorithm,
            resume=is_resume,
        ))
        # C3X Intervention 3: per-episode Q-value distribution across DBB
        # buckets — scientific discriminator between H1 (controller
        # antagonism) and the dense reward dominance weak form. See
        # docs/C3X_CAMPAIGN_PLAN.md §3.3 and §6.4 (IC4 gate).
        callbacks.append(QValueDistributionCallback(
            log_path=os.path.join(args.log_dir, "q_value_distribution.csv"),
        ))
        # Campaign 2b: Unified VecNormalize scheduler (replaces separate freeze/unfreeze)
        if args.unfreeze_normalize_for > 0 and args.resume_from:
            vecnorm_freeze_after = args.unfreeze_normalize_for  # e.g., 100
            vecnorm_unfreeze_at_init = True
        else:
            vecnorm_freeze_after = config.VECNORM_CALIBRATION_EPISODES  # 30
            vecnorm_unfreeze_at_init = False

        callbacks.append(VecNormalizeScheduleCallback(
            vec_env=vec_env,
            freeze_after_episodes=vecnorm_freeze_after,
            unfreeze_at_init=vecnorm_unfreeze_at_init,
            verbose=1,
        ))
        callbacks.append(BufferDilutionCheckCallback(log_dir=args.log_dir, resume=is_resume))
        # v19 P8: TD3-only automated SAC fallback on zero-residual trap
        if algorithm == "td3":
            callbacks.append(ZeroResidualFallbackCallback(
                log_dir=args.log_dir,
                checkpoint_dir=args.checkpoint_dir,
            ))
    else:
        # On-policy (PPO): custom checkpoint + PPO diagnostics
        callbacks.append(CheckpointCallback(
            save_freq=2,
            checkpoint_dir=args.checkpoint_dir,
        ))
        callbacks.append(PolicyDiagnosticsCallback(log_dir=args.log_dir, resume=is_resume))

    # C2: training health monitoring (off-policy only)
    if args.enable_health_monitoring and is_offpolicy:
        callbacks.append(TrainingHealthCallback(
            checkpoint_dir=args.checkpoint_dir,
        ))

    # C2: checkpoint-gated rollback (off-policy only, H3)
    if args.enable_checkpoint_gating and is_offpolicy:
        callbacks.append(CheckpointGatedRollbackCallback(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
        ))

    if args.enable_progress_monitor and is_offpolicy:
        from drl.callbacks import TrainingProgressMonitorCallback
        callbacks.append(TrainingProgressMonitorCallback(
            failure_threshold=args.progress_failure_threshold,
        ))
        logger.info(f"[CALLBACK] TrainingProgressMonitorCallback "
                    f"(failure_threshold={args.progress_failure_threshold})")

    callbacks.append(RewardMonitorCallback(
        log_dir=args.log_dir,
        min_rollouts=10,
        window=8,
        auto_stop=args.auto_stop_on_flat,
    ))

    # C006 Phase 7: exploration noise annealing
    # (P1 Rev 3) default-init so config_snapshot can always reference it.
    _noise_resume_offset: int = 0
    if (is_offpolicy and hasattr(model, "action_noise")
            and model.action_noise is not None):
        # Override initial sigma from CLI
        noise = model.action_noise
        from stable_baselines3.common.noise import VectorizedActionNoise
        if isinstance(noise, VectorizedActionNoise):
            for n in noise.noises:
                n._sigma = np.full(n._sigma.shape, args.exploration_noise_sigma)
        else:
            noise._sigma = np.full(
                noise._sigma.shape, args.exploration_noise_sigma)
        # NOISEAUDIT-792: report the sigma that WILL FLY, read back from the
        # model itself rather than from any config constant. If the overwrite
        # above is ever removed, this line changes with it -- which is the
        # whole point: the record must track the object, not a constant that
        # merely used to describe it.
        _flown = model.action_noise
        _flown_sigma = (_flown.noises[0]._sigma
                        if isinstance(_flown, VectorizedActionNoise)
                        else _flown._sigma)
        logger.info(
            "[NOISE] flown sigma (post-override, per-axis as constructed): %s"
            " | anneal %.3f -> %.3f over %d steps",
            np.array2string(np.asarray(_flown_sigma), precision=4),
            args.exploration_noise_sigma,
            args.exploration_noise_sigma_final,
            args.exploration_noise_anneal_steps)
        if args.reset_exploration_schedule_on_resume:
            _noise_resume_offset = int(model.num_timesteps)
            logger.info(
                "[NOISE_ANNEAL] reset-on-resume ENABLED: "
                "resume_offset_steps=%d (captured from checkpoint); "
                "anneal runs over the next %d steps from initial_sigma=%.3f "
                "to final_sigma=%.3f",
                _noise_resume_offset,
                args.exploration_noise_anneal_steps,
                args.exploration_noise_sigma,
                args.exploration_noise_sigma_final,
            )
        else:
            _noise_resume_offset = 0
        callbacks.append(NoiseAnnealCallback(
            initial_sigma=args.exploration_noise_sigma,
            final_sigma=args.exploration_noise_sigma_final,
            anneal_steps=args.exploration_noise_anneal_steps,
            resume_offset_steps=_noise_resume_offset,
        ))

    # C006 Phase 4→7: critic divergence abort with hysteresis (always active)
    if is_offpolicy:
        callbacks.append(CriticDivergenceAbortCallback(
            threshold=args.critic_abort_threshold,
            hard_ceiling=args.critic_hard_ceiling,
            window_size=args.critic_abort_window,   # GUARDWIN: was a literal 5
            warmup_steps=10000,
            hard_ceiling_persist=args.critic_hard_ceiling_persist,
        ))

    # C3w: structural ceiling halt at ts=50,000 — multi-signal 6-condition
    # consolidation-failure detector. Halts only if reward is firing AND
    # critic stable AND four "stuck" signals all hold (M3b count, DBB
    # trend, entries trend, dwell-event frequency). See callbacks.py
    # StructuralCeilingCallback for full design rationale.
    if is_offpolicy:
        callbacks.append(StructuralCeilingCallback(
            gate_timestep=50000,
            window_size=25,
            head_tail_size=10,
            dbb_growth_threshold=0.05,
            entries_decline_threshold=-2.0,
            m3b_consolidation_threshold=1,
            r_dwell_active_threshold=0.01,
            critic_max_threshold=80.0,
            metrics_csv_path=os.path.join(args.log_dir, "metrics.csv"),
            opd_csv_path=os.path.join(args.log_dir, "offpolicy_diagnostics.csv"),
            log_dir=args.log_dir,
        ))

    # L9.2: H12 conjunction halt (critic grad norm monotonicity + loss gate)
    # E-2: keep the handle — the post-learn() classifier must distinguish an
    # H12 divergence halt from a budget stop in final_run_summary.
    h12_callback = None
    if is_offpolicy:
        # F-3 (T-199): threshold + rise floor are GENERATOR-DERIVED
        # (tools/h12_calibration.py on the banked healthy corpus) after the
        # WS2 false halt — see analysis/h12stage_20260802T052000Z/. The old
        # literal 30.0 sat below the healthy operating level (corpus max
        # L25 43.12) and monotonicity had no floor against ladder slope.
        h12_callback = H12ConjunctionCallback(
            gate_timestep=10000,
            window_size=25,
            n_windows=3,
            loss_threshold=config.H12_LOSS_THRESHOLD,
            min_rise_frac=config.H12_MIN_RISE_FRAC,
            opd_csv_path=os.path.join(args.log_dir, "offpolicy_diagnostics.csv"),
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
        )
        callbacks.append(h12_callback)

    # E-4: behavioral rolling-window early stop (standard equipment; params
    # PENDING on the C006 back-test — see config.py).
    early_stop_callback = None
    if is_offpolicy and config.EARLY_STOP_ENABLED:
        early_stop_callback = BehavioralEarlyStopCallback(
            window_episodes=config.EARLY_STOP_WINDOW_EPISODES,
            delta_frac=config.EARLY_STOP_DELTA_FRAC,
            patience_windows=config.EARLY_STOP_PATIENCE_WINDOWS,
            min_timesteps=config.EARLY_STOP_MIN_TIMESTEPS,
            metric=config.EARLY_STOP_METRIC,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            floor_windows=config.EARLY_STOP_FLOOR_WINDOWS,
            floor_eps=config.EARLY_STOP_FLOOR_EPS,
        )
        callbacks.append(early_stop_callback)

    # C006 Phase 7: passthrough monitoring gate (surveillance only)
    if is_offpolicy:
        callbacks.append(PassthroughMonitorCallback(
            threshold=0.90,
            consecutive_episodes=50,
            action_l2_threshold=config.PASSTHROUGH_L2_THRESHOLD,
            alert_path=os.path.join(args.log_dir, "passthrough_alert.json"),
        ))

    # C006 Phase 4: actor freeze during critic calibration
    if args.actor_freeze_until_critic_stable:
        callbacks.append(ActorFreezeCallback(
            min_steps=args.actor_freeze_min_steps,
            max_steps=args.actor_freeze_max_steps,
            critic_threshold=args.actor_freeze_critic_threshold,
            checkpoint_dir=args.checkpoint_dir,
        ))

    # C006 Phase 7: config verification callback
    if args.config_verify:
        expected_config: dict = {
            "gamma": model.gamma,
            "learning_rate": model.learning_rate,
            "model_class": type(model).__name__,
        }
        if is_offpolicy:
            expected_config["tau"] = model.tau
            expected_config["policy_delay"] = model.policy_delay
            if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
                expected_config["buffer_size"] = model.replay_buffer.buffer_size

        # Campaign 3c: record dwell exponent and weight for config verification
        expected_config["contact_dwell_exponent"] = config.CONTACT_DWELL_EXPONENT
        if args.w_contact_dwell is not None:
            expected_config["w_contact_dwell"] = args.w_contact_dwell
        expected_config["action_scale"] = config.CURRICULUM_STAGES[args.stage].action_scale
        # Campaign 3h: exit penalty config snapshot
        expected_config["exit_dwell_threshold"] = config.EXIT_DWELL_THRESHOLD_S
        expected_config["exit_dwell_lower_gate"] = config.EXIT_DWELL_LOWER_GATE_S
        # C3w: audit-trail for full DWELL_REWARD_THRESHOLD_S dict (stage-dict
        # keys not supported by the mismatch-check whitelist, so this lives
        # in the `expected` section only for forensic record).
        expected_config["dwell_reward_threshold_s_all"] = dict(
            config.DWELL_REWARD_THRESHOLD_S)
        if args.w_exit_penalty is not None:
            expected_config["w_exit_penalty"] = args.w_exit_penalty
        # Campaign 3i: reward pivot config snapshot
        expected_config["approach_suppression_inside_m3b"] = config.APPROACH_SUPPRESSION_INSIDE_M3B
        expected_config["exit_behind_multiplier"] = config.EXIT_BEHIND_MULTIPLIER
        expected_config["dwell_ramp_floor"] = config.DWELL_RAMP_FLOOR
        # Campaign 3j: MRRL-aligned reward config snapshot
        expected_config["approach_taper_midpoint"] = config.APPROACH_TAPER_MIDPOINT
        expected_config["approach_taper_steepness"] = config.APPROACH_TAPER_STEEPNESS
        expected_config["overspeed_corridor_outer"] = config.OVERSPEED_CORRIDOR_OUTER
        expected_config["overspeed_corridor_inner"] = config.OVERSPEED_CORRIDOR_INNER
        if args.w_overspeed is not None:
            expected_config["w_overspeed"] = args.w_overspeed
        # RFL config snapshot
        expected_config["residual_mode"] = config.RESIDUAL_MODE
        expected_config["rfl_activation_tier"] = config.RFL_ACTIVATION_TIER
        expected_config["force_zero_residual"] = config.FORCE_ZERO_RESIDUAL
        # Coordinated intervention stack config snapshot
        expected_config["w_barrier"] = config.W_BARRIER
        expected_config["w_barrier_outside"] = config.W_BARRIER_OUTSIDE
        expected_config["cbf_violation_scale"] = config.CBF_VIOLATION_SCALE
        expected_config["barrier_proximity_gate"] = config.BARRIER_PROXIMITY_GATE
        expected_config["barrier_velocity_ema"] = config.BARRIER_VELOCITY_EMA
        expected_config["forward_bias_ms"] = config.FORWARD_BIAS_MS
        expected_config["clamp_behind_positive_ms"] = config.CLAMP_BEHIND_POSITIVE_MS
        expected_config["bias_smooth_alpha"] = config.BIAS_SMOOTH_ALPHA
        expected_config["ema_alpha_speed"] = config.EMA_ALPHA_SPEED
        expected_config["speed_slew_rate"] = config.SPEED_SLEW_RATE_MPS2
        expected_config["caps_actor_lambda"] = config.CAPS_ACTOR_LAMBDA
        expected_config["caps_lambda_t"] = getattr(config, "CAPS_LAMBDA_T", 0.0)
        expected_config["crrl_beta"] = getattr(args, "crrl_beta", 0.0)
        expected_config["velocity_gate_k"] = config.VELOCITY_GATE_K
        expected_config["w_hold_fwd"] = config.W_HOLD_FWD
        # Phase 0 warmup tracking
        expected_config["phase0_warmup_steps"] = getattr(args, "phase0_warmup_steps", 0)
        expected_config["buffer_seed_steps"] = getattr(args, "buffer_seed_steps", 0)
        # Warm restart config
        expected_config["transfer_actor_weights"] = getattr(args, "transfer_actor_weights", False)
        expected_config["critic_shrink_perturb"] = getattr(args, "critic_shrink_perturb", None)
        expected_config["reset_critic_optimizer"] = getattr(args, "reset_critic_optimizer", False)
        expected_config["critic_layernorm"] = getattr(args, "critic_layernorm", False)
        expected_config["max_grad_norm_critic"] = getattr(args, "max_grad_norm_critic", None)
        # H4-Alpha: exploration-noise schedule provenance
        expected_config["exploration_noise_sigma"] = getattr(args, "exploration_noise_sigma", None)
        expected_config["exploration_noise_sigma_final"] = getattr(args, "exploration_noise_sigma_final", None)
        expected_config["exploration_noise_anneal_steps"] = getattr(args, "exploration_noise_anneal_steps", None)
        expected_config["reset_exploration_schedule_on_resume"] = getattr(args, "reset_exploration_schedule_on_resume", False)
        # SD-NEW-2: provenance fields for provenance + L9 gating
        expected_config["action_dim"] = config.ACTION_DIM
        expected_config["action_scales_reference"] = list(config.ACTION_SCALES_REFERENCE)
        expected_config["learning_starts"] = getattr(args, "learning_starts", None)
        expected_config["n_envs"] = args.n_envs
        expected_config["seed"] = getattr(args, "seed", None)
        expected_config["total_timesteps"] = args.total_timesteps
        expected_config["algorithm"] = args.algorithm
        try:
            _sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
        except Exception:
            _sha = "unknown"
        expected_config["commit_sha"] = _sha
        # P1 Rev 3 (2026-04-23) — schema v3 additions
        expected_config["stage"] = int(getattr(args, "stage", 0) or 0)
        expected_config["resume_offset_steps"] = int(_noise_resume_offset)
        expected_config["train_argv"] = list(sys.argv)
        _wrap_raw = os.environ.get("FLIGHTDOCK_WRAPPER_ARGV_JSON", "[]")
        try:
            _wrap_parsed = json.loads(_wrap_raw)
            expected_config["wrapper_argv"] = (
                _wrap_parsed if isinstance(_wrap_parsed, list) else [])
        except (ValueError, TypeError):
            expected_config["wrapper_argv"] = []
        expected_config["snapshot_schema_version"] = 3

        callback_refs: dict = {}
        for cb in callbacks:
            if isinstance(cb, ActorFreezeCallback):
                callback_refs["actor_freeze"] = cb
                expected_config["actor_freeze_min_steps"] = args.actor_freeze_min_steps
                expected_config["actor_freeze_max_steps"] = args.actor_freeze_max_steps
                expected_config["actor_freeze_critic_threshold"] = args.actor_freeze_critic_threshold
            elif isinstance(cb, CriticDivergenceAbortCallback):
                callback_refs["critic_divergence"] = cb
                expected_config["critic_abort_threshold"] = args.critic_abort_threshold
                expected_config["critic_hard_ceiling"] = args.critic_hard_ceiling
                expected_config["critic_hard_ceiling_persist"] = args.critic_hard_ceiling_persist

        callbacks.insert(0, ConfigVerificationCallback(
            config_snapshot_path=os.path.join(args.log_dir, "config_snapshot.json"),
            expected_config=expected_config,
            callback_refs=callback_refs,
        ))

    if not args.no_curriculum:
        # P1-b (FD_CURRICULUM_V3): gate-freeze snapshotter wired only under v3
        # (None on v1/v2 => CurriculumCallback behavior unchanged).
        _v3_snapshotter = None
        if config.FD_CURRICULUM_V3:
            _v3_snapshotter = AtomicSnapshotter(
                root=os.path.join(args.checkpoint_dir, "v3_gate_snapshots"),
                curriculum=curriculum)
        callbacks.append(CurriculumCallback(
            curriculum=curriculum, pool=pool,
            log_dir=args.log_dir, snapshotter=_v3_snapshotter))
    else:
        logger.info("Curriculum disabled (--no-curriculum). "
                    "Training at fixed Stage %d.", args.stage)

    # Phase 0 warmup — zero-action PID baseline transitions at P=N
    # MUST run BEFORE buffer-seed-steps (clean baseline first, then policy seed).
    # Uses vectorized vec_env (P=24) instead of single-env P=1.
    # VecNormalize is frozen during DAW to preserve W2-era obs statistics.
    if is_offpolicy and args.phase0_warmup_steps > 0:
        n_envs = vec_env.num_envs
        logger.info("[PHASE0_WARMUP] Collecting %d zero-action transitions "
                    "via P=%d vectorized env...", args.phase0_warmup_steps, n_envs)

        # VecNormalize handling during DAW depends on whether we resumed
        # from a checkpoint (protect imported stats) or started fresh
        # (must learn obs distribution from scratch).
        _vecnorm_was_training = False
        if isinstance(vec_env, VecNormalize):
            _vecnorm_was_training = vec_env.training
            if args.resume_from:
                # Freeze VecNormalize — preserve checkpoint obs stats.
                # Zero-action PID baseline may have different obs distribution
                # than the policy that generated the checkpoint.
                vec_env.training = False
                logger.info("[PHASE0_WARMUP] VecNormalize FROZEN "
                            "(preserving checkpoint obs stats).")
            else:
                # Fresh start — VecNormalize must TRAIN during DAW
                # to learn the observation distribution from scratch.
                vec_env.training = True
                logger.info("[PHASE0_WARMUP] VecNormalize in TRAINING MODE "
                            "(cold start — learning obs distribution).")

        zero_action = np.zeros((n_envs, vec_env.action_space.shape[0]),
                               dtype=np.float32)
        obs = vec_env.reset()
        collected = 0
        target = args.phase0_warmup_steps
        # Warmup-active flag: suppress watchdog stale-kill during DAW + bridge
        _WARMUP_FLAG = Path("/tmp/flightdock_warmup_active")
        _WARMUP_FLAG.touch(exist_ok=True)

        # Touch all per-env heartbeats before entering warmup loop
        for _hb_i in range(n_envs):
            Path(config.HEARTBEAT_PER_WORKER_TEMPLATE.format(
                env_id=_hb_i)).touch(exist_ok=True)

        while collected < target:
            next_obs, reward, done, info = vec_env.step(zero_action)
            model.replay_buffer.add(obs, next_obs, zero_action, reward, done, info)
            obs = next_obs
            collected += n_envs
            if collected % 5000 < n_envs:
                logger.info("[PHASE0_WARMUP] %d/%d transitions", collected, target)
            # Keep per-env heartbeats alive for watchdog (every ~1000 transitions)
            if collected % 1008 < n_envs:
                for _hb_i in range(n_envs):
                    Path(config.HEARTBEAT_PER_WORKER_TEMPLATE.format(
                        env_id=_hb_i)).touch(exist_ok=True)

        # Touch heartbeats at DAW completion
        for _hb_i in range(n_envs):
            Path(config.HEARTBEAT_PER_WORKER_TEMPLATE.format(
                env_id=_hb_i)).touch(exist_ok=True)

        # Restore VecNormalize training mode before bridge seeding / training
        if isinstance(vec_env, VecNormalize):
            if args.resume_from:
                vec_env.training = True
                logger.info("[PHASE0_WARMUP] VecNormalize restored to training mode.")
            # Assert training mode is active before proceeding
            assert vec_env.training is True, (
                "[ABORT] VecNormalize must be in training mode after DAW. "
                "A frozen normalizer with no imported stats will produce "
                "unscaled observations."
            )
            logger.info("[PHASE0_WARMUP] VecNormalize post-DAW: training=%s",
                        vec_env.training)

        # Prevent SB3 from overwriting clean baseline with random exploration.
        old_ls = model.learning_starts
        model.learning_starts = 0
        model.num_timesteps = 0
        logger.info("[PHASE0_WARMUP] Complete: %d transitions (P=%d). "
                    "Buffer size: %d. learning_starts: %d -> 0.",
                    collected, n_envs, model.replay_buffer.size(), old_ls)

    # Phase 7: Buffer seeding — collect transitions from current policy
    # Phase 7A: Skip seeding when resuming with existing (non-fresh) buffer data.
    # Do NOT skip when --fresh-buffer is used — the DAW warmup fills the fresh
    # buffer, and the bridge seeding must still run to add W2-policy transitions.
    _skip_seed_on_resume = (
        args.resume_from
        and not args.fresh_buffer
        and hasattr(model, 'replay_buffer')
        and model.replay_buffer.size() > 0
    )
    if is_offpolicy and args.buffer_seed_steps > 0 and _skip_seed_on_resume:
        print(f"[BUFFER_SEED] SKIPPED — resuming with {model.replay_buffer.size():,} existing transitions.")
    elif is_offpolicy and args.buffer_seed_steps > 0:
        # P=24 bridge seeding via vec_env.
        # VecNormalize is in training mode (restored after DAW warmup).
        # Bridge transitions update the normalizer — intentional, as they
        # represent the policy distribution the model will see during training.
        n_envs = vec_env.num_envs
        logger.info("[BUFFER_SEED] Collecting %d transitions from current policy "
                    "via P=%d...", args.buffer_seed_steps, n_envs)
        obs = vec_env.reset()
        collected = 0
        target = args.buffer_seed_steps
        # Touch heartbeats before bridge seeding
        for _hb_i in range(n_envs):
            Path(config.HEARTBEAT_PER_WORKER_TEMPLATE.format(
                env_id=_hb_i)).touch(exist_ok=True)
        while collected < target:
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, done, info = vec_env.step(action)
            model.replay_buffer.add(obs, next_obs, action, reward, done, info)
            obs = next_obs
            collected += n_envs
            if collected % 5000 < n_envs:
                logger.info("[BUFFER_SEED] %d/%d transitions", collected, target)
            # Keep per-env heartbeats alive for watchdog
            if collected % 1008 < n_envs:
                for _hb_i in range(n_envs):
                    Path(config.HEARTBEAT_PER_WORKER_TEMPLATE.format(
                        env_id=_hb_i)).touch(exist_ok=True)
        # Touch heartbeats at bridge completion + remove warmup flag
        for _hb_i in range(n_envs):
            Path(config.HEARTBEAT_PER_WORKER_TEMPLATE.format(
                env_id=_hb_i)).touch(exist_ok=True)
        _WARMUP_FLAG = Path("/tmp/flightdock_warmup_active")
        _WARMUP_FLAG.unlink(missing_ok=True)
        logger.info("[BUFFER_SEED] Complete (P=%d). Buffer size: %d",
                    n_envs, model.replay_buffer.size())

    # #305 determinism instrument (default-OFF): per-step transition-stream
    # dump for A/B replay comparison. Registered ONLY when FD_STEP_STREAM_DUMP
    # is set; unset => no import, no callback => byte-identical behavior.
    if os.environ.get("FD_STEP_STREAM_DUMP", "").strip():
        from drl.step_stream_dump import StepStreamDumpCallback
        callbacks.append(StepStreamDumpCallback(out_dir=args.log_dir))
        logger.info("[CALLBACK] StepStreamDumpCallback (FD_STEP_STREAM_DUMP set)")

    # Train
    logger.info(f"Starting {algorithm.upper()} training: {args.total_timesteps:,} timesteps...")
    # (P1 Rev 3) Initialize final_run_summary.json state BEFORE learn so
    # a signal handler firing during learn can emit the artifact.
    _flightdock_run_summary_state["start_wall_clock"] = time.monotonic()
    _flightdock_run_summary_state["args"] = args
    _flightdock_run_summary_state["model"] = model
    _flightdock_run_summary_state["noise_resume_offset"] = int(
        _noise_resume_offset)
    _flightdock_run_summary_state["noise_schedule_mode"] = (
        "resume_relative"
        if getattr(args, "reset_exploration_schedule_on_resume", False)
        else "absolute_progress")
    _flightdock_run_summary_state["summary_path"] = os.path.join(
        args.log_dir, "final_run_summary.json")
    _flightdock_run_summary_state["emitted"] = False
    try:
        if args.seed is not None:
            model.set_random_seed(args.seed)
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
        )
        # (P1 Rev 3 / E-2) Normal exit — classify the termination reason.
        _final_ts = int(getattr(model, "num_timesteps", 0) or 0)
        _target = int(getattr(args, "total_timesteps", 0) or 0)
        _emit_final_run_summary(_classify_termination(
            _final_ts, _target,
            bool(h12_callback is not None and h12_callback.fired),
            bool(early_stop_callback is not None
                 and early_stop_callback.fired)))
    except BaseException:
        logger.error("[Training] Exception during model.learn() — saving emergency checkpoint")
        # (P1 Rev 3) Exception path — emit final_run_summary BEFORE emergency
        # save so the artifact survives even if save also faults. No-op if
        # a signal handler already emitted (idempotency latch).
        try:
            if not _flightdock_run_summary_state.get("emitted"):
                _emit_final_run_summary("exception")
        except Exception:
            pass
        try:
            emergency_path = os.path.join(args.checkpoint_dir, "model_emergency.zip")
            model.save(emergency_path)
            logger.info(f"[EMERGENCY] Model saved to {emergency_path}")
            if is_offpolicy and hasattr(model, 'replay_buffer') and model.replay_buffer is not None:
                buf_path = os.path.join(args.checkpoint_dir, "replay_buffer_emergency.pkl")
                model.save_replay_buffer(buf_path)
                logger.info(f"[EMERGENCY] Replay buffer saved ({model.replay_buffer.size()} transitions)")
            if isinstance(vec_env, VecNormalize):
                norm_path = os.path.join(args.checkpoint_dir, "vecnormalize_emergency.pkl")
                vec_env.save(norm_path)
                logger.info(f"[EMERGENCY] VecNormalize saved to {norm_path}")
        except Exception as save_err:
            logger.error(f"[EMERGENCY] Checkpoint save failed: {save_err}")
        raise
    finally:
        # Clean up pool on exit (even if training crashes)
        if pool is not None:
            logger.info("[SITLPool] Stopping pool...")
            pool.stop()
        # Close vectorised environment with timeout protection.
        # This was previously OUTSIDE try/finally — the core v9 fix.
        logger.info("[Cleanup] Closing vectorised environment (30s timeout)...")
        try:
            _close_with_timeout(vec_env)
        except Exception as e:
            logger.error(f"[Cleanup] vec_env.close() failed: {e} — atexit will sweep ports")

    # Save final model — INTENTIONALLY outside try/finally.
    # On crash, we do NOT want to save a potentially corrupted model.
    final_path = os.path.join(args.checkpoint_dir, "model_final.zip")
    model.save(final_path)
    if isinstance(vec_env, VecNormalize):
        vec_env.save(os.path.join(args.checkpoint_dir, "vecnormalize_final.pkl"))
    # Save replay buffer for off-policy resume
    if is_offpolicy:
        buffer_path = os.path.join(args.checkpoint_dir, "replay_buffer_final.pkl")
        model.save_replay_buffer(buffer_path)
        logger.info(f"  Replay buffer saved ({model.replay_buffer.size()} transitions)")
    logger.info(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
