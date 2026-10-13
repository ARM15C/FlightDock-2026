"""SB3 callbacks for checkpoint saving, metrics logging, curriculum, and diagnostics.

Twenty BaseCallback subclasses, grouped by algorithm scope:

  All algorithms (TD3/SAC/PPO):
  - MetricsCallback: logs per-episode CSV with reward components
  - CurriculumCallback: drives curriculum stage transitions
  - RewardMonitorCallback: automated no-learning-signal alerting (v11)
  - HeartbeatCallback: touches heartbeat file for watchdog stall detection (v10)

  Off-policy only (TD3/SAC):
  - OffPolicyDiagnosticsCallback: logs off-policy diagnostics + CSV (v18)
  - VecNormalizeFreezeCallback: freezes VecNormalize stats after N episodes (v18)
  - BufferDilutionCheckCallback: verifies warmup data excluded from replay buffer (v18 Phase C)
  - AtomicOffPolicyCheckpointCallback: atomic write-tmp-rename checkpoints (v19 P3)
  - ZeroResidualFallbackCallback: auto-switch TD3→SAC on zero-residual trap (v19 P8)
  - VecNormalizeScheduleCallback: unified freeze/unfreeze scheduler (Campaign 2b)
  - VecNormalizeUnfreezeCallback: temporarily unfreeze stats on resume (C006 Phase 3)
  - ActorFreezeCallback: freeze actor during critic calibration (C006 Phase 3)
  - CriticDivergenceAbortCallback: abort on critic loss divergence (C006 Phase 5)
  - NoiseAnnealCallback: linear exploration noise annealing (C006 Phase 7)
  - PassthroughMonitorCallback: sustained passthrough surveillance gate (C006 Phase 7)
  - ConfigVerificationCallback: one-shot config verification with hard abort (C006 Phase 7)
  - TrainingHealthCallback: multi-trigger health monitor with emergency checkpoint (C2)
  - CheckpointGatedRollbackCallback: checkpoint-gated eval with auto rollback (C2, H3)

  On-policy only (PPO):
  - CheckpointCallback: saves model + VecNormalize every N rollouts
  - PolicyDiagnosticsCallback: logs post-update PPO policy state + CSV (v5/v11)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from . import config
from .env_formation_sitl import check_contact_m3b, check_m3b

logger = logging.getLogger(__name__)


def _check_gradient_active(model, num_timesteps: int) -> bool:
    """Return True if training has passed learning_starts (C006 B6).

    For off-policy (TD3/SAC): True when num_timesteps > model.learning_starts.
    For on-policy (PPO): no learning_starts attribute → always True.
    Defensive against mock objects returning non-int types.
    """
    try:
        ls = int(getattr(model, 'learning_starts', 0))
    except (TypeError, ValueError):
        ls = 0
    return num_timesteps > ls


def _legacy_triplet(arr: np.ndarray, reducer) -> List[float]:
    """Map per-axis action stats to [speed, heading, altitude] CSV layout.

    ACTION_DIM=3 (cross-axis restored 2026-05-06): only n_act==3 is
    valid. The earlier C3X 2D-with-NaN-pad branch has been removed
    since live policies are 3D under the canonical action contract.

    Parameters
    ----------
    arr: np.ndarray
        Shape (T, n_act) — per-timestep action values.
    reducer: Callable[[np.ndarray], float]
        Per-axis reduction function (e.g. ``np.mean``, ``np.std``,
        ``lambda a: np.mean(np.sign(a))``).

    Raises
    ------
    RuntimeError
        If ``arr.shape[1] != 3``. Explicit invariant: converts silent
        shape drift into an auditable hard stop.
    """
    n_act = arr.shape[1]
    if n_act == 3:
        return [
            float(reducer(arr[:, 0])),
            float(reducer(arr[:, 1])),
            float(reducer(arr[:, 2])),
        ]
    raise RuntimeError(
        f"OffPolicyDiagnosticsCallback: unexpected action "
        f"dimension {n_act}; expected 3 (ACTION_DIM=3)"
    )


class CheckpointCallback(BaseCallback):
    """Save model and VecNormalize stats every ``save_freq`` rollouts.

    Parameters
    ----------
    save_freq: int
        Save every this many rollouts (in ``_on_rollout_end``).
    checkpoint_dir: str
        Directory for checkpoint files.
    """

    def __init__(self, save_freq: int = 10, checkpoint_dir: str = "checkpoints") -> None:
        super().__init__()
        self._save_freq = save_freq
        self._checkpoint_dir = checkpoint_dir
        self._rollout_count = 0

    def _init_callback(self) -> None:
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self._save_freq == 0:
            model_path = os.path.join(
                self._checkpoint_dir,
                f"model_rollout_{self._rollout_count}.zip",
            )
            self.model.save(model_path)

            # Save VecNormalize stats if present
            if isinstance(self.training_env, VecNormalize):
                norm_path = os.path.join(
                    self._checkpoint_dir,
                    f"vecnormalize_rollout_{self._rollout_count}.pkl",
                )
                self.training_env.save(norm_path)

            self.logger.record("checkpoint/rollout", self._rollout_count)
            self.logger.record("checkpoint/path", model_path)


# ---------------------------------------------------------------------------
# T-467: schema-checked CSV open (defect T-466)
#
# RESPONSIBILITY: open a per-episode CSV for append/create with the on-disk
#   header VERIFIED against the current column list, so a resume can never
#   append wider rows under a stale narrower header (the T-466 defect:
#   metrics.csv grew 226 -> 231 columns mid-schema and DictReader consumers
#   silently mis-assign from the divergence index on).
# SEMANTICS:
#   fresh/empty file -> create, write header (today's behavior)
#   header == columns (list) -> plain append (byte-identical to today)
#   header a STRICT SET-SUBSET -> one-time migration: bank a backup
#     of columns AND every (<name>.pre_<N>cols.csv, refuse if it
#     data row exactly the already exists), stream-rewrite through a
#     old width sibling .tmp with "" backfill, atomic
#                                os.replace, then append
#   anything else -> RuntimeError, fail-closed at launch,
#                                naming tools/repair_csv_schema_drift.py
# SIDE-EFFECTS: may write <path>.pre_<N>cols.csv and rewrite <path>
#   (atomic, same filesystem). Never deletes; backup exists iff bytes were
#   rewritten. Crash recovery: verify the backup, remove it, relaunch.
# COVERAGE: tests/unit/drl/test_csv_schema_guard.py (executed battery).
# ---------------------------------------------------------------------------


def _read_csv_header(path: str) -> Optional[List[str]]:
    """First physical CSV record of *path* (quote-aware), BOM-stripped."""
    with open(path, "r", newline="") as fh:
        header = next(csv.reader(fh), None)
    if header and header[0].startswith("﻿"):
        header[0] = header[0].lstrip("﻿")
    return header


def _migrate_csv_additive(path: str, old_header: List[str],
                          fieldnames: List[str]) -> None:
    """Rewrite *path* from old_header to fieldnames ("" backfill), banked.

    Refuses (fail-closed) if the backup already exists (a crashed prior
    attempt must be inspected, not overwritten) or if ANY data row width
    differs from the old header (mixed/torn widths are the repair tool's
    job, never the launch path's).
    """
    n_old = len(old_header)
    backup = f"{path}.pre_{n_old}cols.csv"
    if os.path.exists(backup):
        raise RuntimeError(
            f"[T-467] {path}: backup {backup} already exists — a prior "
            f"migration attempt did not complete. Verify the backup, remove "
            f"it, and relaunch; or run tools/repair_csv_schema_drift.py.")
    # VALIDATE FIRST, write second: a refusal must leave ZERO debris (no
    # backup, no .tmp) so the next launch sees the file exactly as it was.
    with open(path, "r", newline="") as src:
        reader = csv.reader(src)
        next(reader, None)  # old header
        for lineno, row in enumerate(reader, start=2):
            if len(row) != n_old:
                raise RuntimeError(
                    f"[T-467] {path}:{lineno}: row width {len(row)} != "
                    f"header width {n_old} — mixed-width file; refusing to "
                    f"migrate. Run tools/repair_csv_schema_drift.py.")
    shutil.copy2(path, backup)
    with open(backup, "rb") as bf:
        os.fsync(bf.fileno())
    tmp = path + ".tmp"
    with open(path, "r", newline="") as src, \
            open(tmp, "w", newline="") as dst:
        reader = csv.reader(src)
        next(reader, None)  # old header
        writer = csv.DictWriter(dst, fieldnames=fieldnames, restval="",
                                extrasaction="raise")
        writer.writeheader()
        for row in reader:
            writer.writerow(dict(zip(old_header, row)))
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, path)
    logger.warning("[T-467] %s: migrated %d -> %d columns; original banked "
                   "at %s", path, n_old, len(fieldnames), backup)


def _open_schema_checked_csv(path: str, fieldnames: List[str], resume: bool,
                             allow_migrate: bool = True):
    """(file, DictWriter) for *path*, header-checked per the block above.

    allow_migrate=False demands a LIST-IDENTICAL header on append (used by
    the OPD legacy branch, which must never be silently migrated).
    """
    fieldnames = list(fieldnames)
    if len(set(fieldnames)) != len(fieldnames):
        raise RuntimeError(
            f"[T-467] {path}: duplicate names in the CURRENT column list")
    append = bool(resume) and os.path.exists(path) \
        and os.path.getsize(path) > 0
    if append:
        header = _read_csv_header(path)
        if header is None:
            append = False  # whitespace-only file: treat as fresh
        elif len(set(header)) != len(header):
            raise RuntimeError(
                f"[T-467] {path}: duplicate names in the EXISTING header — "
                f"run tools/repair_csv_schema_drift.py")
        elif header == fieldnames:
            pass  # plain append
        elif allow_migrate and set(header) < set(fieldnames):
            _migrate_csv_additive(path, header, fieldnames)
        else:
            raise RuntimeError(
                f"[T-467] {path}: existing header ({len(header)} cols) is "
                f"not the current column list ({len(fieldnames)} cols) and "
                f"is not strictly-additive growth"
                f"{'' if allow_migrate else ' (legacy branch: identical only)'}"
                f" — refusing, fail-closed. Run "
                f"tools/repair_csv_schema_drift.py.")
    fh = open(path, "a" if append else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if not append:
        writer.writeheader()
    return fh, writer


class MetricsCallback(BaseCallback):
    """Log per-episode metrics to CSV.

    Watches ``info`` dicts from vectorized envs for episode completions.
    Writes one row per completed episode with reward breakdown, milestone,
    and M3b dwell.

    Parameters
    ----------
    log_path: str
        Path to the CSV output file.
    """

    _FIELDNAMES = [
        "episode", "timestep", "reward", "length",
        "r_ref", "r_prox", "r_dwell", "r_safe", "r_smooth", "r_wake",
        "r_m3a_coarse", "r_m3a_fine",
        "r_contact_dwell",
        "r_action_rate",
        "r_exit",
        "r_terminal",
        "r_range_penalty",
        "r_approach",
        # v13 Commit 9: Per-episode mean reward components
        "r_ref_mean", "r_prox_mean", "r_dwell_mean", "r_safe_mean",
        "r_smooth_mean", "r_wake_mean", "r_m3a_coarse_mean", "r_m3a_fine_mean",
        "r_contact_dwell_mean",
        "r_action_rate_mean",
        "r_exit_mean",
        "r_range_penalty_mean",
        "r_approach_mean",
        # C006 P4: Renamed from m3b_dwell_s → precontact_tier4_dwell_s etc.
        "precontact_tier4_dwell_s", "terminal_type", "precontact_tier",
        "env_id", "pause_count", "max_pause_wall_s",
        "stopped_total_wall_s", "sitl_sim_at_last_stop", "sitl_sim_at_last_resume",
        "best_precontact_tier", "min_range_m",
        "time_to_precontact_tier1_s", "time_to_precontact_tier2_s",
        "time_to_precontact_tier3_s", "time_to_precontact_tier4_s",
        "dwell_precontact_tier1_s", "dwell_precontact_tier2_s",
        "dwell_precontact_tier3_s", "dwell_precontact_tier4_s",
        "max_behind_overshoot_m", "max_cross_excursion_m",
        # T-455: per-episode vertical extrema on BOTH channels + the guard-fire
        # DELTA (never cumulative). The failing axis was the one axis metrics.csv
        # did not record. Telemetry only -- nothing reads these back.
        # NOTE the truth channel is a 10 Hz HOLD: extrema are valid, per-tick
        # rates differenced from these columns are NOT.
        "max_above_boom_fused_m", "max_above_boom_truth_m",
        "min_boom_tip_range_truth_m", "guard_fires_episode",
        "precontact_tier4_mean_residual", "precontact_tier3_mean_residual",
        "learned_precontact_tier4_fraction", "residual_prox_correlation",
        # C006 P4: Actual milestone columns (boom-tip reference)
        "actual_M2_achieved", "actual_M2_dwell_s",
        "actual_M3a_coarse_achieved", "actual_M3a_coarse_dwell_s",
        "actual_M3a_fine_achieved", "actual_M3a_fine_dwell_s",
        "actual_M3b_achieved", "actual_M3b_dwell_s",
        "actual_min_boom_tip_range_m",
        # C006 B6: Gradient-active annotation
        "is_gradient_active",
        # Campaign 3c: dwell-failure decomposition
        "m3b_entry_count", "m3b_break_count", "m3b_mean_dwell_before_break_s",
        "m3b_last_break_axis", "m3b_last_break_margin_m",
        "m3b_last_break_velocity_mps",
        "m3b_breaks_behind", "m3b_breaks_cross", "m3b_breaks_below",
        # Campaign 3h: exit penalty diagnostics
        "penalized_exit_count", "mean_pre_exit_continuous_dwell",
        # Campaign 3i: reward pivot diagnostics
        "approach_suppression_active", "r_approach_suppressed_ticks",
        "behind_axis_exit_count", "non_behind_exit_count",
        "r_exit_behind_total",
        "mean_ttc_at_m3b_entry", "mean_closure_rate_delta_in_m3b",
        # overspeed penalty diagnostics
        "r_overspeed", "r_overspeed_mean",
        "r_overspeed_total", "overspeed_tick_count", "max_v_excess",
        # Decomposed speed + barrier diagnostics
        "policy_speed_delta_mean", "policy_speed_delta_max",
        "bias_speed_delta_mean", "bias_active_ticks",
        "bias_active_while_ahead_ticks",
        "r_barrier", "r_barrier_mean",
        "r_barrier_inside_mean", "r_barrier_outside_mean",
        "barrier_inside_ticks", "barrier_outside_ticks",
        "ticks_to_90pct_bias", "first_tier2_tick", "first_bias_active_tick",
        # Tier-4 hold-phase PSD instrumentation
        "policy_speed_delta_tier4_mean", "policy_speed_delta_tier4_p10",
        "policy_speed_delta_tier4_p50", "policy_speed_delta_tier4_p90",
        "tier4_ticks",
        # Velocity modulation + off-policy + CAPS diagnostics
        "velocity_mod_mean", "velocity_mod_ticks",
        # Braking penalty diagnostics
        "r_hold_fwd", "r_hold_fwd_mean",
        "actor_loss", "critic_loss", "caps_loss",
        "q1_mean", "q2_mean", "q1_max", "q2_max",
        # L9.2: action boundary stats
        "raw_action_max_ep",
        "boundary_hit_count", "boundary_hit_frac",
        "overflow_hit_count", "overflow_hit_frac",
        # L9.2: post-weight reward components (18 contributors)
        "r_ref_weighted", "r_ref_weighted_mean",
        "r_prox_weighted", "r_prox_weighted_mean",
        "r_dwell_weighted", "r_dwell_weighted_mean",
        "r_safe_weighted", "r_safe_weighted_mean",
        "r_smooth_weighted", "r_smooth_weighted_mean",
        "r_wake_weighted", "r_wake_weighted_mean",
        "r_m3a_coarse_weighted", "r_m3a_coarse_weighted_mean",
        "r_m3a_fine_weighted", "r_m3a_fine_weighted_mean",
        "r_contact_dwell_weighted", "r_contact_dwell_weighted_mean",
        "r_action_rate_weighted", "r_action_rate_weighted_mean",
        "r_terminal_weighted", "r_terminal_weighted_mean",
        "r_approach_weighted", "r_approach_weighted_mean",
        "r_passthrough_weighted", "r_passthrough_weighted_mean",
        "r_barrier_weighted", "r_barrier_weighted_mean",
        "r_hold_fwd_weighted", "r_hold_fwd_weighted_mean",
        "r_range_penalty_weighted", "r_range_penalty_weighted_mean",
        "r_exit_weighted", "r_exit_weighted_mean",
        "r_overspeed_weighted", "r_overspeed_weighted_mean",
        # Phase E Batch 3a: per-episode reward-component min/max telemetry
        # (16 raw components x 2 = 32 columns). See env_formation_sitl.py
        # _MRRL_RAW_COMPONENTS for the canonical list.
        "r_ref_max", "r_ref_min",
        "r_prox_max", "r_prox_min",
        "r_dwell_max", "r_dwell_min",
        "r_contact_dwell_max", "r_contact_dwell_min",
        "r_safe_max", "r_safe_min",
        "r_smooth_max", "r_smooth_min",
        "r_wake_max", "r_wake_min",
        "r_m3a_coarse_max", "r_m3a_coarse_min",
        "r_m3a_fine_max", "r_m3a_fine_min",
        "r_range_penalty_max", "r_range_penalty_min",
        "r_approach_max", "r_approach_min",
        "r_action_rate_max", "r_action_rate_min",
        "r_exit_max", "r_exit_min",
        "r_overspeed_max", "r_overspeed_min",
        "r_barrier_max", "r_barrier_min",
        "r_hold_fwd_max", "r_hold_fwd_min",
        # Phase E Batch 3a: explicit per-subtick total
        "total_mean",
        # Phase E Batch 3a: pathology diagnostic ratios
        "ratio_milestone_to_ref_mean",
        "ratio_smoothness_abs_to_total_mean",
        # Phase E Batch 3a: per-episode leader trajectory profile
        "leader_profile",
        # Patch A 2026-05-06: resolved per-episode runtime conditions
        "resolved_wind_max_ms",
        "resolved_wind_turb",
        "resolved_gps_accuracy",
        "resolved_leader_profile",
        "harness_uwb_enable",
        "reward_source_mode",
        # Patch C2 2026-05-06: paired-channel telemetry aggregates
        "condition_profile",
        "paired_csv_path",
        "hallucinated_env_m3b_count",
        "hallucinated_ctrl_m3b_count",
        "paired_env_raw_behind_samples",
        "paired_env_raw_behind_median_abs",
        "paired_env_raw_behind_p95_abs",
        "paired_env_raw_cross_median_abs",
        "paired_env_raw_cross_p95_abs",
        "paired_env_raw_below_median_abs",
        "paired_env_raw_below_p95_abs",
        "paired_ctrl_raw_behind_median_abs",
        "paired_ctrl_raw_behind_p95_abs",
        "paired_ctrl_raw_cross_median_abs",
        "paired_ctrl_raw_cross_p95_abs",
        "paired_ctrl_raw_below_median_abs",
        "paired_ctrl_raw_below_p95_abs",
        # P1-b (FD_CURRICULUM_V3): sampled per-episode cell + health probes.
        # ADDITIVE, appended at the END (binding/timebase indices unchanged);
        # blank on v1/v2 rows except the two probe columns.
        "wind_speed_ms",
        "wind_dir_deg",
        "v3_cell_label",
        "v3_rehearsal",
        "action_saturation_frac",
        "critic_q_bias_ep",
        # #379: per-episode leader-datalink delay config echo (detector c)
        "leader_tx_delay_s",
        # T-451: the FLOWN turbulence (FDM Dryden sigma triple). T-393 gave
        # both eval arms this column; the training path was never covered, so
        # metrics.csv's only turbulence figure was the INERT legacy knob
        # (resolved_wind_turb), which reads 0.0 on a cell flying sigma 0.20.
        "turbulence_sigma_ms",
        # OVSHAPE-20260816 (§5 item 2 GRANTED): endpoint no-pass shaping term
        # observability. ADDITIVE, appended at the END (T-467 migration banks
        # and backfills; a rename or reorder is a hard launch failure on
        # resume). Blank/0.0 whenever FD_NOPASS_SHAPE_ENABLE is off.
        "r_nopass_shape", "r_nopass_shape_mean",
        "r_nopass_shape_max", "r_nopass_shape_min",
    ]

    def __init__(self, log_path: str = "metrics.csv", resume: bool = False) -> None:
        super().__init__()
        self._log_path = log_path
        self._resume = resume
        self._episode_count = 0
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._action_stats_accum: Dict[int, Dict[str, float]] = {}
        # P1-b Q-bias probe: per-env episode-start (obs, action) capture +
        # discounted-return accumulator; one critic forward pass per episode.
        self._qb_state: Dict[int, Dict[str, Any]] = {}
        self._q_bias_ep: Dict[int, float] = {}

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
        # T-467: header verified against the live column list (defect T-466)
        self._file, self._writer = _open_schema_checked_csv(
            self._log_path, self._FIELDNAMES, self._resume)

    def _qbias_step(self, dones) -> None:
        """P1-b health probe: per-episode critic Q-bias vs realized return.

        Captures (s0, a0) at each episode start (model._last_obs = the obs the
        policy consumed; buffer_actions = the action the critic trains on),
        accumulates the discounted return from raw env rewards
        (VecNormalize norm_reward=False), and at episode end emits
        (min(Q1,Q2)(s0,a0) - G0) / max(|G0|, 1e-6). One critic forward per
        episode; every failure path degrades to a blank column, never raises.
        """
        try:
            import torch as th
            model = self.model
            if not hasattr(model, "critic"):
                return
            acts = self.locals.get("buffer_actions")
            if acts is None:
                acts = self.locals.get("actions")
            rewards = self.locals.get("rewards")
            last_obs = getattr(model, "_last_obs", None)
            if acts is None or rewards is None or last_obs is None:
                return
            gamma = float(getattr(model, "gamma", 0.99))
            dev = model.device
            for i in range(len(dones)):
                st = self._qb_state.get(i)
                if st is None:
                    with th.no_grad():
                        obs_t = th.as_tensor(
                            last_obs[i:i + 1], dtype=th.float32, device=dev)
                        act_t = th.as_tensor(
                            acts[i:i + 1], dtype=th.float32, device=dev)
                        qs = model.critic(obs_t, act_t)
                        q0 = float(th.min(th.cat(qs, dim=1), dim=1)[0].item())
                    st = {"q0": q0, "ret": 0.0, "gpow": 1.0}
                    self._qb_state[i] = st
                st["ret"] += st["gpow"] * float(rewards[i])
                st["gpow"] *= gamma
                if dones[i]:
                    g0 = st["ret"]
                    self._q_bias_ep[i] = (
                        (st["q0"] - g0) / max(abs(g0), 1e-6))
                    del self._qb_state[i]
        except Exception:
            self._qb_state.clear()  # fail-silent: probe must never break training

    def _on_step(self) -> bool:
        # Check for episode completions in info dicts
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        action_stats = getattr(self.model, '_last_action_stats', None)
        if action_stats is not None:
            _raw_max = action_stats.get('raw_action_max_per_env')
            _bnd_hit = action_stats.get('boundary_hit_per_env')
            _ovf_hit = action_stats.get('overflow_hit_per_env')
            if _raw_max is not None:
                for _ei in range(min(len(dones), len(_raw_max))):
                    _a = self._action_stats_accum.setdefault(
                        _ei, {'raw_max': 0.0, 'bnd': 0, 'ovf': 0, 'n': 0})
                    _a['raw_max'] = max(_a['raw_max'], float(_raw_max[_ei]))
                    if _bnd_hit is not None and bool(_bnd_hit[_ei]):
                        _a['bnd'] += 1
                    if _ovf_hit is not None and bool(_ovf_hit[_ei]):
                        _a['ovf'] += 1
                    _a['n'] += 1

        self._qbias_step(dones)

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done and "reward_components" in info:
                self._episode_count += 1
                comps = info["reward_components"]
                row = {
                    "episode": self._episode_count,
                    "timestep": self.num_timesteps,
                    "reward": comps.get("total_step", comps.get("total", 0.0)),
                    "length": info.get("episode_length", 0),
                    "r_ref": comps.get("r_ref", 0.0),
                    "r_prox": comps.get("r_prox", 0.0),
                    "r_dwell": comps.get("r_dwell", 0.0),
                    "r_safe": comps.get("r_safe", 0.0),
                    "r_smooth": comps.get("r_smooth", 0.0),
                    "r_wake": comps.get("r_wake", 0.0),
                    "r_m3a_coarse": comps.get("r_m3a_coarse", 0.0),
                    "r_m3a_fine": comps.get("r_m3a_fine", 0.0),
                    "r_contact_dwell": comps.get("r_contact_dwell", 0.0),
                    "r_action_rate": comps.get("r_action_rate", 0.0),
                    "r_exit": comps.get("r_exit", 0.0),
                    "r_terminal": comps.get("r_terminal", 0.0),
                    "r_range_penalty": comps.get("r_range_penalty", 0.0),
                    "r_approach": comps.get("r_approach", 0.0),
                    # v13 Commit 9: Per-episode mean reward components
                    "r_ref_mean": comps.get("r_ref_mean", comps.get("r_ref", 0.0)),
                    "r_prox_mean": comps.get("r_prox_mean", comps.get("r_prox", 0.0)),
                    "r_dwell_mean": comps.get("r_dwell_mean", comps.get("r_dwell", 0.0)),
                    "r_safe_mean": comps.get("r_safe_mean", comps.get("r_safe", 0.0)),
                    "r_smooth_mean": comps.get("r_smooth_mean", comps.get("r_smooth", 0.0)),
                    "r_wake_mean": comps.get("r_wake_mean", comps.get("r_wake", 0.0)),
                    "r_m3a_coarse_mean": comps.get(
                        "r_m3a_coarse_mean", comps.get("r_m3a_coarse", 0.0)),
                    "r_m3a_fine_mean": comps.get("r_m3a_fine_mean", comps.get("r_m3a_fine", 0.0)),
                    "r_contact_dwell_mean": comps.get(
                        "r_contact_dwell_mean", comps.get("r_contact_dwell", 0.0)),
                    "r_action_rate_mean": comps.get(
                        "r_action_rate_mean", comps.get("r_action_rate", 0.0)),
                    "r_exit_mean": comps.get(
                        "r_exit_mean", comps.get("r_exit", 0.0)),
                    "r_range_penalty_mean": comps.get(
                        "r_range_penalty_mean",
                        comps.get("r_range_penalty", 0.0)),
                    "r_approach_mean": comps.get(
                        "r_approach_mean",
                        comps.get("r_approach", 0.0)),
                    # C006 P4: Renamed CSV columns (info dict keys unchanged)
                    "precontact_tier4_dwell_s": info.get("m3b_dwell_s", 0.0),
                    "terminal_type": info.get("terminal_type", ""),
                    "precontact_tier": info.get("milestone_tier", 0),
                    "env_id": info.get("env_id", ""),
                    "pause_count": info.get("pause_count", 0),
                    "max_pause_wall_s": info.get("max_pause_wall_s", 0.0),
                    "stopped_total_wall_s": info.get("stopped_total_wall_s", 0.0),
                    "sitl_sim_at_last_stop": info.get("sitl_sim_at_last_stop", 0.0),
                    "sitl_sim_at_last_resume": info.get("sitl_sim_at_last_resume", 0.0),
                    "best_precontact_tier": info.get("best_milestone_tier", 0),
                    "min_range_m": info.get("min_range_m", 0.0),
                    "time_to_precontact_tier1_s": info.get("time_to_M2_s", -1.0),
                    "time_to_precontact_tier2_s": info.get("time_to_M3a_coarse_s", -1.0),
                    "time_to_precontact_tier3_s": info.get("time_to_M3a_fine_s", -1.0),
                    "time_to_precontact_tier4_s": info.get("time_to_M3b_s", -1.0),
                    "dwell_precontact_tier1_s": info.get("dwell_M2_s", 0.0),
                    "dwell_precontact_tier2_s": info.get("dwell_M3a_coarse_s", 0.0),
                    "dwell_precontact_tier3_s": info.get("dwell_M3a_fine_s", 0.0),
                    "dwell_precontact_tier4_s": info.get("dwell_M3b_s", 0.0),
                    "max_behind_overshoot_m": info.get("max_behind_overshoot_m", 0.0),
                    "max_cross_excursion_m": info.get("max_cross_excursion_m", 0.0),
                    # T-455 (see _FIELDNAMES). -1.0 default on the truth
                    # min-range keeps "no truth sample" distinct from a real
                    # small range; the other three default to 0.
                    "max_above_boom_fused_m": info.get("max_above_boom_fused_m", 0.0),
                    "max_above_boom_truth_m": info.get("max_above_boom_truth_m", 0.0),
                    "min_boom_tip_range_truth_m": info.get(
                        "min_boom_tip_range_truth_m", -1.0),
                    "guard_fires_episode": info.get("guard_fires_episode", 0),
                    "precontact_tier4_mean_residual": info.get("m3b_mean_residual", 0.0),
                    "precontact_tier3_mean_residual": info.get("m3a_fine_mean_residual", 0.0),
                    "learned_precontact_tier4_fraction": info.get("learned_m3b_fraction", 0.0),
                    "residual_prox_correlation": info.get(
                        "residual_prox_correlation", float("nan")),
                    # C006 P4: Actual milestone columns
                    "actual_M2_achieved": info.get("actual_M2_achieved", False),
                    "actual_M2_dwell_s": info.get("actual_M2_dwell_s", 0.0),
                    "actual_M3a_coarse_achieved": info.get("actual_M3a_coarse_achieved", False),
                    "actual_M3a_coarse_dwell_s": info.get("actual_M3a_coarse_dwell_s", 0.0),
                    "actual_M3a_fine_achieved": info.get("actual_M3a_fine_achieved", False),
                    "actual_M3a_fine_dwell_s": info.get("actual_M3a_fine_dwell_s", 0.0),
                    "actual_M3b_achieved": info.get("actual_M3b_achieved", False),
                    "actual_M3b_dwell_s": info.get("actual_M3b_dwell_s", 0.0),
                    "actual_min_boom_tip_range_m": info.get("actual_min_boom_tip_range_m", -1.0),
                    # C006 B6: Gradient-active annotation
                    "is_gradient_active": _check_gradient_active(self.model, self.num_timesteps),
                    # Campaign 3c: dwell-failure decomposition
                    "m3b_entry_count": info.get("m3b_entry_count", 0),
                    "m3b_break_count": info.get("m3b_break_count", 0),
                    "m3b_mean_dwell_before_break_s": info.get("m3b_mean_dwell_before_break_s", 0.0),
                    "m3b_last_break_axis": info.get("m3b_last_break_axis", "none"),
                    "m3b_last_break_margin_m": info.get("m3b_last_break_margin_m", -999.0),
                    "m3b_last_break_velocity_mps": info.get("m3b_last_break_velocity_mps", -1.0),
                    "m3b_breaks_behind": info.get("m3b_breaks_behind", 0),
                    "m3b_breaks_cross": info.get("m3b_breaks_cross", 0),
                    "m3b_breaks_below": info.get("m3b_breaks_below", 0),
                    # Campaign 3h: exit penalty diagnostics
                    "penalized_exit_count": info.get("penalized_exit_count", 0),
                    "mean_pre_exit_continuous_dwell": info.get("mean_pre_exit_continuous_dwell", 0.0),
                    # Campaign 3i: reward pivot diagnostics
                    "approach_suppression_active": info.get("approach_suppression_active", False),
                    "r_approach_suppressed_ticks": info.get("r_approach_suppressed_ticks", 0),
                    "behind_axis_exit_count": info.get("behind_axis_exit_count", 0),
                    "non_behind_exit_count": info.get("non_behind_exit_count", 0),
                    "r_exit_behind_total": info.get("r_exit_behind_total", 0.0),
                    "mean_ttc_at_m3b_entry": info.get("mean_ttc_at_m3b_entry", 0.0),
                    "mean_closure_rate_delta_in_m3b": info.get("mean_closure_rate_delta_in_m3b", 0.0),
                    # overspeed penalty diagnostics
                    "r_overspeed": comps.get("r_overspeed", 0.0),
                    "r_overspeed_mean": comps.get("r_overspeed_mean", comps.get("r_overspeed", 0.0)),
                    "r_overspeed_total": info.get("r_overspeed_total", 0.0),
                    "overspeed_tick_count": info.get("overspeed_tick_count", 0),
                    "max_v_excess": info.get("max_v_excess", 0.0),
                    # Decomposed speed + barrier diagnostics
                    "policy_speed_delta_mean": info.get("policy_speed_delta_mean", 0.0),
                    "policy_speed_delta_max": info.get("policy_speed_delta_max", 0.0),
                    "bias_speed_delta_mean": info.get("bias_speed_delta_mean", 0.0),
                    "bias_active_ticks": info.get("bias_active_ticks", 0),
                    "bias_active_while_ahead_ticks": info.get("bias_active_while_ahead_ticks", 0),
                    "r_barrier": comps.get("r_barrier", 0.0),
                    "r_barrier_mean": comps.get("r_barrier_mean", comps.get("r_barrier", 0.0)),
                    "r_barrier_inside_mean": info.get("r_barrier_inside_mean", 0.0),
                    "r_barrier_outside_mean": info.get("r_barrier_outside_mean", 0.0),
                    "barrier_inside_ticks": info.get("barrier_inside_ticks", 0),
                    "barrier_outside_ticks": info.get("barrier_outside_ticks", 0),
                    "ticks_to_90pct_bias": info.get("ticks_to_90pct_bias", -1),
                    "first_tier2_tick": info.get("first_tier2_tick", -1),
                    "first_bias_active_tick": info.get("first_bias_active_tick", -1),
                    # Tier-4 hold-phase PSD instrumentation
                    "policy_speed_delta_tier4_mean": info.get("policy_speed_delta_tier4_mean", 0.0),
                    "policy_speed_delta_tier4_p10": info.get("policy_speed_delta_tier4_p10", 0.0),
                    "policy_speed_delta_tier4_p50": info.get("policy_speed_delta_tier4_p50", 0.0),
                    "policy_speed_delta_tier4_p90": info.get("policy_speed_delta_tier4_p90", 0.0),
                    "tier4_ticks": info.get("tier4_ticks", 0),
                    # Velocity modulation + off-policy + CAPS diagnostics
                    "velocity_mod_mean": info.get("velocity_mod_mean", 0.0),
                    "velocity_mod_ticks": info.get("velocity_mod_ticks", 0),
                    # Braking penalty diagnostics
                    "r_hold_fwd": comps.get("r_hold_fwd", 0.0),
                    "r_hold_fwd_mean": info.get("r_hold_fwd_mean", 0.0),
                    # OVSHAPE §5 item 2: env emits these into
                    # reward_components only when the flag is on; 0.0 default
                    # keeps flag-off rows well-typed.
                    "r_nopass_shape": comps.get("r_nopass_shape", 0.0),
                    "r_nopass_shape_mean": comps.get("r_nopass_shape_mean", 0.0),
                    "r_nopass_shape_max": comps.get("r_nopass_shape_max", 0.0),
                    "r_nopass_shape_min": comps.get("r_nopass_shape_min", 0.0),
                    "actor_loss": self.model.logger.name_to_value.get("train/actor_loss", 0.0),
                    "critic_loss": self.model.logger.name_to_value.get("train/critic_loss", 0.0),
                    "caps_loss": self.model.logger.name_to_value.get("train/caps_loss", 0.0),
                    "q1_mean": self.model.logger.name_to_value.get("offpolicy/q1_mean", 0.0),
                    "q2_mean": self.model.logger.name_to_value.get("offpolicy/q2_mean", 0.0),
                    "q1_max": self.model.logger.name_to_value.get("offpolicy/q1_max", 0.0),
                    "q2_max": self.model.logger.name_to_value.get("offpolicy/q2_max", 0.0),
                }
                _a = self._action_stats_accum.get(
                    i, {'raw_max': 0.0, 'bnd': 0, 'ovf': 0, 'n': 0})
                row['raw_action_max_ep'] = _a['raw_max']
                row['boundary_hit_count'] = _a['bnd']
                row['boundary_hit_frac'] = _a['bnd'] / max(1, _a['n'])
                row['overflow_hit_count'] = _a['ovf']
                row['overflow_hit_frac'] = _a['ovf'] / max(1, _a['n'])
                self._action_stats_accum[i] = {
                    'raw_max': 0.0, 'bnd': 0, 'ovf': 0, 'n': 0}
                _WK = (
                    "r_ref_weighted", "r_prox_weighted",
                    "r_dwell_weighted", "r_safe_weighted",
                    "r_smooth_weighted", "r_wake_weighted",
                    "r_m3a_coarse_weighted", "r_m3a_fine_weighted",
                    "r_contact_dwell_weighted", "r_action_rate_weighted",
                    "r_terminal_weighted", "r_approach_weighted",
                    "r_passthrough_weighted", "r_barrier_weighted",
                    "r_hold_fwd_weighted", "r_range_penalty_weighted",
                    "r_exit_weighted", "r_overspeed_weighted",
                )
                for _wk in _WK:
                    row[_wk] = comps.get(_wk, 0.0)
                    row[f"{_wk}_mean"] = comps.get(f"{_wk}_mean", 0.0)

                # Phase E Batch 3a: reward-component min/max telemetry
                _TELEMETRY = (
                    "r_ref", "r_prox", "r_dwell", "r_contact_dwell",
                    "r_safe", "r_smooth", "r_wake",
                    "r_m3a_coarse", "r_m3a_fine",
                    "r_range_penalty", "r_approach", "r_action_rate",
                    "r_exit", "r_overspeed", "r_barrier", "r_hold_fwd",
                )
                for _tk in _TELEMETRY:
                    row[f"{_tk}_max"] = comps.get(f"{_tk}_max", 0.0)
                    row[f"{_tk}_min"] = comps.get(f"{_tk}_min", 0.0)
                # Operator M1: explicit per-subtick total mean
                row["total_mean"] = comps.get("total_mean", 0.0)
                # Pathology ratios (passive diagnostics only)
                row["ratio_milestone_to_ref_mean"] = comps.get(
                    "ratio_milestone_to_ref_mean", 0.0)
                row["ratio_smoothness_abs_to_total_mean"] = comps.get(
                    "ratio_smoothness_abs_to_total_mean", 0.0)
                # Per-episode leader trajectory profile
                row["leader_profile"] = info.get("leader_profile", "racetrack")
                # Patch A 2026-05-06: resolved per-episode runtime conditions
                row["resolved_wind_max_ms"] = info.get("wind_max_ms", "")
                row["resolved_wind_turb"] = info.get("wind_turb", "")
                row["resolved_gps_accuracy"] = info.get("gps_accuracy", "")
                row["resolved_leader_profile"] = info.get("leader_profile", "")
                row["harness_uwb_enable"] = info.get("harness_uwb_enable", "")
                row["reward_source_mode"] = info.get("reward_source_mode", "")
                # Patch C2 2026-05-06: paired-channel telemetry aggregates
                row["condition_profile"] = info.get("condition_profile", "")
                row["paired_csv_path"] = info.get("paired_csv_path", "")
                row["hallucinated_env_m3b_count"] = info.get(
                    "hallucinated_env_m3b_count", 0)
                row["hallucinated_ctrl_m3b_count"] = info.get(
                    "hallucinated_ctrl_m3b_count", 0)
                row["paired_env_raw_behind_samples"] = info.get(
                    "paired_env_raw_behind_samples", 0)
                row["paired_env_raw_behind_median_abs"] = info.get(
                    "paired_env_raw_behind_median_abs", 0.0)
                row["paired_env_raw_behind_p95_abs"] = info.get(
                    "paired_env_raw_behind_p95_abs", 0.0)
                row["paired_env_raw_cross_median_abs"] = info.get(
                    "paired_env_raw_cross_median_abs", 0.0)
                row["paired_env_raw_cross_p95_abs"] = info.get(
                    "paired_env_raw_cross_p95_abs", 0.0)
                row["paired_env_raw_below_median_abs"] = info.get(
                    "paired_env_raw_below_median_abs", 0.0)
                row["paired_env_raw_below_p95_abs"] = info.get(
                    "paired_env_raw_below_p95_abs", 0.0)
                row["paired_ctrl_raw_behind_median_abs"] = info.get(
                    "paired_ctrl_raw_behind_median_abs", 0.0)
                row["paired_ctrl_raw_behind_p95_abs"] = info.get(
                    "paired_ctrl_raw_behind_p95_abs", 0.0)
                row["paired_ctrl_raw_cross_median_abs"] = info.get(
                    "paired_ctrl_raw_cross_median_abs", 0.0)
                row["paired_ctrl_raw_cross_p95_abs"] = info.get(
                    "paired_ctrl_raw_cross_p95_abs", 0.0)
                row["paired_ctrl_raw_below_median_abs"] = info.get(
                    "paired_ctrl_raw_below_median_abs", 0.0)
                row["paired_ctrl_raw_below_p95_abs"] = info.get(
                    "paired_ctrl_raw_below_p95_abs", 0.0)

                # P1-b (FD_CURRICULUM_V3): sampled cell + health-probe columns.
                # wind_speed_ms/wind_dir_deg are the SAMPLED per-episode values
                # (env info :2061-2062), distinct from resolved_wind_max_ms
                # (the stage BOUND). Saturation aliases the existing
                # boundary-hit fraction under the health-probe name of record.
                row["wind_speed_ms"] = info.get("wind_speed_ms", "")
                row["wind_dir_deg"] = info.get("wind_dir_deg", "")
                row["v3_cell_label"] = info.get("v3_cell_label", "")
                row["v3_rehearsal"] = info.get("v3_rehearsal", "")
                row["action_saturation_frac"] = row["boundary_hit_frac"]
                row["critic_q_bias_ep"] = self._q_bias_ep.pop(i, "")
                row["leader_tx_delay_s"] = info.get("leader_tx_delay_s", "")
                # T-451: encoding of record, byte-for-byte the eval arms'
                # (eval_drl_checkpoint.py:123) so all three schemas join.
                # Semicolon, not comma -- this is a CSV field. Absent sigma
                # writes "", never a plausible-looking 0.0; S0 flies genuinely
                # clean air and is told apart from "unreported" by
                # v3_cell_label on the same row.
                row["turbulence_sigma_ms"] = ";".join(
                    f"{float(s):.3f}" for s in (info.get("wind_sigma_ms") or []))

                self._writer.writerow(row)
                self._file.flush()

                # Also log to tensorboard (C006 P4: renamed columns)
                self.logger.record("episode/reward", row["reward"])
                self.logger.record("episode/precontact_tier4_dwell_s",
                                   row["precontact_tier4_dwell_s"])
                self.logger.record("episode/r_ref", row["r_ref"])
                self.logger.record("episode/r_prox", row["r_prox"])
                self.logger.record("episode/r_dwell", row["r_dwell"])
                self.logger.record("episode/count", self._episode_count)
                self.logger.record("episode/best_precontact_tier", row["best_precontact_tier"])
                self.logger.record("episode/min_range_m", row["min_range_m"])
                self.logger.record("episode/dwell_precontact_tier4_s",
                                   row["dwell_precontact_tier4_s"])
                # v13 Commit 9: Mean reward variants
                self.logger.record("episode/r_ref_mean", row["r_ref_mean"])
                self.logger.record("episode/r_prox_mean", row["r_prox_mean"])
                self.logger.record("episode/r_m3a_coarse_mean", row["r_m3a_coarse_mean"])
                self.logger.record("episode/r_m3a_fine_mean", row["r_m3a_fine_mean"])
                self.logger.record("episode/r_approach_mean", row["r_approach_mean"])
                # C006 P4: Actual milestone TensorBoard logging
                self.logger.record("episode/actual_M2_dwell_s", row["actual_M2_dwell_s"])
                self.logger.record("episode/actual_M3b_dwell_s", row["actual_M3b_dwell_s"])

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()


# ---------------------------------------------------------------------------
# T-456: the persisted stage-budget ledger (class STRUCTURAL, 2026-08-12)
#
# V3GateStateMachine.decide() is the only automatic brake on how much a stage
# may train, and BOTH of its inputs used to reset on every process restart:
#   * stage_steps = num_timesteps - _v3_stage_start_steps, and
#     _v3_stage_start_steps is 0 in the constructor, updated only on promotion,
#     never persisted;
#   * model.learn() passes no reset_num_timesteps, so SB3's default True zeroes
#     num_timesteps each run.
# Consequence: every relaunch handed the current stage a fresh full budget.
# MEASURED -- stage 15 consumed ~98,062 against a 96,000 cap with no halt, and
# the global pool guard (1,796,000) could only fire inside a single ~145-hour
# process. A budget compared against an in-process counter is a comment, not a
# budget.
#
# The repair is a small JSON ledger beside the run, read once and rewritten as
# the run proceeds. decide() itself is UNCHANGED and stays stateless -- the
# caller simply hands it effective CUMULATIVE figures instead of in-process
# ones, so the decision table remains unit-testable without SB3.
# ---------------------------------------------------------------------------

STAGE_BUDGET_LEDGER_NAME = "v3_stage_budget.json"


def stage_budget_ledger_path(log_dir: str) -> str:
    return os.path.join(log_dir, STAGE_BUDGET_LEDGER_NAME)


def read_stage_budget_ledger(log_dir: str) -> Dict[str, Any]:
    """Load the ledger, or FAIL CLOSED.

    A missing ledger raises unless FD_V3_BUDGET_LEDGER_INIT=1 explicitly
    declares a fresh campaign. Absent evidence is not zero -- assuming zero is
    exactly the defect this ledger exists to fix. A corrupt or truncated ledger
    ALWAYS raises: a half-written file that read back as zero would reintroduce
    the bug at the worst possible moment.
    """
    path = stage_budget_ledger_path(log_dir)
    if not os.path.exists(path):
        if os.environ.get("FD_V3_BUDGET_LEDGER_INIT", "0").strip() == "1":
            return {"stage_consumed": {}, "pool_consumed": 0}
        raise RuntimeError(
            f"[V3-BUDGET] no stage-budget ledger at {path}. Refusing to train: "
            "without it the per-stage cap and the pool guard cannot fire "
            "across a restart (T-454/T-456). Seed it with the campaign's "
            "measured consumption, or set FD_V3_BUDGET_LEDGER_INIT=1 to "
            "declare a genuinely fresh campaign.")
    with open(path) as fh:
        data = json.load(fh)          # a corrupt file raises here, by design
    if (not isinstance(data, dict)
            or not isinstance(data.get("stage_consumed"), dict)
            or not isinstance(data.get("pool_consumed"), int)):
        raise RuntimeError(
            f"[V3-BUDGET] malformed stage-budget ledger at {path}: {data!r}")
    # T-489: the OPTIONAL campaign_prior term (spend that happened OUTSIDE this
    # log dir). Absent means 0 -- every pre-rescope ledger keeps its exact
    # behaviour. Present-but-wrong-typed RAISES rather than coercing: a string
    # or a bool silently read as 0 (or as 1) would restore precisely the
    # fail-open the pool guard exists to close. bool is excluded explicitly
    # because bool IS an int in Python and True would pass as 1.
    prior = data.get("campaign_prior")
    if prior is not None and (isinstance(prior, bool)
                              or not isinstance(prior, int) or prior < 0):
        raise RuntimeError(
            f"[V3-BUDGET] malformed campaign_prior at {path}: {prior!r} — "
            "must be a non-negative int, or absent. Refusing to train: a "
            "campaign prior read as zero puts the pool guard back to sleep.")
    return data


def write_stage_budget_ledger(log_dir: str, data: Dict[str, Any]) -> None:
    """Atomic replace -- the snapshotter's idiom. A crash mid-write must never
    leave a plausible-looking partial ledger."""
    path = stage_budget_ledger_path(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def effective_budget(ledger: Dict[str, Any], stage: int,
                     in_process_stage_steps: int,
                     num_timesteps: int) -> Tuple[int, int]:
    """(stage_steps, total_steps) as CUMULATIVE figures across restarts.

    These are what decide() must compare against the cap and the pool. The
    prior-consumption terms come from the ledger; the in-process terms come
    from this run. A stage with no ledger entry contributes 0, so a genuinely
    fresh stage still receives its full allowance -- the direction that gets
    skipped, and the one FM-2 pins.

    T-489: the pool term carries THREE parts, not two. `pool_consumed` is
    LOG-DIR scoped -- it counts only what this log dir has spent (W1418 =
    stages 14-15) -- so on its own it under-counted campaign spend by 959,618
    steps and the pool guard could not fire. `campaign_prior` supplies the
    spend that happened outside this log dir; absent it is 0, so every
    pre-rescope ledger behaves byte-identically. The two terms stay SEPARATE
    (rather than folded into one bumped number) so each remains independently
    verifiable, and so the writer can keep emitting pool_consumed as log-dir
    spend without ever double-counting the prior.
    """
    prior_stage = int(ledger.get("stage_consumed", {}).get(str(stage), 0))
    prior_pool = int(ledger.get("pool_consumed", 0))
    campaign_prior = int(ledger.get("campaign_prior", 0))
    return (prior_stage + int(in_process_stage_steps),
            campaign_prior + prior_pool + int(num_timesteps))


def stage_budget_payload(prior: Dict[str, Any], stage: int, stage_steps: int,
                         total_steps: int) -> Dict[str, Any]:
    """The dict to persist after a decision. THE seam that keeps the ledger's
    two spend terms from collapsing into each other.

    T-489: `total_steps` is CAMPAIGN scoped (it carries `campaign_prior`),
    while `pool_consumed` must stay LOG-DIR scoped -- so the prior is
    subtracted back out here and re-emitted as its own key. Persisting
    total_steps directly would silently flip the file's meaning on the first
    write and drop the prior, leaving a ledger whose scope depends on how many
    times it has been written. With no prior present this returns exactly what
    the pre-T-489 call site built, key for key.
    """
    cprior = int(prior.get("campaign_prior", 0))
    merged = dict(prior["stage_consumed"])
    merged[str(stage)] = int(stage_steps)
    payload: Dict[str, Any] = {"stage_consumed": merged,
                               "pool_consumed": int(total_steps) - cprior}
    if cprior:
        payload["campaign_prior"] = cprior
    return payload


# ---------------------------------------------------------------------------
# GATEFIRE-627 L2 (2026-08-17): the gate request must not be MISSABLE, and its
# absence must never be SILENT.
#
# MEASURED: s19 consumed its full 48,000-step budget, exited rc 0 with
# termination_reason "normal_budget_exhaustion", and NEVER FILED ITS GATE
# REQUEST. The in-loop request site is reached only at an episode end and only
# every `check_interval` episodes, so with budget == nominal its window was
# ZERO-WIDTH: the last qualifying check landed at 47,263 stage-steps, 737
# short. s18b survived the identical knife-edge at 96,000 -- fortunate, not
# designed, and the condition recurs at every remaining rung.
#
# The real defect is the FALSE GREEN -- rc 0 returned by a run that had failed
# its purpose. L1 (`_v3_end_of_run_gate_check`) stops it happening; THIS is the
# independent limb that makes it loud if it happens anyway. It deliberately
# needs NOTHING from L1: a stale deploy that lacks L1 is precisely the case it
# has to keep catching.
# ---------------------------------------------------------------------------

GATE_AUDIT_SENTINEL_NAME = "GATE_REQUEST_MISSING.json"


def gate_request_audit(log_dir: str) -> Tuple[int, Dict[str, Any]]:
    """(rc, report): did the stage this run was on earn a gate and file one?

    rc 0 == clean; rc 1 == a stage consumed its nominal with no request on
    disk. NEVER RAISES -- the caller is a post-run shell assertion, and an
    audit that crashes the wrapper teaches people to delete it.

    SCOPE IS THE HIGHEST STAGE IN THE LEDGER, and only that one. Under v3 the
    curriculum only ever advances, so every lower key is a stage already
    promoted past, whose request may legitimately live in a PREVIOUS campaign
    log dir (ledgers are carried w1418 -> w1419 -> ...). Auditing those would
    fire on history rather than on this run. They are listed in the report as
    `skipped_historical`, so the narrowing is visible rather than assumed.
    """
    report: Dict[str, Any] = {"log_dir": log_dir, "verdict": "clean",
                              "reason": None, "stage": None, "label": None,
                              "stage_steps": None, "nominal": None,
                              "skipped_historical": []}
    try:
        path = stage_budget_ledger_path(log_dir)
        if not os.path.exists(path):
            # G4: a run with no v3 ledger is not a gated block. Not a finding.
            report["reason"] = "no_v3_ledger"
            return 0, report
        with open(path) as fh:
            data = json.load(fh)
        consumed = data.get("stage_consumed")
        if not isinstance(consumed, dict) or not consumed:
            # A malformed ledger beside a just-finished run IS a finding: read
            # as empty it would silently answer "no stage earned a gate".
            report["verdict"] = "unreadable"
            report["reason"] = "ledger_malformed_or_empty"
            return 1, report
        stages = sorted(int(k) for k in consumed)
        stage = stages[-1]
        report["skipped_historical"] = stages[:-1]
        report["stage"] = stage
        label = config.STAGE_LABELS_V3.get(stage)
        gate_type = config.V3_GATE_TYPE.get(stage)
        report["label"] = label
        if label is None or not gate_type:
            report["reason"] = "stage_has_no_gate"      # G4
            return 0, report
        nominal = (config.STAGE_NOMINAL_TIMESTEPS_V3[stage]
                   + config.STAGE_EXTENSION_TIMESTEPS_V3.get(stage, 0))
        stage_steps = int(consumed[str(stage)])
        report["stage_steps"] = stage_steps
        report["nominal"] = nominal
        report["headroom"] = stage_steps - nominal
        if stage_steps < nominal:
            report["reason"] = "below_nominal"          # G4: nothing earned
            return 0, report
        req = os.path.join(log_dir, "gate_requests", f"{label}.json")
        if os.path.exists(req):
            report["reason"] = "request_present"
            return 0, report
        report["verdict"] = "MISSING"
        report["reason"] = ("stage consumed its nominal but filed no gate "
                            "request — this block earned an exam and did not "
                            "ask for one (GATEFIRE-627)")
        report["expected_request"] = req
        return 1, report
    except Exception as exc:                # noqa: BLE001 - never raise
        report["verdict"] = "unreadable"
        report["reason"] = f"audit_error: {exc}"
        return 1, report


def append_stage_promotion(log_dir: str, timestep: int, old_stage: int,
                           new_stage: int) -> None:
    """F-3 (T-198/T-199): durable per-promotion record for halt forensics.

    Appends one JSON line {timestep, old_stage, new_stage} to
    <log_dir>/stage_promotions.jsonl at every curriculum promotion, so a
    halt post-mortem can partition diagnostics rows by stage directly —
    the T-198 RCA had to reconstruct boundaries from gate_requests/acks.
    Append-only; best-effort (a promotion must never fail on forensics IO).
    """
    if not log_dir:
        return
    try:
        with open(os.path.join(log_dir, "stage_promotions.jsonl"), "a") as fh:
            fh.write(json.dumps({"timestep": int(timestep),
                                 "old_stage": int(old_stage),
                                 "new_stage": int(new_stage)}) + "\n")
    except OSError as exc:
        logger.warning("[F-3] stage_promotions.jsonl append failed: %s", exc)


def _v3_promotion_ack_ok(log_dir: str, label: str, target_stage: int,
                         verdict: dict):
    """E-α(b) (T-128): (ok, reason) for a promotion into `target_stage`.

    DEFAULT IS DENY. Promotions targeting code stage >=
    config.V3_PROMOTION_ACK_STAGE_MIN (WS2 onward) require an operator ack
    file at <log_dir>/promotion_acks/<label>_ack.json whose fields match this
    gate AND this verdict:

        {"gate_label": <label>, "target_stage": <int>,
         "verdict_k": <k>, "verdict_n": <n>}

    Scoping rationale: label+target pin the gate; k+n pin the SPECIFIC
    verdict, so an ack written for a previous attempt (different verdict)
    cannot be silently reused. Any missing, unreadable, or corrupt ack file
    DENIES — defining "corrupt => retry-able" would reopen the fail-open
    class this election exists to close. WS1 targets (< STAGE_MIN) are exempt
    per election (a): every reachable WS1 stage lies inside the band [1,2].

    Escape hatch (documented rollback): FD_V3_PROMOTION_ACK_DISABLE=1
    restores auto-promotion; read at use so a test or an operator can flip
    it without re-import.
    """
    if target_stage < config.V3_PROMOTION_ACK_STAGE_MIN:
        return True, "ws1_exempt"
    if os.environ.get("FD_V3_PROMOTION_ACK_DISABLE", "0").strip() == "1":
        return True, "disabled_by_flag"
    path = os.path.join(log_dir, config.V3_PROMOTION_ACK_DIRNAME,
                        f"{label}_ack.json")
    try:
        with open(path) as fh:
            ack = json.load(fh)
        scoped = (ack.get("gate_label") == label
                  and int(ack.get("target_stage", -1)) == int(target_stage)
                  and int(ack.get("verdict_k", -1)) == int(verdict.get("k", -2))
                  and int(ack.get("verdict_n", -1)) == int(verdict.get("n", -2)))
    except Exception:
        return False, "ack_absent_or_unreadable"
    if not scoped:
        return False, "ack_scope_mismatch"
    return _v3_h1_ack_conjunct(verdict)


def _v3_h1_ack_conjunct(verdict: dict):
    """T-237/B3: (ok, reason) for the H1 safety conjunct on AUTO-promotion.

    Reads the OFFLINE adjudication aggregate out of the published verdict
    FILE — the same class of read as the k/n scope check above; which block
    (whole-leg `h1_truth_peak_up_m` vs A-9's `h1d_truth_peak_up_m` +
    `exceed_legs_v3`) follows the verdict's own safety_criteria (T-362).
    No truth tensor touches actor/critic/reward/replay/observation/
    policy-target (G4 preserved); this runs at an episode boundary on a dict.

    Fail-closed on a missing/malformed block: curriculum_gate_runner ALWAYS
    emits h1_aggregate (B-1), even on invalid_protocol, so absence means a
    malformed verdict, never a legitimately H1-free one.
    """
    if os.environ.get("FD_V3_H1_ACK_GATE", "1").strip() == "0":
        return True, "h1_gate_disabled_by_flag"
    # T-362: criterion-aware per A-9.5 (H1ADJ-A2B2-20260809). The verdict
    # echoes its own safety_criteria (B-1); a v3_windowed panel is judged on
    # the A-9 quantities — exceed_legs_v3 and the H1-D (docking-rows) p95 —
    # under the SAME bars. Unknown criterion fails CLOSED, never defaults.
    crit = verdict.get("safety_criteria")
    if crit == "v3_windowed":
        h1 = verdict.get("h1d_truth_peak_up_m")
        if not isinstance(h1, dict):
            return False, "h1d_block_absent"
        try:
            exceed = int(verdict["exceed_legs_v3"])
            p95 = float(h1["p95"])
        except Exception:
            return False, "h1d_block_absent"
    elif crit in ("v2_real_events", "v1_guard_firings", None):
        h1 = verdict.get("h1_truth_peak_up_m")
        if not isinstance(h1, dict):
            return False, "h1_block_absent"
        try:
            exceed = int(h1["exceed_2m_legs"])
            p95 = float(h1["p95"])
        except Exception:
            return False, "h1_block_absent"
    else:
        return False, "unknown_safety_criteria"
    if exceed > config.V3_H1_ACK_MAX_EXCEED:
        return False, "h1_exceed_gt_%d" % config.V3_H1_ACK_MAX_EXCEED
    if p95 > config.V3_H1_ACK_P95_MAX_M:
        return False, "h1_p95_over_bar"
    return True, "ack_ok"


class V3GateStateMachine:
    """P1-b: pure decision core for the FD_CURRICULUM_V3 sealed-gate lifecycle.

    Per stage: TRAIN -> (stage_steps >= nominal) REQUEST_GATE (checkpoint
    frozen first; training continues while the sealed gate flies on spare
    workers) -> verdict PROMOTE => promote / verdict FAIL => continue to cap;
    stage cap without a pass => HALT; binding inclusive
    pool exhausted => HALT BUDGET_TRUNCATED (a budget outcome, not a
    competence verdict — R23 pool convention). Stateless: all inputs passed
    in, so the decision table is unit-testable without SB3.
    """

    ACTIONS = ("none", "request_gate", "promote",
               "halt_budget_exhausted", "halt_pool_truncated")

    def decide(self, stage: int, stage_steps: int, total_steps: int,
               gate_requested: bool, verdict) -> str:
        # POOLRAISE-20260813: the ceiling is base + every SIGNED raise
        # election, resolved by the ONE definition the G12 launch gate also
        # calls, so the runtime guard and the gate cannot disagree. With no
        # election in force this is byte-identical to the bare constant.
        if total_steps >= config.effective_pool_v3():
            return "halt_pool_truncated"
        if verdict is not None and bool(verdict.get("passed")):
            return "promote"
        # T-456: a one-time elected extension is added to BOTH bounds, so the
        # stage keeps its normal shape. The default is 0 for every stage, so
        # with no election in force this is byte-identical to the old code.
        ext = config.STAGE_EXTENSION_TIMESTEPS_V3.get(stage, 0)
        nominal = config.STAGE_NOMINAL_TIMESTEPS_V3[stage] + ext
        cap = config.STAGE_CAP_TIMESTEPS_V3[stage] + ext
        if not gate_requested and stage_steps >= nominal:
            return "request_gate"
        if stage_steps >= cap:
            return "halt_budget_exhausted"
        return "none"


# --------------------------------------------------------------------------
# T-168 remedy (a): gate-pause. Between filing a sealed-gate request and the
# verdict landing, the learner used to keep training — ~41k steps per stage
# spent on a policy that is NOT the one being judged, and on 2026-07-31 W4
# reached its cap and halted BEFORE its own verdict arrived. The pause makes
# that span ~0. Clock seams are module-level so the detectors can drive a fake
# clock without patching the global `time` module for the whole process.
# --------------------------------------------------------------------------
_now = time.monotonic
_sleep = time.sleep
GATE_PAUSE_POLL_S = 15.0

# --- S-1 (2026-08-06): the pause must OUTLIVE the panel it waits for ---------
# ONE FACT — "how long may a gate panel take" — used to live under two names in
# two files, derived independently and never cross-checked:
#   scripts/gate_relay_core.py PANEL_BUDGET_S mid12 18,900 / anchor30 32,400
#   here GATE_PAUSE_TIMEOUT_S = 300*60 = 18,000 (HARD)
# D1 attempt_001 took 18,505 s and landed in the 900 s gap: the relay still
# considered the panel healthy and on budget while the learner had ALREADY
# halted fail-closed, taking the window with it. The latent case was worse — an
# anchor30 panel is budgeted 32,400 s against that same 18,000 s pause, a
# 14,400 s gap, and the sole remaining ladder anchor30 IS the H2 decisive cell.
#
# The timeout is now DERIVED from the budget table, so the two cannot diverge by
# arithmetic. The table is MIRRORED here rather than imported, deliberately:
# gate_relay_core mutates sys.path at import and lives under scripts/, and this
# module is imported by every training run — a new import edge on that critical
# path buys drift-safety at the cost of availability. The mirror is instead made
# safe by (a) assert_gate_pause_outlives_panels(), run at import below, and
# (b) tests/unit/test_gate_pause_outlives_panel_budget.py, which asserts EXACT
# equality against the relay's own table. A mirror with a fail-closed equality
# test is one fact with one CHECKED name; an UNCHECKED mirror is what killed D1.
PANEL_WALL_BUDGET_S = {"mid12": int(3.5 * 3600 * 1.5),       # 18,900 s
                       "anchor30": int(6.0 * 3600 * 1.5)}    # 32,400 s
GATE_PAUSE_MARGIN_S = 3600               # one hour of slack past the worst panel


def assert_gate_pause_outlives_panels(budget_table=None) -> int:
    """Fail CLOSED unless the pause strictly outlives the worst panel budget.

    Called at import (so it precedes step 1 of any training run) and by the
    relay-equality test with the relay's own table passed in. Raises rather
    than substituting a default: a silently-shortened pause is precisely the
    failure this exists to prevent.
    """
    table = PANEL_WALL_BUDGET_S if budget_table is None else budget_table
    if not table:
        raise RuntimeError("panel budget table is empty — cannot size the "
                           "gate pause; refusing to guess")
    worst = max(table.values())
    if GATE_PAUSE_TIMEOUT_S <= worst:
        raise RuntimeError(
            "gate pause %d s does not outlive the worst panel budget %d s "
            "(%s) — a panel finishing inside its own budget would halt the "
            "learner. Raise GATE_PAUSE_MARGIN_S or lower the budget."
            % (GATE_PAUSE_TIMEOUT_S, worst, table))
    return GATE_PAUSE_TIMEOUT_S


GATE_PAUSE_TIMEOUT_S = max(PANEL_WALL_BUDGET_S.values()) + GATE_PAUSE_MARGIN_S
assert_gate_pause_outlives_panels()      # fail closed at import, before step 1


def _gate_pause_enabled() -> bool:
    """Rollback switch, read AT POINT OF USE (same pattern as the other
    FD_V3_* switches) so a live window can be recovered without a redeploy.
    FD_V3_GATE_PAUSE=0 restores pre-T-168 continue-toward-cap behavior."""
    return os.environ.get("FD_V3_GATE_PAUSE", "1").strip() != "0"


class CurriculumCallback(BaseCallback):
    """Drive curriculum stage transitions (algorithm-agnostic).

    For PPO (on-policy): checks advancement in _on_rollout_end() — fires once
    per rollout (meaningful cadence).

    For TD3/SAC (off-policy): checks advancement in _on_step() every
    CHECK_INTERVAL_EPISODES episodes. _on_rollout_end() is skipped because
    TD3's train_freq=(1,"step") fires it every step — curriculum would
    advance after 20 steps, not 20 episodes.

    Parameters
    ----------
    curriculum: CurriculumManager
        The curriculum manager instance.
    pool: SITLPool or None
        SITL pool reference for warmup target propagation.
    check_interval: int
        For off-policy: check advancement every N episodes (default 4,
        approximates ~1 PPO rollout at P=4).
    """

    # T-456: class-level default so partial-mock callbacks built without
    # __init__ (the established test idiom) still carry the attribute. Same
    # remedy as the T-455 env accumulators; immutable, so the first instance
    # assignment shadows it.
    _v3_budget_prior: Optional[Dict[str, Any]] = None
    # T-459: identity of the HOLD sentinel as it stood when the gate request
    # was filed. Two attributes, not one, because "no sentinel was present"
    # and "no latch was ever taken" are different states and must not collapse.
    _v3_hold_at_request: Optional[str] = None
    _v3_hold_latched: bool = False

    def __init__(self, curriculum, pool=None, check_interval: int = 4,
                 log_dir: str = "logs", snapshotter=None) -> None:
        super().__init__()
        self._curriculum = curriculum
        self._pool = pool
        self._rollout_count = 0
        self._total_episodes = 0
        self._check_interval = check_interval
        self._episode_rewards: list = []
        self._episode_m3b: list = []
        # D2 competence floor: per-stage rolling window of an existing online
        # competence signal (perception-side M3b dwell > 0). Cleared on advance.
        self._stage_competence_window: list = []
        # P1-b (FD_CURRICULUM_V3): sealed-gate lifecycle state — inert on v1/v2.
        self._v3 = config.FD_CURRICULUM_V3
        self._v3_sm = V3GateStateMachine()
        self._v3_log_dir = log_dir
        self._v3_snapshotter = snapshotter
        self._v3_stage_start_steps = 0
        # T-456: cumulative budget baseline, loaded once from the ledger on the
        # first gate-logic call and never mutated thereafter. None = not loaded.
        self._v3_budget_prior: Optional[Dict[str, Any]] = None
        self._v3_gate_requested = False
        self._v3_halted = False
        # T-459: see the class-level defaults. Reset per stage at promotion.
        self._v3_hold_at_request: Optional[str] = None
        self._v3_hold_latched = False
        # T-168: latched so a timed-out pause never re-engages for this stage.
        self._v3_pause_timed_out = False

    def _is_offpolicy(self) -> bool:
        """Check if model is off-policy (has replay_buffer)."""
        return hasattr(self.model, 'replay_buffer')

    def _on_step(self) -> bool:
        # Collect per-episode stats for curriculum decisions
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                self._total_episodes += 1
                comps = info.get("reward_components", {})
                self._episode_rewards.append(comps.get("total", 0.0))
                m3b = info.get("m3b_dwell_s", 0.0)
                self._episode_m3b.append(1.0 if m3b >= 3.0 else 0.0)
                # D2 competence floor signal: positive perception-side M3b dwell
                # (lenient ">0", matching the operator's f9>0 intent; online proxy
                # for the offline simultaneous-f9, no truth / no new env metric).
                self._stage_competence_window.append(1.0 if m3b > 0.0 else 0.0)

                # Off-policy: episode-based advancement check
                if (self._is_offpolicy()
                        and self._total_episodes % self._check_interval == 0):
                    # Guard: skip during learning_starts (random exploration)
                    if self.num_timesteps > getattr(self.model, 'learning_starts', 0):
                        self._check_advancement()

        # P1-b (FD_CURRICULUM_V3): returning False stops training gracefully
        # at a HALT decision (budget exhausted / pool truncated / ladder done).
        return not self._v3_halted

    def _on_rollout_end(self) -> None:
        # PPO only: rollout-based advancement (off-policy skips this)
        if self._is_offpolicy():
            return

        self._rollout_count += 1
        self._check_advancement()

    def _check_advancement(self) -> None:
        """Compute stats and check curriculum advancement."""
        # P1-b (FD_CURRICULUM_V3): the in-loop early-exit / budget-advance path
        # below is DEAD under v3 — sealed gates are the sole authority
        # (detector (d) unit proves check_advance stays False regardless).
        if self._v3:
            self._v3_gate_logic()
            return

        mean_reward = (
            sum(self._episode_rewards) / max(len(self._episode_rewards), 1)
        )
        m3b_rate = (
            sum(self._episode_m3b) / max(len(self._episode_m3b), 1)
        )

        # For off-policy, use total_episodes as rollout_count proxy
        rollout_equiv = (self._rollout_count if not self._is_offpolicy()
                         else self._total_episodes // self._check_interval)

        # D2 competence floor: advance on a stage budget
        # only if the agent achieved positive perception-side M3b dwell in >=3 of
        # the last 5 episodes at this stage (EXISTING online signal; NOT the
        # truth-dependent offline grader-f9, NOT a new env metric). Default-True
        # for non-v2 callers, so the v1 path is unaffected.
        _win = self._stage_competence_window[-5:]
        competence_ok = len(_win) >= 5 and sum(_win) >= 3
        advanced = self._curriculum.check_advance(
            rollout_count=rollout_equiv,
            mean_reward=mean_reward,
            m3b_rate=m3b_rate,
            total_timesteps=self.num_timesteps,  # D2: timestep-budget advance (v2; None=>v1)
            competence_ok=competence_ok,
        )
        self.logger.record("curriculum/held_on_competence",
                           1.0 if self._curriculum.held_on_competence else 0.0)

        if advanced:
            new_stage = self._curriculum.stage
            # F-3: advance() moves exactly one stage in the v2 path.
            append_stage_promotion(self._v3_log_dir, self.num_timesteps,
                                   new_stage - 1, new_stage)
            self.logger.record("curriculum/stage", new_stage)
            self._stage_competence_window.clear()  # D2: reset floor window for the new stage
            self.logger.record("curriculum/advanced_at_episode", self._total_episodes)
            # Propagate new stage to reward computers in all worker envs.
            # SubprocVecEnv.env_method() sends the call across process boundaries.
            # VecNormalize delegates via VecEnvWrapper parent.
            self.training_env.env_method("set_reward_stage", new_stage)
            # Also propagate action scale for the new stage
            new_action_scale = config.CURRICULUM_STAGES[new_stage].action_scale
            self.training_env.env_method("set_action_scale", new_action_scale)
            # Approach F: propagate tighter safety range for later stages
            self.training_env.env_method("set_safety_range", new_stage)
            # Approach D: update pool warmup target for new stage
            if self._pool is not None:
                new_cfg = config.CURRICULUM_STAGES[new_stage]
                self._pool.update_domain_params(
                    warmup_target=new_cfg.warmup_target,
                )

        self.logger.record("curriculum/stage", self._curriculum.stage)
        self.logger.record("curriculum/total_episodes", self._total_episodes)
        self.logger.record("curriculum/m3b_rate", m3b_rate)

        # Reset per-check stats
        self._episode_rewards.clear()
        self._episode_m3b.clear()

    # ------------------------------------------------------------------
    # P1-b (FD_CURRICULUM_V3): sealed-gate lifecycle
    # ------------------------------------------------------------------

    def _v3_read_verdict(self, label: str):
        """Gate-runner verdict of record: logs/gate_verdicts/<label>.json
        (the runner's --out payload; decisive key = "passed")."""
        path = os.path.join(self._v3_log_dir, "gate_verdicts", f"{label}.json")
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return None

    def _v3_pause_touch_heartbeats(self) -> None:
        """Keep the watchdog fed while the learner is blocked.

        The pause blocks _on_step for the WHOLE callback list, so
        HeartbeatCallback cannot run and scripts/training_watchdog.sh would
        `kill -9` on a stale heartbeat. Touches the same two paths
        HeartbeatCallback._touch_all uses. Duplicated ON PURPOSE: extracting a
        shared helper is a STRUCTURAL change and this is a BEHAVIORAL dispatch
        (debt declared in T-168, extraction registered as a follow-on).
        """
        try:
            n = int(getattr(self.training_env, "num_envs", 1) or 1)
        except Exception:
            # `training_env` is a property over self.model; resolve the worker
            # count the way the watchdog itself enumerates rather than
            # silently skipping the per-worker files it polls.
            n = int(os.environ.get("FLIGHTDOCK_N_ENVS") or 1)
        try:
            Path(config.HEARTBEAT_FILE_PATH).touch(exist_ok=True)
            for i in range(n):
                Path(config.HEARTBEAT_PER_WORKER_TEMPLATE.format(
                    env_id=i)).touch(exist_ok=True)
        except Exception as exc:                       # never kill a run here
            logger.warning("[V3] gate pause: heartbeat touch failed (%s) — "
                           "watchdog exposure until the next poll", exc)

    def _v3_pause_timeout(self, label: str) -> None:
        """LEGACY fail-open — reachable ONLY under FD_V3_PAUSE_TIMEOUT_HALT=0
        (T-267); the default timeout path is `_v3_timeout_halt`. Resumes the
        pre-T-168 behavior (train toward cap) and leaves an operator packet.
        Known-legacy defects this path retains, closed by the default:
        the `_v3_pause_timed_out` latch is one-way until promotion, and while
        latched `_v3_gate_pause_if_waiting` returns before `_v3_read_hold`,
        so a recovered panel AND a later HOLD sentinel are both invisible
        (measured: 44,420 un-gated steps at WS34R W12, 2026-08-05)."""
        self._v3_pause_timed_out = True
        payload = {"label": label, "stage": int(self._curriculum.stage),
                   "timeout_s": GATE_PAUSE_TIMEOUT_S,
                   "num_timesteps": int(self.num_timesteps),
                   "note": ("no verdict within the pause window; FAILING OPEN "
                            "to pre-T-168 behavior (training continues toward "
                            "cap). Operator packet — the panel is late or "
                            "dead; do not treat this as a competence signal.")}
        try:
            with open(os.path.join(self._v3_log_dir,
                                   "GATE_PAUSE_TIMEOUT.txt"), "w") as fh:
                json.dump(payload, fh, indent=1)
        except Exception as exc:
            logger.warning("[V3] gate pause: timeout packet unwritable (%s)",
                           exc)
        logger.warning("[V3] %s: gate pause TIMEOUT after %d min — resuming "
                       "toward cap (fail-open)", label,
                       GATE_PAUSE_TIMEOUT_S // 60)

    def _v3_timeout_halt(self, label: str) -> None:
        """T-267: pause deadline expired with NO verdict and NO HOLD — HALT.

        The legacy fail-open resumed training toward cap, a state whose
        steps B1 discards at the window boundary anyway: 45,738 un-gated
        steps at WS34 stage 12, then 44,420 more at WS34R W12 (2026-08-05)
        when the panel recovered but the one-way latch kept the pause (and
        the HOLD check) disabled. Halting banks the GATED checkpoint as the
        resume point — same doctrine as `_v3_hold_halt` (T-237/B3).
        FD_V3_PAUSE_TIMEOUT_HALT=0 restores the legacy fail-open exactly."""
        self._v3_halted = True
        payload = {"label": label, "stage": int(self._curriculum.stage),
                   "timeout_s": GATE_PAUSE_TIMEOUT_S,
                   "num_timesteps": int(self.num_timesteps),
                   "note": ("no verdict within the pause window; HALTING "
                            "(fail-closed) so no un-gated steps accumulate. "
                            "The checkpoint at this step is the resume "
                            "point. Operator packet — the panel is late or "
                            "dead; adjudicate offline, then resume or "
                            "re-dispatch.")}
        try:
            with open(os.path.join(self._v3_log_dir,
                                   f"V3_HALT_GATE_TIMEOUT_{label}.json"),
                      "w") as fh:
                json.dump(payload, fh, indent=1)
        except Exception as exc:
            logger.warning("[V3] gate timeout: halt packet unwritable (%s)",
                           exc)
        logger.warning("[V3] %s: gate pause TIMEOUT after %d min — learner "
                       "HALTED (fail-closed), no un-gated steps", label,
                       GATE_PAUSE_TIMEOUT_S // 60)

    def _v3_hold_halt(self, label: str, info: dict) -> None:
        """T-237/B2: the relay says this gate is DEAD (not late) — STOP.

        Pre-T-237 a HOLD was invisible to the learner, so it waited out the
        300-min fail-open and then trained to cap: 45,738 un-gated steps at
        WS34 stage 12. Halting banks the GATED checkpoint as the resume point
        and pages the operator instead.
        """
        self._v3_halted = True
        payload = {"label": label, "stage": int(self._curriculum.stage),
                   "hold": info, "num_timesteps": int(self.num_timesteps),
                   "note": ("relay published a HOLD sentinel: the panel is "
                            "DEAD, not late. Collection STOPPED so no "
                            "un-gated steps accumulate; the checkpoint at "
                            "this step is the resume point. Operator act "
                            "required (clear the .terminal marker and "
                            "re-dispatch, or resume under a new contract).")}
        try:
            with open(os.path.join(self._v3_log_dir,
                                   f"V3_HALT_GATE_HOLD_{label}.json"),
                      "w") as fh:
                json.dump(payload, fh, indent=1, default=str)
        except Exception as exc:
            logger.warning("[V3] gate hold: halt packet unwritable (%s)", exc)
        logger.warning("[V3] %s: relay HOLD observed (%s) — learner HALTED, "
                       "no un-gated steps", label, info.get("reason"))

    def _v3_hold_path(self, label: str) -> str:
        return os.path.join(self._v3_log_dir, f"GATE_HOLD_{label}.json")

    def _v3_hold_digest(self, label: str) -> Optional[str]:
        """sha256 of the raw sentinel bytes, or None if there is no sentinel.

        Bytes, not parsed JSON, on purpose: a malformed or half-written file
        must still have a stable identity, or the freshness comparison would
        fail exactly when the file is most likely to be mid-write.
        """
        try:
            with open(self._v3_hold_path(label), "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            return None

    def _v3_hold_is_fresh(self, label: str) -> bool:
        """True iff this HOLD can be an answer to the request now in flight.

        LATCH is the mechanism: the request branch records the sentinel's
        identity BEFORE writing the request, so anything that differs from it
        arrived afterwards and is this gate's answer. Anything identical was
        already sitting there and cannot be.

        The un-latched branch is UNREACHABLE in production today — the pause
        only engages when `_v3_gate_requested` is True, and the sole site that
        sets it True is the request branch that takes the latch. It is kept as
        a defence against a future second site (a resumed run restoring the
        flag from disk, say), and `test_v3_hold_freshness.py` pins the
        single-assignment fact so that day is noticed rather than discovered.
        There it falls back to mtime against the request the learner itself
        wrote; if even that is unreadable it reports FRESH, i.e. the exact
        pre-T-459 behaviour — degrade to the old halt, never to a fail-open.
        """
        if self._v3_hold_latched:
            fresh = self._v3_hold_digest(label) != self._v3_hold_at_request
            logger.info("[V3] %s: HOLD freshness by LATCH -> %s", label,
                        "FRESH" if fresh else "STALE (ignored)")
            return fresh
        try:
            req = os.path.join(self._v3_log_dir, "gate_requests",
                               f"{label}.json")
            fresh = os.path.getmtime(self._v3_hold_path(label)) > \
                os.path.getmtime(req)
        except Exception as exc:
            logger.warning("[V3] %s: HOLD freshness undecidable, no latch and "
                           "no readable request (%s) — treated as FRESH", label,
                           exc)
            return True
        logger.info("[V3] %s: HOLD freshness by MTIME -> %s", label,
                    "FRESH" if fresh else "STALE (ignored)")
        return fresh

    def _v3_read_hold(self, label: str):
        """The HOLD sentinel dict IF it answers the request in flight, else None.

        T-459. Read by filename alone — the pre-T-459 behaviour — a stale
        sentinel halts a block it has nothing to do with: on 2026-08-12 the
        stage-15b learner filed its request at 05:50:52 and was stopped in
        that same second by a HOLD published on Aug 8 for attempt_001, four
        days and three attempts earlier.

        Returning None does NOT resume collection. This is called from inside
        `_v3_gate_pause_if_waiting`, where the learner is already blocked, so
        None means "keep waiting" and the pause timeout still governs. That is
        why an unreadable sentinel is reported rather than halted on — no
        un-gated step can accumulate either way (contrast T-237, where the
        fail-open was in the pause itself and donated 45,738 steps).
        """
        if os.environ.get("FD_V3_HOLD_HALT", "1").strip() == "0":
            return None
        try:
            with open(self._v3_hold_path(label)) as fh:
                hold = json.load(fh)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("[V3] %s: HOLD sentinel present but unreadable "
                           "(%s) — NOT halted on; the pause continues and its "
                           "timeout still governs", label, exc)
            return None
        if not isinstance(hold, dict):
            logger.warning("[V3] %s: HOLD sentinel is %s, not an object — NOT "
                           "halted on; the pause continues", label,
                           type(hold).__name__)
            return None
        return hold if self._v3_hold_is_fresh(label) else None

    def _v3_gate_pause_if_waiting(self, label: str) -> bool:
        """Block at THIS episode boundary while a sealed gate is in flight.

        Returns True iff we actually paused AND a verdict is now readable —
        the caller then re-decides immediately, so a PASS promotes in the same
        call instead of `check_interval` episodes (~8.4k steps at W4 rates)
        later. Returns False for every non-engagement case, so the pause can
        only ever exist inside the request→verdict window.
        """
        if not (self._v3_gate_requested and _gate_pause_enabled()):
            return False
        if self._v3_pause_timed_out or self._v3_halted:
            return False
        if self._v3_read_verdict(label) is not None:
            return False
        halt_path = os.path.join(self._v3_log_dir, "HALT_REQUESTED")
        deadline = _now() + GATE_PAUSE_TIMEOUT_S
        logger.info("[V3] %s: sealed gate in flight — learner PAUSED at "
                    "episode boundary (poll %.0fs, timeout %d min)",
                    label, GATE_PAUSE_POLL_S, GATE_PAUSE_TIMEOUT_S // 60)
        while True:
            if self._v3_read_verdict(label) is not None:
                logger.info("[V3] %s: verdict observed — learner RESUMED",
                            label)
                return True
            _hold = self._v3_read_hold(label)
            if _hold is not None:          # DEAD panel — never wait it out
                self._v3_hold_halt(label, _hold)
                return False
            if os.path.exists(halt_path):
                logger.warning("[V3] %s: HALT_REQUESTED during gate pause — "
                               "releasing to the halt path", label)
                return False
            if _now() >= deadline:
                if os.environ.get("FD_V3_PAUSE_TIMEOUT_HALT",
                                  "1").strip() != "0":
                    self._v3_timeout_halt(label)      # T-267 default
                else:
                    self._v3_pause_timeout(label)     # legacy fail-open
                return False
            self._v3_pause_touch_heartbeats()
            _sleep(GATE_PAUSE_POLL_S)

    def _on_training_end(self) -> None:
        """T-456 owed item (closed 2026-08-16, LEDGERFIX-594's OWED clause):
        flush the stage-budget ledger at run end.

        The in-run write fires ONLY inside _v3_gate_logic, i.e. once every
        `check_interval` COMPLETED episodes. A run whose final episodes fall
        off that quantum therefore under-reports its own spend, and the NEXT
        process starts from a stale prior. Measured on stage 18: the ledger
        read 87,747 against 96,000 actually consumed -- an 8,253-step hole
        repaired by hand afterwards (logs_w1418/BUDGET_CORRECTION.json). With
        a cap of 96,000 that hole is large enough to buy an unauthorised
        extra block, which is exactly what the ledger exists to prevent.

        IDEMPOTENT BY CONSTRUCTION: `_v3_budget_prior` is the immutable
        baseline as of process start, so recomputing the payload from it
        yields the same numbers however many times this runs -- flushing after
        an on-quantum write is a no-op, never a double count.

        NEVER RAISES: a bookkeeping flush must not fail a run that otherwise
        completed. A failure here is logged loudly and the run still exits
        normally, leaving exactly the pre-fix behaviour.
        """
        if not self._v3 or not self._v3_log_dir:
            return
        try:
            if self._v3_budget_prior is None:
                self._v3_budget_prior = read_stage_budget_ledger(
                    self._v3_log_dir)
            stage = self._curriculum.stage
            in_process_stage_steps = (self.num_timesteps
                                      - self._v3_stage_start_steps)
            stage_steps, total_steps = effective_budget(
                self._v3_budget_prior, stage, in_process_stage_steps,
                self.num_timesteps)
            write_stage_budget_ledger(
                self._v3_log_dir,
                stage_budget_payload(self._v3_budget_prior, stage,
                                     stage_steps, total_steps))
            logger.info("[V3-BUDGET] end-of-run flush: stage %d = %d steps, "
                        "pool %d (num_timesteps %d)",
                        stage, stage_steps, total_steps, self.num_timesteps)
        except Exception as exc:            # noqa: BLE001 - never fail the run
            logger.error("[V3-BUDGET] end-of-run flush FAILED (ledger may "
                         "under-report this run's final steps): %s", exc)
        # GATEFIRE-627 L1. Deliberately OUTSIDE the flush's try: a bookkeeping
        # failure must not also swallow the gate request. The check recomputes
        # from the in-memory prior, so it stands even if the flush write failed.
        self._v3_end_of_run_gate_check()

    def _v3_end_of_run_gate_check(self) -> None:
        """GATEFIRE-627 L1: file the gate request the episode quantum missed.

        The in-loop request site (`_v3_gate_logic`, via `_on_step`) is reached
        only at an EPISODE END and only every `check_interval` episodes. When a
        block's budget EQUALS its stage nominal, that window has ZERO WIDTH:
        s19's last qualifying check landed at 47,263 stage-steps against a
        48,000 nominal -- 737 short -- and the next would have been ~4 episodes
        (~10,400 steps) PAST the budget. The request was structurally unable to
        fire, and the run still exited 0. This runs once at training end, after
        the budget flush has banked the true cumulative figures, so a block
        that consumes its nominal always files before exit whatever the
        episode phase.

        DELIBERATELY NOT a call to `_v3_gate_logic()`. That path also promotes,
        halts, and enters the gate-pause -- none of which may happen in a
        process that is exiting. This files the request and does nothing else.

        INERT UNLESS NEEDED, so a run that already filed is byte-identically
        unaffected: it is a no-op when a request exists in memory or on disk,
        when the run HALTED, when a verdict already exists, when the stage has
        no gate, or when the stage is short of nominal.

        NEVER RAISES: like the flush it follows, this must not fail a run that
        otherwise completed.
        """
        # EVERY guard below sits INSIDE the try. Read outside it, a callback
        # whose v3 state was never initialised raises AttributeError straight
        # out of _on_training_end and fails a run that had already finished --
        # which is precisely the "never raises" contract this limb inherits
        # from the budget flush it follows. Caught by the existing flush tests.
        try:
            if not self._v3 or not self._v3_log_dir:
                return
            # G2: a HALT is not a completed block. A run that stopped on a
            # truncated pool, an exhausted cap or a held promotion has NOT
            # earned an exam, and manufacturing one here would gate a policy
            # the ladder has already refused. (A crashed run never reaches
            # _on_training_end at all -- SB3 propagates the exception out of
            # learn() first -- so the exception path is closed structurally
            # rather than by a flag.)
            if self._v3_halted:
                return
            if self._v3_gate_requested:      # G1: never a second request
                return
            stage = self._curriculum.stage
            label = config.STAGE_LABELS_V3[stage]
            gate_type = config.V3_GATE_TYPE.get(stage)
            if not gate_type:                # G4: a stage with no gate
                return
            req_path = os.path.join(self._v3_log_dir, "gate_requests",
                                    f"{label}.json")
            if os.path.exists(req_path):     # G1, across a restart
                # Deliberately does NOT set _v3_gate_requested. The existence
                # check above is already idempotent, and setting the flag here
                # would be a second, UN-LATCHED assignment -- exactly the
                # "resumed run restoring the flag from disk" that
                # _v3_hold_is_fresh names as the thing that would make its
                # un-latched branch reachable.
                return
            if self._v3_read_verdict(label) is not None:
                return                       # already judged; nothing owed
            if self._v3_budget_prior is None:
                self._v3_budget_prior = read_stage_budget_ledger(
                    self._v3_log_dir)
            stage_steps, _total = effective_budget(
                self._v3_budget_prior, stage,
                self.num_timesteps - self._v3_stage_start_steps,
                self.num_timesteps)
            nominal = (config.STAGE_NOMINAL_TIMESTEPS_V3[stage]
                       + config.STAGE_EXTENSION_TIMESTEPS_V3.get(stage, 0))
            if stage_steps < nominal:        # G3: nothing earned yet
                return
            ckpt = None
            if self._v3_snapshotter is not None:
                ckpt = self._v3_snapshotter.save(
                    tag=f"gate_{label}", model=self.model,
                    curriculum=self._curriculum)
            # Through the SINGLE filing site, so the T-459 latch travels with
            # the flag rather than being re-implemented here.
            self._v3_file_gate_request(stage, label, ckpt, stage_steps,
                                       filed_by="end_of_run")
            logger.warning(
                "[V3][GATEFIRE-627] %s: nominal %d reached at %d stage-steps, "
                "but the episode quantum never reached the in-loop request "
                "site — gate request FILED AT END OF RUN (checkpoint %s).",
                label, nominal, stage_steps, ckpt)
        except Exception as exc:            # noqa: BLE001 - never fail the run
            logger.error("[V3][GATEFIRE-627] end-of-run gate check FAILED "
                         "(a gate request may be OWED for this block): %s",
                         exc)

    def _v3_file_gate_request(self, stage: int, label: str, ckpt,
                              stage_steps, filed_by: str = "in_loop") -> None:
        """THE single site that files a gate request and latches the flag.

        T-459 ordering lives INSIDE this function on purpose: the HOLD
        sentinel's identity is taken BEFORE the request lands, so a HOLD
        published in the gap reads as FRESH and halts. Taken after, that same
        HOLD would be latched as the baseline and then ignored forever. The
        window is two statements against a 60 s relay poll either way, but
        only one order errs toward stopping.

        GATEFIRE-627 added a SECOND caller (the end-of-run hook). Both callers
        are funnelled through here so the request flag keeps exactly ONE
        assignment site in this module -- the fact
        `test_v3_hold_freshness.py::test_fm3_the_unlatched_branch_is_
        unreachable_today` pins by counting that assignment in the source, and
        the reason the un-latched branch of `_v3_hold_is_fresh` stays
        unreachable: the latch and the flag can no longer be separated by
        anyone adding a third caller. (That test counts a literal string, so
        this docstring must not spell the assignment out.)
        """
        self._v3_hold_at_request = self._v3_hold_digest(label)
        self._v3_hold_latched = True
        req_dir = os.path.join(self._v3_log_dir, "gate_requests")
        os.makedirs(req_dir, exist_ok=True)
        with open(os.path.join(req_dir, f"{label}.json"), "w") as fh:
            json.dump({"stage": stage, "label": label,
                       "gate_type": config.V3_GATE_TYPE[stage],
                       "checkpoint": ckpt,
                       "stage_steps": int(stage_steps),
                       "num_timesteps": int(self.num_timesteps),
                       "filed_by": filed_by}, fh, indent=1)
        self._v3_gate_requested = True

    def _v3_gate_logic(self, _second_pass: bool = False) -> None:
        stage = self._curriculum.stage
        label = config.STAGE_LABELS_V3[stage]
        in_process_stage_steps = self.num_timesteps - self._v3_stage_start_steps
        # T-456: decide() must see CUMULATIVE consumption, not this process's
        # counters -- both of those reset on every relaunch, which is why the
        # stage-15 cap never fired at ~98,062 against 96,000. The ledger is
        # loaded once and cached; a missing or corrupt one fails closed inside
        # read_stage_budget_ledger().
        if self._v3_budget_prior is None:
            self._v3_budget_prior = read_stage_budget_ledger(self._v3_log_dir)
        # _v3_budget_prior is the baseline as it stood when THIS process
        # started and is never mutated. Mutating it would fold this run's
        # in-process steps back into the prior term, so the next call in the
        # same process would count them twice.
        stage_steps, total_steps = effective_budget(
            self._v3_budget_prior, stage, in_process_stage_steps,
            self.num_timesteps)
        # Persist the cumulative figures so the NEXT process starts from truth.
        # Written before the decision, so a halt banks its own final number.
        write_stage_budget_ledger(
            self._v3_log_dir,
            stage_budget_payload(self._v3_budget_prior, stage, stage_steps,
                                 total_steps))
        verdict = self._v3_read_verdict(label)
        action = self._v3_sm.decide(stage, stage_steps, total_steps,
                                    self._v3_gate_requested, verdict)
        if action == "request_gate":
            ckpt = None
            if self._v3_snapshotter is not None:
                ckpt = self._v3_snapshotter.save(
                    tag=f"gate_{label}", model=self.model,
                    curriculum=self._curriculum)
            self._v3_file_gate_request(stage, label, ckpt, stage_steps)
            logger.info("[V3] %s: nominal budget reached at %d stage-steps — "
                        "checkpoint frozen (%s), gate requested; training "
                        "continues toward cap.", label, stage_steps, ckpt)
        elif action == "promote":
            self._v3_promote(label, verdict)
        elif action != "none":
            halt = {"reason": action, "stage": stage, "label": label,
                    "stage_steps": int(stage_steps),
                    "num_timesteps": int(self.num_timesteps),
                    "note": ("BUDGET_TRUNCATED is a budget outcome, not a "
                             "competence verdict (R23 pool convention)"
                             if action == "halt_pool_truncated" else
                             "Q4: cap without gate pass — operator packet")}
            with open(os.path.join(self._v3_log_dir, "V3_HALT.json"), "w") as fh:
                json.dump(halt, fh, indent=1)
            self._v3_halted = True
            logger.warning("[V3] HALT %s at %s (stage_steps=%d)",
                           action, label, stage_steps)
        self.logger.record("curriculum/v3_stage", stage)
        self.logger.record("curriculum/v3_stage_steps", float(stage_steps))
        self.logger.record("curriculum/v3_gate_requested",
                           1.0 if self._v3_gate_requested else 0.0)
        # T-168 remedy (a). Placed AFTER the branches so a freshly filed
        # request pauses with ~0 steps burned, and bounded to ONE re-entry so
        # a verdict that lands while we wait is acted on in THIS call rather
        # than `check_interval` episodes later.
        if (not _second_pass and action in ("none", "request_gate")
                and self._v3_gate_pause_if_waiting(label)):
            self._v3_gate_logic(_second_pass=True)

    def _v3_promote(self, label: str, verdict) -> None:
        import copy
        old_stage = self._curriculum.stage
        if old_stage == config.C1_STAGE_INDEX_V3:
            # C1 gate passed => ladder complete; the final N=50 certification
            # is an OFFLINE cert50 adjudication, never in-loop.
            with open(os.path.join(self._v3_log_dir,
                                   "V3_LADDER_COMPLETE.json"), "w") as fh:
                json.dump({"final_stage": label,
                           "num_timesteps": int(self.num_timesteps)}, fh)
            self._v3_halted = True
            logger.info("[V3] C1 gate PASSED — ladder complete, stopping "
                        "collection (cert50 runs offline).")
            return
        # E-α(b) (T-128): promotions whose TARGET is code stage >=
        # V3_PROMOTION_ACK_STAGE_MIN (WS2 onward) additionally require an
        # operator ack sentinel scoped to this gate AND this verdict. Default
        # DENY, fail-closed. A denied promotion is a WAIT, not a halt:
        # training continues at the current stage toward its cap (the same
        # safe state as awaiting a verdict), and this branch re-runs on every
        # subsequent decide()=="promote" until the ack appears — the existing
        # cap/halt logic already covers "cap reached without promotion".
        target = old_stage + 1
        ok, why = _v3_promotion_ack_ok(self._v3_log_dir, label, target,
                                       verdict if isinstance(verdict, dict)
                                       else {})
        if not ok:
            pend = os.path.join(self._v3_log_dir,
                                f"PROMOTION_ACK_PENDING_{label}.json")
            if not os.path.exists(pend):        # write + warn once per gate
                with open(pend, "w") as fh:
                    json.dump({
                        "gate_label": label, "target_stage": target,
                        "deny_reason": why,
                        "ack_path": os.path.join(
                            self._v3_log_dir,
                            config.V3_PROMOTION_ACK_DIRNAME,
                            f"{label}_ack.json"),
                        "required_fields": {
                            "gate_label": label, "target_stage": target,
                            "verdict_k": (verdict or {}).get("k"),
                            "verdict_n": (verdict or {}).get("n")},
                        "num_timesteps": int(self.num_timesteps)}, fh,
                        indent=1)
                logger.warning(
                    "[V3] PROMOTION HELD %s -> stage %d: %s — operator ack "
                    "required (E-α(b)).", label, target, why)
            # T-237/B3: a held promotion no
            # longer trains on toward cap. Every step past a held gate is
            # un-gated, and B1 discards them at the next window boundary
            # anyway — so collecting them only burns pool. Halting banks the
            # gated checkpoint. FD_V3_HOLD_HALT=0 restores the E-α(b) WAIT.
            if os.environ.get("FD_V3_HOLD_HALT", "1").strip() != "0":
                self._v3_halted = True
                logger.warning(
                    "[V3] %s: collection HALTED on a held promotion (%s) — "
                    "no un-gated steps; operator ack required to resume.",
                    label, why)
            return
        new_stage = self._curriculum.promote()
        append_stage_promotion(self._v3_log_dir, self.num_timesteps,
                               old_stage, new_stage)
        # T-456: fold the finished stage's FINAL cumulative into the baseline
        # before the counters re-zero. Without this the next stage's ledger
        # write would rebuild stage_consumed from a prior that never learned
        # what this stage actually spent, silently discarding it. Appears
        # exactly once (FM-4 pins the no-double-count direction too).
        _prior = getattr(self, "_v3_budget_prior", None)
        if _prior is not None:
            final_old, _ = effective_budget(
                _prior, old_stage,
                self.num_timesteps - self._v3_stage_start_steps,
                self.num_timesteps)
            _prior["stage_consumed"][str(old_stage)] = final_old
        self._v3_gate_requested = False
        # T-459: the finished stage's latch says nothing about the next gate,
        # and a stale latch would make the next stage's genuine HOLD read as
        # already-present. Cleared with the request flag it belongs to.
        self._v3_hold_at_request = None
        self._v3_hold_latched = False
        self._v3_stage_start_steps = self.num_timesteps
        # T-168: a timeout latched on the PREVIOUS stage must not disable the
        # pause for the next one — the panel being late once says nothing
        # about the next gate.
        self._v3_pause_timed_out = False
        self._stage_competence_window.clear()
        # Propagate via the existing channels...
        self.training_env.env_method("set_reward_stage", new_stage)
        self.training_env.env_method(
            "set_action_scale", config.CURRICULUM_STAGES[new_stage].action_scale)
        self.training_env.env_method("set_safety_range", new_stage)
        # ...and sync each worker's OWN manager copy (the sampler runs
        # worker-side; the legacy channels never move the manager's stage).
        try:
            self.training_env.set_attr(
                "curriculum_manager", copy.deepcopy(self._curriculum))
        except Exception:
            logger.exception("[V3] worker curriculum_manager sync failed")
        if self._pool is not None:
            self._pool.update_domain_params(
                warmup_target=config.CURRICULUM_STAGES[new_stage].warmup_target)
        # VecNormalize statistics FROZEN at S0 gate-pass (one-time, v3 Q6).
        if old_stage == 0:
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                vn.training = False
                logger.info("[V3] VecNormalize statistics FROZEN at S0 "
                            "gate-pass (one-time, pre-registered).")
        self.logger.record("curriculum/v3_promoted_to", new_stage)
        logger.info("[V3] PROMOTE %s -> %s (gate verdict k=%s/n=%s)",
                    label, config.STAGE_LABELS_V3[new_stage],
                    (verdict or {}).get("k"), (verdict or {}).get("n"))


class PolicyDiagnosticsCallback(BaseCallback):
    """Log post-update policy state for divergence detection (v5, CSV in v11).

    PPO on-policy only — not used for TD3/SAC (see OffPolicyDiagnosticsCallback).

    After each PPO rollout+update, logs:
      - log_std values (exploration noise per action dim)
      - action_net weight norm (residual output magnitude)
      - gradient_norm (global L2 norm of policy gradients)
      - mean absolute action per axis (from rollout buffer)
      - approx_kl and clip_fraction from SB3 logger
      - explained_variance (critic quality)
      - entropy_loss, value_loss from SB3 logger

    All metrics are written to ``policy_diagnostics.csv`` for durable post-hoc
    analysis, in addition to stdout and tensorboard.

    Note: train/* metrics are from the PREVIOUS PPO update (SB3 calls
    _on_rollout_end BEFORE train()), so they lag by one rollout. Rollout 1
    will have NaN for these fields.
    """

    _CSV_COLUMNS = [
        "rollout", "timestep",
        "approx_kl", "log_std_mean", "action_net_norm", "gradient_norm",
        # v13 Commit 15: Decomposed gradient norms
        "policy_grad_norm", "value_grad_norm", "pv_ratio",
        # C005b Commit 7: SeparateClipPPO per-group clip norms
        "actor_grad_norm", "critic_grad_norm",
        "mean_abs_action_speed", "mean_abs_action_heading", "mean_abs_action_altitude",
        "explained_variance", "clip_fraction", "entropy_loss", "value_loss",
    ]

    def __init__(self, log_dir: str = ".", verbose: int = 0,
                 resume: bool = False) -> None:
        super().__init__(verbose)
        self._log_dir = log_dir
        self._resume = resume
        self._rollout_count = 0
        self._csv_path: Optional[str] = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None

    def _init_callback(self) -> None:
        os.makedirs(self._log_dir, exist_ok=True)
        self._csv_path = os.path.join(self._log_dir, "policy_diagnostics.csv")
        # T-467: header verified against the live column list (defect T-466)
        self._csv_file, self._csv_writer = _open_schema_checked_csv(
            self._csv_path, self._CSV_COLUMNS, self._resume)
        self._csv_file.flush()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1

        policy = self.model.policy

        # --- log_std ---
        log_std_vals = None
        log_std_mean = float("nan")
        if hasattr(policy, 'log_std'):
            log_std_vals = policy.log_std.data.cpu().numpy().tolist()
            log_std_mean = float(policy.log_std.data.mean().item())

        # --- action_net weight norm ---
        action_net_norm = float("nan")
        if hasattr(policy, 'action_net') and hasattr(policy.action_net, 'weight'):
            action_net_norm = float(torch.norm(policy.action_net.weight).item())

        # --- gradient norm (global L2 across all policy parameters) ---
        gradient_norm = float("nan")
        try:
            grad_sq_sum = sum(
                p.grad.norm().item() ** 2
                for p in policy.parameters()
                if p.grad is not None
            )
            if grad_sq_sum > 0:
                gradient_norm = grad_sq_sum ** 0.5
        except Exception:
            pass

        # --- v13 Commit 15: Decomposed policy/value gradient norms ---
        policy_grad_norm = float("nan")
        value_grad_norm = float("nan")
        pv_ratio = float("nan")
        try:
            policy_sq = 0.0
            value_sq = 0.0
            for name, p in policy.named_parameters():
                if p.grad is None:
                    continue
                g2 = p.grad.norm().item() ** 2
                if (name.startswith("mlp_extractor.policy_net")
                        or name.startswith("action_net")
                        or name == "log_std"):
                    policy_sq += g2
                elif (name.startswith("mlp_extractor.value_net")
                      or name.startswith("value_net")):
                    value_sq += g2
            if policy_sq > 0 or value_sq > 0:
                policy_grad_norm = policy_sq ** 0.5
                value_grad_norm = value_sq ** 0.5
                total_norm = policy_grad_norm + value_grad_norm
                pv_ratio = policy_grad_norm / total_norm if total_norm > 1e-8 else float("nan")
        except Exception:
            pass

        # --- mean absolute action per axis (from rollout buffer) ---
        mean_abs_speed = float("nan")
        mean_abs_heading = float("nan")
        mean_abs_altitude = float("nan")
        try:
            # rollout_buffer.actions shape: (n_steps, n_envs, action_dim)
            # Mean over timesteps AND envs to get per-axis absolute mean.
            actions = self.model.rollout_buffer.actions
            if actions is not None and len(actions) > 0:
                import numpy as np
                abs_mean = np.abs(actions).mean(axis=(0, 1))  # → (action_dim,)
                if len(abs_mean) >= 3:
                    mean_abs_speed = float(abs_mean[0])
                    mean_abs_heading = float(abs_mean[1])
                    mean_abs_altitude = float(abs_mean[2])
        except Exception:
            pass

        # --- C005b: SeparateClipPPO per-group gradient norms ---
        actor_grad_norm_val = float("nan")
        critic_grad_norm_val = float("nan")
        if hasattr(self.model, 'last_actor_grad_norm'):
            actor_grad_norm_val = float(self.model.last_actor_grad_norm)
        if hasattr(self.model, 'last_critic_grad_norm'):
            critic_grad_norm_val = float(self.model.last_critic_grad_norm)

        # --- SB3 logged metrics (available after update) ---
        name_to_value = self.model.logger.name_to_value
        approx_kl = name_to_value.get("train/approx_kl")
        clip_fraction = name_to_value.get("train/clip_fraction")
        explained_var = name_to_value.get("train/explained_variance")
        entropy_loss = name_to_value.get("train/entropy_loss")
        value_loss = name_to_value.get("train/value_loss")

        def _fmt(v):
            return f"{v:.4f}" if v == v else "N/A"

        logger.info(
            f"[PolicyDiag] rollout={self._rollout_count} "
            f"log_std={log_std_vals} "
            f"action_net_norm={_fmt(action_net_norm)} "
            f"grad_norm={_fmt(gradient_norm)} "
            f"pv_ratio={_fmt(pv_ratio)} "
            f"actor_gn={_fmt(actor_grad_norm_val)} "
            f"critic_gn={_fmt(critic_grad_norm_val)} "
            f"abs_act=[{mean_abs_speed:.4f},{mean_abs_heading:.4f},{mean_abs_altitude:.4f}] "
            f"[prev_update] approx_kl={approx_kl} clip_frac={clip_fraction} "
            f"expl_var={explained_var} ent_loss={entropy_loss} val_loss={value_loss}"
        )

        # Write CSV row
        if self._csv_writer is not None:
            row = {
                "rollout": self._rollout_count,
                "timestep": self.num_timesteps,
                "approx_kl": approx_kl if approx_kl is not None else float("nan"),
                "log_std_mean": log_std_mean,
                "action_net_norm": action_net_norm,
                "gradient_norm": gradient_norm,
                "policy_grad_norm": policy_grad_norm,
                "value_grad_norm": value_grad_norm,
                "pv_ratio": pv_ratio,
                "actor_grad_norm": actor_grad_norm_val,
                "critic_grad_norm": critic_grad_norm_val,
                "mean_abs_action_speed": mean_abs_speed,
                "mean_abs_action_heading": mean_abs_heading,
                "mean_abs_action_altitude": mean_abs_altitude,
                "explained_variance": explained_var if explained_var is not None else float("nan"),
                "clip_fraction": clip_fraction if clip_fraction is not None else float("nan"),
                "entropy_loss": entropy_loss if entropy_loss is not None else float("nan"),
                "value_loss": value_loss if value_loss is not None else float("nan"),
            }
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        # Also record to tensorboard
        if log_std_vals is not None:
            for i, v in enumerate(log_std_vals):
                self.logger.record(f"policy_diag/log_std_{i}", v)
        if action_net_norm == action_net_norm:  # not NaN
            self.logger.record("policy_diag/action_net_norm", action_net_norm)
        if gradient_norm == gradient_norm:
            self.logger.record("policy_diag/gradient_norm", gradient_norm)
        if policy_grad_norm == policy_grad_norm:
            self.logger.record("policy_diag/policy_grad_norm", policy_grad_norm)
        if value_grad_norm == value_grad_norm:
            self.logger.record("policy_diag/value_grad_norm", value_grad_norm)
        if pv_ratio == pv_ratio:
            self.logger.record("policy_diag/pv_ratio", pv_ratio)
        if actor_grad_norm_val == actor_grad_norm_val:
            self.logger.record("policy_diag/actor_grad_norm", actor_grad_norm_val)
        if critic_grad_norm_val == critic_grad_norm_val:
            self.logger.record("policy_diag/critic_grad_norm", critic_grad_norm_val)
        if approx_kl is not None:
            self.logger.record("policy_diag/approx_kl", approx_kl)
        if clip_fraction is not None:
            self.logger.record("policy_diag/clip_fraction", clip_fraction)
        if explained_var is not None:
            self.logger.record("policy_diag/explained_variance", explained_var)

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()


class OffPolicyDiagnosticsCallback(BaseCallback):
    """Log off-policy (TD3/SAC) diagnostics per episode to CSV (v18).

    Separate from PolicyDiagnosticsCallback because on-policy and off-policy
    metrics diverge significantly:
    - No rollout buffer, no approx_kl, no clip_fraction, no log_std
    - Instead: actor_loss, critic_loss, buffer utilisation, action norms
    - SAC-specific: ent_coef_alpha (learned entropy coefficient)

    Alerts written to files for watchdog integration:
    - ALERT_ZERO_RESIDUAL.txt: action L2 norm < 0.05 for 5 consecutive episodes
    - ALERT_Q_DIVERGENCE.txt: critic_loss > 100 * mean_episode_reward
    - ALERT_DIMENSION_COLLAPSE.txt: one action dimension std < 0.01

    CSV: offpolicy_diagnostics.csv
    """

    _CSV_COLUMNS = [
        "episode", "timestep", "env_id", "algorithm",
        "actor_loss", "critic_loss",
        "action_l2_norm",
        "action_mean_speed", "action_mean_heading", "action_mean_altitude",
        "action_std_speed", "action_std_heading", "action_std_altitude",
        "buffer_utilisation",
        "ent_coef_alpha",
        # C006 B5: Raw Q-value tracking (Perplexity recommendation)
        "q1_mean", "q2_mean", "q1_max", "q2_max",
        # C006 B6: Gradient-active annotation
        "is_gradient_active",
        # Phase 5.1 post-unfreeze diagnostics
        "actor_loss_post_unfreeze",
        "r_prox_r_approach_ratio",
        "zero_residual_alert", "q_divergence_alert", "dimension_collapse_alert",
        # Intra-update critic loss observability (not intervention)
        "critic_loss_max", "critic_loss_p95",
        # Critic gradient norm (pre-clip) for monitoring clipping frequency
        "critic_grad_norm_mean", "critic_grad_norm_max",
        # Per-axis action sign and magnitude
        "action_sign_mean_behind", "action_sign_mean_cross", "action_sign_mean_below",
        "action_mag_mean_behind", "action_mag_mean_cross", "action_mag_mean_below",
        # C3v: Exact per-episode deterministic action decomposition
        "det_action_sign_mean_behind", "det_action_sign_mean_cross", "det_action_sign_mean_below",
        "det_action_mag_mean_behind", "det_action_mag_mean_cross", "det_action_mag_mean_below",
        "det_action_l2_norm",
        # F-2 fix: explicit restart-segment index. The CSV is opened in append
        # mode on --resume-from, so one file can span multiple training
        # processes while `episode`/`timestep` reset to 0 in each; readers
        # (H12, structural ceiling) must never window across that boundary.
        # Trailing position keeps positional consumers unbroken.
        "run_segment",
    ]

    def __init__(self, log_dir: str = ".", algorithm: str = "td3",
                 verbose: int = 0, resume: bool = False) -> None:
        super().__init__(verbose)
        self._log_dir = log_dir
        self._algorithm = algorithm
        self._resume = resume
        self._csv_path: Optional[str] = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._episode_count = 0
        # F-2: restart-segment index for this process's rows (0 for a fresh
        # file; last-segment+1 when appending). Legacy mode: appending onto a
        # pre-migration file whose header lacks run_segment — the open append
        # handle cannot rewrite the header, so rows stay legacy-shaped and
        # readers use their episode-reset fallback.
        self._run_segment = 0
        self._legacy_no_segment = False
        # C006 P2: Per-env state to prevent action/reward interleaving at P>1
        self._episode_actions: Dict[int, List[np.ndarray]] = {}
        self._episode_rewards: Dict[int, List[float]] = {}
        self._episode_obs: Dict[int, List[np.ndarray]] = {}  # C3v: exact decomposition
        self._zero_residual_streak: Dict[int, int] = {}
        # Accumulate per-step critic losses for max/p95 reporting
        self._critic_loss_accum: List[float] = []
        # T-480: per-episode critic grad-norm samples (one pair per train()
        # call): episode mean = mean of the per-call means, episode max = max
        # of the per-call maxes (the true episode extremum).
        self._critic_grad_norm_accum: List[float] = []
        self._critic_grad_norm_max_accum: List[float] = []
        # L9.3: Q1_max L25 rolling history for early-warning sentinel
        self._q1_max_history: List[float] = []
        # T-226: once-per-run latch for the DEMOTED Q_MAX telemetry line. Was
        # the existence of Q_MAX_EARLY_WARNING.txt; that file is no longer
        # written, so the latch has to live in memory.
        self._q1_max_telemetry_logged: bool = False
        # L9.3: Clip saturation tracking for alert sentinel
        self._clip_saturated_history: List[bool] = []

    def _init_callback(self) -> None:
        os.makedirs(self._log_dir, exist_ok=True)
        self._csv_path = os.path.join(self._log_dir, "offpolicy_diagnostics.csv")
        append = self._resume and os.path.exists(self._csv_path)
        if append:
            self._run_segment, self._legacy_no_segment = (
                self._resolve_segment(self._csv_path))
            if self._legacy_no_segment:
                logger.warning(
                    "[OPD] resuming onto pre-run_segment CSV %s; rows stay "
                    "legacy-shaped, readers fall back to episode-reset "
                    "boundary detection", self._csv_path)
        columns = (self._CSV_COLUMNS[:-1] if self._legacy_no_segment
                   else self._CSV_COLUMNS)
        # T-467: header verified; the legacy branch demands a LIST-IDENTICAL
        # header (allow_migrate=False) so the [:-1] trim can never be
        # silently half-migrated by future column growth.
        self._csv_file, self._csv_writer = _open_schema_checked_csv(
            self._csv_path, columns, self._resume,
            allow_migrate=not self._legacy_no_segment)
        self._csv_file.flush()

    @staticmethod
    def _resolve_segment(csv_path: str) -> tuple:
        """F-2: (next run_segment, legacy_mode) for an append onto csv_path.

        Header without run_segment => legacy mode (segment unused). Otherwise
        next segment = max parseable run_segment in the file + 1 (0 rows or
        all-unparseable => 1: something already occupies segment 0).
        """
        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or "run_segment" not in reader.fieldnames:
                    return 0, True
                last_seg = 0
                for row in reader:
                    try:
                        last_seg = max(last_seg, int(float(row["run_segment"])))
                    except (TypeError, ValueError, KeyError):
                        continue
                return last_seg + 1, False
        except OSError as exc:
            logger.warning("[OPD] run_segment resolution failed for %s: %s; "
                           "treating as legacy", csv_path, exc)
            return 0, True

    def _on_step(self) -> bool:
        # C006 P2: Per-env accumulation to prevent action/reward interleaving at P>1
        actions = self.locals.get("actions")   # shape (n_envs, action_dim)
        rewards = self.locals.get("rewards")   # shape (n_envs,)
        dones = self.locals.get("dones", [])

        for i in range(len(dones)):
            # Lazy-init per-env state
            if i not in self._episode_actions:
                self._episode_actions[i] = []
                self._episode_rewards[i] = []
                self._zero_residual_streak[i] = 0

            if actions is not None and len(actions) > i:
                self._episode_actions[i].append(np.array(actions[i], dtype=np.float32))
            if rewards is not None and len(rewards) > i:
                self._episode_rewards[i].append(float(rewards[i]))
            # C3v: store observation that produced this action (exact decomposition)
            if hasattr(self.model, '_last_obs') and self.model._last_obs is not None:
                if len(self.model._last_obs) > i:
                    self._episode_obs.setdefault(i, []).append(
                        np.array(self.model._last_obs[i], dtype=np.float32))

        # C3v: _last_obs ordering verification (fires first 3 episodes only)
        if self._episode_count < 3 and hasattr(self.model, '_last_obs'):
            new_obs = self.locals.get("new_obs")
            if new_obs is not None and self.model._last_obs is not None:
                assert self.model._last_obs is not new_obs, \
                    "FATAL: _last_obs IS new_obs (same object) — callback ordering violated"
                if not hasattr(self, '_last_obs_verified'):
                    logger.info("[C3v] _last_obs is not new_obs confirmed — exact decomposition valid")
                    self._last_obs_verified = True

        # Accumulate critic loss from SB3 logger (shared across envs)
        _cl = self.model.logger.name_to_value.get("train/critic_loss", float("nan"))
        if _cl == _cl:  # not NaN
            self._critic_loss_accum.append(_cl)

        # T-480: accumulate the per-train() critic grad-norm snapshots across
        # the EPISODE. custom_td3 aggregates over one train() call, whose
        # length is `gradient_steps` — 1 at n_envs=1, so mean==max exactly and
        # the pair could never disagree. Consumers (CLIP_SATURATION_ALERT, the
        # H12 monotonicity gate, tools/h12_calibration.py) all read these as
        # EPISODE aggregates; this accumulator makes the value match the name.
        _gn = getattr(self.model, "_critic_grad_norm_mean", float("nan"))
        if _gn == _gn:  # not NaN; absent pre-learning_starts
            self._critic_grad_norm_accum.append(float(_gn))
        _gx = getattr(self.model, "_critic_grad_norm_max", float("nan"))
        if _gx == _gx:
            self._critic_grad_norm_max_accum.append(float(_gx))

        for i in range(len(dones)):
            if dones[i]:
                self._episode_count += 1
                self._write_episode_row(env_id=i)
                self._episode_actions[i].clear()
                self._episode_rewards[i].clear()
                self._episode_obs.pop(i, None)  # C3v: remove key entirely

        return True

    def _sample_q_values(self) -> Dict[str, float]:
        """Sample Q1/Q2 statistics from the critic (C006 B5).

        Draws 256 transitions from the replay buffer, forward-passes through
        the critic networks, and returns mean/max per critic. Returns NaN
        on any error to never crash training.
        """
        nan_result = {"q1_mean": float("nan"), "q2_mean": float("nan"),
                      "q1_max": float("nan"), "q2_max": float("nan")}
        try:
            buf = getattr(self.model, "replay_buffer", None)
            if buf is None or buf.size() < 256:
                return nan_result

            critic = getattr(self.model, "critic", None)
            if critic is None:
                return nan_result

            # Sample from replay buffer
            replay_data = buf.sample(256, env=self.training_env)
            obs = replay_data.observations
            actions = replay_data.actions

            with torch.no_grad():
                q1, q2 = critic(obs, actions)

            return {
                "q1_mean": float(q1.mean().item()),
                "q2_mean": float(q2.mean().item()),
                "q1_max": float(q1.max().item()),
                "q2_max": float(q2.max().item()),
            }
        except Exception:
            logger.debug("[OffPolicyDiag] Q-value sampling failed", exc_info=True)
            return nan_result

    def _write_episode_row(self, env_id: int = 0) -> None:
        """Compute stats and write one CSV row per completed episode."""
        if self._csv_writer is None:
            return

        # --- Action statistics (per-env) ---
        action_l2_norm = float("nan")
        action_mean = [float("nan")] * 3
        action_std = [float("nan")] * 3
        action_sign_mean = [float("nan")] * 3
        action_mag_mean = [float("nan")] * 3
        ep_actions = self._episode_actions.get(env_id, [])
        if ep_actions:
            actions_arr = np.array(ep_actions)  # (T, 2) under 2D, (T, 3) under legacy 3D
            action_l2_norm = float(np.mean(np.linalg.norm(actions_arr, axis=1)))
            # _legacy_triplet maps 2D [speed, altitude] → 3-element [speed, NaN, altitude]
            # to preserve CSV column semantics; legacy 3D path is bit-equivalent.
            action_mean = _legacy_triplet(actions_arr, np.mean)
            action_std = _legacy_triplet(actions_arr, np.std)
            action_sign_mean = _legacy_triplet(
                actions_arr, lambda a: np.mean(np.sign(a)))
            action_mag_mean = _legacy_triplet(
                actions_arr, lambda a: np.mean(np.abs(a)))

        # --- C3v: Exact per-episode deterministic action decomposition ---
        # Computes actor output WITHOUT exploration noise on the SAME observations
        # that produced the stochastic actions logged above.
        det_sign_mean = [float("nan")] * 3
        det_mag_mean = [float("nan")] * 3
        det_l2_norm = float("nan")
        try:
            ep_obs = self._episode_obs.get(env_id, [])
            if ep_obs and len(ep_obs) >= 10:
                obs_arr = np.array(ep_obs)  # (T, obs_dim)
                obs_tensor = torch.as_tensor(
                    obs_arr, device=self.model.device, dtype=torch.float32)
                with torch.no_grad():
                    det_actions = self.model.actor(obs_tensor)  # (T, action_dim) — 2 under 2D, 3 under legacy
                det_np = det_actions.cpu().numpy()
                n_act_det = det_np.shape[1]
                if n_act_det not in (2, 3):
                    raise RuntimeError(
                        f"OffPolicyDiagnosticsCallback (det path): unexpected "
                        f"action dimension {n_act_det}; expected 2 (C3X 2D) or 3 (legacy)"
                    )
                det_sign_mean = _legacy_triplet(
                    det_np, lambda a: np.mean(np.sign(a)))
                det_mag_mean = _legacy_triplet(
                    det_np, lambda a: np.mean(np.abs(a)))
                # det_l2_norm: explicit slice over actual action dim (was hardcoded
                # `det_np[:, :3]`, which silently tolerated under 2D via NumPy slice).
                det_l2_norm = float(np.mean(np.linalg.norm(det_np[:, :n_act_det], axis=1)))
        except Exception:
            logger.debug("[C3v] Deterministic action logging failed", exc_info=True)

        # --- SB3 logged losses (from model.logger) ---
        name_to_value = self.model.logger.name_to_value
        actor_loss = name_to_value.get("train/actor_loss", float("nan"))
        critic_loss = name_to_value.get("train/critic_loss", float("nan"))
        ent_coef_alpha = name_to_value.get("train/ent_coef", float("nan"))

        # --- Buffer utilisation ---
        buffer_util = float("nan")
        if hasattr(self.model, 'replay_buffer'):
            buf = self.model.replay_buffer
            buffer_util = buf.size() / buf.buffer_size if buf.buffer_size > 0 else 0.0

        # --- Alert checks ---
        zero_residual_alert = False
        q_divergence_alert = False
        dimension_collapse_alert = False

        # Zero-residual: action L2 norm < threshold for consecutive episodes (per-env)
        if (action_l2_norm == action_l2_norm
                and action_l2_norm < config.ZERO_RESIDUAL_ALERT_THRESHOLD):
            self._zero_residual_streak[env_id] = self._zero_residual_streak.get(env_id, 0) + 1
        else:
            self._zero_residual_streak[env_id] = 0

        if self._zero_residual_streak.get(env_id, 0) >= config.ZERO_RESIDUAL_ALERT_EPISODES:
            zero_residual_alert = True
            self._write_alert("ALERT_ZERO_RESIDUAL.txt",
                              f"Action L2 norm < {config.ZERO_RESIDUAL_ALERT_THRESHOLD} "
                              f"for {self._zero_residual_streak[env_id]} consecutive episodes "
                              f"(env {env_id}) at episode {self._episode_count}")

        # C006 B5: Sample raw Q-values from critic
        q_stats = self._sample_q_values()

        # C006 B6: Gradient-active annotation
        is_gradient_active = _check_gradient_active(self.model, self.num_timesteps)

        # Phase 5.1 actor_loss_post_unfreeze — NaN during freeze
        actor_loss_post_unfreeze = float("nan")
        if is_gradient_active and actor_loss == actor_loss:
            actor_loss_post_unfreeze = actor_loss

        # Phase 5.1 r_prox_r_approach_ratio from MetricsCallback logger
        # abs(r_prox) because r_prox is negative in the log branch (punishment)
        # and positive in the PBRS branch; abs() normalises to "magnitude of
        # proximity signal" so the ratio is a dimensionless punishment-to-incentive scale.
        r_prox_mean = name_to_value.get("episode/r_prox_mean", float("nan"))
        r_approach_mean = name_to_value.get("episode/r_approach_mean", float("nan"))
        r_prox_r_approach_ratio = float("nan")
        if (r_approach_mean == r_approach_mean and r_approach_mean > 1e-9
                and r_prox_mean == r_prox_mean):
            r_prox_r_approach_ratio = abs(r_prox_mean) / r_approach_mean

        # Q-divergence: dual-condition gate (Phase 7A) — requires BOTH
        # critic_loss >> reward AND |Q1_max| > absolute threshold to fire.
        # Prevents false positives under near-zero reward regimes (|reward| ≈ 0.3).
        q1_max = q_stats.get("q1_max", float("nan"))
        ep_rewards = self._episode_rewards.get(env_id, [])
        if (ep_rewards and critic_loss == critic_loss):
            mean_ep_reward = abs(np.mean(ep_rewards))
            if (mean_ep_reward > 1e-6
                    and critic_loss > config.Q_DIVERGENCE_MULTIPLIER * mean_ep_reward
                    and q1_max == q1_max and abs(q1_max) > config.Q_ABSOLUTE_THRESHOLD):
                q_divergence_alert = True
                self._write_alert("ALERT_Q_DIVERGENCE.txt",
                                  f"critic_loss={critic_loss:.2f} > "
                                  f"{config.Q_DIVERGENCE_MULTIPLIER}x mean_reward="
                                  f"{mean_ep_reward:.4f} AND |Q1_max|={abs(q1_max):.2f} > "
                                  f"{config.Q_ABSOLUTE_THRESHOLD} "
                                  f"at episode {self._episode_count}")

        # T-226: DEMOTED from ALERT to TELEMETRY.
        # NOT a retune — the threshold is unchanged and deliberately so.
        # F-2 (2026-07-30) refuted raising a LEVEL bar on |q1_max|: magnitude is
        # ANTI-correlated with failure (peaks 4453 healthy, falls to 1984 through
        # the real collapse). analysis/qguard_rate_20260803T192000Z extended that
        # refutation to the DERIVATIVE: no rate statistic separates either —
        # detection swings 60%@w10 -> 0%@w25 (a window-choice artifact) and the
        # apparent lead time is indistinguishable from a 6.1% false-positive
        # rate. The defect is STRUCTURAL, not numeric, so no value fixes it.
        # Writing an alert file that fires regardless of health trains operators
        # to ignore alerts; the number keeps forensic value, so it is LOGGED.
        # Full per-episode q1_max already lands in offpolicy_diagnostics.csv.
        # Divergence detection of record is E-4 (dwell) — it watches the
        # OUTCOME rather than a proxy for it.
        if q1_max == q1_max:  # not NaN
            _look = config.Q_MAX_EARLY_WARNING_LOOKBACK
            self._q1_max_history.append(q1_max)
            if len(self._q1_max_history) > _look:
                self._q1_max_history = self._q1_max_history[-_look:]
            _q1_l25_max = max(self._q1_max_history)
            if (_q1_l25_max > config.Q_MAX_EARLY_WARNING_THRESHOLD
                    and not self._q1_max_telemetry_logged):
                self._q1_max_telemetry_logged = True
                logger.info(
                    "[Q_MAX_TELEMETRY] Q1_max L%d rolling max=%.2f > %s at "
                    "episode %d (ADVISORY ONLY, non-discriminating by "
                    "measurement — see analysis/qguard_rate_20260803T192000Z)",
                    _look, _q1_l25_max,
                    config.Q_MAX_EARLY_WARNING_THRESHOLD, self._episode_count)

        # Dimension collapse: any action dimension std < 0.01
        for i, std in enumerate(action_std):
            if std == std and std < 0.01 and self._episode_count > 10:
                dimension_collapse_alert = True
                dim_names = ["speed", "heading", "altitude"]
                name = dim_names[i] if i < len(dim_names) else f"dim{i}"
                self._write_alert("ALERT_DIMENSION_COLLAPSE.txt",
                                  f"Action dimension '{name}' std={std:.6f} < 0.01 "
                                  f"at episode {self._episode_count}")
                break

        row = {
            "episode": self._episode_count,
            "timestep": self.num_timesteps,
            "env_id": env_id,
            "algorithm": self._algorithm,
            "actor_loss": actor_loss if actor_loss == actor_loss else float("nan"),
            "critic_loss": critic_loss if critic_loss == critic_loss else float("nan"),
            "action_l2_norm": action_l2_norm,
            "action_mean_speed": action_mean[0],
            "action_mean_heading": action_mean[1],
            "action_mean_altitude": action_mean[2],
            "action_std_speed": action_std[0],
            "action_std_heading": action_std[1],
            "action_std_altitude": action_std[2],
            "buffer_utilisation": buffer_util,
            "ent_coef_alpha": ent_coef_alpha if ent_coef_alpha == ent_coef_alpha else float("nan"),
            # C006 B5: Raw Q-value tracking
            "q1_mean": q_stats["q1_mean"],
            "q2_mean": q_stats["q2_mean"],
            "q1_max": q_stats["q1_max"],
            "q2_max": q_stats["q2_max"],
            # C006 B6: Gradient-active annotation
            "is_gradient_active": is_gradient_active,
            # Phase 5.1
            "actor_loss_post_unfreeze": actor_loss_post_unfreeze,
            "r_prox_r_approach_ratio": r_prox_r_approach_ratio,
            "zero_residual_alert": zero_residual_alert,
            "q_divergence_alert": q_divergence_alert,
            "dimension_collapse_alert": dimension_collapse_alert,
            # Intra-update critic loss observability
            "critic_loss_max": (float(np.max(self._critic_loss_accum))
                                if self._critic_loss_accum else float("nan")),
            "critic_loss_p95": (float(np.percentile(self._critic_loss_accum, 95))
                                if self._critic_loss_accum else float("nan")),
            # Critic gradient norm (pre-clip)
            # T-480: EPISODE-scoped, not the last train() call's 1-sample window
            "critic_grad_norm_mean": (float(np.mean(self._critic_grad_norm_accum))
                                      if self._critic_grad_norm_accum
                                      else float("nan")),
            "critic_grad_norm_max": (float(np.max(self._critic_grad_norm_max_accum))
                                     if self._critic_grad_norm_max_accum
                                     else float("nan")),
            # Per-axis action sign and magnitude
            "action_sign_mean_behind": action_sign_mean[0],
            "action_sign_mean_cross": action_sign_mean[1],
            "action_sign_mean_below": action_sign_mean[2],
            "action_mag_mean_behind": action_mag_mean[0],
            "action_mag_mean_cross": action_mag_mean[1],
            "action_mag_mean_below": action_mag_mean[2],
            # C3v: exact per-episode deterministic action decomposition
            "det_action_sign_mean_behind": det_sign_mean[0],
            "det_action_sign_mean_cross": det_sign_mean[1],
            "det_action_sign_mean_below": det_sign_mean[2],
            "det_action_mag_mean_behind": det_mag_mean[0],
            "det_action_mag_mean_cross": det_mag_mean[1],
            "det_action_mag_mean_below": det_mag_mean[2],
            "det_action_l2_norm": det_l2_norm,
        }
        if not self._legacy_no_segment:
            row["run_segment"] = self._run_segment
        self._critic_loss_accum.clear()
        self._critic_grad_norm_accum.clear()          # T-480: per-episode
        self._critic_grad_norm_max_accum.clear()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # L9.3: CLIP_SATURATION_ALERT — clip_rate > 95% for 2 consecutive L25 windows
        _grad_max = row.get("critic_grad_norm_max", float("nan"))
        _clip_ceil = getattr(self.model, "max_grad_norm_critic", None)
        if (isinstance(_clip_ceil, (int, float))
                and _clip_ceil == _clip_ceil  # not NaN
                and _grad_max == _grad_max):  # not NaN
            self._clip_saturated_history.append(_grad_max >= _clip_ceil)
            if len(self._clip_saturated_history) > 50:
                self._clip_saturated_history = self._clip_saturated_history[-50:]
            if len(self._clip_saturated_history) >= 50:
                _w1_rate = sum(self._clip_saturated_history[:25]) / 25
                _w2_rate = sum(self._clip_saturated_history[25:]) / 25
                if _w1_rate > 0.95 and _w2_rate > 0.95:
                    _cs_path = os.path.join(self._log_dir, "CLIP_SATURATION_ALERT.txt")
                    if not os.path.exists(_cs_path):
                        self._write_alert(
                            "CLIP_SATURATION_ALERT.txt",
                            f"Clip saturation: 2 consecutive L25 windows > 95% "
                            f"(w1={_w1_rate:.0%}, w2={_w2_rate:.0%}) "
                            f"at episode {self._episode_count}")

        # TensorBoard logging
        self.logger.record("offpolicy/action_l2_norm", action_l2_norm)
        self.logger.record("offpolicy/buffer_utilisation", buffer_util)
        if actor_loss == actor_loss:
            self.logger.record("offpolicy/actor_loss", actor_loss)
        if critic_loss == critic_loss:
            self.logger.record("offpolicy/critic_loss", critic_loss)
        # C006 B5: Q-value TensorBoard logging
        for k in ("q1_mean", "q2_mean", "q1_max", "q2_max"):
            v = q_stats[k]
            if v == v:  # not NaN
                self.logger.record(f"offpolicy/{k}", v)
        self.logger.record("offpolicy/is_gradient_active", int(is_gradient_active))

    def _write_alert(self, filename: str, message: str) -> None:
        """Write alert file for watchdog integration."""
        try:
            alert_path = os.path.join(self._log_dir, filename)
            with open(alert_path, "w") as f:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{now}] {message}\n")
            logger.warning(f"[OffPolicyDiag] {message}")
        except Exception:
            logger.debug("Failed to write alert file", exc_info=True)

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()


class VecNormalizeFreezeCallback(BaseCallback):
    """Freeze VecNormalize observation statistics after N episodes (v18).

    Off-policy algorithms store unnormalized observations in the replay buffer.
    On sample(), SB3 re-normalizes using current VecNormalize stats. If stats
    keep updating, old transitions get re-normalized with different statistics
    than when they were collected, causing distribution shift in the critic.

    Freezing after VECNORM_CALIBRATION_EPISODES (30) episodes ensures stats are
    representative of the observation distribution but remain stable for the
    replay buffer's lifetime (Ball et al., ICML 2023).

    Works identically for all algorithms (episode counting is algorithm-agnostic),
    but primarily important for off-policy methods with replay buffers.

    One-shot: once frozen, all future checks are skipped.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_count = 0
        self._frozen = False

    def _on_step(self) -> bool:
        if self._frozen:
            return True

        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._episode_count += 1

                if self._episode_count >= config.VECNORM_CALIBRATION_EPISODES:
                    self._freeze()
                    return True

        return True

    def _freeze(self) -> None:
        """Freeze VecNormalize stats."""
        self._frozen = True
        if isinstance(self.training_env, VecNormalize):
            self.training_env.training = False
            logger.info(f"[VecNormFreeze] Froze VecNormalize stats after "
                        f"{self._episode_count} episodes "
                        f"(threshold: {config.VECNORM_CALIBRATION_EPISODES})")
            self.logger.record("vecnorm/frozen_at_episode", self._episode_count)
        else:
            logger.warning("[VecNormFreeze] training_env is not VecNormalize — "
                           "cannot freeze")


class VecNormalizeScheduleCallback(BaseCallback):
    """Unified VecNormalize freeze/unfreeze scheduler (Campaign 2b).

    Single authority over vec_normalize.training, replacing the conflicting
    VecNormalizeFreezeCallback (hardcoded 30-ep) and VecNormalizeUnfreezeCallback
    (CLI-specified N-ep).

    Behavior:
    - If unfreeze_at_init=True: sets training=True at construction
    - Counts completed episodes via dones array (P>1 safe)
    - At episode >= freeze_after_episodes: sets training=False (one-shot)
    """

    def __init__(self, vec_env: VecNormalize, freeze_after_episodes: int,
                 unfreeze_at_init: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self._vec_env = vec_env
        self._freeze_after_episodes = freeze_after_episodes
        self._unfreeze_at_init = unfreeze_at_init
        self._episode_count = 0
        self._frozen = False

        if self._unfreeze_at_init:
            self._vec_env.training = True
            logger.info(f"[VecNormSchedule] Unfrozen at init. "
                        f"Will freeze after {freeze_after_episodes} episodes.")

    def _on_step(self) -> bool:
        if self._frozen:
            return True
        dones = self.locals.get("dones", [])
        self._episode_count += sum(dones)
        if self._episode_count >= self._freeze_after_episodes:
            self._vec_env.training = False
            self._frozen = True
            logger.info(f"[VecNormSchedule] Frozen at episode {self._episode_count} "
                        f"(threshold: {self._freeze_after_episodes})")
            self.logger.record("vecnorm/frozen_at_episode", self._episode_count)
        return True


class BufferDilutionCheckCallback(BaseCallback):
    """Verify warmup data does NOT enter the off-policy replay buffer (v18 Phase C).

    Risk R4 from the governing document (§2.7) requires empirical proof that
    warmup transitions are excluded. This callback records replay_buffer.size()
    at episode boundaries and checks that the delta equals only the DRL steps
    in the previous episode — not steps + warmup ticks.

    Writes results to ``buffer_dilution_check.csv``. Designed as a Phase C
    diagnostic but safe to keep running as defense evidence.

    Only meaningful for off-policy algorithms (TD3/SAC) that use a replay buffer.
    """

    _CSV_COLUMNS = [
        "episode", "timestep", "env_id", "pre_reset_size", "post_reset_size",
        "delta", "episode_length", "verdict",
    ]

    def __init__(self, log_dir: str = ".", verbose: int = 0,
                 resume: bool = False) -> None:
        super().__init__(verbose)
        self._log_dir = log_dir
        self._resume = resume
        self._csv_path: Optional[str] = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._episode_count = 0
        # C006 P2: Per-env state to handle simultaneous dones at P>1
        self._episode_step_count: Dict[int, int] = {}
        self._pre_reset_size: Dict[int, Optional[int]] = {}
        self._awaiting_post_reset: Dict[int, bool] = {}

    def _init_callback(self) -> None:
        os.makedirs(self._log_dir, exist_ok=True)
        self._csv_path = os.path.join(self._log_dir, "buffer_dilution_check.csv")
        # T-467: header verified against the live column list (defect T-466)
        self._csv_file, self._csv_writer = _open_schema_checked_csv(
            self._csv_path, self._CSV_COLUMNS, self._resume)
        self._csv_file.flush()

    def _get_buffer_size(self) -> int:
        """Return current replay buffer size, or -1 if unavailable."""
        if hasattr(self.model, 'replay_buffer'):
            return self.model.replay_buffer.size()
        return -1

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])

        for i in range(len(dones)):
            # Lazy-init per-env state
            if i not in self._episode_step_count:
                self._episode_step_count[i] = 0
                self._pre_reset_size[i] = None
                self._awaiting_post_reset[i] = False

            # Track per-env step count
            self._episode_step_count[i] += 1

            # After a reset (new episode start), record post-reset buffer size
            if self._awaiting_post_reset[i] and self._pre_reset_size[i] is not None:
                post_size = self._get_buffer_size()
                delta = post_size - self._pre_reset_size[i]
                verdict = "PASS" if delta <= 1 else "CHECK"
                if self._csv_writer is not None:
                    row = {
                        "episode": self._episode_count,
                        "timestep": self.num_timesteps,
                        "env_id": i,
                        "pre_reset_size": self._pre_reset_size[i],
                        "post_reset_size": post_size,
                        "delta": delta,
                        "episode_length": self._episode_step_count[i] - 1,
                        "verdict": verdict,
                    }
                    self._csv_writer.writerow(row)
                    self._csv_file.flush()
                if verdict != "PASS":
                    logger.warning(
                        f"[BufferDilution] Episode {self._episode_count} (env {i}): "
                        f"delta={delta} (pre={self._pre_reset_size[i]}, post={post_size}). "
                        f"Possible warmup contamination."
                    )
                self._awaiting_post_reset[i] = False

            # Check for episode completion
            if dones[i]:
                self._episode_count += 1
                self._pre_reset_size[i] = self._get_buffer_size()
                self._awaiting_post_reset[i] = True
                self._episode_step_count[i] = 0

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()


class AtomicOffPolicyCheckpointCallback(BaseCallback):
    """Atomic checkpoint saves for off-policy algorithms (v19 P3).

    Replaces SB3's built-in CheckpointCallback which uses direct model.save()
    — if the process is killed mid-write (OOM, watchdog), the checkpoint is
    corrupted. The replay buffer is the single most valuable asset in
    off-policy training; losing it wastes hours of SITL simulation.

    Atomic pattern: write to ``<path>.tmp``, then ``os.replace()`` (POSIX
    atomic rename). The final file is either complete or absent — never
    partially written.

    Parameters
    ----------
    save_freq: int
        Save checkpoint every this many steps.
    save_path: str
        Directory for checkpoint files.
    name_prefix: str
        Prefix for checkpoint filenames.
    save_replay_buffer: bool
        Also save the replay buffer (recommended for off-policy resume).
    save_vecnormalize: bool
        Also save VecNormalize statistics.
    """

    def __init__(
        self,
        save_freq: int = 3600,
        save_path: str = "checkpoints",
        name_prefix: str = "model",
        save_replay_buffer: bool = True,
        save_vecnormalize: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._save_freq = save_freq
        self._save_path = save_path
        self._name_prefix = name_prefix
        self._save_replay_buffer = save_replay_buffer
        self._save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        os.makedirs(self._save_path, exist_ok=True)

    def _atomic_save(self, save_fn, final_path: str) -> None:
        """Write to .tmp then os.replace() for POSIX atomic rename."""
        tmp_path = final_path + ".tmp"
        try:
            save_fn(tmp_path)
            os.replace(tmp_path, final_path)
        except Exception:
            # Clean up partial .tmp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            self._save_checkpoint()
        return True

    def _save_checkpoint(self) -> None:
        """Atomically save model, replay buffer, and VecNormalize."""
        step = self.num_timesteps
        prefix = f"{self._name_prefix}_{step}_steps"

        # Model
        model_path = os.path.join(self._save_path, f"{prefix}.zip")
        self._atomic_save(self.model.save, model_path)
        logger.info(f"[AtomicCheckpoint] Model saved: {model_path}")

        # Replay buffer
        if self._save_replay_buffer and hasattr(self.model, 'replay_buffer'):
            buf_path = os.path.join(self._save_path, f"replay_buffer_{step}_steps.pkl")
            self._atomic_save(self.model.save_replay_buffer, buf_path)
            logger.info(f"[AtomicCheckpoint] Replay buffer saved: {buf_path} "
                        f"({self.model.replay_buffer.size()} transitions)")

        # VecNormalize
        if self._save_vecnormalize and isinstance(self.training_env, VecNormalize):
            norm_path = os.path.join(self._save_path, f"vecnormalize_{step}_steps.pkl")
            self._atomic_save(self.training_env.save, norm_path)
            logger.info(f"[AtomicCheckpoint] VecNormalize saved: {norm_path}")


class RewardMonitorCallback(BaseCallback):
    """Detect "flat learning" campaigns and alert the operator (v11).

    Watches ``r_prox`` (proximity reward) from the MetricsCallback CSV and
    ``explained_variance`` from the SB3 logger. If both are flat over a
    sliding window, the campaign is not learning — alert immediately rather
    than burning GPU hours on a dead run.

    Alerts:
      1. Writes ``ALERT_NO_LEARNING.txt`` to ``log_dir`` (always)
      2. Desktop notification via ``notify-send`` (best-effort)
      3. Console bell (best-effort)

    Parameters
    ----------
    log_dir: str
        Directory for the alert file.
    min_rollouts: int
        Skip checks until this many rollouts have completed.
    window: int
        Number of recent rollouts to compare.
    r_prox_threshold: float
        If the range (max-min) of r_prox over the window is below this, r_prox
        is considered flat.
    ev_threshold: float
        If the mean explained_variance over the window is below this, the
        critic is considered uninformative.
    auto_stop: bool
        If True, cleanly stop training when flat learning is detected.
    """

    def __init__(
        self,
        log_dir: str = ".",
        min_rollouts: int = 10,
        window: int = 8,
        r_prox_threshold: float = 0.05,
        ev_threshold: float = 0.10,
        auto_stop: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._log_dir = log_dir
        self._min_rollouts = min_rollouts
        self._window = window
        self._r_prox_threshold = r_prox_threshold
        self._ev_threshold = ev_threshold
        self._auto_stop = auto_stop
        self._should_stop = False
        self._rollout_count = 0
        self._r_prox_history: Deque[float] = deque(maxlen=window)
        self._ev_history: Deque[float] = deque(maxlen=window)
        self._alerted = False

    def _on_step(self) -> bool:
        # Collect per-episode r_prox values for the rolling window.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                comps = info.get("reward_components", {})
                r_prox = comps.get("r_prox", None)
                if r_prox is not None:
                    self._r_prox_history.append(float(r_prox))
        return not self._should_stop

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count < self._min_rollouts:
            return

        # Collect explained_variance from SB3 logger
        ev = self.model.logger.name_to_value.get("train/explained_variance")
        if ev is not None:
            self._ev_history.append(float(ev))

        self._check_learning_signal()

    def _check_learning_signal(self) -> None:
        """Check if both r_prox and explained_variance are flat."""
        if self._alerted:
            return
        if len(self._r_prox_history) < self._window:
            return
        if len(self._ev_history) < self._window:
            return

        # r_prox range over last `window` episodes (deque is already bounded)
        rprox_range = max(self._r_prox_history) - min(self._r_prox_history)

        # explained_variance mean over last `window` rollouts
        ev_mean = sum(self._ev_history) / len(self._ev_history)

        if rprox_range < self._r_prox_threshold and ev_mean < self._ev_threshold:
            self._trigger_alert(rprox_range, ev_mean)

    def _trigger_alert(self, rprox_range: float, ev_mean: float) -> None:
        """Write alert file and attempt desktop/audio notification."""
        self._alerted = True
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"[ALERT] No learning signal detected at rollout {self._rollout_count}\n"
            f"  Time:           {now}\n"
            f"  r_prox range:   {rprox_range:.4f} (threshold: {self._r_prox_threshold})\n"
            f"  EV mean:        {ev_mean:.4f} (threshold: {self._ev_threshold})\n"
            f"  Window:         {self._window} rollouts\n"
            f"  Auto-stop:      {self._auto_stop}\n"
        )
        logger.warning(msg)

        # 1. Write durable alert file
        alert_path = os.path.join(self._log_dir, "ALERT_NO_LEARNING.txt")
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            with open(alert_path, "w") as f:
                f.write(msg)
        except Exception:
            logger.debug("Failed to write alert file", exc_info=True)

        # 2. Desktop notification (best-effort)
        try:
            subprocess.run(
                ["notify-send", "-u", "critical", "FlightDock DRL",
                 f"No learning signal at rollout {self._rollout_count}"],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass  # No desktop environment or notify-send unavailable

        # 3. Console bell (best-effort)
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass

        # 4. Auto-stop if configured
        if self._auto_stop:
            logger.warning("[RewardMonitor] auto_stop=True — will stop training at next step()")
            self._should_stop = True


class HeartbeatCallback(BaseCallback):
    """Touch per-worker heartbeat files for watchdog stall detection (v13 Commit 12).

    The external watchdog (scripts/training_watchdog.sh) monitors per-worker
    heartbeat file mtimes. If any goes stale for >5 minutes, the watchdog
    assumes the training loop is hung and kills the process.

    v13 three-layer defense:
      Layer 1: WallClockStallDetector (30s) unblocks step_wait()
      Layer 2: In-worker heartbeat touch in step() (backup liveness)
      Layer 3: This callback + PPO-gap daemon + watchdog (5 min kill threshold)

    During PPO train() (~210s at P=4), no _on_step fires. A daemon thread
    launched at _on_rollout_end keeps heartbeat files fresh until _on_step
    resumes.
    """

    def __init__(self, interval: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.interval = interval
        self._onpolicy_hb_stop: Optional[threading.Event] = None
        self._onpolicy_hb_thread: Optional[threading.Thread] = None

    def _touch_all(self) -> None:
        """Touch all per-worker heartbeat files."""
        n = self.training_env.num_envs
        for i in range(n):
            Path(config.HEARTBEAT_PER_WORKER_TEMPLATE.format(env_id=i)).touch(
                exist_ok=True
            )

    def _on_step(self) -> bool:
        # Kill on-policy update daemon thread (if running from previous rollout end)
        self._stop_onpolicy_heartbeat()
        if self.n_calls % self.interval == 0:
            self._touch_all()
        return True

    def _on_rollout_end(self) -> None:
        # PPO train() is about to run — no _on_step fires during it.
        # Launch daemon thread to keep heartbeat files fresh.
        # For off-policy (TD3/SAC), train() is <15s so no daemon needed.
        self._touch_all()
        # On-policy algorithms (PPO) lack replay_buffer — need daemon heartbeat
        # during long training updates that block step() for minutes.
        if not hasattr(self.model, 'replay_buffer'):
            self._onpolicy_hb_stop = threading.Event()

            def _onpolicy_heartbeat():
                while not self._onpolicy_hb_stop.wait(30.0):
                    self._touch_all()

            self._onpolicy_hb_thread = threading.Thread(
                target=_onpolicy_heartbeat, daemon=True
            )
            self._onpolicy_hb_thread.start()

    def _stop_onpolicy_heartbeat(self) -> None:
        if self._onpolicy_hb_stop is not None:
            self._onpolicy_hb_stop.set()
        if self._onpolicy_hb_thread is not None:
            self._onpolicy_hb_thread.join(timeout=5.0)
        self._onpolicy_hb_stop = None
        self._onpolicy_hb_thread = None

    def _on_training_end(self) -> None:
        self._stop_onpolicy_heartbeat()


class ZeroResidualFallbackCallback(BaseCallback):
    """Auto-switch TD3→SAC on zero-residual detection (v19 P8).

    For unattended overnight campaigns: if the TD3 policy gets trapped at
    zero-residual (action L2 norm < threshold for consecutive episodes),
    this callback:
      1. Forces an atomic checkpoint (model + replay buffer + VecNormalize)
      2. Writes a ``FALLBACK_REQUEST.json`` sentinel to the log directory
      3. Returns ``False`` to cleanly stop training

    The wrapper script (``run_drl_training.sh``) detects the sentinel and
    relaunches with ``--algorithm sac --resume-from <checkpoint>``.

    Only active for TD3 — SAC is the fallback target, so no further fallback.

    Parameters
    ----------
    log_dir: str
        Directory for the sentinel file.
    checkpoint_dir: str
        Directory for the fallback checkpoint.
    min_episodes: int
        Skip detection until this many episodes complete (let VecNormalize calibrate).
    """

    def __init__(
        self,
        log_dir: str = ".",
        checkpoint_dir: str = "checkpoints",
        min_episodes: int = 15,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._log_dir = log_dir
        self._checkpoint_dir = checkpoint_dir
        self._min_episodes = min_episodes
        self._episode_count = 0
        # C006 P7: Per-env streak tracking (was shared scalar — bug at P>1)
        self._zero_streak: Dict[int, int] = {}      # env_id → consecutive zero episodes
        self._env_episode_count: Dict[int, int] = {}  # env_id → episode count
        self._triggered = False

    def _on_step(self) -> bool:
        if self._triggered:
            return False  # Stop training

        dones = self.locals.get("dones", [])
        actions = self.locals.get("actions")

        for i, done in enumerate(dones):
            if done:
                self._episode_count += 1
                # C006 P7: Per-env lazy init
                if i not in self._zero_streak:
                    self._zero_streak[i] = 0
                    self._env_episode_count[i] = 0
                self._env_episode_count[i] += 1

                if self._env_episode_count[i] < self._min_episodes:
                    continue

                # Check terminal action L2 norm per-env
                if actions is not None and len(actions) > i:
                    action_l2 = float(np.linalg.norm(actions[i]))
                    if action_l2 < config.ZERO_RESIDUAL_ALERT_THRESHOLD:
                        self._zero_streak[i] += 1
                    else:
                        self._zero_streak[i] = 0

                    # Trigger if ANY env has sustained zero-residual streak
                    if self._zero_streak[i] >= config.ZERO_RESIDUAL_ALERT_EPISODES:
                        self._trigger_fallback(env_id=i)
                        return False

        return True

    def _trigger_fallback(self, env_id: int = 0) -> None:
        """Force checkpoint + write sentinel + stop training."""
        self._triggered = True
        from datetime import datetime as _dt

        logger.warning(
            f"[ZeroResidualFallback] TD3 zero-residual trap detected at episode "
            f"{self._episode_count} (env_id={env_id}, "
            f"streak={self._zero_streak.get(env_id, 0)}). "
            f"Forcing checkpoint and writing FALLBACK_REQUEST."
        )

        # 1. Force atomic checkpoint
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self._checkpoint_dir, f"model_fallback_{self.num_timesteps}_steps.zip"
        )
        try:
            tmp_path = checkpoint_path + ".tmp"
            self.model.save(tmp_path)
            os.replace(tmp_path, checkpoint_path)
            logger.info(f"[ZeroResidualFallback] Checkpoint: {checkpoint_path}")

            # Save replay buffer
            buf_path = checkpoint_path.replace("model_", "replay_buffer_").replace(".zip", ".pkl")
            tmp_buf = buf_path + ".tmp"
            self.model.save_replay_buffer(tmp_buf)
            os.replace(tmp_buf, buf_path)
            logger.info(f"[ZeroResidualFallback] Replay buffer: {buf_path}")

            # Save VecNormalize
            if isinstance(self.training_env, VecNormalize):
                norm_path = checkpoint_path.replace(
                    "model_", "vecnormalize_").replace(".zip", ".pkl")
                tmp_norm = norm_path + ".tmp"
                self.training_env.save(tmp_norm)
                os.replace(tmp_norm, norm_path)
        except Exception:
            logger.error("[ZeroResidualFallback] Checkpoint save failed", exc_info=True)
            # Clean up tmp files
            rb_tmp = checkpoint_path.replace(
                "model_", "replay_buffer_"
            ).replace(".zip", ".pkl") + ".tmp"
            for p in [checkpoint_path + ".tmp", rb_tmp]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        # 2. Write sentinel
        sentinel_path = os.path.join(self._log_dir, "FALLBACK_REQUEST.json")
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            sentinel = {
                "timestamp": _dt.now().isoformat(),
                "reason": "zero_residual_trap",
                "episode": self._episode_count,
                "timestep": self.num_timesteps,
                "streak": self._zero_streak,
                "checkpoint": checkpoint_path,
                "fallback_algorithm": "sac",
            }
            with open(sentinel_path, "w") as f:
                json.dump(sentinel, f, indent=2)
            logger.warning(f"[ZeroResidualFallback] Sentinel written: {sentinel_path}")
        except Exception:
            logger.error("[ZeroResidualFallback] Failed to write sentinel", exc_info=True)


# ---------------------------------------------------------------------------
# VecNormalize unfreeze callback (C006 Phase 3)
# Moved from train.py during structural cleanup.
# ---------------------------------------------------------------------------


class VecNormalizeUnfreezeCallback(BaseCallback):
    """Temporarily unfreeze VecNormalize for N episodes after resume.

    When resuming with a new reward distribution (e.g., r_prox decay),
    the frozen VecNormalize running stats may be stale. This callback
    unfreezes stats collection for a calibration window, then re-freezes.
    """

    def __init__(self, unfreeze_episodes: int, vec_env: VecNormalize,
                 verbose: int = 0):
        super().__init__(verbose)
        self._unfreeze_episodes = unfreeze_episodes
        self._vec_env = vec_env
        self._episode_count = 0
        self._frozen = False
        # Unfreeze at init
        self._vec_env.training = True
        logger.info(f"  VecNormalize UNFROZEN for {unfreeze_episodes} "
                    f"calibration episodes")

    def _on_step(self) -> bool:
        if self._frozen:
            return True
        # Count completed episodes via 'dones'
        dones = self.locals.get("dones", [])
        self._episode_count += sum(dones)
        if self._episode_count >= self._unfreeze_episodes:
            self._vec_env.training = False
            self._frozen = True
            logger.info(f"  VecNormalize re-frozen after "
                        f"{self._episode_count} calibration episodes")
        return True


class ActorFreezeCallback(BaseCallback):
    """Freeze actor during critic calibration, unfreeze when critic stabilizes.

    Prevents "learned helplessness" — actor retreating to zero-residual under
    noisy gradients from an uncalibrated critic (observed in C006 Phase 3).

    Unfreeze when BOTH:
      1. num_timesteps >= min_steps
      2. critic_loss < threshold for n_consecutive checks

    Forced unfreeze at max_steps with warning.

    SB3 note: With requires_grad=False, actor_loss.backward() computes no
    actor gradients — optimizer step is a no-op. Critic trains normally.
    polyak_update() uses .data directly (ignores requires_grad), so the
    target actor drifts toward the frozen actor — correct behavior.
    """

    def __init__(self, min_steps: int = 16000, max_steps: int = 40000,
                 critic_threshold: float = 3.0, n_consecutive: int = 5,
                 checkpoint_dir: str | None = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._threshold = critic_threshold
        self._n_consecutive = n_consecutive
        self._checkpoint_dir = checkpoint_dir
        self._consecutive_below = 0
        self._unfrozen = False
        self._initialized = False
        self._log_interval = 500

    def _freeze_actor(self) -> None:
        """Freeze actor and actor_target with hard assertions."""
        self.model.actor.requires_grad_(False)
        self.model.actor_target.requires_grad_(False)
        assert not any(p.requires_grad for p in self.model.actor.parameters()), \
            "[ACTOR_FREEZE] FATAL: Actor params still require grad after freeze!"
        assert not any(p.requires_grad for p in self.model.actor_target.parameters()), \
            "[ACTOR_FREEZE] FATAL: Actor_target params still require grad after freeze!"
        logger.info(f"[ACTOR_FREEZE] Actor frozen at step={self.num_timesteps} "
                    f"(min={self._min_steps}, max={self._max_steps}, "
                    f"threshold={self._threshold})")

    def _check_unfreeze(self) -> bool:
        """Determine whether the actor should be unfrozen.

        HARD FLOOR (Phase 7): min_steps takes absolute precedence.
        The actor is never unfrozen before min_steps, regardless of
        critic loss or max_steps misconfiguration.
        """
        if self.num_timesteps < self._min_steps:
            return False
        if self._consecutive_below >= self._n_consecutive:
            return True
        if self.num_timesteps >= self._max_steps:
            return True
        return False

    def _on_step(self) -> bool:
        if not self._initialized:
            self._freeze_actor()
            self._initialized = True
        if self._unfrozen:
            return True

        ts = self.num_timesteps
        critic_loss = self.model.logger.name_to_value.get(
            "train/critic_loss", float("inf"))

        # Update consecutive counter (only increments after min_steps)
        if ts >= self._min_steps and critic_loss < self._threshold:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        if ts % self._log_interval == 0:
            logger.info(f"[ACTOR_FREEZE] step={ts}, critic_loss={critic_loss:.3f}, "
                        f"consecutive_below={self._consecutive_below}/{self._n_consecutive}, "
                        f"unfreeze_ready={self._check_unfreeze()}")

        if self._check_unfreeze():
            self.model.actor.requires_grad_(True)
            self.model.actor_target.requires_grad_(True)
            self._unfrozen = True
            reason = ("convergence"
                      if self._consecutive_below >= self._n_consecutive
                      else "max_steps")
            logger.info(f"[ACTOR_FREEZE] Actor UNFROZEN at step={ts}, "
                        f"reason={reason}, critic_loss={critic_loss:.3f}")

            # Phase 5.1: Save checkpoint at unfreeze for post-hoc analysis.
            # Per Batch 3d/3e forensic investigation: removed `or "."` CWD fallback
            # to prevent unit-test pollution of repo-root model_unfreeze_*.zip.
            ckpt_dir = self._checkpoint_dir or self.model.tensorboard_log
            if ckpt_dir is None:
                logger.warning(
                    "[ACTOR_FREEZE] No checkpoint_dir or tensorboard_log set; "
                    "skipping unfreeze checkpoint save to prevent CWD pollution."
                )
            else:
                try:
                    unfreeze_path = os.path.join(ckpt_dir, f"model_unfreeze_{ts}_steps.zip")
                    self.model.save(unfreeze_path)
                    logger.info(f"[ACTOR_FREEZE] Unfreeze checkpoint saved: {unfreeze_path}")
                except Exception as e:
                    logger.warning(f"[ACTOR_FREEZE] Unfreeze checkpoint save failed: {e}")

        return True


class CriticDivergenceAbortCallback(BaseCallback):
    """Abort training if critic loss shows sustained divergence.

    Uses a rolling window mean to prevent false triggers from single-batch
    spikes (Phase 6: spike to 112.32 after stable 2.87). Retains a hard
    ceiling for truly catastrophic events.

    Phase 7 upgrade: replaces single-sample threshold with hysteresis.

    Parameters
    ----------
    threshold: float
        Rolling mean abort threshold (default 100.0).
    hard_ceiling: float
        Abort if ``hard_ceiling_persist`` consecutive losses exceed this (default 500.0).
    hard_ceiling_persist: int
        Consecutive over-ceiling samples required to abort (default 1 = legacy single-sample).
    window_size: int
        Rolling window size for mean computation (default 5).
    warmup_steps: int
        Minimum timesteps before the callback becomes active (default 10000).
    """

    def __init__(self, threshold: float = 500.0, hard_ceiling: float = 500.0,
                 window_size: int = 1000, warmup_steps: int = 10000,
                 verbose: int = 0, hard_ceiling_persist: int = 1):
        # GUARDWIN (2026-08-26): window 5 -> 1000 readings and threshold 100 -> 500. A reading
        # is one train() call (one batch at n_envs=1), so 5 was ~0.25 s; against a bar of 500
        # that fired on single-batch spikes -- three healthy legs in 36 h -- while no leg ever
        # posted an EPISODE-mean critic loss above 256 (analysis/guardprice_20260826T2000Z).
        # 1000 readings is about one episode. The hard-ceiling limb below is unchanged and
        # still catches a true blow-up within `hard_ceiling_persist` readings.
        super().__init__(verbose)
        self._threshold = threshold
        self._hard_ceiling = hard_ceiling
        self._window_size = window_size
        self._warmup_steps = warmup_steps
        self._loss_history: deque[float] = deque(maxlen=window_size)
        # RB-67: the ceiling fires only on a RUN of `hard_ceiling_persist` consecutive samples
        # over it. 1 == the legacy single-sample trigger. Candidate-A leg 3 died on ONE sample
        # of 503.7 with the rolling mean at 251.8 -- a spike, which the docstring says the
        # rolling window exists to survive. Each sample is already a gradient_steps-batch mean.
        self._hard_ceiling_persist = max(1, int(hard_ceiling_persist))
        self._ceiling_run = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self._warmup_steps:
            return True

        critic_loss = self.model.logger.name_to_value.get(
            "train/critic_loss", 0.0)
        self._loss_history.append(critic_loss)

        # Hard ceiling: abort on a RUN of consecutive samples over it (RB-67); a sample under
        # the ceiling resets the run, so spikes spread across a leg never accumulate.
        self._ceiling_run = self._ceiling_run + 1 if critic_loss > self._hard_ceiling else 0
        if self._ceiling_run >= self._hard_ceiling_persist:
            self._write_abort("hard_ceiling", critic_loss)
            return False

        # Rolling mean: sustained divergence
        if (len(self._loss_history) == self._window_size
                and np.mean(self._loss_history) > self._threshold):
            self._write_abort("rolling_mean", critic_loss)
            return False

        return True

    def _write_abort(self, reason: str, critic_loss: float) -> None:
        """Write ABORT_REQUEST.json and log the abort."""
        abort_path = os.path.join(
            self.model.tensorboard_log or ".", "ABORT_REQUEST.json")
        try:
            with open(abort_path, "w") as f:
                json.dump({
                    "reason": f"critic_divergence_{reason}",
                    "critic_loss": float(critic_loss),
                    "threshold": self._threshold,
                    "hard_ceiling": self._hard_ceiling,
                    "hard_ceiling_persist": self._hard_ceiling_persist,
                    "persist_run": self._ceiling_run,
                    "rolling_mean": (float(np.mean(self._loss_history))
                                     if self._loss_history else 0.0),
                    "timestep": self.num_timesteps,
                }, f, indent=2)
        except Exception:
            pass
        logger.error("[ABORT] Critic diverged (%s): loss=%.1f, "
                     "rolling_mean=%.1f. Wrote %s",
                     reason, critic_loss,
                     float(np.mean(self._loss_history))
                     if self._loss_history else 0.0,
                     abort_path)


def current_segment_rows(rows: list) -> list:
    """F-2: restrict diagnostics-CSV rows to the CURRENT run segment.

    Training diagnostics CSVs are opened in append mode on --resume-from, so
    one file can span several training processes; rolling-window guards must
    never window across that boundary (the C006 H12 window-1 contamination).

    Selection rule keyed on the LAST row:
      - last row has a parseable ``run_segment`` (writer is F-2-aware):
        return exactly the rows of that segment;
      - otherwise (legacy file, or legacy rows appended after a rollback):
        return the suffix after the last ``episode``-counter reset — the
        per-process episode counter restarts at 0/1 in a resumed process,
        so a decrease marks the boundary. Rows with unparseable episodes
        are boundary-neutral.
    """
    if not rows:
        return rows

    def _seg(row):
        try:
            return int(float(row.get("run_segment")))
        except (TypeError, ValueError):
            return None

    last_seg = _seg(rows[-1])
    if last_seg is not None:
        return [r for r in rows if _seg(r) == last_seg]
    start, prev = 0, None
    for i, row in enumerate(rows):
        try:
            ep = float(row.get("episode"))
        except (TypeError, ValueError):
            continue
        if prev is not None and ep < prev:
            start = i
        prev = ep
    return rows[start:]


class StructuralCeilingCallback(BaseCallback):
    """Halt training at gate_timestep if a multi-signal consolidation
    failure is detected: agent shows occasional M3b crossings but no
    sustained improvement, despite stable critic and active reward.

    The C3w audit's 2 M3b achievements at eps 15-16 followed by zero
    through ep 49 establish that this 'W2 peak then decay' failure
    pattern is a real risk profile under the new C3w reward landscape.
    A simple 'global zero M3b' or 'L25 zero M3b' halt would not catch
    this; a multi-signal consolidation check is required.

    Halt fires when ALL six conditions hold simultaneously, evaluated
    ONCE when self.num_timesteps >= gate_timestep:

    SANITY (must hold for halt to be meaningful):
      S1. r_dwell_mean L25 > 0.01 -- reward gradient firing
      S2. critic_loss_max L25 < 80 -- critic stable, not diverging

    STUCK SIGNALS (all four must hold):
      T1. L25 M3b count <= 1 -- no consolidation
      T2. DBB L10-tail mean - L10-head mean <= +0.05s -- DBB not rising
      T3. Entries L10-tail mean - L10-head mean >= -2 -- entries not declining
      T4. Dwell-event count (eps with actual_M3b_dwell_s >= 1.0)
          tail (last 10) <= head (first 10) -- frequency not growing

    Where L10-head = first 10 of L25, L10-tail = last 10 of L25
    (middle 5 episodes ignored to give clean head/tail separation).

    Minimum-window guard: T2 and T3 head/tail deltas are sensitive to
    outlier episodes. If the L25 window contains fewer than 20 valid
    episodes (non-NaN values for DBB and entry count), halt evaluation
    is SKIPPED entirely (returns True without firing) and a warning is
    logged. This prevents false halts during the ts=25K-50K ramp-up
    if the run accumulates very long episodes.

    On halt, writes STRUCTURAL_CEILING_DETECTED.txt to log_dir with
    all 6 condition values for forensic analysis. The halt note is
    empirical-only -- it records the observed signature without
    committing to a specific causal hypothesis.

    Parameters
    ----------
    gate_timestep: int
        SB3 num_timesteps at which to evaluate the gate (default 50000).
    window_size: int
        L25 window size (default 25 episodes).
    head_tail_size: int
        L10-head and L10-tail window sizes (default 10 episodes each).
    dbb_growth_threshold: float
        Min DBB tail-vs-head delta required to count as "rising" (default 0.05 s).
    entries_decline_threshold: float
        Min entries tail-vs-head decline (negative) required to count as
        "declining" (default -2.0 entries).
    m3b_consolidation_threshold: int
        Max L25 M3b count tolerated without halt (default 1).
    r_dwell_active_threshold: float
        Min r_dwell_mean L25 required for sanity (default 0.01).
    critic_max_threshold: float
        Max critic_loss_max L25 allowed for sanity (default 80.0).
    metrics_csv_path: str
        Path to per-episode metrics.csv (read at gate fire time).
    opd_csv_path: str
        Path to offpolicy_diagnostics.csv (for critic_loss_max).
    log_dir: str
        Directory to write STRUCTURAL_CEILING_DETECTED.txt.
    """

    def __init__(self, gate_timestep: int = 50000,
                 window_size: int = 25,
                 head_tail_size: int = 10,
                 dbb_growth_threshold: float = 0.05,
                 entries_decline_threshold: float = -2.0,
                 m3b_consolidation_threshold: int = 1,
                 r_dwell_active_threshold: float = 0.01,
                 critic_max_threshold: float = 80.0,
                 metrics_csv_path: Optional[str] = None,
                 opd_csv_path: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self._gate_ts = gate_timestep
        self._L25 = window_size
        self._L10 = head_tail_size
        self._dbb_growth = dbb_growth_threshold
        self._ent_decline = entries_decline_threshold
        self._m3b_consol = m3b_consolidation_threshold
        self._r_dwell_min = r_dwell_active_threshold
        self._critic_max = critic_max_threshold
        self._metrics_path = metrics_csv_path
        self._opd_path = opd_csv_path
        self._log_dir = log_dir
        self._fired = False

    def _on_step(self) -> bool:
        if self._fired or self.num_timesteps < self._gate_ts:
            return True
        self._fired = True

        # Read both CSVs
        if not (self._metrics_path and os.path.exists(self._metrics_path)
                and self._opd_path and os.path.exists(self._opd_path)):
            logger.warning(
                "[STRUCTURAL_CEILING] CSV files missing at ts=%d "
                "(metrics=%s, opd=%s); skipping evaluation",
                self.num_timesteps, self._metrics_path, self._opd_path)
            return True
        try:
            with open(self._metrics_path) as f:
                rows = list(csv.DictReader(f))
            with open(self._opd_path) as f:
                opd_rows = list(csv.DictReader(f))
        except Exception as exc:
            logger.warning(
                "[STRUCTURAL_CEILING] CSV read failed at ts=%d: %s",
                self.num_timesteps, exc)
            return True
        # F-2: never window across a restart boundary
        rows = current_segment_rows(rows)
        opd_rows = current_segment_rows(opd_rows)
        if len(rows) < self._L25 or len(opd_rows) < self._L25:
            logger.warning(
                "[STRUCTURAL_CEILING] Insufficient rows for L25 at ts=%d "
                "(metrics=%d, opd=%d, need>=%d); skipping",
                self.num_timesteps, len(rows), len(opd_rows), self._L25)
            return True

        l25 = rows[-self._L25:]
        l25_opd = opd_rows[-self._L25:]
        head = l25[:self._L10]
        tail = l25[-self._L10:]

        def _f(v, d=0.0):
            try:
                if v in ("", "nan", "NaN", None):
                    return d
                fv = float(v)
                import math
                if math.isnan(fv):
                    return d
                return fv
            except (ValueError, TypeError):
                return d

        def _bool(v):
            return str(v).strip() in ("True", "true", "1")

        # Minimum-window guard for T2 and T3.
        # T2/T3 head/tail deltas are sensitive to outlier episodes. If the
        # L25 window has fewer than MIN_VALID episodes with non-NaN values
        # for DBB and entry count, do NOT fire halt -- log warning and skip.
        MIN_VALID = 20
        dbb_valid_count = sum(
            1 for r in l25
            if _f(r.get("m3b_mean_dwell_before_break_s",
                        float("nan")), -1.0) >= 0
        )
        entry_valid_count = sum(
            1 for r in l25
            if _f(r.get("m3b_entry_count", float("nan")), -1.0) >= 0
        )
        if dbb_valid_count < MIN_VALID or entry_valid_count < MIN_VALID:
            logger.warning(
                "[STRUCTURAL_CEILING] Insufficient valid data for T2/T3 "
                "at ts=%d (dbb_valid=%d, entry_valid=%d, need >= %d) -- "
                "skip evaluation",
                self.num_timesteps, dbb_valid_count, entry_valid_count,
                MIN_VALID,
            )
            return True

        # SANITY S1: r_dwell active in L25
        # Use the per-episode r_dwell_mean column directly (the mean field
        # explicitly named in the gate definition). Fall back to averaging
        # the per-episode r_dwell totals if the mean column is missing.
        rdm_vals = [_f(r.get("r_dwell_mean", float("nan"))) for r in l25]
        if any(v != 0.0 for v in rdm_vals):
            r_dwell_mean_metric = sum(rdm_vals) / len(rdm_vals)
        else:
            r_dwell_mean_metric = (
                sum(_f(r.get("r_dwell", 0)) for r in l25) / len(l25)
            )
        S1 = r_dwell_mean_metric > self._r_dwell_min

        # SANITY S2: critic stable
        # Use critic_loss_max column (per-episode max) when valid, NaN-safe.
        # If the column is unpopulated/missing, fall back to per-episode mean.
        clm_vals = [_f(r.get("critic_loss_max", float("nan"))) for r in l25_opd]
        clm_valid = [v for v in clm_vals if v > 0.0]
        if clm_valid:
            critic_max_metric = max(clm_valid)
        else:
            critic_max_metric = max(
                _f(r.get("critic_loss", 0)) for r in l25_opd
            )
        S2 = critic_max_metric < self._critic_max

        # STUCK T1: L25 M3b count
        m3b_count = sum(
            1 for r in l25
            if _bool(r.get("actual_M3b_achieved", ""))
        )
        T1 = m3b_count <= self._m3b_consol

        # STUCK T2: DBB trend
        dbb_head = sum(
            _f(r.get("m3b_mean_dwell_before_break_s", 0)) for r in head
        ) / len(head)
        dbb_tail = sum(
            _f(r.get("m3b_mean_dwell_before_break_s", 0)) for r in tail
        ) / len(tail)
        dbb_delta = dbb_tail - dbb_head
        T2 = dbb_delta <= self._dbb_growth

        # STUCK T3: Entry count trend
        ent_head = sum(
            _f(r.get("m3b_entry_count", 0)) for r in head
        ) / len(head)
        ent_tail = sum(
            _f(r.get("m3b_entry_count", 0)) for r in tail
        ) / len(tail)
        ent_delta = ent_tail - ent_head
        T3 = ent_delta >= self._ent_decline

        # STUCK T4: Dwell event frequency (eps with actual_M3b_dwell_s >= 1.0)
        head_events = sum(
            1 for r in head
            if _f(r.get("actual_M3b_dwell_s", 0)) >= 1.0
        )
        tail_events = sum(
            1 for r in tail
            if _f(r.get("actual_M3b_dwell_s", 0)) >= 1.0
        )
        T4 = tail_events <= head_events

        all_conditions = [S1, S2, T1, T2, T3, T4]
        if all(all_conditions):
            self._write_halt(
                r_dwell_mean_metric, critic_max_metric, m3b_count,
                dbb_head, dbb_tail, dbb_delta,
                ent_head, ent_tail, ent_delta,
                head_events, tail_events,
            )
            return False  # HALT
        else:
            logger.info(
                "[STRUCTURAL_CEILING] Gate at ts=%d: "
                "S1(r_dwell_mean>%.3f? %s, val=%.4f) "
                "S2(critic_max<%.0f? %s, val=%.2f) "
                "T1(M3b<=%d? %s, count=%d) "
                "T2(DBB delta<=%.2f? %s, %.3f) "
                "T3(ent delta>=%.0f? %s, %.1f) "
                "T4(events tail<=head? %s, %d/%d). "
                "Continuing training.",
                self.num_timesteps,
                self._r_dwell_min, S1, r_dwell_mean_metric,
                self._critic_max, S2, critic_max_metric,
                self._m3b_consol, T1, m3b_count,
                self._dbb_growth, T2, dbb_delta,
                self._ent_decline, T3, ent_delta,
                T4, tail_events, head_events,
            )
        return True

    def _write_halt(self, *args) -> None:
        (r_dwell_mean_metric, critic_max_metric, m3b_count,
         dbb_head, dbb_tail, dbb_delta,
         ent_head, ent_tail, ent_delta,
         head_events, tail_events) = args
        halt_path = os.path.join(
            self._log_dir or ".", "STRUCTURAL_CEILING_DETECTED.txt"
        )
        try:
            with open(halt_path, "w") as f:
                f.write(
                    f"Multi-signal consolidation failure detected at "
                    f"ts={self.num_timesteps}\n\n"
                    f"All 6 halt conditions met simultaneously:\n\n"
                    f"SANITY:\n"
                    f"  S1 r_dwell_mean L25 > {self._r_dwell_min}: "
                    f"{r_dwell_mean_metric:.4f} (firing)\n"
                    f"  S2 critic_loss_max L25 < {self._critic_max}: "
                    f"{critic_max_metric:.2f} (stable)\n\n"
                    f"STUCK SIGNALS:\n"
                    f"  T1 L25 M3b count <= {self._m3b_consol}: "
                    f"{m3b_count} (no consolidation)\n"
                    f"  T2 DBB tail-head delta <= +{self._dbb_growth}s: "
                    f"head={dbb_head:.3f}, tail={dbb_tail:.3f}, "
                    f"delta={dbb_delta:+.3f} (not rising)\n"
                    f"  T3 entries tail-head delta >= {self._ent_decline}: "
                    f"head={ent_head:.1f}, tail={ent_tail:.1f}, "
                    f"delta={ent_delta:+.1f} (not declining)\n"
                    f"  T4 dwell-events>=1.0s tail <= head: "
                    f"head={head_events}, tail={tail_events} "
                    f"(frequency not growing)\n\n"
                    f"Empirical observation: the agent has stabilized into "
                    f"a non-improving pattern despite active reward gradient "
                    f"and stable critic. Further training is unlikely to "
                    f"produce consolidation under the current configuration.\n"
                )
        except Exception:
            pass
        logger.error(
            "[STRUCTURAL_CEILING] HALT at ts=%d: "
            "M3b_count=%d, dbb_delta=%+.3f. See %s",
            self.num_timesteps, m3b_count, dbb_delta, halt_path,
        )


class H12ConjunctionCallback(BaseCallback):
    """H12 conjunction halt: critic grad-norm monotonicity + loss gate.

    Fires halt when BOTH conditions co-occur at a 25-episode window boundary:
      1. critic_grad_norm_mean strictly increasing (w1 < w2 < w3) across 3
         consecutive non-overlapping 25-episode windows, evaluated only when
         num_timesteps >= gate_timestep.
      2. critic_loss rolling mean over last 25 episodes (L25) > threshold.

    Monotonicity is strict (w1 < w2 < w3). Equal consecutive window means
    do NOT trigger. Conjoined with critic_loss L25 rolling mean > threshold.

    Note: Originally used critic_loss_max, but that column has ~40% NaN in
    multi-env training (episodes sharing a timestep batch get NaN because no
    gradient update occurs between them). Switched to critic_loss (0% NaN,
    ~1.0-1.1x magnitude of critic_loss_max). See L9.3 resilience autopsy.
    """

    def __init__(self, gate_timestep=10000, window_size=25, n_windows=3,
                 loss_threshold=40.0, opd_csv_path="", log_dir="",
                 checkpoint_dir="", min_rise_frac=0.0, verbose=0):
        super().__init__(verbose=verbose)
        self._gate_ts = int(gate_timestep)
        self._window_size = int(window_size)
        self._n_windows = int(n_windows)
        self._loss_threshold = float(loss_threshold)
        # F-3 (T-198/T-199): noise floor on the monotone rise — strict `<`
        # alone is satisfied by the curriculum wind-ladder slope (max healthy
        # rise 3.27% vs the 3.3% spread that FALSE-halted WS2). 0.0 preserves
        # the legacy conjunction; train.py wires config.H12_MIN_RISE_FRAC.
        self._min_rise_frac = float(min_rise_frac)
        self._opd_csv = opd_csv_path
        self._log_dir = log_dir
        # E-2: enables the last-clean-checkpoint hint in the halt record
        self._checkpoint_dir = checkpoint_dir
        self._fired = False
        self._last_check_ep = 0
        import logging as _logging
        _logging.getLogger(__name__).info(
            "H12ConjunctionCallback registered: gate_ts=%d, "
            "window_size=%d, n_windows=%d, loss_threshold=%.1f, "
            "min_rise_frac=%.3f, monotonicity=strict",
            gate_timestep, window_size, n_windows, loss_threshold,
            self._min_rise_frac)

    def _on_step(self) -> bool:
        if self._fired or self.num_timesteps < self._gate_ts:
            return True
        rows = self._read_opd()
        if not rows:
            return True
        n_rows = len(rows)
        if n_rows <= self._last_check_ep:
            return True
        self._last_check_ep = n_rows
        # F-2: window only within the current run segment — a resumed
        # process must accumulate its own 75 clean rows, never straddle
        # the pre-crash tail (the C006 window-1 contamination).
        rows = current_segment_rows(rows)
        min_rows = self._n_windows * self._window_size
        if len(rows) < min_rows:
            return True
        tail = rows[-min_rows:]
        grad_norms = [self._safe_f(r.get('critic_grad_norm_mean')) for r in tail]
        if any(v != v for v in grad_norms):
            return True
        window_means = []
        for w_idx in range(self._n_windows):
            start = w_idx * self._window_size
            end = start + self._window_size
            window_means.append(
                float(np.mean(grad_norms[start:end])))
        monotonic = all(
            window_means[j] < window_means[j + 1]
            for j in range(len(window_means) - 1))
        if not monotonic:
            return True
        # F-3: the rise must clear the calibrated noise floor, not merely be
        # monotone — a stage promotion inside the window (W8 ran fewer eps
        # than one window when H12 false-halted WS2) shifts the level by the
        # ladder slope, never by a divergence-scale factor.
        if window_means[0] > 0.0:
            rise_frac = window_means[-1] / window_means[0] - 1.0
        else:
            rise_frac = float("inf")
        if rise_frac < self._min_rise_frac:
            return True
        loss_rows = rows[-self._window_size:]
        loss_vals = [self._safe_f(r.get('critic_loss')) for r in loss_rows]
        if any(v != v for v in loss_vals):
            return True
        loss_mean = float(np.mean(loss_vals))
        if loss_mean <= self._loss_threshold:
            return True
        self._fired = True
        self._write_halt(window_means, loss_mean,
                         window1_first_ts=self._safe_f(tail[0].get("timestep")),
                         run_segment=str(tail[-1].get("run_segment", "")))
        return False

    @property
    def fired(self) -> bool:
        """E-2: public halt flag for the post-learn() termination classifier."""
        return self._fired

    def _read_opd(self):
        if not self._opd_csv or not os.path.exists(self._opd_csv):
            return []
        rows = []
        with open(self._opd_csv) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def _safe_f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('nan')

    def _write_halt(self, window_means, loss_mean,
                    window1_first_ts=float("nan"), run_segment=""):
        # E-2: append-mode (a re-fire after resume must not overwrite the
        # prior halt record) + rollback evidence: run_segment of the fired
        # window and a last-clean-checkpoint HINT (newest periodic checkpoint
        # strictly older than window 1 — human-verified before any resume).
        import datetime
        halt_path = os.path.join(self._log_dir, "H12_FIRED.txt")
        with open(halt_path, "a") as fh:
            fh.write(
                f"[{datetime.datetime.utcnow().isoformat()}Z] "
                f"H12 CONJUNCTION HALT\n")
            fh.write(f"num_timesteps={self.num_timesteps}\n")
            fh.write(f"gate_timestep={self._gate_ts}\n")
            fh.write(f"window_size={self._window_size}\n")
            fh.write(f"n_windows={self._n_windows}\n")
            fh.write(f"loss_threshold={self._loss_threshold}\n")
            fh.write(f"window_means={window_means}\n")
            fh.write(f"monotonicity=strict (w1 < w2 < w3)\n")
            fh.write(f"min_rise_frac={self._min_rise_frac}\n")
            fh.write(f"rise_frac_observed="
                     f"{window_means[-1] / window_means[0] - 1.0:.6f}\n")
            fh.write(f"critic_loss_L25_mean={loss_mean:.6f}\n")
            fh.write(f"both_conditions_met=True\n")
            fh.write(f"run_segment={run_segment}\n")
            fh.write(f"window1_first_timestep={window1_first_ts}\n")
            fh.write(f"last_clean_checkpoint_hint="
                     f"{self._last_clean_checkpoint_hint(window1_first_ts)}\n")
            fh.write("---\n")
        logger.error(
            "[H12_CONJUNCTION] HALT at ts=%d: window_means=%s, "
            "L25_loss=%.4f > %.1f. See %s",
            self.num_timesteps, window_means, loss_mean,
            self._loss_threshold, halt_path)

    def _last_clean_checkpoint_hint(self, before_ts):
        return last_clean_checkpoint_hint(self._checkpoint_dir, before_ts)


def last_clean_checkpoint_hint(checkpoint_dir, before_ts):
    """Newest model_<N>_steps.zip with N strictly before before_ts (E-2/E-4).

    Hint only — the divergence-halt rollback procedure requires the operator
    to verify the candidate before any resume. Diagnostics timestep and
    checkpoint N share the same counter within a run segment.
    """
    if (not checkpoint_dir or not os.path.isdir(checkpoint_dir)
            or before_ts != before_ts):
        return "unavailable"
    best = None
    for name in os.listdir(checkpoint_dir):
        m = re.match(r"model_(\d+)_steps\.zip$", name)
        if m:
            n = int(m.group(1))
            if n < before_ts and (best is None or n > best):
                best = n
    return f"model_{best}_steps.zip" if best is not None else "none_found"


class BehavioralEarlyStopCallback(BaseCallback):
    """E-4: rolling-window behavioral early stop (negative-learning terminator).

    Watches the POLICY where H12 watches the CRITIC: the online M3b success
    rate over non-overlapping `window_episodes` windows. The rate is a
    labeled ctrl-side proxy (training episodes include exploration noise);
    this callback has STOP authority ONLY — it never certifies or promotes,
    gates remain the sole certifier (M3b framing directive).

    Rule: R* = best window rate so far. Once num_timesteps >=
    `min_timesteps`, `patience_windows` CONSECUTIVE windows with
    rate <= R* * (1 - delta_frac) (and R* > 0) => append an evidence block
    to EARLY_STOP_FIRED.txt and return False from _on_step. No automatic
    resume — per docs/DIVERGENCE_HALT_PROCEDURE.md.

    Restart-segment note: all state is in-memory, so windows can never span
    a restart (each process re-baselines its own R* — conservative).
    Standalone rather than an extension of CheckpointGatedRollbackCallback:
    that class's restore path is dead under FD_CURRICULUM_V3 and its
    enablement is opt-in, while E-4 is standard equipment; the shared
    "metric" is one line (window mean over info['actual_M3b_achieved']).
    """

    #: supported metrics -> info-dict key (both emitted by
    #: env_formation_sitl; no env change involved). achieved_rate SATURATES
    #: at ~1.0 on C006 (back-test 2026-07-29) — mean_dwell carries the
    #: dynamic range. Election pending.
    _METRIC_KEYS = {"achieved_rate": "actual_M3b_achieved",
                    "mean_dwell": "actual_M3b_dwell_s"}

    def __init__(self, window_episodes=25, delta_frac=0.5, patience_windows=3,
                 min_timesteps=15000, metric="achieved_rate", log_dir="",
                 checkpoint_dir="", floor_windows=0, floor_eps=0.2, verbose=0):
        super().__init__(verbose=verbose)
        if metric not in self._METRIC_KEYS:
            raise ValueError(f"unknown early-stop metric: {metric!r}")
        self._window_episodes = int(window_episodes)
        self._delta_frac = float(delta_frac)
        self._patience = int(patience_windows)
        self._min_ts = int(min_timesteps)
        self._metric = metric
        self._log_dir = log_dir
        self._checkpoint_dir = checkpoint_dir
        # W-D floor mode: never-achieved detector (0 = disabled)
        self._floor_windows = int(floor_windows)
        self._floor_eps = float(floor_eps)
        self._floor_streak = 0
        self._window: List[float] = []
        self._best_rate = 0.0
        self._streak_rates: List[float] = []
        self._streak_first_ts: Optional[int] = None
        self._fired = False
        logger.info(
            "BehavioralEarlyStopCallback registered: metric=%s, window=%d "
            "eps, delta_frac=%.2f, patience=%d windows, min_timesteps=%d",
            self._metric, self._window_episodes, self._delta_frac,
            self._patience, self._min_ts)

    @property
    def fired(self) -> bool:
        return self._fired

    def _on_step(self) -> bool:
        if self._fired:
            return True
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i in range(len(dones)):
            if not dones[i]:
                continue
            info = infos[i] if i < len(infos) else {}
            raw = info.get(self._METRIC_KEYS[self._metric], 0.0)
            self._window.append(float(raw if raw == raw else 0.0))
            if len(self._window) < self._window_episodes:
                continue
            if not self._close_window():
                return False
        return True

    def _close_window(self) -> bool:
        """Evaluate one completed window; False => halt training."""
        rate = sum(self._window) / len(self._window)
        self._window = []
        # W-D floor mode: the regression rule below needs best>0; a run that
        # NEVER achieves has no peak to regress from (measured blind spot,
        # W0 2026-07-29). Floor: consecutive sub-eps windows while best is
        # also sub-eps => halt. A single decent window (best >= eps) kills
        # floor mode for the rest of the run — regression takes over.
        if self._floor_windows > 0 and self._best_rate < self._floor_eps:
            if rate < self._floor_eps:
                self._floor_streak += 1
                logger.warning(
                    "[EARLY_STOP] floor window %d/%d: rate=%.3f < eps=%.3f "
                    "(best=%.3f) at ts=%d", self._floor_streak,
                    self._floor_windows, rate, self._floor_eps,
                    self._best_rate, self.num_timesteps)
                if (self._floor_streak >= self._floor_windows
                        and self.num_timesteps >= self._min_ts):
                    self._fired = True
                    self._write_halt(reason="floor_never_achieved")
                    return False
            else:
                self._floor_streak = 0
        degraded = (self._best_rate > 0
                    and rate <= self._best_rate * (1.0 - self._delta_frac))
        if rate > self._best_rate:
            self._best_rate = rate
        if not degraded:
            self._streak_rates = []
            self._streak_first_ts = None
            return True
        if self._streak_first_ts is None:
            self._streak_first_ts = int(self.num_timesteps)
        self._streak_rates.append(rate)
        logger.warning(
            "[EARLY_STOP] degraded window %d/%d: rate=%.3f vs best=%.3f "
            "(bar %.3f) at ts=%d",
            len(self._streak_rates), self._patience, rate, self._best_rate,
            self._best_rate * (1.0 - self._delta_frac), self.num_timesteps)
        if (len(self._streak_rates) >= self._patience
                and self.num_timesteps >= self._min_ts):
            self._fired = True
            self._write_halt()
            return False
        return True

    def _write_halt(self, reason: str = "regression_from_peak") -> None:
        import datetime
        halt_path = os.path.join(self._log_dir, "EARLY_STOP_FIRED.txt")
        with open(halt_path, "a") as fh:
            fh.write(
                f"[{datetime.datetime.utcnow().isoformat()}Z] "
                f"BEHAVIORAL EARLY STOP\n")
            fh.write(f"reason={reason}\n")
            if reason == "floor_never_achieved":
                fh.write(f"floor_windows={self._floor_windows}\n")
                fh.write(f"floor_eps={self._floor_eps}\n")
                fh.write(f"floor_streak={self._floor_streak}\n")
            fh.write(f"num_timesteps={self.num_timesteps}\n")
            fh.write(f"window_episodes={self._window_episodes}\n")
            fh.write(f"delta_frac={self._delta_frac}\n")
            fh.write(f"patience_windows={self._patience}\n")
            fh.write(f"min_timesteps={self._min_ts}\n")
            fh.write(f"best_rate={self._best_rate:.6f}\n")
            fh.write(f"degraded_window_rates={self._streak_rates}\n")
            fh.write(f"streak_first_timestep={self._streak_first_ts}\n")
            fh.write(f"metric={self._metric} "
                     f"[{self._METRIC_KEYS[self._metric]}] (ctrl-side "
                     f"PROXY; stop authority only — gates certify)\n")
            fh.write(f"last_clean_checkpoint_hint="
                     f"{last_clean_checkpoint_hint(self._checkpoint_dir, float(self._streak_first_ts or 0))}\n")
            fh.write("---\n")
        if reason == "floor_never_achieved":
            logger.error(
                "[EARLY_STOP] HALT at ts=%d: %d consecutive windows < "
                "floor eps %.3f with best %.3f never achieved. See %s",
                self.num_timesteps, self._floor_streak, self._floor_eps,
                self._best_rate, halt_path)
        else:
            logger.error(
                "[EARLY_STOP] HALT at ts=%d: %d consecutive windows <= "
                "%.1f%% of best %.3f. See %s",
                self.num_timesteps, len(self._streak_rates),
                (1.0 - self._delta_frac) * 100, self._best_rate, halt_path)


class NoiseAnnealCallback(BaseCallback):
    """Linearly anneal TD3 exploration noise sigma over training.

    Reduces exploration noise from initial_sigma to final_sigma over
    anneal_steps timesteps. After anneal_steps, sigma stays at final_sigma.

    C006 Phase 7: Start with high exploration (0.25) to escape the
    zero-residual attractor, then reduce to 0.15 for fine-grained docking.
    """

    def __init__(self, initial_sigma: float, final_sigma: float,
                 anneal_steps: int, resume_offset_steps: int = 0,
                 verbose: int = 0):
        super().__init__(verbose)
        self._initial_sigma = initial_sigma
        self._final_sigma = final_sigma
        self._anneal_steps = anneal_steps
        # H4-Alpha: when >0, progress is computed relative to this offset so
        # that a resumed run re-bases the anneal schedule to its own first
        # step rather than inheriting the parent checkpoint's num_timesteps.
        self._resume_offset_steps = resume_offset_steps

    def _on_step(self) -> bool:
        if self._anneal_steps <= 0:
            progress = 1.0
        elif self._resume_offset_steps > 0:
            effective = max(0.0,
                            self.num_timesteps - self._resume_offset_steps)
            progress = min(1.0, effective / self._anneal_steps)
        else:
            progress = min(1.0, self.num_timesteps / self._anneal_steps)
        current_sigma = (self._initial_sigma
                         + progress * (self._final_sigma - self._initial_sigma))
        if (hasattr(self.model, "action_noise")
                and self.model.action_noise is not None):
            noise = self.model.action_noise
            from stable_baselines3.common.noise import VectorizedActionNoise
            if isinstance(noise, VectorizedActionNoise):
                for n in noise.noises:
                    n._sigma = np.full(n._sigma.shape, current_sigma)
            else:
                noise._sigma = np.full(noise._sigma.shape, current_sigma)
        return True


class PassthroughMonitorCallback(BaseCallback):
    """Monitor passthrough rate and emit alert if sustained above threshold.

    Does NOT abort — only writes an alert JSON for post-campaign analysis.
    Passthrough is detected when the per-episode mean action L2 norm is
    below ``action_l2_threshold``.

    C006 Phase 7: surveillance gate for the anti-passthrough interventions.
    """

    def __init__(self, threshold: float = 0.90,
                 consecutive_episodes: int = 50,
                 action_l2_threshold: float = 0.05,
                 alert_path: str | None = None, verbose: int = 0):
        super().__init__(verbose)
        self._threshold = threshold
        self._consecutive_episodes = consecutive_episodes
        self._action_l2_threshold = action_l2_threshold
        self._alert_path = alert_path
        self._consecutive_count = 0
        self._alerted = False
        self._episode_count = 0
        self._passthrough_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if dones is None or infos is None:
            return True

        for i, done in enumerate(dones):
            if not done:
                continue
            self._episode_count += 1
            info = infos[i] if i < len(infos) else {}
            # Check if this episode was passthrough
            action_l2 = info.get("mean_action_l2", None)
            is_passthrough = (action_l2 is not None
                              and action_l2 < self._action_l2_threshold)
            if is_passthrough:
                self._passthrough_episodes += 1
                self._consecutive_count += 1
            else:
                self._consecutive_count = 0

            # Check alert condition
            if (not self._alerted
                    and self._consecutive_count >= self._consecutive_episodes):
                self._alerted = True
                rate = (self._passthrough_episodes / self._episode_count
                        if self._episode_count > 0 else 0.0)
                logger.warning(
                    "[PASSTHROUGH_MONITOR] ALERT: %d consecutive passthrough "
                    "episodes (rate=%.1f%%). Threshold=%d.",
                    self._consecutive_count, rate * 100,
                    self._consecutive_episodes)
                if self._alert_path:
                    try:
                        with open(self._alert_path, "w") as f:
                            json.dump({
                                "alert": "sustained_passthrough",
                                "consecutive_count": self._consecutive_count,
                                "total_episodes": self._episode_count,
                                "passthrough_rate": rate,
                                "timestep": self.num_timesteps,
                            }, f, indent=2)
                    except Exception:
                        pass
        return True


class ConfigVerificationCallback(BaseCallback):
    """One-shot config verification on the first training step.

    Writes a JSON snapshot of expected vs actual parameters, then compares
    every entry. Hard-aborts via ``sys.exit(1)`` on ANY mismatch.

    Addresses the systemic config propagation failures that affected
    C006 Phases 5, 5.1, and 6 — where CLI flags were set correctly but
    parameters never reached the model or callbacks at runtime.

    Parameters
    ----------
    config_snapshot_path: str
        Path to write the JSON audit snapshot.
    expected_config: dict
        Parameter name → expected value mapping. Keys may include both
        model-level params (gamma, learning_rate, tau, policy_delay,
        buffer_size, model_class) and callback-level params
        (actor_freeze_min_steps, critic_abort_threshold, etc.).
    callback_refs: dict or None
        Direct references to sibling callbacks for attribute verification.
        Keys: ``"actor_freeze"`` → ActorFreezeCallback instance,
              ``"critic_divergence"`` → CriticDivergenceAbortCallback instance.
    verbose: int
        Verbosity level.
    """

    def __init__(self, config_snapshot_path: str, expected_config: dict,
                 callback_refs: dict | None = None, verbose: int = 0):
        super().__init__(verbose)
        self._config_snapshot_path = config_snapshot_path
        self._expected_config = expected_config
        self._callback_refs = callback_refs or {}
        self._verified = False

    # ------------------------------------------------------------------
    # Callback-level parameter → (callback_refs key, private attr name)
    # ------------------------------------------------------------------
    _CB_CHECKS = {
        "critic_abort_threshold":       ("critic_divergence", "_threshold"),
        "critic_hard_ceiling":          ("critic_divergence", "_hard_ceiling"),
        "critic_hard_ceiling_persist":  ("critic_divergence", "_hard_ceiling_persist"),
        "actor_freeze_min_steps":       ("actor_freeze", "_min_steps"),
        "actor_freeze_max_steps":       ("actor_freeze", "_max_steps"),
        "actor_freeze_critic_threshold": ("actor_freeze", "_threshold"),
    }

    def _on_step(self) -> bool:
        if self._verified:
            return True
        self._verified = True

        mismatches: list[tuple[str, object, object]] = []

        # --- Model-level parameters ---
        model_checks: dict[str, object] = {
            "gamma": getattr(self.model, "gamma", None),
            "learning_rate": getattr(self.model, "learning_rate", None),
            "tau": getattr(self.model, "tau", None),
            "policy_delay": getattr(self.model, "policy_delay", None),
            "buffer_size": (
                self.model.replay_buffer.buffer_size
                if hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
                else None
            ),
            "model_class": type(self.model).__name__,
        }

        for key, actual in model_checks.items():
            if key not in self._expected_config:
                continue
            expected = self._expected_config[key]
            if isinstance(expected, float) and isinstance(actual, (int, float)):
                if abs(float(actual) - expected) > 1e-9:
                    mismatches.append((key, expected, actual))
            elif actual != expected:
                mismatches.append((key, expected, actual))

        # --- Callback-level parameters ---
        cb_actuals: dict[str, object] = {}
        for key, (ref_name, attr_name) in self._CB_CHECKS.items():
            if key not in self._expected_config:
                continue
            expected = self._expected_config[key]
            cb_ref = self._callback_refs.get(ref_name)
            if cb_ref is None:
                mismatches.append((key, expected, "CALLBACK_NOT_FOUND"))
                cb_actuals[key] = "CALLBACK_NOT_FOUND"
                continue
            actual = getattr(cb_ref, attr_name, "ATTR_NOT_FOUND")
            cb_actuals[key] = actual
            if isinstance(expected, float) and isinstance(actual, (int, float)):
                if abs(float(actual) - expected) > 1e-9:
                    mismatches.append((key, expected, actual))
            elif actual != expected:
                mismatches.append((key, expected, actual))

        # --- Config-module parameters (extended to intervention keys) ---
        from drl import config as _runtime_cfg
        _CONFIG_CHECKS = {
            "w_barrier": "W_BARRIER",
            "w_barrier_outside": "W_BARRIER_OUTSIDE",
            "cbf_violation_scale": "CBF_VIOLATION_SCALE",
            "barrier_proximity_gate": "BARRIER_PROXIMITY_GATE",
            "barrier_velocity_ema": "BARRIER_VELOCITY_EMA",
            "forward_bias_ms": "FORWARD_BIAS_MS",
            "bias_smooth_alpha": "BIAS_SMOOTH_ALPHA",
            "ema_alpha_speed": "EMA_ALPHA_SPEED",
            "speed_slew_rate": "SPEED_SLEW_RATE_MPS2",
            "caps_actor_lambda": "CAPS_ACTOR_LAMBDA",
            "exit_behind_multiplier": "EXIT_BEHIND_MULTIPLIER",
            "velocity_gate_k": "VELOCITY_GATE_K",
            "w_hold_fwd": "W_HOLD_FWD",
            "exit_dwell_threshold": "EXIT_DWELL_THRESHOLD_S",
            "exit_dwell_lower_gate": "EXIT_DWELL_LOWER_GATE_S",
            "overspeed_corridor_outer": "OVERSPEED_CORRIDOR_OUTER",
            "overspeed_corridor_inner": "OVERSPEED_CORRIDOR_INNER",
            "residual_mode": "RESIDUAL_MODE",
            "force_zero_residual": "FORCE_ZERO_RESIDUAL",
        }
        config_actuals: dict[str, object] = {}
        for snap_key, attr_name in _CONFIG_CHECKS.items():
            if snap_key not in self._expected_config:
                continue
            expected = self._expected_config[snap_key]
            actual = getattr(_runtime_cfg, attr_name, "ATTR_NOT_FOUND")
            config_actuals[snap_key] = actual
            if isinstance(expected, float) and isinstance(actual, (int, float)):
                if abs(float(actual) - expected) > 1e-9:
                    mismatches.append((snap_key, expected, actual))
            elif actual != expected:
                mismatches.append((snap_key, expected, actual))

        # --- Worker-side config propagation check ---
        worker_actuals: dict[str, dict] = {}
        if hasattr(self.training_env, 'env_method'):
            try:
                worker_configs = self.training_env.env_method(
                    "get_runtime_config_subset")
                for env_id, wc in enumerate(worker_configs):
                    worker_actuals[f"env_{env_id}"] = wc
                    for key, worker_val in wc.items():
                        if key in self._expected_config:
                            expected = self._expected_config[key]
                            if (isinstance(expected, float)
                                    and isinstance(worker_val, (int, float))):
                                if abs(float(worker_val) - expected) > 1e-9:
                                    mismatches.append(
                                        (f"worker_{env_id}/{key}",
                                         expected, worker_val))
                            elif worker_val != expected:
                                mismatches.append(
                                    (f"worker_{env_id}/{key}",
                                     expected, worker_val))
                logger.info(
                    "[CONFIG_VERIFY] Worker-side propagation: %d workers checked",
                    len(worker_configs))
            except Exception as exc:
                logger.warning(
                    "[CONFIG_VERIFY] Worker echo failed: %s", exc)

        # --- Write JSON snapshot (audit trail) ---
        snapshot = {
            "expected": self._expected_config,
            "actual_model": {k: v for k, v in model_checks.items()
                            if k in self._expected_config},
            "actual_callbacks": cb_actuals,
            "actual_config": config_actuals,
            "actual_workers": worker_actuals,
            "mismatches": [
                {"param": p, "expected": str(e), "actual": str(a)}
                for p, e, a in mismatches
            ],
            "verified": len(mismatches) == 0,
        }
        try:
            with open(self._config_snapshot_path, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("[CONFIG_VERIFY] Failed to write snapshot: %s", exc)

        # --- Verdict ---
        if mismatches:
            logger.error(
                "[CONFIG_VERIFY] MISMATCH DETECTED — aborting training")
            for param, expected, actual in mismatches:
                logger.error("  %s: expected=%s, actual=%s",
                             param, expected, actual)
            sys.exit(1)

        logger.info(
            "[CONFIG_VERIFY] All parameters verified OK "
            "(%d model + %d callback + %d config + %d worker checks)",
            sum(1 for k in model_checks if k in self._expected_config),
            len(cb_actuals),
            len(config_actuals),
            len(worker_actuals),
        )
        return True


class TrainingHealthCallback(BaseCallback):
    """Multi-trigger training health monitor with emergency checkpoint (C2).

    Fires every ``check_interval_episodes`` completed episodes and evaluates
    three abort triggers:

      T1: >= ``collision_threshold`` SAFETY_COLLISION terminal events in the
          most recent ``check_interval_episodes`` episode window.
      T2: Critic loss L10 > ``critic_loss_threshold``.
      T3: Q1 mean L10 < ``q1_mean_threshold``.

    On trigger:
      1. Emergency atomic checkpoint (model + replay buffer + VecNormalize).
      2. Sentinel file written to ``/tmp/flightdock_health_alert.json``.
      3. CRITICAL log message.

    On clear (no trigger fires at interval):
      INFO log with brief status: M3b dwell L20, critic loss L10, Q1 mean L10,
      safety collisions L20.

    Parameters
    ----------
    check_interval_episodes: int
        Check health every this many completed episodes (default 20).
    collision_threshold: int
        T1 fires if >= this many SAFETY_COLLISION events in the window.
    critic_loss_threshold: float
        T2 fires if critic loss L10 exceeds this.
    q1_mean_threshold: float
        T3 fires if Q1 mean L10 drops below this.
    checkpoint_dir: str
        Directory for emergency checkpoint files.
    verbose: int
        Verbosity level.
    """

    _SENTINEL_PATH = "/tmp/flightdock_health_alert.json"

    def __init__(
        self,
        check_interval_episodes: int = 20,
        collision_threshold: int = 5,
        critic_loss_threshold: float = 80.0,
        q1_mean_threshold: float = -5.0,
        checkpoint_dir: str = "checkpoints",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._check_interval = check_interval_episodes
        self._collision_threshold = collision_threshold
        self._critic_loss_threshold = critic_loss_threshold
        self._q1_mean_threshold = q1_mean_threshold
        self._checkpoint_dir = checkpoint_dir

        self._episode_count = 0
        # Rolling windows (bounded deques)
        self._collision_history: Deque[int] = deque(maxlen=check_interval_episodes)
        self._critic_loss_history: Deque[float] = deque(maxlen=10)
        self._q1_mean_history: Deque[float] = deque(maxlen=10)
        self._m3b_dwell_history: Deque[float] = deque(maxlen=20)
        self._q_consecutive_decline_threshold = 5

    # ------------------------------------------------------------------
    # Core callback
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            self._episode_count += 1
            info = infos[i] if i < len(infos) else {}

            # Track SAFETY_COLLISION events
            terminal_type = info.get("terminal_type", "")
            self._collision_history.append(
                1 if terminal_type == "SAFETY_COLLISION" else 0
            )

            # Track M3b dwell for status reporting
            m3b_dwell = info.get("m3b_dwell_s", 0.0)
            self._m3b_dwell_history.append(float(m3b_dwell))

            # Interval check
            if self._episode_count % self._check_interval == 0:
                self._run_health_check()

        return True

    # ------------------------------------------------------------------
    # Health check logic
    # ------------------------------------------------------------------

    def _run_health_check(self) -> None:
        """Evaluate all triggers at the current interval boundary."""
        # Sample critic / Q-value metrics
        critic_loss = self.model.logger.name_to_value.get(
            "train/critic_loss", float("nan"))
        if critic_loss == critic_loss:  # not NaN
            self._critic_loss_history.append(float(critic_loss))

        q_stats = self._sample_q_values()
        q1_mean = q_stats.get("q1_mean", float("nan"))
        if q1_mean == q1_mean:
            self._q1_mean_history.append(float(q1_mean))

        # --- T1: collision count in window ---
        collision_count = sum(self._collision_history)
        if collision_count >= self._collision_threshold:
            self._trigger(
                trigger="T1",
                value=float(collision_count),
                threshold=float(self._collision_threshold),
            )
            return

        # --- T2: critic loss L10 ---
        if len(self._critic_loss_history) >= 10:
            critic_l10 = sum(self._critic_loss_history) / len(self._critic_loss_history)
            if critic_l10 > self._critic_loss_threshold:
                self._trigger(
                    trigger="T2",
                    value=critic_l10,
                    threshold=self._critic_loss_threshold,
                )
                return

        # --- T3: Q1 mean L10 ---
        if len(self._q1_mean_history) >= 10:
            q1_l10 = sum(self._q1_mean_history) / len(self._q1_mean_history)
            if q1_l10 < self._q1_mean_threshold:
                self._trigger(
                    trigger="T3",
                    value=q1_l10,
                    threshold=self._q1_mean_threshold,
                )
                return

        # --- T4 advisory: Q1 consecutive decline ---
        if len(self._q1_mean_history) >= 2:
            consecutive_declines = 0
            hist = list(self._q1_mean_history)
            for i in range(len(hist) - 1, 0, -1):
                if hist[i] < hist[i - 1]:
                    consecutive_declines += 1
                else:
                    break
            if consecutive_declines >= self._q_consecutive_decline_threshold:
                logger.warning(
                    "[HealthMonitor] Q1 consecutive decline: %d checks. "
                    "Current=%.2f. Recent: %s",
                    consecutive_declines, hist[-1],
                    [f"{q:.2f}" for q in hist[-6:]],
                )

        # --- Clear: no trigger fired ---
        m3b_l20 = (sum(self._m3b_dwell_history) / len(self._m3b_dwell_history)
                   if self._m3b_dwell_history else 0.0)
        critic_l10 = (sum(self._critic_loss_history) / len(self._critic_loss_history)
                      if self._critic_loss_history else float("nan"))
        q1_l10 = (sum(self._q1_mean_history) / len(self._q1_mean_history)
                  if self._q1_mean_history else float("nan"))
        collision_count = sum(self._collision_history)

        logger.info(
            "[HealthMonitor] ep=%d OK: M3b_dwell_L20=%.2f, "
            "critic_loss_L10=%.2f, Q1_mean_L10=%.2f, collisions_L20=%d",
            self._episode_count, m3b_l20, critic_l10, q1_l10,
            collision_count,
        )

    # ------------------------------------------------------------------
    # Q-value sampling (mirrors OffPolicyDiagnosticsCallback._sample_q_values)
    # ------------------------------------------------------------------

    def _sample_q_values(self) -> Dict[str, float]:
        """Sample Q1/Q2 statistics from the critic.

        Draws 256 transitions from the replay buffer, forward-passes through
        the critic networks, and returns mean/max per critic. Returns NaN
        on any error to never crash training.
        """
        nan_result = {"q1_mean": float("nan"), "q2_mean": float("nan"),
                      "q1_max": float("nan"), "q2_max": float("nan")}
        try:
            buf = getattr(self.model, "replay_buffer", None)
            if buf is None or buf.size() < 256:
                return nan_result

            critic = getattr(self.model, "critic", None)
            if critic is None:
                return nan_result

            replay_data = buf.sample(256, env=self.training_env)
            with torch.no_grad():
                q1, q2 = critic(replay_data.observations, replay_data.actions)

            return {
                "q1_mean": float(q1.mean().item()),
                "q2_mean": float(q2.mean().item()),
                "q1_max": float(q1.max().item()),
                "q2_max": float(q2.max().item()),
            }
        except Exception:
            logger.debug("[HealthMonitor] Q-value sampling failed", exc_info=True)
            return nan_result

    # ------------------------------------------------------------------
    # Trigger + emergency checkpoint
    # ------------------------------------------------------------------

    def _trigger(self, trigger: str, value: float, threshold: float) -> None:
        """Handle a health trigger: checkpoint, sentinel, log."""
        logger.critical(
            "[HealthMonitor] TRIGGER %s: value=%.3f, threshold=%.3f, "
            "episode=%d, timestep=%d",
            trigger, value, threshold, self._episode_count, self.num_timesteps,
        )

        # 1. Emergency checkpoint
        self._save_emergency_checkpoint()

        # 2. Sentinel file
        try:
            sentinel = {
                "trigger": trigger,
                "value": value,
                "threshold": threshold,
                "episode": self._episode_count,
                "timestep": self.num_timesteps,
                "timestamp": datetime.now().isoformat(),
            }
            with open(self._SENTINEL_PATH, "w") as f:
                json.dump(sentinel, f, indent=2)
        except Exception:
            logger.error("[HealthMonitor] Failed to write sentinel", exc_info=True)

    def _save_emergency_checkpoint(self) -> None:
        """Save model + replay buffer + VecNormalize with atomic write pattern."""
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        step = self.num_timesteps
        prefix = f"emergency_{step}_steps"

        # Model
        model_path = os.path.join(self._checkpoint_dir, f"{prefix}.zip")
        self._atomic_save(self.model.save, model_path)
        logger.info("[HealthMonitor] Emergency model saved: %s", model_path)

        # Replay buffer
        if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            buf_path = os.path.join(
                self._checkpoint_dir, f"replay_buffer_{step}_steps.pkl")
            self._atomic_save(self.model.save_replay_buffer, buf_path)
            logger.info("[HealthMonitor] Emergency replay buffer saved: %s", buf_path)

        # VecNormalize
        if isinstance(self.training_env, VecNormalize):
            norm_path = os.path.join(
                self._checkpoint_dir, f"vecnormalize_{step}_steps.pkl")
            self._atomic_save(self.training_env.save, norm_path)
            logger.info("[HealthMonitor] Emergency VecNormalize saved: %s", norm_path)

    @staticmethod
    def _atomic_save(save_fn, final_path: str) -> None:
        """Write to .tmp then os.replace() for POSIX atomic rename."""
        tmp_path = final_path + ".tmp"
        try:
            save_fn(tmp_path)
            os.replace(tmp_path, final_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# =============================================================================
# CHECKPOINT-GATED ROLLBACK (C2, H3)
# =============================================================================


class CheckpointGatedRollbackCallback(BaseCallback):
    """Checkpoint-gated evaluation with automatic rollback (H3).

    Every eval_interval training episodes (after grace period), computes
    M3b sustained dwell rate from recent training episodes. If rate drops
    below trigger_ratio * rolling best, restores model weights from best
    checkpoint. Coordinates with TrainingHealthCallback — defers rollback
    if health alert sentinel exists.
    """

    _HEALTH_SENTINEL = "/tmp/flightdock_health_alert.json"
    _EXHAUSTED_SENTINEL = "/tmp/flightdock_rollback_exhausted.json"

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        eval_interval: int = config.ROLLBACK_EVAL_INTERVAL_EPISODES,
        grace_episodes: int = config.ROLLBACK_GRACE_EPISODES,
        trigger_ratio: float = config.ROLLBACK_TRIGGER_RATIO,
        cooldown_episodes: int = config.ROLLBACK_COOLDOWN_EPISODES,
        max_rollbacks: int = config.MAX_ROLLBACKS,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._checkpoint_dir = checkpoint_dir
        self._log_dir = log_dir
        self._eval_interval = eval_interval
        self._grace_episodes = grace_episodes
        self._trigger_ratio = trigger_ratio
        self._cooldown_episodes = cooldown_episodes
        self._max_rollbacks = max_rollbacks

        # State
        self._episode_count: int = 0
        self._episodes_since_last_eval: int = 0
        self._m3b_window: Deque[bool] = deque(maxlen=eval_interval)
        self._best_eval_m3b_rate: float = 0.0
        self._best_eval_episode: int = 0
        self._rollback_count: int = 0
        self._in_cooldown: bool = False
        self._cooldown_episodes_remaining: int = 0

    def _init_callback(self) -> None:
        best_dir = os.path.join(
            self._checkpoint_dir, config.ROLLBACK_BEST_CHECKPOINT_DIR)
        os.makedirs(best_dir, exist_ok=True)
        os.makedirs(self._log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            self._episode_count += 1
            info = infos[i] if i < len(infos) else {}

            # Track M3b achieved
            m3b = bool(info.get("actual_M3b_achieved", False))
            self._m3b_window.append(m3b)

            # Cooldown decrement
            if self._in_cooldown:
                self._cooldown_episodes_remaining -= 1
                if self._cooldown_episodes_remaining <= 0:
                    self._in_cooldown = False
                    self._episodes_since_last_eval = 0
                    logger.info(
                        "[RollbackGate] Cooldown ended at episode %d",
                        self._episode_count,
                    )
                continue

            # Grace period
            if self._episode_count <= self._grace_episodes:
                continue

            # Post-grace: count episodes toward next eval
            self._episodes_since_last_eval += 1

            if self._episodes_since_last_eval >= self._eval_interval:
                self._evaluate()
                self._episodes_since_last_eval = 0

        return True

    def _evaluate(self) -> None:
        """Evaluate M3b rate and decide on checkpoint save or rollback."""
        if len(self._m3b_window) == 0:
            return

        rate = sum(self._m3b_window) / len(self._m3b_window)

        logger.info(
            "[RollbackGate] Eval at ep %d: M3b rate=%.3f "
            "(best=%.3f at ep %d, window=%d)",
            self._episode_count, rate, self._best_eval_m3b_rate,
            self._best_eval_episode, len(self._m3b_window),
        )

        # New best?
        if rate > self._best_eval_m3b_rate:
            self._best_eval_m3b_rate = rate
            self._best_eval_episode = self._episode_count
            self._save_best_checkpoint()
            logger.info(
                "[RollbackGate] New best M3b rate=%.3f at ep %d — "
                "checkpoint saved",
                rate, self._episode_count,
            )

        # P1-b (FD_CURRICULUM_V3): demoted to checkpoint-save-ONLY — sealed
        # gates are the sole demote/rollback authority under v3 (an in-loop
        # restore would race the gate state machine). Best-checkpoint saving
        # above stays live; the restore path below is dead.
        if config.FD_CURRICULUM_V3:
            return

        # Rollback check: rate dropped below trigger_ratio * best
        if (self._best_eval_m3b_rate > 0
                and rate < self._trigger_ratio * self._best_eval_m3b_rate):
            # Defer to health alert if active
            if os.path.exists(self._HEALTH_SENTINEL):
                logger.warning(
                    "[RollbackGate] M3b degradation detected (%.3f < %.3f * %.3f) "
                    "but health sentinel active — deferring rollback",
                    rate, self._trigger_ratio, self._best_eval_m3b_rate,
                )
                return

            # Max rollbacks check
            if self._rollback_count >= self._max_rollbacks:
                logger.warning(
                    "[RollbackGate] Rollback budget exhausted (%d/%d) — "
                    "continuing without rollback",
                    self._rollback_count, self._max_rollbacks,
                )
                return

            self._rollback(rate, self._best_eval_m3b_rate)

    def _save_best_checkpoint(self) -> None:
        """Save model weights and VecNormalize stats to best_policy dir."""
        best_dir = os.path.join(
            self._checkpoint_dir, config.ROLLBACK_BEST_CHECKPOINT_DIR)
        model_path = os.path.join(best_dir, "model.zip")
        self._atomic_save(self.model.save, model_path)

        if isinstance(self.training_env, VecNormalize):
            norm_path = os.path.join(best_dir, "vecnormalize.pkl")
            self._atomic_save(self.training_env.save, norm_path)

    def _rollback(self, current_rate: float, best_rate: float) -> None:
        """Restore model weights from best checkpoint."""
        best_dir = os.path.join(
            self._checkpoint_dir, config.ROLLBACK_BEST_CHECKPOINT_DIR)
        best_model_path = os.path.join(best_dir, "model.zip")

        if not os.path.isfile(best_model_path):
            logger.error(
                "[RollbackGate] Cannot rollback — best checkpoint not found: %s",
                best_model_path,
            )
            return

        logger.warning(
            "[RollbackGate] ROLLBACK #%d: M3b rate %.3f < %.1f%% of best %.3f "
            "(ep %d → best ep %d). Restoring weights.",
            self._rollback_count + 1, current_rate,
            self._trigger_ratio * 100, best_rate,
            self._episode_count, self._best_eval_episode,
        )

        # 1. Restore model weights (NOT replay buffer)
        self.model.set_parameters(best_model_path)

        # 2. Restore VecNormalize stats if available
        norm_path = os.path.join(best_dir, "vecnormalize.pkl")
        if os.path.isfile(norm_path) and isinstance(
                self.training_env, VecNormalize):
            import pickle
            with open(norm_path, "rb") as f:
                saved_env = pickle.load(f)
            self.training_env.obs_rms = saved_env.obs_rms
            if hasattr(saved_env, "ret_rms"):
                self.training_env.ret_rms = saved_env.ret_rms
            logger.info("[RollbackGate] VecNormalize stats restored")

        # 3. Increment count
        self._rollback_count += 1

        # 4. Log event
        self._log_rollback_event(current_rate, best_rate)

        # 5. Enter cooldown
        self._in_cooldown = True
        self._cooldown_episodes_remaining = self._cooldown_episodes
        logger.info(
            "[RollbackGate] Entering %d-episode cooldown",
            self._cooldown_episodes,
        )

        # 6. Exhaustion check
        if self._rollback_count >= self._max_rollbacks:
            logger.critical(
                "[RollbackGate] Rollback budget EXHAUSTED (%d/%d). "
                "Writing sentinel. Training continues without further rollbacks.",
                self._rollback_count, self._max_rollbacks,
            )
            sentinel_data = {
                "event": "rollback_budget_exhausted",
                "rollback_count": self._rollback_count,
                "max_rollbacks": self._max_rollbacks,
                "episode": self._episode_count,
                "timestep": self.num_timesteps,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            with open(self._EXHAUSTED_SENTINEL, "w") as f:
                json.dump(sentinel_data, f, indent=2)

    def _log_rollback_event(
        self, current_rate: float, best_rate: float
    ) -> None:
        """Append rollback event to JSON-lines log."""
        log_path = os.path.join(self._log_dir, "rollback_events.json")
        event = {
            "event": "rollback",
            "rollback_number": self._rollback_count,
            "episode": self._episode_count,
            "timestep": self.num_timesteps,
            "current_m3b_rate": current_rate,
            "best_m3b_rate": best_rate,
            "trigger_ratio": self._trigger_ratio,
            "best_episode": self._best_eval_episode,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        # L9.3: Idempotent sentinel for operator/watchdog
        sentinel_path = os.path.join(self._log_dir, "ROLLBACK_FIRED.txt")
        if not os.path.exists(sentinel_path):
            with open(sentinel_path, "w") as sf:
                sf.write(f"[{event['timestamp']}] Rollback #{event['rollback_number']} "
                         f"at ep {event['episode']} ts={event['timestep']}: "
                         f"rate={current_rate:.4f} < "
                         f"{self._trigger_ratio:.2f}*best={best_rate:.4f}\n")

    @staticmethod
    def _atomic_save(save_fn, final_path: str) -> None:
        """Write to .tmp then os.replace() for POSIX atomic rename."""
        tmp_path = final_path + ".tmp"
        try:
            save_fn(tmp_path)
            os.replace(tmp_path, final_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


class TrainingProgressMonitorCallback(BaseCallback):
    """Monitor training M3b rate for plateau/failure early stopping (Campaign 3).

    IMPORTANT: Training M3b rate (stochastic) is systematically lower than
    deterministic M3b rate due to exploration noise. The failure threshold
    is set conservatively low (default 5%) to avoid false failure stops.
    The SUCCESS criterion (>50% M3b) MUST be validated via separate
    deterministic evaluation at checkpoint boundaries.
    """

    def __init__(
        self,
        window_size: int = 50,
        failure_threshold: float = 0.05,
        failure_timestep: int = 100_000,
        plateau_window: int = 3,
        plateau_eval_interval: int = 25_000,
        plateau_tolerance: float = 0.02,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self._window_size = window_size
        self._failure_threshold = failure_threshold
        self._failure_timestep = failure_timestep
        self._plateau_window = plateau_window
        self._plateau_eval_interval = plateau_eval_interval
        self._plateau_tolerance = plateau_tolerance
        self._m3b_history: list = []
        self._eval_rates: list = []
        self._last_eval_ts: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                m3b = info.get("actual_M3b_achieved", False)
                self._m3b_history.append(bool(m3b))

        if (self.num_timesteps - self._last_eval_ts) < self._plateau_eval_interval:
            return True
        if len(self._m3b_history) < self._window_size:
            return True

        self._last_eval_ts = self.num_timesteps
        recent = self._m3b_history[-self._window_size:]
        m3b_rate = sum(recent) / len(recent)
        self._eval_rates.append(m3b_rate)

        self.logger.record("progress/m3b_rate_training", m3b_rate)
        self.logger.record("progress/n_episodes_seen", len(self._m3b_history))

        # Failure: conservative 5% accounts for stochastic-deterministic gap
        if (self.num_timesteps >= self._failure_timestep
                and m3b_rate < self._failure_threshold):
            if self.verbose:
                print(f"[ProgressMonitor] FAILURE: M3b rate {m3b_rate:.1%} "
                      f"< {self._failure_threshold:.1%} after "
                      f"{self.num_timesteps} timesteps")
            self.logger.record("progress/early_stop", "FAILURE")
            return False

        # Plateau: relative change detection (less affected by stoch-det gap)
        if len(self._eval_rates) >= self._plateau_window:
            window = self._eval_rates[-self._plateau_window:]
            if max(window) - min(window) < self._plateau_tolerance:
                if self.verbose:
                    print(f"[ProgressMonitor] PLATEAU: {self._plateau_window} "
                          f"consecutive evals within {self._plateau_tolerance:.1%}: "
                          f"{[f'{r:.1%}' for r in window]}")
                self.logger.record("progress/early_stop", "PLATEAU")
                return False

        return True


class QValueDistributionCallback(BaseCallback):
    """C3X Intervention 3 — per-episode Q-value distribution across DBB buckets.

    Dual-variant logging (C3X IC4 baseline rework, 2026-04-08): the
    callback computes BOTH the contact-DBB (primary, boom-tip-relative —
    matches the campaign plan's nominal "DBB" semantics and the Phase 0
    baseline 0.344 s reference) and the band-DBB (auxiliary,
    precontact-relative — DRL ``hold_m3b`` reward shaping reference) for
    each cadence event, in a single buffer pass.

    Scientific discriminator between H1 (controller antagonism) and the
    dense reward dominance weak form. Under H1, the critic's Q-values
    are high in the long-dwell DBB bins but the agent cannot reach them;
    the distribution shows a high-variance, right-tailed shape. Under
    the weak form, Q-values are flat across bins because the terminal
    bonus is too small to produce a gradient toward the threshold.

    Per-episode cadence: fires once per ``_on_step`` call in which any
    env signals ``done=True``. Each cadence event:

    1. Iterates the replay buffer in **chronological storage order**
       and accumulates BOTH band-DBB and contact-DBB per transition in
       a single buffer pass. The band-DBB uses
       :func:`drl.env_formation_sitl.check_m3b` over the precontact
       envelope errors at obs[7]/[8]/[9]; the contact-DBB uses
       :func:`drl.env_formation_sitl.check_contact_m3b` over the raw
       boom-tip-relative geometry at obs[0]/[1]/[2]. Per-env
       independence is preserved via two n_envs-wide accumulators.
    2. Stratifies the contact-DBB and band-DBB arrays separately into
       6 buckets each with non-linear edges
       ``[0.0, 0.2, 0.5, 1.0, 2.0, 3.0, ∞)``. The layout provides finer
       resolution at the low end where the Phase 0 baseline DBB mean
       (0.344 s) lives.
    3. Samples up to 1024 transitions per variant, distributing 170 per
       bucket with deficit redistribution to populated buckets.
    4. For each sampled state, computes the deterministic actor action
       and twin-critic Q-values ``(Q1, Q2, min(Q1, Q2))``. The contact
       variant emits all three aggregations (Q1, Q2, Qmin); the band
       variant emits only Qmin (auxiliary cross-validation channel).
    5. Aggregates mean per ``(bucket, q-aggregation)``, computes the
       right-tail slope from contact-Qmin, and appends a single row to
       a CSV file.

    The CSV has **26 columns**:

    - ``timestep`` (1 column)
    - 18 contact-variant columns: ``bucket_X_q1, bucket_X_q2,
      bucket_X_qmin`` for each of 6 buckets (operator-prescribed
      schema; the contact-Qmin columns serve as the IC4 directional
      criterion source)
    - ``right_tail_slope`` (1 column, computed from contact-Qmin)
    - 6 band-variant auxiliary columns: ``bucket_X_qmin_band`` for
      each of 6 buckets (cross-validation channel; the band variant
      tracks the precontact envelope which is the DRL reward shaping
      reference but NOT the campaign plan's "DBB")

    Notes on observation handling:

    - Replay-buffer obs are **double-normalized**: the env's
      :class:`ObservationBuilder` divides each element by
      :data:`drl.config.OBS_NORMS`, then ``VecNormalize`` z-scores. To
      recover raw error values for ``check_m3b`` and ``check_contact_m3b``,
      the callback first ``unnormalize_obs``-es via the live
      ``VecNormalize`` wrapper, then multiplies by ``OBS_NORMS[i]``.
      Since the callback uses the *current* VecNormalize running stats,
      recovered values for older buffer entries may carry small
      calibration drift; for the smallest bucket width (0.2 s, bucket
      0) the drift is well below the bucket resolution.
    - Q-value computation uses the buffer obs *as stored* (normalized),
      because the policy / critic networks were trained on normalized
      obs and expect that format at inference.

    Parameters
    ----------
    log_path: str
        Path to the output CSV (e.g.,
        ``logs/campaign3x/q_value_distribution.csv``). Parent directory
        will be created if missing.
    sample_target: int, default 1024
        Hard-pinned sample budget per cadence event per variant.
    verbose: int, default 0
        BaseCallback verbosity.
    """

    BUCKET_EDGES_LO = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0)  # 6 left edges (non-linear, C3X)
    BUCKET_NAMES = (
        "0_0s_to_0_2s",
        "0_2s_to_0_5s",
        "0_5s_to_1_0s",
        "1_0s_to_2_0s",
        "2_0s_to_3_0s",
        "3_0s_plus",
    )
    N_BUCKETS = 6

    # Obs index layout (must match observation_builder.py raw construction).
    # Band variant: precontact envelope errors (geom - target_behind, etc.)
    OBS_INDEX_BEHIND_ERR = 7
    OBS_INDEX_CROSS_ERR = 8
    OBS_INDEX_BELOW_ERR = 9
    # Contact variant: raw boom-tip-relative geometry
    OBS_INDEX_BEHIND_M = 0
    OBS_INDEX_CROSS_M = 1
    OBS_INDEX_BELOW_M = 2
    # Temporal
    OBS_INDEX_DT_S = 31

    def __init__(
        self,
        log_path: str,
        sample_target: int = 1024,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_path = log_path
        self.sample_target = int(sample_target)
        self.per_bucket_target = self.sample_target // self.N_BUCKETS  # 170
        self._header_written = False
        self._rng = np.random.default_rng()
        parent = os.path.dirname(self.log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # Header & schema
    # ------------------------------------------------------------------

    def _column_names(self) -> List[str]:
        """Return the 26-column header (contact-primary + band-auxiliary).

        Schema:
          - timestep (1 column)
          - 18 contact-variant Q columns: bucket_X_{q1, q2, qmin}
            for each of 6 buckets (boom-tip-relative DBB)
          - right_tail_slope (1 column, computed from contact-Qmin)
          - 6 band-variant auxiliary columns: bucket_X_qmin_band
            for each of 6 buckets (precontact-envelope DBB cross-check)
        """
        cols = ["timestep"]
        # Contact-variant columns (primary, all three Q aggregations)
        for bname in self.BUCKET_NAMES:
            cols.append(f"bucket_{bname}_q1")
            cols.append(f"bucket_{bname}_q2")
            cols.append(f"bucket_{bname}_qmin")
        cols.append("right_tail_slope")
        # Band-variant auxiliary columns (qmin only)
        for bname in self.BUCKET_NAMES:
            cols.append(f"bucket_{bname}_qmin_band")
        return cols

    def _write_header(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._column_names())
        self._header_written = True

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        """Trigger one Q-value distribution emission on any env-done event."""
        dones = self.locals.get("dones", None)
        if dones is None or not bool(np.any(dones)):
            return True
        try:
            self._log_distribution()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "[QValueDistribution] failed at ts=%d: %s",
                self.num_timesteps, exc,
            )
        return True

    # ------------------------------------------------------------------
    # Core: dwell computation, sampling, Q-value aggregation
    # ------------------------------------------------------------------

    def _log_distribution(self) -> None:
        rb = getattr(self.model, "replay_buffer", None)
        if rb is None:
            return

        n_stored = rb.buffer_size if rb.full else rb.pos
        if n_stored == 0:
            logger.warning(
                "[QValueDistribution] Replay buffer empty at ts=%d, "
                "skipping cadence trigger (pre-warmup).",
                self.num_timesteps,
            )
            return

        n_envs = rb.n_envs

        # Compute BOTH variants in a single buffer pass.
        dwell_band, dwell_contact = self._compute_dwell_chronological(
            rb, n_stored, n_envs,
        )

        # Stratify each variant independently (different distributions
        # produce different per-bucket sample sets).
        sampled_contact = self._stratified_sample(dwell_contact, n_envs)
        sampled_band = self._stratified_sample(dwell_band, n_envs)

        n_contact = sum(len(idx) for idx in sampled_contact)
        n_band = sum(len(idx) for idx in sampled_band)
        if n_contact == 0 and n_band == 0:
            logger.warning(
                "[QValueDistribution] No samples after stratification at "
                "ts=%d (both variants degenerate).", self.num_timesteps,
            )
            return

        # Contact (primary): q1, q2, qmin per bucket.
        bucket_q1, bucket_q2, bucket_qmin = self._compute_q_values(
            rb, sampled_contact,
        )
        right_tail_slope = self._right_tail_slope(bucket_qmin)

        # Band (auxiliary): qmin only per bucket. We compute Q1/Q2/Qmin
        # internally and discard q1/q2 to keep the schema minimal.
        _, _, bucket_qmin_band = self._compute_q_values(rb, sampled_band)

        if not self._header_written:
            self._write_header()

        row: List = [self.num_timesteps]
        for b in range(self.N_BUCKETS):
            row.append(bucket_q1[b])
            row.append(bucket_q2[b])
            row.append(bucket_qmin[b])
        row.append(right_tail_slope)
        for b in range(self.N_BUCKETS):
            row.append(bucket_qmin_band[b])

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _compute_dwell_chronological(
        self, rb, n_stored: int, n_envs: int,
    ) -> tuple:
        """Replay the buffer in storage order, computing BOTH band-DBB
        and contact-DBB per transition in a single pass.

        Returns
        -------
        tuple of two np.ndarray
            ``(dwell_band, dwell_contact)``. Each array has shape
            ``(n_stored, n_envs)``, dtype float64. ``dwell_band[i, env]``
            is the continuous M3b dwell using the band-variant
            (precontact envelope errors at obs[7]/[8]/[9]) at transition
            ``(i, env)``. ``dwell_contact[i, env]`` is the same using
            the contact-variant (raw boom-tip-relative geometry at
            obs[0]/[1]/[2]). Both reset on non-M3b transitions and on
            episode boundaries; per-env independence preserved.
        """
        dwell_band = np.zeros((n_stored, n_envs), dtype=np.float64)
        dwell_contact = np.zeros((n_stored, n_envs), dtype=np.float64)
        accum_band = np.zeros(n_envs, dtype=np.float64)
        accum_contact = np.zeros(n_envs, dtype=np.float64)

        # Chronological order: oldest first.
        if rb.full:
            order = np.concatenate([
                np.arange(rb.pos, rb.buffer_size),
                np.arange(0, rb.pos),
            ])
        else:
            order = np.arange(rb.pos)

        vec_norm = self._get_vec_normalize()
        obs_norms = np.asarray(config.OBS_NORMS, dtype=np.float64)

        # Cache per-element scaling factors for both variants + dt_s.
        scale_behind_err = float(obs_norms[self.OBS_INDEX_BEHIND_ERR])
        scale_cross_err = float(obs_norms[self.OBS_INDEX_CROSS_ERR])
        scale_below_err = float(obs_norms[self.OBS_INDEX_BELOW_ERR])
        scale_behind_m = float(obs_norms[self.OBS_INDEX_BEHIND_M])
        scale_cross_m = float(obs_norms[self.OBS_INDEX_CROSS_M])
        scale_below_m = float(obs_norms[self.OBS_INDEX_BELOW_M])
        scale_dt = float(obs_norms[self.OBS_INDEX_DT_S])

        for write_pos, chrono_pos in enumerate(order):
            obs_step = rb.observations[chrono_pos]  # (n_envs, obs_dim)
            done_step = rb.dones[chrono_pos]        # (n_envs,)

            if vec_norm is not None:
                obs_unscaled = vec_norm.unnormalize_obs(obs_step)
            else:
                obs_unscaled = obs_step

            for env_idx in range(n_envs):
                obs_env = obs_unscaled[env_idx]

                # Band-variant inputs: precontact envelope errors
                behind_err = float(
                    obs_env[self.OBS_INDEX_BEHIND_ERR]) * scale_behind_err
                cross_err = float(
                    obs_env[self.OBS_INDEX_CROSS_ERR]) * scale_cross_err
                below_err = float(
                    obs_env[self.OBS_INDEX_BELOW_ERR]) * scale_below_err

                # Contact-variant inputs: raw boom-tip-relative geometry
                behind_m = float(
                    obs_env[self.OBS_INDEX_BEHIND_M]) * scale_behind_m
                cross_m = float(
                    obs_env[self.OBS_INDEX_CROSS_M]) * scale_cross_m
                below_m = float(
                    obs_env[self.OBS_INDEX_BELOW_M]) * scale_below_m

                # Shared dt_s
                dt_s = float(
                    obs_env[self.OBS_INDEX_DT_S]) * scale_dt

                # Band accumulator
                if check_m3b(behind_err, cross_err, below_err):
                    accum_band[env_idx] += dt_s
                else:
                    accum_band[env_idx] = 0.0
                dwell_band[write_pos, env_idx] = accum_band[env_idx]

                # Contact accumulator
                if check_contact_m3b(behind_m, cross_m, below_m):
                    accum_contact[env_idx] += dt_s
                else:
                    accum_contact[env_idx] = 0.0
                dwell_contact[write_pos, env_idx] = accum_contact[env_idx]

                # Reset BOTH accumulators on episode boundary AFTER
                # recording: the dwell at the done step is the dwell
                # at the done step itself; the next transition begins
                # a fresh episode.
                if bool(done_step[env_idx]):
                    accum_band[env_idx] = 0.0
                    accum_contact[env_idx] = 0.0

        return dwell_band, dwell_contact

    def _stratified_sample(
        self, dwell_arr: np.ndarray, n_envs: int,
    ) -> List[np.ndarray]:
        """Stratified sampling by bucket with deficit redistribution.

        Returns
        -------
        list of np.ndarray
            Length ``N_BUCKETS``. Each entry is a ``(k_b, 2)`` int array
            of ``(chronological_index, env_idx)`` pairs sampled from
            bucket ``b``. Empty buckets yield ``(0, 2)`` arrays.
        """
        flat_dwell = dwell_arr.reshape(-1)  # row-major: i*n_envs + env_idx
        bucket_idx = bucket_index_array(flat_dwell)

        per_bucket_idx = [
            np.where(bucket_idx == b)[0] for b in range(self.N_BUCKETS)
        ]
        bucket_sizes = [len(idx) for idx in per_bucket_idx]

        takes = [min(self.per_bucket_target, sz) for sz in bucket_sizes]
        deficit = self.sample_target - sum(takes)

        # Redistribute deficit uniformly among buckets that have remaining
        # capacity (size > current take).
        while deficit > 0:
            has_capacity = [
                b for b in range(self.N_BUCKETS)
                if bucket_sizes[b] > takes[b]
            ]
            if not has_capacity:
                break

            per_extra = deficit // len(has_capacity)
            if per_extra == 0:
                # Distribute the remainder one slot at a time, in order.
                for b in has_capacity:
                    if deficit == 0:
                        break
                    if takes[b] < bucket_sizes[b]:
                        takes[b] += 1
                        deficit -= 1
                break

            for b in has_capacity:
                extra = min(per_extra, bucket_sizes[b] - takes[b])
                takes[b] += extra
                deficit -= extra

        sampled = []
        for b in range(self.N_BUCKETS):
            if takes[b] == 0:
                sampled.append(np.empty((0, 2), dtype=np.int64))
                continue
            chosen_flat = self._rng.choice(
                per_bucket_idx[b], size=takes[b], replace=False,
            )
            i_arr = chosen_flat // n_envs
            env_arr = chosen_flat % n_envs
            sampled.append(np.stack([i_arr, env_arr], axis=1).astype(np.int64))
        return sampled

    def _compute_q_values(
        self, rb, sampled_per_bucket: List[np.ndarray],
    ) -> tuple:
        """Aggregate mean Q1, Q2, Qmin per bucket via current actor + twin critic.

        Returns
        -------
        tuple of three lists, each length ``N_BUCKETS`` of float (NaN if
        bucket empty): ``(bucket_q1, bucket_q2, bucket_qmin)``.
        """
        bucket_q1: List[float] = [float("nan")] * self.N_BUCKETS
        bucket_q2: List[float] = [float("nan")] * self.N_BUCKETS
        bucket_qmin: List[float] = [float("nan")] * self.N_BUCKETS

        device = self.model.device

        for b in range(self.N_BUCKETS):
            idx_pairs = sampled_per_bucket[b]
            if len(idx_pairs) == 0:
                continue
            i_arr = idx_pairs[:, 0]
            env_arr = idx_pairs[:, 1]
            obs_batch = rb.observations[i_arr, env_arr]  # (k, obs_dim)

            obs_tensor = torch.as_tensor(
                obs_batch, dtype=torch.float32, device=device,
            )

            with torch.no_grad():
                # Deterministic actor action; cross-algorithm idiom that
                # works for both TD3 and SAC. For TD3 _predict returns
                # actor(obs); for SAC it returns the distribution mean.
                actions_tensor = self.model.policy._predict(
                    obs_tensor, deterministic=True,
                )
                # Twin critic forward — returns a tuple (q1, q2).
                q_tuple = self.model.critic(obs_tensor, actions_tensor)
                q1 = q_tuple[0].detach().cpu().numpy().reshape(-1)
                q2 = q_tuple[1].detach().cpu().numpy().reshape(-1)
                qmin = np.minimum(q1, q2)

            bucket_q1[b] = float(np.mean(q1))
            bucket_q2[b] = float(np.mean(q2))
            bucket_qmin[b] = float(np.mean(qmin))

        return bucket_q1, bucket_q2, bucket_qmin

    def _right_tail_slope(self, bucket_qmin: List[float]) -> float:
        """Return scalar slope between the open tail and the first bucket.

        Slope = ``(mean_qmin[bucket_3_0s_plus] - mean_qmin[bucket_0_0s_to_0_2s]) / 5``.
        Denominator 5 = number of bucket-index intervals between the two
        endpoints (the 6-bucket non-linear layout has 5 inter-bucket gaps).

        NOTE: This returns an index-normalized heuristic, not a true
        kinematic derivative with respect to time, due to the non-linear
        bucket widths. The 5 inter-bucket gaps span an inhomogeneous time
        range (0.2 + 0.3 + 0.5 + 1.0 + 1.0 = 3.0 s of finite-width domain
        plus an open ∞ tail), so the divisor 5 is a bucket-index step
        count, NOT a seconds count. The statistic is monotone-preserving
        and useful for tracking the qualitative "right-tail rises" trend
        the IC4 gate language specifies, but should not be interpreted as
        a Q-per-second gradient.

        Returns NaN if either endpoint bucket is empty.
        """
        lo = bucket_qmin[0]
        hi = bucket_qmin[self.N_BUCKETS - 1]
        if np.isnan(lo) or np.isnan(hi):
            return float("nan")
        # Index-normalized heuristic (see docstring): denominator is the
        # number of bucket-index intervals (5), NOT a time interval in
        # seconds. Non-linear bucket widths preclude a clean kinematic
        # derivative interpretation.
        return (hi - lo) / 5.0

    def _get_vec_normalize(self) -> Optional[VecNormalize]:
        """Return the live VecNormalize wrapper, walking nested wrappers."""
        env = None
        getter = getattr(self.model, "get_vec_normalize_env", None)
        if callable(getter):
            env = getter()
            if isinstance(env, VecNormalize):
                return env
        env = self.model.get_env() if hasattr(self.model, "get_env") else None
        if env is None:
            return None
        if isinstance(env, VecNormalize):
            return env
        cur = env
        while hasattr(cur, "venv"):
            cur = cur.venv
            if isinstance(cur, VecNormalize):
                return cur
        return None


def bucket_index_array(dwell_values: np.ndarray) -> np.ndarray:
    """Vectorized bucket assignment for the C3X 6-bucket non-linear layout.

    Edges: ``[0.0, 0.2, 0.5, 1.0, 2.0, 3.0, ∞)``. Each bucket
    ``[edges[i], edges[i+1])`` maps to index ``i`` (left edge inclusive,
    right edge exclusive). Returns int32 array of the same shape as
    ``dwell_values``.
    """
    arr = np.asarray(dwell_values, dtype=np.float64)
    # Internal edges (excluding leading 0.0 and trailing ∞).
    # searchsorted(side="right") puts a value v at index i where
    # edges[i-1] <= v < edges[i], which matches the half-open bucket
    # convention exactly: a value equal to a left edge falls into the
    # bucket starting at that edge.
    edges = np.array([0.2, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)
    out = np.searchsorted(edges, arr, side="right").astype(np.int32)
    out[arr < 0.0] = 0  # defensive: clamp negatives into bucket 0
    return out
